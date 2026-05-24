"""Backend orchestration for negative-context retrieval."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Sequence, cast

import torch
from tqdm import tqdm

from config import config
from model.turingllm import TuringLLMConfig

from .ann import (
    TorchANNIndex,
    _record_ann_memory_estimate,
    check_neg_ctx_memory_guardrail,
    estimate_neg_ctx_ann_memory,
    estimate_neg_ctx_ann_memory_for_shape,
)
from .component import _process_component, _process_component_sharded
from .devices import (
    _ann_device,
    _validate_cuda_devices,
    parse_neg_ctx_devices,
    partition_components,
)
from .sharded_ann import ShardedANNIndex, partition_index_slots
from .stats import NegCtxStats
from .validation import validate_neg_ctx_output

if TYPE_CHECKING:
    from store.context import Context
    from store.seq_repr import SeqRepr


def build_neg_ctx(
    seq_repr: "SeqRepr",
    top_ctx: "Context",
    mid_ctx: "Context",
    neg_ctx: "Context",
    *,
    selected_devices: Sequence[int | str | torch.device] | None = None,
) -> NegCtxStats:
    """
    Populate neg_ctx for all latents with sufficient PosCtx data.

    Each of the 36 SAE components is processed fully vectorised:
    no Python loops over the ~40K active latents per component.

    Returns a NegCtxStats instance with fill-rate distribution and timing.
    """
    backend = cast(str, config.latents.neg_ctx.backend or "single_gpu_exact")
    if backend == "multi_gpu_exact":
        return build_neg_ctx_multi_gpu(
            seq_repr,
            top_ctx,
            mid_ctx,
            neg_ctx,
            selected_devices=selected_devices,
        )
    if backend == "multi_gpu_index_sharded_exact":
        return build_neg_ctx_index_sharded(
            seq_repr,
            top_ctx,
            mid_ctx,
            neg_ctx,
            selected_devices=selected_devices,
        )

    return build_neg_ctx_single_gpu_exact(seq_repr, top_ctx, mid_ctx, neg_ctx)


def build_neg_ctx_single_gpu_exact(
    seq_repr: "SeqRepr",
    top_ctx: "Context",
    mid_ctx: "Context",
    neg_ctx: "Context",
) -> NegCtxStats:
    """Single-device exact negative-context backend used as the correctness baseline."""

    llm_cfg = TuringLLMConfig()
    n_comp = llm_cfg.n_layer * 3
    n_neg = neg_ctx.num_ctx_sequences
    n_neighbors = cast(int, config.latents.neg_ctx.n_neighbors or 500)
    min_pos_ctx = cast(int, config.latents.neg_ctx.min_pos_ctx or 8)

    stats = NegCtxStats(backend="single_gpu_exact")
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    total_n_seqs = seq_repr.n_seqs     # full dataset count (for stride + valid filter)
    n_stored = seq_repr.n_stored       # actual ANN index size (<= total_n_seqs)
    device = _ann_device()
    stats.devices = [str(device)]
    stats.ann_device = str(device)
    stats.record_seq_repr(seq_repr)
    ann_memory_estimate = estimate_neg_ctx_ann_memory(
        seq_repr,
        query_chunk_size=4096 if device.type == "cuda" else 512,
    )
    guardrail = check_neg_ctx_memory_guardrail(
        device,
        ann_memory_estimate,
        fraction=float(config.latents.neg_ctx.memory_guardrail_fraction),
        fail_on_exceed=bool(config.latents.neg_ctx.fail_on_memory_guardrail),
    )
    _record_ann_memory_estimate(stats, ann_memory_estimate, guardrail)

    raw_vecs = seq_repr.repr_buf[1:n_stored + 1].float()   # [n_stored, D] float32, CPU
    index = TorchANNIndex(raw_vecs, device=device)
    stats.t_index_build = time.perf_counter() - t0

    K = min(n_neighbors, n_stored)

    # Move slot-mapping tensors to compute device once (reused across components).
    # None when uncapped (slot == seq_id, no mapping needed).
    if seq_repr.is_capped and seq_repr.slot_to_id is not None and seq_repr.id_to_slot is not None:
        slot_to_id_d: torch.Tensor | None = seq_repr.slot_to_id.to(device)
        id_to_slot_d: torch.Tensor | None = seq_repr.id_to_slot.to(device)
    else:
        slot_to_id_d = None
        id_to_slot_d = None

    pbar = tqdm(range(n_comp), desc="  [neg_ctx]", unit="comp", leave=True)
    for comp_idx in pbar:
        tc0 = time.perf_counter()

        timing = _process_component(
            comp_idx, top_ctx, mid_ctx, neg_ctx,
            index, K, n_neg, min_pos_ctx, stats,
            total_n_seqs, slot_to_id_d, id_to_slot_d,
        )

        comp_s = time.perf_counter() - tc0
        stats.t_pos_collect += timing.get("pos", 0.0)
        stats.t_qmat_build += timing.get("qmat", 0.0)
        stats.t_query += timing.get("query", 0.0)
        stats.t_filter += timing.get("filter", 0.0)
        stats.t_write += timing.get("write", 0.0)

        active = stats.n_latents_attempted - stats.n_latents_skipped_low_pos
        pbar.set_postfix({
            "active": f"{active // (comp_idx + 1):,}",
            "ms": f"{comp_s * 1000:.0f}",
            "pos_s": f"{stats.t_pos_collect:.1f}",
            "qmat_s": f"{stats.t_qmat_build:.1f}",
            "srch_s": f"{stats.t_query:.1f}",
            "flt_s": f"{stats.t_filter:.1f}",
        })

    pbar.close()
    stats.t_total = time.perf_counter() - t_start
    validate_neg_ctx_output(neg_ctx, total_n_seqs=total_n_seqs, n_sequences=n_neg)
    return stats


def build_neg_ctx_multi_gpu(
    seq_repr: "SeqRepr",
    top_ctx: "Context",
    mid_ctx: "Context",
    neg_ctx: "Context",
    *,
    selected_devices: Sequence[int | str | torch.device] | None = None,
) -> NegCtxStats:
    """
    Component-parallel exact backend.

    Each selected GPU builds its own exact ANN index and owns a disjoint subset
    of SAE components. Component writes target disjoint neg_ctx slices, so the
    final artifact shape and semantics match the single-device backend.
    """
    llm_cfg = TuringLLMConfig()
    n_comp = llm_cfg.n_layer * 3
    n_neg = neg_ctx.num_ctx_sequences
    n_neighbors = cast(int, config.latents.neg_ctx.n_neighbors or 500)
    min_pos_ctx = cast(int, config.latents.neg_ctx.min_pos_ctx or 8)

    configured_devices = (
        list(selected_devices)
        if selected_devices is not None
        else list(config.latents.neg_ctx.devices)
    )
    devices = parse_neg_ctx_devices(configured_devices)
    _validate_cuda_devices(devices)
    assignments = partition_components(n_comp, devices)

    print("  [neg_ctx] backend=multi_gpu_exact")
    for device in devices:
        comps = assignments[str(device)]
        if comps:
            print(f"  [neg_ctx] {device}: {len(comps)} components ({comps[0]}..{comps[-1]})")
        else:
            print(f"  [neg_ctx] {device}: 0 components")

    total_n_seqs = seq_repr.n_seqs
    n_stored = seq_repr.n_stored
    raw_vecs = seq_repr.repr_buf[1:n_stored + 1].float()
    K = min(n_neighbors, n_stored)

    final_stats = NegCtxStats(
        backend="multi_gpu_exact",
        devices=[str(device) for device in devices],
        ann_device="multi_gpu_exact",
        component_assignments=assignments,
    )
    final_stats.record_seq_repr(seq_repr)
    ann_memory_estimate = estimate_neg_ctx_ann_memory(
        seq_repr,
        query_chunk_size=4096,
    )
    for device in devices:
        guardrail = check_neg_ctx_memory_guardrail(
            device,
            ann_memory_estimate,
            fraction=float(config.latents.neg_ctx.memory_guardrail_fraction),
            fail_on_exceed=bool(config.latents.neg_ctx.fail_on_memory_guardrail),
        )
        _record_ann_memory_estimate(final_stats, ann_memory_estimate, guardrail)
    t_start = time.perf_counter()

    def worker(device: torch.device, component_indices: list[int]) -> NegCtxStats:
        torch.cuda.set_device(device)
        worker_stats = NegCtxStats(
            backend="multi_gpu_exact",
            devices=[str(device)],
            ann_device=str(device),
        )
        worker_stats.record_seq_repr(seq_repr)
        _record_ann_memory_estimate(
            worker_stats,
            ann_memory_estimate,
            {
                "fraction": float(config.latents.neg_ctx.memory_guardrail_fraction),
                "limit_bytes": final_stats.ann_memory_guardrail_limit_bytes,
            },
        )

        t0 = time.perf_counter()
        index = TorchANNIndex(raw_vecs, device=device)
        worker_stats.t_index_build = time.perf_counter() - t0

        if seq_repr.is_capped and seq_repr.slot_to_id is not None and seq_repr.id_to_slot is not None:
            slot_to_id_d: torch.Tensor | None = seq_repr.slot_to_id.to(device)
            id_to_slot_d: torch.Tensor | None = seq_repr.id_to_slot.to(device)
        else:
            slot_to_id_d = None
            id_to_slot_d = None

        for comp_idx in component_indices:
            tc0 = time.perf_counter()
            timing = _process_component(
                comp_idx, top_ctx, mid_ctx, neg_ctx,
                index, K, n_neg, min_pos_ctx, worker_stats,
                total_n_seqs, slot_to_id_d, id_to_slot_d,
            )
            comp_s = time.perf_counter() - tc0
            worker_stats.t_pos_collect += timing.get("pos", 0.0)
            worker_stats.t_qmat_build += timing.get("qmat", 0.0)
            worker_stats.t_query += timing.get("query", 0.0)
            worker_stats.t_filter += timing.get("filter", 0.0)
            worker_stats.t_write += timing.get("write", 0.0)
            print(f"  [neg_ctx:{device}] comp={comp_idx} {comp_s * 1000:.0f} ms")
        worker_stats.per_device_timing_ms[str(device)] = {
            "index_build_ms": round(worker_stats.t_index_build * 1000, 1),
            "pos_collect_ms": round(worker_stats.t_pos_collect * 1000, 1),
            "qmat_build_ms": round(worker_stats.t_qmat_build * 1000, 1),
            "query_ms": round(worker_stats.t_query * 1000, 1),
            "filter_ms": round(worker_stats.t_filter * 1000, 1),
            "write_ms": round(worker_stats.t_write * 1000, 1),
        }
        return worker_stats

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, device, assignments[str(device)])
            for device in devices
            if assignments[str(device)]
        ]
        for future in as_completed(futures):
            final_stats.merge_from(future.result())

    final_stats.t_total = time.perf_counter() - t_start
    validate_neg_ctx_output(neg_ctx, total_n_seqs=total_n_seqs, n_sequences=n_neg)
    return final_stats


def build_neg_ctx_index_sharded(
    seq_repr: "SeqRepr",
    top_ctx: "Context",
    mid_ctx: "Context",
    neg_ctx: "Context",
    *,
    selected_devices: Sequence[int | str | torch.device] | None = None,
) -> NegCtxStats:
    """
    Exact backend that shards seq_repr rows across CUDA devices.

    Every query is searched against every shard, then shard-local top-K rows are
    merged into global slot IDs before applying the standard positive filter.
    """

    llm_cfg = TuringLLMConfig()
    n_comp = llm_cfg.n_layer * 3
    n_neg = neg_ctx.num_ctx_sequences
    n_neighbors = cast(int, config.latents.neg_ctx.n_neighbors or 500)
    min_pos_ctx = cast(int, config.latents.neg_ctx.min_pos_ctx or 8)

    configured_devices = (
        list(selected_devices)
        if selected_devices is not None
        else list(config.latents.neg_ctx.devices)
    )
    devices = parse_neg_ctx_devices(configured_devices)
    _validate_cuda_devices(devices)

    total_n_seqs = seq_repr.n_seqs
    n_stored = seq_repr.n_stored
    raw_vecs = seq_repr.repr_buf[1:n_stored + 1].float()
    K = min(n_neighbors, n_stored)
    query_device = devices[0]
    shard_assignments = partition_index_slots(n_stored, devices)

    print("  [neg_ctx] backend=multi_gpu_index_sharded_exact")
    for device in devices:
        start, end = shard_assignments[str(device)]
        print(f"  [neg_ctx] {device}: index slots [{start}, {end}) ({end - start:,} rows)")

    final_stats = NegCtxStats(
        backend="multi_gpu_index_sharded_exact",
        devices=[str(device) for device in devices],
        ann_device="multi_gpu_index_sharded_exact",
        index_shard_assignments={
            str(device): {"start_slot": start, "end_slot": end, "n_rows": end - start}
            for device, (start, end) in (
                (device, shard_assignments[str(device)]) for device in devices
            )
        },
    )
    final_stats.record_seq_repr(seq_repr)

    largest_estimate: dict[str, int] | None = None
    largest_guardrail: dict[str, object] | None = None
    for device in devices:
        start, end = shard_assignments[str(device)]
        shard_estimate = estimate_neg_ctx_ann_memory_for_shape(
            n_stored=end - start,
            n_seqs=total_n_seqs,
            repr_dim=seq_repr.repr_dim,
            is_capped=seq_repr.is_capped,
            query_chunk_size=4096,
        )
        final_stats.ann_shard_memory_estimates[str(device)] = shard_estimate
        guardrail = check_neg_ctx_memory_guardrail(
            device,
            shard_estimate,
            fraction=float(config.latents.neg_ctx.memory_guardrail_fraction),
            fail_on_exceed=bool(config.latents.neg_ctx.fail_on_memory_guardrail),
        )
        if largest_estimate is None or shard_estimate["total_bytes"] > largest_estimate["total_bytes"]:
            largest_estimate = shard_estimate
            largest_guardrail = guardrail
    if largest_estimate is not None and largest_guardrail is not None:
        _record_ann_memory_estimate(final_stats, largest_estimate, largest_guardrail)

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    index = ShardedANNIndex(raw_vecs, devices)
    final_stats.t_index_build = time.perf_counter() - t0

    if seq_repr.is_capped and seq_repr.slot_to_id is not None and seq_repr.id_to_slot is not None:
        slot_to_id_d: torch.Tensor | None = seq_repr.slot_to_id.to(query_device)
        id_to_slot_d: torch.Tensor | None = seq_repr.id_to_slot.to(query_device)
    else:
        slot_to_id_d = None
        id_to_slot_d = None

    pbar = tqdm(range(n_comp), desc="  [neg_ctx:sharded]", unit="comp", leave=True)
    for comp_idx in pbar:
        tc0 = time.perf_counter()
        timing = _process_component_sharded(
            comp_idx, top_ctx, mid_ctx, neg_ctx, seq_repr,
            index, K, n_neg, min_pos_ctx, final_stats,
            total_n_seqs, query_device, slot_to_id_d, id_to_slot_d,
        )
        comp_s = time.perf_counter() - tc0
        final_stats.t_pos_collect += timing.get("pos", 0.0)
        final_stats.t_qmat_build += timing.get("qmat", 0.0)
        final_stats.t_query += timing.get("query", 0.0)
        final_stats.t_filter += timing.get("filter", 0.0)
        final_stats.t_write += timing.get("write", 0.0)
        pbar.set_postfix({
            "ms": f"{comp_s * 1000:.0f}",
            "qmat_s": f"{final_stats.t_qmat_build:.1f}",
            "srch_s": f"{final_stats.t_query:.1f}",
            "flt_s": f"{final_stats.t_filter:.1f}",
        })
    pbar.close()

    final_stats.t_total = time.perf_counter() - t_start
    validate_neg_ctx_output(neg_ctx, total_n_seqs=total_n_seqs, n_sequences=n_neg)
    return final_stats


__all__ = [
    "build_neg_ctx",
    "build_neg_ctx_index_sharded",
    "build_neg_ctx_multi_gpu",
    "build_neg_ctx_single_gpu_exact",
]
