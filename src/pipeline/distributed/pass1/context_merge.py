"""Top- and mid-context merge helpers for distributed pass 1."""

from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Sequence

import torch
from tqdm import tqdm

from ..pass1_partials import Pass1PartialMetadata, load_pass1_partial, validate_pass1_partial
from .contracts import (
    MID_CTX_CANDIDATE_POOL_DEFAULTS,
    MidCtxCandidatesPartial,
    TopCtxPartial,
)


MID_CTX_WEIGHTED_RESERVOIR_HASH_VERSION = "weighted-reservoir-v1"

# SplitMix64 constants, represented as signed int64 values so Torch int64
# arithmetic wraps deterministically while mixing per-candidate priority keys.
_SPLITMIX_GOLDEN_GAMMA = -7046029254386353131  # unsigned: 0x9E3779B97F4A7C15
_SPLITMIX_MIX_1 = -4658895280553007687        # unsigned: 0xBF58476D1CE4E5B9
_SPLITMIX_MIX_2 = -7723592293110705685        # unsigned: 0x94D049BB133111EB
_WEIGHTED_RESERVOIR_CHUNK_ROWS = 8192


def load_and_merge_top_ctx_partials(
    partial_paths: Sequence[str | Path],
    *,
    expected_config_hash: str | None = None,
) -> Dict[str, object]:
    """Load top-context partial files and merge them into one canonical payload."""

    partials = [
        load_pass1_partial(
            path,
            expected_artifact_name="top_ctx",
            expected_config_hash=expected_config_hash,
        )
        for path in partial_paths
    ]
    return merge_top_ctx_partials(partials)


def merge_top_ctx_partials(
    partials: Sequence[TopCtxPartial],
) -> Dict[str, object]:
    """Merge top-context partial payloads by global top-K per component/latent."""

    if not partials:
        raise ValueError("at least one top_ctx partial is required")
    _validate_top_ctx_partial_set(partials)

    first_payload = partials[0][1]
    candidate_indices = torch.cat(
        [payload["ctx_seq_idx"].to(torch.int64) for _metadata, payload in partials],
        dim=2,
    )
    candidate_values = torch.cat(
        [payload["ctx_seq_val"].to(torch.float32) for _metadata, payload in partials],
        dim=2,
    )
    candidate_values = torch.where(
        (candidate_indices > 0) & (candidate_values > 0),
        candidate_values,
        torch.zeros_like(candidate_values),
    )
    candidate_indices = torch.where(
        candidate_values > 0,
        candidate_indices,
        torch.zeros_like(candidate_indices),
    )

    output_k = int(first_payload["ctx_seq_idx"].shape[2])
    order_by_sequence = torch.argsort(candidate_indices, dim=2, stable=True)
    values_by_sequence = candidate_values.gather(2, order_by_sequence)
    indices_by_sequence = candidate_indices.gather(2, order_by_sequence)
    order_by_value = torch.argsort(values_by_sequence, dim=2, descending=True, stable=True)
    top_positions = order_by_value[:, :, :output_k]
    merged_values = values_by_sequence.gather(2, top_positions)
    merged_indices = indices_by_sequence.gather(2, top_positions)

    valid = merged_values > 0
    merged_values = merged_values.masked_fill(~valid, 0.0)
    merged_indices = merged_indices.masked_fill(~valid, 0)
    merged = {
        "ctx_seq_idx": merged_indices.to(first_payload["ctx_seq_idx"].dtype),
        "ctx_seq_val": merged_values.to(first_payload["ctx_seq_val"].dtype),
        "ctx_type": "top",
    }
    _validate_merged_top_ctx(merged, partials)
    return merged


def load_and_merge_mid_ctx_candidate_partials(
    partial_paths: Sequence[str | Path],
    *,
    latent_stats_payload: Dict[str, object],
    expected_config_hash: str | None = None,
    num_ctx_sequences: int | None = None,
    band_low_sigma: float = 0.5,
    band_high_sigma: float = 1.5,
    on_truncation: str = "replay_fallback",
    replay_fallback_fn: Callable[..., Dict[str, object]] | None = None,
) -> Dict[str, object]:
    """Load mid-context candidate partial files and merge into canonical mid_ctx."""

    partials = [
        load_pass1_partial(
            path,
            expected_artifact_name="mid_ctx_candidates",
            expected_config_hash=expected_config_hash,
        )
        for path in partial_paths
    ]
    return merge_mid_ctx_candidate_partials(
        partials,
        latent_stats_payload=latent_stats_payload,
        num_ctx_sequences=num_ctx_sequences,
        band_low_sigma=band_low_sigma,
        band_high_sigma=band_high_sigma,
        on_truncation=on_truncation,
        replay_fallback_fn=replay_fallback_fn,
    )


def load_and_merge_mid_ctx_reservoir_partials(
    partial_paths: Sequence[str | Path],
    *,
    expected_config_hash: str | None = None,
    num_ctx_sequences: int | None = None,
    band_low_sigma: float = 0.5,
    band_high_sigma: float = 1.5,
    sampling_seed: int = 0,
    dataset_fingerprint: str = "",
) -> Dict[str, object]:
    """Load compact worker reservoir partials and merge into canonical mid_ctx."""

    load_start = time.perf_counter()
    partials = []
    print(
        f"[pass1_merge] loading {len(partial_paths)} compact mid_ctx reservoir partials",
        flush=True,
    )
    for index, path in enumerate(partial_paths, start=1):
        partial_start = time.perf_counter()
        print(
            f"[pass1_merge] loading mid_ctx reservoir partial {index}/{len(partial_paths)} -> {path}",
            flush=True,
        )
        partials.append(
            load_pass1_partial(
                path,
                expected_artifact_name="mid_ctx_candidates",
                expected_config_hash=expected_config_hash,
            )
        )
        print(
            "[pass1_merge] loaded mid_ctx reservoir partial "
            f"{index}/{len(partial_paths)} elapsed={time.perf_counter() - partial_start:.1f}s",
            flush=True,
        )
    print(
        f"[pass1_merge] loaded all compact mid_ctx reservoir partials elapsed={time.perf_counter() - load_start:.1f}s",
        flush=True,
    )
    merge_start = time.perf_counter()
    merged = merge_mid_ctx_reservoir_partials(
        partials,
        num_ctx_sequences=num_ctx_sequences,
        band_low_sigma=band_low_sigma,
        band_high_sigma=band_high_sigma,
        sampling_seed=sampling_seed,
        dataset_fingerprint=dataset_fingerprint,
    )
    print(
        f"[pass1_merge] compact mid_ctx reservoir merge elapsed={time.perf_counter() - merge_start:.1f}s",
        flush=True,
    )
    return merged


def merge_mid_ctx_candidate_partials(
    partials: Sequence[MidCtxCandidatesPartial],
    *,
    latent_stats_payload: Dict[str, object],
    num_ctx_sequences: int | None = None,
    band_low_sigma: float = 0.5,
    band_high_sigma: float = 1.5,
    on_truncation: str = "replay_fallback",
    replay_fallback_fn: Callable[..., Dict[str, object]] | None = None,
) -> Dict[str, object]:
    """Filter candidate pools with global stats and select deterministic priorities."""

    if not partials:
        raise ValueError("at least one mid_ctx candidate partial is required")
    if on_truncation not in {"fail", "replay_fallback", "allow_bounded_approx"}:
        raise ValueError("unsupported mid_ctx candidate truncation policy")
    _validate_mid_ctx_candidate_partial_set(partials)
    _validate_latent_stats_for_mid_ctx(latent_stats_payload, partials[0][0])

    metadata = partials[0][0]
    first_payload = partials[0][1]
    output_k = int(
        num_ctx_sequences
        or first_payload.get("candidate_pool_settings", {}).get("final_num_ctx_sequences", 0)
        or first_payload.get("candidate_pool_settings", {}).get("num_ctx_sequences", 0)
        or first_payload["ctx_seq_idx"].shape[2]
    )
    if output_k < 1:
        raise ValueError("num_ctx_sequences must be >= 1")

    component_count = metadata.component_count
    d_sae = metadata.d_sae
    ctx_seq_idx = torch.zeros((component_count, d_sae, output_k), dtype=torch.int32)
    ctx_seq_val = torch.zeros((component_count, d_sae, output_k), dtype=torch.float32)
    reservoir_fill = torch.zeros((component_count, d_sae), dtype=torch.int32)
    reservoir_n = torch.zeros((component_count, d_sae), dtype=torch.int64)
    candidate_count = torch.zeros((component_count, d_sae), dtype=torch.int64)
    truncation_counters = torch.zeros((component_count, d_sae), dtype=torch.int64)

    candidates = _concatenate_mid_ctx_candidates(partials)
    if candidates["component_ids"].numel() > 0:
        component_ids = candidates["component_ids"].to(torch.long)
        latent_ids = candidates["latent_ids"].to(torch.long)
        sequence_ids = candidates["sequence_ids"].to(torch.int32)
        activation_values = candidates["activation_values"].to(torch.float32)
        priorities = candidates["priorities"].to(torch.int64)
        flat_ids = component_ids * d_sae + latent_ids
        candidate_count.view(-1).scatter_add_(
            0,
            flat_ids,
            torch.ones_like(flat_ids, dtype=torch.int64),
        )

        mean_seq = latent_stats_payload["mean_seq"].to(torch.float32)
        std_seq = _std_seq_from_latent_stats(latent_stats_payload)
        low = mean_seq[component_ids, latent_ids] + band_low_sigma * std_seq[component_ids, latent_ids]
        high = mean_seq[component_ids, latent_ids] + band_high_sigma * std_seq[component_ids, latent_ids]
        valid = (activation_values > low) & (activation_values < high)
        if bool(valid.any()):
            valid_components = component_ids[valid]
            valid_latents = latent_ids[valid]
            valid_sequences = sequence_ids[valid]
            valid_values = activation_values[valid]
            valid_priorities = priorities[valid]
            valid_flat_ids = flat_ids[valid]
            reservoir_n.view(-1).scatter_add_(
                0,
                valid_flat_ids,
                torch.ones_like(valid_flat_ids, dtype=torch.int64),
            )
            _select_mid_ctx_candidates(
                ctx_seq_idx,
                ctx_seq_val,
                reservoir_fill,
                valid_components,
                valid_latents,
                valid_sequences,
                valid_values,
                valid_priorities,
                output_k,
            )

    for _metadata, payload in partials:
        truncation_counters += payload["truncation_counters"].to(torch.int64)
    requires_replay_fallback = bool((truncation_counters > 0).any())
    if requires_replay_fallback and on_truncation == "fail":
        raise ValueError("mid_ctx candidate pool truncation detected")
    if requires_replay_fallback and on_truncation == "replay_fallback" and replay_fallback_fn:
        replayed = replay_fallback_fn(
            partials=partials,
            latent_stats_payload=latent_stats_payload,
            num_ctx_sequences=output_k,
            band_low_sigma=band_low_sigma,
            band_high_sigma=band_high_sigma,
        )
        replayed.setdefault("merge_report", {})
        replayed["merge_report"].update(
            {
                "mode": "stats_aware_replay_fallback",
                "replay_fallback_executed": True,
                "truncation_counters": truncation_counters,
                "num_ctx_sequences": int(output_k),
            }
        )
        _validate_merged_mid_ctx(replayed, partials)
        return replayed

    merge_report = {
        "candidate_count": candidate_count,
        "valid_count": reservoir_n,
        "selected_count": reservoir_fill.to(torch.int64),
        "truncation_counters": truncation_counters,
        "requires_replay_fallback": requires_replay_fallback
        and on_truncation == "replay_fallback",
        "replay_fallback_executed": False,
        "bounded_approximation": requires_replay_fallback
        and on_truncation == "allow_bounded_approx",
        "candidate_pool_cleanup_eligible": not requires_replay_fallback,
        "fill_rate": reservoir_fill.float() / float(output_k),
        "band_low_sigma": float(band_low_sigma),
        "band_high_sigma": float(band_high_sigma),
        "num_ctx_sequences": int(output_k),
        "priority_mode": "deterministic_priority_reservoir",
        "candidate_pool_defaults": {
            **MID_CTX_CANDIDATE_POOL_DEFAULTS,
            "max_candidates_per_latent": max(256, 4 * output_k),
        },
        "candidate_pool_settings": dict(first_payload.get("candidate_pool_settings", {})),
        "priority_hash_version": first_payload.get("candidate_pool_settings", {}).get(
            "priority_hash_version"
        ),
    }
    merged = {
        "ctx_seq_idx": ctx_seq_idx,
        "ctx_seq_val": ctx_seq_val,
        "ctx_type": "mid",
        "mode": "distributed_priority_reservoir",
        "band_low_sigma": float(band_low_sigma),
        "band_high_sigma": float(band_high_sigma),
        "num_ctx_sequences": int(output_k),
        "reservoir_fill": reservoir_fill,
        "reservoir_n": reservoir_n,
        "merge_report": merge_report,
    }
    _validate_merged_mid_ctx(merged, partials)
    return merged


def merge_mid_ctx_reservoir_partials(
    partials: Sequence[MidCtxCandidatesPartial],
    *,
    num_ctx_sequences: int | None = None,
    band_low_sigma: float = 0.5,
    band_high_sigma: float = 1.5,
    sampling_seed: int = 0,
    dataset_fingerprint: str = "",
) -> Dict[str, object]:
    """Merge compact worker-local mid-context reservoirs by weighted sampling."""

    if not partials:
        raise ValueError("at least one mid_ctx reservoir partial is required")
    _validate_mid_ctx_reservoir_partial_set(partials)

    metadata = partials[0][0]
    first_payload = partials[0][1]
    output_k = int(num_ctx_sequences or first_payload["ctx_seq_idx"].shape[2])
    if output_k < 1:
        raise ValueError("num_ctx_sequences must be >= 1")

    component_count = metadata.component_count
    d_sae = metadata.d_sae
    total_rows = component_count * d_sae
    merge_start = time.perf_counter()
    print(
        "[pass1_merge] merging mid_ctx weighted reservoirs "
        f"workers={len(partials)} components={component_count} d_sae={d_sae} "
        f"rows={total_rows} output_k={output_k}",
        flush=True,
    )
    ctx_seq_idx, ctx_seq_val, reservoir_fill, reservoir_n, empty_worker_rows = (
        _merge_mid_ctx_reservoir_partials_chunked(
            partials,
            component_count=component_count,
            d_sae=d_sae,
            output_k=output_k,
            sampling_seed=sampling_seed,
            dataset_fingerprint=dataset_fingerprint,
            merge_start=merge_start,
        )
    )

    merge_report = {
        "mode": "weighted_reservoir",
        "merge_mode": "weighted_reservoir",
        "selected_count": reservoir_fill.to(torch.int64),
        "valid_count": reservoir_n,
        "total_reservoir_n": reservoir_n,
        "empty_worker_rows": empty_worker_rows,
        "any_worker_reservoir_empty": bool((empty_worker_rows > 0).any()),
        "fill_rate": reservoir_fill.float() / float(output_k),
        "band_low_sigma": float(band_low_sigma),
        "band_high_sigma": float(band_high_sigma),
        "num_ctx_sequences": int(output_k),
        "sampling_seed": int(sampling_seed),
        "dataset_fingerprint": str(dataset_fingerprint),
        "priority_mode": "deterministic_weighted_reservoir",
        "priority_hash_version": MID_CTX_WEIGHTED_RESERVOIR_HASH_VERSION,
    }
    merged = {
        "ctx_seq_idx": ctx_seq_idx,
        "ctx_seq_val": ctx_seq_val,
        "ctx_type": "mid",
        "mode": "distributed_weighted_reservoir",
        "band_low_sigma": float(band_low_sigma),
        "band_high_sigma": float(band_high_sigma),
        "num_ctx_sequences": int(output_k),
        "reservoir_fill": reservoir_fill,
        "reservoir_n": reservoir_n,
        "merge_report": merge_report,
    }
    _validate_merged_mid_ctx(merged, partials)
    return merged


def _merge_mid_ctx_reservoir_partials_chunked(
    partials: Sequence[MidCtxCandidatesPartial],
    *,
    component_count: int,
    d_sae: int,
    output_k: int,
    sampling_seed: int,
    dataset_fingerprint: str,
    merge_start: float,
    chunk_rows: int = _WEIGHTED_RESERVOIR_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked tensor implementation of weighted reservoir merging."""

    sorted_partials = sorted(partials, key=lambda item: item[0].worker_id)
    total_rows = component_count * d_sae
    input_k = int(sorted_partials[0][1]["ctx_seq_idx"].shape[2])
    candidates_per_row = len(sorted_partials) * input_k
    selected_k = min(output_k, candidates_per_row)
    worker_ids = torch.tensor(
        [int(metadata.worker_id) for metadata, _payload in sorted_partials],
        dtype=torch.int64,
    ).view(-1, 1, 1)
    base_seed = _weighted_reservoir_hash_base(
        sampling_seed=sampling_seed,
        dataset_fingerprint=dataset_fingerprint,
    )
    base_seed_tensor = torch.tensor(base_seed, dtype=torch.int64)
    slot_ids = torch.arange(input_k, dtype=torch.int64).view(1, 1, input_k)

    ctx_seq_idx = torch.zeros((component_count, d_sae, output_k), dtype=torch.int32)
    ctx_seq_val = torch.zeros((component_count, d_sae, output_k), dtype=torch.float32)
    reservoir_fill = torch.zeros((component_count, d_sae), dtype=torch.int32)
    reservoir_n = torch.zeros((component_count, d_sae), dtype=torch.int64)
    empty_worker_rows = torch.zeros((component_count, d_sae), dtype=torch.int64)

    chunk_ranges = range(0, total_rows, chunk_rows)
    progress = tqdm(
        chunk_ranges,
        total=math.ceil(total_rows / chunk_rows),
        desc="  [pass1_merge:mid_ctx]",
        unit="chunk",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for row_start in progress:
        row_end = min(total_rows, row_start + chunk_rows)
        row_count = row_end - row_start
        chunk_start = time.perf_counter()

        idx_chunk = torch.stack(
            [
                payload["ctx_seq_idx"].reshape(total_rows, input_k)[row_start:row_end].to(torch.int64)
                for _metadata, payload in sorted_partials
            ],
            dim=0,
        )
        val_chunk = torch.stack(
            [
                payload["ctx_seq_val"].reshape(total_rows, input_k)[row_start:row_end].to(torch.float32)
                for _metadata, payload in sorted_partials
            ],
            dim=0,
        )
        fill_chunk = torch.stack(
            [
                payload["reservoir_fill"].reshape(total_rows)[row_start:row_end].to(torch.int64)
                for _metadata, payload in sorted_partials
            ],
            dim=0,
        )
        n_chunk = torch.stack(
            [
                payload["reservoir_n"].reshape(total_rows)[row_start:row_end].to(torch.int64)
                for _metadata, payload in sorted_partials
            ],
            dim=0,
        )

        row_ids = torch.arange(row_start, row_end, dtype=torch.int64)
        component_ids = (row_ids // d_sae).view(1, row_count, 1)
        latent_ids = (row_ids % d_sae).view(1, row_count, 1)
        fill_expanded = fill_chunk.unsqueeze(-1)
        n_expanded = n_chunk.unsqueeze(-1)
        slot_mask = slot_ids[:, :, :input_k] < fill_expanded
        valid = slot_mask & (idx_chunk > 0) & torch.isfinite(val_chunk)

        weights = n_expanded.to(torch.float64) / fill_expanded.clamp(min=1).to(torch.float64)
        uniforms = _weighted_reservoir_uniform_tensor(
            base_seed=base_seed_tensor,
            component_ids=component_ids,
            latent_ids=latent_ids,
            worker_ids=worker_ids,
            sequence_ids=idx_chunk,
            slot_ids=slot_ids[:, :, :input_k],
            reservoir_n=n_expanded,
        )
        keys = -torch.log(uniforms) / weights.clamp(min=1e-12)
        keys = keys.masked_fill(~valid, float("inf"))

        flat_keys = keys.permute(1, 0, 2).reshape(row_count, candidates_per_row)
        flat_idx = idx_chunk.permute(1, 0, 2).reshape(row_count, candidates_per_row)
        flat_val = val_chunk.permute(1, 0, 2).reshape(row_count, candidates_per_row)
        selected_keys, selected_positions = torch.topk(
            flat_keys,
            k=selected_k,
            dim=1,
            largest=False,
        )
        selected_idx = flat_idx.gather(1, selected_positions).to(torch.int32)
        selected_val = flat_val.gather(1, selected_positions).to(torch.float32)
        selected_valid = torch.isfinite(selected_keys)
        selected_idx = selected_idx.masked_fill(~selected_valid, 0)
        selected_val = selected_val.masked_fill(~selected_valid, 0.0)

        out_idx = torch.zeros((row_count, output_k), dtype=torch.int32)
        out_val = torch.zeros((row_count, output_k), dtype=torch.float32)
        out_idx[:, :selected_k] = selected_idx
        out_val[:, :selected_k] = selected_val

        ctx_seq_idx.view(total_rows, output_k)[row_start:row_end] = out_idx
        ctx_seq_val.view(total_rows, output_k)[row_start:row_end] = out_val
        reservoir_fill.view(-1)[row_start:row_end] = selected_valid.sum(dim=1).to(torch.int32)
        reservoir_n.view(-1)[row_start:row_end] = n_chunk.sum(dim=0)
        empty_worker_rows.view(-1)[row_start:row_end] = (
            (n_chunk <= 0) | (fill_chunk <= 0)
        ).sum(dim=0).to(torch.int64)

        rows_done = row_end
        progress.set_postfix(
            rows=f"{rows_done}/{total_rows}",
            chunk_s=f"{time.perf_counter() - chunk_start:.2f}",
        )

    return ctx_seq_idx, ctx_seq_val, reservoir_fill, reservoir_n, empty_worker_rows


def merge_mid_ctx_reservoir_row(
    partials: Sequence[MidCtxCandidatesPartial],
    *,
    component_id: int,
    latent_id: int,
    output_k: int,
    sampling_seed: int = 0,
    dataset_fingerprint: str = "",
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Merge one compact worker-local reservoir row with weighted keys."""

    if output_k < 1:
        raise ValueError("output_k must be >= 1")
    if component_id < 0 or latent_id < 0:
        raise ValueError("component_id and latent_id must be non-negative")

    total_reservoir_n = 0
    candidates: list[tuple[float, int, int, int, float]] = []
    for metadata, payload in sorted(partials, key=lambda item: item[0].worker_id):
        if component_id >= metadata.component_count or latent_id >= metadata.d_sae:
            raise ValueError("component_id or latent_id outside partial shape")
        worker_id = int(metadata.worker_id)
        worker_n = int(payload["reservoir_n"][component_id, latent_id].item())
        total_reservoir_n += worker_n
        worker_fill = int(payload["reservoir_fill"][component_id, latent_id].item())
        if worker_n <= 0 or worker_fill <= 0:
            continue

        row_indices = payload["ctx_seq_idx"][component_id, latent_id].to(torch.int64)
        row_values = payload["ctx_seq_val"][component_id, latent_id].to(torch.float32)
        selected_slots = min(worker_fill, int(row_indices.numel()))
        weight = float(worker_n) / float(max(worker_fill, 1))
        for slot_id in range(selected_slots):
            sequence_id = int(row_indices[slot_id].item())
            activation_value = float(row_values[slot_id].item())
            if sequence_id == 0 or not math.isfinite(activation_value):
                continue
            uniform = _weighted_reservoir_uniform(
                sampling_seed=sampling_seed,
                dataset_fingerprint=dataset_fingerprint,
                component_id=component_id,
                latent_id=latent_id,
                worker_id=worker_id,
                sequence_id=sequence_id,
                slot_id=slot_id,
                reservoir_n=worker_n,
            )
            key = -math.log(uniform) / weight
            candidates.append((key, sequence_id, worker_id, slot_id, activation_value))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    selected = candidates[:output_k]
    merged_indices = torch.zeros(output_k, dtype=torch.int32)
    merged_values = torch.zeros(output_k, dtype=torch.float32)
    for output_slot, (_key, sequence_id, _worker_id, _slot_id, activation_value) in enumerate(selected):
        merged_indices[output_slot] = int(sequence_id)
        merged_values[output_slot] = float(activation_value)
    return merged_indices, merged_values, len(selected), total_reservoir_n


def _weighted_reservoir_uniform(
    *,
    sampling_seed: int,
    dataset_fingerprint: str,
    component_id: int,
    latent_id: int,
    worker_id: int,
    sequence_id: int,
    slot_id: int,
    reservoir_n: int,
) -> float:
    base_seed = torch.tensor(
        _weighted_reservoir_hash_base(
            sampling_seed=sampling_seed,
            dataset_fingerprint=dataset_fingerprint,
        ),
        dtype=torch.int64,
    )
    uniform = _weighted_reservoir_uniform_tensor(
        base_seed=base_seed,
        component_ids=torch.tensor([[[int(component_id)]]], dtype=torch.int64),
        latent_ids=torch.tensor([[[int(latent_id)]]], dtype=torch.int64),
        worker_ids=torch.tensor([[[int(worker_id)]]], dtype=torch.int64),
        sequence_ids=torch.tensor([[[int(sequence_id)]]], dtype=torch.int64),
        slot_ids=torch.tensor([[[int(slot_id)]]], dtype=torch.int64),
        reservoir_n=torch.tensor([[[int(reservoir_n)]]], dtype=torch.int64),
    )
    return float(uniform.item())


def _weighted_reservoir_hash_base(
    *,
    sampling_seed: int,
    dataset_fingerprint: str,
) -> int:
    material = "|".join(
        [
            MID_CTX_WEIGHTED_RESERVOIR_HASH_VERSION,
            str(int(sampling_seed)),
            str(dataset_fingerprint),
        ]
    )
    value = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    if value >= (1 << 63):
        value -= 1 << 64
    return value


def _weighted_reservoir_uniform_tensor(
    *,
    base_seed: torch.Tensor,
    component_ids: torch.Tensor,
    latent_ids: torch.Tensor,
    worker_ids: torch.Tensor,
    sequence_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    reservoir_n: torch.Tensor,
) -> torch.Tensor:
    values = base_seed + component_ids.to(torch.int64) * _SPLITMIX_GOLDEN_GAMMA
    values = values + latent_ids.to(torch.int64) * _SPLITMIX_MIX_1
    values = values + worker_ids.to(torch.int64) * _SPLITMIX_MIX_2
    values = _splitmix64(values)
    values = values + sequence_ids.to(torch.int64) * _SPLITMIX_MIX_1
    values = values + slot_ids.to(torch.int64) * _SPLITMIX_MIX_2
    values = values + reservoir_n.to(torch.int64) * _SPLITMIX_GOLDEN_GAMMA
    values = _splitmix64(values)
    positive = torch.bitwise_and(values, 0x7FFFFFFFFFFFFFFF).to(torch.float64)
    return (positive + 1.0) / float(1 << 63)


def _splitmix64(values: torch.Tensor) -> torch.Tensor:
    values = values + _SPLITMIX_GOLDEN_GAMMA
    values = torch.bitwise_xor(values, values >> 30) * _SPLITMIX_MIX_1
    values = torch.bitwise_xor(values, values >> 27) * _SPLITMIX_MIX_2
    return torch.bitwise_xor(values, values >> 31)


def _validate_top_ctx_partial_set(
    partials: Sequence[TopCtxPartial],
) -> None:
    seen_workers: set[int] = set()
    first_metadata = partials[0][0]
    first_shape = partials[0][1]["ctx_seq_idx"].shape
    for metadata, payload in partials:
        if metadata.artifact_name != "top_ctx":
            raise ValueError("all partials must be top_ctx artifacts")
        if metadata.worker_id in seen_workers:
            raise ValueError(f"duplicate top_ctx partial for worker {metadata.worker_id}")
        seen_workers.add(metadata.worker_id)
        if metadata.run_id != first_metadata.run_id:
            raise ValueError("top_ctx partial run_id mismatch")
        if metadata.config_hash != first_metadata.config_hash:
            raise ValueError("top_ctx partial config hash mismatch")
        if metadata.component_count != first_metadata.component_count:
            raise ValueError("top_ctx partial component count mismatch")
        if metadata.d_sae != first_metadata.d_sae:
            raise ValueError("top_ctx partial d_sae mismatch")
        if payload["ctx_seq_idx"].shape != first_shape:
            raise ValueError("top_ctx partial shape mismatch")
        validate_pass1_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_artifact_name="top_ctx",
            expected_config_hash=first_metadata.config_hash,
        )
        _validate_top_ctx_sequence_range(metadata, payload)


def _validate_mid_ctx_candidate_partial_set(
    partials: Sequence[MidCtxCandidatesPartial],
) -> None:
    seen_workers: set[int] = set()
    first_metadata = partials[0][0]
    for metadata, payload in partials:
        if metadata.artifact_name != "mid_ctx_candidates":
            raise ValueError("all partials must be mid_ctx_candidates artifacts")
        if metadata.worker_id in seen_workers:
            raise ValueError(
                f"duplicate mid_ctx_candidates partial for worker {metadata.worker_id}"
            )
        seen_workers.add(metadata.worker_id)
        if metadata.run_id != first_metadata.run_id:
            raise ValueError("mid_ctx_candidates partial run_id mismatch")
        if metadata.config_hash != first_metadata.config_hash:
            raise ValueError("mid_ctx_candidates partial config hash mismatch")
        if metadata.component_count != first_metadata.component_count:
            raise ValueError("mid_ctx_candidates partial component count mismatch")
        if metadata.d_sae != first_metadata.d_sae:
            raise ValueError("mid_ctx_candidates partial d_sae mismatch")
        validate_pass1_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_artifact_name="mid_ctx_candidates",
            expected_config_hash=first_metadata.config_hash,
        )


def _validate_top_ctx_sequence_range(
    metadata: Pass1PartialMetadata,
    payload: Dict[str, object],
) -> None:
    sequence_ids = payload["ctx_seq_idx"]
    valid = sequence_ids != 0
    if not bool(valid.any()):
        return
    if metadata.sequence_id_min is not None and int(sequence_ids[valid].min()) < metadata.sequence_id_min:
        raise ValueError("top_ctx sequence IDs below worker range")
    if metadata.sequence_id_max is not None and int(sequence_ids[valid].max()) > metadata.sequence_id_max:
        raise ValueError("top_ctx sequence IDs above worker range")


def _validate_merged_top_ctx(
    merged: Dict[str, object],
    partials: Sequence[TopCtxPartial],
) -> None:
    metadata = partials[0][0]
    validate_pass1_partial(
        {"metadata": metadata.model_copy(update={"worker_id": 0}).model_dump(mode="json"), "payload": merged},
        expected_artifact_name="top_ctx",
        expected_config_hash=metadata.config_hash,
    )
    values = merged["ctx_seq_val"].float()
    indices = merged["ctx_seq_idx"]
    if bool((values < 0).any()):
        raise ValueError("merged top_ctx values must be non-negative")
    global_min = min(
        item.sequence_id_min for item, _payload in partials if item.sequence_id_min is not None
    )
    global_max = max(
        item.sequence_id_max for item, _payload in partials if item.sequence_id_max is not None
    )
    valid = indices != 0
    if bool(valid.any()):
        if int(indices[valid].min()) < global_min:
            raise ValueError("merged top_ctx sequence IDs below global range")
        if int(indices[valid].max()) > global_max:
            raise ValueError("merged top_ctx sequence IDs above global range")
    invalid = ~valid
    if bool((values[invalid] != 0).any()):
        raise ValueError("merged top_ctx invalid sentinel values must be zero")


def _validate_latent_stats_for_mid_ctx(
    latent_stats_payload: Dict[str, object],
    metadata: Pass1PartialMetadata,
) -> None:
    shape = (metadata.component_count, metadata.d_sae)
    for name in ["seq_count", "mean_seq", "m2_seq"]:
        tensor = latent_stats_payload.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"latent_stats payload missing valid {name}")
        if name != "seq_count" and not torch.isfinite(tensor.float()).all():
            raise ValueError(f"latent_stats {name} contains non-finite values")


def _std_seq_from_latent_stats(latent_stats_payload: Dict[str, object]) -> torch.Tensor:
    seq_count = latent_stats_payload["seq_count"].to(torch.float32)
    m2_seq = latent_stats_payload["m2_seq"].to(torch.float32)
    return (m2_seq / (seq_count - 1).clamp(min=1)).clamp(min=0).sqrt()


def _validate_mid_ctx_reservoir_partial_set(
    partials: Sequence[MidCtxCandidatesPartial],
) -> None:
    _validate_mid_ctx_candidate_partial_set(partials)
    for _metadata, payload in partials:
        if payload.get("merge_source") != "worker_local_reservoir":
            raise ValueError(
                "weighted reservoir merge requires compact worker_local_reservoir partials"
            )


def _concatenate_mid_ctx_candidates(
    partials: Sequence[MidCtxCandidatesPartial],
) -> Dict[str, torch.Tensor]:
    names = [
        "component_ids",
        "latent_ids",
        "sequence_ids",
        "activation_values",
        "priorities",
    ]
    return {
        name: torch.cat([payload[name] for _metadata, payload in partials], dim=0)
        for name in names
    }


def _select_mid_ctx_candidates(
    ctx_seq_idx: torch.Tensor,
    ctx_seq_val: torch.Tensor,
    reservoir_fill: torch.Tensor,
    component_ids: torch.Tensor,
    latent_ids: torch.Tensor,
    sequence_ids: torch.Tensor,
    activation_values: torch.Tensor,
    priorities: torch.Tensor,
    output_k: int,
) -> None:
    groups = torch.stack([component_ids, latent_ids], dim=1).unique(dim=0)
    for component_id, latent_id in groups.tolist():
        mask = (component_ids == component_id) & (latent_ids == latent_id)
        group_priorities = priorities[mask]
        group_sequences = sequence_ids[mask].to(torch.long)
        group_values = activation_values[mask]
        order_by_sequence = torch.argsort(group_sequences, stable=True)
        priority_sorted = group_priorities[order_by_sequence]
        sequences_sorted = group_sequences[order_by_sequence]
        values_sorted = group_values[order_by_sequence]
        order_by_priority = torch.argsort(priority_sorted, stable=True)
        selected = order_by_priority[:output_k]
        selected_count = int(selected.numel())
        if selected_count == 0:
            continue
        ctx_seq_idx[component_id, latent_id, :selected_count] = sequences_sorted[
            selected
        ].to(torch.int32)
        ctx_seq_val[component_id, latent_id, :selected_count] = values_sorted[
            selected
        ].to(torch.float32)
        reservoir_fill[component_id, latent_id] = selected_count


def _validate_merged_mid_ctx(
    merged: Dict[str, object],
    partials: Sequence[MidCtxCandidatesPartial],
) -> None:
    metadata = partials[0][0]
    idx = merged["ctx_seq_idx"]
    vals = merged["ctx_seq_val"]
    if idx.dtype != torch.int32:
        raise ValueError("merged mid_ctx sequence IDs must be int32")
    if idx.shape != vals.shape:
        raise ValueError("merged mid_ctx tensors must have matching shapes")
    if idx.ndim != 3 or idx.shape[0] != metadata.component_count or idx.shape[1] != metadata.d_sae:
        raise ValueError("merged mid_ctx tensors have invalid shape")
    if not torch.isfinite(vals.float()).all():
        raise ValueError("merged mid_ctx values contain non-finite values")
    if bool((vals < 0).any()):
        raise ValueError("merged mid_ctx values must be non-negative")
    invalid = idx == 0
    if bool((vals[invalid] != 0).any()):
        raise ValueError("merged mid_ctx invalid sentinel values must be zero")


__all__ = [
    "MID_CTX_WEIGHTED_RESERVOIR_HASH_VERSION",
    "load_and_merge_mid_ctx_candidate_partials",
    "load_and_merge_mid_ctx_reservoir_partials",
    "load_and_merge_top_ctx_partials",
    "merge_mid_ctx_reservoir_row",
    "merge_mid_ctx_candidate_partials",
    "merge_mid_ctx_reservoir_partials",
    "merge_top_ctx_partials",
]
