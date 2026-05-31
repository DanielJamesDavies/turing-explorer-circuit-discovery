"""Pass-1 distributed worker implementation."""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from config import config
from data.loader import DataLoader
from model.inference import Inference
from pipeline.first_pass import run_first_pass
from pipeline.runtime import build_distributed_worker_runtime, clear_runtime, get_runtime, set_runtime
from sae.bank import SAEBank
from store.context import mid_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.seq_repr import SeqRepr

from ..interfaces import get_worker_shard_ids
from ..layout import build_run_layout, build_worker_marker, write_worker_marker
from ..manifest import DistributedRunManifest
from ..pass1_partials import (
    build_pass1_partial_metadata,
    latent_stats_payload,
    logit_ctx_payload,
    mid_ctx_candidates_payload,
    save_pass1_partial,
    seq_repr_payload,
    top_ctx_payload,
)
from ..seq_repr_mapping import build_seq_repr_cap_mapping, shard_table_fingerprint
from ..shard_table import sequence_ids_for_shards, validate_shard_table
from ..worker_common import (
    _device_assignment_for_worker,
    _total_sequences,
    _utc_now,
    _worker_batch_count,
)
from .contracts import PASS1_PARTIAL_FILENAMES


def run_pass1_worker(
    manifest: DistributedRunManifest,
    worker_id: int,
    *,
    initialize_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None,
    run_first_pass_fn: Callable[..., None] = run_first_pass,
    save_partials_fn: Optional[Callable[[DistributedRunManifest, int], Dict[str, str]]] = None,
    validate_inputs_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None,
) -> Dict[str, str]:
    """Run first-pass work for one worker and save worker-local partials."""

    initialize = initialize_fn or initialize_pass1_worker_resources
    save_partials = save_partials_fn or save_pass1_partials
    validate_inputs = validate_inputs_fn or validate_pass1_worker_inputs
    layout = build_run_layout(manifest)
    worker_layout = layout.workers[worker_id]
    shard_ids = get_worker_shard_ids(manifest, worker_id)
    sequence_count = manifest.work_assignments.pass1_sequence_totals.get(str(worker_id), 0)
    batch_count = 0
    start_time = _utc_now()
    started_at = time.perf_counter()

    write_worker_marker(
        build_worker_marker(
            manifest,
            worker_id,
            phase="pass1",
            status="started",
            start_time=start_time,
            sequence_count=sequence_count,
        ),
        worker_layout.started_marker,
    )

    try:
        validate_inputs(manifest, worker_id)
        initialize(manifest, worker_id)
        batch_count = _worker_batch_count(shard_ids)
        run_first_pass_fn(
            assigned_shard_ids=shard_ids,
            seq_latent_index_output_dir=str(worker_layout.pass1_dir / "seq_latent_index"),
        )
        artifacts = save_partials(manifest, worker_id)
        end_time = _utc_now()
        write_worker_marker(
            build_worker_marker(
                manifest,
                worker_id,
                phase="pass1",
                status="completed",
                start_time=start_time,
                end_time=end_time,
                duration_s=time.perf_counter() - started_at,
                batch_count=batch_count,
                sequence_count=sequence_count,
                artifacts=artifacts,
            ),
            worker_layout.completed_marker,
        )
        return artifacts
    except Exception as error:
        write_worker_marker(
            build_worker_marker(
                manifest,
                worker_id,
                phase="pass1",
                status="failed",
                start_time=start_time,
                end_time=_utc_now(),
                duration_s=time.perf_counter() - started_at,
                batch_count=batch_count,
                sequence_count=sequence_count,
                error=str(error),
            ),
            worker_layout.failed_marker,
        )
        raise
    finally:
        clear_runtime()


def initialize_pass1_worker_resources(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> None:
    """Initialize DataLoader, model, SAE bank, and sequence store for one worker."""

    validate_pass1_worker_inputs(manifest, worker_id)
    assignment = _device_assignment_for_worker(manifest, worker_id)
    runtime = build_distributed_worker_runtime(assignment)
    set_runtime(runtime)

    runtime.loader = DataLoader(device=runtime.device, pin_memory=runtime.fast)
    seq_repr_mapping = build_seq_repr_cap_mapping(
        total_sequence_count=_total_sequences(manifest.shard_table),
        max_repr_seqs=(
            int(config.latents.neg_ctx.max_repr_seqs)
            if config.latents.neg_ctx.max_repr_seqs is not None
            else None
        ),
        sampling_seed=manifest.sampling_seed,
        dataset_fingerprint=shard_table_fingerprint(manifest.shard_table),
    )
    runtime.seq_repr = SeqRepr(
        n_seqs=int(seq_repr_mapping["n_seqs"]),
        slot_to_id=seq_repr_mapping["slot_to_id"],
        id_to_slot=seq_repr_mapping["id_to_slot"],
    )
    runtime.model = Inference(device=runtime.device, compile=runtime.compile)
    runtime.bank = SAEBank(
        devices=runtime.devices,
        load_decoders=runtime.fast,
        compile=runtime.compile,
    )
    configure_mid_ctx_candidate_pool(manifest)


def configure_mid_ctx_candidate_pool(manifest: DistributedRunManifest) -> None:
    """Widen and oversize worker mid_ctx collection for merge-side filtering."""

    final_n_sequences = int(config.latents.mid_ctx.n_sequences or 64)
    pool_cfg = config.distributed.mid_ctx_candidate_pool
    band_margin_sigma = float(pool_cfg.band_margin_sigma)
    max_candidates_per_latent = int(
        pool_cfg.max_candidates_per_latent
        if pool_cfg.max_candidates_per_latent is not None
        else max(256, 4 * final_n_sequences)
    )
    final_band_low = float(config.latents.mid_ctx.band_low_sigma or 0.5)
    final_band_high = float(config.latents.mid_ctx.band_high_sigma or 1.5)

    if getattr(mid_ctx, "_allocated", False):
        raise RuntimeError("mid_ctx candidate pool must be configured before allocation")
    mid_ctx._distributed_candidate_pool = True
    mid_ctx._final_num_ctx_sequences = final_n_sequences
    mid_ctx._candidate_band_margin = band_margin_sigma
    mid_ctx._final_band_low = final_band_low
    mid_ctx._final_band_high = final_band_high
    mid_ctx._band_low = max(0.0, final_band_low - band_margin_sigma)
    mid_ctx._band_high = final_band_high + band_margin_sigma
    mid_ctx.num_ctx_sequences = max_candidates_per_latent
    mid_ctx._candidate_pool_dataset_fingerprint = shard_table_fingerprint(manifest.shard_table)


def validate_pass1_worker_inputs(
    manifest: DistributedRunManifest,
    worker_id: int,
    *,
    validate_on_disk: bool = True,
) -> None:
    """Validate shard table and pass-1 assignments before loading model resources."""

    if worker_id < 0 or worker_id >= manifest.worker_count:
        raise ValueError("worker_id out of range")
    if not manifest.shard_table:
        raise ValueError("manifest shard_table is required for pass1 workers")
    if validate_on_disk:
        validate_shard_table(manifest.dataset_path, manifest.shard_table)

    expected_shards = {record.shard_index for record in manifest.shard_table}
    assigned_by_worker = manifest.work_assignments.pass1_shards
    seen_shards: Dict[int, str] = {}
    for worker_key, shard_ids in assigned_by_worker.items():
        try:
            parsed_worker_id = int(worker_key)
        except ValueError as exc:
            raise ValueError("pass1 assignment worker keys must be integer strings") from exc
        if parsed_worker_id < 0 or parsed_worker_id >= manifest.worker_count:
            raise ValueError("pass1 assignment worker key out of range")
        for shard_id in shard_ids:
            if shard_id not in expected_shards:
                raise ValueError(f"assigned shard index out of range: {shard_id}")
            if shard_id in seen_shards:
                raise ValueError(
                    f"assigned shard index duplicated across workers: {shard_id}"
                )
            seen_shards[shard_id] = worker_key

    missing_shards = expected_shards - set(seen_shards)
    if missing_shards:
        raise ValueError(f"pass1 shard assignments missing shards: {sorted(missing_shards)}")

    extra_shards = set(seen_shards) - expected_shards
    if extra_shards:
        raise ValueError(f"pass1 shard assignments include extra shards: {sorted(extra_shards)}")

    assigned_shards = assigned_by_worker.get(str(worker_id), [])
    sequence_ids = sequence_ids_for_shards(manifest.shard_table, assigned_shards)
    expected_total = len(sequence_ids)
    declared_total = manifest.work_assignments.pass1_sequence_totals.get(str(worker_id))
    if declared_total is not None and declared_total != expected_total:
        raise ValueError("pass1 sequence total does not match assigned shards")


def save_pass1_partials(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> Dict[str, str]:
    """Save current first-pass stores under the worker pass1 directory."""

    worker_layout = build_run_layout(manifest).workers[worker_id]
    worker_layout.pass1_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "latent_stats": worker_layout.pass1_dir / PASS1_PARTIAL_FILENAMES["latent_stats"],
        "top_ctx": worker_layout.pass1_dir / PASS1_PARTIAL_FILENAMES["top_ctx"],
        "mid_ctx_candidates": worker_layout.pass1_dir
        / PASS1_PARTIAL_FILENAMES["mid_ctx_candidates"],
        "seq_repr": worker_layout.pass1_dir / PASS1_PARTIAL_FILENAMES["seq_repr"],
        "logit_ctx": worker_layout.pass1_dir / PASS1_PARTIAL_FILENAMES["logit_ctx"],
    }

    payload_builders = {
        "latent_stats": lambda: latent_stats_payload(latent_stats),
        "top_ctx": lambda: top_ctx_payload(top_ctx),
        # Phase 6 later replaces this worker-local mid_ctx checkpoint with the
        # oversampled candidate-pool collection semantics planned for exact merge.
        "mid_ctx_candidates": lambda: mid_ctx_candidates_payload(
            mid_ctx,
            sampling_seed=manifest.sampling_seed,
            dataset_fingerprint=shard_table_fingerprint(manifest.shard_table),
        ),
        "seq_repr": _runtime_seq_repr_payload,
        "logit_ctx": lambda: logit_ctx_payload(logit_ctx),
    }
    saved: Dict[str, str] = {}
    for artifact_name, artifact_path in artifact_paths.items():
        payload = payload_builders[artifact_name]()
        metadata = build_pass1_partial_metadata(
            manifest,
            worker_id,
            artifact_name,  # type: ignore[arg-type]
            component_count=_component_count(),
            d_sae=_d_sae(),
            store_mode=_store_mode_for(artifact_name),
        )
        save_pass1_partial(artifact_path, metadata, payload)
        saved[artifact_name] = str(artifact_path)
    return saved


def _runtime_seq_repr_payload() -> Dict[str, object]:
    seq_repr = get_runtime().seq_repr
    if seq_repr is None:
        raise RuntimeError("runtime seq_repr is not initialized")
    return seq_repr_payload(seq_repr)


def _component_count() -> int:
    return int(latent_stats.num_components)


def _d_sae() -> int:
    return int(latent_stats.sae_config.d_sae)


def _store_mode_for(artifact_name: str) -> Dict[str, object]:
    if artifact_name == "top_ctx":
        return {"ctx_type": top_ctx.ctx_type}
    if artifact_name == "mid_ctx_candidates":
        return {
            "ctx_type": mid_ctx.ctx_type,
            "candidate_schema": "widened_worker_candidate_pool",
            "mid_mode": mid_ctx.mid_mode,
            "candidate_band_low_sigma": float(mid_ctx._band_low),
            "candidate_band_high_sigma": float(mid_ctx._band_high),
            "band_low_sigma": float(getattr(mid_ctx, "_final_band_low", mid_ctx._band_low)),
            "band_high_sigma": float(getattr(mid_ctx, "_final_band_high", mid_ctx._band_high)),
            "num_ctx_sequences": int(
                getattr(mid_ctx, "_final_num_ctx_sequences", mid_ctx.num_ctx_sequences)
            ),
            "max_candidates_per_latent": int(mid_ctx.num_ctx_sequences),
        }
    if artifact_name == "seq_repr":
        seq_repr = get_runtime().seq_repr
        return {"repr_mode": seq_repr.repr_mode if seq_repr is not None else None}
    return {}


__all__ = [
    "configure_mid_ctx_candidate_pool",
    "initialize_pass1_worker_resources",
    "run_pass1_worker",
    "save_pass1_partials",
    "validate_pass1_worker_inputs",
]
