"""Distributed worker entrypoint for manifest-assigned pipeline work."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import torch

from data.loader import DataLoader
from config import config
from model.inference import Inference
from sae.bank import SAEBank
from store.context import mid_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.seq_repr import SeqRepr
from store.top_coactivation import top_coactivation

from pipeline.first_pass import run_first_pass
from pipeline.runtime import (
    build_distributed_worker_runtime,
    clear_runtime,
    get_runtime,
    set_runtime,
)
from pipeline.second_pass import SecondPassDumpResult, run_second_pass_dump

from .interfaces import get_worker_shard_ids
from .layout import (
    build_worker_marker,
    build_run_layout,
    write_worker_marker,
)
from .manifest import DeviceAssignment, DistributedRunManifest, load_manifest
from .pass1_partials import (
    build_pass1_partial_metadata,
    latent_stats_payload,
    logit_ctx_payload,
    mid_ctx_candidates_payload,
    save_pass1_partial,
    seq_repr_payload,
    top_ctx_payload,
)
from .pass2_partials import (
    build_candidate_dump_metadata,
    candidate_dump_payload,
    check_candidate_dump_memory_guardrail,
    load_candidate_dump_partial,
    save_candidate_dump_partial,
)
from .pass2_replay import get_pass2_worker_input, validate_pass2_replay_assignments
from .seq_repr_mapping import build_seq_repr_cap_mapping, shard_table_fingerprint
from .shard_table import (
    ShardRecord,
    sequence_ids_for_shards,
    validate_shard_table,
)


PASS1_PARTIAL_FILENAMES = {
    "latent_stats": "latent_stats.partial.pt",
    "top_ctx": "top_ctx.partial.pt",
    "mid_ctx_candidates": "mid_ctx_candidates.partial.pt",
    "seq_repr": "seq_repr.partial.pt",
    "logit_ctx": "logit_ctx.partial.pt",
}

PASS2_PARTIAL_FILENAMES = {
    "candidate_dump": "candidate_dump.partial.pt",
    "pass2_summary": "pass2_summary.json",
}


def run_worker(
    manifest_path: str | Path,
    worker_id: int,
    *,
    phase: str = "pass1",
) -> Dict[str, str]:
    """Run one manifest-assigned worker phase from the command-line contract."""

    manifest = load_manifest(manifest_path)
    if phase == "pass1":
        return run_pass1_worker(manifest, worker_id)
    if phase == "pass2":
        return run_pass2_worker(manifest, worker_id)
    raise ValueError(f"unsupported worker phase: {phase}")


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


def run_pass2_worker(
    manifest: DistributedRunManifest,
    worker_id: int,
    *,
    validate_inputs_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None,
    load_artifacts_fn: Optional[Callable[[DistributedRunManifest], None]] = None,
    initialize_fn: Optional[Callable[[DistributedRunManifest, int], None]] = None,
    run_dump_fn: Callable[[list[int]], SecondPassDumpResult] = run_second_pass_dump,
    save_dump_fn: Optional[
        Callable[[DistributedRunManifest, int, SecondPassDumpResult], Dict[str, str]]
    ] = None,
) -> Dict[str, str]:
    """Run pass-2 candidate dump work for one worker without reducing globally."""

    validate_inputs = validate_inputs_fn or validate_pass2_worker_inputs
    load_artifacts = load_artifacts_fn or load_pass2_global_artifacts
    initialize = initialize_fn or initialize_pass2_worker_resources
    save_dump = save_dump_fn or save_pass2_candidate_dump
    layout = build_run_layout(manifest)
    worker_layout = layout.workers[worker_id]
    worker_input = get_pass2_worker_input(manifest, worker_id)
    start_time = _utc_now()
    started_at = time.perf_counter()

    write_worker_marker(
        build_worker_marker(
            manifest,
            worker_id,
            phase="pass2",
            status="started",
            start_time=start_time,
        ),
        worker_layout.started_marker,
    )

    try:
        validate_inputs(manifest, worker_id)
        check_candidate_dump_memory_guardrail(
            worker_input.sequence_count,
            int(top_coactivation.M),
            guardrail_bytes=config.latents.top_coactivation.dump_memory_guardrail_bytes,
            fail_on_guardrail=bool(config.latents.top_coactivation.fail_on_dump_memory_guardrail),
        )
        load_artifacts(manifest)
        initialize(manifest, worker_id)
        dump_result = run_dump_fn(worker_input.sequence_ids)
        artifacts = save_dump(manifest, worker_id, dump_result)
        end_time = _utc_now()
        write_worker_marker(
            build_worker_marker(
                manifest,
                worker_id,
                phase="pass2",
                status="completed",
                start_time=start_time,
                end_time=end_time,
                duration_s=time.perf_counter() - started_at,
                batch_count=dump_result.batch_count,
                peak_cuda_memory_bytes=_peak_cuda_memory_bytes(),
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
                phase="pass2",
                status="failed",
                start_time=start_time,
                end_time=_utc_now(),
                duration_s=time.perf_counter() - started_at,
                error=str(error),
            ),
            worker_layout.failed_marker,
        )
        raise
    finally:
        clear_runtime()


def validate_pass2_worker_inputs(
    manifest: DistributedRunManifest,
    worker_id: int,
    *,
    validate_on_disk: bool = True,
) -> None:
    """Validate pass-2 worker inputs before model/SAE initialization."""

    get_pass2_worker_input(manifest, worker_id)
    validate_pass2_replay_assignments(manifest)
    if validate_on_disk:
        validate_shard_table(manifest.dataset_path, manifest.shard_table)
    output_root = Path(manifest.output_root)
    required_artifacts = {
        "top_ctx": output_root / "top_ctx.pt",
        "latent_stats": output_root / "latent_stats.pt",
    }
    missing = [name for name, path in required_artifacts.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing pass2 input artifacts: {missing}")


def load_pass2_global_artifacts(manifest: DistributedRunManifest) -> None:
    """Load merged global stores needed for pass-2 candidate dumping."""

    output_root = Path(manifest.output_root)
    top_ctx.load(str(output_root / "top_ctx.pt"))
    latent_stats.load(str(output_root / "latent_stats.pt"))


def initialize_pass2_worker_resources(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> None:
    """Initialize DataLoader, model, and SAE bank for one pass-2 worker."""

    assignment = _device_assignment_for_worker(manifest, worker_id)
    runtime = build_distributed_worker_runtime(assignment)
    set_runtime(runtime)
    runtime.loader = DataLoader(device=runtime.device, pin_memory=runtime.fast)
    runtime.model = Inference(device=runtime.device, compile=runtime.compile)
    runtime.bank = SAEBank(
        devices=runtime.devices,
        load_decoders=runtime.fast,
        compile=runtime.compile,
    )


def save_pass2_candidate_dump(
    manifest: DistributedRunManifest,
    worker_id: int,
    dump_result: SecondPassDumpResult,
) -> Dict[str, str]:
    """Save the worker-local simple exact candidate dump partial."""

    worker_input = get_pass2_worker_input(manifest, worker_id)
    worker_layout = build_run_layout(manifest).workers[worker_id]
    artifact_path = worker_layout.pass2_dir / PASS2_PARTIAL_FILENAMES["candidate_dump"]
    summary_path = worker_layout.pass2_dir / PASS2_PARTIAL_FILENAMES["pass2_summary"]
    metadata = build_candidate_dump_metadata(
        manifest,
        worker_id,
        top_coactivation,
        dump_result,
    )
    payload = candidate_dump_payload(top_coactivation, worker_input.sequence_ids)
    save_started_at = time.perf_counter()
    save_candidate_dump_partial(artifact_path, metadata, payload)
    save_elapsed_s = time.perf_counter() - save_started_at
    # Reload after the atomic write so completion is gated on the durable artifact.
    load_candidate_dump_partial(
        artifact_path,
        expected_config_hash=manifest.normalized_config_hash,
    )
    summary = build_pass2_worker_summary(
        manifest,
        worker_id,
        dump_result,
        artifact_path=artifact_path,
        save_elapsed_s=save_elapsed_s,
    )
    _atomic_write_json(summary_path, summary)
    return {
        "candidate_dump": str(artifact_path),
        "pass2_summary": str(summary_path),
    }


def build_pass2_worker_summary(
    manifest: DistributedRunManifest,
    worker_id: int,
    dump_result: SecondPassDumpResult,
    *,
    artifact_path: str | Path,
    save_elapsed_s: float,
) -> Dict[str, object]:
    """Build a compact pass-2 worker report for benchmark/debug logs."""

    worker_input = get_pass2_worker_input(manifest, worker_id)
    artifact = Path(artifact_path)
    dump_timing = dict(getattr(top_coactivation, "dump_timing", {}) or {})
    return {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "worker_id": worker_id,
        "phase": "pass2",
        "sequence_count": worker_input.sequence_count,
        "sequence_id_min": worker_input.sequence_id_min,
        "sequence_id_max": worker_input.sequence_id_max,
        "replay_sequence_hash": worker_input.replay_sequence_hash,
        "batch_count": int(dump_result.batch_count),
        "seq_len": int(dump_result.seq_len),
        "dump_elapsed_s": float(dump_result.elapsed_s),
        "model_forward_s": float(dump_result.model_forward_s),
        "sae_encode_s": float(dump_result.sae_encode_s),
        "update_dump_s": float(dump_result.update_dump_s),
        "save_elapsed_s": float(save_elapsed_s),
        "artifact_path": str(artifact),
        "artifact_size_bytes": artifact.stat().st_size if artifact.exists() else 0,
        "peak_cuda_memory_bytes": _peak_cuda_memory_bytes(),
        "dump_timing": {
            key: float(value)
            for key, value in dump_timing.items()
        },
        "timing_available": {
            "model_forward": "not_separated_yet",
            "sae_encode": "not_separated_yet",
            "update_dump": "total_update" in dump_timing,
            "cpu_transfer": "cpu_transfer" in dump_timing or "final_cpu_transfer" in dump_timing,
            "save": True,
        },
    }


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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a distributed pipeline worker")
    parser.add_argument("--manifest", required=True, help="Path to distributed manifest JSON")
    parser.add_argument("--worker-id", required=True, type=int, help="Worker ID from manifest")
    parser.add_argument("--phase", default="pass1", choices=["pass1", "pass2"], help="Worker phase to run")
    args = parser.parse_args(argv)
    run_worker(args.manifest, args.worker_id, phase=args.phase)


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


def _device_assignment_for_worker(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> DeviceAssignment:
    for assignment in manifest.devices:
        if assignment.worker_id == worker_id:
            return assignment
    raise ValueError(f"manifest has no device assignment for worker {worker_id}")


def _total_sequences(shard_table: Sequence[ShardRecord]) -> int:
    if not shard_table:
        raise ValueError("manifest shard_table is required for pass1 workers")
    return max(record.global_end_id for record in shard_table) - 1


def _worker_batch_count(shard_ids: Sequence[int]) -> int:
    try:
        runtime = get_runtime()
    except RuntimeError:
        return 0
    if runtime.loader is None:
        return 0
    return runtime.loader.num_batches_for_shards(list(shard_ids))


def _peak_cuda_memory_bytes() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def _atomic_write_json(path: str | Path, data: Dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, output_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
