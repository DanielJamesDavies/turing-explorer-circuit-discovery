"""Pass-2 distributed worker implementation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Optional

from config import config
from data.loader import DataLoader
from model.inference import Inference
from pipeline.runtime import build_distributed_worker_runtime, clear_runtime, set_runtime
from pipeline.second_pass import SecondPassDumpResult, run_second_pass_dump
from sae.bank import SAEBank
from store.context import top_ctx
from store.latent_stats import latent_stats
from store.top_coactivation import top_coactivation

from ..layout import build_run_layout
from ..manifest import DistributedRunManifest
from ..pass2_partials import (
    build_candidate_dump_metadata,
    candidate_dump_payload,
    check_candidate_dump_memory_guardrail,
    load_candidate_dump_partial,
    save_candidate_dump_partial,
)
from ..pass2_replay import get_pass2_worker_input, validate_pass2_replay_assignments
from ..shard_table import validate_shard_table
from ..worker_common import (
    _atomic_write_json,
    _device_assignment_for_worker,
    _peak_cuda_memory_bytes,
    _utc_now,
    _write_worker_phase_marker,
)


PASS2_PARTIAL_FILENAMES = {
    "candidate_dump": "candidate_dump.partial.pt",
    "pass2_summary": "pass2_summary.json",
}


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

    _write_worker_phase_marker(
        manifest,
        worker_layout,
        worker_id,
        phase="pass2",
        status="started",
        start_time=start_time,
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
        _write_worker_phase_marker(
            manifest,
            worker_layout,
            worker_id,
            phase="pass2",
            status="completed",
            start_time=start_time,
            end_time=end_time,
            duration_s=time.perf_counter() - started_at,
            batch_count=dump_result.batch_count,
            peak_cuda_memory_bytes=_peak_cuda_memory_bytes(),
            artifacts=artifacts,
        )
        return artifacts
    except Exception as error:
        _write_worker_phase_marker(
            manifest,
            worker_layout,
            worker_id,
            phase="pass2",
            status="failed",
            start_time=start_time,
            end_time=_utc_now(),
            duration_s=time.perf_counter() - started_at,
            error=str(error),
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


__all__ = [
    "PASS2_PARTIAL_FILENAMES",
    "build_pass2_worker_summary",
    "initialize_pass2_worker_resources",
    "load_pass2_global_artifacts",
    "run_pass2_worker",
    "save_pass2_candidate_dump",
    "validate_pass2_worker_inputs",
]
