"""Benchmark readiness helpers for distributed pass-2 candidate dumps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from .layout import build_run_layout, read_worker_marker
from .manifest import DistributedRunManifest
from .pass2_partials import estimate_candidate_dump_bytes
from .pass2_replay import get_pass2_worker_input, validate_pass2_replay_assignments


class Pass2WorkerBenchmarkEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: int
    physical_id: Optional[int] = None
    logical_id: str
    sequence_count: int
    sequence_id_min: Optional[int] = None
    sequence_id_max: Optional[int] = None
    m: int
    estimated_dump_bytes: int

    @field_validator("worker_id", "sequence_count", "m", "estimated_dump_bytes")
    @classmethod
    def counts_are_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("pass2 benchmark estimate counts must be >= 0")
        return value


class Pass2BenchmarkEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    worker_count: int
    replay_sequence_count: int
    replay_sequence_hash: Optional[str] = None
    m: int
    total_estimated_dump_bytes: int
    max_worker_sequences: int
    min_worker_sequences: int
    assignment_imbalance_ratio: float
    workers: list[Pass2WorkerBenchmarkEstimate]

    @field_validator(
        "schema_version",
        "worker_count",
        "replay_sequence_count",
        "m",
        "total_estimated_dump_bytes",
        "max_worker_sequences",
        "min_worker_sequences",
    )
    @classmethod
    def counts_are_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("pass2 benchmark estimate counts must be >= 0")
        return value


class Pass2WorkerBenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: int
    sequence_count: int
    batch_count: int
    duration_s: float
    dump_elapsed_s: float
    avg_batch_s: float
    model_forward_s: float = 0.0
    sae_encode_s: float = 0.0
    update_dump_s: float = 0.0
    save_elapsed_s: float = 0.0
    artifact_size_bytes: int = 0
    peak_cuda_memory_bytes: Optional[int] = None


class Pass2BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    run_id: str
    worker_count: int
    completed_worker_count: int
    total_replay_sequences: int
    total_candidate_dump_bytes: int
    total_wall_time_s: float
    total_worker_time_s: float
    max_worker_sequences: int
    min_worker_sequences: int
    assignment_imbalance_ratio: float
    max_peak_cuda_memory_bytes: Optional[int] = None
    workers: list[Pass2WorkerBenchmarkSummary]


def build_pass2_benchmark_estimate(
    manifest: DistributedRunManifest,
    *,
    m: int,
) -> Pass2BenchmarkEstimate:
    """Estimate pass-2 dump shape and bytes without loading model resources."""

    validate_pass2_replay_assignments(manifest)
    workers: list[Pass2WorkerBenchmarkEstimate] = []
    assignments = {device.worker_id: device for device in manifest.devices}
    for worker_id in range(manifest.worker_count):
        worker_input = get_pass2_worker_input(manifest, worker_id)
        device = assignments.get(worker_id)
        estimate = estimate_candidate_dump_bytes(worker_input.sequence_count, m)
        workers.append(
            Pass2WorkerBenchmarkEstimate(
                worker_id=worker_id,
                physical_id=device.physical_id if device is not None else None,
                logical_id=device.logical_id if device is not None else "cpu",
                sequence_count=worker_input.sequence_count,
                sequence_id_min=worker_input.sequence_id_min,
                sequence_id_max=worker_input.sequence_id_max,
                m=m,
                estimated_dump_bytes=estimate.total_bytes,
            )
        )
    counts = [worker.sequence_count for worker in workers]
    max_sequences = max(counts, default=0)
    min_sequences = min(counts, default=0)
    return Pass2BenchmarkEstimate(
        run_id=manifest.run_id,
        worker_count=manifest.worker_count,
        replay_sequence_count=sum(counts),
        replay_sequence_hash=manifest.work_assignments.pass2_replay_sequence_hash,
        m=m,
        total_estimated_dump_bytes=sum(worker.estimated_dump_bytes for worker in workers),
        max_worker_sequences=max_sequences,
        min_worker_sequences=min_sequences,
        assignment_imbalance_ratio=_imbalance_ratio(max_sequences, min_sequences),
        workers=workers,
    )


def format_pass2_benchmark_estimate(estimate: Pass2BenchmarkEstimate) -> str:
    lines = [
        "pass2 candidate dump estimate:",
        f"  replay_sequences: {estimate.replay_sequence_count}",
        f"  M: {estimate.m}",
        f"  total_dump_bytes: {estimate.total_estimated_dump_bytes}",
        f"  assignment_imbalance_ratio: {estimate.assignment_imbalance_ratio:.3f}",
    ]
    for worker in estimate.workers:
        lines.append(
            f"  worker_{worker.worker_id:03d}: sequences={worker.sequence_count} "
            f"dump_bytes={worker.estimated_dump_bytes} logical={worker.logical_id}"
        )
    return "\n".join(lines)


def save_pass2_benchmark_estimate(
    estimate: Pass2BenchmarkEstimate,
    path: str | Path,
) -> None:
    _atomic_write_json(path, estimate.model_dump(mode="json"))


def build_pass2_benchmark_report(manifest: DistributedRunManifest) -> Pass2BenchmarkReport:
    """Aggregate completed pass-2 worker summaries for benchmark comparison."""

    layout = build_run_layout(manifest)
    workers: list[Pass2WorkerBenchmarkSummary] = []
    for worker_id in range(manifest.worker_count):
        worker_layout = layout.workers[worker_id]
        marker = read_worker_marker(worker_layout.completed_marker)
        if marker.phase != "pass2":
            raise ValueError(f"worker_{worker_id:03d} completed marker is not pass2")
        summary_path = Path(marker.artifacts.get("pass2_summary", ""))
        if not summary_path.exists():
            raise FileNotFoundError(f"missing pass2 summary for worker_{worker_id:03d}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        batch_count = int(summary.get("batch_count", marker.batch_count))
        dump_elapsed_s = float(summary.get("dump_elapsed_s", 0.0))
        workers.append(
            Pass2WorkerBenchmarkSummary(
                worker_id=worker_id,
                sequence_count=int(summary.get("sequence_count", marker.sequence_count)),
                batch_count=batch_count,
                duration_s=float(marker.duration_s or 0.0),
                dump_elapsed_s=dump_elapsed_s,
                avg_batch_s=dump_elapsed_s / batch_count if batch_count else 0.0,
                model_forward_s=float(summary.get("model_forward_s", 0.0)),
                sae_encode_s=float(summary.get("sae_encode_s", 0.0)),
                update_dump_s=float(summary.get("update_dump_s", 0.0)),
                save_elapsed_s=float(summary.get("save_elapsed_s", 0.0)),
                artifact_size_bytes=int(summary.get("artifact_size_bytes", 0)),
                peak_cuda_memory_bytes=(
                    int(summary["peak_cuda_memory_bytes"])
                    if summary.get("peak_cuda_memory_bytes") is not None
                    else marker.peak_cuda_memory_bytes
                ),
            )
        )
    sequence_counts = [worker.sequence_count for worker in workers]
    peak_values = [
        int(worker.peak_cuda_memory_bytes)
        for worker in workers
        if worker.peak_cuda_memory_bytes is not None
    ]
    max_sequences = max(sequence_counts, default=0)
    min_sequences = min(sequence_counts, default=0)
    return Pass2BenchmarkReport(
        run_id=manifest.run_id,
        worker_count=manifest.worker_count,
        completed_worker_count=len(workers),
        total_replay_sequences=sum(sequence_counts),
        total_candidate_dump_bytes=sum(worker.artifact_size_bytes for worker in workers),
        total_wall_time_s=max((worker.duration_s for worker in workers), default=0.0),
        total_worker_time_s=sum(worker.duration_s for worker in workers),
        max_worker_sequences=max_sequences,
        min_worker_sequences=min_sequences,
        assignment_imbalance_ratio=_imbalance_ratio(max_sequences, min_sequences),
        max_peak_cuda_memory_bytes=max(peak_values) if peak_values else None,
        workers=workers,
    )


def save_pass2_benchmark_report(
    report: Pass2BenchmarkReport,
    path: str | Path,
) -> None:
    _atomic_write_json(path, report.model_dump(mode="json"))


def _imbalance_ratio(max_sequences: int, min_sequences: int) -> float:
    if min_sequences == 0:
        return float(max_sequences) if max_sequences else 0.0
    return float(max_sequences) / float(min_sequences)


def _atomic_write_json(path: str | Path, data: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)
