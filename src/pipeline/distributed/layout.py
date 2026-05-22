"""Output layout, marker, metrics, and cleanup contracts for distributed runs."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .manifest import CleanupPolicy, DeviceAssignment, DistributedRunManifest


MARKER_SCHEMA_VERSION = 1
METRIC_EVENT_SCHEMA_VERSION = 1
WORKER_PARTS = ("pass1", "pass2", "discovery")
LARGE_PARTIAL_SUFFIXES = {".pt", ".pth", ".npy", ".npz", ".bin", ".parquet"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class WorkerLayout:
    worker_id: int
    root: Path
    metrics_path: Path
    pass1_dir: Path
    pass2_dir: Path
    discovery_dir: Path
    started_marker: Path
    completed_marker: Path
    failed_marker: Path


@dataclass(frozen=True)
class RunLayout:
    run_root: Path
    distributed_root: Path
    manifest_path: Path
    reports_dir: Path
    run_metrics_path: Path
    run_summary_path: Path
    workers_root: Path
    workers: Dict[int, WorkerLayout]


class WorkerMarker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    marker_schema_version: int = MARKER_SCHEMA_VERSION
    run_id: str
    worker_id: int
    phase: str
    status: Literal["started", "completed", "failed"]
    timestamp: str = Field(default_factory=_utc_now)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_s: Optional[float] = None
    physical_id: Optional[int] = None
    logical_id: str
    pid: int = Field(default_factory=os.getpid)
    hostname: str = Field(default_factory=socket.gethostname)
    shard_ids: List[int] = Field(default_factory=list)
    shard_ranges: List[Dict[str, int]] = Field(default_factory=list)
    batch_count: int = 0
    sequence_count: int = 0
    seed_count: int = 0
    peak_cpu_ram_bytes: Optional[int] = None
    peak_cuda_memory_bytes: Optional[int] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None

    @field_validator("marker_schema_version")
    @classmethod
    def marker_schema_is_supported(cls, value: int) -> int:
        if value != MARKER_SCHEMA_VERSION:
            raise ValueError("unsupported marker schema version")
        return value

    @field_validator("worker_id", "batch_count", "sequence_count", "seed_count")
    @classmethod
    def non_negative_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("worker marker counts must be >= 0")
        return value

    @field_validator("duration_s")
    @classmethod
    def duration_is_non_negative(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("duration_s must be >= 0")
        return value

    @model_validator(mode="after")
    def failed_markers_have_error(self) -> "WorkerMarker":
        if self.status == "failed" and not self.error:
            raise ValueError("failed worker markers must include error")
        return self


class MetricEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_schema_version: int = METRIC_EVENT_SCHEMA_VERSION
    run_id: str
    worker_id: Optional[int] = None
    phase: str
    event: str
    timestamp: str = Field(default_factory=_utc_now)
    elapsed_s: Optional[float] = None
    physical_id: Optional[int] = None
    logical_id: Optional[str] = None
    pid: int = Field(default_factory=os.getpid)
    hostname: str = Field(default_factory=socket.gethostname)
    artifact_path: Optional[str] = None
    artifact_size_bytes: Optional[int] = None
    counters: Dict[str, int] = Field(default_factory=dict)

    @field_validator("metric_schema_version")
    @classmethod
    def metric_schema_is_supported(cls, value: int) -> int:
        if value != METRIC_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported metric schema version")
        return value

    @field_validator("elapsed_s")
    @classmethod
    def elapsed_is_non_negative(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("elapsed_s must be >= 0")
        return value

    @field_validator("artifact_size_bytes")
    @classmethod
    def artifact_size_is_non_negative(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("artifact_size_bytes must be >= 0")
        return value

    @field_validator("counters")
    @classmethod
    def counters_are_non_negative(cls, value: Dict[str, int]) -> Dict[str, int]:
        for counter_value in value.values():
            if counter_value < 0:
                raise ValueError("metric counters must be >= 0")
        return value


def build_run_layout(manifest: DistributedRunManifest) -> RunLayout:
    workers_root = Path(manifest.distributed_root) / "workers"
    workers = {
        worker_id: _worker_layout(workers_root, worker_id)
        for worker_id in range(manifest.worker_count)
    }
    return RunLayout(
        run_root=Path(manifest.output_root),
        distributed_root=Path(manifest.distributed_root),
        manifest_path=Path(manifest.manifest_path),
        reports_dir=Path(manifest.distributed_root) / "reports",
        run_metrics_path=Path(manifest.metrics_path),
        run_summary_path=Path(manifest.run_summary_path),
        workers_root=workers_root,
        workers=workers,
    )


def create_output_layout(manifest: DistributedRunManifest) -> RunLayout:
    """Create canonical run directories without creating outputs/latest."""

    layout = build_run_layout(manifest)
    for path in [
        layout.run_root,
        layout.distributed_root,
        layout.reports_dir,
        layout.workers_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for worker in layout.workers.values():
        for path in [worker.root, worker.pass1_dir, worker.pass2_dir, worker.discovery_dir]:
            path.mkdir(parents=True, exist_ok=True)
    return layout


def build_worker_marker(
    manifest: DistributedRunManifest,
    worker_id: int,
    *,
    phase: str,
    status: Literal["started", "completed", "failed"],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    duration_s: Optional[float] = None,
    batch_count: int = 0,
    sequence_count: int = 0,
    seed_count: int = 0,
    peak_cpu_ram_bytes: Optional[int] = None,
    peak_cuda_memory_bytes: Optional[int] = None,
    artifacts: Optional[Dict[str, str]] = None,
    error: Optional[str] = None,
) -> WorkerMarker:
    assignment = _device_for_worker(manifest, worker_id)
    shard_ids = manifest.work_assignments.pass1_shards.get(str(worker_id), [])
    shard_ranges = _shard_ranges_for_worker(manifest, shard_ids)
    return WorkerMarker(
        run_id=manifest.run_id,
        worker_id=worker_id,
        phase=phase,
        status=status,
        start_time=start_time,
        end_time=end_time,
        duration_s=duration_s,
        physical_id=assignment.physical_id if assignment is not None else None,
        logical_id=assignment.logical_id if assignment is not None else "cpu",
        shard_ids=list(shard_ids),
        shard_ranges=shard_ranges,
        batch_count=batch_count,
        sequence_count=sequence_count,
        seed_count=seed_count,
        peak_cpu_ram_bytes=peak_cpu_ram_bytes,
        peak_cuda_memory_bytes=peak_cuda_memory_bytes,
        artifacts=artifacts or {},
        error=error,
    )


def write_worker_marker(marker: WorkerMarker, path: str | Path) -> None:
    _atomic_write_json(path, marker.model_dump(mode="json"))


def read_worker_marker(path: str | Path) -> WorkerMarker:
    return WorkerMarker.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def append_metric_event(event: MetricEvent, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(event.model_dump_json() + "\n")


def validate_worker_completed(
    marker: WorkerMarker,
    worker_dir: str | Path,
    *,
    required_artifacts: Iterable[str],
) -> None:
    if marker.status != "completed":
        raise ValueError("worker marker status must be completed")
    if marker.start_time is None or marker.end_time is None or marker.duration_s is None:
        raise ValueError("completed worker marker missing timing metadata")

    root = Path(worker_dir)
    if not (root / "started.json").exists():
        raise FileNotFoundError("completed worker missing started.json marker")
    for artifact_name in required_artifacts:
        if artifact_name not in marker.artifacts:
            raise ValueError(f"completed worker missing declared artifact: {artifact_name}")
        artifact_path = Path(marker.artifacts[artifact_name])
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        if not artifact_path.exists():
            raise FileNotFoundError(f"declared artifact does not exist: {artifact_path}")


def cleanup_candidates(
    manifest: DistributedRunManifest,
    *,
    run_failed: bool,
) -> List[Path]:
    """Return files/directories eligible for cleanup without deleting them."""

    if run_failed:
        return []
    policy = manifest.cleanup_policy
    if policy in {CleanupPolicy.KEEP_ALL, CleanupPolicy.MANUAL_CLEANUP_ONLY}:
        return []

    distributed_root = Path(manifest.distributed_root)
    cleanup_roots = [
        distributed_root / "workers",
        distributed_root / "partials",
    ]
    candidates: List[Path] = []
    for root in cleanup_roots:
        if not root.exists():
            continue
        if policy == CleanupPolicy.DELETE_ALL_PARTIALS_ON_SUCCESS:
            candidates.append(root)
            continue
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in LARGE_PARTIAL_SUFFIXES
        )
    return sorted(candidates)


def _worker_layout(workers_root: Path, worker_id: int) -> WorkerLayout:
    worker_root = workers_root / f"worker_{worker_id:03d}"
    return WorkerLayout(
        worker_id=worker_id,
        root=worker_root,
        metrics_path=worker_root / "metrics.jsonl",
        pass1_dir=worker_root / "pass1",
        pass2_dir=worker_root / "pass2",
        discovery_dir=worker_root / "discovery",
        started_marker=worker_root / "started.json",
        completed_marker=worker_root / "completed.json",
        failed_marker=worker_root / "failed.json",
    )


def _device_for_worker(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> Optional[DeviceAssignment]:
    for assignment in manifest.devices:
        if assignment.worker_id == worker_id:
            return assignment
    return None


def _shard_ranges_for_worker(
    manifest: DistributedRunManifest,
    shard_ids: Iterable[int],
) -> List[Dict[str, int]]:
    by_index = {record.shard_index: record for record in manifest.shard_table}
    ranges: List[Dict[str, int]] = []
    for shard_id in shard_ids:
        record = by_index.get(shard_id)
        if record is None:
            continue
        ranges.append(
            {
                "shard_index": record.shard_index,
                "global_start_id": record.global_start_id,
                "global_end_id": record.global_end_id,
                "sequence_count": record.sequence_count,
            }
        )
    return ranges


def _atomic_write_json(path: str | Path, data: Dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, output_path)
