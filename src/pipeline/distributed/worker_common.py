"""Shared helpers for distributed worker phase implementations."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from pipeline.runtime import get_runtime

from .layout import WorkerLayout, build_worker_marker, write_worker_marker
from .manifest import DeviceAssignment, DistributedRunManifest
from .shard_table import ShardRecord


def _device_assignment_for_worker(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> DeviceAssignment:
    for assignment in manifest.devices:
        if assignment.worker_id == worker_id:
            return assignment
    raise ValueError(f"manifest has no device assignment for worker {worker_id}")


def _validate_worker_id(manifest: DistributedRunManifest, worker_id: int) -> None:
    if worker_id < 0 or worker_id >= manifest.worker_count:
        raise ValueError("worker_id out of range")


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


def _atomic_torch_save(data: object, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(data, tmp_path)
    os.replace(tmp_path, output_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_worker_phase_marker(
    manifest: DistributedRunManifest,
    worker_layout: WorkerLayout,
    worker_id: int,
    *,
    phase: str,
    status: str,
    start_time: str,
    end_time: str | None = None,
    duration_s: float | None = None,
    batch_count: int | None = None,
    sequence_count: int | None = None,
    seed_count: int | None = None,
    peak_cuda_memory_bytes: int | None = None,
    artifacts: Dict[str, str] | None = None,
    error: str | None = None,
) -> None:
    """Write a worker marker while leaving phase-specific metadata explicit."""

    marker_path = {
        "started": worker_layout.started_marker,
        "completed": worker_layout.completed_marker,
        "failed": worker_layout.failed_marker,
    }.get(status)
    if marker_path is None:
        raise ValueError(f"unsupported worker marker status: {status}")
    marker_kwargs: Dict[str, object] = {
        "phase": phase,
        "status": status,
        "start_time": start_time,
    }
    optional_values = {
        "end_time": end_time,
        "duration_s": duration_s,
        "batch_count": batch_count,
        "sequence_count": sequence_count,
        "seed_count": seed_count,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "artifacts": artifacts,
        "error": error,
    }
    marker_kwargs.update(
        {name: value for name, value in optional_values.items() if value is not None}
    )
    write_worker_marker(
        build_worker_marker(
            manifest,
            worker_id,
            **marker_kwargs,
        ),
        marker_path,
    )


__all__ = [
    "_atomic_torch_save",
    "_atomic_write_json",
    "_device_assignment_for_worker",
    "_peak_cuda_memory_bytes",
    "_total_sequences",
    "_utc_now",
    "_validate_worker_id",
    "_write_worker_phase_marker",
    "_worker_batch_count",
]
