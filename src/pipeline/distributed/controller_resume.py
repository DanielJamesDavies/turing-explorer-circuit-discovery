"""Resume classification helpers for distributed controller planning."""

from __future__ import annotations

from typing import Dict, List, Optional

from .layout import build_run_layout, read_worker_marker
from .manifest import DistributedRunManifest


def classify_resume_workers(
    manifest: DistributedRunManifest,
    *,
    current_config_hash: Optional[str] = None,
) -> Dict[str, List[int]]:
    """Classify workers from marker files as pending/completed/failed/stale."""

    result = {"pending": [], "completed": [], "failed": [], "stale": []}
    if current_config_hash is not None and current_config_hash != manifest.normalized_config_hash:
        result["stale"] = list(range(manifest.worker_count))
        return result

    layout = build_run_layout(manifest)
    for worker_id, worker_layout in layout.workers.items():
        try:
            if worker_layout.failed_marker.exists():
                marker = read_worker_marker(worker_layout.failed_marker)
                _validate_marker_identity(manifest, worker_id, marker.run_id, marker.worker_id)
                result["failed"].append(worker_id)
            elif worker_layout.completed_marker.exists():
                marker = read_worker_marker(worker_layout.completed_marker)
                _validate_marker_identity(manifest, worker_id, marker.run_id, marker.worker_id)
                result["completed"].append(worker_id)
            else:
                result["pending"].append(worker_id)
        except Exception:
            result["stale"].append(worker_id)
    return result


def _validate_marker_identity(
    manifest: DistributedRunManifest,
    worker_id: int,
    marker_run_id: str,
    marker_worker_id: int,
) -> None:
    if marker_run_id != manifest.run_id or marker_worker_id != worker_id:
        raise ValueError("worker marker identity does not match manifest")


__all__ = [
    "classify_resume_workers",
]
