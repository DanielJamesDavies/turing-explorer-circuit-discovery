"""Discovery worker stats and process-state helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from store.circuits import circuit_store

from ..layout import build_run_layout
from ..manifest import DistributedRunManifest
from ..worker_common import _atomic_write_json
from .method_filtering import seed_free_methods_for_worker


def _discovery_output_artifacts(worker_layout) -> Dict[str, str]:
    circuits_dir = worker_layout.discovery_dir / "circuits"
    artifact_paths = {
        "discovered_circuits": circuits_dir / "discovered_circuits.pt",
        "summary": circuits_dir / "summary.json",
        "summary_xlsx": circuits_dir / "summary.xlsx",
    }
    return {
        name: str(path)
        for name, path in artifact_paths.items()
        if path.exists()
    }


def save_worker_discovery_stats(
    manifest: DistributedRunManifest,
    worker_id: int,
    assigned_candidates: List[Dict[str, Any]],
    *,
    task_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Save a small worker-local discovery stats/provenance JSON."""

    worker_layout = build_run_layout(manifest).workers[worker_id]
    stats_path = worker_layout.discovery_dir / "worker_discovery_stats.json"
    methods = sorted(
        {
            method
            for assignment in manifest.work_assignments.discovery_candidate_assignments.get(str(worker_id), [])
            for method in assignment.methods
        }
        | set(seed_free_methods_for_worker(manifest, worker_id))
    )
    stats = {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "worker_id": worker_id,
        "config_hash": manifest.normalized_config_hash,
        "candidate_count": len(assigned_candidates),
        "planned_task_count": len(
            manifest.work_assignments.discovery_task_assignments.get(str(worker_id), [])
        ),
        "estimated_task_cost": manifest.work_assignments.discovery_worker_estimated_costs.get(
            str(worker_id),
            0.0,
        ),
        "failed_task_ranges": manifest.work_assignments.discovery_failed_task_ranges.get(
            str(worker_id),
            [],
        ),
        "method_count": len(methods),
        "methods": methods,
        "accepted_circuit_count": len(circuit_store.circuits),
        "circuit_uuids": sorted(circuit_store.circuits.keys()),
        "task_metrics": list(task_metrics or []),
    }
    _atomic_write_json(stats_path, stats)
    return stats_path


def reset_discovery_worker_state() -> None:
    """Reset process-global discovery state before a worker-local run."""

    circuit_store.circuits.clear()
    try:
        from observability.tracking import obs

        obs.forward_passes = 0
        obs.total_forward_time = 0.0
        obs.attempt_forward_passes = 0
        obs.attempt_start_time = 0.0
    except Exception:
        pass


__all__ = [
    "_discovery_output_artifacts",
    "reset_discovery_worker_state",
    "save_worker_discovery_stats",
]
