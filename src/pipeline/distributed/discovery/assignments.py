"""Discovery assignment persistence helpers for distributed workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from pipeline.discovery_artifacts import hash_discovery_artifacts

from ..layout import build_run_layout
from ..manifest import DistributedRunManifest
from ..worker_common import _atomic_torch_save, _atomic_write_json, _validate_worker_id
from .method_filtering import seed_free_methods_for_worker


def load_assigned_discovery_candidates(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> List[Dict[str, Any]]:
    """Load the canonical candidate list and return this worker's assigned subset."""

    _validate_worker_id(manifest, worker_id)
    candidates_path = Path(manifest.output_root) / "candidates.pt"
    candidates: List[Dict[str, Any]] = torch.load(candidates_path, weights_only=False)
    artifact_hashes = hash_discovery_artifacts(
        manifest.output_root,
        candidates_path=candidates_path,
    )
    assignments = manifest.work_assignments.discovery_candidate_assignments.get(str(worker_id), [])
    assigned: List[Dict[str, Any]] = []
    for assignment in assignments:
        if assignment.candidate_index >= len(candidates):
            raise ValueError("assigned candidate index out of range")
        candidate = dict(candidates[assignment.candidate_index])
        if int(candidate.get("comp_idx", -1)) != assignment.comp_idx:
            raise ValueError("assigned candidate comp_idx mismatch")
        if int(candidate.get("latent_idx", -1)) != assignment.latent_idx:
            raise ValueError("assigned candidate latent_idx mismatch")
        candidate["candidate_index"] = assignment.candidate_index
        candidate["run_id"] = manifest.run_id
        candidate["worker_id"] = worker_id
        candidate["config_hash"] = manifest.normalized_config_hash
        candidate["artifact_hashes"] = artifact_hashes
        candidate["methods"] = list(assignment.methods)
        assigned.append(candidate)
    return assigned


def save_discovery_worker_inputs(
    manifest: DistributedRunManifest,
    worker_id: int,
    assigned_candidates: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Save assigned candidates and assignment metadata for traceability."""

    worker_layout = build_run_layout(manifest).workers[worker_id]
    worker_layout.discovery_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = worker_layout.discovery_dir / "assigned_candidates.pt"
    metadata_path = worker_layout.discovery_dir / "assignment_metadata.json"
    _atomic_torch_save(assigned_candidates, candidates_path)
    metadata = {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "worker_id": worker_id,
        "candidate_count": len(assigned_candidates),
        "owned_seed_free_methods": seed_free_methods_for_worker(manifest, worker_id),
        "assignments": [
            assignment.model_dump(mode="json")
            for assignment in manifest.work_assignments.discovery_candidate_assignments.get(str(worker_id), [])
        ],
    }
    _atomic_write_json(metadata_path, metadata)
    return {
        "assigned_candidates": str(candidates_path),
        "assignment_metadata": str(metadata_path),
    }


__all__ = [
    "load_assigned_discovery_candidates",
    "save_discovery_worker_inputs",
]
