"""Stable integration interfaces consumed by later distributed pipeline parts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .layout import WorkerLayout, build_run_layout
from .manifest import DistributedRunManifest


CANONICAL_ARTIFACT_FILENAMES = {
    "latent_stats": "latent_stats.pt",
    "top_ctx": "top_ctx.pt",
    "mid_ctx": "mid_ctx.pt",
    "seq_repr": "seq_repr.pt",
    "logit_ctx": "logit_ctx.pt",
    "neg_ctx": "neg_ctx.pt",
    "global_negctx_ids": "global_negctx_ids.pt",
    "top_coactivation": "top_coactivation.pt",
    "candidates": "candidates.pt",
    "search_cache": "search_cache.parquet",
}

MANIFEST_FIELD_CONSUMERS = {
    "run_id": ["all parts", "metrics", "resume"],
    "run_mode": ["controller", "worker runtime", "preflight"],
    "cleanup_policy": ["controller", "cleanup"],
    "normalized_config_hash": ["preflight", "resume stale detection"],
    "output_root": ["canonical artifact persistence", "display/search"],
    "distributed_root": ["partials", "metrics", "worker markers"],
    "devices": ["worker launch", "runtime device isolation", "observability"],
    "shard_table": ["pass 1", "pass 2 replay", "display/search references"],
    "work_assignments.pass1_shards": ["pass 1 workers"],
    "work_assignments.pass1_sequence_totals": ["preflight", "metrics sanity checks"],
    "work_assignments.pass2_sequence_ids": ["pass 2 dump workers"],
    "work_assignments.discovery_seed_ids": ["distributed discovery workers"],
}


@dataclass(frozen=True)
class PipelineOutputPaths:
    run_root: Path
    latent_stats: Path
    top_ctx: Path
    mid_ctx: Path
    seq_repr: Path
    logit_ctx: Path
    neg_ctx: Path
    global_negctx_ids: Path
    top_coactivation: Path
    candidates: Path
    search_cache: Path
    seq_latent_index_dir: Path
    circuits_dir: Path
    cluster_circuits_dir: Path


def build_output_paths(run_root: str | Path = "outputs") -> PipelineOutputPaths:
    """Resolve canonical artifact paths under a run root."""

    root = Path(run_root)
    return PipelineOutputPaths(
        run_root=root,
        latent_stats=root / CANONICAL_ARTIFACT_FILENAMES["latent_stats"],
        top_ctx=root / CANONICAL_ARTIFACT_FILENAMES["top_ctx"],
        mid_ctx=root / CANONICAL_ARTIFACT_FILENAMES["mid_ctx"],
        seq_repr=root / CANONICAL_ARTIFACT_FILENAMES["seq_repr"],
        logit_ctx=root / CANONICAL_ARTIFACT_FILENAMES["logit_ctx"],
        neg_ctx=root / CANONICAL_ARTIFACT_FILENAMES["neg_ctx"],
        global_negctx_ids=root / CANONICAL_ARTIFACT_FILENAMES["global_negctx_ids"],
        top_coactivation=root / CANONICAL_ARTIFACT_FILENAMES["top_coactivation"],
        candidates=root / CANONICAL_ARTIFACT_FILENAMES["candidates"],
        search_cache=root / CANONICAL_ARTIFACT_FILENAMES["search_cache"],
        seq_latent_index_dir=root / "seq_latent_index",
        circuits_dir=root / "circuits",
        cluster_circuits_dir=root / "cluster_circuits",
    )


def resolve_output_path(run_root: str | Path, relative_path: str | Path) -> Path:
    """Resolve a schema-stable artifact path inside a run root."""

    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    return Path(run_root) / relative


def get_worker_shard_ids(manifest: DistributedRunManifest, worker_id: int) -> List[int]:
    _validate_worker_id(manifest, worker_id)
    return list(manifest.work_assignments.pass1_shards.get(str(worker_id), []))


def get_worker_sequence_ids(manifest: DistributedRunManifest, worker_id: int) -> List[int]:
    _validate_worker_id(manifest, worker_id)
    return list(manifest.work_assignments.pass2_sequence_ids.get(str(worker_id), []))


def get_worker_seed_ids(manifest: DistributedRunManifest, worker_id: int) -> List[int]:
    _validate_worker_id(manifest, worker_id)
    return list(manifest.work_assignments.discovery_seed_ids.get(str(worker_id), []))


def get_worker_output_paths(
    manifest: DistributedRunManifest,
    worker_id: int,
) -> WorkerLayout:
    _validate_worker_id(manifest, worker_id)
    return build_run_layout(manifest).workers[worker_id]


def manifest_field_consumers() -> Dict[str, List[str]]:
    """Document which later parts consume each manifest field."""

    return {field: list(consumers) for field, consumers in MANIFEST_FIELD_CONSUMERS.items()}


def _validate_worker_id(manifest: DistributedRunManifest, worker_id: int) -> None:
    if worker_id < 0 or worker_id >= manifest.worker_count:
        raise ValueError("worker_id out of range")
