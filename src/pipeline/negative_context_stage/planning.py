"""Planning and resume classification for the negative-context stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from config import config
from pipeline.distributed.manifest import (
    DistributedRunManifest,
    NegativeContextRunConfig,
    load_manifest,
)

from .inputs import NegativeContextInputs, load_negative_context_inputs


@dataclass(frozen=True)
class NegativeContextStagePlan:
    output_root: Path
    part_dir: Path
    metadata: Dict[str, object]
    resume_status: str
    reason: str


@dataclass(frozen=True)
class NegativeContextStageClassification:
    resume_status: str
    reason: str


def plan_negative_context_stage(
    output_root: str | Path = "outputs",
    *,
    expected_config_hash: Optional[str] = None,
    manifest_path: str | Path | None = None,
) -> NegativeContextStagePlan:
    manifest = load_manifest(manifest_path) if manifest_path is not None else None
    effective_config_hash = expected_config_hash or (
        manifest.normalized_config_hash if manifest is not None else None
    )
    inputs = load_negative_context_inputs(
        output_root,
        expected_config_hash=None,
    )
    metadata = build_negative_context_stage_metadata(
        inputs,
        manifest=manifest,
        expected_config_hash=effective_config_hash,
        selected_devices=_manifest_neg_ctx_devices_from_manifest(manifest),
    )
    classification = classify_negative_context_stage(inputs.paths.run_root, metadata=metadata)
    return NegativeContextStagePlan(
        output_root=inputs.paths.run_root,
        part_dir=_neg_ctx_part_dir(inputs.paths.run_root),
        metadata=metadata,
        resume_status=classification.resume_status,
        reason=classification.reason,
    )


def classify_negative_context_stage(
    run_root: str | Path,
    *,
    metadata: Dict[str, object],
) -> NegativeContextStageClassification:
    root = Path(run_root)
    part_dir = _neg_ctx_part_dir(root)
    failed_marker = part_dir / "failed.json"
    completed_marker = part_dir / "completed.json"
    if failed_marker.exists():
        return NegativeContextStageClassification("failed", "failed marker exists")
    if not completed_marker.exists():
        return NegativeContextStageClassification("missing", "completed marker missing")
    required_outputs = [
        root / "neg_ctx.pt",
        root / "neg_ctx_stats.json",
        part_dir / "neg_ctx_sanity_report.json",
    ]
    if any(not path.exists() for path in required_outputs):
        return NegativeContextStageClassification("missing", "required neg_ctx outputs missing")
    try:
        marker = json.loads(completed_marker.read_text(encoding="utf-8"))
        sanity = json.loads((part_dir / "neg_ctx_sanity_report.json").read_text(encoding="utf-8"))
    except Exception:
        return NegativeContextStageClassification("stale", "status metadata is unreadable")
    if marker.get("metadata") != metadata:
        return NegativeContextStageClassification("stale", "completed marker metadata mismatch")
    if sanity.get("metadata") != metadata:
        return NegativeContextStageClassification("stale", "sanity report metadata mismatch")
    return NegativeContextStageClassification("completed", "outputs and metadata match")


def build_negative_context_stage_metadata(
    inputs: NegativeContextInputs,
    *,
    manifest: DistributedRunManifest | None,
    expected_config_hash: Optional[str],
    selected_devices: Sequence[int] | None,
) -> Dict[str, object]:
    backend = str(config.latents.neg_ctx.backend or "single_gpu_exact")
    configured_devices = [str(device) for device in list(config.latents.neg_ctx.devices)]
    if selected_devices:
        selected_device_labels = [f"cuda:{device}" for device in selected_devices]
        device_source = "manifest_declared_devices"
    elif configured_devices:
        selected_device_labels = configured_devices
        device_source = "config_override"
    elif backend in {"multi_gpu_exact", "multi_gpu_index_sharded_exact"}:
        selected_device_labels = []
        device_source = "standalone_all_visible"
    else:
        selected_device_labels = [str(config.hardware.ann_device or "auto")]
        device_source = "single_device"
    return {
        "schema_version": 1,
        "run_id": manifest.run_id if manifest is not None else None,
        "config_hash": expected_config_hash
        or (manifest.normalized_config_hash if manifest is not None else None),
        "backend": backend,
        "used_backend": backend,
        "selected_devices": selected_device_labels,
        "device_selection_source": device_source,
        "n_neighbors": int(config.latents.neg_ctx.n_neighbors or 512),
        "n_sequences": int(config.latents.neg_ctx.n_sequences or 64),
        "min_pos_ctx": int(config.latents.neg_ctx.min_pos_ctx or 8),
        "repr_mode": str(config.latents.neg_ctx.repr_mode or "mean_pool"),
        "max_repr_seqs": config.latents.neg_ctx.max_repr_seqs,
        "memory_guardrail_fraction": float(config.latents.neg_ctx.memory_guardrail_fraction),
        "fail_on_memory_guardrail": bool(config.latents.neg_ctx.fail_on_memory_guardrail),
        "inputs": {
            "top_ctx": _artifact_metadata(inputs.paths.top_ctx),
            "mid_ctx": _artifact_metadata(inputs.paths.mid_ctx),
            "seq_repr": _artifact_metadata(inputs.paths.seq_repr),
        },
    }


def _manifest_neg_ctx_devices(manifest_path: str | Path | None) -> list[int] | None:
    if manifest_path is None:
        return None
    manifest = load_manifest(manifest_path)
    return _manifest_neg_ctx_devices_from_manifest(manifest)


def _manifest_neg_ctx_devices_from_manifest(
    manifest: DistributedRunManifest | None,
) -> list[int] | None:
    if manifest is None:
        return None
    physical_ids = [
        int(device.physical_id)
        for device in sorted(manifest.devices, key=lambda assignment: assignment.worker_id)
        if device.physical_id is not None
    ]
    if not physical_ids:
        return None
    return physical_ids


def _manifest_neg_ctx_config(metadata: Dict[str, object]) -> NegativeContextRunConfig:
    return NegativeContextRunConfig(
        backend=str(metadata["backend"]),
        selected_devices=[str(device) for device in metadata["selected_devices"]],
        device_selection_source=metadata["device_selection_source"],  # type: ignore[arg-type]
        n_neighbors=int(metadata["n_neighbors"]),
        n_sequences=int(metadata["n_sequences"]),
        min_pos_ctx=int(metadata["min_pos_ctx"]),
        repr_mode=str(metadata["repr_mode"]),
        max_repr_seqs=metadata["max_repr_seqs"],  # type: ignore[arg-type]
        memory_guardrail_fraction=float(metadata["memory_guardrail_fraction"]),
        fail_on_memory_guardrail=bool(metadata["fail_on_memory_guardrail"]),
    )


def _neg_ctx_part_dir(run_root: str | Path) -> Path:
    return Path(run_root) / "distributed" / "parts" / "neg_ctx"


def _artifact_metadata(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


__all__ = [
    "NegativeContextStageClassification",
    "NegativeContextStagePlan",
    "build_negative_context_stage_metadata",
    "classify_negative_context_stage",
    "plan_negative_context_stage",
]
