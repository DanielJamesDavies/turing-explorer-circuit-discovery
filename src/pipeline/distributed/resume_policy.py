"""Mode-level resume, skip, failure, and cleanup policy helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .layout import build_run_layout, cleanup_candidates, read_worker_marker
from .manifest import DistributedRunManifest, RunMode


RESUMABLE_PARTS_BY_MODE: Mapping[RunMode, tuple[str, ...]] = {
    RunMode.SINGLE_PROCESS: (),
    RunMode.DISTRIBUTED_SIMPLE_EXACT: (
        "pass1",
        "pass1_merge",
        "neg_ctx",
        "pass2_dump",
        "pass2_reduce",
        "candidate_selection",
        "discovery",
        "circuit_merge",
    ),
    RunMode.DISTRIBUTED_MAPREDUCE_EXACT: (
        "pass1",
        "pass1_merge",
        "neg_ctx",
        "pass2_dump",
        "pass2_reduce",
        "pass2_mapreduce_reduce",
        "candidate_selection",
        "discovery",
        "circuit_merge",
    ),
    RunMode.DISTRIBUTED_EXPERIMENTAL_FAST: (
        "pass1",
        "pass1_merge",
        "neg_ctx",
        "pass2_dump",
        "pass2_reduce",
        "candidate_selection",
        "discovery",
        "circuit_merge",
        "experimental_fast_report",
    ),
}


@dataclass(frozen=True)
class PartResumeState:
    part: str
    status: str
    can_skip: bool
    reason: str
    marker_path: Path | None
    required_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class CleanupPlan:
    cleanup_policy: str
    run_failed: bool
    candidates: tuple[Path, ...]
    preserve_reason: str | None = None


def resumable_parts_for_mode(mode: RunMode | str) -> tuple[str, ...]:
    """Return the parts that may be resumed for an operating mode."""

    parsed_mode = mode if isinstance(mode, RunMode) else RunMode(mode)
    return RESUMABLE_PARTS_BY_MODE[parsed_mode]


def classify_part_resume_state(
    manifest: DistributedRunManifest,
    part: str,
    *,
    required_outputs: Sequence[str | Path] = (),
    marker_path: str | Path | None = None,
    current_config_hash: str | None = None,
) -> PartResumeState:
    """Classify whether a part can be skipped because valid outputs already exist."""

    output_paths = tuple(Path(path) for path in required_outputs)
    marker = Path(marker_path) if marker_path is not None else None
    if part not in resumable_parts_for_mode(manifest.run_mode):
        return PartResumeState(part, "not_resumable", False, "part is not resumable for mode", marker, output_paths)
    if current_config_hash is not None and current_config_hash != manifest.normalized_config_hash:
        return PartResumeState(part, "stale", False, "current config hash differs", marker, output_paths)
    missing_outputs = [path for path in output_paths if not path.exists()]
    if marker is None:
        if missing_outputs:
            return PartResumeState(part, "missing", False, "required outputs are missing", marker, output_paths)
        return PartResumeState(part, "partial", False, "required outputs exist but no completion marker was provided", marker, output_paths)
    if not marker.exists():
        return PartResumeState(part, "missing", False, "completion marker is missing", marker, output_paths)

    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as error:
        return PartResumeState(part, "stale", False, f"completion marker is invalid JSON: {error}", marker, output_paths)
    status = str(payload.get("status", "")).lower()
    if status == "failed":
        return PartResumeState(part, "failed", False, "part has a failed marker", marker, output_paths)
    if status not in {"completed", "passed", "ok"}:
        return PartResumeState(part, "partial", False, "part marker is not completed", marker, output_paths)
    marker_hash = _marker_config_hash(payload)
    if marker_hash is not None and marker_hash != manifest.normalized_config_hash:
        return PartResumeState(part, "stale", False, "part marker config hash differs", marker, output_paths)
    if missing_outputs:
        return PartResumeState(part, "partial", False, "completion marker exists but outputs are missing", marker, output_paths)
    return PartResumeState(part, "completed", True, "completion marker and outputs are valid", marker, output_paths)


def completed_worker_ids_for_merge(
    manifest: DistributedRunManifest,
    *,
    phase: str,
) -> tuple[int, ...]:
    """Return completed workers or fail if any worker is failed, pending, or stale."""

    layout = build_run_layout(manifest)
    completed: list[int] = []
    blockers: list[str] = []
    for worker_id, worker_layout in layout.workers.items():
        if worker_layout.failed_marker.exists():
            blockers.append(f"worker_{worker_id:03d}:failed")
            continue
        if not worker_layout.completed_marker.exists():
            blockers.append(f"worker_{worker_id:03d}:pending")
            continue
        try:
            marker = read_worker_marker(worker_layout.completed_marker)
        except Exception:
            blockers.append(f"worker_{worker_id:03d}:stale")
            continue
        if marker.phase != phase or marker.run_id != manifest.run_id or marker.worker_id != worker_id:
            blockers.append(f"worker_{worker_id:03d}:stale")
            continue
        completed.append(worker_id)
    if blockers:
        raise ValueError("worker outputs are not mergeable: " + ", ".join(blockers))
    return tuple(completed)


def build_cleanup_plan(
    manifest: DistributedRunManifest,
    *,
    run_failed: bool,
) -> CleanupPlan:
    """Build a cleanup plan without deleting files."""

    candidates = tuple(cleanup_candidates(manifest, run_failed=run_failed))
    preserve_reason = None
    if run_failed:
        preserve_reason = "failed runs preserve partials, logs, metrics, and markers"
    elif not candidates:
        preserve_reason = f"cleanup policy {manifest.cleanup_policy.value} preserves distributed partials"
    return CleanupPlan(
        cleanup_policy=manifest.cleanup_policy.value,
        run_failed=run_failed,
        candidates=candidates,
        preserve_reason=preserve_reason,
    )


def _marker_config_hash(payload: Mapping[str, object]) -> str | None:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("config_hash")
        if value is not None:
            return str(value)
    value = payload.get("config_hash")
    return str(value) if value is not None else None
