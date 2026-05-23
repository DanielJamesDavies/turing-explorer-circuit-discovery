"""Rollout gate checks for exact distributed operating modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .manifest import DistributedRunManifest, RunMode, load_manifest


ONE_WORKER_EQUIVALENCE_REPORT = "equivalence_one_worker.json"
TINY_SYNTHETIC_EQUIVALENCE_REPORT = "equivalence_tiny_synthetic.json"
REDUCED_REAL_EQUIVALENCE_REPORT = "equivalence_reduced_real.json"
MAPREDUCE_EQUIVALENCE_REPORT = "equivalence_mapreduce_vs_simple.json"
VERIFICATION_STATUS_REPORT = "verification_status.json"
BENCHMARK_REPORT = "benchmark_report.json"


@dataclass(frozen=True)
class RolloutGateReport:
    run_id: str
    run_mode: str
    ok: bool
    issues: tuple[str, ...]
    required_paths: tuple[Path, ...]


def validate_rollout_gates(
    manifest: DistributedRunManifest,
    *,
    current_config_hash: str | None = None,
    paper_facing: bool = False,
    recommend_as_default: bool = False,
    required_sanity_reports: Sequence[str | Path] | None = None,
    raise_on_failure: bool = False,
) -> RolloutGateReport:
    """Validate mode rollout gates before trusting or recommending a run."""

    issues: list[str] = []
    required_paths: list[Path] = []
    reports_dir = Path(manifest.distributed_root) / "reports"

    _check_manifest_current(
        manifest,
        current_config_hash=current_config_hash,
        issues=issues,
        required_paths=required_paths,
    )
    _require_report(
        reports_dir / VERIFICATION_STATUS_REPORT,
        issues,
        required_paths,
        "verification status",
    )
    for path in _default_sanity_reports(manifest, required_sanity_reports):
        _require_report(path, issues, required_paths, "sanity report")

    if manifest.run_mode == RunMode.DISTRIBUTED_SIMPLE_EXACT:
        if manifest.worker_count > 1:
            _require_report(
                reports_dir / ONE_WORKER_EQUIVALENCE_REPORT,
                issues,
                required_paths,
                "one-worker equivalence",
            )
        if paper_facing:
            _require_report(
                reports_dir / TINY_SYNTHETIC_EQUIVALENCE_REPORT,
                issues,
                required_paths,
                "tiny synthetic equivalence",
            )
            _require_report(
                reports_dir / REDUCED_REAL_EQUIVALENCE_REPORT,
                issues,
                required_paths,
                "reduced real-data equivalence",
            )

    if manifest.run_mode == RunMode.DISTRIBUTED_MAPREDUCE_EXACT:
        _require_report(
            reports_dir / MAPREDUCE_EQUIVALENCE_REPORT,
            issues,
            required_paths,
            "MapReduce vs simple exact equivalence",
        )

    if recommend_as_default:
        _require_report(
            reports_dir / BENCHMARK_REPORT,
            issues,
            required_paths,
            "benchmark report",
        )

    report = RolloutGateReport(
        run_id=manifest.run_id,
        run_mode=manifest.run_mode.value,
        ok=not issues,
        issues=tuple(issues),
        required_paths=tuple(required_paths),
    )
    if raise_on_failure and not report.ok:
        raise ValueError("; ".join(report.issues))
    return report


def write_rollout_gate_report(report: RolloutGateReport, path: str | Path) -> None:
    """Persist a rollout gate report for later run summaries."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": report.run_id,
                "run_mode": report.run_mode,
                "ok": report.ok,
                "issues": list(report.issues),
                "required_paths": [str(path) for path in report.required_paths],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _check_manifest_current(
    manifest: DistributedRunManifest,
    *,
    current_config_hash: str | None,
    issues: list[str],
    required_paths: list[Path],
) -> None:
    manifest_path = Path(manifest.manifest_path)
    required_paths.append(manifest_path)
    if not manifest_path.exists():
        issues.append(f"missing manifest: {manifest_path}")
        return
    try:
        loaded = load_manifest(manifest_path)
    except Exception as error:
        issues.append(f"manifest failed to load: {error}")
        return
    if loaded.run_id != manifest.run_id:
        issues.append("manifest run_id does not match requested run")
    if loaded.normalized_config_hash != manifest.normalized_config_hash:
        issues.append("manifest config hash does not match requested run")
    if current_config_hash is not None and current_config_hash != manifest.normalized_config_hash:
        issues.append("stale manifest: current config hash differs")


def _default_sanity_reports(
    manifest: DistributedRunManifest,
    required_sanity_reports: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    if required_sanity_reports is not None:
        return tuple(Path(path) for path in required_sanity_reports)
    root = Path(manifest.distributed_root)
    return (
        root / "reports" / "pass1_sanity_report.json",
        root / "parts" / "neg_ctx" / "neg_ctx_sanity_report.json",
        root / "reports" / "pass2_reduce_report.json",
        root / "reports" / "discovery_merge_report.json",
    )


def _require_report(
    path: Path,
    issues: list[str],
    required_paths: list[Path],
    label: str,
) -> None:
    required_paths.append(path)
    if not path.exists():
        issues.append(f"missing {label}: {path}")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        issues.append(f"{label} is not valid JSON: {path}: {error}")
        return
    if not _report_payload_passed(payload):
        issues.append(f"{label} is not passing: {path}")


def _report_payload_passed(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    status = payload.get("status")
    if isinstance(status, str) and status.lower() in {"passed", "completed", "ok"}:
        return True
    ok = payload.get("ok")
    if ok is True:
        return True
    validation = payload.get("validation")
    if isinstance(validation, Mapping) and validation.get("ok") is True:
        return True
    equivalence = payload.get("equivalence")
    if isinstance(equivalence, Mapping) and equivalence.get("passed") is True:
        return True
    benchmark = payload.get("benchmark")
    if isinstance(benchmark, Mapping) and benchmark.get("completed") is True:
        return True
    return False
