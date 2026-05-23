"""Guardrails for opt-in experimental distributed fast modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .manifest import DistributedRunManifest, ExperimentalFastModeRunConfig, RunMode


EXPERIMENTAL_WARNING_BANNER = (
    "EXPERIMENTAL FAST MODE: outputs are not exact and are not paper-eligible "
    "unless compared against exact baseline artifacts."
)


@dataclass(frozen=True)
class ExperimentalFastModeReport:
    run_id: str
    ok: bool
    warning_banner: str
    exact_baseline_root: Path | None
    quality_toggles: Mapping[str, bool | int | float | str]
    issues: tuple[str, ...]


def build_experimental_fast_config(
    *,
    acknowledged: bool,
    exact_baseline_root: str | Path | None,
    quality_toggles: Mapping[str, bool | int | float | str],
) -> ExperimentalFastModeRunConfig:
    """Build the manifest sub-config for an acknowledged experimental run."""

    return ExperimentalFastModeRunConfig(
        acknowledged=acknowledged,
        exact_baseline_root=str(exact_baseline_root) if exact_baseline_root is not None else None,
        quality_toggles=dict(quality_toggles),
        warning_banner=EXPERIMENTAL_WARNING_BANNER if acknowledged else "",
    )


def validate_experimental_fast_mode(
    manifest: DistributedRunManifest,
    *,
    exact_baseline_root: str | Path | None = None,
    quality_toggles: Mapping[str, bool | int | float | str] | None = None,
    raise_on_failure: bool = False,
) -> ExperimentalFastModeReport:
    """Validate experimental mode acknowledgement, baseline, toggles, and output marking."""

    issues: list[str] = []
    configured = manifest.experimental_fast
    baseline_root = Path(
        exact_baseline_root or configured.exact_baseline_root or ""
    ) if (exact_baseline_root or configured.exact_baseline_root) else None
    toggles = dict(quality_toggles if quality_toggles is not None else configured.quality_toggles)

    if manifest.run_mode != RunMode.DISTRIBUTED_EXPERIMENTAL_FAST:
        issues.append("manifest run_mode is not distributed_experimental_fast")
    if not configured.acknowledged:
        issues.append("experimental fast mode requires explicit acknowledgement")
    if baseline_root is None:
        issues.append("experimental fast mode requires exact baseline artifacts")
    elif not baseline_root.exists():
        issues.append(f"missing exact baseline root: {baseline_root}")
    if not toggles:
        issues.append("experimental fast mode requires quality-changing toggles")
    if not _output_root_is_experimentally_marked(Path(manifest.output_root)):
        issues.append("experimental fast output_root must be separate or clearly marked")
    if not configured.warning_banner:
        issues.append("experimental fast mode requires a warning banner")

    report = ExperimentalFastModeReport(
        run_id=manifest.run_id,
        ok=not issues,
        warning_banner=configured.warning_banner or EXPERIMENTAL_WARNING_BANNER,
        exact_baseline_root=baseline_root,
        quality_toggles=toggles,
        issues=tuple(issues),
    )
    if raise_on_failure and not report.ok:
        raise ValueError("; ".join(report.issues))
    return report


def write_experimental_fast_report(
    report: ExperimentalFastModeReport,
    path: str | Path,
) -> None:
    """Write a JSON report recording warning banner and quality-changing toggles."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": report.run_id,
                "ok": report.ok,
                "warning_banner": report.warning_banner,
                "exact_baseline_root": (
                    str(report.exact_baseline_root)
                    if report.exact_baseline_root is not None
                    else None
                ),
                "quality_toggles": dict(report.quality_toggles),
                "issues": list(report.issues),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _output_root_is_experimentally_marked(output_root: Path) -> bool:
    normalized_parts = [part.lower() for part in output_root.parts]
    try:
        outputs_index = len(normalized_parts) - 1 - normalized_parts[::-1].index("outputs")
    except ValueError:
        return False
    marked_parts = normalized_parts[outputs_index + 1 : -1]
    return any("experimental" in part or "fast" in part for part in marked_parts)
