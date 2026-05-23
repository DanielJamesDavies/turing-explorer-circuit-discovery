"""Synthetic/local equivalence helpers for canonical run-root artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


DEFAULT_CANONICAL_ARTIFACTS = (
    "latent_stats.pt",
    "top_ctx.pt",
    "mid_ctx.pt",
    "neg_ctx.pt",
    "logit_ctx.pt",
    "top_coactivation.pt",
    "candidates.pt",
    "circuits/summary.json",
)


def compare_run_roots(
    baseline_root: str | Path,
    candidate_root: str | Path,
    *,
    artifact_paths: Sequence[str] = DEFAULT_CANONICAL_ARTIFACTS,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Compare canonical artifacts from two run roots and return a stable report."""

    baseline = Path(baseline_root)
    candidate = Path(candidate_root)
    artifact_reports = [
        compare_artifact(
            baseline / relative_path,
            candidate / relative_path,
            artifact_name=relative_path,
            atol=atol,
            rtol=rtol,
        )
        for relative_path in artifact_paths
    ]
    equivalent = all(report["equivalent"] for report in artifact_reports)
    return {
        "schema_version": 1,
        "status": "equivalent" if equivalent else "different",
        "ok": equivalent,
        "equivalence": {"passed": equivalent},
        "baseline_root": str(baseline),
        "candidate_root": str(candidate),
        "tolerances": {"atol": float(atol), "rtol": float(rtol)},
        "artifacts": artifact_reports,
    }


def compare_artifact(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    artifact_name: str | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Compare one torch or JSON artifact."""

    baseline = Path(baseline_path)
    candidate = Path(candidate_path)
    name = artifact_name or baseline.name
    report: dict[str, Any] = {
        "artifact": name,
        "baseline_path": str(baseline),
        "candidate_path": str(candidate),
        "equivalent": False,
        "issues": [],
    }
    if not baseline.exists():
        report["issues"].append("missing baseline artifact")
        return report
    if not candidate.exists():
        report["issues"].append("missing candidate artifact")
        return report

    baseline_payload = _load_artifact(baseline)
    candidate_payload = _load_artifact(candidate)
    equivalent, details = _compare_payloads(
        baseline_payload,
        candidate_payload,
        path=name,
        atol=atol,
        rtol=rtol,
    )
    report["equivalent"] = equivalent
    report.update(details)
    return report


def save_equivalence_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Persist an equivalence report as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def _load_artifact(path: Path) -> Any:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return torch.load(path, map_location="cpu", weights_only=False)


def _compare_payloads(
    baseline: Any,
    candidate: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
) -> tuple[bool, dict[str, Any]]:
    if isinstance(baseline, torch.Tensor) or isinstance(candidate, torch.Tensor):
        return _compare_tensors(baseline, candidate, path=path, atol=atol, rtol=rtol)

    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        baseline_keys = set(baseline)
        candidate_keys = set(candidate)
        issues = []
        if baseline_keys != candidate_keys:
            issues.append(
                {
                    "path": path,
                    "kind": "mapping_keys",
                    "baseline_only": sorted(str(key) for key in baseline_keys - candidate_keys),
                    "candidate_only": sorted(str(key) for key in candidate_keys - baseline_keys),
                }
            )
        child_reports = []
        equivalent = not issues
        for key in sorted(baseline_keys & candidate_keys, key=str):
            child_equivalent, child_details = _compare_payloads(
                baseline[key],
                candidate[key],
                path=f"{path}.{key}",
                atol=atol,
                rtol=rtol,
            )
            equivalent = equivalent and child_equivalent
            if not child_equivalent:
                child_reports.append(child_details)
        return equivalent, {"issues": issues, "children": child_reports}

    if isinstance(baseline, (list, tuple)) and isinstance(candidate, (list, tuple)):
        issues = []
        if len(baseline) != len(candidate):
            issues.append(
                {
                    "path": path,
                    "kind": "sequence_length",
                    "baseline_length": len(baseline),
                    "candidate_length": len(candidate),
                }
            )
        child_reports = []
        equivalent = not issues
        for idx, (baseline_item, candidate_item) in enumerate(zip(baseline, candidate)):
            child_equivalent, child_details = _compare_payloads(
                baseline_item,
                candidate_item,
                path=f"{path}[{idx}]",
                atol=atol,
                rtol=rtol,
            )
            equivalent = equivalent and child_equivalent
            if not child_equivalent:
                child_reports.append(child_details)
        return equivalent, {"issues": issues, "children": child_reports}

    equivalent = baseline == candidate
    return (
        equivalent,
        {
            "issues": []
            if equivalent
            else [
                {
                    "path": path,
                    "kind": "value",
                    "baseline": repr(baseline),
                    "candidate": repr(candidate),
                }
            ]
        },
    )


def _compare_tensors(
    baseline: Any,
    candidate: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(baseline, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        return (
            False,
            {
                "issues": [
                    {
                        "path": path,
                        "kind": "type",
                        "baseline_type": type(baseline).__name__,
                        "candidate_type": type(candidate).__name__,
                    }
                ]
            },
        )
    issues = []
    if baseline.shape != candidate.shape:
        issues.append(
            {
                "path": path,
                "kind": "tensor_shape",
                "baseline_shape": list(baseline.shape),
                "candidate_shape": list(candidate.shape),
            }
        )
    if baseline.dtype != candidate.dtype:
        issues.append(
            {
                "path": path,
                "kind": "tensor_dtype",
                "baseline_dtype": str(baseline.dtype),
                "candidate_dtype": str(candidate.dtype),
            }
        )
    if issues:
        return False, {"issues": issues}

    if baseline.is_floating_point() or candidate.is_floating_point():
        equivalent = torch.allclose(baseline, candidate, atol=atol, rtol=rtol)
        diff = (baseline - candidate).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        return (
            bool(equivalent),
            {
                "max_abs_diff": max_abs_diff,
                "issues": []
                if equivalent
                else [
                    {
                        "path": path,
                        "kind": "tensor_values",
                        "max_abs_diff": max_abs_diff,
                    }
                ],
            },
        )

    equivalent = torch.equal(baseline, candidate)
    return (
        bool(equivalent),
        {
            "issues": []
            if equivalent
            else [{"path": path, "kind": "tensor_values"}],
        },
    )
