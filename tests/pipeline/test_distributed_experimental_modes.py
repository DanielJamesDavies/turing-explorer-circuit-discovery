import json
from pathlib import Path

import pytest

from pipeline.distributed.experimental_modes import (
    EXPERIMENTAL_WARNING_BANNER,
    build_experimental_fast_config,
    validate_experimental_fast_mode,
    write_experimental_fast_report,
)
from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    ManifestStatus,
    RunMode,
)


def _manifest(tmp_path: Path, **overrides) -> DistributedRunManifest:
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / "experimental_fast" / run_id
    distributed_root = output_root / "distributed"
    data = {
        "run_id": run_id,
        "run_mode": RunMode.DISTRIBUTED_EXPERIMENTAL_FAST,
        "status": ManifestStatus.PLANNED,
        "cleanup_policy": CleanupPolicy.KEEP_ALL,
        "created_at": "2026-05-17T00:25:00Z",
        "config_path": str(tmp_path / "config.yaml"),
        "normalized_config_hash": "abcdef1234567890",
        "project_root": str(tmp_path),
        "output_root": str(output_root),
        "distributed_root": str(distributed_root),
        "manifest_path": str(distributed_root / "manifest.json"),
        "metrics_path": str(distributed_root / "reports" / "run_metrics.jsonl"),
        "run_summary_path": str(distributed_root / "reports" / "run_summary.json"),
        "model_path": str(tmp_path / "model.pt"),
        "sae_path": str(tmp_path / "sae"),
        "dataset_path": str(tmp_path / "data"),
        "worker_count": 1,
        "devices": [DeviceAssignment(worker_id=0, physical_id=0, logical_id="cuda:0")],
        "experimental_fast": build_experimental_fast_config(
            acknowledged=True,
            exact_baseline_root=tmp_path / "outputs" / "exact-baseline",
            quality_toggles={
                "local_topk_merge": True,
                "mid_ctx_semantics": "bounded_approx",
            },
        ),
    }
    data.update(overrides)
    return DistributedRunManifest.model_validate(data)


def test_experimental_fast_config_records_warning_baseline_and_toggles(tmp_path):
    config = build_experimental_fast_config(
        acknowledged=True,
        exact_baseline_root=tmp_path / "outputs" / "exact-baseline",
        quality_toggles={"local_topk_merge": True},
    )

    assert config.acknowledged is True
    assert config.exact_baseline_root is not None
    assert config.quality_toggles == {"local_topk_merge": True}
    assert config.warning_banner == EXPERIMENTAL_WARNING_BANNER


def test_experimental_fast_validation_requires_existing_baseline_root(tmp_path):
    manifest = _manifest(tmp_path)

    report = validate_experimental_fast_mode(manifest)

    assert report.ok is False
    assert any("missing exact baseline root" in issue for issue in report.issues)

    Path(manifest.experimental_fast.exact_baseline_root).mkdir(parents=True)
    passed = validate_experimental_fast_mode(manifest)
    assert passed.ok is True
    assert passed.warning_banner == EXPERIMENTAL_WARNING_BANNER
    assert passed.quality_toggles["local_topk_merge"] is True


def test_experimental_fast_validation_rejects_unmarked_output_root(tmp_path):
    baseline_root = tmp_path / "outputs" / "exact-baseline"
    baseline_root.mkdir(parents=True)
    manifest = _manifest(
        tmp_path,
        output_root=str(tmp_path / "outputs" / "20260517-002500-abcdef12"),
        distributed_root=str(tmp_path / "outputs" / "20260517-002500-abcdef12" / "distributed"),
        manifest_path=str(
            tmp_path / "outputs" / "20260517-002500-abcdef12" / "distributed" / "manifest.json"
        ),
        metrics_path=str(
            tmp_path
            / "outputs"
            / "20260517-002500-abcdef12"
            / "distributed"
            / "reports"
            / "run_metrics.jsonl"
        ),
        run_summary_path=str(
            tmp_path
            / "outputs"
            / "20260517-002500-abcdef12"
            / "distributed"
            / "reports"
            / "run_summary.json"
        ),
    )

    report = validate_experimental_fast_mode(manifest)

    assert report.ok is False
    assert "experimental fast output_root must be separate or clearly marked" in report.issues


def test_experimental_fast_validation_rejects_non_experimental_manifest(tmp_path):
    manifest = _manifest(tmp_path, run_mode=RunMode.DISTRIBUTED_SIMPLE_EXACT)

    report = validate_experimental_fast_mode(manifest)

    assert report.ok is False
    assert "manifest run_mode is not distributed_experimental_fast" in report.issues


def test_experimental_fast_report_writes_quality_toggles(tmp_path):
    baseline_root = tmp_path / "outputs" / "exact-baseline"
    baseline_root.mkdir(parents=True)
    manifest = _manifest(tmp_path)
    report = validate_experimental_fast_mode(manifest)
    output_path = Path(manifest.distributed_root) / "reports" / "experimental_fast_report.json"

    write_experimental_fast_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["warning_banner"] == EXPERIMENTAL_WARNING_BANNER
    assert payload["quality_toggles"]["mid_ctx_semantics"] == "bounded_approx"


def test_experimental_fast_validation_can_raise(tmp_path):
    manifest = _manifest(tmp_path)

    with pytest.raises(ValueError, match="missing exact baseline root"):
        validate_experimental_fast_mode(manifest, raise_on_failure=True)
