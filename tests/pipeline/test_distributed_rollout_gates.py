import json
from pathlib import Path

import pytest

from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    ManifestStatus,
    RunMode,
    save_manifest,
)
from pipeline.distributed.rollout_gates import (
    BENCHMARK_REPORT,
    MAPREDUCE_EQUIVALENCE_REPORT,
    ONE_WORKER_EQUIVALENCE_REPORT,
    REDUCED_REAL_EQUIVALENCE_REPORT,
    TINY_SYNTHETIC_EQUIVALENCE_REPORT,
    VERIFICATION_STATUS_REPORT,
    validate_rollout_gates,
    write_rollout_gate_report,
)


def _manifest(
    tmp_path: Path,
    *,
    run_mode: RunMode = RunMode.DISTRIBUTED_SIMPLE_EXACT,
    worker_count: int = 2,
    config_hash: str = "abcdef1234567890",
) -> DistributedRunManifest:
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    return DistributedRunManifest(
        run_id=run_id,
        run_mode=run_mode,
        status=ManifestStatus.PLANNED,
        cleanup_policy=CleanupPolicy.KEEP_ALL,
        created_at="2026-05-17T00:25:00Z",
        config_path=str(tmp_path / "config.yaml"),
        normalized_config_hash=config_hash,
        project_root=str(tmp_path),
        output_root=str(output_root),
        distributed_root=str(distributed_root),
        manifest_path=str(distributed_root / "manifest.json"),
        metrics_path=str(distributed_root / "reports" / "run_metrics.jsonl"),
        run_summary_path=str(distributed_root / "reports" / "run_summary.json"),
        model_path=str(tmp_path / "model.pt"),
        sae_path=str(tmp_path / "sae"),
        dataset_path=str(tmp_path / "data"),
        worker_count=worker_count,
        devices=[
            DeviceAssignment(worker_id=worker_id, physical_id=worker_id, logical_id="cuda:0")
            for worker_id in range(worker_count)
        ],
    )


def _write_report(path: Path, payload=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"status": "passed"}, indent=2), encoding="utf-8")


def _write_manifest_and_required_reports(manifest: DistributedRunManifest) -> None:
    save_manifest(manifest, manifest.manifest_path)
    reports_dir = Path(manifest.distributed_root) / "reports"
    _write_report(reports_dir / VERIFICATION_STATUS_REPORT)
    _write_report(reports_dir / "pass1_sanity_report.json", {"status": "completed"})
    _write_report(Path(manifest.distributed_root) / "parts" / "neg_ctx" / "neg_ctx_sanity_report.json")
    _write_report(reports_dir / "pass2_reduce_report.json", {"ok": True})
    _write_report(reports_dir / "discovery_merge_report.json", {"validation": {"ok": True}})


def test_rollout_gates_reject_missing_manifest_and_sanity_reports(tmp_path):
    manifest = _manifest(tmp_path, worker_count=1)

    report = validate_rollout_gates(manifest)

    assert report.ok is False
    assert any("missing manifest" in issue for issue in report.issues)
    assert any("missing verification status" in issue for issue in report.issues)
    assert any("missing sanity report" in issue for issue in report.issues)


def test_rollout_gates_reject_stale_manifest_config_hash(tmp_path):
    manifest = _manifest(tmp_path, worker_count=1)
    _write_manifest_and_required_reports(manifest)

    report = validate_rollout_gates(manifest, current_config_hash="newhash")

    assert report.ok is False
    assert "stale manifest: current config hash differs" in report.issues


def test_simple_exact_multi_worker_requires_one_worker_equivalence(tmp_path):
    manifest = _manifest(tmp_path, worker_count=2)
    _write_manifest_and_required_reports(manifest)

    report = validate_rollout_gates(manifest)

    assert report.ok is False
    assert any("one-worker equivalence" in issue for issue in report.issues)

    _write_report(Path(manifest.distributed_root) / "reports" / ONE_WORKER_EQUIVALENCE_REPORT)
    passed = validate_rollout_gates(manifest)
    assert passed.ok is True


def test_paper_facing_simple_exact_requires_tiny_and_reduced_equivalence(tmp_path):
    manifest = _manifest(tmp_path, worker_count=2)
    _write_manifest_and_required_reports(manifest)
    reports_dir = Path(manifest.distributed_root) / "reports"
    _write_report(reports_dir / ONE_WORKER_EQUIVALENCE_REPORT)

    report = validate_rollout_gates(manifest, paper_facing=True)

    assert report.ok is False
    assert any("tiny synthetic equivalence" in issue for issue in report.issues)
    assert any("reduced real-data equivalence" in issue for issue in report.issues)

    _write_report(reports_dir / TINY_SYNTHETIC_EQUIVALENCE_REPORT)
    _write_report(reports_dir / REDUCED_REAL_EQUIVALENCE_REPORT)
    assert validate_rollout_gates(manifest, paper_facing=True).ok is True


def test_mapreduce_exact_requires_equivalence_against_simple_exact(tmp_path):
    manifest = _manifest(tmp_path, run_mode=RunMode.DISTRIBUTED_MAPREDUCE_EXACT, worker_count=8)
    _write_manifest_and_required_reports(manifest)

    report = validate_rollout_gates(manifest)

    assert report.ok is False
    assert any("MapReduce vs simple exact equivalence" in issue for issue in report.issues)

    _write_report(Path(manifest.distributed_root) / "reports" / MAPREDUCE_EQUIVALENCE_REPORT)
    assert validate_rollout_gates(manifest).ok is True


def test_recommending_defaults_requires_benchmark_report(tmp_path):
    manifest = _manifest(tmp_path, worker_count=2)
    _write_manifest_and_required_reports(manifest)
    reports_dir = Path(manifest.distributed_root) / "reports"
    _write_report(reports_dir / ONE_WORKER_EQUIVALENCE_REPORT)

    report = validate_rollout_gates(manifest, recommend_as_default=True)

    assert report.ok is False
    assert any("benchmark report" in issue for issue in report.issues)

    _write_report(reports_dir / BENCHMARK_REPORT, {"benchmark": {"completed": True}})
    assert validate_rollout_gates(manifest, recommend_as_default=True).ok is True


def test_rollout_gate_report_can_be_written_and_failures_can_raise(tmp_path):
    manifest = _manifest(tmp_path, worker_count=1)
    _write_manifest_and_required_reports(manifest)
    report = validate_rollout_gates(manifest)
    output_path = Path(manifest.distributed_root) / "reports" / "rollout_gate_report.json"

    write_rollout_gate_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["run_id"] == manifest.run_id

    (Path(manifest.distributed_root) / "reports" / VERIFICATION_STATUS_REPORT).unlink()
    with pytest.raises(ValueError, match="verification status"):
        validate_rollout_gates(
            manifest,
            required_sanity_reports=[],
            raise_on_failure=True,
        )
