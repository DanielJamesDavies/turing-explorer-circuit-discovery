import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.distributed.layout import MetricEvent, append_metric_event
from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    ManifestStatus,
    RunMode,
)
from pipeline.distributed.reporting import (
    ObservabilitySample,
    append_observability_sample,
    build_final_run_report,
    build_hardware_context,
    build_mode_summary_report,
    save_mode_summary_report,
    save_run_report,
)
from pipeline.distributed.rollout_gates import RolloutGateReport


def _manifest(
    tmp_path: Path,
    *,
    run_mode: RunMode = RunMode.DISTRIBUTED_SIMPLE_EXACT,
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
        normalized_config_hash="abcdef1234567890",
        project_root=str(tmp_path),
        output_root=str(output_root),
        distributed_root=str(distributed_root),
        manifest_path=str(distributed_root / "manifest.json"),
        metrics_path=str(distributed_root / "reports" / "run_metrics.jsonl"),
        run_summary_path=str(distributed_root / "reports" / "run_summary.json"),
        model_path=str(tmp_path / "model.pt"),
        sae_path=str(tmp_path / "sae"),
        dataset_path=str(tmp_path / "data"),
        worker_count=2,
        devices=[
            DeviceAssignment(
                worker_id=0,
                physical_id=0,
                logical_id="cuda:0",
                uuid="GPU-0",
                name="H100",
                pci_bus_id="0000:01:00.0",
                total_vram_bytes=80 * 1024**3,
                hostname="host-a",
            ),
            DeviceAssignment(
                worker_id=1,
                physical_id=1,
                logical_id="cuda:0",
                uuid="GPU-1",
                name="H100",
                pci_bus_id="0000:02:00.0",
                total_vram_bytes=80 * 1024**3,
                hostname="host-a",
            ),
        ],
    )


def test_mode_summary_report_includes_exactness_and_hardware_context(tmp_path):
    manifest = _manifest(tmp_path)
    rollout = RolloutGateReport(
        run_id=manifest.run_id,
        run_mode=manifest.run_mode.value,
        ok=True,
        issues=(),
        required_paths=(),
    )

    report = build_mode_summary_report(manifest, rollout_report=rollout, warnings=["none"])

    assert report["run_mode"] == "distributed_simple_exact"
    assert report["exactness_status"] == "exact_equivalent"
    assert report["rollout_ok"] is True
    assert report["warnings"] == ["none"]
    assert report["hardware"]["device_count"] == 2
    assert report["hardware"]["devices"][0]["uuid"] == "GPU-0"


def test_exactness_statuses_are_visible_for_all_modes(tmp_path):
    assert build_mode_summary_report(_manifest(tmp_path, run_mode=RunMode.SINGLE_PROCESS))[
        "exactness_status"
    ] == "single_process_oracle"
    assert build_mode_summary_report(
        _manifest(tmp_path, run_mode=RunMode.DISTRIBUTED_MAPREDUCE_EXACT)
    )["exactness_status"] == "exact_mapreduce_equivalent"
    assert build_mode_summary_report(
        _manifest(tmp_path, run_mode=RunMode.DISTRIBUTED_EXPERIMENTAL_FAST)
    )["exactness_status"] == "experimental_non_exact"


def test_final_run_report_links_parts_artifacts_equivalence_and_benchmark(tmp_path):
    manifest = _manifest(tmp_path)
    equivalence_path = Path(manifest.distributed_root) / "reports" / "equivalence_one_worker.json"
    benchmark_path = Path(manifest.distributed_root) / "reports" / "benchmark_report.json"
    equivalence_path.parent.mkdir(parents=True, exist_ok=True)
    equivalence_path.write_text(json.dumps({"equivalence": {"passed": True}}), encoding="utf-8")
    benchmark_path.write_text(json.dumps({"benchmark": {"completed": True}}), encoding="utf-8")
    rollout = RolloutGateReport(
        run_id=manifest.run_id,
        run_mode=manifest.run_mode.value,
        ok=False,
        issues=("missing reduced real-data equivalence",),
        required_paths=(equivalence_path,),
    )

    report = build_final_run_report(
        manifest,
        part_statuses={"pass1": "completed", "pass2_reduce": "completed"},
        artifacts={"latent_stats": Path(manifest.output_root) / "latent_stats.pt"},
        rollout_report=rollout,
        equivalence_reports={"one_worker": equivalence_path},
        benchmark_report=benchmark_path,
        warnings=["reduced real-data comparison pending"],
    )

    assert report["mode_summary"]["exactness_status"] == "exact_equivalent"
    assert report["part_statuses"]["pass1"] == "completed"
    assert report["artifacts"]["latent_stats"].endswith("latent_stats.pt")
    assert report["equivalence_reports"]["one_worker"]["equivalence"]["passed"] is True
    assert report["benchmark_report"]["benchmark"]["completed"] is True
    assert report["rollout"]["ok"] is False
    assert report["warnings"] == ["reduced real-data comparison pending"]


def test_report_save_helpers_write_stable_json(tmp_path):
    manifest = _manifest(tmp_path)

    mode_path = save_mode_summary_report(manifest)
    final_path = Path(manifest.run_summary_path)
    save_run_report(
        build_final_run_report(
            manifest,
            part_statuses={"pass1": "planned"},
            artifacts={},
        ),
        final_path,
    )

    assert mode_path == Path(manifest.distributed_root) / "reports" / "mode_summary.json"
    assert json.loads(mode_path.read_text(encoding="utf-8"))["run_id"] == manifest.run_id
    assert json.loads(final_path.read_text(encoding="utf-8"))["mode_summary"]["run_id"] == manifest.run_id


def test_hardware_context_handles_missing_vram(tmp_path):
    manifest = _manifest(
        tmp_path,
    ).model_copy(
        update={
            "devices": [
                DeviceAssignment(worker_id=0, physical_id=None, logical_id="cpu"),
            ],
            "worker_count": 1,
        }
    )

    context = build_hardware_context(manifest)

    assert context["device_count"] == 1
    assert context["total_vram_bytes"] is None
    assert context["devices"][0]["logical_id"] == "cpu"


def test_metric_event_and_observability_jsonl_append(tmp_path):
    metrics_path = tmp_path / "run_metrics.jsonl"
    metric = MetricEvent(
        run_id="20260517-002500-abcdef12",
        worker_id=0,
        phase="pass1",
        event="worker_started",
        counters={"sequence_count": 4},
    )
    append_metric_event(metric, metrics_path)

    observability_path = tmp_path / "observability.jsonl"
    sample = ObservabilitySample(
        run_id="20260517-002500-abcdef12",
        phase="pass1",
        timestamp="2026-05-17T00:25:00Z",
        worker_id=0,
        physical_id=0,
        logical_id="cuda:0",
        gpu_utilization_percent=91.5,
        vram_used_bytes=10,
        vram_total_bytes=20,
        power_watts=400.0,
        temperature_c=65.0,
        cpu_ram_used_bytes=30,
        cpu_ram_total_bytes=40,
        disk_used_bytes=50,
        disk_total_bytes=60,
        disk_write_bytes_per_s=70.0,
    )
    append_observability_sample(sample, observability_path)

    assert MetricEvent.model_validate_json(metrics_path.read_text(encoding="utf-8").strip()) == metric
    assert ObservabilitySample.model_validate_json(
        observability_path.read_text(encoding="utf-8").strip()
    ) == sample


def test_observability_sample_rejects_invalid_values():
    with pytest.raises(ValidationError, match="gpu_utilization_percent"):
        ObservabilitySample(
            run_id="20260517-002500-abcdef12",
            phase="pass1",
            timestamp="2026-05-17T00:25:00Z",
            gpu_utilization_percent=101.0,
        )
    with pytest.raises(ValidationError, match="observability byte values"):
        ObservabilitySample(
            run_id="20260517-002500-abcdef12",
            phase="pass1",
            timestamp="2026-05-17T00:25:00Z",
            vram_used_bytes=-1,
        )
