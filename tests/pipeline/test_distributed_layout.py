from pathlib import Path
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pipeline.distributed.layout import (
    MetricEvent,
    WorkerMarker,
    append_metric_event,
    build_run_layout,
    build_worker_marker,
    cleanup_candidates,
    create_output_layout,
    read_worker_marker,
    validate_worker_completed,
    write_worker_marker,
)
from pipeline.distributed.manifest import (
    CleanupPolicy,
    DistributedRunManifest,
    generate_run_id,
)


def _manifest(tmp_path: Path, **overrides) -> DistributedRunManifest:
    config_hash = "abcdef1234567890"
    run_id = generate_run_id(
        config_hash,
        timestamp=datetime(2026, 5, 17, 0, 25, 0, tzinfo=timezone.utc),
    )
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    data = {
        "run_id": run_id,
        "run_mode": "distributed_simple_exact",
        "cleanup_policy": CleanupPolicy.KEEP_ALL,
        "created_at": "2026-05-17T00:25:00Z",
        "config_path": str(tmp_path / "config.yaml"),
        "normalized_config_hash": config_hash,
        "project_root": str(tmp_path),
        "output_root": str(output_root),
        "distributed_root": str(distributed_root),
        "manifest_path": str(distributed_root / "manifest.json"),
        "metrics_path": str(distributed_root / "reports" / "run_metrics.jsonl"),
        "run_summary_path": str(distributed_root / "reports" / "run_summary.json"),
        "model_path": str(tmp_path / "model.pt"),
        "sae_path": str(tmp_path / "sae"),
        "dataset_path": str(tmp_path / "data"),
        "worker_count": 2,
        "devices": [
            {
                "worker_id": 0,
                "physical_id": 0,
                "logical_id": "cuda:0",
                "hostname": "host-a",
            },
            {
                "worker_id": 1,
                "physical_id": 1,
                "logical_id": "cuda:0",
                "hostname": "host-a",
            },
        ],
        "work_assignments": {
            "pass1_shards": {"0": [0, 2], "1": [1]},
            "pass2_sequence_ids": {},
            "discovery_seed_ids": {},
        },
    }
    data.update(overrides)
    return DistributedRunManifest.model_validate(data)


def test_create_output_layout_uses_canonical_run_root_without_latest(tmp_path):
    manifest = _manifest(tmp_path)
    layout = create_output_layout(manifest)

    assert layout.run_root == Path(manifest.output_root)
    assert layout.distributed_root == Path(manifest.output_root) / "distributed"
    assert layout.manifest_path == Path(manifest.distributed_root) / "manifest.json"
    assert layout.run_metrics_path == Path(manifest.metrics_path)
    assert layout.run_summary_path == Path(manifest.run_summary_path)
    assert layout.workers[0].root == (
        Path(manifest.distributed_root) / "workers" / "worker_000"
    )
    assert layout.workers[0].metrics_path == layout.workers[0].root / "metrics.jsonl"
    assert layout.workers[0].pass1_dir.exists()
    assert layout.workers[0].pass2_dir.exists()
    assert layout.workers[0].discovery_dir.exists()
    assert not (tmp_path / "outputs" / "latest").exists()


def test_worker_marker_atomic_write_replaces_existing_marker(tmp_path):
    manifest = _manifest(tmp_path)
    layout = create_output_layout(manifest)
    first = build_worker_marker(
        manifest,
        0,
        phase="pass1",
        status="started",
        sequence_count=2,
    )
    second = build_worker_marker(
        manifest,
        0,
        phase="pass1",
        status="started",
        sequence_count=5,
    )

    write_worker_marker(first, layout.workers[0].started_marker)
    write_worker_marker(second, layout.workers[0].started_marker)
    loaded = read_worker_marker(layout.workers[0].started_marker)

    assert loaded.sequence_count == 5
    assert not layout.workers[0].started_marker.with_name("started.json.tmp").exists()


def test_failed_worker_marker_requires_error_and_records_metadata(tmp_path):
    manifest = _manifest(tmp_path)
    marker = build_worker_marker(
        manifest,
        0,
        phase="pass1",
        status="failed",
        sequence_count=10,
        error="boom",
    )

    assert marker.status == "failed"
    assert marker.error == "boom"
    assert marker.physical_id == 0
    assert marker.logical_id == "cuda:0"
    assert marker.shard_ids == [0, 2]

    with pytest.raises(ValidationError, match="failed worker markers must include error"):
        WorkerMarker(
            run_id=manifest.run_id,
            worker_id=0,
            phase="pass1",
            status="failed",
            logical_id="cuda:0",
        )


def test_metric_event_jsonl_schema_and_append(tmp_path):
    path = tmp_path / "metrics.jsonl"
    event = MetricEvent(
        run_id="20260517-002509-abcdef12",
        worker_id=0,
        phase="pass1",
        event="artifact_written",
        elapsed_s=1.5,
        physical_id=0,
        logical_id="cuda:0",
        artifact_path="pass1/latent_stats.partial.pt",
        artifact_size_bytes=123,
        counters={"sequence_count": 4},
    )

    append_metric_event(event, path)
    append_metric_event(event.model_copy(update={"event": "worker_completed"}), path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert MetricEvent.model_validate_json(lines[0]) == event

    with pytest.raises(ValidationError, match="metric counters must be >= 0"):
        MetricEvent(
            run_id="20260517-002509-abcdef12",
            phase="pass1",
            event="bad",
            counters={"sequence_count": -1},
        )


def test_completed_worker_validation_requires_started_marker_and_artifacts(tmp_path):
    manifest = _manifest(tmp_path)
    layout = create_output_layout(manifest)
    artifact = layout.workers[0].pass1_dir / "latent_stats.partial.pt"
    artifact.write_text("ok", encoding="utf-8")
    marker = build_worker_marker(
        manifest,
        0,
        phase="pass1",
        status="completed",
        start_time="2026-05-17T00:00:00Z",
        end_time="2026-05-17T00:00:01Z",
        duration_s=1.0,
        artifacts={"latent_stats": str(artifact)},
    )

    with pytest.raises(FileNotFoundError, match="started.json"):
        validate_worker_completed(
            marker,
            layout.workers[0].root,
            required_artifacts=["latent_stats"],
        )

    started = build_worker_marker(manifest, 0, phase="pass1", status="started")
    write_worker_marker(started, layout.workers[0].started_marker)
    validate_worker_completed(
        marker,
        layout.workers[0].root,
        required_artifacts=["latent_stats"],
    )

    missing = marker.model_copy(update={"artifacts": {"latent_stats": "missing.pt"}})
    with pytest.raises(FileNotFoundError, match="declared artifact does not exist"):
        validate_worker_completed(
            missing,
            layout.workers[0].root,
            required_artifacts=["latent_stats"],
        )


def test_completed_worker_validation_requires_timing_metadata(tmp_path):
    manifest = _manifest(tmp_path)
    layout = create_output_layout(manifest)
    write_worker_marker(
        build_worker_marker(manifest, 0, phase="pass1", status="started"),
        layout.workers[0].started_marker,
    )
    marker = build_worker_marker(
        manifest,
        0,
        phase="pass1",
        status="completed",
        artifacts={},
    )

    with pytest.raises(ValueError, match="missing timing metadata"):
        validate_worker_completed(marker, layout.workers[0].root, required_artifacts=[])


def test_cleanup_candidates_preserve_failed_runs_and_unrelated_files(tmp_path):
    manifest = _manifest(
        tmp_path,
        cleanup_policy=CleanupPolicy.DELETE_LARGE_PARTIALS_ON_SUCCESS,
    )
    layout = create_output_layout(manifest)
    large_partial = layout.workers[0].pass1_dir / "latent_stats.partial.pt"
    small_marker = layout.workers[0].started_marker
    unrelated = Path(manifest.output_root) / "notes.txt"
    large_partial.write_text("large", encoding="utf-8")
    small_marker.write_text("marker", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")

    assert cleanup_candidates(manifest, run_failed=True) == []
    assert cleanup_candidates(manifest, run_failed=False) == [large_partial]
    assert unrelated.exists()


def test_cleanup_candidates_for_delete_all_partials_scopes_to_distributed_partials(tmp_path):
    manifest = _manifest(
        tmp_path,
        cleanup_policy=CleanupPolicy.DELETE_ALL_PARTIALS_ON_SUCCESS,
    )
    layout = create_output_layout(manifest)
    unrelated = Path(manifest.output_root) / "latent_stats.pt"
    unrelated.write_text("canonical", encoding="utf-8")

    assert cleanup_candidates(manifest, run_failed=False) == [layout.workers_root]
    assert unrelated.exists()


def test_keep_all_and_manual_cleanup_return_no_candidates(tmp_path):
    for policy in [CleanupPolicy.KEEP_ALL, CleanupPolicy.MANUAL_CLEANUP_ONLY]:
        manifest = _manifest(tmp_path, cleanup_policy=policy)
        create_output_layout(manifest)
        assert cleanup_candidates(manifest, run_failed=False) == []


def test_build_run_layout_does_not_create_directories(tmp_path):
    manifest = _manifest(tmp_path)
    layout = build_run_layout(manifest)

    assert layout.run_root == Path(manifest.output_root)
    assert not layout.run_root.exists()
