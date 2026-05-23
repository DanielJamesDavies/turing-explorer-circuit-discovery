import json
from pathlib import Path

import pytest

from pipeline.distributed.layout import (
    build_run_layout,
    build_worker_marker,
    create_output_layout,
    write_worker_marker,
)
from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    ManifestStatus,
    RunMode,
)
from pipeline.distributed.resume_policy import (
    build_cleanup_plan,
    classify_part_resume_state,
    completed_worker_ids_for_merge,
    resumable_parts_for_mode,
)


def _manifest(
    tmp_path: Path,
    *,
    run_mode: RunMode = RunMode.DISTRIBUTED_SIMPLE_EXACT,
    worker_count: int = 2,
    cleanup_policy: CleanupPolicy = CleanupPolicy.KEEP_ALL,
) -> DistributedRunManifest:
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    return DistributedRunManifest(
        run_id=run_id,
        run_mode=run_mode,
        status=ManifestStatus.PLANNED,
        cleanup_policy=cleanup_policy,
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
        worker_count=worker_count,
        devices=[
            DeviceAssignment(worker_id=worker_id, physical_id=worker_id, logical_id="cuda:0")
            for worker_id in range(worker_count)
        ],
    )


def _write_part_marker(path: Path, *, status: str = "completed", config_hash: str = "abcdef1234567890") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": status, "metadata": {"config_hash": config_hash}}, indent=2),
        encoding="utf-8",
    )


def test_resumable_parts_are_defined_by_mode():
    assert resumable_parts_for_mode(RunMode.SINGLE_PROCESS) == ()
    assert "pass1" in resumable_parts_for_mode(RunMode.DISTRIBUTED_SIMPLE_EXACT)
    assert "pass2_mapreduce_reduce" in resumable_parts_for_mode(RunMode.DISTRIBUTED_MAPREDUCE_EXACT)
    assert "experimental_fast_report" in resumable_parts_for_mode(RunMode.DISTRIBUTED_EXPERIMENTAL_FAST)


def test_part_resume_state_classifies_completed_and_skippable(tmp_path):
    manifest = _manifest(tmp_path)
    output = Path(manifest.output_root) / "latent_stats.pt"
    marker = Path(manifest.distributed_root) / "parts" / "pass1_merge" / "completed.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("ok", encoding="utf-8")
    _write_part_marker(marker)

    state = classify_part_resume_state(
        manifest,
        "pass1_merge",
        required_outputs=[output],
        marker_path=marker,
    )

    assert state.status == "completed"
    assert state.can_skip is True


def test_part_resume_state_detects_missing_partial_failed_and_stale(tmp_path):
    manifest = _manifest(tmp_path)
    marker = Path(manifest.distributed_root) / "parts" / "neg_ctx" / "completed.json"

    missing = classify_part_resume_state(
        manifest,
        "neg_ctx",
        required_outputs=[Path(manifest.output_root) / "neg_ctx.pt"],
        marker_path=marker,
    )
    assert missing.status == "missing"

    _write_part_marker(marker, status="running")
    partial = classify_part_resume_state(manifest, "neg_ctx", marker_path=marker)
    assert partial.status == "partial"

    _write_part_marker(marker, status="failed")
    failed = classify_part_resume_state(manifest, "neg_ctx", marker_path=marker)
    assert failed.status == "failed"

    _write_part_marker(marker, status="completed", config_hash="oldhash")
    stale = classify_part_resume_state(manifest, "neg_ctx", marker_path=marker)
    assert stale.status == "stale"

    current_stale = classify_part_resume_state(
        manifest,
        "neg_ctx",
        marker_path=marker,
        current_config_hash="different",
    )
    assert current_stale.status == "stale"


def test_non_resumable_part_and_unmarked_outputs_do_not_skip(tmp_path):
    manifest = _manifest(tmp_path, run_mode=RunMode.SINGLE_PROCESS, worker_count=1)
    output = Path(manifest.output_root) / "latent_stats.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("ok", encoding="utf-8")

    state = classify_part_resume_state(manifest, "pass1", required_outputs=[output])

    assert state.status == "not_resumable"
    assert state.can_skip is False

    distributed_manifest = _manifest(tmp_path)
    unmarked = classify_part_resume_state(
        distributed_manifest,
        "pass1_merge",
        required_outputs=[output],
    )
    assert unmarked.status == "partial"
    assert unmarked.can_skip is False


def test_completed_worker_ids_for_merge_rejects_failed_pending_and_stale(tmp_path):
    manifest = _manifest(tmp_path, worker_count=3)
    layout = create_output_layout(manifest)
    write_worker_marker(
        build_worker_marker(
            manifest,
            0,
            phase="pass1",
            status="completed",
            start_time="2026-05-17T00:00:00Z",
            end_time="2026-05-17T00:00:01Z",
            duration_s=1.0,
        ),
        layout.workers[0].completed_marker,
    )
    write_worker_marker(
        build_worker_marker(manifest, 1, phase="pass1", status="failed", error="boom"),
        layout.workers[1].failed_marker,
    )

    with pytest.raises(ValueError, match="worker_001:failed"):
        completed_worker_ids_for_merge(manifest, phase="pass1")

    layout.workers[1].failed_marker.unlink()
    with pytest.raises(ValueError, match="worker_001:pending"):
        completed_worker_ids_for_merge(manifest, phase="pass1")

    write_worker_marker(
        build_worker_marker(
            manifest,
            1,
            phase="pass2",
            status="completed",
            start_time="2026-05-17T00:00:00Z",
            end_time="2026-05-17T00:00:01Z",
            duration_s=1.0,
        ),
        layout.workers[1].completed_marker,
    )
    with pytest.raises(ValueError, match="worker_001:stale"):
        completed_worker_ids_for_merge(manifest, phase="pass1")


def test_completed_worker_ids_for_merge_accepts_all_completed_workers(tmp_path):
    manifest = _manifest(tmp_path, worker_count=2)
    layout = create_output_layout(manifest)
    for worker_id in range(2):
        write_worker_marker(
            build_worker_marker(
                manifest,
                worker_id,
                phase="pass1",
                status="completed",
                start_time="2026-05-17T00:00:00Z",
                end_time="2026-05-17T00:00:01Z",
                duration_s=1.0,
            ),
            layout.workers[worker_id].completed_marker,
        )

    assert completed_worker_ids_for_merge(manifest, phase="pass1") == (0, 1)


def test_cleanup_plan_preserves_failed_runs_and_keep_all(tmp_path):
    manifest = _manifest(
        tmp_path,
        cleanup_policy=CleanupPolicy.DELETE_LARGE_PARTIALS_ON_SUCCESS,
    )
    layout = create_output_layout(manifest)
    partial = layout.workers[0].pass1_dir / "latent_stats.partial.pt"
    partial.write_text("partial", encoding="utf-8")

    failed_plan = build_cleanup_plan(manifest, run_failed=True)
    assert failed_plan.candidates == ()
    assert "failed runs preserve" in failed_plan.preserve_reason

    keep_all_plan = build_cleanup_plan(
        _manifest(tmp_path, cleanup_policy=CleanupPolicy.KEEP_ALL),
        run_failed=False,
    )
    assert keep_all_plan.candidates == ()
    assert "keep_all" in keep_all_plan.preserve_reason


def test_cleanup_plan_scopes_to_distributed_partials(tmp_path):
    manifest = _manifest(
        tmp_path,
        cleanup_policy=CleanupPolicy.DELETE_LARGE_PARTIALS_ON_SUCCESS,
    )
    layout = create_output_layout(manifest)
    partial = layout.workers[0].pass1_dir / "latent_stats.partial.pt"
    marker = layout.workers[0].started_marker
    unrelated = Path(manifest.output_root) / "notes.pt"
    partial.write_text("partial", encoding="utf-8")
    marker.write_text("marker", encoding="utf-8")
    unrelated.write_text("do not clean", encoding="utf-8")

    plan = build_cleanup_plan(manifest, run_failed=False)

    assert plan.candidates == (partial,)
    assert unrelated not in plan.candidates
    assert marker not in plan.candidates

    delete_all = _manifest(
        tmp_path,
        cleanup_policy=CleanupPolicy.DELETE_ALL_PARTIALS_ON_SUCCESS,
    )
    create_output_layout(delete_all)
    delete_all_plan = build_cleanup_plan(delete_all, run_failed=False)
    assert build_run_layout(delete_all).workers_root in delete_all_plan.candidates
