import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.distributed.layout import build_run_layout, build_worker_marker, write_worker_marker
from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    ShardRecord,
    WorkAssignments,
)
from pipeline.distributed.pass2_benchmark import (
    build_pass2_benchmark_estimate,
    build_pass2_benchmark_report,
    format_pass2_benchmark_estimate,
    save_pass2_benchmark_estimate,
    save_pass2_benchmark_report,
)
from pipeline.distributed.pass2_replay import hash_replay_sequence_ids


def _manifest(tmp_path: Path) -> DistributedRunManifest:
    replay_ids = [1, 2, 3, 4, 5]
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    return DistributedRunManifest(
        run_id=run_id,
        run_mode="distributed_simple_exact",
        status="planned",
        cleanup_policy=CleanupPolicy.KEEP_ALL,
        created_at=datetime(2026, 5, 17, 0, 25, 0, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
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
            DeviceAssignment(worker_id=0, physical_id=0, logical_id="cuda:0"),
            DeviceAssignment(worker_id=1, physical_id=1, logical_id="cuda:0"),
        ],
        shard_table=[
            ShardRecord(
                shard_index=0,
                shard_filename="shard_0.npy",
                sequence_count=len(replay_ids),
                global_start_id=1,
                global_end_id=len(replay_ids) + 1,
                shard_size_bytes=1,
                shard_mtime_ns=1,
                index_filename=".shard_indices/shard_0.npy_sft1.idx.npy",
                index_size_bytes=1,
                index_mtime_ns=1,
            ),
        ],
        work_assignments=WorkAssignments(
            pass1_shards={"0": [], "1": []},
            pass1_sequence_totals={"0": 0, "1": 0},
            pass2_sequence_ids={"0": [1, 2, 3], "1": [4, 5]},
            pass2_replay_sequence_count=len(replay_ids),
            pass2_replay_sequence_hash=hash_replay_sequence_ids(replay_ids),
        ),
    )


def test_pass2_benchmark_estimate_reports_worker_dump_sizes(tmp_path):
    estimate = build_pass2_benchmark_estimate(_manifest(tmp_path), m=6)

    assert estimate.replay_sequence_count == 5
    assert estimate.total_estimated_dump_bytes == 5 * 6 * 8
    assert estimate.max_worker_sequences == 3
    assert estimate.min_worker_sequences == 2
    assert estimate.assignment_imbalance_ratio == 1.5
    assert [worker.estimated_dump_bytes for worker in estimate.workers] == [144, 96]
    assert "worker_001: sequences=2 dump_bytes=96" in format_pass2_benchmark_estimate(estimate)


def test_pass2_benchmark_estimate_round_trip(tmp_path):
    path = tmp_path / "estimate.json"
    estimate = build_pass2_benchmark_estimate(_manifest(tmp_path), m=4)

    save_pass2_benchmark_estimate(estimate, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["total_estimated_dump_bytes"] == 5 * 4 * 8
    assert loaded["workers"][0]["sequence_count"] == 3


def test_pass2_benchmark_report_aggregates_completed_worker_summaries(tmp_path):
    manifest = _manifest(tmp_path)
    layout = build_run_layout(manifest)
    for worker_id, sequence_count in [(0, 3), (1, 2)]:
        worker_layout = layout.workers[worker_id]
        worker_layout.pass2_dir.mkdir(parents=True, exist_ok=True)
        summary_path = worker_layout.pass2_dir / "pass2_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "sequence_count": sequence_count,
                    "batch_count": worker_id + 1,
                    "dump_elapsed_s": 2.0 + worker_id,
                    "model_forward_s": 1.0 + worker_id,
                    "sae_encode_s": 0.5 + worker_id,
                    "update_dump_s": 0.25 + worker_id,
                    "save_elapsed_s": 0.1 + worker_id,
                    "artifact_size_bytes": 100 + worker_id,
                    "peak_cuda_memory_bytes": 1000 + worker_id,
                }
            ),
            encoding="utf-8",
        )
        write_worker_marker(
            build_worker_marker(
                manifest,
                worker_id,
                phase="pass2",
                status="completed",
                start_time="2026-05-17T00:00:00Z",
                end_time="2026-05-17T00:00:01Z",
                duration_s=10.0 + worker_id,
                batch_count=worker_id + 1,
                peak_cuda_memory_bytes=900 + worker_id,
                artifacts={"pass2_summary": str(summary_path)},
            ),
            worker_layout.completed_marker,
        )

    report = build_pass2_benchmark_report(manifest)

    assert report.completed_worker_count == 2
    assert report.total_replay_sequences == 5
    assert report.total_candidate_dump_bytes == 201
    assert report.total_wall_time_s == 11.0
    assert report.total_worker_time_s == 21.0
    assert report.max_peak_cuda_memory_bytes == 1001
    assert report.workers[1].avg_batch_s == 1.5

    report_path = layout.reports_dir / "pass2_benchmark_report.json"
    save_pass2_benchmark_report(report, report_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["total_candidate_dump_bytes"] == 201
