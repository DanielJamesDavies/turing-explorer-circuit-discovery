from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline.distributed import controller
from pipeline.distributed import pass1_merge, pass2_reduce
from pipeline.distributed.controller import (
    build_arg_parser,
    build_worker_commands,
    build_discovery_dry_run_estimate,
    classify_resume_workers,
    launch_worker_processes,
    load_and_hash_config,
    main as controller_main,
    plan_distributed_run,
    plan_distributed_run_from_args,
    run_parts_1_to_3,
)
from pipeline.distributed.controller_config import _candidate_dump_m_from_config
from pipeline.distributed.layout import build_worker_marker, create_output_layout, write_worker_marker
from pipeline.distributed.manifest import CleanupPolicy, DeviceAssignment


def _write_config(tmp_path: Path, dataset_path: Path) -> Path:
    return _write_config_with_n_shards(tmp_path, dataset_path, n_shards=2)


def _write_config_with_n_shards(tmp_path: Path, dataset_path: Path, n_shards: int) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": {
                    "model_path": "weights/model.pt",
                    "sae_path": "weights/sae",
                },
                "data": {
                    "dataset_path": str(dataset_path),
                    "n_shards": n_shards,
                    "batch_size": 2,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config_path


def _write_shards(dataset_path: Path) -> None:
    dataset_path.mkdir(parents=True, exist_ok=True)
    np.save(dataset_path / "shard_0.npy", np.asarray([1, 2, 3, -1, 4, 5, 6], dtype=np.int64))
    np.save(dataset_path / "shard_1.npy", np.asarray([7, 8, 9, -1], dtype=np.int64))


def _write_counted_shards(dataset_path: Path, counts: list[int]) -> None:
    dataset_path.mkdir(parents=True, exist_ok=True)
    token = 1
    for shard_idx, count in enumerate(counts):
        values = []
        for _ in range(count):
            values.extend([token, token + 1, token + 2, -1])
            token += 3
        np.save(dataset_path / f"shard_{shard_idx}.npy", np.asarray(values, dtype=np.int64))


def _timestamp() -> datetime:
    return datetime(2026, 5, 17, 0, 25, 0, tzinfo=timezone.utc)


def test_controller_dry_run_creates_manifest_and_one_worker_layout(tmp_path):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=1,
        use_cpu=True,
        timestamp=_timestamp(),
    )

    assert plan.manifest.worker_count == 1
    assert plan.manifest.devices[0].worker_id == 0
    assert plan.manifest.devices[0].physical_id is None
    assert plan.manifest.devices[0].logical_id == "cpu"
    assert plan.manifest.work_assignments.pass1_shards == {"0": [0, 1]}
    assert plan.layout.manifest_path.exists()
    assert plan.layout.workers[0].pass1_dir.exists()
    assert not (tmp_path / "outputs" / "latest").exists()
    assert "worker_000" in plan.dry_run_text
    assert "CUDA_VISIBLE_DEVICES=" in plan.dry_run_text
    assert "discovery estimate:" in plan.dry_run_text
    assert "mode: local_one_worker" in plan.dry_run_text
    assert "probe_batch_size: 16" in plan.dry_run_text
    assert "neg_ctx_eval_max: 16" in plan.dry_run_text
    assert "worker_000: discovery_tasks=256" in plan.dry_run_text
    assert plan.discovery_estimate.mode == "local_one_worker"
    assert plan.local_compatibility.mode == "local_one_worker"
    assert plan.local_compatibility.device_mode == "cpu"
    assert plan.local_compatibility.h100_required is False
    assert "local compatibility:" in plan.dry_run_text
    assert "h100_required: false" in plan.dry_run_text
    assert plan.discovery_estimate.candidate_count == 128
    assert plan.discovery_estimate.seed_method_count == 2
    assert plan.worker_commands[0].environment["CUDA_VISIBLE_DEVICES"] == ""
    assert str(tmp_path / "src") in plan.worker_commands[0].environment["PYTHONPATH"]
    assert plan.worker_commands[0].command[-2:] == ["--worker-id", "0"]
    assert plan.worker_commands[0].command[-4:-2] == ["--phase", "pass1"]


def test_controller_cli_help_documents_phase_three_flags(capsys):
    parser = build_arg_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--dry-run" in output
    assert "--resume" in output
    assert "--run-id" in output
    assert "--phase" in output
    assert "--launch" in output


def test_part_specific_merge_and_reduce_parsers_document_required_commands(capsys):
    with pytest.raises(SystemExit) as pass1_exc:
        pass1_merge.build_arg_parser().parse_args(["--help"])
    assert pass1_exc.value.code == 0
    pass1_output = capsys.readouterr().out
    assert "--manifest" in pass1_output
    assert "--mid-ctx-on-truncation" in pass1_output

    with pytest.raises(SystemExit) as pass2_exc:
        pass2_reduce.build_arg_parser().parse_args(["--help"])
    assert pass2_exc.value.code == 0
    pass2_output = capsys.readouterr().out
    assert "--output-root" in pass2_output
    assert "--candidate-dump" in pass2_output
    assert "--top-ctx" in pass2_output


def test_controller_cli_dry_run_uses_run_id_and_prints_worker_commands(tmp_path, capsys):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)
    run_id = "20260517-002500-abcdef12"

    controller_main(
        [
            "--config",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--output-base",
            str(tmp_path / "outputs"),
            "--mode",
            "distributed_simple_exact",
            "--run-id",
            run_id,
            "--worker-count",
            "1",
            "--use-cpu",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert f"run_id: {run_id}" in output
    assert "run_mode: distributed_simple_exact" in output
    assert "worker_000" in output
    assert "CUDA_VISIBLE_DEVICES=" in output
    assert (tmp_path / "outputs" / run_id / "distributed" / "manifest.json").exists()


def test_controller_cli_plan_can_emit_pass2_worker_commands(tmp_path):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--output-base",
            str(tmp_path / "outputs"),
            "--mode",
            "distributed_simple_exact",
            "--run-id",
            "20260517-002500-abcdef12",
            "--worker-count",
            "1",
            "--use-cpu",
            "--phase",
            "pass2",
        ]
    )

    plan = plan_distributed_run_from_args(args)

    assert plan.worker_commands[0].command[-4:-2] == ["--phase", "pass2"]
    assert " --phase pass2 " in plan.dry_run_text


def test_controller_one_worker_cuda_style_dry_run_for_local_gpu(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)
    monkeypatch.setattr(controller, "_visible_cuda_device_count", lambda: 1)

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=1,
        physical_ids=[0],
        create_layout=False,
        timestamp=_timestamp(),
    )

    assert plan.manifest.devices[0].physical_id == 0
    assert plan.manifest.devices[0].logical_id == "cuda:0"
    assert plan.local_compatibility.mode == "local_one_worker"
    assert plan.local_compatibility.device_mode == "single_cuda"
    assert plan.local_compatibility.h100_required is False
    assert plan.worker_commands[0].environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert str(tmp_path / "src") in plan.worker_commands[0].environment["PYTHONPATH"]
    assert plan.manifest.work_assignments.pass1_shards == {"0": [0, 1]}


def test_controller_local_compatibility_report_reflects_rtx_style_config(tmp_path, monkeypatch):
    dataset_path = tmp_path / "data"
    _write_counted_shards(dataset_path, [1, 1, 1, 1])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": {
                    "model_path": "weights/model.pt",
                    "sae_path": "weights/sae",
                },
                "data": {
                    "dataset_path": str(dataset_path),
                    "n_shards": 4,
                    "batch_size": 128,
                },
                "hardware": {
                    "memory": "efficient",
                    "keep_model_loaded_for_neg_ctx": False,
                },
                "distributed": {
                    "mode": "distributed_simple_exact",
                    "worker_count": 1,
                    "devices": [0],
                },
                "discovery": {
                    "n_seeds": 16,
                    "probe_batch_size": 4,
                    "neg_ctx_eval_max": 16,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(controller, "_visible_cuda_device_count", lambda: 1)

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--output-base",
            str(tmp_path / "outputs"),
            "--run-id",
            "20260517-002500-abcdef12",
        ]
    )
    plan = plan_distributed_run_from_args(args)

    assert plan.manifest.worker_count == 1
    assert plan.manifest.devices[0].physical_id == 0
    assert plan.layout.manifest_path == (
        tmp_path / "outputs" / "20260517-002500-abcdef12" / "distributed" / "manifest.json"
    )
    assert not (tmp_path / "outputs" / "latest").exists()
    assert plan.local_compatibility.memory == "efficient"
    assert plan.local_compatibility.keep_model_loaded_for_neg_ctx is False
    assert plan.local_compatibility.search_cache_deferred is True
    assert plan.local_compatibility.n_shards == 4
    assert plan.local_compatibility.n_seeds == 16
    assert plan.local_compatibility.probe_batch_size == 4
    assert "search_cache_deferred: true" in plan.dry_run_text
    assert "n_shards: 4" in plan.dry_run_text


def test_controller_prints_worker_commands_with_cuda_visible_devices(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)
    monkeypatch.setattr(controller, "_visible_cuda_device_count", lambda: 8)

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=2,
        physical_ids=[3, 1],
        create_layout=False,
        timestamp=_timestamp(),
    )

    assert [cmd.environment["CUDA_VISIBLE_DEVICES"] for cmd in plan.worker_commands] == ["3", "1"]
    assert all(str(tmp_path / "src") in cmd.environment["PYTHONPATH"] for cmd in plan.worker_commands)
    assert "CUDA_VISIBLE_DEVICES=3" in plan.dry_run_text
    assert "CUDA_VISIBLE_DEVICES=1" in plan.dry_run_text
    assert "--manifest" in plan.worker_commands[0].command
    assert "--phase" in plan.worker_commands[0].command


def test_controller_dry_run_includes_pass2_dump_estimate(tmp_path, monkeypatch):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)
    monkeypatch.setattr(controller, "_visible_cuda_device_count", lambda: 2)

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=2,
        physical_ids=[0, 1],
        pass2_sequence_ids=[1, 2, 3],
        create_layout=False,
        timestamp=_timestamp(),
    )

    assert "pass2 candidate dump estimate:" in plan.dry_run_text
    assert "total_dump_bytes: 6144" in plan.dry_run_text
    assert "worker_001: sequences=1 dump_bytes=2048" in plan.dry_run_text


def test_candidate_dump_m_uses_configured_oversample_factor():
    assert _candidate_dump_m_from_config(
        {
            "latents": {
                "top_coactivation": {
                    "n_latents_per_latent": 128,
                    "n_candidates_per_component": 64,
                    "candidate_oversample_factor": 8,
                }
            }
        }
    ) == 1024


def test_controller_h100_style_8_worker_dry_run_has_stable_assignments(
    tmp_path,
    monkeypatch,
):
    counts = [8, 7, 6, 5, 4, 3, 2, 1]
    dataset_path = tmp_path / "data"
    _write_counted_shards(dataset_path, counts)
    config_path = _write_config_with_n_shards(tmp_path, dataset_path, n_shards=len(counts))
    monkeypatch.setattr(controller, "_visible_cuda_device_count", lambda: 8)

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=8,
        physical_ids=list(range(8)),
        create_layout=False,
        timestamp=_timestamp(),
    )

    assert plan.manifest.work_assignments.pass1_shards == {
        str(worker_id): [worker_id] for worker_id in range(8)
    }
    assert plan.manifest.work_assignments.pass1_sequence_totals == {
        str(worker_id): count for worker_id, count in enumerate(counts)
    }
    assert [command.environment["CUDA_VISIBLE_DEVICES"] for command in plan.worker_commands] == [
        str(worker_id) for worker_id in range(8)
    ]
    assert plan.discovery_estimate.mode == "h100_one_worker_per_gpu"
    assert plan.discovery_estimate.replicated_model_sae_workers == 8
    assert plan.h100_exact_mode is not None
    assert plan.h100_exact_mode.one_worker_per_gpu is True
    assert plan.h100_exact_mode.worker_logical_device == "cuda:0"
    assert plan.h100_exact_mode.manifest_declared_devices == tuple(range(8))
    assert plan.h100_exact_mode.neg_ctx_device_source == "manifest_declared_devices"
    assert plan.h100_exact_mode.pass2_reduce_strategy == (
        "simple_exact_candidate_dump_reduce:single_process"
    )
    assert plan.discovery_estimate.expected_worker_task_counts == {
        str(worker_id): 32 for worker_id in range(8)
    }
    assert "worker_007" in plan.dry_run_text
    assert "mode: h100_one_worker_per_gpu" in plan.dry_run_text
    assert "h100 exact mode:" in plan.dry_run_text
    assert "one_worker_per_gpu: true" in plan.dry_run_text
    assert "manifest_declared_devices: [0, 1, 2, 3, 4, 5, 6, 7]" in plan.dry_run_text
    assert "neg_ctx_device_source: manifest_declared_devices" in plan.dry_run_text
    assert "worker_007: discovery_tasks=32" in plan.dry_run_text


def test_controller_h100_mapreduce_mode_reports_entry_criterion(tmp_path, monkeypatch):
    counts = [1] * 8
    dataset_path = tmp_path / "data"
    _write_counted_shards(dataset_path, counts)
    config_path = _write_config_with_n_shards(tmp_path, dataset_path, n_shards=len(counts))
    monkeypatch.setattr(controller, "_visible_cuda_device_count", lambda: 8)

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=8,
        run_mode=controller.RunMode.DISTRIBUTED_MAPREDUCE_EXACT,
        physical_ids=list(range(8)),
        create_layout=False,
        timestamp=_timestamp(),
    )

    assert plan.h100_exact_mode is not None
    assert plan.h100_exact_mode.pass2_reduce_strategy == "mapreduce_target_range_reduce"
    assert "candidate-dump merge or reducer input memory" in (
        plan.h100_exact_mode.mapreduce_entry_criterion
    )
    assert "pass2_reduce_strategy: mapreduce_target_range_reduce" in plan.dry_run_text


def test_controller_builds_manual_discovery_worker_commands(tmp_path):
    manifest = _minimal_manifest(tmp_path, worker_count=2)

    commands = build_worker_commands(manifest, tmp_path, phase="discovery")

    assert commands[0].command[-4:] == ["--phase", "discovery", "--worker-id", "0"]
    assert commands[1].command[-4:] == ["--phase", "discovery", "--worker-id", "1"]


def test_discovery_dry_run_estimate_uses_config_values():
    estimate = build_discovery_dry_run_estimate(
        {
            "weights": {"model_path": "weights/model.pt", "sae_path": "weights/sae"},
            "data": {"dataset_path": "data"},
            "discovery": {
                "n_seeds": 5,
                "probe_batch_size": 4,
                "neg_ctx_eval_max": 3,
                "methods": ["coactivation_statistical", "cluster_contrast"],
            },
        },
        2,
    )

    assert estimate.candidate_count == 5
    assert estimate.seed_method_count == 1
    assert estimate.seed_free_method_count == 1
    assert estimate.probe_batch_size == 4
    assert estimate.neg_ctx_eval_max == 3
    assert estimate.expected_worker_task_counts == {"0": 4, "1": 2}


def test_subprocess_launcher_uses_planned_commands(tmp_path, monkeypatch):
    calls = []

    class FakePopen:
        def __init__(self, command, cwd, env):
            self.command = command
            self.cwd = cwd
            self.env = env
            calls.append((command, cwd, env))

    monkeypatch.setattr(controller.subprocess, "Popen", FakePopen)
    command = build_worker_commands(
        _minimal_manifest(tmp_path),
        tmp_path,
    )[0]

    processes = launch_worker_processes([command])

    assert len(processes) == 1
    assert calls[0][0] == command.command
    assert calls[0][1] == command.cwd
    assert calls[0][2]["CUDA_VISIBLE_DEVICES"] == "0"
    assert str(tmp_path / "src") in calls[0][2]["PYTHONPATH"]


def test_run_parts_1_to_3_composes_workers_merge_and_neg_ctx(tmp_path):
    manifest = _minimal_manifest(tmp_path, worker_count=2)
    calls = []

    def fake_worker_runner(seen_manifest, worker_id):
        calls.append(("worker", worker_id))
        assert seen_manifest is manifest
        return {"latent_stats": f"worker_{worker_id}.pt"}

    def fake_merge_runner(seen_manifest, **kwargs):
        calls.append(("merge", kwargs["seq_latent_index_enabled"], kwargs["vocab_size"]))
        assert seen_manifest is manifest
        return {"status": "completed"}

    def fake_neg_ctx_runner(output_root, **kwargs):
        calls.append(("neg_ctx", output_root, kwargs["manifest_path"], kwargs["resume"]))
        return {"status": "completed"}

    result = run_parts_1_to_3(
        manifest,
        worker_runner=fake_worker_runner,
        merge_runner=fake_merge_runner,
        neg_ctx_runner=fake_neg_ctx_runner,
        seq_latent_index_enabled=False,
        vocab_size=100,
    )

    assert calls == [
        ("worker", 0),
        ("worker", 1),
        ("merge", False, 100),
        ("neg_ctx", manifest.output_root, manifest.manifest_path, True),
    ]
    assert result.worker_artifacts == {
        0: {"latent_stats": "worker_0.pt"},
        1: {"latent_stats": "worker_1.pt"},
    }
    assert result.pass1_merge == {"status": "completed"}
    assert result.negative_context == {"status": "completed"}


def test_resume_classification_completed_failed_pending_and_stale(tmp_path):
    manifest = _minimal_manifest(tmp_path, worker_count=3)
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
        build_worker_marker(
            manifest,
            1,
            phase="pass1",
            status="failed",
            error="boom",
        ),
        layout.workers[1].failed_marker,
    )

    assert classify_resume_workers(manifest) == {
        "pending": [2],
        "completed": [0],
        "failed": [1],
        "stale": [],
    }
    assert classify_resume_workers(manifest, current_config_hash="different") == {
        "pending": [],
        "completed": [],
        "failed": [],
        "stale": [0, 1, 2],
    }


def test_controller_rejects_run_id_collision_unless_resume(tmp_path):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)
    _, config_hash = load_and_hash_config(config_path)
    run_id = f"20260517-002500-{config_hash[:8]}"
    (tmp_path / "outputs" / run_id).mkdir(parents=True)

    with pytest.raises(FileExistsError, match="run ID collision"):
        plan_distributed_run(
            config_path=config_path,
            project_root=tmp_path,
            output_base=tmp_path / "outputs",
            worker_count=1,
            run_id=run_id,
            use_cpu=True,
        )

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=1,
        run_id=run_id,
        use_cpu=True,
        resume=True,
    )
    assert plan.preflight.run_id_collision is True
    assert plan.preflight.free_disk_bytes > 0
    assert plan.preflight.rough_required_disk_bytes > 0


def test_controller_preflight_reports_shard_table_construction(tmp_path):
    dataset_path = tmp_path / "data"
    _write_counted_shards(dataset_path, [2, 1, 3])
    config_path = _write_config_with_n_shards(tmp_path, dataset_path, n_shards=3)

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=1,
        use_cpu=True,
        create_layout=False,
        timestamp=_timestamp(),
    )

    assert [record.sequence_count for record in plan.manifest.shard_table] == [2, 1, 3]
    assert plan.preflight.shard_count == 3
    assert plan.preflight.total_shard_bytes > 0
    assert "preflight_shards: 3" in plan.dry_run_text
    assert [
        (record.global_start_id, record.global_end_id)
        for record in plan.manifest.shard_table
    ] == [(1, 3), (3, 4), (4, 7)]


def test_controller_preflight_rejects_missing_dataset_before_manifest(tmp_path):
    config_path = _write_config(tmp_path, tmp_path / "missing-data")

    with pytest.raises(FileNotFoundError, match="dataset path does not exist"):
        plan_distributed_run(
            config_path=config_path,
            project_root=tmp_path,
            output_base=tmp_path / "outputs",
            worker_count=1,
            use_cpu=True,
        )

    assert not list((tmp_path / "outputs").glob("*/distributed/manifest.json"))


def test_controller_preflight_rejects_unwritable_output_root(tmp_path, monkeypatch):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)

    def fail_writable(_path):
        raise PermissionError("blocked")

    monkeypatch.setattr(controller, "_check_output_writable", fail_writable)

    with pytest.raises(PermissionError, match="blocked"):
        plan_distributed_run(
            config_path=config_path,
            project_root=tmp_path,
            output_base=tmp_path / "outputs",
            worker_count=1,
            use_cpu=True,
        )


def test_controller_preflight_rejects_invisible_or_duplicate_devices(tmp_path, monkeypatch):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)
    monkeypatch.setattr(controller, "_visible_cuda_device_count", lambda: 1)

    with pytest.raises(ValueError, match="physical device IDs not visible"):
        plan_distributed_run(
            config_path=config_path,
            project_root=tmp_path,
            output_base=tmp_path / "outputs",
            worker_count=1,
            physical_ids=[2],
        )

    monkeypatch.setattr(controller, "_visible_cuda_device_count", lambda: 2)
    with pytest.raises(ValueError, match="physical device IDs must be unique"):
        plan_distributed_run(
            config_path=config_path,
            project_root=tmp_path,
            output_base=tmp_path / "outputs",
            worker_count=2,
            physical_ids=[0, 0],
        )


def test_controller_preflight_preserves_cpu_one_worker_fallback_only(tmp_path):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=1,
        use_cpu=True,
        create_layout=False,
    )
    assert plan.manifest.devices[0].logical_id == "cpu"

    with pytest.raises(ValueError, match="CPU fallback supports exactly one worker"):
        plan_distributed_run(
            config_path=config_path,
            project_root=tmp_path,
            output_base=tmp_path / "outputs",
            worker_count=2,
            use_cpu=True,
        )


def test_controller_preflight_rejects_insufficient_disk_space(tmp_path, monkeypatch):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)

    class TinyDisk:
        free = 1

    monkeypatch.setattr(controller.shutil, "disk_usage", lambda _path: TinyDisk())

    with pytest.raises(OSError, match="insufficient disk space"):
        plan_distributed_run(
            config_path=config_path,
            project_root=tmp_path,
            output_base=tmp_path / "outputs",
            worker_count=1,
            use_cpu=True,
        )


def test_controller_accepts_distributed_config_and_records_sampling_seed(tmp_path):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": {"model_path": "weights/model.pt", "sae_path": "weights/sae"},
                "data": {"dataset_path": str(dataset_path), "n_shards": 2, "batch_size": 2},
                "distributed": {
                    "sampling_seed": 123,
                    "mid_ctx_candidate_pool": {
                        "enabled": True,
                        "band_margin_sigma": 0.75,
                        "max_candidates_per_latent": 128,
                        "on_truncation": "fail",
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    plan = plan_distributed_run(
        config_path=config_path,
        project_root=tmp_path,
        output_base=tmp_path / "outputs",
        worker_count=1,
        use_cpu=True,
        create_layout=False,
        timestamp=_timestamp(),
    )

    assert plan.manifest.sampling_seed == 123


def test_controller_rejects_stale_or_unknown_config_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": {"model_path": "model.pt", "sae_path": "sae"},
                "data": {"dataset_path": "data", "n_shards": 1, "batch_size": 1},
                "unknown": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_and_hash_config(config_path)


def test_controller_rejects_missing_native_extensions(tmp_path, monkeypatch):
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    config_path = _write_config(tmp_path, dataset_path)
    monkeypatch.setattr(controller.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="required native extensions unavailable"):
        plan_distributed_run(
            config_path=config_path,
            project_root=tmp_path,
            output_base=tmp_path / "outputs",
            worker_count=1,
            use_cpu=True,
            selected_parts=["pass2_reduce"],
        )


def _minimal_manifest(tmp_path: Path, worker_count: int = 1):
    config_hash = "abcdef1234567890"
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    return controller.DistributedRunManifest(
        run_id=run_id,
        run_mode="distributed_simple_exact",
        status="planned",
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
