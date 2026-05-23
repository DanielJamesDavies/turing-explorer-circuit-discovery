from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from config import config
from pipeline.distributed.layout import build_run_layout, read_worker_marker
from pipeline.distributed.manifest import (
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    ShardRecord,
    WorkAssignments,
)
from pipeline.distributed.worker import (
    PASS1_PARTIAL_FILENAMES,
    PASS2_PARTIAL_FILENAMES,
    configure_mid_ctx_candidate_pool,
    discovery_methods_for_worker_filter,
    initialize_discovery_worker_resources,
    initialize_pass1_worker_resources,
    initialize_pass2_worker_resources,
    load_assigned_discovery_candidates,
    load_discovery_global_artifacts,
    load_pass2_global_artifacts,
    run_discovery_worker,
    run_pass1_worker,
    run_pass2_worker,
    run_worker,
    save_discovery_worker_inputs,
    save_pass1_partials,
    save_pass2_candidate_dump,
    save_worker_discovery_stats,
    seed_free_methods_for_worker,
    validate_discovery_worker_inputs,
    validate_pass1_worker_inputs,
    validate_pass2_worker_inputs,
)
from pipeline.distributed.pass1_partials import load_pass1_partial
from pipeline.distributed.pass2_partials import load_candidate_dump_partial
from pipeline.distributed.pass2_replay import hash_replay_sequence_ids
from pipeline.distributed.shard_table import build_shard_table
from pipeline.second_pass import SecondPassDumpResult
from pipeline.runtime import clear_runtime, get_runtime
from store.circuits import Circuit, circuit_store


def _manifest(tmp_path: Path, worker_count: int = 2) -> DistributedRunManifest:
    run_id = "20260517-002500-abcdef12"
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    if worker_count == 1:
        work_assignments = WorkAssignments(
            pass1_shards={"0": [0, 1]},
            pass1_sequence_totals={"0": 5},
        )
    else:
        work_assignments = WorkAssignments(
            pass1_shards={"0": [0], "1": [1]},
            pass1_sequence_totals={"0": 2, "1": 3},
        )
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
        worker_count=worker_count,
        devices=[
            DeviceAssignment(worker_id=worker_id, physical_id=worker_id, logical_id="cuda:0")
            for worker_id in range(worker_count)
        ],
        shard_table=[
            ShardRecord(
                shard_index=0,
                shard_filename="shard_0.npy",
                sequence_count=2,
                global_start_id=1,
                global_end_id=3,
                shard_size_bytes=1,
                shard_mtime_ns=1,
                index_filename=".shard_indices/shard_0.npy_sft1.idx.npy",
                index_size_bytes=1,
                index_mtime_ns=1,
            ),
            ShardRecord(
                shard_index=1,
                shard_filename="shard_1.npy",
                sequence_count=3,
                global_start_id=3,
                global_end_id=6,
                shard_size_bytes=1,
                shard_mtime_ns=1,
                index_filename=".shard_indices/shard_1.npy_sft1.idx.npy",
                index_size_bytes=1,
                index_mtime_ns=1,
            ),
        ],
        work_assignments=work_assignments,
    )


def _pass2_manifest(tmp_path: Path, worker_count: int = 2) -> DistributedRunManifest:
    manifest = _manifest(tmp_path, worker_count=worker_count)
    if worker_count == 1:
        pass2_sequence_ids = {"0": [1, 3, 5]}
        replay_ids = [1, 3, 5]
    else:
        pass2_sequence_ids = {"0": [1, 3], "1": [4, 5]}
        replay_ids = [1, 3, 4, 5]
    return manifest.model_copy(
        update={
            "work_assignments": WorkAssignments(
                pass1_shards=manifest.work_assignments.pass1_shards,
                pass1_sequence_totals=manifest.work_assignments.pass1_sequence_totals,
                pass2_sequence_ids=pass2_sequence_ids,
                pass2_replay_sequence_count=len(replay_ids),
                pass2_replay_sequence_hash=hash_replay_sequence_ids(replay_ids),
            )
        }
    )


def _discovery_manifest(tmp_path: Path, worker_count: int = 2) -> DistributedRunManifest:
    manifest = _manifest(tmp_path, worker_count=worker_count)
    if worker_count == 1:
        seed_ids = {"0": [0, 1, 2]}
        candidate_assignments = {
            "0": [
                {
                    "candidate_index": 0,
                    "comp_idx": 1,
                    "latent_idx": 10,
                    "methods": ["coactivation_statistical"],
                    "estimated_task_count": 1,
                },
                {
                    "candidate_index": 1,
                    "comp_idx": 2,
                    "latent_idx": 20,
                    "methods": ["coactivation_statistical"],
                    "estimated_task_count": 1,
                },
                {
                    "candidate_index": 2,
                    "comp_idx": 3,
                    "latent_idx": 30,
                    "methods": ["coactivation_statistical"],
                    "estimated_task_count": 1,
                },
            ]
        }
    else:
        seed_ids = {"0": [0], "1": [1, 2]}
        candidate_assignments = {
            "0": [
                {
                    "candidate_index": 0,
                    "comp_idx": 1,
                    "latent_idx": 10,
                    "methods": ["coactivation_statistical"],
                    "estimated_task_count": 1,
                }
            ],
            "1": [
                {
                    "candidate_index": 1,
                    "comp_idx": 2,
                    "latent_idx": 20,
                    "methods": ["coactivation_statistical"],
                    "estimated_task_count": 1,
                },
                {
                    "candidate_index": 2,
                    "comp_idx": 3,
                    "latent_idx": 30,
                    "methods": ["coactivation_statistical"],
                    "estimated_task_count": 1,
                },
            ],
        }
    return manifest.model_copy(
        update={
            "work_assignments": WorkAssignments(
                pass1_shards=manifest.work_assignments.pass1_shards,
                pass1_sequence_totals=manifest.work_assignments.pass1_sequence_totals,
                pass2_sequence_ids=manifest.work_assignments.pass2_sequence_ids,
                pass2_replay_sequence_count=manifest.work_assignments.pass2_replay_sequence_count,
                pass2_replay_sequence_hash=manifest.work_assignments.pass2_replay_sequence_hash,
                discovery_seed_ids=seed_ids,
                discovery_candidate_assignments=candidate_assignments,
            )
        }
    )


def _write_shards(dataset_path: Path) -> None:
    dataset_path.mkdir(parents=True, exist_ok=True)
    np.save(dataset_path / "shard_0.npy", np.asarray([1, 2, 3, -1, 4, 5, 6], dtype=np.int64))
    np.save(dataset_path / "shard_1.npy", np.asarray([7, 8, 9, -1, 10, 11, 12], dtype=np.int64))


def _write_discovery_artifacts(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    mode = str(config.latents.top_coactivation.mode or "freq_weighted")
    shape = (2, 3)
    ctx_shape = (2, 3, 2)
    topk_shape = (2, 3, 4)
    torch.save(
        {
            "active_count": torch.ones(shape, dtype=torch.int64),
            "seq_count": torch.ones(shape, dtype=torch.int64),
            "mean_seq": torch.ones(shape, dtype=torch.float32),
        },
        run_root / "latent_stats.pt",
    )
    for name in ("top_ctx", "mid_ctx", "neg_ctx"):
        torch.save(
            {
                "ctx_seq_idx": torch.ones(ctx_shape, dtype=torch.int32),
                "ctx_seq_val": torch.ones(ctx_shape, dtype=torch.float32),
            },
            run_root / f"{name}.pt",
        )
    torch.save(
        {
            "latent_counts": torch.ones(shape, dtype=torch.int64),
            "top_tokens": torch.ones(ctx_shape, dtype=torch.int32),
            "top_probs": torch.ones(ctx_shape, dtype=torch.float32),
        },
        run_root / "logit_ctx.pt",
    )
    torch.save(
        {
            "top_indices": torch.ones(topk_shape, dtype=torch.int32),
            "top_values": torch.ones(topk_shape, dtype=torch.float32),
            "mode": mode,
        },
        run_root / "top_coactivation.pt",
    )
    torch.save(
        [
            {"comp_idx": 1, "latent_idx": 10},
            {"comp_idx": 2, "latent_idx": 20},
            {"comp_idx": 3, "latent_idx": 30},
        ],
        run_root / "candidates.pt",
    )


def _real_manifest(tmp_path: Path) -> DistributedRunManifest:
    dataset_path = tmp_path / "data"
    _write_shards(dataset_path)
    shard_table = build_shard_table(dataset_path, n_shards=2)
    return _manifest(tmp_path).model_copy(
        update={
            "dataset_path": str(dataset_path),
            "shard_table": shard_table,
            "work_assignments": WorkAssignments(
                pass1_shards={"0": [0], "1": [1]},
                pass1_sequence_totals={"0": 2, "1": 2},
            ),
        }
    )


def test_run_pass1_worker_uses_assigned_shards_and_writes_markers(tmp_path):
    manifest = _manifest(tmp_path)
    seen = {}

    def fake_initialize(observed_manifest, worker_id):
        seen["initialize"] = (observed_manifest.run_id, worker_id)

    def fake_run_first_pass(**kwargs):
        seen["first_pass_kwargs"] = kwargs

    def fake_save_partials(observed_manifest, worker_id):
        path = build_run_layout(observed_manifest).workers[worker_id].pass1_dir / "latent_stats.partial.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("latent_stats", encoding="utf-8")
        return {"latent_stats": str(path)}

    artifacts = run_pass1_worker(
        manifest,
        1,
        initialize_fn=fake_initialize,
        run_first_pass_fn=fake_run_first_pass,
        save_partials_fn=fake_save_partials,
        validate_inputs_fn=lambda _manifest, _worker_id: None,
    )

    worker_layout = build_run_layout(manifest).workers[1]
    completed = read_worker_marker(worker_layout.completed_marker)
    started = read_worker_marker(worker_layout.started_marker)
    assert artifacts == {"latent_stats": str(worker_layout.pass1_dir / "latent_stats.partial.pt")}
    assert seen["initialize"] == (manifest.run_id, 1)
    assert seen["first_pass_kwargs"]["assigned_shard_ids"] == [1]
    assert seen["first_pass_kwargs"]["seq_latent_index_output_dir"] == str(
        worker_layout.pass1_dir / "seq_latent_index"
    )
    assert started.status == "started"
    assert completed.status == "completed"
    assert completed.sequence_count == 3
    assert completed.shard_ranges == [
        {
            "shard_index": 1,
            "global_start_id": 3,
            "global_end_id": 6,
            "sequence_count": 3,
        }
    ]
    assert completed.artifacts == artifacts


def test_run_pass2_worker_replays_assigned_sequences_and_writes_markers(tmp_path):
    manifest = _pass2_manifest(tmp_path)
    seen = {}

    def fake_load(observed_manifest):
        seen["load"] = observed_manifest.run_id

    def fake_initialize(observed_manifest, worker_id):
        seen["initialize"] = (observed_manifest.run_id, worker_id)

    def fake_run_dump(sequence_ids):
        seen["sequence_ids"] = sequence_ids
        return SecondPassDumpResult(
            sequence_count=len(sequence_ids),
            batch_count=2,
            seq_len=64,
            elapsed_s=0.1,
        )

    def fake_save_dump(observed_manifest, worker_id, dump_result):
        path = build_run_layout(observed_manifest).workers[worker_id].pass2_dir / "candidate_dump.partial.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("candidate_dump", encoding="utf-8")
        seen["save_dump"] = (worker_id, dump_result.batch_count)
        return {"candidate_dump": str(path)}

    artifacts = run_pass2_worker(
        manifest,
        1,
        validate_inputs_fn=lambda _manifest, _worker_id: None,
        load_artifacts_fn=fake_load,
        initialize_fn=fake_initialize,
        run_dump_fn=fake_run_dump,
        save_dump_fn=fake_save_dump,
    )

    worker_layout = build_run_layout(manifest).workers[1]
    completed = read_worker_marker(worker_layout.completed_marker)
    started = read_worker_marker(worker_layout.started_marker)
    assert artifacts == {"candidate_dump": str(worker_layout.pass2_dir / "candidate_dump.partial.pt")}
    assert seen["load"] == manifest.run_id
    assert seen["initialize"] == (manifest.run_id, 1)
    assert seen["sequence_ids"] == [4, 5]
    assert started.status == "started"
    assert started.phase == "pass2"
    assert completed.status == "completed"
    assert completed.phase == "pass2"
    assert completed.sequence_count == 2
    assert completed.sequence_id_min == 4
    assert completed.sequence_id_max == 5
    assert completed.replay_sequence_hash == hash_replay_sequence_ids([1, 3, 4, 5])
    assert completed.batch_count == 2
    assert completed.artifacts == artifacts
    assert seen["save_dump"] == (1, 2)


def test_run_pass2_worker_checks_dump_memory_before_model_init(monkeypatch, tmp_path):
    manifest = _pass2_manifest(tmp_path)
    seen = {}

    monkeypatch.setattr(
        "pipeline.distributed.worker.top_coactivation.M",
        3,
        raising=False,
    )
    monkeypatch.setattr(
        "pipeline.distributed.worker.config.latents.top_coactivation.dump_memory_guardrail_bytes",
        1,
    )
    monkeypatch.setattr(
        "pipeline.distributed.worker.config.latents.top_coactivation.fail_on_dump_memory_guardrail",
        True,
    )

    def should_not_initialize(_manifest, _worker_id):
        seen["initialized"] = True

    with pytest.raises(MemoryError, match="exceeds guardrail"):
        run_pass2_worker(
            manifest,
            1,
            validate_inputs_fn=lambda _manifest, _worker_id: None,
            load_artifacts_fn=lambda _manifest: None,
            initialize_fn=should_not_initialize,
            run_dump_fn=lambda _sequence_ids: SecondPassDumpResult(2, 1, 64, 0.1),
            save_dump_fn=lambda _manifest, _worker_id, _dump_result: {},
        )

    assert "initialized" not in seen


def test_run_worker_dispatches_pass2_phase(monkeypatch, tmp_path):
    manifest = _pass2_manifest(tmp_path)
    manifest_path = Path(manifest.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    seen = {}

    def fake_run_pass2(observed_manifest, worker_id):
        seen["pass2"] = (observed_manifest.run_id, worker_id)
        return {}

    monkeypatch.setattr("pipeline.distributed.worker.run_pass2_worker", fake_run_pass2)

    run_worker(manifest_path, 0, phase="pass2")

    assert seen["pass2"] == (manifest.run_id, 0)


def test_run_worker_dispatches_discovery_phase(monkeypatch, tmp_path):
    manifest = _discovery_manifest(tmp_path)
    manifest_path = Path(manifest.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    seen = {}

    def fake_run_discovery(observed_manifest, worker_id):
        seen["discovery"] = (observed_manifest.run_id, worker_id)
        return {}

    monkeypatch.setattr("pipeline.distributed.worker.run_discovery_worker", fake_run_discovery)

    run_worker(manifest_path, 1, phase="discovery")

    assert seen["discovery"] == (manifest.run_id, 1)


def test_run_discovery_worker_saves_assigned_candidates_and_worker_outputs(tmp_path):
    manifest = _discovery_manifest(tmp_path)
    output_root = Path(manifest.output_root)
    _write_discovery_artifacts(output_root)
    seen = {}

    def fake_load(observed_manifest):
        seen["load"] = observed_manifest.run_id

    def fake_initialize(observed_manifest, worker_id):
        seen["initialize"] = (observed_manifest.run_id, worker_id)

    def fake_run_discovery(candidates, output_dir):
        seen["candidates"] = candidates
        seen["output_dir"] = Path(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "discovered_circuits.pt").write_text("circuits", encoding="utf-8")
        (Path(output_dir) / "summary.json").write_text("[]", encoding="utf-8")

    artifacts = run_discovery_worker(
        manifest,
        1,
        validate_inputs_fn=lambda _manifest, _worker_id: None,
        load_artifacts_fn=fake_load,
        initialize_fn=fake_initialize,
        run_discovery_fn=fake_run_discovery,
    )

    worker_layout = build_run_layout(manifest).workers[1]
    completed = read_worker_marker(worker_layout.completed_marker)
    started = read_worker_marker(worker_layout.started_marker)
    assert seen["load"] == manifest.run_id
    assert seen["initialize"] == (manifest.run_id, 1)
    assert [candidate["candidate_index"] for candidate in seen["candidates"]] == [1, 2]
    assert seen["output_dir"] == worker_layout.discovery_dir / "circuits"
    assert started.phase == "discovery"
    assert completed.status == "completed"
    assert completed.phase == "discovery"
    assert completed.seed_count == 2
    assert artifacts["assigned_candidates"] == str(worker_layout.discovery_dir / "assigned_candidates.pt")
    assert artifacts["assignment_metadata"] == str(worker_layout.discovery_dir / "assignment_metadata.json")
    assert artifacts["worker_discovery_stats"] == str(worker_layout.discovery_dir / "worker_discovery_stats.json")
    assert artifacts["discovered_circuits"] == str(worker_layout.discovery_dir / "circuits" / "discovered_circuits.pt")
    assert artifacts["summary"] == str(worker_layout.discovery_dir / "circuits" / "summary.json")
    assert completed.artifacts == artifacts
    assert not (Path(manifest.output_root) / "circuits").exists()


def test_initialize_discovery_worker_resources_uses_single_worker_device(monkeypatch, tmp_path):
    manifest = _discovery_manifest(tmp_path, worker_count=1).model_copy(
        update={"devices": [DeviceAssignment(worker_id=0, physical_id=0, logical_id="cuda:0")]}
    )
    seen = {}

    class FakeDataLoader:
        def __init__(self, device, pin_memory):
            seen["loader"] = (device, pin_memory)

    class FakeInference:
        def __init__(self, device, compile):
            seen["model"] = (device, compile)

    class FakeSAEBank:
        def __init__(self, devices, load_decoders, compile):
            seen["sae_devices"] = devices

    monkeypatch.setattr("pipeline.distributed.worker.DataLoader", FakeDataLoader)
    monkeypatch.setattr("pipeline.distributed.worker.Inference", FakeInference)
    monkeypatch.setattr("pipeline.distributed.worker.SAEBank", FakeSAEBank)

    try:
        initialize_discovery_worker_resources(manifest, 0)
        runtime = get_runtime()
    finally:
        clear_runtime()

    assert runtime.devices == [torch.device("cuda:0")]
    assert runtime.multi_gpu is False
    assert seen["loader"][0] == torch.device("cuda:0")
    assert seen["model"][0] == torch.device("cuda:0")
    assert seen["sae_devices"] == [torch.device("cuda:0")]


def test_load_assigned_discovery_candidates_rejects_manifest_drift(tmp_path):
    manifest = _discovery_manifest(tmp_path)
    output_root = Path(manifest.output_root)
    _write_discovery_artifacts(output_root)
    torch.save(
        [
            {"comp_idx": 1, "latent_idx": 10},
            {"comp_idx": 999, "latent_idx": 20},
            {"comp_idx": 3, "latent_idx": 30},
        ],
        output_root / "candidates.pt",
    )

    with pytest.raises(ValueError, match="assigned candidate comp_idx mismatch"):
        load_assigned_discovery_candidates(manifest, 1)


def test_load_assigned_discovery_candidates_attaches_worker_metadata(tmp_path):
    manifest = _discovery_manifest(tmp_path)
    output_root = Path(manifest.output_root)
    _write_discovery_artifacts(output_root)

    assigned = load_assigned_discovery_candidates(manifest, 1)

    assert [candidate["candidate_index"] for candidate in assigned] == [1, 2]
    assert {candidate["run_id"] for candidate in assigned} == {manifest.run_id}
    assert {candidate["worker_id"] for candidate in assigned} == {1}
    assert {candidate["config_hash"] for candidate in assigned} == {manifest.normalized_config_hash}
    assert [candidate["methods"] for candidate in assigned] == [
        ["coactivation_statistical"],
        ["coactivation_statistical"],
    ]
    assert "top_coactivation" in assigned[0]["artifact_hashes"]


def test_save_worker_discovery_stats_records_worker_circuits(tmp_path):
    manifest = _discovery_manifest(tmp_path)
    circuit_store.circuits.clear()
    try:
        circuit = Circuit(name="demo")
        circuit_store.add_circuit(circuit)
        stats_path = save_worker_discovery_stats(
            manifest,
            1,
            [{"comp_idx": 2, "latent_idx": 20}],
            task_metrics=[
                {
                    "task_key": "1:coactivation_statistical",
                    "method": "coactivation_statistical",
                    "duration_s": 0.25,
                    "forward_pass_count": 2,
                    "accepted_circuit_count": 1,
                    "peak_cuda_memory_bytes": None,
                }
            ],
        )
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    finally:
        circuit_store.circuits.clear()

    assert stats["candidate_count"] == 1
    assert stats["accepted_circuit_count"] == 1
    assert stats["circuit_uuids"] == [circuit.uuid]
    assert stats["methods"] == ["coactivation_statistical"]
    assert stats["planned_task_count"] == 0
    assert stats["estimated_task_cost"] == 0.0
    assert stats["task_metrics"][0]["forward_pass_count"] == 2


def test_seed_free_methods_are_owned_by_one_worker(tmp_path):
    manifest = _discovery_manifest(tmp_path).model_copy(
        update={
            "work_assignments": WorkAssignments(
                pass1_shards={"0": [0], "1": [1]},
                pass1_sequence_totals={"0": 2, "1": 3},
                discovery_seed_ids={"0": [0], "1": [1, 2]},
                discovery_candidate_assignments={
                    "0": [
                        {
                            "candidate_index": 0,
                            "comp_idx": 1,
                            "latent_idx": 10,
                            "methods": ["coactivation_statistical"],
                            "estimated_task_count": 1,
                        }
                    ],
                    "1": [
                        {
                            "candidate_index": 1,
                            "comp_idx": 2,
                            "latent_idx": 20,
                            "methods": ["coactivation_statistical"],
                            "estimated_task_count": 1,
                        },
                        {
                            "candidate_index": 2,
                            "comp_idx": 3,
                            "latent_idx": 30,
                            "methods": ["coactivation_statistical"],
                            "estimated_task_count": 1,
                        },
                    ],
                },
                discovery_seed_free_method_owners={"cluster_contrast": 0},
            )
        }
    )

    assert seed_free_methods_for_worker(manifest, 0) == ["cluster_contrast"]
    assert seed_free_methods_for_worker(manifest, 1) == []


def test_discovery_methods_filter_prevents_duplicate_cluster_contrast():
    methods = ["coactivation_statistical", "cluster_contrast", "logit_attribution"]

    assert discovery_methods_for_worker_filter(methods, []) == [
        "coactivation_statistical",
        "logit_attribution",
    ]
    assert discovery_methods_for_worker_filter(methods, ["cluster_contrast"]) == methods


def test_validate_discovery_worker_inputs_requires_assignments_and_artifacts(tmp_path):
    manifest = _discovery_manifest(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing discovery input artifacts"):
        validate_discovery_worker_inputs(manifest, 0, validate_on_disk=False)

    output_root = Path(manifest.output_root)
    _write_discovery_artifacts(output_root)

    validate_discovery_worker_inputs(manifest, 0, validate_on_disk=False)


def test_load_discovery_global_artifacts_loads_required_stores(monkeypatch, tmp_path):
    manifest = _discovery_manifest(tmp_path)
    _write_discovery_artifacts(Path(manifest.output_root))
    seen = {}

    class FakeStore:
        def __init__(self, key):
            self.key = key

        def load(self, path):
            seen[self.key] = Path(path)

    monkeypatch.setattr("pipeline.discovery_artifacts.latent_stats", FakeStore("latent_stats"))
    monkeypatch.setattr("pipeline.discovery_artifacts.top_ctx", FakeStore("top_ctx"))
    monkeypatch.setattr("pipeline.discovery_artifacts.mid_ctx", FakeStore("mid_ctx"))
    monkeypatch.setattr("pipeline.discovery_artifacts.neg_ctx", FakeStore("neg_ctx"))
    monkeypatch.setattr("pipeline.discovery_artifacts.logit_ctx", FakeStore("logit_ctx"))
    monkeypatch.setattr("pipeline.discovery_artifacts.top_coactivation", FakeStore("top_coactivation"))

    load_discovery_global_artifacts(manifest)

    output_root = Path(manifest.output_root)
    assert seen == {
        "latent_stats": output_root / "latent_stats.pt",
        "top_ctx": output_root / "top_ctx.pt",
        "mid_ctx": output_root / "mid_ctx.pt",
        "neg_ctx": output_root / "neg_ctx.pt",
        "logit_ctx": output_root / "logit_ctx.pt",
        "top_coactivation": output_root / "top_coactivation.pt",
    }


def test_save_pass1_partials_writes_expected_worker_artifact_names(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path, worker_count=1).model_copy(
        update={
            "devices": [DeviceAssignment(worker_id=0, physical_id=0, logical_id="cuda:0")],
            "work_assignments": WorkAssignments(
                pass1_shards={"0": [0, 1]},
                pass1_sequence_totals={"0": 5},
            ),
        }
    )

    shape = (2, 3)
    ctx_idx = torch.zeros((2, 3, 1), dtype=torch.int32)
    ctx_val = torch.zeros((2, 3, 1), dtype=torch.float32)
    monkeypatch.setattr("pipeline.distributed.worker._component_count", lambda: 2)
    monkeypatch.setattr("pipeline.distributed.worker._d_sae", lambda: 3)
    monkeypatch.setattr("pipeline.distributed.worker._store_mode_for", lambda _name: {})
    monkeypatch.setattr(
        "pipeline.distributed.worker.latent_stats_payload",
        lambda _store: {
            "active_count": torch.zeros(shape, dtype=torch.int64),
            "mean": torch.zeros(shape, dtype=torch.float32),
            "mean_abs": torch.zeros(shape, dtype=torch.float32),
            "m2": torch.zeros(shape, dtype=torch.float32),
            "m2_abs": torch.zeros(shape, dtype=torch.float32),
            "seq_count": torch.zeros(shape, dtype=torch.int64),
            "mean_seq": torch.zeros(shape, dtype=torch.float32),
            "m2_seq": torch.zeros(shape, dtype=torch.float32),
            "component_steps": {},
        },
    )
    monkeypatch.setattr(
        "pipeline.distributed.worker.top_ctx_payload",
        lambda _store: {
            "ctx_seq_idx": ctx_idx,
            "ctx_seq_val": ctx_val,
            "ctx_type": "top",
        },
    )
    monkeypatch.setattr(
        "pipeline.distributed.worker.mid_ctx_candidates_payload",
        lambda _store, **_kwargs: {
            "component_ids": torch.zeros(0, dtype=torch.int16),
            "latent_ids": torch.zeros(0, dtype=torch.int32),
            "sequence_ids": torch.zeros(0, dtype=torch.int32),
            "activation_values": torch.zeros(0, dtype=torch.float32),
            "priorities": torch.zeros(0, dtype=torch.float32),
            "candidate_pool_settings": {},
            "truncation_counters": torch.zeros(shape, dtype=torch.int64),
            "ctx_seq_idx": ctx_idx,
            "ctx_seq_val": ctx_val,
        },
    )
    monkeypatch.setattr(
        "pipeline.distributed.worker._runtime_seq_repr_payload",
        lambda: {
            "repr_buf": torch.zeros((6, 4), dtype=torch.float16),
            "repr_mode": "mean_pool",
            "repr_dim": 4,
            "n_seqs": 5,
            "n_stored": 5,
            "is_capped": False,
        },
    )
    monkeypatch.setattr(
        "pipeline.distributed.worker.logit_ctx_payload",
        lambda _store: {
            "latent_counts": torch.zeros(shape, dtype=torch.int64),
            "top_tokens": torch.zeros((2, 3, 1), dtype=torch.int32),
            "top_probs": torch.zeros((2, 3, 1), dtype=torch.float32),
        },
    )

    artifacts = save_pass1_partials(manifest, 0)

    assert artifacts == {
        artifact_name: str(
            build_run_layout(manifest).workers[0].pass1_dir / partial_filename
        )
        for artifact_name, partial_filename in PASS1_PARTIAL_FILENAMES.items()
    }
    for artifact_name, artifact_path in artifacts.items():
        metadata, payload = load_pass1_partial(
            artifact_path,
            expected_artifact_name=artifact_name,
            expected_config_hash=manifest.normalized_config_hash,
        )
        assert metadata.worker_id == 0
        assert metadata.shard_ids == [0, 1]
        assert isinstance(payload, dict)


def test_initialize_pass1_worker_resources_uses_manifest_total_sequences(
    monkeypatch,
    tmp_path,
):
    manifest = _manifest(tmp_path, worker_count=1).model_copy(
        update={"devices": [DeviceAssignment(worker_id=0, physical_id=0, logical_id="cuda:0")]}
    )
    seen = {}

    class FakeDataLoader:
        def __init__(self, device, pin_memory):
            seen["loader"] = (device, pin_memory)

    class FakeSeqRepr:
        def __init__(self, n_seqs, **kwargs):
            seen["n_seqs"] = n_seqs
            seen["seq_repr_kwargs"] = kwargs

    class FakeInference:
        def __init__(self, device, compile):
            seen["model"] = (device, compile)

    class FakeSAEBank:
        def __init__(self, devices, load_decoders, compile):
            seen["sae_devices"] = devices

    monkeypatch.setattr("pipeline.distributed.worker.DataLoader", FakeDataLoader)
    monkeypatch.setattr("pipeline.distributed.worker.SeqRepr", FakeSeqRepr)
    monkeypatch.setattr("pipeline.distributed.worker.Inference", FakeInference)
    monkeypatch.setattr("pipeline.distributed.worker.SAEBank", FakeSAEBank)
    monkeypatch.setattr(
        "pipeline.distributed.worker.validate_pass1_worker_inputs",
        lambda _manifest, _worker_id: None,
    )

    try:
        initialize_pass1_worker_resources(manifest, 0)
        runtime = get_runtime()
    finally:
        clear_runtime()

    assert runtime.devices == [torch.device("cuda:0")]
    assert runtime.multi_gpu is False
    assert seen["n_seqs"] == 5
    assert "slot_to_id" in seen["seq_repr_kwargs"]
    assert "id_to_slot" in seen["seq_repr_kwargs"]
    assert seen["sae_devices"] == [torch.device("cuda:0")]


def test_initialize_pass2_worker_resources_uses_single_worker_device(monkeypatch, tmp_path):
    manifest = _pass2_manifest(tmp_path, worker_count=1).model_copy(
        update={"devices": [DeviceAssignment(worker_id=0, physical_id=0, logical_id="cuda:0")]}
    )
    seen = {}

    class FakeDataLoader:
        def __init__(self, device, pin_memory):
            seen["loader"] = (device, pin_memory)

    class FakeInference:
        def __init__(self, device, compile):
            seen["model"] = (device, compile)

    class FakeSAEBank:
        def __init__(self, devices, load_decoders, compile):
            seen["sae_devices"] = devices

    monkeypatch.setattr("pipeline.distributed.worker.DataLoader", FakeDataLoader)
    monkeypatch.setattr("pipeline.distributed.worker.Inference", FakeInference)
    monkeypatch.setattr("pipeline.distributed.worker.SAEBank", FakeSAEBank)

    try:
        initialize_pass2_worker_resources(manifest, 0)
        runtime = get_runtime()
    finally:
        clear_runtime()

    assert runtime.devices == [torch.device("cuda:0")]
    assert runtime.multi_gpu is False
    assert seen["loader"][0] == torch.device("cuda:0")
    assert seen["model"][0] == torch.device("cuda:0")
    assert seen["sae_devices"] == [torch.device("cuda:0")]


def test_validate_pass2_worker_inputs_requires_global_artifacts(tmp_path):
    manifest = _pass2_manifest(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing pass2 input artifacts"):
        validate_pass2_worker_inputs(manifest, 0, validate_on_disk=False)

    output_root = Path(manifest.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "top_ctx.pt").write_text("top", encoding="utf-8")
    (output_root / "latent_stats.pt").write_text("stats", encoding="utf-8")

    validate_pass2_worker_inputs(manifest, 0, validate_on_disk=False)


def test_load_pass2_global_artifacts_loads_top_ctx_and_latent_stats(monkeypatch, tmp_path):
    manifest = _pass2_manifest(tmp_path)
    seen = {}

    class FakeStore:
        def __init__(self, key):
            self.key = key

        def load(self, path):
            seen[self.key] = Path(path)

    monkeypatch.setattr("pipeline.distributed.worker.top_ctx", FakeStore("top_ctx"))
    monkeypatch.setattr("pipeline.distributed.worker.latent_stats", FakeStore("latent_stats"))

    load_pass2_global_artifacts(manifest)

    assert seen["top_ctx"] == Path(manifest.output_root) / "top_ctx.pt"
    assert seen["latent_stats"] == Path(manifest.output_root) / "latent_stats.pt"


def test_save_pass2_candidate_dump_writes_expected_partial(monkeypatch, tmp_path):
    manifest = _pass2_manifest(tmp_path)

    class FakeTopCoactivation:
        mode = "raw"
        M = 3
        n_candidates_per_component = 2
        n_latents_per_latent = 4
        num_components = 2
        d_sae = 8
        total_tokens_processed = 128
        candidate_ids = torch.tensor(
            [
                [1, 2, 0],
                [4, 5, 0],
            ],
            dtype=torch.int32,
        )
        candidate_vals = torch.tensor(
            [
                [1.5, 0.5, 0.0],
                [2.5, 0.25, 0.0],
            ],
            dtype=torch.float32,
        )

    monkeypatch.setattr("pipeline.distributed.worker.top_coactivation", FakeTopCoactivation())
    dump_result = SecondPassDumpResult(
        sequence_count=2,
        batch_count=1,
        seq_len=64,
        elapsed_s=0.1,
    )

    artifacts = save_pass2_candidate_dump(manifest, 1, dump_result)

    expected_path = (
        build_run_layout(manifest).workers[1].pass2_dir
        / PASS2_PARTIAL_FILENAMES["candidate_dump"]
    )
    summary_path = (
        build_run_layout(manifest).workers[1].pass2_dir
        / PASS2_PARTIAL_FILENAMES["pass2_summary"]
    )
    assert artifacts == {
        "candidate_dump": str(expected_path),
        "pass2_summary": str(summary_path),
    }
    metadata, payload = load_candidate_dump_partial(
        expected_path,
        expected_config_hash=manifest.normalized_config_hash,
    )
    assert metadata.worker_id == 1
    assert metadata.mode == "raw"
    assert metadata.sequence_count == 2
    assert metadata.sequence_id_min == 4
    assert metadata.sequence_id_max == 5
    assert metadata.token_count == 128
    assert payload["sequence_ids"].tolist() == [4, 5]
    assert torch.equal(payload["candidate_ids"], FakeTopCoactivation.candidate_ids)
    assert torch.allclose(payload["candidate_vals"], FakeTopCoactivation.candidate_vals)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["sequence_count"] == 2
    assert summary["batch_count"] == 1
    assert summary["artifact_size_bytes"] > 0
    assert summary["timing_available"]["save"] is True


def test_run_pass2_worker_marks_failed_when_dump_validation_fails(tmp_path):
    manifest = _pass2_manifest(tmp_path)

    def bad_save_dump(_manifest, _worker_id, _dump_result):
        raise ValueError("invalid candidate dump")

    with pytest.raises(ValueError, match="invalid candidate dump"):
        run_pass2_worker(
            manifest,
            1,
            validate_inputs_fn=lambda _manifest, _worker_id: None,
            load_artifacts_fn=lambda _manifest: None,
            initialize_fn=lambda _manifest, _worker_id: None,
            run_dump_fn=lambda _sequence_ids: SecondPassDumpResult(2, 1, 64, 0.1),
            save_dump_fn=bad_save_dump,
        )

    worker_layout = build_run_layout(manifest).workers[1]
    failed = read_worker_marker(worker_layout.failed_marker)
    assert failed.status == "failed"
    assert failed.phase == "pass2"
    assert failed.error == "invalid candidate dump"
    assert not worker_layout.completed_marker.exists()


def test_configure_mid_ctx_candidate_pool_widens_band_and_capacity(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)

    class FakeMidCtx:
        _allocated = False
        _band_low = 0.5
        _band_high = 1.5
        num_ctx_sequences = 64
        mid_mode = "reservoir_cpu"

    fake_mid_ctx = FakeMidCtx()
    monkeypatch.setattr("pipeline.distributed.worker.mid_ctx", fake_mid_ctx)

    configure_mid_ctx_candidate_pool(manifest)

    assert fake_mid_ctx._distributed_candidate_pool is True
    assert fake_mid_ctx._final_num_ctx_sequences == 64
    assert fake_mid_ctx.num_ctx_sequences == 256
    assert fake_mid_ctx._band_low == 0.0
    assert fake_mid_ctx._band_high == 2.5
    assert fake_mid_ctx._candidate_band_margin == 1.0
    assert fake_mid_ctx._candidate_pool_dataset_fingerprint


def test_validate_pass1_worker_inputs_accepts_full_disjoint_real_shards(tmp_path):
    manifest = _real_manifest(tmp_path)

    validate_pass1_worker_inputs(manifest, 0)
    validate_pass1_worker_inputs(manifest, 1)


def test_validate_pass1_worker_inputs_rejects_missing_assignment(tmp_path):
    manifest = _real_manifest(tmp_path).model_copy(
        update={
            "work_assignments": WorkAssignments(
                pass1_shards={"0": [0], "1": []},
                pass1_sequence_totals={"0": 2, "1": 0},
            )
        }
    )

    with pytest.raises(ValueError, match="missing shards"):
        validate_pass1_worker_inputs(manifest, 0)


def test_validate_pass1_worker_inputs_rejects_duplicate_assignment(tmp_path):
    manifest = _real_manifest(tmp_path).model_copy(
        update={
            "work_assignments": WorkAssignments(
                pass1_shards={"0": [0], "1": [0]},
                pass1_sequence_totals={"0": 2, "1": 2},
            )
        }
    )

    with pytest.raises(ValueError, match="duplicated across workers"):
        validate_pass1_worker_inputs(manifest, 0)


def test_validate_pass1_worker_inputs_rejects_out_of_range_assignment(tmp_path):
    manifest = _real_manifest(tmp_path).model_copy(
        update={
            "work_assignments": WorkAssignments(
                pass1_shards={"0": [0], "1": [999]},
                pass1_sequence_totals={"0": 2, "1": 0},
            )
        }
    )

    with pytest.raises(ValueError, match="assigned shard index out of range"):
        validate_pass1_worker_inputs(manifest, 0)


def test_validate_pass1_worker_inputs_rejects_stale_or_missing_shard_table(tmp_path):
    manifest = _real_manifest(tmp_path)
    Path(manifest.dataset_path, "shard_1.npy").unlink()

    with pytest.raises(ValueError, match="shard files or shard order differ"):
        validate_pass1_worker_inputs(manifest, 0)
