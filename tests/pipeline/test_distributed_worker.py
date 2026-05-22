from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

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
    configure_mid_ctx_candidate_pool,
    initialize_pass1_worker_resources,
    run_pass1_worker,
    save_pass1_partials,
    validate_pass1_worker_inputs,
)
from pipeline.distributed.pass1_partials import load_pass1_partial
from pipeline.distributed.shard_table import build_shard_table
from pipeline.runtime import clear_runtime, get_runtime


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


def _write_shards(dataset_path: Path) -> None:
    dataset_path.mkdir(parents=True, exist_ok=True)
    np.save(dataset_path / "shard_0.npy", np.asarray([1, 2, 3, -1, 4, 5, 6], dtype=np.int64))
    np.save(dataset_path / "shard_1.npy", np.asarray([7, 8, 9, -1, 10, 11, 12], dtype=np.int64))


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
