from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.distributed.manifest import (
    ArtifactSchemaVersions,
    CleanupPolicy,
    DeviceAssignment,
    DistributedRunManifest,
    ManifestStatus,
    RunMode,
    ShardRecord,
    generate_run_id,
    load_manifest,
    save_manifest,
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
        "run_mode": RunMode.DISTRIBUTED_SIMPLE_EXACT,
        "status": ManifestStatus.PLANNED,
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
                "uuid": "GPU-0",
                "name": "H100",
                "pci_bus_id": "0000:01:00.0",
                "total_vram_bytes": 80 * 1024**3,
                "hostname": "host-a",
            },
            {
                "worker_id": 1,
                "physical_id": 1,
                "logical_id": "cuda:0",
                "uuid": "GPU-1",
                "name": "H100",
                "pci_bus_id": "0000:02:00.0",
                "total_vram_bytes": 80 * 1024**3,
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


def _shard_table():
    return [
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
            sequence_count=1,
            global_start_id=3,
            global_end_id=4,
            shard_size_bytes=1,
            shard_mtime_ns=1,
            index_filename=".shard_indices/shard_1.npy_sft1.idx.npy",
            index_size_bytes=1,
            index_mtime_ns=1,
        ),
    ]


def test_generate_run_id_uses_timestamp_and_config_hash():
    run_id = generate_run_id(
        "ABCDEF123456",
        timestamp=datetime(2026, 5, 17, 0, 25, 9, tzinfo=timezone.utc),
    )

    assert run_id == "20260517-002509-abcdef12"


def test_manifest_round_trip(tmp_path):
    manifest = _manifest(tmp_path)
    path = Path(manifest.manifest_path)

    save_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded == manifest
    assert path.exists()


def test_manifest_rejects_unknown_fields(tmp_path):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _manifest(tmp_path, unexpected=True)


def test_manifest_rejects_unsupported_schema_version(tmp_path):
    with pytest.raises(ValidationError, match="unsupported schema version"):
        _manifest(tmp_path, manifest_schema_version=999)

    with pytest.raises(ValidationError, match="unsupported schema version"):
        _manifest(tmp_path, metrics_schema_version=999)

    with pytest.raises(ValidationError, match="unsupported schema version"):
        _manifest(tmp_path, run_summary_schema_version=999)

    with pytest.raises(ValidationError, match="unsupported schema version"):
        _manifest(tmp_path, sanity_report_schema_version=999)


def test_manifest_rejects_invalid_artifact_schema_versions(tmp_path):
    with pytest.raises(ValidationError, match="schema versions must be >= 1"):
        _manifest(
            tmp_path,
            artifact_schema_versions=ArtifactSchemaVersions(
                latent_stats=1,
                top_ctx=1,
            ).model_copy(update={"mid_ctx_candidates": 0}).model_dump(),
        )


def test_manifest_rejects_invalid_run_mode(tmp_path):
    with pytest.raises(ValidationError):
        _manifest(tmp_path, run_mode="distributed_maybe")


def test_manifest_rejects_invalid_cleanup_policy(tmp_path):
    with pytest.raises(ValidationError):
        _manifest(tmp_path, cleanup_policy="delete_everything")


def test_manifest_accepts_negative_context_run_config(tmp_path):
    manifest = _manifest(
        tmp_path,
        neg_ctx={
            "backend": "multi_gpu_exact",
            "selected_devices": ["cuda:0", "cuda:1"],
            "device_selection_source": "manifest_declared_devices",
            "n_neighbors": 128,
            "n_sequences": 16,
            "min_pos_ctx": 4,
            "repr_mode": "mean_pool",
            "max_repr_seqs": 1000,
            "memory_guardrail_fraction": 0.8,
            "fail_on_memory_guardrail": True,
        },
    )

    assert manifest.neg_ctx.backend == "multi_gpu_exact"
    assert manifest.neg_ctx.selected_devices == ["cuda:0", "cuda:1"]
    assert manifest.neg_ctx.n_neighbors == 128
    assert manifest.neg_ctx.memory_guardrail_fraction == 0.8


def test_manifest_rejects_invalid_negative_context_counts(tmp_path):
    with pytest.raises(ValidationError, match="negative-context counts must be >= 1"):
        _manifest(tmp_path, neg_ctx={"n_neighbors": 0})

    with pytest.raises(ValidationError, match="memory_guardrail_fraction"):
        _manifest(tmp_path, neg_ctx={"memory_guardrail_fraction": 1.5})

    with pytest.raises(ValidationError, match="negative-context backend"):
        _manifest(tmp_path, neg_ctx={"backend": "approximate_magic"})


def test_manifest_rejects_duplicate_worker_and_physical_device_ids(tmp_path):
    with pytest.raises(ValidationError, match="device worker IDs must be unique"):
        _manifest(
            tmp_path,
            devices=[
                DeviceAssignment(worker_id=0, physical_id=0).model_dump(),
                DeviceAssignment(worker_id=0, physical_id=1).model_dump(),
            ],
        )

    with pytest.raises(ValidationError, match="physical device IDs must be unique"):
        _manifest(
            tmp_path,
            devices=[
                DeviceAssignment(worker_id=0, physical_id=0).model_dump(),
                DeviceAssignment(worker_id=1, physical_id=0).model_dump(),
            ],
        )


def test_manifest_rejects_non_run_root_output(tmp_path):
    with pytest.raises(ValidationError, match="output_root must end with outputs/<run_id>"):
        _manifest(tmp_path, output_root=str(tmp_path / "outputs"))


def test_manifest_rejects_wrong_manifest_path(tmp_path):
    manifest = _manifest(tmp_path)

    with pytest.raises(
        ValidationError,
        match="manifest_path must be distributed_root/manifest.json",
    ):
        _manifest(tmp_path, manifest_path=str(Path(manifest.output_root) / "manifest.json"))


def test_manifest_rejects_out_of_range_assigned_shards(tmp_path):
    with pytest.raises(ValidationError, match="assigned shard index out of range"):
        _manifest(
            tmp_path,
            shard_table=_shard_table(),
            work_assignments={
                "pass1_shards": {"0": [0, 999]},
                "pass2_sequence_ids": {},
                "discovery_seed_ids": {},
            },
        )


def test_manifest_rejects_duplicate_assigned_sequence_ids(tmp_path):
    with pytest.raises(ValidationError, match="duplicated assigned sequence ID"):
        _manifest(
            tmp_path,
            shard_table=_shard_table(),
            work_assignments={
                "pass1_shards": {"0": [0], "1": [1]},
                "pass2_sequence_ids": {"0": [1, 2], "1": [2, 3]},
                "discovery_seed_ids": {},
            },
        )


def test_manifest_rejects_out_of_range_assigned_sequence_ids(tmp_path):
    with pytest.raises(ValidationError, match="assigned sequence ID out of range"):
        _manifest(
            tmp_path,
            shard_table=_shard_table(),
            work_assignments={
                "pass1_shards": {"0": [0], "1": [1]},
                "pass2_sequence_ids": {"0": [1], "1": [4]},
                "discovery_seed_ids": {},
            },
        )
