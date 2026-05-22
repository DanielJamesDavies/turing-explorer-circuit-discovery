from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from pipeline.distributed.manifest import (
    CleanupPolicy,
    DistributedRunManifest,
    ManifestStatus,
    RunMode,
    ShardRecord,
    generate_run_id,
)
from pipeline.distributed.pass2_replay import (
    assign_pass2_replay_sequences,
    build_pass2_replay_list,
    get_pass2_worker_input,
    hash_replay_sequence_ids,
    validate_pass2_replay_assignments,
)
from pipeline.distributed.layout import build_worker_marker


class FakeTopCtx:
    def __init__(self, sequence_ids: list[int]) -> None:
        self._sequence_ids = sequence_ids

    def get_all_sequence_ids(self) -> list[int]:
        return list(self._sequence_ids)


def _shard_table() -> list[ShardRecord]:
    return [
        ShardRecord(
            shard_index=0,
            shard_filename="shard_0.npy",
            sequence_count=3,
            global_start_id=1,
            global_end_id=4,
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
            global_start_id=4,
            global_end_id=7,
            shard_size_bytes=1,
            shard_mtime_ns=1,
            index_filename=".shard_indices/shard_1.npy_sft1.idx.npy",
            index_size_bytes=1,
            index_mtime_ns=1,
        ),
    ]


def _manifest(tmp_path: Path, *, worker_count: int = 2) -> DistributedRunManifest:
    config_hash = "abcdef1234567890"
    run_id = generate_run_id(
        config_hash,
        timestamp=datetime(2026, 5, 17, 0, 25, 0, tzinfo=timezone.utc),
    )
    output_root = tmp_path / "outputs" / run_id
    distributed_root = output_root / "distributed"
    if worker_count == 1:
        pass1_shards = {"0": [0, 1]}
        pass1_totals = {"0": 6}
    else:
        pass1_shards = {str(worker_id): [] for worker_id in range(worker_count)}
        pass1_totals = {str(worker_id): 0 for worker_id in range(worker_count)}
        pass1_shards["0"] = [0]
        pass1_totals["0"] = 3
        pass1_shards["1"] = [1]
        pass1_totals["1"] = 3

    return DistributedRunManifest.model_validate(
        {
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
            "worker_count": worker_count,
            "devices": [
                {
                    "worker_id": worker_id,
                    "physical_id": worker_id,
                    "logical_id": "cuda:0",
                }
                for worker_id in range(worker_count)
            ],
            "shard_table": [record.model_dump() for record in _shard_table()],
            "work_assignments": {
                "pass1_shards": pass1_shards,
                "pass1_sequence_totals": pass1_totals,
                "pass2_sequence_ids": {},
                "discovery_seed_ids": {},
            },
        }
    )


def test_build_pass2_replay_list_sorts_deduplicates_and_excludes_zero():
    replay = build_pass2_replay_list(
        FakeTopCtx([5, 0, 3, 5, 2, 0]),
        shard_table=_shard_table(),
    )

    assert replay.sequence_ids == [2, 3, 5]
    assert replay.sequence_count == 3
    assert replay.sequence_hash == hash_replay_sequence_ids([2, 3, 5])


def test_build_pass2_replay_list_accepts_top_ctx_payload_tensor():
    replay = build_pass2_replay_list(
        {
            "ctx_seq_idx": torch.tensor(
                [
                    [[0, 4, 2], [6, 4, 0]],
                ],
                dtype=torch.int32,
            ),
        },
        shard_table=_shard_table(),
    )

    assert replay.sequence_ids == [2, 4, 6]


def test_build_pass2_replay_list_rejects_missing_sequence_ids():
    with pytest.raises(ValueError, match="replay sequence ID out of range: 7"):
        build_pass2_replay_list(
            FakeTopCtx([2, 7]),
            shard_table=_shard_table(),
        )


def test_assign_pass2_replay_sequences_updates_manifest_assignments_and_summary(tmp_path):
    manifest = _manifest(tmp_path)

    updated = assign_pass2_replay_sequences(
        manifest,
        FakeTopCtx([6, 2, 4, 3, 0, 2]),
    )

    assert updated.work_assignments.pass2_sequence_ids == {
        "0": [2, 3],
        "1": [4, 6],
    }
    assert updated.work_assignments.pass2_replay_sequence_count == 4
    assert updated.work_assignments.pass2_replay_sequence_hash == hash_replay_sequence_ids(
        [2, 3, 4, 6]
    )
    assert manifest.work_assignments.pass2_sequence_ids == {}


def test_manifest_rejects_pass2_replay_count_mismatch(tmp_path):
    manifest_data = _manifest(tmp_path).model_dump()
    manifest_data["work_assignments"]["pass2_sequence_ids"] = {"0": [1], "1": [2]}
    manifest_data["work_assignments"]["pass2_replay_sequence_count"] = 3
    manifest_data["work_assignments"]["pass2_replay_sequence_hash"] = hash_replay_sequence_ids([1, 2])

    with pytest.raises(ValueError, match="pass2 replay sequence count does not match"):
        DistributedRunManifest.model_validate(manifest_data)


def test_pass2_replay_assignments_cover_replay_list_with_remainder_chunks(tmp_path):
    manifest = _manifest(tmp_path, worker_count=3)

    updated = assign_pass2_replay_sequences(
        manifest,
        FakeTopCtx([1, 2, 3, 4, 5]),
    )

    assert updated.work_assignments.pass2_sequence_ids == {
        "0": [1, 2],
        "1": [3, 4],
        "2": [5],
    }
    validate_pass2_replay_assignments(updated)
    flattened = [
        sequence_id
        for worker_id in range(updated.worker_count)
        for sequence_id in updated.work_assignments.pass2_sequence_ids[str(worker_id)]
    ]
    assert flattened == [1, 2, 3, 4, 5]


def test_pass2_replay_assignments_support_one_worker_mode(tmp_path):
    manifest = _manifest(tmp_path, worker_count=1)

    updated = assign_pass2_replay_sequences(
        manifest,
        FakeTopCtx([6, 2, 4]),
    )

    assert updated.work_assignments.pass2_sequence_ids == {"0": [2, 4, 6]}
    worker_input = get_pass2_worker_input(updated, 0)
    assert worker_input.sequence_ids == [2, 4, 6]
    assert worker_input.sequence_count == 3
    assert worker_input.sequence_id_min == 2
    assert worker_input.sequence_id_max == 6
    assert worker_input.replay_sequence_hash == hash_replay_sequence_ids([2, 4, 6])


def test_validate_pass2_replay_assignments_rejects_non_contiguous_chunks(tmp_path):
    manifest = assign_pass2_replay_sequences(
        _manifest(tmp_path),
        FakeTopCtx([1, 2, 3, 4]),
    )
    manifest_data = manifest.model_dump()
    manifest_data["work_assignments"]["pass2_sequence_ids"] = {
        "0": [1, 3],
        "1": [2, 4],
    }
    manifest_data["work_assignments"]["pass2_replay_sequence_hash"] = hash_replay_sequence_ids(
        [1, 3, 2, 4]
    )
    non_contiguous = DistributedRunManifest.model_validate(manifest_data)

    with pytest.raises(ValueError, match="sorted replay-list order"):
        validate_pass2_replay_assignments(non_contiguous)


def test_validate_pass2_replay_assignments_rejects_stale_hash(tmp_path):
    manifest = assign_pass2_replay_sequences(
        _manifest(tmp_path),
        FakeTopCtx([1, 2, 3, 4]),
    )
    manifest_data = manifest.model_dump()
    manifest_data["work_assignments"]["pass2_replay_sequence_hash"] = hash_replay_sequence_ids(
        [1, 2, 3]
    )
    stale = DistributedRunManifest.model_validate(manifest_data)

    with pytest.raises(ValueError, match="hash does not match"):
        validate_pass2_replay_assignments(stale)


def test_pass2_worker_marker_records_sequence_metadata(tmp_path):
    manifest = assign_pass2_replay_sequences(
        _manifest(tmp_path),
        FakeTopCtx([1, 2, 4, 6]),
    )

    marker = build_worker_marker(
        manifest,
        1,
        phase="pass2",
        status="started",
    )

    assert marker.sequence_count == 2
    assert marker.sequence_id_min == 4
    assert marker.sequence_id_max == 6
    assert marker.replay_sequence_hash == hash_replay_sequence_ids([1, 2, 4, 6])
