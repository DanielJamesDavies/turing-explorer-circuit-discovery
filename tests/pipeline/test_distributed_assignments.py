from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline.distributed.assignments import (
    build_work_assignments,
    partition_contiguous,
    partition_pass1_shards,
    partition_seed_ids,
    partition_sequence_ids,
)
from pipeline.distributed.devices import build_device_assignments
from pipeline.distributed.manifest import DistributedRunManifest, generate_run_id
from pipeline.distributed.shard_table import ShardRecord


def _records(counts):
    records = []
    next_id = 1
    for shard_index, count in enumerate(counts):
        records.append(
            ShardRecord(
                shard_index=shard_index,
                shard_filename=f"shard_{shard_index}.npy",
                sequence_count=count,
                global_start_id=next_id,
                global_end_id=next_id + count,
                shard_size_bytes=1,
                shard_mtime_ns=1,
                index_filename=f".shard_indices/shard_{shard_index}.npy_sft1.idx.npy",
                index_size_bytes=1,
                index_mtime_ns=1,
            )
        )
        next_id += count
    return records


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
    }
    data.update(overrides)
    return DistributedRunManifest.model_validate(data)


def test_pass1_shards_are_balanced_by_sequence_count():
    assignments, totals = partition_pass1_shards(_records([10, 9, 5, 4, 1]), 3)

    assert assignments == {"0": [0], "1": [1, 4], "2": [2, 3]}
    assert totals == {"0": 10, "1": 10, "2": 9}


def test_pass1_balancing_uses_deterministic_tie_breaks():
    assignments, totals = partition_pass1_shards(_records([5, 5, 5, 5]), 2)

    assert assignments == {"0": [0, 2], "1": [1, 3]}
    assert totals == {"0": 10, "1": 10}


def test_pass1_handles_empty_inputs_one_worker_and_more_workers_than_shards():
    assert partition_pass1_shards([], 3) == (
        {"0": [], "1": [], "2": []},
        {"0": 0, "1": 0, "2": 0},
    )
    assert partition_pass1_shards(_records([2, 1]), 1) == (
        {"0": [0, 1]},
        {"0": 3},
    )
    assert partition_pass1_shards(_records([3, 1]), 4) == (
        {"0": [0], "1": [1], "2": [], "3": []},
        {"0": 3, "1": 1, "2": 0, "3": 0},
    )


def test_pass1_rejects_duplicate_and_out_of_range_selected_shards():
    table = _records([2, 1])

    with pytest.raises(ValueError, match="duplicate selected shard"):
        partition_pass1_shards(table, 2, selected_shards=[0, 0])

    with pytest.raises(ValueError, match="selected shard index out of range"):
        partition_pass1_shards(table, 2, selected_shards=[0, 99])


def test_contiguous_partitioner_never_drops_remainders():
    assert partition_contiguous(list(range(10)), 3) == {
        "0": [0, 1, 2, 3],
        "1": [4, 5, 6],
        "2": [7, 8, 9],
    }
    assert partition_contiguous([7, 8], 4) == {
        "0": [7],
        "1": [8],
        "2": [],
        "3": [],
    }


def test_sequence_partitioner_preserves_order_and_rejects_bad_ids():
    table = _records([2, 2])

    assert partition_sequence_ids([4, 1, 3, 2], 2, shard_table=table) == {
        "0": [4, 1],
        "1": [3, 2],
    }

    with pytest.raises(ValueError, match="duplicate sequence ID"):
        partition_sequence_ids([1, 2, 2], 2, shard_table=table)

    with pytest.raises(ValueError, match="sequence ID out of range"):
        partition_sequence_ids([1, 5], 2, shard_table=table)


def test_seed_partitioner_is_deterministic_and_rejects_duplicates():
    assert partition_seed_ids([10, 20, 30, 40, 50], 2) == {
        "0": [10, 20, 30],
        "1": [40, 50],
    }

    with pytest.raises(ValueError, match="duplicate seed ID"):
        partition_seed_ids([10, 10], 2)


def test_build_work_assignments_is_manifest_ready(tmp_path):
    table = _records([3, 2, 1])
    assignments = build_work_assignments(
        table,
        2,
        pass2_sequence_ids=[1, 2, 3, 4, 5, 6],
        discovery_seed_ids=[100, 200, 300],
    )
    manifest = _manifest(
        tmp_path,
        worker_count=2,
        shard_table=table,
        work_assignments=assignments.model_dump(),
    )

    assert isinstance(manifest, DistributedRunManifest)
    assert assignments.pass1_shards == {"0": [0], "1": [1, 2]}
    assert assignments.pass1_sequence_totals == {"0": 3, "1": 3}
    assert assignments.pass2_sequence_ids == {"0": [1, 2, 3], "1": [4, 5, 6]}
    assert assignments.discovery_seed_ids == {"0": [100, 200], "1": [300]}


def test_manifest_rejects_mismatched_pass1_sequence_totals(tmp_path):
    table = _records([3, 2])

    with pytest.raises(ValueError, match="pass1 sequence total does not match"):
        _manifest(
            tmp_path,
            worker_count=2,
            shard_table=table,
            work_assignments={
                "pass1_shards": {"0": [0], "1": [1]},
                "pass1_sequence_totals": {"0": 999, "1": 2},
                "pass2_sequence_ids": {},
                "discovery_seed_ids": {},
            },
        )


def test_device_assignment_validates_explicit_visible_cuda_ids():
    assignments = build_device_assignments(
        worker_count=2,
        physical_ids=[3, 1],
        visible_device_count=4,
    )

    assert [assignment.physical_id for assignment in assignments] == [3, 1]

    with pytest.raises(ValueError, match="physical device IDs not visible"):
        build_device_assignments(
            worker_count=2,
            physical_ids=[0, 4],
            visible_device_count=4,
        )

    with pytest.raises(ValueError, match="physical device IDs must be >= 0"):
        build_device_assignments(
            worker_count=1,
            physical_ids=[-1],
            visible_device_count=4,
        )
