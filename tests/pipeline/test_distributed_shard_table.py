import os
import time

import numpy as np
import pytest
import torch

from config import config
from data.loader import DataLoader
from pipeline.distributed.shard_table import (
    ShardRecord,
    build_shard_table,
    sequence_ids_for_shards,
    validate_shard_records,
    validate_shard_table,
)


def _write_shards(tmp_path, shards):
    for shard_idx, values in enumerate(shards):
        np.save(tmp_path / f"shard_{shard_idx}.npy", np.asarray(values, dtype=np.int64))


@pytest.fixture
def uneven_shards(tmp_path, monkeypatch):
    shards = [
        [10, 11, 12, -1, 20, 21, 22, 23, -1, 30, 31, 32],
        [40, 41, 42, -1, 50, 51, 52],
        [60, 61, 62, -1],
    ]
    _write_shards(tmp_path, shards)
    monkeypatch.setattr(config.data, "dataset_path", str(tmp_path))
    monkeypatch.setattr(config.data, "n_shards", len(shards))
    monkeypatch.setattr(config.data, "batch_size", 2)
    return shards


def test_build_shard_table_uses_prefix_sum_and_allows_short_final_shard(
    uneven_shards,
    tmp_path,
):
    table = build_shard_table(tmp_path, n_shards=len(uneven_shards))

    assert [
        (
            record.shard_index,
            record.shard_filename,
            record.sequence_count,
            record.global_start_id,
            record.global_end_id,
        )
        for record in table
    ] == [
        (0, "shard_0.npy", 3, 1, 4),
        (1, "shard_1.npy", 2, 4, 6),
        (2, "shard_2.npy", 1, 6, 7),
    ]


def test_worker_assigned_shards_emit_single_process_global_ids(
    uneven_shards,
    tmp_path,
):
    table = build_shard_table(tmp_path, n_shards=len(uneven_shards))
    loader = DataLoader(torch.device("cpu"), skip_first_token=True)

    assert sequence_ids_for_shards(table, [0, 2]) == [1, 2, 3, 6]
    assert sequence_ids_for_shards(table, [1]) == [4, 5]

    all_worker_ids = sequence_ids_for_shards(table, [0, 2]) + sequence_ids_for_shards(
        table, [1]
    )
    single_process_ids = [
        sequence_id
        for start_id, end_id in loader.shard_id_ranges
        if start_id != -1
        for sequence_id in range(start_id, end_id + 1)
    ]
    assert sorted(all_worker_ids) == single_process_ids


def test_validate_shard_table_rejects_missing_shards(uneven_shards, tmp_path):
    table = build_shard_table(tmp_path, n_shards=len(uneven_shards))
    os.remove(tmp_path / "shard_1.npy")

    with pytest.raises(ValueError, match="shard files or shard order differ"):
        validate_shard_table(tmp_path, table)


def test_validate_shard_table_rejects_reordered_shards(uneven_shards, tmp_path):
    table = build_shard_table(tmp_path, n_shards=len(uneven_shards))
    reordered = [
        table[0],
        table[2].model_copy(update={"shard_index": 1}),
        table[1].model_copy(update={"shard_index": 2}),
    ]

    with pytest.raises(ValueError, match="shard files or shard order differ"):
        validate_shard_table(tmp_path, reordered)


def test_validate_shard_table_rejects_stale_index(uneven_shards, tmp_path):
    table = build_shard_table(tmp_path, n_shards=len(uneven_shards))
    shard_path = tmp_path / "shard_0.npy"

    time.sleep(0.01)
    now_ns = time.time_ns()
    os.utime(shard_path, ns=(now_ns, now_ns))

    with pytest.raises(ValueError, match="stale shard index"):
        validate_shard_table(tmp_path, table)


def test_validate_shard_table_rejects_changed_sequence_count(uneven_shards, tmp_path):
    table = build_shard_table(tmp_path, n_shards=len(uneven_shards))
    index_path = tmp_path / table[0].index_filename
    np.save(index_path, np.zeros((1, 2), dtype=np.int64))
    stat = index_path.stat()
    table[0] = table[0].model_copy(
        update={"index_size_bytes": stat.st_size, "index_mtime_ns": stat.st_mtime_ns}
    )

    with pytest.raises(ValueError, match="sequence count changed"):
        validate_shard_table(tmp_path, table)


def test_validate_shard_records_rejects_duplicate_sequence_ranges(tmp_path):
    table = build_shard_table(tmp_path, n_shards=0)
    base = ShardRecord(
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
    )
    duplicated = base.model_copy(
        update={"shard_index": 1, "global_start_id": 2, "global_end_id": 4}
    )

    assert table == []
    with pytest.raises(ValueError, match="global sequence IDs must be contiguous"):
        validate_shard_records([base, duplicated])


def test_sequence_ids_for_shards_rejects_out_of_range_assigned_shard(
    uneven_shards,
    tmp_path,
):
    table = build_shard_table(tmp_path, n_shards=len(uneven_shards))

    with pytest.raises(ValueError, match="assigned shard index out of range"):
        sequence_ids_for_shards(table, [999])
