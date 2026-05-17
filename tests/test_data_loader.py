import math

import numpy as np
import pytest
import torch

from config import config
from data.loader import DataLoader


def _write_shards(tmp_path, shards):
    for shard_idx, values in enumerate(shards):
        np.save(tmp_path / f"shard_{shard_idx}.npy", np.asarray(values, dtype=np.int64))


def _reference_sequences(values, skip_first_token):
    raw = np.asarray(values, dtype=np.int64)
    if len(raw) == 0:
        return []

    split_at = np.where(raw == -1)[0]
    sequences = []
    for chunk in np.split(raw, split_at):
        seq = chunk[chunk != -1]
        if skip_first_token:
            seq = seq[1:]
        if len(seq) > 1:
            sequences.append(seq)
    return sequences


@pytest.fixture
def synthetic_shards(tmp_path, monkeypatch):
    shards = [
        [10, 11, 12, -1, 20, 21, 22, 23, -1, 30, 31],
        [],
        [40, 41, 42, 43, -1, 50, 51, 52],
    ]
    _write_shards(tmp_path, shards)

    monkeypatch.setattr(config.data, "dataset_path", str(tmp_path))
    monkeypatch.setattr(config.data, "n_shards", len(shards))
    monkeypatch.setattr(config.data, "batch_size", 3)
    return shards


def _collect_padded_batches(loader, max_length=5):
    batches = list(loader.get_batches(pad_to_max=True, max_length=max_length))
    ids = torch.cat([batch_ids.cpu() for batch_ids, _tokens in batches], dim=0)
    tokens = torch.cat([tokens.cpu() for _batch_ids, tokens in batches], dim=0)
    return ids, tokens


def _padded(sequences, max_length):
    out = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        out[idx, :length] = torch.from_numpy(seq[:length].astype(np.int64))
    return out


def test_get_batches_uses_indexed_mmap_and_preserves_ids_remainders(
    synthetic_shards, monkeypatch
):
    loader = DataLoader(torch.device("cpu"), skip_first_token=True)

    def fail_load_shard(_shard_index):
        raise AssertionError("get_batches should not materialize full shard lists")

    monkeypatch.setattr(loader, "_load_shard", fail_load_shard)

    expected_sequences = [
        seq
        for shard in synthetic_shards
        for seq in _reference_sequences(shard, skip_first_token=True)
    ]
    ids, tokens = _collect_padded_batches(loader, max_length=5)

    assert loader.shard_id_ranges == [(1, 2), (-1, -1), (3, 4)]
    assert len(loader) == math.ceil(len(expected_sequences) / config.data.batch_size)
    assert ids.tolist() == [1, 2, 3, 4]
    assert torch.equal(tokens, _padded(expected_sequences, max_length=5))


def test_skip_first_false_public_load_shard_matches_reference(synthetic_shards):
    loader = DataLoader(torch.device("cpu"), skip_first_token=False)

    expected_shard_0 = _reference_sequences(synthetic_shards[0], skip_first_token=False)
    actual_shard_0 = loader.load_shard(0)

    assert [seq.tolist() for seq in actual_shard_0] == [
        seq.tolist() for seq in expected_shard_0
    ]
    assert loader.shard_id_ranges == [(1, 3), (-1, -1), (4, 5)]

    ids, tokens = _collect_padded_batches(loader, max_length=5)
    expected_sequences = [
        seq
        for shard in synthetic_shards
        for seq in _reference_sequences(shard, skip_first_token=False)
    ]
    assert ids.tolist() == [1, 2, 3, 4, 5]
    assert torch.equal(tokens, _padded(expected_sequences, max_length=5))


def test_get_sequence_uses_global_ids_across_empty_shards(synthetic_shards):
    loader = DataLoader(torch.device("cpu"), skip_first_token=True)

    assert loader.get_sequence(1).tolist() == [11, 12]
    assert loader.get_sequence(3).tolist() == [41, 42, 43]
    assert loader.get_sequence(4).tolist() == [51, 52]

    with pytest.raises(IndexError, match="Sequence ID 999 not found"):
        loader.get_sequence(999)


def test_get_batches_by_ids_preserves_requested_order_and_skips_missing_ids(
    synthetic_shards,
):
    loader = DataLoader(torch.device("cpu"), skip_first_token=True)

    batches = list(loader.get_batches_by_ids([4, 1, 999, 3], max_length=5))
    ids = torch.cat([batch_ids.cpu() for batch_ids, _tokens in batches], dim=0)
    tokens = torch.cat([tokens.cpu() for _batch_ids, tokens in batches], dim=0)

    assert ids.tolist() == [4, 1, 3]
    assert torch.equal(
        tokens,
        _padded(
            [
                np.asarray([51, 52], dtype=np.int64),
                np.asarray([11, 12], dtype=np.int64),
                np.asarray([41, 42, 43], dtype=np.int64),
            ],
            max_length=5,
        ),
    )
