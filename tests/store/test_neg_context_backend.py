from types import SimpleNamespace

import pytest
import torch

from store.neg_context import (
    NegCtxStats,
    TorchANNIndex,
    _process_component,
    parse_neg_ctx_devices,
    partition_components,
)


def test_parse_neg_ctx_devices_defaults_and_explicit_ids():
    assert parse_neg_ctx_devices([], cuda_count=3) == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:2"),
    ]
    assert parse_neg_ctx_devices([0, "1", "cuda:2", "cuda:2"], cuda_count=4) == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:2"),
    ]

    with pytest.raises(ValueError, match="Invalid neg_ctx device"):
        parse_neg_ctx_devices(["cpu"], cuda_count=1)


def test_partition_components_round_robin_and_complete():
    devices = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")]
    parts = partition_components(8, devices)

    assert parts == {
        "cuda:0": [0, 3, 6],
        "cuda:1": [1, 4, 7],
        "cuda:2": [2, 5],
    }
    assert sorted(comp for comps in parts.values() for comp in comps) == list(range(8))


def test_neg_ctx_stats_merge_adds_counts_and_timings():
    left = NegCtxStats(
        n_latents_attempted=1,
        n_latents_skipped_low_pos=2,
        n_latents_populated=3,
        n_latents_zero_negatives=4,
        fill_counts=[1, 2],
        t_query=0.5,
    )
    right = NegCtxStats(
        n_latents_attempted=10,
        n_latents_skipped_low_pos=20,
        n_latents_populated=30,
        n_latents_zero_negatives=40,
        fill_counts=[3],
        t_query=1.5,
    )

    left.merge_from(right)

    assert left.n_latents_attempted == 11
    assert left.n_latents_skipped_low_pos == 22
    assert left.n_latents_populated == 33
    assert left.n_latents_zero_negatives == 44
    assert left.fill_counts == [1, 2, 3]
    assert left.t_query == 2.0


def test_process_component_writes_expected_cpu_artifact_rows():
    top_ctx = SimpleNamespace(
        ctx_seq_idx=torch.tensor([[[1, 2], [3, 0]]], dtype=torch.int32),
        ctx_seq_val=torch.tensor([[[1.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
    )
    mid_ctx = SimpleNamespace(
        ctx_seq_idx=torch.zeros((1, 2, 2), dtype=torch.int32),
        ctx_seq_val=torch.zeros((1, 2, 2), dtype=torch.float32),
    )
    neg_ctx = SimpleNamespace(
        ctx_seq_idx=torch.full((1, 2, 2), 999, dtype=torch.int32),
        ctx_seq_val=torch.full((1, 2, 2), 999.0, dtype=torch.float32),
    )
    index = TorchANNIndex(
        torch.tensor(
            [
                [1.0, 0.2],   # seq 1, positive for latent 0
                [1.0, 0.1],   # seq 2, positive for latent 0
                [0.0, 1.0],   # seq 3, positive for latent 1
                [-1.0, -1.0], # seq 4
            ],
            dtype=torch.float32,
        ),
        device=torch.device("cpu"),
    )
    stats = NegCtxStats()

    _process_component(
        0,
        top_ctx,
        mid_ctx,
        neg_ctx,
        index,
        K=4,
        n_neg=2,
        min_pos_ctx=1,
        stats=stats,
        total_n_seqs=4,
        slot_to_id_d=None,
        id_to_slot_d=None,
    )

    assert neg_ctx.ctx_seq_idx[0, 0].tolist() == [3, 4]
    assert neg_ctx.ctx_seq_idx[0, 1].tolist() == [1, 2]
    assert neg_ctx.ctx_seq_val[0, 0, 0].item() >= neg_ctx.ctx_seq_val[0, 0, 1].item()
    assert stats.n_latents_attempted == 2
    assert stats.n_latents_populated == 2
    assert stats.fill_counts == [2, 2]
