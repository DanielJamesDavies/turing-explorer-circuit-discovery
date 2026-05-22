import json
from types import SimpleNamespace

import pytest
import torch

from store.neg_context import (
    NegCtxStats,
    ShardedANNIndex,
    TorchANNIndex,
    _ann_device,
    _validate_cuda_devices,
    check_neg_ctx_memory_guardrail,
    estimate_neg_ctx_ann_memory,
    merge_shard_search_results,
    _process_component,
    parse_neg_ctx_devices,
    partition_index_slots,
    partition_components,
    validate_neg_ctx_output,
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


def test_validate_cuda_devices_rejects_unavailable_or_out_of_range(monkeypatch):
    monkeypatch.setattr("store.neg_context.torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        _validate_cuda_devices([torch.device("cuda:0")])

    monkeypatch.setattr("store.neg_context.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("store.neg_context.torch.cuda.device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="outside visible range"):
        _validate_cuda_devices([torch.device("cuda:1")])


def test_partition_components_round_robin_and_complete():
    devices = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")]
    parts = partition_components(8, devices)

    assert parts == {
        "cuda:0": [0, 3, 6],
        "cuda:1": [1, 4, 7],
        "cuda:2": [2, 5],
    }
    assert sorted(comp for comps in parts.values() for comp in comps) == list(range(8))


def test_partition_index_slots_splits_rows_contiguously():
    parts = partition_index_slots(
        10,
        [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")],
    )

    assert parts == {
        "cuda:0": (0, 4),
        "cuda:1": (4, 7),
        "cuda:2": (7, 10),
    }


def test_merge_shard_search_results_matches_global_topk():
    shard_a_sims = torch.tensor([[0.9, 0.2], [0.6, 0.3]], dtype=torch.float32)
    shard_a_idxs = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    shard_b_sims = torch.tensor([[0.8, 0.7], [0.95, 0.1]], dtype=torch.float32)
    shard_b_idxs = torch.tensor([[2, 3], [2, 3]], dtype=torch.int64)

    sims, idxs = merge_shard_search_results(
        [(shard_a_sims, shard_a_idxs), (shard_b_sims, shard_b_idxs)],
        k=3,
        device=torch.device("cpu"),
    )

    assert torch.allclose(
        sims,
        torch.tensor([[0.9, 0.8, 0.7], [0.95, 0.6, 0.3]], dtype=torch.float32),
    )
    assert idxs.tolist() == [[0, 2, 3], [2, 0, 1]]


def test_sharded_ann_single_cpu_shard_matches_replicated_index():
    vecs = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    replicated = TorchANNIndex(vecs, torch.device("cpu"))
    sharded = ShardedANNIndex(vecs, [torch.device("cpu")])

    rep_sims, rep_idxs = replicated.search(queries, k=3)
    shard_sims, shard_idxs = sharded.search(queries, k=3, merge_device=torch.device("cpu"))

    assert torch.allclose(shard_sims, rep_sims)
    assert torch.equal(shard_idxs, rep_idxs)


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


def test_neg_ctx_stats_records_seq_repr_metadata_and_save_payload(tmp_path):
    stats = NegCtxStats(
        fill_counts=[1, 2],
        backend="single_gpu_exact",
        ann_device="cpu",
        devices=["cpu"],
    )
    seq_repr = SimpleNamespace(
        n_seqs=10,
        n_stored=4,
        repr_dim=3,
        is_capped=True,
    )

    stats.record_seq_repr(seq_repr)
    output_path = tmp_path / "neg_ctx_stats.json"
    stats.save(str(output_path))

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "single_gpu_exact"
    assert payload["ann_device"] == "cpu"
    assert payload["seq_repr_n_seqs"] == 10
    assert payload["seq_repr_n_stored"] == 4
    assert payload["seq_repr_cap_percent"] == 40.0
    assert payload["fill_counts"] == [1, 2]


def test_estimate_neg_ctx_ann_memory_includes_index_mappings_and_query_working():
    seq_repr = SimpleNamespace(
        n_seqs=10,
        n_stored=4,
        repr_dim=3,
        is_capped=True,
    )

    estimate = estimate_neg_ctx_ann_memory(seq_repr, query_chunk_size=2)

    assert estimate["index_bytes"] == 4 * 3 * 4
    assert estimate["slot_to_id_bytes"] == 5 * 8
    assert estimate["id_to_slot_bytes"] == 11 * 4
    assert estimate["query_working_bytes"] == 2 * 4 * 4
    assert estimate["total_bytes"] == 48 + 40 + 44 + 32


def test_memory_guardrail_skips_cpu_and_checks_cuda_fraction():
    estimate = {"total_bytes": 900, "index_bytes": 100, "query_working_bytes": 800}

    cpu_result = check_neg_ctx_memory_guardrail(
        torch.device("cpu"),
        estimate,
        fraction=0.5,
        fail_on_exceed=True,
    )
    assert cpu_result["checked"] is False

    ok_result = check_neg_ctx_memory_guardrail(
        torch.device("cuda:0"),
        estimate,
        fraction=0.9,
        fail_on_exceed=True,
        total_vram_bytes=1000,
    )
    assert ok_result["exceeds_limit"] is False

    with pytest.raises(RuntimeError, match="ANN memory estimate"):
        check_neg_ctx_memory_guardrail(
            torch.device("cuda:0"),
            estimate,
            fraction=0.5,
            fail_on_exceed=True,
            total_vram_bytes=1000,
        )


def test_memory_guardrail_can_warn_without_failing(capsys):
    result = check_neg_ctx_memory_guardrail(
        torch.device("cuda:0"),
        {"total_bytes": 900, "index_bytes": 100, "query_working_bytes": 800},
        fraction=0.5,
        fail_on_exceed=False,
        total_vram_bytes=1000,
    )

    assert result["exceeds_limit"] is True
    assert "WARNING" in capsys.readouterr().out


def test_neg_ctx_stats_saves_component_assignments_and_per_device_timings(tmp_path):
    stats = NegCtxStats(
        backend="multi_gpu_exact",
        devices=["cuda:0", "cuda:1"],
        component_assignments={"cuda:0": [0, 2], "cuda:1": [1, 3]},
        per_device_timing_ms={"cuda:0": {"query_ms": 1.5}},
    )
    output_path = tmp_path / "neg_ctx_stats.json"

    stats.save(str(output_path))

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["component_assignments"] == {"cuda:0": [0, 2], "cuda:1": [1, 3]}
    assert payload["per_device_timing_ms"] == {"cuda:0": {"query_ms": 1.5}}


def test_ann_device_supports_cpu_and_auto_cpu_fallback(monkeypatch):
    monkeypatch.setattr("store.neg_context.config.hardware.ann_device", "cpu")
    assert _ann_device() == torch.device("cpu")

    monkeypatch.setattr("store.neg_context.config.hardware.ann_device", "auto")
    monkeypatch.setattr("store.neg_context.torch.cuda.is_available", lambda: False)
    assert _ann_device() == torch.device("cpu")


def test_ann_device_supports_cuda_aliases_when_available(monkeypatch):
    monkeypatch.setattr("store.neg_context.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("store.neg_context.torch.cuda.device_count", lambda: 2)

    monkeypatch.setattr("store.neg_context.config.hardware.ann_device", "gpu")
    assert _ann_device() == torch.device("cuda")

    monkeypatch.setattr("store.neg_context.config.hardware.ann_device", "cuda")
    assert _ann_device() == torch.device("cuda")

    monkeypatch.setattr("store.neg_context.config.hardware.ann_device", "cuda:1")
    assert _ann_device() == torch.device("cuda:1")

    monkeypatch.setattr("store.neg_context.config.hardware.ann_device", "cuda:2")
    with pytest.raises(RuntimeError, match="outside visible CUDA range"):
        _ann_device()


def test_ann_device_rejects_unknown_value(monkeypatch):
    monkeypatch.setattr("store.neg_context.config.hardware.ann_device", "tpu")

    with pytest.raises(ValueError, match="hardware.ann_device"):
        _ann_device()


def test_torch_ann_index_cpu_search_returns_sorted_neighbors():
    index = TorchANNIndex(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        device=torch.device("cpu"),
    )

    similarities, indices = index.search(torch.tensor([[1.0, 0.0]]), k=2)

    assert indices.tolist() == [[0, 1]]
    assert similarities[0, 0].item() >= similarities[0, 1].item()


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


def test_validate_neg_ctx_output_accepts_valid_rows_and_rejects_bad_values():
    neg_ctx = SimpleNamespace(
        ctx_seq_idx=torch.tensor([[[1, 2], [0, 0]]], dtype=torch.int32),
        ctx_seq_val=torch.tensor([[[0.9, 0.8], [0.0, 0.0]]], dtype=torch.float32),
    )
    validate_neg_ctx_output(neg_ctx, total_n_seqs=2, n_sequences=2)

    neg_ctx.ctx_seq_idx[0, 0, 0] = 3
    with pytest.raises(ValueError, match="exceeds seq_repr"):
        validate_neg_ctx_output(neg_ctx, total_n_seqs=2, n_sequences=2)


def test_validate_neg_ctx_output_rejects_non_finite_or_negative_similarity():
    neg_ctx = SimpleNamespace(
        ctx_seq_idx=torch.tensor([[[1]]], dtype=torch.int32),
        ctx_seq_val=torch.tensor([[[-0.1]]], dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="non-negative"):
        validate_neg_ctx_output(neg_ctx, total_n_seqs=1, n_sequences=1)

    neg_ctx.ctx_seq_val[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_neg_ctx_output(neg_ctx, total_n_seqs=1, n_sequences=1)
