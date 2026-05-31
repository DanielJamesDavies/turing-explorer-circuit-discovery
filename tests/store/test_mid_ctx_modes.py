import torch

from config import MidCtxConfig, config
from store.context import Context


def _tiny_mid_ctx(monkeypatch, mode: str, *, sampling_seed: int = 0) -> Context:
    monkeypatch.setattr(config.latents.mid_ctx, "mode", mode)
    monkeypatch.setattr(config.latents.mid_ctx, "n_sequences", 2)
    monkeypatch.setattr(config.latents.mid_ctx, "band_low_sigma", 0.5)
    monkeypatch.setattr(config.latents.mid_ctx, "band_high_sigma", 1.5)
    monkeypatch.setattr(config.distributed, "sampling_seed", sampling_seed)

    ctx = Context("mid")
    ctx.num_components = 1
    ctx.d_sae = 3
    ctx.num_ctx_sequences = 2
    return ctx


def test_mid_ctx_config_accepts_gpu_priority_reservoir():
    assert MidCtxConfig(mode="gpu_priority_reservoir").mode == "gpu_priority_reservoir"


def test_gpu_topk_mid_selects_sequences_closest_to_band_midpoint(monkeypatch):
    ctx = _tiny_mid_ctx(monkeypatch, "gpu_topk_mid")
    sequence_ids = torch.tensor([101, 102, 103, 104], dtype=torch.int32)

    # Scores are mean activation over seq_len=1. The band is (0.5, 1.5),
    # midpoint 1.0. For latent 0, seq 102 (1.0) and seq 103 (1.2) are closest.
    top_acts = torch.tensor([[[0.6]], [[1.0]], [[1.2]], [[1.4]]], dtype=torch.float32)
    top_indices = torch.zeros((4, 1, 1), dtype=torch.long)
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.ones(3, dtype=torch.float32)

    ctx.update_component(0, sequence_ids, (top_acts, top_indices), mean, std)

    assert ctx.ctx_seq_idx[0, 0].tolist() == [102, 103]
    assert torch.allclose(ctx.ctx_seq_val[0, 0], torch.tensor([1.0, 1.2]))
    assert ctx.ctx_seq_idx[0, 1].tolist() == [0, 0]
    assert ctx.reservoir_fill[0, 0].item() == 2
    assert ctx.reservoir_n[0, 0].item() == 4


def test_gpu_priority_reservoir_selects_lowest_stable_priorities(monkeypatch):
    ctx = _tiny_mid_ctx(monkeypatch, "gpu_priority_reservoir", sampling_seed=123)
    sequence_ids = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
    top_acts = torch.tensor([[[0.6]], [[1.0]], [[1.2]], [[1.4]]], dtype=torch.float32)
    top_indices = torch.zeros((4, 1, 1), dtype=torch.long)
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.ones(3, dtype=torch.float32)

    expected_priorities = ctx._mid_priority_values(0, sequence_ids, torch.device("cpu"))[0]
    expected_order = torch.argsort(expected_priorities, stable=True)[:2]
    expected_ids = sequence_ids[expected_order].tolist()
    expected_vals = top_acts.flatten()[expected_order].float()

    ctx.update_component(0, sequence_ids, (top_acts, top_indices), mean, std)

    assert ctx.ctx_seq_idx[0, 0].tolist() == expected_ids
    assert torch.allclose(ctx.ctx_seq_val[0, 0], expected_vals)
    assert ctx.ctx_seq_idx[0, 1].tolist() == [0, 0]
    assert ctx.reservoir_fill[0, 0].item() == 2
    assert ctx.reservoir_n[0, 0].item() == 4


def test_gpu_priority_reservoir_is_batch_order_invariant(monkeypatch):
    sequence_ids = torch.tensor([201, 202, 203, 204], dtype=torch.int32)
    top_acts = torch.tensor([[[0.75]], [[1.35]], [[1.0]], [[1.45]]], dtype=torch.float32)
    top_indices = torch.zeros((4, 1, 1), dtype=torch.long)
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.ones(3, dtype=torch.float32)

    combined = _tiny_mid_ctx(monkeypatch, "gpu_priority_reservoir", sampling_seed=456)
    combined.update_component(0, sequence_ids, (top_acts, top_indices), mean, std)

    split = _tiny_mid_ctx(monkeypatch, "gpu_priority_reservoir", sampling_seed=456)
    split.update_component(0, sequence_ids[2:], (top_acts[2:], top_indices[2:]), mean, std)
    split.update_component(0, sequence_ids[:2], (top_acts[:2], top_indices[:2]), mean, std)

    assert torch.equal(combined.ctx_seq_idx, split.ctx_seq_idx)
    assert torch.equal(combined.ctx_seq_val, split.ctx_seq_val)
    assert torch.equal(combined.reservoir_fill, split.reservoir_fill)
    assert torch.equal(combined.reservoir_n, split.reservoir_n)


def test_gpu_priority_reservoir_is_multi_update_order_invariant(monkeypatch):
    sequence_ids = torch.tensor([301, 302, 303, 304, 305, 306], dtype=torch.int32)
    top_acts = torch.tensor([[[0.7]], [[0.9]], [[1.1]], [[1.3]], [[0.8]], [[1.4]]], dtype=torch.float32)
    top_indices = torch.zeros((6, 1, 1), dtype=torch.long)
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.ones(3, dtype=torch.float32)

    combined = _tiny_mid_ctx(monkeypatch, "gpu_priority_reservoir", sampling_seed=789)
    combined.update_component(0, sequence_ids, (top_acts, top_indices), mean, std)

    chunked = _tiny_mid_ctx(monkeypatch, "gpu_priority_reservoir", sampling_seed=789)
    for indices in ([4, 5], [0, 1], [2, 3]):
        idx = torch.tensor(indices, dtype=torch.long)
        chunked.update_component(0, sequence_ids[idx], (top_acts[idx], top_indices[idx]), mean, std)

    assert torch.equal(combined.ctx_seq_idx, chunked.ctx_seq_idx)
    assert torch.equal(combined.ctx_seq_val, chunked.ctx_seq_val)
    assert torch.equal(combined.reservoir_fill, chunked.reservoir_fill)
    assert torch.equal(combined.reservoir_n, chunked.reservoir_n)


def test_gpu_priority_reservoir_inclusion_frequency_matches_uniform_expectation(monkeypatch):
    sequence_ids = torch.arange(401, 409, dtype=torch.int32)
    top_acts = torch.ones((8, 1, 1), dtype=torch.float32)
    top_indices = torch.zeros((8, 1, 1), dtype=torch.long)
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.ones(3, dtype=torch.float32)
    seed_count = 512
    selected_per_seed = 2
    expected = seed_count * selected_per_seed / float(sequence_ids.numel())
    max_abs_deviation = expected * 0.375
    counts = torch.zeros(sequence_ids.numel(), dtype=torch.int32)

    for seed in range(seed_count):
        ctx = _tiny_mid_ctx(monkeypatch, "gpu_priority_reservoir", sampling_seed=seed)
        ctx.update_component(0, sequence_ids, (top_acts, top_indices), mean, std)
        selected = ctx.ctx_seq_idx[0, 0]
        for sequence_id in selected.tolist():
            position = int((sequence_ids == sequence_id).nonzero()[0].item())
            counts[position] += 1

    deviations = (counts.float() - expected).abs()
    assert int(counts.sum().item()) == seed_count * selected_per_seed
    assert bool((deviations <= max_abs_deviation).all()), counts.tolist()


def test_gpu_priority_reservoir_handles_large_component_ids(monkeypatch):
    ctx = _tiny_mid_ctx(monkeypatch, "gpu_priority_reservoir", sampling_seed=321)
    sequence_ids = torch.tensor([501, 502, 503, 504], dtype=torch.int32)
    priorities = ctx._mid_priority_values(35, sequence_ids, torch.device("cpu"))

    assert priorities.shape == (3, 4)
    assert priorities.dtype == torch.int64
    assert bool((priorities >= 0).all())


def test_gpu_priority_sparse_candidate_priorities_match_dense(monkeypatch):
    ctx = _tiny_mid_ctx(monkeypatch, "gpu_priority_reservoir", sampling_seed=654)
    sequence_ids = torch.tensor([601, 602, 603, 604], dtype=torch.int32)
    latent_indices = torch.tensor([0, 2, 1], dtype=torch.long)
    batch_indices = torch.tensor([3, 1, 2], dtype=torch.long)

    dense = ctx._mid_priority_values(17, sequence_ids, torch.device("cpu"))
    sparse = ctx._mid_priority_values_for_candidates(
        17,
        latent_indices,
        sequence_ids[batch_indices],
        torch.device("cpu"),
    )

    assert torch.equal(sparse, dense[latent_indices, batch_indices])


def test_gpu_priority_reservoir_differs_from_gpu_topk_midpoint_selection(monkeypatch):
    sequence_ids = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
    top_acts = torch.tensor([[[0.6]], [[1.0]], [[1.2]], [[1.4]]], dtype=torch.float32)
    top_indices = torch.zeros((4, 1, 1), dtype=torch.long)
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.ones(3, dtype=torch.float32)

    topk = _tiny_mid_ctx(monkeypatch, "gpu_topk_mid")
    topk.update_component(0, sequence_ids, (top_acts, top_indices), mean, std)

    priority = _tiny_mid_ctx(monkeypatch, "gpu_priority_reservoir", sampling_seed=0)
    priority.update_component(0, sequence_ids, (top_acts, top_indices), mean, std)

    assert topk.ctx_seq_idx[0, 0].tolist() == [102, 103]
    assert set(priority.ctx_seq_idx[0, 0].tolist()) == {101, 104}


def test_gpu_topk_mid_merges_existing_and_new_candidates(monkeypatch):
    ctx = _tiny_mid_ctx(monkeypatch, "gpu_topk_mid")
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.ones(3, dtype=torch.float32)

    first_ids = torch.tensor([201, 202], dtype=torch.int32)
    first_acts = torch.tensor([[[0.75]], [[1.35]]], dtype=torch.float32)
    first_indices = torch.zeros((2, 1, 1), dtype=torch.long)
    ctx.update_component(0, first_ids, (first_acts, first_indices), mean, std)

    second_ids = torch.tensor([203, 204], dtype=torch.int32)
    second_acts = torch.tensor([[[1.0]], [[1.45]]], dtype=torch.float32)
    second_indices = torch.zeros((2, 1, 1), dtype=torch.long)
    ctx.update_component(0, second_ids, (second_acts, second_indices), mean, std)

    assert ctx.ctx_seq_idx[0, 0].tolist() == [203, 201]
    assert torch.allclose(ctx.ctx_seq_val[0, 0], torch.tensor([1.0, 0.75]))
    assert ctx.reservoir_n[0, 0].item() == 4


def test_mid_ctx_modes_allocate_same_shapes_and_save_metadata(monkeypatch, tmp_path):
    for mode in ("reservoir_cpu", "gpu_topk_mid", "gpu_priority_reservoir"):
        ctx = _tiny_mid_ctx(monkeypatch, mode)
        ctx.allocate()

        assert ctx.ctx_seq_idx.shape == (1, 3, 2)
        assert ctx.ctx_seq_val.shape == (1, 3, 2)
        assert ctx.reservoir_fill.shape == (1, 3)
        assert ctx.reservoir_n.shape == (1, 3)

        path = tmp_path / f"{mode}.pt"
        ctx.save(str(path))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["ctx_seq_idx"].shape == (1, 3, 2)
        assert checkpoint["ctx_seq_val"].shape == (1, 3, 2)
        assert checkpoint["ctx_type"] == "mid"
        assert checkpoint["mode"] == mode
        assert checkpoint["band_low_sigma"] == 0.5
        assert checkpoint["band_high_sigma"] == 1.5
        assert "_priority_val" not in checkpoint
