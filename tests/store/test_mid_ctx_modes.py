import torch

from config import config
from store.context import Context


def _tiny_mid_ctx(monkeypatch, mode: str) -> Context:
    monkeypatch.setattr(config.latents.mid_ctx, "mode", mode)
    monkeypatch.setattr(config.latents.mid_ctx, "n_sequences", 2)
    monkeypatch.setattr(config.latents.mid_ctx, "band_low_sigma", 0.5)
    monkeypatch.setattr(config.latents.mid_ctx, "band_high_sigma", 1.5)

    ctx = Context("mid")
    ctx.num_components = 1
    ctx.d_sae = 3
    ctx.num_ctx_sequences = 2
    return ctx


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
    for mode in ("reservoir_cpu", "gpu_topk_mid"):
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
