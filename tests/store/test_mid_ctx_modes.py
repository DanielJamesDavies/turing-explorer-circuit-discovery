import torch

from config import DistributedMidCtxMergeConfig, MidCtxConfig, config
from store.context import Context


def _tiny_mid_ctx(monkeypatch) -> Context:
    mode = "reservoir_cpu"
    monkeypatch.setattr(config.latents.mid_ctx, "mode", mode)
    monkeypatch.setattr(config.latents.mid_ctx, "n_sequences", 2)
    monkeypatch.setattr(config.latents.mid_ctx, "band_low_sigma", 0.5)
    monkeypatch.setattr(config.latents.mid_ctx, "band_high_sigma", 1.5)

    ctx = Context("mid")
    ctx.num_components = 1
    ctx.d_sae = 3
    ctx.num_ctx_sequences = 2
    return ctx


def test_mid_ctx_config_only_accepts_reservoir_cpu():
    assert MidCtxConfig(mode="reservoir_cpu").mode == "reservoir_cpu"
    for mode in ("gpu_topk_mid", "gpu_priority_reservoir"):
        try:
            MidCtxConfig(mode=mode)
        except ValueError as exc:
            assert "reservoir_cpu" in str(exc)
        else:
            raise AssertionError(f"{mode} should not be accepted")


def test_distributed_mid_ctx_merge_config_accepts_weighted_and_candidate_pool():
    assert DistributedMidCtxMergeConfig().mode == "weighted_reservoir"
    assert DistributedMidCtxMergeConfig(mode="weighted_reservoir").mode == "weighted_reservoir"
    assert DistributedMidCtxMergeConfig(mode="candidate_pool").mode == "candidate_pool"
    try:
        DistributedMidCtxMergeConfig(mode="gpu_priority_reservoir")
    except ValueError as exc:
        assert "weighted_reservoir" in str(exc)
        assert "candidate_pool" in str(exc)
    else:
        raise AssertionError("unsupported mid_ctx merge mode should not be accepted")


def test_mid_ctx_modes_allocate_same_shapes_and_save_metadata(monkeypatch, tmp_path):
    ctx = _tiny_mid_ctx(monkeypatch)
    ctx.allocate()

    assert ctx.ctx_seq_idx.shape == (1, 3, 2)
    assert ctx.ctx_seq_val.shape == (1, 3, 2)
    assert ctx.reservoir_fill.shape == (1, 3)
    assert ctx.reservoir_n.shape == (1, 3)

    path = tmp_path / "reservoir_cpu.pt"
    ctx.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["ctx_seq_idx"].shape == (1, 3, 2)
    assert checkpoint["ctx_seq_val"].shape == (1, 3, 2)
    assert checkpoint["ctx_type"] == "mid"
    assert checkpoint["mode"] == "reservoir_cpu"
    assert checkpoint["band_low_sigma"] == 0.5
    assert checkpoint["band_high_sigma"] == 1.5
    assert "_priority_val" not in checkpoint
