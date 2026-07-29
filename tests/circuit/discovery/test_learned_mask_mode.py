"""Mode wiring for the learned-mask family (config + dispatch).

Engine correctness lives in tests/circuit/instrument/test_learned_mask.py;
here we pin the mode surface: validator membership (mask is abl-only, the
_negctx-suffixed and contrast modes are cf-only, per the ig_negctx
precedent), the hop contracts (supports-only for mask/mask_contrast,
inhibitors-only with negative scores for mask_negctx), and the loud
position_aware failure.
"""
import pytest
import torch
from unittest.mock import MagicMock

from circuit.types.feature_id import FeatureID
from config import (AblationGradientConfig, CounterfactualGradientConfig,
                    LearnedMaskConfig, config)
from tests.conftest import MockSAEBank


class TestConfig:
    def test_abl_accepts_mask_and_rejects_cf_only_modes(self):
        assert AblationGradientConfig(attribution_mode="mask").attribution_mode == "mask"
        for cf_only in ("mask_contrast", "mask_negctx", "ig_negctx"):
            with pytest.raises(ValueError, match="attribution_mode"):
                AblationGradientConfig(attribution_mode=cf_only)

    def test_cf_accepts_mask_variants_and_rejects_bare_mask(self):
        for ok in ("mask_contrast", "mask_negctx"):
            assert CounterfactualGradientConfig(attribution_mode=ok).attribution_mode == ok
        # bare "mask" is abl-only: hosting it under cf would recreate the
        # cf-ig_mean == abl-ig_mean duplication under a new name.
        with pytest.raises(ValueError, match="attribution_mode"):
            CounterfactualGradientConfig(attribution_mode="mask")

    def test_learned_mask_defaults_and_validators(self):
        cfg = LearnedMaskConfig()
        # Calibrated defaults (L2/L8/L10 wd sweep, 2026-07-25). Decay is
        # schedule-coupled: only steps*lr*wd matters, calibrated at ~1.0.
        assert cfg.steps == 400 and cfg.lr == 0.05
        assert cfg.optimizer == "adamw" and cfg.weight_decay == 0.05
        assert cfg.steps * cfg.lr * cfg.weight_decay == pytest.approx(1.0, abs=1e-6)
        assert cfg.code_dtype == "stream"
        assert cfg.keep_threshold == 0.5
        assert config.discovery.learned_mask.beta == 1.0
        with pytest.raises(ValueError, match="code_dtype"):
            LearnedMaskConfig(code_dtype="bf16")
        with pytest.raises(ValueError):
            LearnedMaskConfig(steps=0)
        with pytest.raises(ValueError):
            LearnedMaskConfig(keep_threshold=1.0)
        with pytest.raises(ValueError):
            LearnedMaskConfig(holdout_frac=1.0)
        with pytest.raises(ValueError):
            LearnedMaskConfig(lr=0.0)
        with pytest.raises(ValueError):
            LearnedMaskConfig(beta=-0.1)


def _abl(monkeypatch, mode="mask", position_aware=False):
    from circuit.discovery.ablation_gradient import AblationGradientDiscovery
    monkeypatch.setattr(config.discovery.ablation_gradient, "attribution_mode", mode)
    monkeypatch.setattr(config.discovery, "position_aware", position_aware)
    bank = MockSAEBank()
    return AblationGradientDiscovery(MagicMock(), bank, torch.zeros(1), MagicMock()), bank


def _cf(monkeypatch, mode, position_aware=False):
    from circuit.discovery.counterfactual_gradient import CounterfactualGradientDiscovery
    monkeypatch.setattr(config.discovery.counterfactual_gradient, "attribution_mode", mode)
    monkeypatch.setattr(config.discovery, "position_aware", position_aware)
    bank = MockSAEBank()
    return CounterfactualGradientDiscovery(MagicMock(), bank, torch.zeros(1), MagicMock()), bank


TOKENS = torch.zeros(4, 2, dtype=torch.long)
ARGMAX = torch.zeros(4, dtype=torch.long)


class TestAblMaskHop:
    def test_dispatch_reaches_engine_and_returns_supports(self, monkeypatch):
        method, bank = _abl(monkeypatch)
        import circuit.instrument.learned_mask as lm
        captured = {}

        def stub(inference, bank_, **kw):
            captured.update(kw)
            return ({FeatureID(0, "attn", 7): 0.9},
                    {"loss_initial": 1.0, "loss_final": 0.1})

        monkeypatch.setattr(lm, "run_learned_mask", stub)
        # seed at comp (1, resid) -> comp_idx = 1*3 + 2 = 5
        scores, floor, natural = method._run_ablation_hop(
            5, 11, TOKENS, ARGMAX, MagicMock())
        assert captured["objective"] == "pos"
        assert scores == {FeatureID(0, "attn", 7): 0.9}
        assert method._pending_inhibitors == {}
        assert (floor, natural) == (1.0, 0.1)

    def test_position_aware_raises_loudly(self, monkeypatch):
        method, bank = _abl(monkeypatch, position_aware=True)
        with pytest.raises(ValueError, match="position_aware"):
            method._run_ablation_hop(5, 11, TOKENS, ARGMAX, MagicMock())


class TestCfMaskHops:
    def test_contrast_delivers_supports(self, monkeypatch):
        method, bank = _cf(monkeypatch, "mask_contrast")
        import circuit.instrument.learned_mask as lm
        captured = {}

        def stub(inference, bank_, **kw):
            captured.update(kw)
            return ({FeatureID(0, "mlp", 3): 0.8}, {"loss_initial": 1, "loss_final": 0})

        monkeypatch.setattr(lm, "run_learned_mask", stub)
        acts, inhs = method._run_mask_hop(5, 11, TOKENS, TOKENS, ARGMAX, 2.0,
                                          MagicMock())
        assert captured["objective"] == "contrast"
        assert captured["target_act"] == 2.0
        assert acts == {FeatureID(0, "mlp", 3): 0.8}
        assert inhs == {}

    def test_negctx_delivers_inhibitors_with_negative_scores(self, monkeypatch):
        method, bank = _cf(monkeypatch, "mask_negctx")
        import circuit.instrument.learned_mask as lm

        def stub(inference, bank_, **kw):
            assert kw["objective"] == "negctx"
            return ({FeatureID(0, "attn", 5): -0.7}, {"loss_initial": 1, "loss_final": 0})

        monkeypatch.setattr(lm, "run_learned_mask", stub)
        acts, inhs = method._run_mask_hop(5, 11, TOKENS, TOKENS, ARGMAX, 2.0,
                                          MagicMock())
        assert acts == {}
        assert inhs == {FeatureID(0, "attn", 5): -0.7}
        assert all(v < 0 for v in inhs.values())

    def test_position_aware_raises_loudly(self, monkeypatch):
        method, bank = _cf(monkeypatch, "mask_negctx", position_aware=True)
        with pytest.raises(ValueError, match="position_aware"):
            method._run_mask_hop(5, 11, TOKENS, TOKENS, ARGMAX, 2.0, MagicMock())
