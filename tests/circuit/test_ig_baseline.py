"""
Unit tests for the ig_baseline attribution mode (SFC-style integrated
gradients from the mean-ablation floor; Marks et al. 2025).

Part 1  TestInterpolatedCodeInstrument — transform arithmetic at the path
        endpoints, seed capture, out-of-scope passthrough.
Part 2  TestIntegratedBaselineScores — IG completeness on a fully linear
        pipeline (exact for linear paths), endpoint metrics, pass protocol.
Part 3  TestExtractSignedRoles — sign split, scope semantics, count mask.
"""

import pytest
import torch
from unittest.mock import MagicMock

from circuit.instrument.ig_baseline import (
    InterpolatedCodeInstrument,
    collect_natural_codes,
    extract_signed_roles,
    integrated_baseline_scores,
)
from circuit.types.feature_id import FeatureID
from sae.dense import sparse_topk_to_dense

from tests.conftest import D_MODEL, D_SAE, KINDS, N_LAYERS, MockSAEBank

B, T = 2, 4
SEED_LAYER, SEED_KIND = 1, "resid"
SITE = (0, "attn")


def _cache_from(bank, x, site):
    ta, ti = bank.encode(x, site[1], site[0])
    dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
    residual = x - bank.decode(dense, site[1], site[0])
    return {site: (ta, ti)}, {site: residual}, dense


def _make_instrument(bank, codes, residuals, floors, alpha, w_seed=None, b_seed=None):
    return InterpolatedCodeInstrument(
        bank,
        {SITE},
        codes,
        residuals,
        floors,
        alpha,
        SEED_LAYER,
        SEED_KIND,
        w_seed if w_seed is not None else torch.randn(D_MODEL),
        b_seed if b_seed is not None else torch.tensor(0.0),
    )


class TestInterpolatedCodeInstrument:
    def test_alpha_one_reproduces_input(self, mock_sae_bank):
        """At alpha=1 with codes cached from the same x, the output must be
        decode(natural) + (x - decode(natural)) == x exactly."""
        x = torch.randn(B, T, D_MODEL)
        codes, residuals, _ = _cache_from(mock_sae_bank, x, SITE)
        floors = {SITE: torch.rand(D_SAE)}
        instrument = _make_instrument(mock_sae_bank, codes, residuals, floors, alpha=1.0)
        out = instrument.transform(*SITE, x.clone())
        assert torch.allclose(out, x, atol=1e-5)

    def test_alpha_zero_is_floor_plus_residual(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        codes, residuals, dense = _cache_from(mock_sae_bank, x, SITE)
        floor = torch.rand(D_SAE)
        instrument = _make_instrument(mock_sae_bank, codes, residuals, {SITE: floor}, alpha=0.0)
        out = instrument.transform(*SITE, x.clone())
        expected = mock_sae_bank.decode(floor.expand_as(dense), "attn", 0) + residuals[SITE]
        assert torch.allclose(out, expected, atol=1e-5)

    def test_seed_site_captures_and_passes_through(self, mock_sae_bank):
        w_seed = torch.randn(D_MODEL)
        b_seed = torch.tensor(0.5)
        instrument = _make_instrument(mock_sae_bank, {}, {}, {}, 0.5, w_seed, b_seed)
        x = torch.randn(B, T, D_MODEL)
        out = instrument.transform(SEED_LAYER, SEED_KIND, x)
        assert torch.equal(out, x)
        assert torch.allclose(instrument.seed_pre_act, x @ w_seed + b_seed, atol=1e-5)

    def test_out_of_scope_site_untouched(self, mock_sae_bank):
        instrument = _make_instrument(mock_sae_bank, {}, {}, {}, 0.5)
        x = torch.randn(B, T, D_MODEL)
        assert torch.equal(instrument.transform(0, "mlp", x), x)

    def test_anchor_receives_gradient(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        codes, residuals, _ = _cache_from(mock_sae_bank, x, SITE)
        instrument = _make_instrument(mock_sae_bank, codes, residuals, {SITE: torch.zeros(D_SAE)}, 0.5)
        out = instrument.transform(*SITE, x.clone())
        grad = torch.autograd.grad(out.sum(), instrument.anchors[SITE])[0]
        assert grad is not None and grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Stub inference driving a fully linear pipeline:
#   x0 --site(0,attn) transform--> out0 --@ W_map--> x_seed --seed capture
# Linear decode + linear map + linear metric => IG is exact.
# ---------------------------------------------------------------------------


def _linear_stub(bank, x0, w_map):
    def forward_fn(tokens, activations_callback=None, patcher=None, **kwargs):
        if activations_callback is not None:
            acts = tuple(x0.clone() for _ in KINDS)
            for layer in range(N_LAYERS):
                activations_callback(layer, acts)
            return
        out0 = patcher.transform(*SITE, x0.clone())
        x_seed = out0 @ w_map
        patcher.transform(SEED_LAYER, SEED_KIND, x_seed)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    return inf


class TestIntegratedBaselineScores:
    def _run(self, objective="drive", ig_steps=4, target_act=0.0):
        torch.manual_seed(7)
        bank = MockSAEBank()
        x0 = torch.randn(B, T, D_MODEL)
        w_map = torch.randn(D_MODEL, D_MODEL) * 0.2
        inf = _linear_stub(bank, x0, w_map)
        scores, m_floor, m_nat = integrated_baseline_scores(
            inf,
            bank,
            tokens=torch.zeros(B, T, dtype=torch.long),
            substitute_sites={SITE},
            site_floors={SITE: torch.rand(D_SAE) * 0.3},
            seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND,
            w_seed=torch.randn(D_MODEL),
            b_seed=torch.tensor(0.1),
            pos_argmax=torch.zeros(B, dtype=torch.long),
            objective=objective,
            target_act=target_act,
            ig_steps=ig_steps,
        )
        return scores, m_floor, m_nat, inf

    def test_completeness_exact_on_linear_pipeline(self):
        """IG completeness: sum of scores == metric(natural) - metric(floor)
        exactly, because every map in the stub pipeline is linear."""
        scores, m_floor, m_nat, _ = self._run(objective="drive")
        total = float(scores[SITE].sum().item())
        assert total == pytest.approx(m_nat - m_floor, abs=1e-3)

    def test_endpoint_metrics_ordered_passes(self):
        """ig_steps grad passes + 1 endpoint pass + 1 clean caching pass."""
        _, _, _, inf = self._run(ig_steps=4)
        assert inf.forward.call_count == 4 + 1 + 1

    def test_gap_objective_runs_and_scores_nonzero(self):
        scores, m_floor, m_nat, _ = self._run(objective="gap", target_act=2.0)
        assert scores[SITE].abs().sum() > 0
        assert m_nat >= m_floor - 1e-6  # gap metric is -(peak-target)^2; natural closer

    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError):
            self._run(objective="banana")


SITE_A = (0, "attn")
SITE_B = (0, "mlp")


def _chain_stub(bank, x0, w_ab, w_bs):
    """Two substituted sites in series: A -> B -> seed. The metric depends on
    site A ONLY through site B, so A's anchors must receive gradient via the
    identity passthrough (SFC's pass-through gradients) or A scores zero."""

    def forward_fn(tokens, activations_callback=None, patcher=None, **kwargs):
        if activations_callback is not None:
            xb_nat = x0 @ w_ab
            acts_by_kind = {"attn": x0, "mlp": xb_nat, "resid": x0}
            acts = tuple(acts_by_kind[kind].clone() for kind in KINDS)
            for layer in range(N_LAYERS):
                activations_callback(layer, acts)
            return
        out_a = patcher.transform(*SITE_A, x0.clone())
        out_b = patcher.transform(*SITE_B, out_a @ w_ab)
        patcher.transform(SEED_LAYER, SEED_KIND, out_b @ w_bs)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    return inf


class TestChainGradientFlow:
    """Regression tests for the severed-gradient bug: without the identity
    passthrough, upstream-of-upstream sites received grad=None and the mode
    silently degenerated into direct-edge attribution."""

    def _run_chain(self):
        torch.manual_seed(11)
        bank = MockSAEBank()
        x0 = torch.randn(B, T, D_MODEL)
        w_ab = torch.randn(D_MODEL, D_MODEL) * 0.2
        w_bs = torch.randn(D_MODEL, D_MODEL) * 0.2
        inf = _chain_stub(bank, x0, w_ab, w_bs)
        return integrated_baseline_scores(
            inf,
            bank,
            tokens=torch.zeros(B, T, dtype=torch.long),
            substitute_sites={SITE_A, SITE_B},
            site_floors={SITE_A: torch.rand(D_SAE) * 0.3, SITE_B: torch.rand(D_SAE) * 0.3},
            seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND,
            w_seed=torch.randn(D_MODEL),
            b_seed=torch.tensor(0.1),
            pos_argmax=torch.zeros(B, dtype=torch.long),
            objective="drive",
            ig_steps=4,
        )

    def test_mediated_site_receives_credit(self):
        """Site A influences the metric only through substituted site B; its
        scores must be nonzero (total-effect attribution, not direct-edge)."""
        scores, _, _ = self._run_chain()
        assert scores[SITE_A].abs().sum().item() > 1e-6

    def test_final_cut_alone_matches_endpoint_gap(self):
        """The metric value at each alpha is carried entirely by the final
        cut (site B), so B's scores alone recover the endpoint gap on a
        linear chain; A's flow-through credit comes on top of that."""
        scores, m_floor, m_nat = self._run_chain()
        total_b = float(scores[SITE_B].sum().item())
        assert total_b == pytest.approx(m_nat - m_floor, abs=1e-3)


class TestExtractSignedRoles:
    def _scores(self):
        site_scores = torch.zeros(D_SAE)
        site_scores[1] = 3.0
        site_scores[2] = 2.0
        site_scores[3] = -4.0
        site_scores[4] = -1.0
        return {(0, "attn"): site_scores}

    def test_sign_split_and_global_topk(self):
        positives, negatives = extract_signed_roles(
            self._scores(), kinds=KINDS, n_kinds=len(KINDS),
            top_k_positive=1, top_k_negative=1,
            min_active_count=0, active_count=None, top_k_scope="global",
        )
        assert list(positives) == [FeatureID(0, "attn", 1)]
        assert list(negatives) == [FeatureID(0, "attn", 3)]
        assert positives[FeatureID(0, "attn", 1)] == pytest.approx(3.0)
        assert negatives[FeatureID(0, "attn", 3)] == pytest.approx(-4.0)

    def test_active_count_mask_excludes_dead_latents(self):
        active = torch.zeros(N_LAYERS * len(KINDS), D_SAE)
        active[0, 2] = 100  # only latent 2 at component 0 is alive
        positives, negatives = extract_signed_roles(
            self._scores(), kinds=KINDS, n_kinds=len(KINDS),
            top_k_positive=8, top_k_negative=8,
            min_active_count=10, active_count=active, top_k_scope="global",
        )
        assert list(positives) == [FeatureID(0, "attn", 2)]
        assert negatives == {}

    def test_layer_kind_scope_keeps_per_site_topk(self):
        scores = self._scores()
        scores[(1, "mlp")] = torch.zeros(D_SAE)
        scores[(1, "mlp")][7] = 0.5
        positives, _ = extract_signed_roles(
            scores, kinds=KINDS, n_kinds=len(KINDS),
            top_k_positive=1, top_k_negative=0,
            min_active_count=0, active_count=None, top_k_scope="layer_kind",
        )
        assert set(positives) == {FeatureID(0, "attn", 1), FeatureID(1, "mlp", 7)}


class TestCollectNaturalCodes:
    def test_raises_on_missing_site(self):
        bank = MockSAEBank()
        inf = MagicMock()
        inf.forward.side_effect = lambda *a, **k: None
        with pytest.raises(RuntimeError):
            collect_natural_codes(inf, bank, torch.zeros(B, T, dtype=torch.long), {(99, "attn")})
