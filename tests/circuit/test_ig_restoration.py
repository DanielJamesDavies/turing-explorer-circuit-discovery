"""
Tests for attribution_mode="ig_restoration" (per-round integrated-gradients
scoring inside the restoration loop) and the final_ig_polish pass.

Part 1  Instrument alpha arithmetic (MockSAEBank): alpha=0 byte-compatible
        with classic restoration; alpha=1 -> decode(natural)+residual;
        alpha=0.5 partial-mask exact; alpha!=0 without natural_dense raises.
Part 2  Scorer equivalence: ig_restoration_scores(ig_steps=1) must equal
        restoration_scores exactly (only sample is alpha=0) — the key
        regression tying the new mode to the old one.
Part 3  Loop behaviour under the IG scorer on the TinyBank chain: chain
        recruitment, alpha-0 trajectory metric, certificate early stop.
Part 4  run_restoration_selection routing (scorer="ig"), bogus scorer, and
        final_ig_polish (ranking-only re-scores + provenance stamping).
Part 5  Config + runner surface.
"""

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from circuit.discovery.iterative_selection import run_iterative_selection
from circuit.instrument.restoration import (
    MaskedRestorationInstrument,
    ig_restoration_scores,
    restoration_scores,
    run_restoration_selection,
    stamp_restoration_provenance,
)
from sae.dense import sparse_topk_to_dense

from tests.conftest import D_MODEL, D_SAE
from tests.circuit.test_iterative_restoration import TinyBank

B, T = 2, 4
SITE = (0, "attn")
SEED = (1, "resid")
SITE_A, SITE_B = (0, "attn"), (0, "mlp")


def _cached(bank, x, site):
    ta, ti = bank.encode(x, site[1], site[0])
    dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
    residual = (x - bank.decode(dense, site[1], site[0])).detach()
    return residual, dense


def _instrument(bank, residuals, floors, masks, natural_dense=None, alpha=0.0):
    return MaskedRestorationInstrument(
        bank, {SITE}, residuals, floors, masks,
        SEED[0], SEED[1], torch.randn(D_MODEL), torch.tensor(0.0),
        natural_dense=natural_dense, alpha=alpha,
    )


class TestInstrumentAlpha:
    def test_alpha_zero_matches_classic_restoration(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, _ = _cached(mock_sae_bank, x, SITE)
        floor = torch.rand(D_SAE) * 0.4
        natural = torch.rand(D_SAE)
        masks = {SITE: torch.zeros(D_SAE, dtype=torch.bool)}
        classic = MaskedRestorationInstrument(
            mock_sae_bank, {SITE}, {SITE: residual}, {SITE: floor}, masks,
            SEED[0], SEED[1], torch.randn(D_MODEL), torch.tensor(0.0),
        )
        with_alpha = _instrument(
            mock_sae_bank, {SITE: residual}, {SITE: floor}, masks,
            natural_dense={SITE: natural}, alpha=0.0,
        )
        out_classic = classic.transform(*SITE, x.clone())
        out_alpha = with_alpha.transform(*SITE, x.clone())
        assert torch.equal(out_classic, out_alpha)

    def test_alpha_one_unrestored_is_natural_plus_residual(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, dense = _cached(mock_sae_bank, x, SITE)
        floor = torch.rand(D_SAE) * 0.4
        natural = torch.rand(D_SAE)
        inst = _instrument(
            mock_sae_bank, {SITE: residual}, {SITE: floor},
            {SITE: torch.zeros(D_SAE, dtype=torch.bool)},
            natural_dense={SITE: natural}, alpha=1.0,
        )
        out = inst.transform(*SITE, x.clone())
        expected = mock_sae_bank.decode(natural.expand_as(dense), "attn", 0) + residual
        assert torch.allclose(out, expected, atol=1e-5)

    def test_alpha_half_partial_mask_exact(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, dense = _cached(mock_sae_bank, x, SITE)
        floor = torch.rand(D_SAE) * 0.4
        natural = torch.rand(D_SAE)
        mask = torch.zeros(D_SAE, dtype=torch.bool)
        mask[2] = True
        mask[5] = True
        inst = _instrument(
            mock_sae_bank, {SITE: residual}, {SITE: floor}, {SITE: mask},
            natural_dense={SITE: natural}, alpha=0.5,
        )
        out = inst.transform(*SITE, x.clone())
        midpoint = floor + 0.5 * (natural - floor)
        code = midpoint.expand_as(dense).clone()
        code[:, :, [2, 5]] = dense[:, :, [2, 5]]  # restored dims stay live
        expected = mock_sae_bank.decode(code, "attn", 0) + residual
        assert torch.allclose(out, expected, atol=1e-5)

    def test_nonzero_alpha_without_natural_raises(self, mock_sae_bank):
        with pytest.raises(ValueError, match="natural_dense"):
            _instrument(
                mock_sae_bank, {}, {}, {},
                natural_dense=None, alpha=0.5,
            )

    def test_leaf_receives_gradient_at_alpha_half(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, _ = _cached(mock_sae_bank, x, SITE)
        inst = _instrument(
            mock_sae_bank, {SITE: residual}, {SITE: torch.zeros(D_SAE)},
            {SITE: torch.zeros(D_SAE, dtype=torch.bool)},
            natural_dense={SITE: torch.rand(D_SAE)}, alpha=0.5,
        )
        out = inst.transform(*SITE, x.clone())
        grad = torch.autograd.grad(out.sum(), inst.leaves[SITE])[0]
        assert grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# TinyBank chain harness shared by parts 2-4 (mirrors the classic tests:
# latent u at site A drives the seed only through latent d at site B).
# ---------------------------------------------------------------------------


def _chain_setup(c=2.0):
    bank = TinyBank(c)
    x0 = torch.zeros(1, 1, 3)
    x0[0, 0, 0] = c

    residuals, naturals = {}, {}
    for site in (SITE_A, SITE_B):
        ta, ti = bank.encode(x0, site[1], site[0])
        dense = sparse_topk_to_dense(ta, ti, 3, dtype=x0.dtype)
        residuals[site] = (x0 - bank.decode(dense, site[1], site[0])).detach()
        naturals[site] = dense[0, 0].detach()

    def forward_fn(tokens, patcher=None, **kwargs):
        out_a = patcher.transform(0, "attn", x0.clone())
        out_b = patcher.transform(0, "mlp", out_a)
        patcher.transform(1, "attn", out_b)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn

    common = dict(
        tokens=torch.zeros(1, 1, dtype=torch.long),
        substitute_sites={SITE_A, SITE_B},
        residuals=residuals,
        site_floors={SITE_A: torch.zeros(3), SITE_B: torch.zeros(3)},
        natural_dense=naturals,
        seed_layer=1, seed_kind="attn",
        w_seed=torch.tensor([0.0, 0.0, 1.0]), b_seed=torch.tensor(0.0),
        pos_argmax=torch.zeros(1, dtype=torch.long),
        target_act=0.0,
    )
    masks = {
        SITE_A: torch.zeros(3, dtype=torch.bool),
        SITE_B: torch.zeros(3, dtype=torch.bool),
    }
    return inf, bank, common, masks, c


class TestScorerEquivalence:
    def test_ig_steps_one_equals_point_scorer_empty_mask(self):
        inf, bank, common, masks, _ = _chain_setup()
        point_scores, point_metric = restoration_scores(inf, bank, masks=masks, **common)
        ig_scores, ig_metric = ig_restoration_scores(
            inf, bank, masks=masks, ig_steps=1, **common
        )
        assert ig_metric == point_metric
        for site in point_scores:
            assert torch.allclose(ig_scores[site], point_scores[site], atol=0, rtol=0)

    def test_ig_steps_one_equals_point_scorer_partial_mask(self):
        inf, bank, common, masks, _ = _chain_setup()
        masks[SITE_B][1] = True  # d restored
        point_scores, point_metric = restoration_scores(inf, bank, masks=masks, **common)
        ig_scores, ig_metric = ig_restoration_scores(
            inf, bank, masks=masks, ig_steps=1, **common
        )
        assert ig_metric == point_metric
        for site in point_scores:
            assert torch.allclose(ig_scores[site], point_scores[site], atol=0, rtol=0)

    def test_ig_steps_below_one_raises(self):
        inf, bank, common, masks, _ = _chain_setup()
        with pytest.raises(ValueError, match="ig_steps"):
            ig_restoration_scores(inf, bank, masks=masks, ig_steps=0, **common)


class TestIGChainRecruitment:
    def _score_fn(self, inf, bank, common, ig_steps=4):
        def score_fn(masks):
            return ig_restoration_scores(
                inf, bank, masks=masks, ig_steps=ig_steps, **common
            )
        return score_fn

    def test_round_one_sees_direct_parent_only(self):
        inf, bank, common, masks, c = _chain_setup()
        scores, metric = self._score_fn(inf, bank, common)(masks)
        # Trajectory metric is the alpha=0 pass: the restored-state metric.
        assert metric == pytest.approx(-c * c, abs=1e-5)
        assert scores[SITE_B][1].item() > 0.1     # d scores high
        assert abs(scores[SITE_A][0].item()) < 1e-6  # u severed at every alpha

    def test_iteration_recruits_the_chain(self):
        inf, bank, common, masks, _ = _chain_setup()
        result = run_iterative_selection(
            self._score_fn(inf, bank, common), masks=masks, rounds=2, per_round_k=1
        )
        assert result.round_of[(SITE_B, 1)] == 0
        assert result.round_of[(SITE_A, 0)] == 1
        assert result.positives[(SITE_A, 0)] > 0
        assert result.metric_trajectory[1] > result.metric_trajectory[0]

    def test_certificate_early_stop(self):
        # After u and d are both restored the gap metric reaches 0 exactly
        # (fully-restored chain reproduces the clean stream), so round 3's
        # certificate check trips.
        inf, bank, common, masks, _ = _chain_setup()
        result = run_iterative_selection(
            self._score_fn(inf, bank, common), masks=masks, rounds=5, per_round_k=1,
            certificate_tol=1e-6, target_metric=0.0,
        )
        assert result.stopped_early is True
        assert result.rounds_used == 2
        assert result.metric_trajectory[-1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# run_restoration_selection integration (saes shims as in the classic test).
# ---------------------------------------------------------------------------


def _selection_harness(c=2.0):
    bank = TinyBank(c)
    seed_sae = SimpleNamespace(encoder=SimpleNamespace(weight=None))
    seed_sae.encoder.weight = torch.zeros(3, 3)
    seed_sae.encoder.weight[0] = torch.tensor([0.0, 0.0, 1.0])
    seed_sae._get_bias_eff = lambda: torch.zeros(3)
    bank.saes = {"attn": [None, seed_sae], "mlp": [None, None]}

    # Two positions: signal at t=0, zeros at t=1 — otherwise the mean floor
    # equals the natural values and every restoration delta is 0.
    x0 = torch.zeros(1, 2, 3)
    x0[0, 0, 0] = c

    def forward_fn(tokens, patcher=None, activations_callback=None, **kwargs):
        if activations_callback is not None:
            acts = (x0.clone(), x0.clone())
            activations_callback(0, acts)
            activations_callback(1, acts)
            return
        out_a = patcher.transform(0, "attn", x0.clone())
        out_b = patcher.transform(0, "mlp", out_a)
        patcher.transform(1, "attn", out_b)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    return inf, bank


class TestRunRestorationSelectionIG:
    def test_ig_scorer_recruits_chain_with_provenance(self):
        from circuit.types.feature_id import FeatureID

        inf, bank = _selection_harness()
        positives, negatives, result = run_restoration_selection(
            inf, bank,
            tokens=torch.zeros(1, 2, dtype=torch.long),
            pos_argmax=torch.zeros(1, dtype=torch.long),
            seed_layer=1, seed_kind="attn", seed_latent_idx=0,
            target_act=0.0,
            rounds=2, per_round_k=1, certificate_tol=0.0,
            scorer="ig", ig_steps=2,
        )
        assert FeatureID(0, "mlp", 1) in positives  # d, round 0
        assert FeatureID(0, "attn", 0) in positives  # u, round 1
        assert result.round_of[((0, "mlp"), 1)] == 0
        assert result.rounds_used == 2
        assert result.polish_scores is None

    def test_bogus_scorer_raises(self):
        inf, bank = _selection_harness()
        with pytest.raises(ValueError, match="scorer"):
            run_restoration_selection(
                inf, bank,
                tokens=torch.zeros(1, 2, dtype=torch.long),
                pos_argmax=torch.zeros(1, dtype=torch.long),
                seed_layer=1, seed_kind="attn", seed_latent_idx=0,
                target_act=0.0,
                rounds=1, per_round_k=1, certificate_tol=0.0,
                scorer="bogus",
            )


class TestFinalIGPolish:
    def _run(self, final_ig_polish):
        inf, bank = _selection_harness()
        return run_restoration_selection(
            inf, bank,
            tokens=torch.zeros(1, 2, dtype=torch.long),
            pos_argmax=torch.zeros(1, dtype=torch.long),
            seed_layer=1, seed_kind="attn", seed_latent_idx=0,
            target_act=0.0,
            rounds=2, per_round_k=1, certificate_tol=0.0,
            final_ig_polish=final_ig_polish, polish_ig_steps=4,
        )

    def test_polish_scores_cover_selected_membership_unchanged(self):
        plain_pos, plain_neg, plain_result = self._run(False)
        pol_pos, pol_neg, pol_result = self._run(True)
        assert plain_result.polish_scores is None
        assert pol_result.polish_scores is not None
        assert set(pol_result.polish_scores) == set(pol_result.round_of)
        # Membership and the loop scores used for thresholds are unchanged.
        assert set(pol_pos) == set(plain_pos)
        assert set(pol_neg) == set(plain_neg)
        for fid, value in plain_pos.items():
            assert pol_pos[fid] == pytest.approx(value)

    def test_stamp_applies_polish_to_scores_and_edges(self):
        from circuit.types.feature_id import FeatureID
        from store.circuits import Circuit, CircuitNode

        circuit = Circuit(name="polished")
        seed = circuit.add_node(
            CircuitNode(metadata={"feature_id": FeatureID(1, "attn", 0), "role": "seed"})
        )
        member = circuit.add_node(
            CircuitNode(metadata={
                "feature_id": FeatureID(0, "mlp", 1),
                "role": "counterfactual_activator",
                "attribution_score": 0.25,
            })
        )
        circuit.add_edge(member.uuid, seed.uuid, weight=0.25)
        result = SimpleNamespace(
            round_of={(((0, "mlp")), 1): 0},
            rounds_used=1,
            stopped_early=False,
            metric_trajectory=[-4.0],
            polish_scores={(((0, "mlp")), 1): 0.9},
        )
        stamp_restoration_provenance(circuit, result)
        assert member.metadata["selected_round"] == 0
        assert member.metadata["selection_score"] == 0.25
        assert member.metadata["attribution_score"] == 0.9
        assert circuit.edges[0].weight == 0.9
        assert circuit.metadata["restoration_ig_polished"] is True

    def test_stamp_without_polish_leaves_scores_untouched(self):
        from circuit.types.feature_id import FeatureID
        from store.circuits import Circuit, CircuitNode

        circuit = Circuit(name="plain")
        seed = circuit.add_node(
            CircuitNode(metadata={"feature_id": FeatureID(1, "attn", 0), "role": "seed"})
        )
        member = circuit.add_node(
            CircuitNode(metadata={
                "feature_id": FeatureID(0, "mlp", 1),
                "role": "counterfactual_activator",
                "attribution_score": 0.25,
            })
        )
        circuit.add_edge(member.uuid, seed.uuid, weight=0.25)
        result = SimpleNamespace(
            round_of={(((0, "mlp")), 1): 0},
            rounds_used=1,
            stopped_early=False,
            metric_trajectory=[-4.0],
            polish_scores=None,
        )
        stamp_restoration_provenance(circuit, result)
        assert member.metadata["attribution_score"] == 0.25
        assert "selection_score" not in member.metadata
        assert circuit.edges[0].weight == 0.25
        assert circuit.metadata["restoration_ig_polished"] is False


class TestConfigAndRunnerSurface:
    def test_attribution_mode_accepts_ig_restoration(self):
        from config import AblationGradientConfig, CounterfactualGradientConfig

        assert CounterfactualGradientConfig(attribution_mode="ig_restoration").attribution_mode == "ig_restoration"
        assert AblationGradientConfig(attribution_mode="ig_restoration").attribution_mode == "ig_restoration"
        with pytest.raises(ValueError):
            CounterfactualGradientConfig(attribution_mode="bogus")

    def test_restoration_config_defaults(self):
        from config import RestorationConfig

        cfg = RestorationConfig()
        assert cfg.ig_steps == 4
        assert cfg.final_ig_polish is False

    def test_runner_round_prefix_modes(self):
        from analysis.circuits.gradient_size_sweep_runner import ROUND_PREFIX_MODES

        assert "restoration" in ROUND_PREFIX_MODES
        assert "ig_restoration" in ROUND_PREFIX_MODES
        assert "ig_mean" not in ROUND_PREFIX_MODES


# ---------------------------------------------------------------------------
# Round-admission cells (round_select x position_aware) and round chunking.
# ---------------------------------------------------------------------------


class TestRoundSelectCells:
    """The chain (u at site A -> d at site B -> seed) must be recruited over
    two rounds under EVERY admission rule — the rules change how much each
    round admits, not the re-linearisation logic that recruits chains."""

    def _run(self, **kw):
        from circuit.types.feature_id import FeatureID

        inf, bank = _selection_harness()
        positives, negatives, result = run_restoration_selection(
            inf, bank,
            tokens=torch.zeros(1, 2, dtype=torch.long),
            pos_argmax=torch.zeros(1, dtype=torch.long),
            seed_layer=1, seed_kind="attn", seed_latent_idx=0,
            target_act=0.0,
            rounds=2, per_round_k=1, certificate_tol=0.0,
            **kw,
        )
        return positives, result, FeatureID

    def test_default_top_k_unchanged(self):
        positives, result, FeatureID = self._run()
        assert FeatureID(0, "mlp", 1) in positives
        assert FeatureID(0, "attn", 0) in positives
        assert result.round_of[((0, "mlp"), 1)] == 0
        assert result.round_of[((0, "attn"), 0)] == 1

    def test_abs_pctl_recruits_chain(self):
        """Round admission by pooled percentile: each round's only nonzero
        candidate clears its own distribution's cut, so the chain recruits
        identically — but with a variable (rule-owned) budget."""
        positives, result, FeatureID = self._run(
            round_select="abs_pctl", round_abs_pctl=50.0
        )
        assert FeatureID(0, "mlp", 1) in positives
        assert FeatureID(0, "attn", 0) in positives
        assert result.round_of[((0, "mlp"), 1)] == 0
        assert result.round_of[((0, "attn"), 0)] == 1

    def test_pa_top_k_recruits_chain(self):
        positives, result, FeatureID = self._run(position_aware=True)
        assert FeatureID(0, "mlp", 1) in positives
        assert FeatureID(0, "attn", 0) in positives
        assert result.round_of[((0, "mlp"), 1)] == 0
        assert result.round_of[((0, "attn"), 0)] == 1

    def test_pa_abs_pctl_recruits_chain(self):
        positives, result, FeatureID = self._run(
            position_aware=True, round_select="abs_pctl", round_abs_pctl=50.0
        )
        assert FeatureID(0, "mlp", 1) in positives
        assert FeatureID(0, "attn", 0) in positives

    def test_bogus_round_select_raises(self):
        inf, bank = _selection_harness()
        with pytest.raises(ValueError, match="round_select"):
            run_restoration_selection(
                inf, bank,
                tokens=torch.zeros(1, 2, dtype=torch.long),
                pos_argmax=torch.zeros(1, dtype=torch.long),
                seed_layer=1, seed_kind="attn", seed_latent_idx=0,
                target_act=0.0, rounds=1, per_round_k=1, certificate_tol=0.0,
                round_select="banana",
            )


class TestRoundChunking:
    """Sequence count vs batch size in the round scorer: chunked rounds must
    merge to the single-pass result (identical replicated sequences, so the
    weighted mean is exactly the single-sequence value)."""

    def _chunk_setup(self, n=4):
        inf, bank, common, masks, c = _chain_setup()
        x0 = torch.zeros(n, 1, 3)
        x0[:, 0, 0] = c

        def forward_fn(tokens, patcher=None, **kwargs):
            # Chunked calls see exactly their own sequences — a mismatched
            # per-sequence cache must therefore fail loudly, not broadcast.
            x = x0[: tokens.shape[0]]
            out_a = patcher.transform(0, "attn", x.clone())
            out_b = patcher.transform(0, "mlp", out_a)
            patcher.transform(1, "attn", out_b)

        inf.forward.side_effect = forward_fn
        common = dict(common)
        common["tokens"] = torch.zeros(n, 1, dtype=torch.long)
        common["pos_argmax"] = torch.zeros(n, dtype=torch.long)
        # Residuals are PER-SEQUENCE state ([B, T, d_model]): expand the
        # single-sequence cache to n rows so the chunk-slicing path is
        # actually exercised (a [1, T, d] cache would silently broadcast —
        # exactly the bug the production smoke caught).
        common["residuals"] = {
            site: r.expand(n, -1, -1).clone() for site, r in common["residuals"].items()
        }
        return inf, bank, common, masks

    def test_chunked_round_matches_single_pass(self):
        from circuit.instrument.restoration import _round_scores

        inf, bank, common, masks = self._chunk_setup()
        single, m_single, _ = _round_scores(
            inf, bank, masks=masks, alphas=[0.0], **common)
        inf2, bank2, common2, masks2 = self._chunk_setup()
        chunked, m_chunked, _ = _round_scores(
            inf2, bank2, masks=masks2, alphas=[0.0], batch_size=2, **common2)
        assert m_chunked == pytest.approx(m_single, abs=1e-6)
        for site in single:
            assert torch.allclose(chunked[site], single[site], atol=1e-6)

    def test_chunked_pa_selection_matches_single_pass(self):
        from circuit.instrument.position_aware import PositionAwareSpec
        from circuit.instrument.restoration import _round_scores

        spec = PositionAwareSpec(peaks=torch.zeros(4, dtype=torch.long), top_n=1)
        inf, bank, common, masks = self._chunk_setup()
        _, _, sel_single = _round_scores(
            inf, bank, masks=masks, alphas=[0.0], position_select=spec, **common)
        inf2, bank2, common2, masks2 = self._chunk_setup()
        _, _, sel_chunked = _round_scores(
            inf2, bank2, masks=masks2, alphas=[0.0], batch_size=2,
            position_select=spec, **common2)
        assert set(sel_chunked) == set(sel_single)
        for key, val in sel_single.items():
            assert sel_chunked[key] == pytest.approx(val, rel=1e-5, abs=1e-7)


def test_restoration_config_round_select_validators():
    from config import RestorationConfig

    assert RestorationConfig().round_select == "top_k"
    assert RestorationConfig(round_select="abs_pctl", round_abs_pctl=95).round_abs_pctl == 95
    with pytest.raises(ValueError, match="round_select"):
        RestorationConfig(round_select="banana")
    with pytest.raises(ValueError, match="percentile"):
        RestorationConfig(round_abs_pctl=0)
