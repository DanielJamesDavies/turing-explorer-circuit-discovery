"""
Tests for MaskedRestorationInstrument, restoration_scores and the iterative
selector.

Part 1  Instrument arithmetic (MockSAEBank): mask=0 -> floor+residual;
        mask=1 -> identity; partial masks exact.
Part 2  Selector logic (stub score_fn): budget, no re-selection, mask
        mutation visible round-to-round, roles, certificate stop.
Part 3  Chain recruitment (hand-built 3-dim bank): u drives the seed ONLY
        through d. Round 1 must select d while u scores ~0 (leaves sever
        feature paths); round 2, with d restored CONNECTED, must select u.
        This test fails by construction under pinned restoration — it locks
        in connected-restoration semantics.
"""

import pytest
import torch
from unittest.mock import MagicMock

from circuit.discovery.iterative_selection import run_iterative_selection
from circuit.instrument.restoration import MaskedRestorationInstrument, restoration_scores
from sae.dense import sparse_topk_to_dense

from tests.conftest import D_MODEL, D_SAE, MockSAEBank

B, T = 2, 4
SITE = (0, "attn")
SEED = (1, "resid")


def _cached(bank, x, site):
    ta, ti = bank.encode(x, site[1], site[0])
    dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
    residual = (x - bank.decode(dense, site[1], site[0])).detach()
    return residual, dense


def _instrument(bank, residuals, floors, masks):
    return MaskedRestorationInstrument(
        bank, {SITE}, residuals, floors, masks,
        SEED[0], SEED[1], torch.randn(D_MODEL), torch.tensor(0.0),
    )


class TestMaskedRestorationInstrument:
    def test_mask_zero_is_floor_plus_residual(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, dense = _cached(mock_sae_bank, x, SITE)
        floor = torch.rand(D_SAE) * 0.4
        inst = _instrument(mock_sae_bank, {SITE: residual}, {SITE: floor},
                           {SITE: torch.zeros(D_SAE, dtype=torch.bool)})
        out = inst.transform(*SITE, x.clone())
        expected = mock_sae_bank.decode(floor.expand_as(dense), "attn", 0) + residual
        assert torch.allclose(out, expected, atol=1e-5)

    def test_mask_one_is_identity(self, mock_sae_bank):
        """All restored + live encode + cached residual == x exactly."""
        x = torch.randn(B, T, D_MODEL)
        residual, _ = _cached(mock_sae_bank, x, SITE)
        inst = _instrument(mock_sae_bank, {SITE: residual}, {SITE: torch.zeros(D_SAE)},
                           {SITE: torch.ones(D_SAE, dtype=torch.bool)})
        out = inst.transform(*SITE, x.clone())
        assert torch.allclose(out, x, atol=1e-5)

    def test_partial_mask_mixes_floor_and_live(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, dense = _cached(mock_sae_bank, x, SITE)
        floor = torch.rand(D_SAE) * 0.4
        mask = torch.zeros(D_SAE, dtype=torch.bool)
        mask[2] = True
        mask[5] = True
        inst = _instrument(mock_sae_bank, {SITE: residual}, {SITE: floor}, {SITE: mask})
        out = inst.transform(*SITE, x.clone())
        code = floor.expand_as(dense).clone()
        code[:, :, [2, 5]] = dense[:, :, [2, 5]]
        expected = mock_sae_bank.decode(code, "attn", 0) + residual
        assert torch.allclose(out, expected, atol=1e-5)

    def test_unrestored_leaf_receives_gradient(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, _ = _cached(mock_sae_bank, x, SITE)
        inst = _instrument(mock_sae_bank, {SITE: residual}, {SITE: torch.zeros(D_SAE)},
                           {SITE: torch.zeros(D_SAE, dtype=torch.bool)})
        out = inst.transform(*SITE, x.clone())
        grad = torch.autograd.grad(out.sum(), inst.leaves[SITE])[0]
        assert grad.abs().sum() > 0


class TestIterativeSelector:
    def _stub(self, per_round_scores, metrics):
        calls = {"masks_seen": []}

        def score_fn(masks):
            round_index = len(calls["masks_seen"])
            calls["masks_seen"].append({s: m.clone() for s, m in masks.items()})
            return per_round_scores[min(round_index, len(per_round_scores) - 1)], metrics[
                min(round_index, len(metrics) - 1)
            ]

        return score_fn, calls

    def test_budget_roles_and_no_reselection(self):
        site = (0, "attn")
        s1 = {site: torch.tensor([5.0, -4.0, 3.0, 0.0])}
        score_fn, calls = self._stub([s1], [-10.0])
        masks = {site: torch.zeros(4, dtype=torch.bool)}
        result = run_iterative_selection(score_fn, masks=masks, rounds=3, per_round_k=1)
        assert result.rounds_used == 3
        assert result.round_of == {(site, 0): 0, (site, 1): 1, (site, 2): 2}
        assert (site, 0) in result.positives and (site, 1) in result.negatives
        # Round 2 saw round 1's selection restored in the mask it received.
        assert calls["masks_seen"][1][site][0].item() is True
        assert calls["masks_seen"][2][site][1].item() is True

    def test_allow_negative_false_skips_negative_candidates(self):
        """Negatives are never selected, restored, or counted against the
        round budget when allow_negative is off."""
        site = (0, "attn")
        scores = {site: torch.tensor([-9.0, 5.0, -7.0, 3.0])}
        score_fn, calls = self._stub([scores], [-10.0])
        masks = {site: torch.zeros(4, dtype=torch.bool)}
        result = run_iterative_selection(
            score_fn, masks=masks, rounds=2, per_round_k=1, allow_negative=False,
        )
        assert result.negatives == {}
        assert result.round_of == {(site, 1): 0, (site, 3): 1}
        assert not masks[site][0] and not masks[site][2]

    def test_certificate_stop(self):
        site = (0, "attn")
        scores = {site: torch.tensor([5.0, 4.0])}
        score_fn, _ = self._stub([scores, scores, scores], [-10.0, -0.005, -0.001])
        masks = {site: torch.zeros(2, dtype=torch.bool)}
        result = run_iterative_selection(
            score_fn, masks=masks, rounds=5, per_round_k=1,
            certificate_tol=0.01, target_metric=0.0,
        )
        assert result.stopped_early is True
        assert result.rounds_used == 1  # selected once, stopped at round 2's check
        assert len(result.metric_trajectory) == 2


# ---------------------------------------------------------------------------
# Part 3 — chain recruitment on a hand-built bank.
# Geometry (d_model=3, basis e0/e1/e2, d_sae=3, k=2, no decoder bias):
#   site A: latent u=0 reads e0, writes e1.       x0 = c*e0 (u fires at c)
#   site B: latent d=1 reads e0+0.5*e1, writes e2. seed reads e2 only.
# Direct path u->seed: decode_A(u)=e1, seed weight e2 -> zero. u matters
# only because d's encoder reads dim 1. Floor state: seed pre = -c.
# ---------------------------------------------------------------------------


class TinyBank:
    d_sae = 3
    kinds = ["attn", "mlp"]

    def __init__(self, c=2.0):
        self.weights = {}
        w_enc_a = torch.zeros(3, 3); w_enc_a[0, 0] = 1.0
        w_dec_a = torch.zeros(3, 3); w_dec_a[1, 0] = 1.0
        w_enc_b = torch.zeros(3, 3); w_enc_b[1, 0] = 1.0; w_enc_b[1, 1] = 0.5
        w_dec_b = torch.zeros(3, 3); w_dec_b[2, 1] = 1.0
        self.weights[(0, "attn")] = (w_enc_a, w_dec_a)
        self.weights[(0, "mlp")] = (w_enc_b, w_dec_b)

    def encode(self, x, kind, layer):
        w_enc, _ = self.weights[(layer, kind)]
        pre = torch.relu(x @ w_enc.T)
        return pre.topk(2, dim=-1)

    def decode(self, latents, kind, layer):
        _, w_dec = self.weights[(layer, kind)]
        return latents @ w_dec.T


class TestChainRecruitment:
    def _setup(self, c=2.0):
        bank = TinyBank(c)
        x0 = torch.zeros(1, 1, 3); x0[0, 0, 0] = c
        site_a, site_b = (0, "attn"), (0, "mlp")

        # Clean pass: both sites see x0 (instrument output is identity).
        residuals, naturals = {}, {}
        for site in (site_a, site_b):
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

        def score_fn(masks):
            return restoration_scores(
                inf, bank,
                tokens=torch.zeros(1, 1, dtype=torch.long),
                substitute_sites={site_a, site_b},
                residuals=residuals,
                site_floors={site_a: torch.zeros(3), site_b: torch.zeros(3)},
                natural_dense=naturals,
                masks=masks,
                seed_layer=1, seed_kind="attn",
                w_seed=torch.tensor([0.0, 0.0, 1.0]), b_seed=torch.tensor(0.0),
                pos_argmax=torch.zeros(1, dtype=torch.long),
                target_act=0.0,
            )

        masks = {site_a: torch.zeros(3, dtype=torch.bool), site_b: torch.zeros(3, dtype=torch.bool)}
        return score_fn, masks, site_a, site_b, c

    def test_round_one_sees_direct_parent_only(self):
        score_fn, masks, site_a, site_b, c = self._setup()
        scores, metric = score_fn(masks)
        assert metric == pytest.approx(-c * c, abs=1e-5)  # floor: pre = -c
        assert scores[site_b][1].item() > 0.1              # d scores high
        assert abs(scores[site_a][0].item()) < 1e-6        # u severed at floor

    def test_run_restoration_selection_maps_featureids_and_provenance(self):
        """The integration helper returns extractor-shaped FeatureID dicts
        and provenance; verified on the TinyBank chain via saes shims."""
        from types import SimpleNamespace
        from circuit.instrument.restoration import run_restoration_selection
        from circuit.types.feature_id import FeatureID

        score_fn, masks, site_a, site_b, c = self._setup()
        bank = TinyBank(c)
        # Shims the helper reads: seed SAE surface + saes structure.
        w_enc_seed = torch.zeros(3, 3); w_enc_seed[2, 2] = 1.0
        seed_sae = SimpleNamespace(
            encoder=SimpleNamespace(weight=torch.eye(3)[2:3].expand(3, 3).clone()),
        )
        seed_sae.encoder.weight = torch.zeros(3, 3); seed_sae.encoder.weight[0] = torch.tensor([0.0, 0.0, 1.0])
        seed_sae._get_bias_eff = lambda: torch.zeros(3)
        bank.saes = {"attn": [None, seed_sae], "mlp": [None, None]}

        # Two positions: signal at t=0, zeros at t=1 — otherwise the mean
        # floor equals the natural values and every restoration delta is 0.
        x0 = torch.zeros(1, 2, 3); x0[0, 0, 0] = c

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

        positives, negatives, result = run_restoration_selection(
            inf, bank,
            tokens=torch.zeros(1, 2, dtype=torch.long),
            pos_argmax=torch.zeros(1, dtype=torch.long),
            seed_layer=1, seed_kind="attn", seed_latent_idx=0,
            target_act=0.0,
            rounds=2, per_round_k=1, certificate_tol=0.0,
        )
        assert FeatureID(0, "mlp", 1) in positives  # d, round 0
        assert FeatureID(0, "attn", 0) in positives  # u, round 1
        assert result.round_of[((0, "mlp"), 1)] == 0
        assert result.rounds_used == 2

    def test_iteration_recruits_the_chain(self):
        """d first; then, with d restored CONNECTED, u lights up. Under
        pinned restoration u would stay at zero forever."""
        score_fn, masks, site_a, site_b, _ = self._setup()
        result = run_iterative_selection(score_fn, masks=masks, rounds=2, per_round_k=1)
        assert result.round_of[(site_b, 1)] == 0
        assert result.round_of[(site_a, 0)] == 1
        assert result.positives[(site_a, 0)] > 0
        # Restoring d improved the restored-state metric (closure progress).
        assert result.metric_trajectory[1] > result.metric_trajectory[0]
