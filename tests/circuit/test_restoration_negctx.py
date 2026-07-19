"""
Tests for attribution_mode="restoration_negctx" (the restoration loop on
ig_negctx's negctx -> posctx-target trajectory) and the "fused" neg_mode's
config surface.

Part 1  Instrument target_inject arithmetic (MockSAEBank): mask=0 -> exact
        identity (live connected negctx state — the on-manifold alpha=0,
        OPPOSITE of floor_restore's mask=0); mask=1 -> decode(targets) +
        residual (the fully-injected state); partial masks exact; deltas
        stored as target - live; constructor guards.
Part 2  Grad-pass semantics on a tiny chain bank: round 1 scores BOTH chain
        latents (unrestored latents stay connected, so gradients thread the
        whole live chain — unlike floor_restore, where floors sever paths);
        after injecting the upstream latent, the downstream latent's LIVE
        DELTA collapses to ~0 (the stream already carries its target) — the
        adaptive-admission semantics this mode exists for. Objective knob:
        "drive" backward, gap-form metric either way.
Part 3  Loop + driver: certificate early-stop once the injected state makes
        the seed hit its posctx target; run_negctx_restoration_selection
        end-to-end (FeatureID roles, provenance result); guards.
Part 4  Config surface: restoration_negctx cf-only; "fused" neg_mode
        accepted on both gradient configs.
"""

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from circuit.instrument.restoration import (
    MaskedRestorationInstrument,
    _restoration_grad_pass,
    run_negctx_restoration_selection,
)
from sae.dense import sparse_topk_to_dense

from tests.conftest import D_MODEL, D_SAE

B, T = 2, 4
SITE = (0, "attn")
SEED = (1, "resid")
SITE_A, SITE_B = (0, "attn"), (0, "mlp")


def _cached(bank, x, site):
    ta, ti = bank.encode(x, site[1], site[0])
    dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
    residual = (x - bank.decode(dense, site[1], site[0])).detach()
    return residual, dense


def _inject_instrument(bank, residuals, masks, targets):
    return MaskedRestorationInstrument(
        bank, {SITE}, residuals, {}, masks,
        SEED[0], SEED[1], torch.randn(D_MODEL), torch.tensor(0.0),
        mode="target_inject", inject_targets=targets,
    )


class TestTargetInjectInstrument:
    def test_mask_zero_is_identity(self, mock_sae_bank):
        """Nothing restored = the live negctx state exactly (on-manifold),
        the opposite of floor_restore's mask=0 (floor + residual)."""
        x = torch.randn(B, T, D_MODEL)
        residual, _ = _cached(mock_sae_bank, x, SITE)
        inst = _inject_instrument(
            mock_sae_bank, {SITE: residual},
            {SITE: torch.zeros(D_SAE, dtype=torch.bool)},
            {SITE: torch.rand(D_SAE)},
        )
        out = inst.transform(*SITE, x.clone())
        assert torch.allclose(out, x, atol=1e-5)

    def test_mask_one_is_injected_targets(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, dense = _cached(mock_sae_bank, x, SITE)
        target = torch.rand(D_SAE)
        inst = _inject_instrument(
            mock_sae_bank, {SITE: residual},
            {SITE: torch.ones(D_SAE, dtype=torch.bool)},
            {SITE: target},
        )
        out = inst.transform(*SITE, x.clone())
        expected = mock_sae_bank.decode(target.expand_as(dense), "attn", 0) + residual
        assert torch.allclose(out, expected, atol=1e-5)

    def test_partial_mask_mixes_live_and_targets(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, dense = _cached(mock_sae_bank, x, SITE)
        target = torch.rand(D_SAE)
        mask = torch.zeros(D_SAE, dtype=torch.bool)
        mask[2] = True
        mask[5] = True
        inst = _inject_instrument(
            mock_sae_bank, {SITE: residual}, {SITE: mask}, {SITE: target},
        )
        out = inst.transform(*SITE, x.clone())
        code = dense.clone()  # live everywhere...
        code[:, :, [2, 5]] = target[[2, 5]]  # ...except the injected dims
        expected = mock_sae_bank.decode(code, "attn", 0) + residual
        assert torch.allclose(out, expected, atol=1e-5)

    def test_live_is_stashed_for_scorer(self, mock_sae_bank):
        """The scorer reads instrument.live (f_conn) to weigh the activator
        move (posctx - live) against the inhibitor move (0 - live)."""
        x = torch.randn(B, T, D_MODEL)
        residual, dense = _cached(mock_sae_bank, x, SITE)
        inst = _inject_instrument(
            mock_sae_bank, {SITE: residual},
            {SITE: torch.zeros(D_SAE, dtype=torch.bool)},
            {SITE: torch.rand(D_SAE)},
        )
        inst.transform(*SITE, x.clone())
        assert torch.allclose(inst.live[SITE], dense, atol=1e-5)

    def test_unrestored_leaf_receives_gradient(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        residual, _ = _cached(mock_sae_bank, x, SITE)
        inst = _inject_instrument(
            mock_sae_bank, {SITE: residual},
            {SITE: torch.zeros(D_SAE, dtype=torch.bool)},
            {SITE: torch.rand(D_SAE)},
        )
        out = inst.transform(*SITE, x.clone())
        grad = torch.autograd.grad(out.sum(), inst.leaves[SITE])[0]
        assert grad.abs().sum() > 0

    def test_target_inject_without_targets_raises(self, mock_sae_bank):
        with pytest.raises(ValueError, match="inject_targets"):
            MaskedRestorationInstrument(
                mock_sae_bank, {SITE}, {}, {}, {},
                SEED[0], SEED[1], torch.randn(D_MODEL), torch.tensor(0.0),
                mode="target_inject",
            )

    def test_target_inject_alpha_raises(self, mock_sae_bank):
        with pytest.raises(ValueError, match="point scorer"):
            MaskedRestorationInstrument(
                mock_sae_bank, {SITE}, {}, {}, {},
                SEED[0], SEED[1], torch.randn(D_MODEL), torch.tensor(0.0),
                mode="target_inject", inject_targets={SITE: torch.rand(D_SAE)},
                alpha=0.5,
            )

    def test_unknown_mode_raises(self, mock_sae_bank):
        with pytest.raises(ValueError, match="mode"):
            MaskedRestorationInstrument(
                mock_sae_bank, {SITE}, {}, {}, {},
                SEED[0], SEED[1], torch.randn(D_MODEL), torch.tensor(0.0),
                mode="bogus",
            )


# ---------------------------------------------------------------------------
# Tiny chain bank for the trajectory semantics: latent u (site A, idx 0)
# writes dim 1; latent d (site B, idx 1) reads dim 1 and writes dim 2; the
# seed reads dim 2. The negctx input carries small positive values so the
# relus are live (relu'(0)=0 would sever chain gradients at exactly zero).
# Targets are chosen self-consistently: B's target equals the live value B
# takes once A is injected, so the round-2 delta for B is EXACTLY zero.
# ---------------------------------------------------------------------------

C = 2.0
EPS_A, EPS_B = 0.1, 0.05
TARGET_A = torch.tensor([C, 0.0, 0.0])
# residual_A[1] = -EPS_B (see _neg_setup): injected stream gives
# x_B[1] = C - EPS_B, so B's live value under injection is C - EPS_B.
TARGET_B = torch.tensor([0.0, C - EPS_B, 0.0])
# Seed's posctx target: B injected at C-EPS_B plus residual_B[2] = -EPS_B*?
# (computed in _neg_setup; the loop's certificate uses this).


class TinyNegBank:
    d_sae = 3
    kinds = ["attn", "mlp"]

    def __init__(self):
        w_enc_a = torch.zeros(3, 3); w_enc_a[0, 0] = 1.0   # A0 = x[0]
        w_dec_a = torch.zeros(3, 3); w_dec_a[1, 0] = 1.0   # A0 -> dim 1
        w_enc_b = torch.zeros(3, 3); w_enc_b[1, 1] = 1.0   # B1 = x[1]
        w_dec_b = torch.zeros(3, 3); w_dec_b[2, 1] = 1.0   # B1 -> dim 2
        self.weights = {
            (0, "attn"): (w_enc_a, w_dec_a),
            (0, "mlp"): (w_enc_b, w_dec_b),
        }
        eye = torch.eye(3)
        self.saes = {
            "attn": {1: SimpleNamespace(
                encoder=SimpleNamespace(weight=eye),
                _get_bias_eff=lambda: torch.zeros(3),
            )},
        }

    def encode(self, x, kind, layer):
        w_enc, _ = self.weights[(layer, kind)]
        pre = torch.relu(x @ w_enc.T)
        return pre.topk(2, dim=-1)

    def decode(self, latents, kind, layer):
        _, w_dec = self.weights[(layer, kind)]
        return latents @ w_dec.T


def _neg_setup():
    bank = TinyNegBank()
    x_neg = torch.zeros(1, 1, 3)
    x_neg[0, 0, 0] = EPS_A
    x_neg[0, 0, 1] = EPS_B

    residuals = {}
    for site in (SITE_A, SITE_B):
        ta, ti = bank.encode(x_neg, site[1], site[0])
        dense = sparse_topk_to_dense(ta, ti, 3, dtype=x_neg.dtype)
        residuals[site] = (x_neg - bank.decode(dense, site[1], site[0])).detach()

    def forward_fn(tokens, patcher=None, **kwargs):
        out_a = patcher.transform(0, "attn", x_neg.clone())
        out_b = patcher.transform(0, "mlp", out_a)
        patcher.transform(1, "attn", out_b)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn

    # Seed target: with B1 injected at C-EPS_B, seed pre-act =
    # (C-EPS_B) + residual_B[2].
    target_act = float((C - EPS_B) + residuals[SITE_B][0, 0, 2])

    common = dict(
        tokens=torch.zeros(1, 1, dtype=torch.long),
        substitute_sites={SITE_A, SITE_B},
        residuals=residuals,
        site_floors={},
        natural_dense={},
        seed_layer=1, seed_kind="attn",
        w_seed=torch.tensor([0.0, 0.0, 1.0]), b_seed=torch.tensor(0.0),
        pos_argmax=torch.zeros(1, dtype=torch.long),
        target_act=target_act,
        mode="target_inject",
        inject_targets={SITE_A: TARGET_A.clone(), SITE_B: TARGET_B.clone()},
        posctx_targets={SITE_A: TARGET_A.clone(), SITE_B: TARGET_B.clone()},
    )
    return bank, inf, common, residuals


class TestNegctxTrajectorySemantics:
    def test_round_one_scores_both_chain_latents(self):
        """Unrestored latents stay CONNECTED, so round 1's gradient threads
        the whole live chain — u scores nonzero immediately (in
        floor_restore it cannot, because the floor severs its path)."""
        bank, inf, common, _ = _neg_setup()
        masks = {SITE_A: torch.zeros(3, dtype=torch.bool),
                 SITE_B: torch.zeros(3, dtype=torch.bool)}
        scores, _, _ = _restoration_grad_pass(inf, bank, masks=masks, **common)
        assert scores[SITE_A][0] > 0.01
        assert scores[SITE_B][1] > 0.01

    def test_injected_upstream_collapses_downstream_delta(self):
        """THE load-bearing semantics: after injecting u, the modified
        stream already carries d's target value, so d's live delta — and
        with it d's score — collapses to ~0. Admission adaptively moves off
        latents the injection has satisfied."""
        bank, inf, common, _ = _neg_setup()
        mask_a = torch.zeros(3, dtype=torch.bool)
        mask_a[0] = True  # u injected at its posctx target
        masks = {SITE_A: mask_a, SITE_B: torch.zeros(3, dtype=torch.bool)}
        scores, _, _ = _restoration_grad_pass(inf, bank, masks=masks, **common)
        assert scores[SITE_A][0] == 0.0        # restored: zeroed
        assert abs(scores[SITE_B][1]) < 1e-5   # delta = target - live = 0

    def test_drive_objective_scores_with_gap_metric(self):
        """objective picks the BACKWARD scalar only; the returned metric is
        the gap form either way (certificate semantics objective-invariant)."""
        bank, inf, common, _ = _neg_setup()
        masks = {SITE_A: torch.zeros(3, dtype=torch.bool),
                 SITE_B: torch.zeros(3, dtype=torch.bool)}
        s_gap, m_gap, _ = _restoration_grad_pass(
            inf, bank, masks={k: v.clone() for k, v in masks.items()},
            **{**common, "objective": "gap"},
        )
        s_drive, m_drive, _ = _restoration_grad_pass(
            inf, bank, masks={k: v.clone() for k, v in masks.items()},
            **{**common, "objective": "drive"},
        )
        assert m_gap == pytest.approx(m_drive)      # same (gap) metric
        assert s_drive[SITE_B][1] > 0.01            # drive grads still score
        # gap backward carries the 2*(target - peak) factor, drive does not:
        # same sign, different magnitude.
        assert s_gap[SITE_B][1] != pytest.approx(float(s_drive[SITE_B][1]))


class RoleBank:
    """Single site, one latent of each kind wired straight to the seed's dim:
    latent 0 EXCITES (decodes +dim2), latent 1 INHIBITS (decodes -dim2),
    latent 2 is IRRELEVANT (decodes dim0, which the seed never reads)."""
    d_sae = 3
    kinds = ["attn"]

    def __init__(self):
        self.w_enc = torch.eye(3)
        w_dec = torch.zeros(3, 3)
        w_dec[2, 0] = 1.0     # activator -> +dim2
        w_dec[2, 1] = -1.0    # inhibitor -> -dim2
        w_dec[0, 2] = 1.0     # irrelevant -> dim0
        self.w_dec = w_dec

    def encode(self, x, kind, layer):
        pre = torch.relu(x @ self.w_enc.T)
        return pre.topk(3, dim=-1)

    def decode(self, latents, kind, layer):
        return latents @ self.w_dec.T


def _role_setup():
    bank = RoleBank()
    site = (0, "attn")
    x_neg = torch.zeros(1, 1, 3)
    x_neg[0, 0, 1] = 2.0   # inhibitor present on negctx; activator/irrelevant absent
    ta, ti = bank.encode(x_neg, "attn", 0)
    dense = sparse_topk_to_dense(ta, ti, 3, dtype=x_neg.dtype)
    residual = (x_neg - bank.decode(dense, "attn", 0)).detach()

    def forward_fn(tokens, patcher=None, **kwargs):
        out = patcher.transform(0, "attn", x_neg.clone())
        patcher.transform(1, "attn", out)   # seed site reads the modified stream

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    # activator posctx target high; inhibitor partial (present but lower on
    # posctx, so its removal-to-0 beats its posctx-injection); irrelevant 0.
    posctx = torch.tensor([1.0, 0.5, 0.0])
    common = dict(
        tokens=torch.zeros(1, 1, dtype=torch.long),
        substitute_sites={site},
        residuals={site: residual},
        site_floors={}, natural_dense={},
        seed_layer=1, seed_kind="attn",
        w_seed=torch.tensor([0.0, 0.0, 1.0]), b_seed=torch.tensor(0.0),
        pos_argmax=torch.zeros(1, dtype=torch.long),
        target_act=3.0,
        mode="target_inject",
        inject_targets={site: posctx.clone()},
        posctx_targets={site: posctx.clone()},
    )
    return bank, inf, common, site


class TestRoleDirectionalScoring:
    """THE fix: score each candidate by its best HELPING move (raise to posctx
    OR remove to 0), sign it by role, and never admit a latent that helps in
    neither direction — so we stop injecting seed-suppressing latents."""

    def test_activator_positive_inhibitor_negative_irrelevant_zero(self):
        bank, inf, common, site = _role_setup()
        masks = {site: torch.zeros(3, dtype=torch.bool)}
        scores, _, _ = _restoration_grad_pass(inf, bank, masks=masks, **common)
        s = scores[site]
        assert s[0] > 0.01     # activator: raising toward posctx helps -> +
        assert s[1] < -0.01    # inhibitor: removing to 0 helps -> - (role sign)
        assert abs(s[2]) < 1e-6  # irrelevant: no helping move -> not selectable

    def test_inhibitor_role_pins_to_zero_not_posctx(self):
        """The driver stamps the inhibitor's inject value to 0 (the eval's
        semantics), never its nonzero posctx target."""
        bank, inf, common, site = _role_setup()
        inject_values = {site: torch.zeros(3)}
        # Emulate one admission: score, then apply the driver's role stamp.
        masks = {site: torch.zeros(3, dtype=torch.bool)}
        scores, _, _ = _restoration_grad_pass(inf, bank, masks=masks, **common)
        for latent, value in enumerate(scores[site].tolist()):
            if value == 0:
                continue
            inject_values[site][latent] = (
                float(common["posctx_targets"][site][latent]) if value >= 0 else 0.0
            )
        assert inject_values[site][0] == pytest.approx(1.0)   # activator -> posctx
        assert inject_values[site][1] == 0.0                  # inhibitor -> 0, NOT 0.5


class NoHelpBank:
    """latent 0 EXCITES and is absent on negctx (clean activator); latent 1
    EXCITES but is PRESENT on negctx with a posctx target BELOW its live value
    — so neither raising it to posctx nor removing it to 0 helps the seed (no
    helping move), yet it has a large |posctx-injection effect|; latent 2 is
    irrelevant."""
    d_sae = 3
    kinds = ["attn"]

    def __init__(self):
        self.w_enc = torch.eye(3)
        w_dec = torch.zeros(3, 3)
        w_dec[2, 0] = 1.0     # activator -> +dim2
        w_dec[2, 1] = 1.0     # present-excitatory -> +dim2
        w_dec[0, 2] = 1.0     # irrelevant -> dim0
        self.w_dec = w_dec

    def encode(self, x, kind, layer):
        return torch.relu(x @ self.w_enc.T).topk(3, dim=-1)

    def decode(self, latents, kind, layer):
        return latents @ self.w_dec.T


def _nohelp_setup():
    bank = NoHelpBank()
    site = (0, "attn")
    x_neg = torch.zeros(1, 1, 3)
    x_neg[0, 0, 1] = 2.0
    ta, ti = bank.encode(x_neg, "attn", 0)
    dense = sparse_topk_to_dense(ta, ti, 3, dtype=x_neg.dtype)
    residual = (x_neg - bank.decode(dense, "attn", 0)).detach()

    def forward_fn(tokens, patcher=None, **kwargs):
        out = patcher.transform(0, "attn", x_neg.clone())
        patcher.transform(1, "attn", out)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    posctx = torch.tensor([1.0, 0.5, 0.0])   # latent1 target 0.5 < live 2.0
    common = dict(
        tokens=torch.zeros(1, 1, dtype=torch.long),
        substitute_sites={site},
        residuals={site: residual},
        site_floors={}, natural_dense={},
        seed_layer=1, seed_kind="attn",
        w_seed=torch.tensor([0.0, 0.0, 1.0]), b_seed=torch.tensor(0.0),
        pos_argmax=torch.zeros(1, dtype=torch.long),
        target_act=5.0,
        mode="target_inject",
        inject_targets={site: posctx.clone()},
        posctx_targets={site: posctx.clone()},
    )
    return bank, inf, common, site


class TestInjectModes:
    """The no-helping-move latent (idx 1) is exactly where the three modes
    diverge: directional drops it, both_sign/posctx keep it (free0 members)."""

    def _score1(self, inject_mode):
        bank, inf, common, site = _nohelp_setup()
        masks = {site: torch.zeros(3, dtype=torch.bool)}
        scores, _, _ = _restoration_grad_pass(
            inf, bank, masks=masks, inject_mode=inject_mode, **common)
        return scores[site]

    def test_directional_drops_no_helping_move(self):
        s = self._score1("directional")
        assert s[0] > 0.01          # clean activator kept
        assert abs(s[1]) < 1e-6     # no helping move -> dropped

    def test_both_sign_keeps_no_helping_move(self):
        s = self._score1("both_sign")
        assert s[0] > 0.01          # activator kept
        assert abs(s[1]) > 0.01     # kept by |posctx effect| (free0 membership)

    def test_posctx_keeps_and_signs_by_raw_effect(self):
        s = self._score1("posctx")
        assert s[0] > 0.01          # activator: effect positive
        # latent1 posctx effect = g*(0.5 - 2.0) < 0 -> inhibitor-role by sign.
        assert s[1] < -0.01

    def test_genuine_inhibitor_pins_zero_in_both_sign_posctx_in_posctx(self):
        """A GENUINE inhibitor (present, removing helps) pins to 0 under
        both_sign (matching the cf eval) but to its posctx target under the
        original 'posctx' mode — the exact difference the fix is about. Uses
        _role_setup, whose latent 1 has b_inh > b_act (removal is the win)."""
        posctx = torch.tensor([1.0, 0.5, 0.0])
        for inject_mode, expected1 in (("both_sign", 0.0), ("posctx", 0.5)):
            bank, inf, common, site = _role_setup()
            scores, _, _ = _restoration_grad_pass(
                inf, bank, masks={site: torch.zeros(3, dtype=torch.bool)},
                inject_mode=inject_mode, **common)
            inject_values = {site: torch.zeros(3)}
            for latent, value in enumerate(scores[site].tolist()):
                if value == 0:
                    continue
                if inject_mode == "posctx":
                    inject_values[site][latent] = float(posctx[latent])
                else:
                    inject_values[site][latent] = (
                        float(posctx[latent]) if value >= 0 else 0.0)
            assert inject_values[site][1] == pytest.approx(expected1)


class TestNegctxRestorationDriver:
    def _run(self, monkeypatch, **overrides):
        bank, inf, common, residuals = _neg_setup()
        monkeypatch.setattr(
            "circuit.instrument.ig_baseline.collect_natural_codes",
            lambda *a, **k: ({}, residuals),
        )
        kwargs = dict(
            neg_tokens=common["tokens"],
            neg_anchor=common["pos_argmax"],
            inject_targets=common["inject_targets"],
            sites={SITE_A, SITE_B},
            seed_layer=1, seed_kind="attn", seed_latent_idx=2,
            target_act=common["target_act"],
            rounds=4, per_round_k=2, certificate_tol=0.05,
        )
        kwargs.update(overrides)
        return run_negctx_restoration_selection(inf, bank, **kwargs)

    def test_certificate_stops_once_injection_hits_target(self, monkeypatch):
        positives, negatives, result = self._run(monkeypatch)
        # Round 1 injects u and d (per_round_k=2); round 2's metric is the
        # injected state where the seed sits AT its posctx target -> stop.
        assert result.stopped_early
        assert result.metric_trajectory[-1] == pytest.approx(0.0, abs=1e-6)
        from circuit.types.feature_id import FeatureID
        assert FeatureID(0, "attn", 0) in positives
        assert FeatureID(0, "mlp", 1) in positives
        assert not negatives

    def test_no_sites_returns_empty(self, monkeypatch):
        positives, negatives, result = self._run(monkeypatch, sites=set())
        assert positives == {} and negatives == {} and result is None

    def test_bad_round_select_raises(self, monkeypatch):
        with pytest.raises(ValueError, match="round_select"):
            self._run(monkeypatch, round_select="bogus")

    def test_bad_objective_raises(self, monkeypatch):
        with pytest.raises(ValueError, match="objective"):
            self._run(monkeypatch, objective="bogus")


class TestConfigSurface:
    def test_cf_accepts_restoration_negctx(self):
        from config import CounterfactualGradientConfig
        cfg = CounterfactualGradientConfig(attribution_mode="restoration_negctx")
        assert cfg.attribution_mode == "restoration_negctx"

    def test_abl_rejects_restoration_negctx(self):
        from config import AblationGradientConfig
        with pytest.raises(ValueError):
            AblationGradientConfig(attribution_mode="restoration_negctx")

    def test_both_accept_fused_neg_mode(self):
        from config import AblationGradientConfig, CounterfactualGradientConfig
        assert CounterfactualGradientConfig(neg_mode="fused").neg_mode == "fused"
        assert AblationGradientConfig(neg_mode="fused").neg_mode == "fused"

    def test_restoration_negctx_mode_validates(self):
        from config import CounterfactualGradientConfig
        assert CounterfactualGradientConfig().restoration_negctx_mode == "posctx"
        for m in ("posctx", "directional", "both_sign"):
            assert CounterfactualGradientConfig(restoration_negctx_mode=m).restoration_negctx_mode == m
        with pytest.raises(ValueError):
            CounterfactualGradientConfig(restoration_negctx_mode="bogus")
