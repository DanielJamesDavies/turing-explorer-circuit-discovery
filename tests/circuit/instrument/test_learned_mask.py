"""Engine tests for the learned continuous mask (instrument/learned_mask.py).

Controlled geometry so every expectation is hand-computable:
  d_model = d_sae = 8, one upstream site (0, "attn"), seed at (1, "resid").
  Encoder = identity (relu), so latent i mirrors input dim i.
  Decoder = identity EXCEPT column 5 = -e0: latent 5 is a genuine suppressor
  of the seed direction (w_seed = e0). Latent 3 decodes to e3: present but
  irrelevant to the seed.

The stub inference chains patcher.transform through the two sites, so mask
edits at the upstream site propagate to the seed tap — a minimal residual
pipe. All CPU, no model.
"""
import pytest
import torch

from circuit.instrument.learned_mask import (
    LearnedMaskPatcher, OBJECTIVES, run_learned_mask)
from circuit.types.feature_id import FeatureID

D = 8
SITE = (0, "attn")
SEED_LAYER, SEED_KIND, SEED_LATENT = 1, "resid", 0


class _SAE:
    """Identity encoder (relu, full top-k); decoder = I with column 5 = -e0."""

    def __init__(self):
        self.encoder = type("E", (), {})()
        self.encoder.weight = torch.eye(D)
        W = torch.eye(D)
        W[:, 5] = 0.0
        W[0, 5] = -1.0
        self._W_dec = W

    def _get_bias_eff(self):
        return torch.zeros(D)


class _Bank:
    kinds = ["attn", "mlp", "resid"]
    d_sae = D
    device = torch.device("cpu")

    def __init__(self):
        sae = _SAE()
        self.saes = {k: {0: sae, 1: sae} for k in self.kinds}
        self._sae = sae

    def encode(self, x, kind, layer):
        pre = torch.relu(x.float())
        acts, idx = pre.topk(D, dim=-1)
        return acts, idx

    def decode(self, latents, kind, layer):
        return latents @ self._sae._W_dec.T


class _Inference:
    """tokens[0,0] selects the stored stream: 0 -> pos, 1 -> neg. Chains the
    patcher across (0,'attn') then the seed tap at (1,'resid')."""

    def __init__(self, x_pos, x_neg):
        self.streams = {0: x_pos, 1: x_neg}
        self._compiled = False

    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, patcher=None, **kwargs):
        x = self.streams[int(tokens[0, 0])][: tokens.shape[0]].clone()
        x = patcher.transform(0, "attn", x)
        patcher.transform(1, "resid", x)


def _setup(pos_dim0=2.0, pos_dim3=1.0, neg_lat5=1.0):
    B, T = 4, 2
    x_pos = torch.zeros(B, T, D)
    x_pos[:, :, 0] = pos_dim0          # drives the seed via latent 0
    x_pos[:, :, 3] = pos_dim3          # present but irrelevant (decodes to e3)
    x_neg = torch.zeros(B, T, D)
    x_neg[:, :, 5] = neg_lat5          # suppressor present on negctx
    bank = _Bank()
    inf = _Inference(x_pos, x_neg)
    pos_tokens = torch.zeros(B, T, dtype=torch.long)       # marker 0 -> pos
    neg_tokens = torch.ones(B, T, dtype=torch.long)        # marker 1 -> neg
    pos_argmax = torch.zeros(B, dtype=torch.long)
    return bank, inf, pos_tokens, neg_tokens, pos_argmax


def _run(objective, bank, inf, pt, nt, pa, **kw):
    args = dict(objective=objective, sites=[SITE], seed_layer=SEED_LAYER,
                seed_kind=SEED_KIND, seed_latent_idx=SEED_LATENT,
                pos_tokens=pt, pos_argmax=pa, steps=120, lr=0.3,
                l1_lambda=0.05, keep_threshold=0.5, batch_size=4,
                holdout_frac=0.25, log_every=0)
    args.update(kw)
    return run_learned_mask(inf, bank, **args)


class TestValidation:
    def test_unknown_objective_raises(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="objective"):
            _run("posctx", bank, inf, pt, nt, pa)

    def test_negctx_requires_neg_tokens(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="neg_tokens"):
            _run("negctx", bank, inf, pt, nt, pa, target_act=1.0)

    def test_negctx_requires_target(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="target_act"):
            _run("negctx", bank, inf, pt, nt, pa, neg_tokens=nt)


class TestPatcher:
    def test_high_theta_is_near_identity(self):
        bank, inf, pt, nt, pa = _setup()
        thetas = {SITE: torch.full((D,), 8.0)}       # sigmoid(8) ~ 0.99966
        p = LearnedMaskPatcher(bank, thetas, SEED_LAYER, SEED_KIND,
                               torch.eye(D)[0], torch.tensor(0.0))
        x = inf.streams[0].clone()
        out = p.transform(0, "attn", x)
        assert torch.allclose(out, x, atol=5e-3)

    def test_seed_site_taps_and_passes_through(self):
        bank, inf, pt, nt, pa = _setup()
        p = LearnedMaskPatcher(bank, {}, SEED_LAYER, SEED_KIND,
                               torch.eye(D)[0], torch.tensor(0.0))
        x = inf.streams[0].clone()
        out = p.transform(SEED_LAYER, SEED_KIND, x)
        assert torch.equal(out, x)
        assert p.seed_pre is not None
        assert p.seed_pre[0, 0] == pytest.approx(2.0, abs=1e-5)


class TestPosObjective:
    def test_loss_decreases_and_selection_is_correct(self):
        bank, inf, pt, nt, pa = _setup()
        scores, prov = _run("pos", bank, inf, pt, nt, pa)
        assert prov["loss_final"] < prov["loss_initial"]
        kept = set(scores)
        assert FeatureID(0, "attn", 0) in kept        # the driver survives
        assert FeatureID(0, "attn", 3) not in kept    # irrelevant is pruned
        assert all(v > 0 for v in scores.values())    # supports only

    def test_holdout_loss_reported(self):
        bank, inf, pt, nt, pa = _setup()
        _, prov = _run("pos", bank, inf, pt, nt, pa)
        assert prov["holdout_data_loss"] is not None
        assert prov["objective"] == "pos"


class TestNegctxObjective:
    def test_gate_opening_selects_the_suppressor_as_inhibitor(self):
        """Natural negctx pre-act = 1 - m5 (suppressor at value 1 against the
        error-preserved stream). Target 0.9 is reachable only by editing
        latent 5 down; the edit must be selected with a NEGATIVE score, and
        the driver latent (absent on negctx) must not appear."""
        bank, inf, pt, nt, pa = _setup()
        scores, prov = _run("negctx", bank, inf, pt, nt, pa,
                            neg_tokens=nt, target_act=0.9)
        assert FeatureID(0, "attn", 5) in scores
        assert scores[FeatureID(0, "attn", 5)] < 0    # delivered as inhibitor
        assert FeatureID(0, "attn", 0) not in scores
        assert prov["loss_final"] < prov["loss_initial"]


class TestContrastObjective:
    def test_beta_zero_matches_pos_selection(self):
        bank, inf, pt, nt, pa = _setup()
        s_pos, _ = _run("pos", bank, inf, pt, nt, pa)
        s_con, _ = _run("contrast", bank, inf, pt, nt, pa,
                        neg_tokens=nt, beta=0.0)
        assert set(s_pos) == set(s_con)

    def test_contrast_runs_and_keeps_driver(self):
        bank, inf, pt, nt, pa = _setup()
        scores, prov = _run("contrast", bank, inf, pt, nt, pa,
                            neg_tokens=nt, beta=1.0)
        assert FeatureID(0, "attn", 0) in scores
        assert prov["objective"] == "contrast"


class TestGradientAccumulation:
    """The deep-site guard shrinks the micro-batch but preserves the step
    gradient via accumulation — deep and shallow seeds share one optimisation
    regime. Guarded runs must therefore match unguarded runs on the same data
    (identical selection; provenance records the split)."""

    def test_guarded_matches_unguarded_selection(self):
        bank, inf, pt, nt, pa = _setup()
        common = dict(objective="pos", sites=[SITE], seed_layer=SEED_LAYER,
                      seed_kind=SEED_KIND, seed_latent_idx=SEED_LATENT,
                      pos_tokens=pt, pos_argmax=pa, steps=60, lr=0.3,
                      l1_lambda=0.05, keep_threshold=0.5, batch_size=4,
                      holdout_frac=0.0, log_every=0)
        s_plain, p_plain = run_learned_mask(inf, bank, **common)
        s_accum, p_accum = run_learned_mask(
            inf, bank, **common, deep_site_threshold=0, deep_batch_size=2)
        assert p_plain["accum_chunks"] == 1
        assert p_accum["accum_chunks"] == 2
        assert p_accum["micro_batch"] == 2
        assert p_accum["batch_size_used"] == 4     # effective batch preserved
        assert set(s_plain) == set(s_accum)
        for fid in s_plain:
            assert s_plain[fid] == pytest.approx(s_accum[fid], abs=0.05)

    def test_guard_inactive_below_threshold(self):
        bank, inf, pt, nt, pa = _setup()
        _, prov = run_learned_mask(
            inf, bank, objective="pos", sites=[SITE], seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND, seed_latent_idx=SEED_LATENT,
            pos_tokens=pt, pos_argmax=pa, steps=5, lr=0.3, l1_lambda=0.05,
            keep_threshold=0.5, batch_size=4, holdout_frac=0.0, log_every=0,
            deep_site_threshold=21, deep_batch_size=2)
        assert prov["micro_batch"] == 4
        assert prov["accum_chunks"] == 1
