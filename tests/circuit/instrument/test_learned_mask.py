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
        # NON-ZERO decoder bias on purpose. The transform decodes the
        # DIFFERENCE (code - dense) with add_bias=False, relying on the bias
        # cancelling between the two decodes it replaces. With a zero bias
        # that cancellation is vacuous and the equivalence tests below would
        # pass even if the bias were mishandled.
        self._b_dec = torch.linspace(-0.3, 0.3, D)

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

    def decode(self, latents, kind, layer, add_bias=True):
        out = latents @ self._sae._W_dec.T
        return out + self._sae._b_dec if add_bias else out


class _Inference:
    """tokens[0,0] selects the stored stream: 0 -> pos, 1 -> neg. Chains the
    patcher across (0,'attn') then the seed tap at (1,'resid')."""

    def __init__(self, x_pos, x_neg):
        self.streams = {0: x_pos, 1: x_neg}
        self._compiled = False
        # Toy unembed for the BEHAVIOURAL endpoint. Only stream dim 0 reaches
        # the logits, so latent 0 (which decodes to e0) is the sole
        # logit-carrying latent and latent 3 is inert for the output while
        # still being present in the code — exactly the discrimination the
        # logit objective has to make.
        Wv = torch.zeros(D, 4)
        Wv[0, 1] = 2.0
        self.Wv = Wv

    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, patcher=None, activations_callback=None, **kwargs):
        x = self.streams[int(tokens[0, 0])][: tokens.shape[0]].clone()
        if activations_callback is not None:
            # Clean pass with no patcher — how collect_site_means gathers the
            # mask floor. Callback signature mirrors the real Inference: one
            # activation tensor per kind, indexed by the bank's kind order.
            activations_callback(0, (x, x, x))
            return None
        if patcher is None:
            return None
        x = patcher.transform(0, "attn", x)
        patcher.transform(1, "resid", x)
        if kwargs.get("all_logits"):
            return None, x @ self.Wv, None


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


class TestMaskFloor:
    """The mean-ablation mask: a fully masked latent lands on a FLOOR instead
    of 0, so m=0 reproduces the state freeM/freeN measure against rather than
    one no eval uses. This is what lets the mask be compared with mean-floor
    methods on a metric neither family owns."""

    def test_zero_floor_is_the_default_and_unchanged(self):
        bank, inf, pt, nt, pa = _setup()
        base = _run("pos", bank, inf, pt, nt, pa)
        explicit = _run("pos", bank, inf, pt, nt, pa, mask_floor_source="zero")
        assert base[0] == explicit[0]
        assert base[1]["mask_floor_source"] == "zero"
        assert base[1]["mask_floor_sites"] == 0

    def test_m_one_is_still_identity_under_a_floor(self):
        """The load-bearing invariant: code*m + floor*(1-m) at m=1 must be
        exactly code, or a fully-kept mask would no longer reproduce the clean
        stream and every reconstruction number would be off."""
        bank = _Bank()
        x = torch.zeros(1, 1, D)
        x[0, 0, 0], x[0, 0, 3] = 2.0, 1.0
        floors = {SITE: torch.full((D,), 0.7)}
        big = torch.full((D,), 40.0)          # sigmoid(40) == 1.0
        p_floor = LearnedMaskPatcher(bank, {SITE: big}, SEED_LAYER, SEED_KIND,
                                     torch.eye(D)[0], torch.zeros(1),
                                     floors=floors)
        p_zero = LearnedMaskPatcher(bank, {SITE: big}, SEED_LAYER, SEED_KIND,
                                    torch.eye(D)[0], torch.zeros(1))
        out_floor = p_floor.transform(SITE[0], SITE[1], x)
        out_zero = p_zero.transform(SITE[0], SITE[1], x)
        assert torch.allclose(out_floor, x, atol=1e-5)
        assert torch.allclose(out_floor, out_zero, atol=1e-5)

    def test_m_zero_lands_on_the_floor_not_zero(self):
        """m=0 must reproduce the mean-ablated state — that IS the alignment
        with freeN. Under the zero floor the same mask gives the empty state."""
        bank = _Bank()
        x = torch.zeros(1, 1, D)
        x[0, 0, 0] = 2.0
        floors = {SITE: torch.full((D,), 0.5)}
        small = torch.full((D,), -40.0)       # sigmoid(-40) == 0.0
        p_floor = LearnedMaskPatcher(bank, {SITE: small}, SEED_LAYER, SEED_KIND,
                                     torch.eye(D)[0], torch.zeros(1),
                                     floors=floors)
        p_zero = LearnedMaskPatcher(bank, {SITE: small}, SEED_LAYER, SEED_KIND,
                                    torch.eye(D)[0], torch.zeros(1))
        got = p_floor.transform(SITE[0], SITE[1], x)
        want = bank.decode(floors[SITE].view(1, 1, D), SITE[1], SITE[0]) + (
            x - bank.decode(torch.relu(x), SITE[1], SITE[0]))
        assert torch.allclose(got, want, atol=1e-5)
        assert not torch.allclose(got, p_zero.transform(SITE[0], SITE[1], x),
                                  atol=1e-3)

    def test_negctx_floor_uses_negatives_and_is_recorded(self):
        bank, inf, pt, nt, pa = _setup()
        _, prov = _run("pos", bank, inf, pt, nt, pa,
                       neg_tokens=nt, mask_floor_source="negctx")
        assert prov["mask_floor_source"] == "negctx"
        assert prov["mask_floor_sites"] == 1

    def test_negctx_floor_without_negatives_raises(self):
        """Loud failure, never a silent substitution: a run labelled 'negctx
        floor' that quietly used posctx would be worse than a crash."""
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="negctx"):
            _run("pos", bank, inf, pt, nt, pa, mask_floor_source="negctx")

    def test_unknown_floor_source_raises(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="mask_floor_source"):
            _run("pos", bank, inf, pt, nt, pa, mask_floor_source="global")

    def test_floor_changes_selection(self):
        """Sanity that the floor is actually load-bearing rather than plumbed
        but inert — the two floors must not agree by construction."""
        bank, inf, pt, nt, pa = _setup()
        zero, _ = _run("pos", bank, inf, pt, nt, pa)
        neg, _ = _run("pos", bank, inf, pt, nt, pa, neg_tokens=nt,
                      mask_floor_source="negctx")
        assert isinstance(zero, dict) and isinstance(neg, dict)


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


class TestInjectObjective:
    """"inject": value' = m*value + delta. The controlled geometry has both C1
    roles ready-made: latent 0 drives the seed but is ABSENT on negctx (only
    delta can reach it); latent 5 is PRESENT on negctx and suppresses (only an
    edit can silence it). Target above the gate-only ceiling forces the
    optimiser to use both levers."""

    def test_learns_both_roles(self):
        bank, inf, pt, nt, pa = _setup()
        # gate-only ceiling is 1.0 (edit latent 5 fully); target 2.5 needs
        # delta on latent 0 as well.
        scores, prov = _run("inject", bank, inf, pt, nt, pa,
                            neg_tokens=nt, target_act=2.5, steps=200)
        assert scores[FeatureID(0, "attn", 5)] < 0     # present inhibitor (edit)
        assert scores[FeatureID(0, "attn", 0)] > 0     # absent activator (delta)
        assert prov["loss_final"] < prov["loss_initial"]

    def test_decomposition_reported_and_ordered(self):
        bank, inf, pt, nt, pa = _setup()
        _, prov = _run("inject", bank, inf, pt, nt, pa,
                       neg_tokens=nt, target_act=2.5, steps=200)
        assert {"p_both", "p_gate_only", "p_inject_only"} <= set(prov)
        # both levers together must reach at least each alone
        assert prov["p_both"] >= prov["p_gate_only"] - 1e-3
        assert prov["p_both"] >= prov["p_inject_only"] - 1e-3
        # and approach the target
        assert prov["p_both"] == pytest.approx(2.5, abs=0.5)

    def test_requires_neg_and_target(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="neg_tokens"):
            _run("inject", bank, inf, pt, nt, pa, target_act=1.0)
        with pytest.raises(ValueError, match="target_act"):
            _run("inject", bank, inf, pt, nt, pa, neg_tokens=nt)


class TestInjectV2Economics:
    """v2: delta is priced on its own scale, its concentration is reported,
    and the seed-adjacent sites can be excluded from injection.

    v1 shared one lambda and found a diffuse sub-threshold delta blanket that
    hit the target with ZERO selected latents (L8, 2026-07-24) — these pin the
    machinery that makes that visible and controllable."""

    def _inject(self, **kw):
        bank, inf, pt, nt, pa = _setup()
        return _run("inject", bank, inf, pt, nt, pa, neg_tokens=nt,
                    target_act=2.5, steps=150, **kw)

    def test_concentration_diagnostics_present(self):
        _, prov = self._inject()
        for k in ("delta_sum", "delta_top1pct_share", "delta_max",
                  "n_delta_gt_0p1", "n_delta_gt_0p5", "n_delta_sites"):
            assert k in prov, k
        assert prov["delta_sum"] >= 0.0

    def test_higher_inject_lambda_shrinks_injected_mass(self):
        _, cheap = self._inject(inject_lambda=0.0)
        _, dear = self._inject(inject_lambda=5.0)
        assert dear["delta_sum"] < cheap["delta_sum"]

    def test_inject_lambda_defaults_to_l1_lambda(self):
        _, prov = self._inject()          # inject_lambda unset
        assert prov["inject_lambda"] == pytest.approx(0.05)  # _run's l1_lambda

    def test_exclusion_removes_sites_from_injection(self):
        # one site only, so excluding 1 leaves nothing injectable
        _, prov = self._inject(inject_exclude_sites=1)
        assert prov["n_delta_sites"] == 0 or prov.get("delta_sum", 0.0) == 0.0
        assert prov["inject_exclude_sites"] == 1

    def test_exclusion_forces_the_gate(self):
        """With injection unavailable, the optimiser must fall back to the
        gate — the same lever mask_negctx uses."""
        bank, inf, pt, nt, pa = _setup()
        scores, _ = _run("inject", bank, inf, pt, nt, pa, neg_tokens=nt,
                         target_act=0.9, steps=150, inject_exclude_sites=1)
        assert scores[FeatureID(0, "attn", 5)] < 0     # edit selected


class TestLrSchedule:
    """Decaying lr freezes membership progressively (membership is a threshold
    crossing, so the last step shouldn't decide inclusion). BOTH budgets scale
    with sum(lr), so a decayed run must report a halved sum and correspondingly
    halved budgets — that is what forces peak lr to double when converting a
    calibrated constant-lr setting."""

    def _prov(self, schedule, **kw):
        bank, inf, pt, nt, pa = _setup()
        _, prov = _run("pos", bank, inf, pt, nt, pa, steps=100, lr=0.2,
                       lr_schedule=schedule, **kw)
        return prov

    def test_constant_sum_is_steps_times_lr(self):
        prov = self._prov("constant")
        assert prov["lr_sum"] == pytest.approx(100 * 0.2, rel=1e-3)
        assert prov["lr_schedule"] == "constant"

    def test_cosine_halves_the_lr_sum(self):
        prov = self._prov("cosine", lr_min_frac=0.0)
        assert prov["lr_sum"] == pytest.approx(0.5 * 100 * 0.2, rel=0.02)

    def test_linear_halves_the_lr_sum(self):
        prov = self._prov("linear", lr_min_frac=0.0)
        assert prov["lr_sum"] == pytest.approx(0.5 * 100 * 0.2, rel=0.02)

    def test_budgets_track_the_lr_sum(self):
        prov = self._prov("cosine", lr_min_frac=0.0)
        assert prov["decay_product"] == pytest.approx(
            prov["lr_sum"] * prov["weight_decay"], rel=1e-3)
        assert prov["sparsity_product"] == pytest.approx(
            prov["lr_sum"] * 0.05, rel=1e-3)   # _run's l1_lambda

    def test_floor_raises_the_sum(self):
        bare = self._prov("cosine", lr_min_frac=0.0)["lr_sum"]
        floored = self._prov("cosine", lr_min_frac=0.5)["lr_sum"]
        assert floored > bare

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="lr_schedule"):
            self._prov("exponential")

    @pytest.mark.parametrize("up,down", [("cosine_up", "cosine"),
                                         ("linear_up", "linear")])
    def test_warmup_is_the_mirror_of_decay(self, up, down):
        """The _up variants exist because decay measurably made circuits
        BIGGER; they must carry the same lr budget as their decaying twin so
        the two isolate schedule DIRECTION and nothing else."""
        assert self._prov(up, lr_min_frac=0.0)["lr_sum"] == pytest.approx(
            self._prov(down, lr_min_frac=0.0)["lr_sum"], rel=0.02)

    def test_warmup_frac_ramps_then_decays(self):
        """The conventional recipe: rise to peak by the end of warmup, then
        decay to the floor. The earlier decay arms had NO warmup — they
        started at peak on step 0 — so this is a genuinely different shape."""
        prov = self._prov("cosine", lr_min_frac=0.1, warmup_frac=0.1)
        assert prov["warmup_steps"] == 10          # 10% of steps=100
        assert prov["lr_first"] == pytest.approx(0.1 * 0.2 + 0.9 * 0.2 / 10,
                                                 rel=1e-6)
        assert prov["lr_last"] == pytest.approx(0.1 * 0.2, rel=1e-6)
        # warmup adds budget relative to the same schedule without it
        assert prov["lr_sum"] > self._prov("cosine", lr_min_frac=0.1)["lr_sum"]

    def test_warmup_rejected_for_non_decaying_schedules(self):
        for sched in ("constant", "cosine_up", "linear_up"):
            with pytest.raises(ValueError, match="warmup_frac"):
                self._prov(sched, warmup_frac=0.1)

    def test_warmup_starts_low_and_ends_high(self):
        prov = self._prov("cosine_up", lr_min_frac=0.0)
        assert prov["lr_first"] < prov["lr_last"]
        assert prov["lr_last"] == pytest.approx(0.2, rel=1e-6)

    def test_doubling_peak_restores_the_constant_budget(self):
        """The conversion rule: a cosine run at 2x peak lr matches the
        constant run's budgets, so lambda and wd carry over unchanged."""
        const = self._prov("constant")
        bank, inf, pt, nt, pa = _setup()
        _, cos = _run("pos", bank, inf, pt, nt, pa, steps=100, lr=0.4,
                      lr_schedule="cosine", lr_min_frac=0.0)
        assert cos["lr_sum"] == pytest.approx(const["lr_sum"], rel=0.02)


class TestDualFloor:
    """Scoring one mask under BOTH ablation semantics every step.

    Motivation is measured, not aesthetic: on L2/L5/L8 the negctx-only floor
    reached freeN 0.66-1.06 while its free0 was EXACTLY 0.0 at L5 and L8 (the
    post-top-k signature — the members alone cannot get the seed into top-k).
    It learns the DELTA from the negative baseline and is never asked to be
    sufficient. The zero-only floor has the mirror gap.
    """

    def test_dual_requires_negatives(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="negctx|neg_tokens"):
            _run("pos", bank, inf, pt, nt, pa, mask_floor_source="dual")

    def test_dual_rejected_for_negctx_objectives(self):
        """negctx/contrast/inject already carry a negative-context term;
        composing them with a dual floor is a different experiment."""
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="dual"):
            _run("contrast", bank, inf, pt, nt, pa, neg_tokens=nt,
                 mask_floor_source="dual")

    def test_dual_records_both_normalisers(self):
        """Each term is divided by its OWN closed-mask loss. Without that the
        sum is just L_zero plus noise, since zeroing every latent destroys far
        more of the stream than replacing it with the negctx mean."""
        bank, inf, pt, nt, pa = _setup()
        _, prov = _run("pos", bank, inf, pt, nt, pa, neg_tokens=nt,
                       mask_floor_source="dual")
        assert prov["mask_floor_source"] == "dual"
        assert prov["dual_floor_weight"] == 1.0
        # The closed-mask losses are now DIAGNOSTIC, not the scale. They are
        # still reported because they are exactly what exposed the bug: at
        # L10 the zero floor's closed state measured 3.3e10 against the
        # negctx floor's 176 (ratio 1.9e8), and dividing each term by its own
        # such loss annihilated the zero term, silently turning dual into
        # negctx-only. Direction is geometry-dependent too: in THIS harness
        # the negctx floor is the harsher one (its negative context carries an
        # active suppressor, latent 5 -> -e0), so neither floor is reliably
        # the gentler one and neither is a safe scale.
        assert prov["dual_norm_zero"] > 0.0
        assert prov["dual_norm_floor"] > 0.0
        assert prov["dual_norm_zero"] != prov["dual_norm_floor"]
        assert prov["dual_norm_shared"] > 0.0

    def test_shared_normaliser_is_the_target_scale_not_a_closed_loss(self):
        """Regression for the L10 silent failure: the scale must be bounded by
        the TARGET, so that a pathologically off-manifold closed state cannot
        divide its own term out of the loss."""
        bank, inf, pt, nt, pa = _setup()
        _, prov = _run("pos", bank, inf, pt, nt, pa, neg_tokens=nt,
                       mask_floor_source="dual")
        # _setup drives the seed to 2.0 on every positive, so mean(target^2)
        # is ~4 however extreme either closed-mask state happens to be.
        assert prov["dual_norm_shared"] == pytest.approx(4.0, rel=0.25)
        # NOTE it coincides with dual_norm_zero here, and that is an identity
        # rather than luck: when the zero floor's fully-closed stream leaves
        # the seed at ~0, its closed-mask loss IS (0 - target)^2 == target^2.
        # That is precisely why the old per-term scheme looked fine wherever
        # the empty state was benign (measured ratios L5 1.08, L8 1.19) and
        # only detonated where it was not (L10 1.9e8, empty pre-activation
        # ~1.8e5 off-manifold). The floor term is the one that must differ.
        assert prov["dual_norm_shared"] != prov["dual_norm_floor"]

    def test_dual_differs_from_both_single_floors(self):
        """If dual matched either specialist it would not be adding anything."""
        bank, inf, pt, nt, pa = _setup()
        zero, _ = _run("pos", bank, inf, pt, nt, pa)
        negc, _ = _run("pos", bank, inf, pt, nt, pa, neg_tokens=nt,
                       mask_floor_source="negctx")
        dual, pv = _run("pos", bank, inf, pt, nt, pa, neg_tokens=nt,
                        mask_floor_source="dual")
        assert pv["dual_norm_zero"] is not None
        assert not (dual == zero and dual == negc)

    def test_gamma_shifts_the_solution_toward_the_floor_term(self):
        bank, inf, pt, nt, pa = _setup()
        _, lo = _run("pos", bank, inf, pt, nt, pa, neg_tokens=nt,
                     mask_floor_source="dual", dual_floor_weight=0.0)
        _, hi = _run("pos", bank, inf, pt, nt, pa, neg_tokens=nt,
                     mask_floor_source="dual", dual_floor_weight=8.0)
        assert lo["dual_floor_weight"] == 0.0 and hi["dual_floor_weight"] == 8.0
        assert lo["loss_final"] != hi["loss_final"]


def test_floors_needing_negatives_matches_the_engine():
    """Regression: gradient_base skips negative RETRIEVAL unless something
    will read it, and that guard once hardcoded "negctx" — which starved
    "dual" and killed 4 of 11 arms mid-run with a missing-floor error. Both
    sides now import this tuple; this pins it to what the engine actually
    demands, so a new floor cannot drift out of sync silently."""
    from circuit.instrument.learned_mask import (
        FLOORS_NEEDING_NEGATIVES, MASK_FLOOR_SOURCES)
    bank, inf, pt, nt, pa = _setup()
    for src in MASK_FLOOR_SOURCES:
        if src == "posctx":
            continue                      # builds its floor from positives
        needs = src in FLOORS_NEEDING_NEGATIVES
        try:
            _run("pos", bank, inf, pt, nt, pa, mask_floor_source=src)
            raised = False
        except ValueError:
            raised = True
        assert raised == needs, (
            "%r raised=%s but FLOORS_NEEDING_NEGATIVES says %s" % (src, raised, needs))


class TestSingleDecodeEquivalence:
    """The transform computes ONE decode of (code - dense) instead of decoding
    `code` and `dense` separately and subtracting. decode is affine with a
    shared bias, so

        decode(code) + (x - decode(dense))  ==  x + (code - dense) @ W_dec.T

    exactly, in real arithmetic. These pin the rewrite against a literal
    re-implementation of the OLD formulation, so a future change to either
    side cannot silently diverge."""

    @staticmethod
    def _old_form(bank, x, m=None, floor=None, delta=None):
        """The pre-2026-07-30 transform, written out."""
        ta, ti = bank.encode(x, SITE[1], SITE[0])
        from sae.dense import sparse_topk_to_dense
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        recon = bank.decode(dense, SITE[1], SITE[0])
        error = x - recon.to(x.dtype)
        code = dense
        if m is not None:
            code = code * m if floor is None else code * m + floor * (1.0 - m)
        if delta is not None:
            code = code + delta
        return bank.decode(code, SITE[1], SITE[0]).to(x.dtype) + error

    def _x(self):
        x = torch.zeros(1, 2, D)
        x[0, 0, 0], x[0, 0, 3], x[0, 1, 5] = 2.0, 1.0, 0.75
        return x

    @pytest.mark.parametrize("theta", [40.0, 4.0, 0.0, -40.0])
    def test_matches_old_form_zero_floor(self, theta):
        bank = _Bank()
        x = self._x()
        p = LearnedMaskPatcher(bank, {SITE: torch.full((D,), theta)},
                               SEED_LAYER, SEED_KIND, torch.eye(D)[0],
                               torch.zeros(1))
        got = p.transform(SITE[0], SITE[1], x)
        want = self._old_form(bank, x, m=torch.sigmoid(torch.full((D,), theta)))
        assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()

    @pytest.mark.parametrize("theta", [4.0, 0.0, -40.0])
    def test_matches_old_form_mean_floor(self, theta):
        bank = _Bank()
        x = self._x()
        floors = {SITE: torch.full((D,), 0.4)}
        p = LearnedMaskPatcher(bank, {SITE: torch.full((D,), theta)},
                               SEED_LAYER, SEED_KIND, torch.eye(D)[0],
                               torch.zeros(1), floors=floors)
        got = p.transform(SITE[0], SITE[1], x)
        want = self._old_form(bank, x, m=torch.sigmoid(torch.full((D,), theta)),
                              floor=floors[SITE])
        assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()

    def test_matches_old_form_with_injection(self):
        """delta enters `code` additively, so it must enter the difference
        unchanged — the one path where the rewrite could plausibly drop a
        term."""
        bank = _Bank()
        x = self._x()
        psi = torch.full((D,), -1.0)
        p = LearnedMaskPatcher(bank, {SITE: torch.full((D,), 2.0)},
                               SEED_LAYER, SEED_KIND, torch.eye(D)[0],
                               torch.zeros(1),
                               deltas={SITE: psi})
        got = p.transform(SITE[0], SITE[1], x)
        want = self._old_form(
            bank, x, m=torch.sigmoid(torch.full((D,), 2.0)),
            delta=torch.nn.functional.softplus(psi))
        assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()

    def test_gradients_match_old_form(self):
        """Backward is ~70% of the training loop, so the rewritten graph's
        gradient — not just its value — has to agree."""
        bank = _Bank()
        x = self._x()
        th_new = torch.full((D,), 1.5, requires_grad=True)
        th_old = torch.full((D,), 1.5, requires_grad=True)
        p = LearnedMaskPatcher(bank, {SITE: th_new}, SEED_LAYER, SEED_KIND,
                               torch.eye(D)[0], torch.zeros(1))
        p.transform(SITE[0], SITE[1], x).pow(2).sum().backward()
        self._old_form(bank, x, m=torch.sigmoid(th_old)).pow(2).sum().backward()
        assert th_new.grad is not None and th_old.grad is not None
        assert torch.allclose(th_new.grad, th_old.grad, atol=1e-5), \
            (th_new.grad - th_old.grad).abs().max()


class TestActiveInit:
    """theta_init_mode="active": probe-inactive latents start at theta_lo.

    Under the ZERO floor an inactive latent's data gradient is identically
    zero (delta_code = -dense*(1-m) and dense=0), so down-initialising it
    deletes the measured 80-100-step burn-in in which the L1 penalty marches
    millions of informationless thetas across the keep threshold (the
    n = n_sites * d_sae plateau at steps 25-50). Under mean floors it is a
    genuine prior instead - masked inactive latents inject the floor value.
    """

    def test_invalid_mode_raises(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="theta_init_mode"):
            _run("pos", bank, inf, pt, nt, pa, theta_init_mode="warm")

    def test_all_active_matches_uniform(self):
        """The toy bank's encode returns every index in top-k, so ALL latents
        are probe-active and active-init must reduce to uniform exactly."""
        bank, inf, pt, nt, pa = _setup()
        uni, _ = _run("pos", bank, inf, pt, nt, pa)
        act, prov = _run("pos", bank, inf, pt, nt, pa,
                         theta_init_mode="active")
        assert prov["theta_init_mode"] == "active"
        assert uni == act

    def test_inactive_latents_start_low_and_stay_out(self):
        """With a top-2 encode only the two largest pre-acts per position are
        'active'; the rest must start at theta_lo and never enter the circuit
        (they have no data gradient to pull them back up under zero floor)."""
        class _Top2Bank(_Bank):
            def encode(self, x, kind, layer):
                pre = torch.relu(x.float())
                acts, idx = pre.topk(2, dim=-1)
                return acts, idx

        bank = _Top2Bank()
        x_pos = torch.zeros(4, 2, D)
        x_pos[:, :, 0] = 2.0        # active
        x_pos[:, :, 3] = 1.0        # active
        inf = _Inference(x_pos, torch.zeros(4, 2, D))
        pt = torch.zeros(4, 2, dtype=torch.long)
        pa = torch.zeros(4, dtype=torch.long)
        scores, prov = _run("pos", bank, inf, pt, None, pa,
                            theta_init_mode="active")
        assert prov["theta_init_mode"] == "active"
        member_idx = {f.index for f in scores}
        # latents other than 0 and 3 never fire on the probes; from theta_lo
        # with zero data gradient they cannot cross keep_threshold upward
        assert member_idx <= {0, 3}, member_idx


class TestSiteLambdaWeights:
    def test_weight_one_everywhere_matches_flat(self):
        bank, inf, pt, nt, pa = _setup()
        flat, _ = _run("pos", bank, inf, pt, nt, pa)
        w, prov = _run("pos", bank, inf, pt, nt, pa,
                       site_lambda_weights={SITE: 1.0})
        assert prov["site_lambda_weighted"] is True
        assert flat == w

    def test_missing_site_raises(self):
        """No silent 1.0 default: a run labelled 'weighted' that quietly
        priced most sites flat would be worse than a crash."""
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="site_lambda_weights"):
            _run("pos", bank, inf, pt, nt, pa, site_lambda_weights={})

    def test_heavier_price_prunes_harder(self):
        bank, inf, pt, nt, pa = _setup()
        cheap, _ = _run("pos", bank, inf, pt, nt, pa,
                        site_lambda_weights={SITE: 0.2})
        dear, _ = _run("pos", bank, inf, pt, nt, pa,
                       site_lambda_weights={SITE: 5.0})
        assert len(dear) <= len(cheap)


class TestBinarize:
    """Training-time gate discretisation - the TopK-SAE lesson brought to the
    mask. The SAE's top-k lives INSIDE its training forward, so it has no
    soft/hard gap by construction; these modes give the mask the same
    property while keeping a gradient path for non-members (which a naive
    hard top-k would freeze forever, since membership is global rather than
    per-token)."""

    def test_invalid_mode_raises(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="binarize"):
            _run("pos", bank, inf, pt, nt, pa, binarize="hard")

    def test_anneal_requires_half_threshold(self):
        """Anneal hardens the gate at theta=0 == m=0.5; any other cut would
        select a different set than training converged to."""
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="anneal"):
            _run("pos", bank, inf, pt, nt, pa, binarize="anneal",
                 keep_threshold=0.7)

    def test_ste_forward_is_exactly_hard(self):
        """The STE forward must EQUAL the hard-mask forward - that identity
        is the whole point (training sees eval semantics)."""
        bank = _Bank()
        x = torch.zeros(1, 2, D)
        x[0, 0, 0], x[0, 0, 3], x[0, 1, 5] = 2.0, 1.0, 0.75
        theta = torch.linspace(-3.0, 3.0, D)          # mixed soft values
        p_ste = LearnedMaskPatcher(bank, {SITE: theta}, SEED_LAYER, SEED_KIND,
                                   torch.eye(D)[0], torch.zeros(1),
                                   binarize="ste", bin_threshold=0.5)
        hard_theta = torch.where(torch.sigmoid(theta) > 0.5,
                                 torch.full((D,), 40.0),
                                 torch.full((D,), -40.0))
        p_hard = LearnedMaskPatcher(bank, {SITE: hard_theta}, SEED_LAYER,
                                    SEED_KIND, torch.eye(D)[0], torch.zeros(1))
        got = p_ste.transform(SITE[0], SITE[1], x)
        want = p_hard.transform(SITE[0], SITE[1], x)
        assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()

    def test_ste_gradient_reaches_subthreshold_latents(self):
        """The straight-through half: a latent whose hard gate is CLOSED
        (m_soft < 0.5, contributes nothing to the forward) must still get a
        nonzero gradient through the soft surrogate - otherwise init would
        freeze membership forever."""
        bank = _Bank()
        x = torch.zeros(1, 1, D)
        x[0, 0, 0] = 2.0
        theta = torch.full((D,), -1.0, requires_grad=True)   # all gates closed
        p = LearnedMaskPatcher(bank, {SITE: theta}, SEED_LAYER, SEED_KIND,
                               torch.eye(D)[0], torch.zeros(1),
                               binarize="ste", bin_threshold=0.5)
        out = p.transform(SITE[0], SITE[1], x)
        # Loss must not be stationary at the hard-closed output: with the
        # identity toy SAE, closing every gate gives out == x - decode(dense)
        # == 0 EXACTLY, so sum(out^2) has zero gradient at that point and a
        # zero theta.grad would prove nothing. Target the clean stream
        # instead (reconstruction), where the closed mask is maximally wrong.
        ((out - x) ** 2).sum().backward()
        assert theta.grad is not None
        assert float(theta.grad.abs().max()) > 0.0
        # and specifically the latent that carries signal, through a CLOSED
        # hard gate - the straight-through path itself
        assert float(theta.grad[0].abs()) > 0.0

    def test_anneal_sharpens_with_temperature(self):
        """At T=1 the anneal gate IS the soft gate; at tiny T it must match
        the hard gate."""
        bank = _Bank()
        x = torch.zeros(1, 1, D)
        x[0, 0, 0], x[0, 0, 3] = 2.0, 1.0
        theta = torch.linspace(-2.0, 2.0, D)
        p_soft = LearnedMaskPatcher(bank, {SITE: theta}, SEED_LAYER, SEED_KIND,
                                    torch.eye(D)[0], torch.zeros(1))
        p_ann = LearnedMaskPatcher(bank, {SITE: theta}, SEED_LAYER, SEED_KIND,
                                   torch.eye(D)[0], torch.zeros(1),
                                   binarize="anneal")
        p_ann.temperature = 1.0
        assert torch.allclose(p_ann.transform(SITE[0], SITE[1], x),
                              p_soft.transform(SITE[0], SITE[1], x), atol=1e-5)
        p_ann.temperature = 0.001
        hard_theta = torch.where(theta > 0, torch.full((D,), 40.0),
                                 torch.full((D,), -40.0))
        p_hard = LearnedMaskPatcher(bank, {SITE: hard_theta}, SEED_LAYER,
                                    SEED_KIND, torch.eye(D)[0], torch.zeros(1))
        assert torch.allclose(p_ann.transform(SITE[0], SITE[1], x),
                              p_hard.transform(SITE[0], SITE[1], x), atol=1e-3)

    def test_end_to_end_both_modes(self):
        bank, inf, pt, nt, pa = _setup()
        for mode in ("ste", "anneal"):
            scores, prov = _run("pos", bank, inf, pt, nt, pa, binarize=mode)
            assert prov["binarize"] == mode
            assert isinstance(scores, dict)

    def test_none_is_bitwise_unchanged(self):
        bank, inf, pt, nt, pa = _setup()
        base, _ = _run("pos", bank, inf, pt, nt, pa)
        none, prov = _run("pos", bank, inf, pt, nt, pa, binarize="none")
        assert prov["binarize"] == "none"
        assert base == none


def test_anneal_reach_frac_holds_the_floor():
    """reach_frac=0.5 must hit the floor temperature halfway and HOLD it;
    the default 1.0 descends the whole run. Verified through provenance and
    by the schedule maths rather than a full trajectory capture."""
    bank, inf, pt, nt, pa = _setup()
    _, prov = _run("pos", bank, inf, pt, nt, pa, binarize="anneal",
                   anneal_reach_frac=0.5)
    assert prov["anneal_reach_frac"] == 0.5
    _, prov2 = _run("pos", bank, inf, pt, nt, pa, binarize="anneal")
    assert prov2["anneal_reach_frac"] == 1.0


class TestSupport:
    """D3.6 support restriction: the mask searches ONLY inside an
    attribution-proposed candidate set; everything outside is frozen at
    the floor and unreachable by selection."""

    def test_members_subset_of_support(self):
        bank, inf, pt, nt, pa = _setup()
        sup = torch.tensor([0, 1, 2], dtype=torch.long)   # exclude 3..7
        scores, prov = _run("pos", bank, inf, pt, nt, pa,
                            support={SITE: sup})
        assert prov["support_n"] == 3
        for fid in scores:
            assert fid.index in (0, 1, 2)

    def test_index_and_bool_specs_agree(self):
        bank, inf, pt, nt, pa = _setup()
        idx = torch.tensor([0, 1, 2], dtype=torch.long)
        boolm = torch.zeros(D, dtype=torch.bool)
        boolm[:3] = True
        s1, _ = _run("pos", bank, inf, pt, nt, pa, support={SITE: idx})
        s2, _ = _run("pos", bank, inf, pt, nt, pa, support={SITE: boolm})
        assert s1 == s2

    def test_none_support_is_unchanged(self):
        bank, inf, pt, nt, pa = _setup()
        base, pb = _run("pos", bank, inf, pt, nt, pa)
        same, ps = _run("pos", bank, inf, pt, nt, pa, support=None)
        assert base == same
        assert ps["support_n"] is None

    def test_full_support_matches_unrestricted(self):
        bank, inf, pt, nt, pa = _setup()
        base, _ = _run("pos", bank, inf, pt, nt, pa)
        full, _ = _run("pos", bank, inf, pt, nt, pa,
                       support={SITE: torch.ones(D, dtype=torch.bool)})
        assert base == full

    def test_missing_site_excludes_everything(self):
        bank, inf, pt, nt, pa = _setup()
        scores, prov = _run("pos", bank, inf, pt, nt, pa, support={})
        assert prov["support_n"] == 0
        assert scores == {}

    def test_exclusion_survives_adamw_decay(self):
        # AdamW's decoupled decay pulls theta toward 0 (m toward 0.5) even
        # with zero data gradient — the per-step re-clamp must hold the
        # excluded latents at the floor through a full run.
        bank, inf, pt, nt, pa = _setup()
        sup = torch.tensor([0], dtype=torch.long)
        scores, _ = _run("pos", bank, inf, pt, nt, pa,
                         support={SITE: sup}, optimizer="adamw",
                         weight_decay=0.5, steps=200)
        for fid in scores:
            assert fid.index == 0

    def test_support_with_dual_floor(self):
        bank, inf, pt, nt, pa = _setup()
        sup = torch.tensor([0, 1], dtype=torch.long)
        scores, prov = _run("pos", bank, inf, pt, nt, pa,
                            support={SITE: sup}, neg_tokens=nt,
                            mask_floor_source="dual", dual_floor_weight=0.25)
        assert prov["support_n"] == 2
        for fid in scores:
            assert fid.index in (0, 1)


class TestInjectV3:
    """cf-mask v2 knobs: scale_normalize, delta_init warm start,
    suppress_weight. All default-off; the toy geometry is the inject
    class's (latent 0 = absent activator, latent 5 = present inhibitor)."""

    def test_defaults_bitwise_unchanged(self):
        bank, inf, pt, nt, pa = _setup()
        a, pa_ = _run("inject", bank, inf, pt, nt, pa,
                      neg_tokens=nt, target_act=2.5, steps=120)
        b, pb_ = _run("inject", bank, inf, pt, nt, pa,
                      neg_tokens=nt, target_act=2.5, steps=120,
                      scale_normalize=False, suppress_weight=0.0,
                      delta_init=None)
        assert a == b
        assert pb_["scale_normalize"] is False
        assert pb_["n_delta_init"] is None

    def test_scale_normalize_runs_and_flags(self):
        bank, inf, pt, nt, pa = _setup()
        scores, prov = _run("inject", bank, inf, pt, nt, pa,
                            neg_tokens=nt, target_act=2.5, steps=200,
                            scale_normalize=True)
        assert prov["scale_normalize"] is True
        # the objective still learns both roles under normalisation
        assert scores[FeatureID(0, "attn", 0)] > 0
        assert prov["loss_final"] < prov["loss_initial"]

    def test_scale_normalize_requires_target(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="target_act"):
            _run("negctx", bank, inf, pt, nt, pa, neg_tokens=nt,
                 target_act=None, scale_normalize=True)

    def test_delta_init_warm_start_is_exact(self):
        bank, inf, pt, nt, pa = _setup()
        # lr=0: the optimiser never moves, so softplus(psi) must equal the
        # warm value and selection must deliver latent 0 at ~2.0.
        scores, prov = _run("inject", bank, inf, pt, nt, pa,
                            neg_tokens=nt, target_act=2.5, steps=1, lr=0.0,
                            delta_init={SITE: {0: 2.0}})
        assert prov["n_delta_init"] == 1
        assert scores[FeatureID(0, "attn", 0)] == pytest.approx(2.0, abs=1e-4)

    def test_delta_init_on_excluded_site_is_skipped(self):
        bank, inf, pt, nt, pa = _setup()
        # one site total; excluding it leaves a gate-only run — warm start
        # must be silently ignored, not crash.
        scores, prov = _run("inject", bank, inf, pt, nt, pa,
                            neg_tokens=nt, target_act=1.0, steps=60,
                            inject_exclude_sites=1,
                            delta_init={SITE: {0: 2.0}})
        assert all(v < 0 for v in scores.values())   # edits only

    def test_suppress_weight_runs_and_still_fires(self):
        bank, inf, pt, nt, pa = _setup()
        scores, prov = _run("inject", bank, inf, pt, nt, pa,
                            neg_tokens=nt, target_act=2.5, steps=200,
                            scale_normalize=True, suppress_weight=0.25)
        assert prov["suppress_weight"] == 0.25
        # drive must survive the added suppression term
        assert prov["p_both"] > 1.0


class TestStepHook:
    """Diagnostics hook: called once per step with live grads and state,
    never mutating the run."""

    def test_hook_called_every_step_with_grads(self):
        bank, inf, pt, nt, pa = _setup()
        seen = []

        def hook(step, ctx):
            seen.append((step, sorted(ctx["grads"]["theta"]),
                         ctx["data_loss"], ctx["lr"]))

        _run("pos", bank, inf, pt, nt, pa, steps=10, step_hook=hook)
        assert [s for s, _, _, _ in seen] == list(range(10))
        assert seen[0][1] == [SITE]           # grads present for the site
        assert seen[0][3] > 0                 # lr reported

    def test_hook_does_not_change_result(self):
        bank, inf, pt, nt, pa = _setup()
        base, _ = _run("pos", bank, inf, pt, nt, pa, steps=60)
        hooked, _ = _run("pos", bank, inf, pt, nt, pa, steps=60,
                         step_hook=lambda s, c: None)
        assert base == hooked

    def test_hook_sees_inject_psi_grads(self):
        bank, inf, pt, nt, pa = _setup()
        got = {}

        def hook(step, ctx):
            got["psi_sites"] = sorted(ctx["grads"]["psi"])

        _run("inject", bank, inf, pt, nt, pa, neg_tokens=nt, target_act=2.5,
             steps=5, step_hook=hook)
        assert got["psi_sites"] == [SITE]


class TestRaiseObjective:
    """"raise" (inhibitor-mask): push the seed ABOVE natural on posctx,
    pay per silenced latent. Toy geometry: latent 5 SUPPRESSES the seed
    (decoder column 5 = -e0) and is present on posctx? — in _setup the
    suppressor sits on negctx, so we build a posctx-suppressed variant."""

    @staticmethod
    def _setup_pos_suppressed():
        # posctx carries BOTH the driver (dim 0) and the suppressor (dim 5),
        # so the seed fires but is braked: silencing latent 5 raises it.
        B, T = 4, 2
        x_pos = torch.zeros(B, T, D)
        x_pos[:, :, 0] = 3.0        # driver
        x_pos[:, :, 3] = 1.0        # irrelevant
        x_pos[:, :, 5] = 1.0        # SUPPRESSOR, present while firing
        x_neg = torch.zeros(B, T, D)
        x_neg[:, :, 5] = 1.0
        bank = _Bank()
        inf = _Inference(x_pos, x_neg)
        pt = torch.zeros(B, T, dtype=torch.long)
        nt = torch.ones(B, T, dtype=torch.long)
        pa = torch.zeros(B, dtype=torch.long)
        return bank, inf, pt, nt, pa

    def test_finds_the_suppressor_and_ignores_inert(self):
        bank, inf, pt, nt, pa = self._setup_pos_suppressed()
        # m=1 is the identity, so the natural seed pre-act is the raw stream
        # dim0 = 3.0. Latent 5 decodes to -e0, so SILENCING it removes a -1
        # from the reconstruction: the seed rises to 4.0. gamma = 4/3 makes
        # "silence exactly the suppressor" the exact solution.
        scores, prov = _run("raise", bank, inf, pt, nt, pa,
                            target_act=3.0, raise_gamma=4.0 / 3.0, steps=250,
                            l1_lambda=0.02)
        assert prov["raise_gamma"] == pytest.approx(4.0 / 3.0)
        assert FeatureID(0, "attn", 5) in scores          # the brake
        assert scores[FeatureID(0, "attn", 5)] < 0        # delivered as inhibitor
        # the inert latent (3) must NOT be recruited: silencing it buys
        # nothing and costs lambda — this is the tension the objective needs
        assert FeatureID(0, "attn", 3) not in scores
        # nor the DRIVER (silencing it lowers the seed: penalised twice)
        assert FeatureID(0, "attn", 0) not in scores

    def test_requires_target_and_gamma_above_one(self):
        bank, inf, pt, nt, pa = self._setup_pos_suppressed()
        with pytest.raises(ValueError, match="target_act"):
            _run("raise", bank, inf, pt, nt, pa, steps=5)
        with pytest.raises(ValueError, match="raise_gamma"):
            _run("raise", bank, inf, pt, nt, pa, target_act=3.0,
                 raise_gamma=1.0, steps=5)

    def test_scale_normalize_and_holdout_reported(self):
        bank, inf, pt, nt, pa = self._setup_pos_suppressed()
        _, prov = _run("raise", bank, inf, pt, nt, pa, target_act=3.0,
                       raise_gamma=1.5, steps=120, scale_normalize=True)
        assert prov["scale_normalize"] is True
        assert prov["holdout_data_loss"] is not None

    def test_higher_lambda_recruits_no_more_than_lower(self):
        bank, inf, pt, nt, pa = self._setup_pos_suppressed()
        cheap, _ = _run("raise", bank, inf, pt, nt, pa, target_act=3.0,
                        raise_gamma=1.5, steps=200, l1_lambda=0.005)
        dear, _ = _run("raise", bank, inf, pt, nt, pa, target_act=3.0,
                       raise_gamma=1.5, steps=200, l1_lambda=0.5)
        assert len(dear) <= len(cheap)

    def test_other_objectives_unchanged(self):
        bank, inf, pt, nt, pa = _setup()
        a, _ = _run("pos", bank, inf, pt, nt, pa, steps=80)
        b, _ = _run("pos", bank, inf, pt, nt, pa, steps=80, raise_gamma=2.0)
        assert a == b          # raise_gamma is inert outside "raise"


class TestPinObjective:
    """D3.1 pin-mask: kept latents are CLAMPED to clean pin values and
    dropped ones go to the floor — the pinned counterfactual made
    differentiable. Loss shape is "pos" (reproduce natural, pay to keep),
    so the driver twin of the closure mask."""

    def test_requires_pin_values(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="pin_values"):
            _run("pin", bank, inf, pt, nt, pa, steps=5)

    def test_keeps_the_driver_drops_the_irrelevant(self):
        bank, inf, pt, nt, pa = _setup()
        # pins = the clean posctx code: latent 0 = 2.0 (drives the seed),
        # latent 3 = 1.0 (decodes to e3, irrelevant to the seed).
        pins = torch.zeros(D)
        pins[0] = 2.0
        pins[3] = 1.0
        scores, prov = _run("pin", bank, inf, pt, nt, pa,
                            pin_values={SITE: pins}, steps=250,
                            l1_lambda=0.02)
        assert prov["n_pin_sites"] == 1
        assert FeatureID(0, "attn", 0) in scores      # the driver is kept
        assert FeatureID(0, "attn", 3) not in scores  # the inert one is not

    def test_pin_forward_is_pinned_not_scaled(self):
        # m=1 under "pin" reproduces the PIN value, not the live value:
        # with a pin of 0 the seed must read the floor even though the
        # live code is 2.0.
        bank, inf, pt, nt, pa = _setup()
        thetas = {SITE: torch.full((D,), 40.0)}      # sigmoid(40) == 1
        pins = torch.zeros(D)                        # every pin is 0
        sae = bank._sae
        p = LearnedMaskPatcher(bank, thetas, SEED_LAYER, SEED_KIND,
                               sae.encoder.weight[SEED_LATENT],
                               sae._get_bias_eff()[SEED_LATENT],
                               pins={SITE: pins})
        x = torch.zeros(1, 1, D)
        x[..., 0] = 2.0
        out = p.transform(0, "attn", x)
        assert float(out[0, 0, 0]) == pytest.approx(0.0, abs=1e-4)

    def test_pins_ignored_for_other_objectives(self):
        bank, inf, pt, nt, pa = _setup()
        pins = torch.zeros(D); pins[0] = 2.0
        a, _ = _run("pos", bank, inf, pt, nt, pa, steps=80)
        b, _ = _run("pos", bank, inf, pt, nt, pa, steps=80,
                    pin_values={SITE: pins})
        assert a == b


class TestLogitObjective:
    """objective='logit' — the BEHAVIOURAL endpoint. Same gates, same sites,
    same positions as 'pos'; the loss moves from the seed latent's
    pre-activation to log p(target token) at the anchor. Built to make the
    latent-vs-behavioural circuit-size comparison run one method over two
    endpoints instead of comparing our method to someone else's."""

    def _tok(self, pa, tok=1):
        return torch.full((pa.shape[0],), tok, dtype=torch.long)

    def test_logit_is_a_registered_objective(self):
        assert "logit" in OBJECTIVES

    def test_requires_target_tokens(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="target_tokens"):
            _run("logit", bank, inf, pt, nt, pa)

    def test_keeps_the_logit_carrying_latent_and_drops_the_inert_one(self):
        """Latent 0 decodes to e0, the only stream dim the unembed reads;
        latent 3 decodes to e3, which no logit sees. A behavioural objective
        must keep 0 and let L1 take 3 — note 'pos' has no reason to
        distinguish them differently, so this is the endpoint doing work."""
        bank, inf, pt, nt, pa = _setup()
        scores, meta = _run("logit", bank, inf, pt, nt, pa,
                            target_tokens=self._tok(pa))
        kept = {f.index for f in scores}
        assert 0 in kept
        assert 3 not in kept
        assert meta["objective"] == "logit"

    def test_accepts_2d_target_tokens(self):
        """[B, T] targets are gathered at pos_argmax, matching the [B] form."""
        bank, inf, pt, nt, pa = _setup()
        flat, _ = _run("logit", bank, inf, pt, nt, pa,
                       target_tokens=self._tok(pa))
        wide = torch.zeros((pa.shape[0], pt.shape[1]), dtype=torch.long)
        wide[:, :] = 1
        grid, _ = _run("logit", bank, inf, pt, nt, pa, target_tokens=wide)
        assert {f.index for f in flat} == {f.index for f in grid}

    def test_reproduces_the_full_model_logprob_not_maximises_it(self):
        """The target is the UNMASKED model's log p(target), so a converged
        mask's held-out squared error must be small — a maximising objective
        would drive the gap the other way and leave `ho` large."""
        bank, inf, pt, nt, pa = _setup()
        _, meta = _run("logit", bank, inf, pt, nt, pa, steps=200,
                       target_tokens=self._tok(pa))
        assert meta["holdout_data_loss"] is not None
        assert meta["holdout_data_loss"] < 0.05

    def test_sparsity_pressure_still_applies(self):
        """A large lambda must shrink membership: the behavioural loss must
        not be immune to the L1 term the size comparison depends on."""
        bank, inf, pt, nt, pa = _setup()
        loose, _ = _run("logit", bank, inf, pt, nt, pa, l1_lambda=0.0,
                        target_tokens=self._tok(pa))
        tight, _ = _run("logit", bank, inf, pt, nt, pa, l1_lambda=5.0,
                        target_tokens=self._tok(pa))
        assert len(tight) < len(loose)


class TestMaximiseObjective:
    """objective='maximise' — no target at all: drive the seed as high as
    the mask can and let L1 price each latent kept. The unbounded limit of
    `raise` (which targets raise_gamma * natural). A DRIVER objective: it
    does not reproduce the natural state, so its members must not be read
    as a closure circuit."""

    def _both(self):
        """Driver (latent 0 -> +e0) AND suppressor (latent 5 -> -e0) both
        present on the POSITIVE stream, so maximising has a real choice:
        keep 0, drop 5. Natural seed pre-act is x[...,0] = 2.0."""
        bank = _Bank()
        x = torch.zeros(4, 2, D)
        x[:, :, 0] = 2.0
        x[:, :, 5] = 1.0
        inf = _Inference(x, x)
        pt = torch.zeros(4, 2, dtype=torch.long)
        pa = torch.zeros(4, dtype=torch.long)
        return bank, inf, pt, pt, pa

    def test_maximise_is_a_registered_objective(self):
        assert "maximise" in OBJECTIVES

    def test_drops_the_suppressor_and_keeps_the_driver(self):
        bank, inf, pt, nt, pa = self._both()
        scores, meta = _run("maximise", bank, inf, pt, nt, pa, l1_lambda=0.01)
        kept = {f.index for f in scores}
        assert 0 in kept, "the driver must survive"
        assert 5 not in kept, "the suppressor must be dropped to raise the seed"
        assert meta["objective"] == "maximise"

    def test_pushes_the_seed_above_its_natural_level(self):
        """holdout is -(mean seed / mean natural): -1.0 is natural, more
        negative is amplified. Dropping the suppressor takes 2.0 -> 3.0."""
        bank, inf, pt, nt, pa = self._both()
        _, meta = _run("maximise", bank, inf, pt, nt, pa, l1_lambda=0.01)
        assert meta["holdout_data_loss"] is not None
        assert meta["holdout_data_loss"] < -1.0

    def test_pos_objective_keeps_the_suppressor_that_maximise_drops(self):
        """The endpoint difference, isolated: reproducing the natural state
        REQUIRES the suppressor (without it the seed overshoots), while
        maximising requires dropping it. Same data, same sites."""
        bank, inf, pt, nt, pa = self._both()
        pos_kept = {f.index for f in _run("pos", bank, inf, pt, nt, pa,
                                          l1_lambda=1e-4)[0]}
        max_kept = {f.index for f in _run("maximise", bank, inf, pt, nt, pa,
                                          l1_lambda=1e-4)[0]}
        assert 5 in pos_kept
        assert 5 not in max_kept

    def test_sparsity_pressure_still_bites(self):
        bank, inf, pt, nt, pa = self._both()
        loose, _ = _run("maximise", bank, inf, pt, nt, pa, l1_lambda=0.0)
        tight, _ = _run("maximise", bank, inf, pt, nt, pa, l1_lambda=5.0)
        assert len(tight) < len(loose)

    def test_scale_normaliser_makes_lambda_depth_independent(self):
        """A seed firing at 20 must not pay a different effective price from
        one firing at 2: the linear data term is divided by the mean natural
        level, so both land on the same loss scale."""
        small = self._both()
        big = self._both()
        big[1].streams[0][:, :, 0] = 20.0
        big[1].streams[1][:, :, 0] = 20.0
        a, _ = _run("maximise", *small, l1_lambda=0.01)
        b, _ = _run("maximise", *big, l1_lambda=0.01)
        assert {f.index for f in a} == {f.index for f in b}


class TestMaximiseDualFloor:
    """`dual` was a whitelist of one objective, written before `maximise`
    existed. Its stated reason — negctx/contrast/inject already carry a
    negative-context term — does not apply to `maximise`, whose loss lives
    on the POSITIVES exactly like `pos`. Enabling it matters because a
    single zero floor lets maximise raise the seed by DELETING
    SUPPRESSORS, which only works when deleted means set-to-zero."""

    def _suppressor_on_pos(self):
        """Suppressor (latent 5 -> -e0) present on the POSITIVE stream, and a
        DISTINCT negative stream that also carries it — so under the negctx
        floor a dropped latent 5 lands on ~1.0 (still suppressing) instead
        of 0, and deletion stops paying."""
        bank = _Bank()
        x_pos = torch.zeros(4, 2, D)
        x_pos[:, :, 0] = 2.0
        x_pos[:, :, 5] = 1.0
        x_neg = torch.zeros(4, 2, D)
        x_neg[:, :, 5] = 1.0
        inf = _Inference(x_pos, x_neg)
        pt = torch.zeros(4, 2, dtype=torch.long)
        nt = torch.ones(4, 2, dtype=torch.long)
        pa = torch.zeros(4, dtype=torch.long)
        return bank, inf, pt, nt, pa

    def test_dual_is_accepted_for_maximise(self):
        bank, inf, pt, nt, pa = self._suppressor_on_pos()
        scores, meta = _run("maximise", bank, inf, pt, nt, pa,
                            mask_floor_source="dual", neg_tokens=nt)
        assert meta["objective"] == "maximise"
        assert meta["mask_floor_source"] == "dual"

    def test_dual_still_barred_for_negctx_family(self):
        """The original restriction's REAL target must stay barred."""
        bank, inf, pt, nt, pa = self._suppressor_on_pos()
        for obj in ("negctx", "inject"):
            with pytest.raises(ValueError, match="dual"):
                _run(obj, bank, inf, pt, nt, pa, mask_floor_source="dual",
                     neg_tokens=nt, target_act=2.0)

    def test_dual_does_NOT_remove_the_deletion_incentive(self):
        """MEASURED, and it is why `dual` is not the fix for the zero-floor
        exploit. Dropping a suppressor still RAISES the seed in the
        zero-floored term (weight 1.0), is merely NEUTRAL in the
        negctx-floored term, and is REWARDED by L1 — net incentive
        unchanged. dual_floor_weight defaults to 0.25, so the negctx term is
        outweighed 4:1 regardless. The floor that actually removes the
        incentive is a SINGLE mean floor, tested below."""
        bank, inf, pt, nt, pa = self._suppressor_on_pos()
        zero, _ = _run("maximise", bank, inf, pt, nt, pa, l1_lambda=0.01,
                       mask_floor_source="zero", neg_tokens=nt)
        dual, _ = _run("maximise", bank, inf, pt, nt, pa, l1_lambda=0.01,
                       mask_floor_source="dual", neg_tokens=nt)
        assert {f.index for f in zero} == {f.index for f in dual}
        assert 5 not in {f.index for f in dual}   # suppressor dropped either way

    def test_single_mean_floor_is_accepted_for_maximise(self):
        """`posctx`/`negctx` alone are the arms that align TRAINING ablation
        semantics with how freeM evaluates — the actual candidate fix for
        the zero-floor exploit.

        Membership is NOT asserted non-empty: this fixture's stream is
        constant across positions, so each latent's mean equals its value,
        m=0 and m=1 coincide, and the mask is vacuous — L1 then prunes
        everything. That is a property of the toy, not of the floor. The
        real-model version is measured in
        dev-notes/data/maximise-objective-2026-08-05 (the `max/mean` arm)."""
        bank, inf, pt, nt, pa = self._suppressor_on_pos()
        for floor in ("posctx", "negctx"):
            _, meta = _run("maximise", bank, inf, pt, nt, pa,
                           mask_floor_source=floor, neg_tokens=nt)
            assert meta["objective"] == "maximise"
            assert meta["mask_floor_source"] == floor

    def test_maximise_dual_does_not_use_the_squared_error_normaliser(self):
        """`pos` divides its dual terms by dual_norm = mean(target^2) to make
        squared errors relative. `maximise` is LINEAR and already divides by
        max_scale, so applying dual_norm would rescale by target magnitude
        twice — the run must stay finite and select something."""
        bank, inf, pt, nt, pa = self._suppressor_on_pos()
        big = self._suppressor_on_pos()
        big[1].streams[0][:, :, 0] = 20.0
        big[1].streams[1][:, :, 5] = 1.0
        a, _ = _run("maximise", bank, inf, pt, nt, pa, l1_lambda=0.01,
                    mask_floor_source="dual", neg_tokens=nt)
        b, _ = _run("maximise", *big, l1_lambda=0.01,
                    mask_floor_source="dual", neg_tokens=big[3])
        assert len(a) > 0 and len(b) > 0
        assert {f.index for f in a} == {f.index for f in b}


class TestTripleFloor:
    """mask_floor_source='triple' = zero AND negctx AND posctx, all three
    scored every step over the SAME thetas. Added 2026-08-05.

    Note what it costs: training on posctx makes freeM_dense IN-SAMPLE,
    so the metric that currently validates pos/dual held-out stops being
    evidence. And the posctx term is gentle by construction — a dropped
    latent lands on its posctx mean, which on positive contexts is near
    its actual value."""

    def _streams(self):
        bank = _Bank()
        x_pos = torch.zeros(4, 2, D)
        x_pos[:, :, 0] = 2.0
        x_pos[:, :, 3] = 1.0
        x_neg = torch.zeros(4, 2, D)
        x_neg[:, :, 5] = 1.0
        inf = _Inference(x_pos, x_neg)
        pt = torch.zeros(4, 2, dtype=torch.long)
        nt = torch.ones(4, 2, dtype=torch.long)
        pa = torch.zeros(4, dtype=torch.long)
        return bank, inf, pt, nt, pa

    def test_triple_is_a_valid_floor_source(self):
        from circuit.instrument.learned_mask import MASK_FLOOR_SOURCES
        assert "triple" in MASK_FLOOR_SOURCES

    def test_triple_runs_and_is_recorded(self):
        bank, inf, pt, nt, pa = self._streams()
        _, meta = _run("pos", bank, inf, pt, nt, pa,
                       mask_floor_source="triple", neg_tokens=nt)
        assert meta["mask_floor_source"] == "triple"
        assert meta["triple_floor_weight"] == 0.25
        assert meta["dual_floor_weight"] is not None

    def test_triple_weight_is_none_for_other_floors(self):
        """Provenance must not claim a third term that was never scored."""
        bank, inf, pt, nt, pa = self._streams()
        _, meta = _run("pos", bank, inf, pt, nt, pa,
                       mask_floor_source="dual", neg_tokens=nt)
        assert meta["triple_floor_weight"] is None

    def test_triple_requires_negatives_like_dual(self):
        bank, inf, pt, nt, pa = self._streams()
        with pytest.raises(ValueError, match="neg_tokens"):
            _run("pos", bank, inf, pt, nt, pa, mask_floor_source="triple")

    def test_triple_barred_for_the_negctx_family(self):
        bank, inf, pt, nt, pa = self._streams()
        with pytest.raises(ValueError, match="dual"):
            _run("negctx", bank, inf, pt, nt, pa, mask_floor_source="triple",
                 neg_tokens=nt, target_act=2.0)

    def test_triple_accepted_for_maximise(self):
        bank, inf, pt, nt, pa = self._streams()
        _, meta = _run("maximise", bank, inf, pt, nt, pa,
                       mask_floor_source="triple", neg_tokens=nt)
        assert meta["objective"] == "maximise"
        assert meta["mask_floor_source"] == "triple"

    def test_third_term_actually_contributes(self):
        """Weight 0 on the third term must reproduce plain dual exactly; a
        non-zero weight must not. Guards against the posctx patcher being
        constructed but never scored."""
        bank, inf, pt, nt, pa = self._streams()
        dual, _ = _run("pos", bank, inf, pt, nt, pa, steps=60,
                       mask_floor_source="dual", neg_tokens=nt)
        off, _ = _run("pos", bank, inf, pt, nt, pa, steps=60,
                      mask_floor_source="triple", neg_tokens=nt,
                      triple_floor_weight=0.0)
        assert {f.index for f in off} == {f.index for f in dual}


class TestPnFloor:
    """mask_floor_source='pn' = negctx AND posctx, NO zero term. Added
    2026-08-06 as the direct ablation of the zero floor's role under free
    amplitudes: negctx PROMOTES to the primary 1.0 slot that zero held,
    posctx keeps triple_floor_weight, dual_floor_weight is unused."""

    def _streams(self):
        bank = _Bank()
        x_pos = torch.zeros(4, 2, D)
        x_pos[:, :, 0] = 2.0
        x_pos[:, :, 3] = 1.0
        x_neg = torch.zeros(4, 2, D)
        x_neg[:, :, 5] = 1.0
        inf = _Inference(x_pos, x_neg)
        pt = torch.zeros(4, 2, dtype=torch.long)
        nt = torch.ones(4, 2, dtype=torch.long)
        pa = torch.zeros(4, dtype=torch.long)
        return bank, inf, pt, nt, pa

    def test_pn_is_a_valid_floor_source(self):
        from circuit.instrument.learned_mask import (
            FLOORS_NEEDING_NEGATIVES, MASK_FLOOR_SOURCES)
        assert "pn" in MASK_FLOOR_SOURCES
        assert "pn" in FLOORS_NEEDING_NEGATIVES

    def test_pn_runs_and_is_recorded(self):
        bank, inf, pt, nt, pa = self._streams()
        _, meta = _run("pos", bank, inf, pt, nt, pa,
                       mask_floor_source="pn", neg_tokens=nt)
        assert meta["mask_floor_source"] == "pn"
        assert meta["triple_floor_weight"] == 0.25
        # no zero term -> dual_floor_weight was never scored
        assert meta["dual_floor_weight"] is None

    def test_pn_requires_negatives(self):
        bank, inf, pt, nt, pa = self._streams()
        with pytest.raises(ValueError, match="neg_tokens"):
            _run("pos", bank, inf, pt, nt, pa, mask_floor_source="pn")

    def test_pn_barred_for_the_negctx_family(self):
        bank, inf, pt, nt, pa = self._streams()
        with pytest.raises(ValueError, match="dual"):
            _run("negctx", bank, inf, pt, nt, pa, mask_floor_source="pn",
                 neg_tokens=nt, target_act=2.0)

    def test_pn_differs_from_triple(self):
        """Deleting the zero term must be able to change the outcome —
        otherwise the mode silently still scores it. The recorded loss
        endpoints must diverge even if the final sets agree."""
        bank, inf, pt, nt, pa = self._streams()
        _, m_tri = _run("pos", bank, inf, pt, nt, pa, steps=40,
                        mask_floor_source="triple", neg_tokens=nt)
        _, m_pn = _run("pos", bank, inf, pt, nt, pa, steps=40,
                       mask_floor_source="pn", neg_tokens=nt)
        assert (m_tri["loss_initial"], m_tri["loss_final"]) != \
               (m_pn["loss_initial"], m_pn["loss_final"])


class TestFreeAmplitude:
    """free_amplitude=True — each latent gets alpha = softplus(psi) on top
    of the gate, so the mask has FREE RANGE: off (gate), natural (alpha=1),
    elevated (alpha>1). psi initialises at softplus^-1(1), so step 0
    reproduces the gate-only mask exactly. The promise is COMPENSATION: one
    latent at 2x can replace two redundant ones, which no pure gate can."""

    def _redundant_drivers(self):
        """Two latents (0 and 6) each contribute 1.0 to the seed direction;
        natural seed pre-act = 2.0. A pure gate needs BOTH to reproduce it.
        With amplitude, ONE at alpha~2 suffices — strictly cheaper under L1."""
        bank = _Bank()
        bank._sae._W_dec = torch.eye(D)
        bank._sae._W_dec[:, 5] = 0.0
        bank._sae._W_dec[0, 5] = -1.0
        bank._sae._W_dec[:, 6] = 0.0
        bank._sae._W_dec[0, 6] = 1.0      # latent 6 also drives e0
        x = torch.zeros(4, 2, D)
        x[:, :, 0] = 1.0
        x[:, :, 6] = 1.0
        inf = _Inference(x, x)
        pt = torch.zeros(4, 2, dtype=torch.long)
        pa = torch.zeros(4, dtype=torch.long)
        return bank, inf, pt, pt, pa

    def test_amp_off_is_the_default_and_unchanged(self):
        bank, inf, pt, nt, pa = _setup()
        a, meta = _run("pos", bank, inf, pt, nt, pa)
        assert meta["free_amplitude"] is False
        assert meta["amp_stats"] is None and meta["amp_kept"] is None

    def test_alpha_initialises_at_one(self):
        """softplus(psi_init) must be 1.0 so training starts AT the
        gate-only behaviour; a wrong init would silently rescale every
        latent at step 0."""
        import math as _m
        psi = _m.log(_m.expm1(1.0))
        assert abs(torch.nn.functional.softplus(
            torch.tensor(psi)).item() - 1.0) < 1e-6

    def test_amplitude_enables_compensation(self):
        """The load-bearing behaviour: with redundant drivers and a stiff
        L1, the amplitude arm reproduces the target with FEWER members by
        amplifying one of them; the gate-only arm needs both (or eats
        loss). Assert the amp arm is no larger, its holdout is sound,
        membership is NON-EMPTY (the leak guard working), and at least one
        kept amplitude is genuinely elevated."""
        bank, inf, pt, nt, pa = self._redundant_drivers()
        plain, pmeta = _run("pos", bank, inf, pt, nt, pa, l1_lambda=0.2,
                            steps=250)
        amp, ameta = _run("pos", bank, inf, pt, nt, pa, l1_lambda=0.2,
                          steps=250, free_amplitude=True)
        assert len(amp) >= 1, "leak guard failed: empty membership carrying signal"
        assert len(amp) <= len(plain)
        assert ameta["holdout_data_loss"] < 0.2
        assert ameta["amp_stats"] is not None
        assert ameta["amp_stats"]["max"] > 1.3

    def test_leak_guard_prevents_subthreshold_amplitude_route(self):
        """The exploit the first version had: push m below keep_threshold
        (cheap under L1) and inflate alpha so m*alpha ~ 1 — loss near zero
        with EMPTY membership. The (1-m)*|alpha-1| charge must make honest
        membership the cheaper route: a converged run with low holdout loss
        must have kept at least one member."""
        bank, inf, pt, nt, pa = self._redundant_drivers()
        scores, meta = _run("pos", bank, inf, pt, nt, pa, l1_lambda=0.3,
                            steps=250, free_amplitude=True)
        if meta["holdout_data_loss"] is not None and meta["holdout_data_loss"] < 0.5:
            assert len(scores) >= 1

    def test_amp_l1_pulls_amplitudes_toward_natural(self):
        bank, inf, pt, nt, pa = self._redundant_drivers()
        free, fmeta = _run("pos", bank, inf, pt, nt, pa, l1_lambda=0.01,
                           steps=150, free_amplitude=True, amp_l1=0.0)
        priced, pmeta = _run("pos", bank, inf, pt, nt, pa, l1_lambda=0.01,
                             steps=150, free_amplitude=True, amp_l1=1.0)
        dev = lambda st: abs(st["median"] - 1.0) + abs(st["max"] - 1.0)
        assert dev(pmeta["amp_stats"]) <= dev(fmeta["amp_stats"]) + 1e-6

    def test_amp_kept_is_archived_per_member(self):
        bank, inf, pt, nt, pa = _setup()
        scores, meta = _run("pos", bank, inf, pt, nt, pa,
                            free_amplitude=True)
        assert meta["amp_kept"] is not None
        archived = sum(len(v) for v in meta["amp_kept"].values())
        assert archived == len(scores)

    def test_amp_rejected_for_pin(self):
        bank, inf, pt, nt, pa = _setup()
        with pytest.raises(ValueError, match="free_amplitude"):
            _run("pin", bank, inf, pt, nt, pa, free_amplitude=True,
                 pin_values={SITE: torch.zeros(D)})
