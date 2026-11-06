"""The resample term R (2026-10-31, experiment 101): learned_mask resample_weight / resample_tokens / resample_k /
resample_seed, the native port of 079's donor-resample term with 095's sparsity-preserving donor fill. CPU only.

  - off (resample_weight 0, the default) is the historical engine bit for bit, whatever resample_tokens says;
  - on, a fourth term ("resample") is trained, last, at gamma_R / dual_norm, and changes the loss;
  - donors are redrawn at every forward of the term (one row per training row) from a seeded generator: same seed,
    same fit; another seed, another fit; resample_k draws per row per step;
  - the donor patcher: floor = the drawn donor codes (never cached), under train_fill="sparse" donor_keep_mask's fill
    (members at alpha x live, then the donor's non-member entries largest first up to K - #active members);
  - unweighted (gate-only) fits take the term too, dense or sparse;
  - capture_donor_codes / DonorPool return what the engine trains against; invalid settings are refused.
(The bit-identity with 079 / 095's in-memory patch on this fake model is experiments/101-r-native/test_101_equiv_cpu.py.)
"""
from types import SimpleNamespace

import pytest
import torch

from circuit.instrument import learned_mask as LM
from circuit.instrument.learned_mask import (
    DonorPool, LearnedMaskPatcher, capture_donor_codes, donor_keep_mask, run_learned_mask)
from sae.dense import sparse_topk_to_dense

D_MODEL, D_SAE, K = 16, 64, 8
SITE = (0, "resid")


class FakeBank:
    """Top-K SAE: encode = top-k of relu(x W^T); decode = code W_dec^T (+ bias)."""
    d_sae, k = D_SAE, K
    kinds = ["attn", "mlp", "resid"]
    device = torch.device("cpu")

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(D_SAE, D_MODEL, generator=g)
        self.Wd = torch.randn(D_MODEL, D_SAE, generator=g)
        self.bd = torch.randn(D_MODEL, generator=g)
        sae = SimpleNamespace(encoder=SimpleNamespace(weight=self.W), _get_bias_eff=lambda: torch.zeros(D_SAE))
        self.saes = {k: {0: sae, 1: sae} for k in self.kinds}

    def encode(self, x, kind, layer):
        return torch.relu(x.float() @ self.W.T).topk(K, dim=-1)

    def decode(self, code, kind, layer, add_bias=True):
        return code @ self.Wd.T + (self.bd if add_bias else 0)


class FakeInference:
    """Two-site pipe: the patcher's transform at (0, 'resid'), then the seed tap at (1, 'resid'). Row r of a batch is
    stream row tokens[r, 0] (so donor contexts are distinct rows: token ids index the stored stream)."""

    def __init__(self, x):
        self.x = x

    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, patcher=None, activations_callback=None, **kw):
        x = self.x[tokens[:, 0].long()].clone()
        if activations_callback is not None:
            activations_callback(0, (x, x, x))
            return None
        if patcher is not None:
            x = patcher.transform(0, "resid", x)
            patcher.transform(1, "resid", x)


def _model():
    g = torch.Generator().manual_seed(11)
    x = torch.randn(24, 4, D_MODEL, generator=g)
    x[:8] += 0.5                                                   # rows 0-7 activating, 8-15 contrast, 16-23 other
    return FakeInference(x), FakeBank()


def _tok(rows):
    return torch.tensor(rows, dtype=torch.long)[:, None].repeat(1, 4)


PT, NT, DONORS = _tok(range(8)), _tok(range(8, 16)), _tok(range(8, 14))      # donors = the 6 contrast-train rows


def _engine(**kw):
    inf, bank = _model()
    args = dict(objective="pos", sites=[SITE], seed_layer=1, seed_kind="resid", seed_latent_idx=3, pos_tokens=PT,
                pos_argmax=torch.zeros(8, dtype=torch.long), neg_tokens=NT, steps=12, lr=0.1, l1_lambda=1e-3,
                batch_size=4, log_every=0, mask_floor_source="triple", free_amplitude=True, binarize="anneal",
                triple_floor_weight=0.25, dual_floor_weight=0.25, train_fill="sparse", sparse_grad="ste")
    args.update(kw)
    return run_learned_mask(inf, bank, **args)


# ------------------------------------------------------------------------------------------- off = unchanged
@pytest.mark.parametrize("fill", ["dense", "sparse"])
@pytest.mark.parametrize("free_amp", [True, False])
def test_off_is_the_historical_engine(fill, free_amp):
    s0, p0 = _engine(train_fill=fill, free_amplitude=free_amp)
    s1, p1 = _engine(train_fill=fill, free_amplitude=free_amp, resample_weight=0.0, resample_tokens=DONORS,
                     resample_k=3, resample_seed=5)
    assert s0 == s1 and p0["curve"] == p1["curve"] and p0["amp_kept"] == p1["amp_kept"]
    assert sorted(p0["curve"]["terms"]) == ["floor", "pos", "zero"]
    for k in ("resample_weight", "resample_k", "resample_n_donors", "resample_draws", "resample_fill",
              "resample_fill_calls"):
        assert p0[k] is None and p1[k] is None


def test_off_builds_no_donor_patcher_and_captures_nothing(monkeypatch):
    seen = []
    monkeypatch.setattr(LM, "capture_donor_codes", lambda *a, **k: seen.append(1))
    orig = LM.LearnedMaskPatcher.__init__

    def spy(self, *a, **k):
        orig(self, *a, **k)
        seen.append(self.donors)
    monkeypatch.setattr(LM.LearnedMaskPatcher, "__init__", spy)
    _engine(resample_tokens=DONORS)
    assert seen and all(d is None for d in seen)


# ------------------------------------------------------------------------------------------- on
@pytest.mark.parametrize("fill", ["dense", "sparse"])
def test_on_adds_the_resample_term_and_changes_the_loss(fill):
    _, p0 = _engine(train_fill=fill)
    s1, p1 = _engine(train_fill=fill, resample_weight=0.25, resample_tokens=DONORS)
    assert list(p1["curve"]["terms"]) == ["zero", "floor", "pos", "resample"]
    assert p1["curve"]["loss"] != p0["curve"]["loss"]
    assert p1["curve"]["terms"]["zero"][0] == p0["curve"]["terms"]["zero"][0]       # same state at step 0
    assert p1["resample_weight"] == 0.25 and p1["resample_k"] == 1 and p1["resample_n_donors"] == 6
    assert p1["resample_draws"] == 12                                                # one per step (accum 1, k 1)
    assert p1["resample_fill"] == fill
    assert (p1["resample_fill_calls"] == 12) if fill == "sparse" else (p1["resample_fill_calls"] is None)
    # sparse_fill_calls still counts the C and A patchers only (095's provenance)
    _, p2 = _engine(train_fill=fill)
    assert p1["sparse_fill_calls"] == p2["sparse_fill_calls"]


def test_term_weight_is_gamma_r_over_dual_norm():
    """The resample entry is (training rows, gamma_R / dual_norm, the donor patcher), appended after the A term:
    at step 0 its curve value is gamma_R x the donor term's relative squared error, and scaling gamma_R scales it."""
    _, pa = _engine(resample_weight=0.25, resample_tokens=DONORS, steps=1)
    _, pb = _engine(resample_weight=0.5, resample_tokens=DONORS, steps=1)
    a, b = pa["curve"]["terms"]["resample"][0], pb["curve"]["terms"]["resample"][0]
    assert a > 0 and b == pytest.approx(2 * a, rel=1e-6)


def test_donors_are_redrawn_every_forward_and_seeded(monkeypatch):
    rows = []
    orig = LM.DonorPool.draw

    def spy(self, B):
        orig(self, B)
        rows.append(self.rows.clone())
    monkeypatch.setattr(LM.DonorPool, "draw", spy)
    s1, p1 = _engine(resample_weight=0.25, resample_tokens=DONORS)
    # one draw per forward, one donor per row of that micro-batch (6 training rows, batch 4: slices of 4, 2, 4, ...)
    sizes = [len(range(6)[(s * 4) % 6:(s * 4) % 6 + 4]) for s in range(12)]
    assert [int(r.shape[0]) for r in rows] == sizes
    assert len({tuple(r.tolist()) for r in rows}) > 1                                # redrawn, not fixed
    g = torch.Generator().manual_seed(LM.RESAMPLE_SEED)
    assert all(torch.equal(r, torch.randint(6, (n,), generator=g)) for r, n in zip(rows, sizes))  # seeded sequence
    rows.clear()
    s2, p2 = _engine(resample_weight=0.25, resample_tokens=DONORS)
    assert s1 == s2 and p1["curve"] == p2["curve"]                                   # same seed, same fit
    _, p3 = _engine(resample_weight=0.25, resample_tokens=DONORS, resample_seed=7)
    assert p3["curve"]["loss"] != p1["curve"]["loss"]


def test_k_draws_per_row_per_step():
    _, p = _engine(resample_weight=0.25, resample_tokens=DONORS, resample_k=3)
    assert list(p["curve"]["terms"]) == ["zero", "floor", "pos", "resample", "resample2", "resample3"]
    assert p["resample_draws"] == 36 and p["resample_k"] == 3
    _, p1 = _engine(resample_weight=0.25, resample_tokens=DONORS, resample_k=3, steps=1)
    r = [p1["curve"]["terms"][n][0] for n in ("resample", "resample2", "resample3")]
    assert len(set(r)) > 1                                                           # three independent draws
    # one donor context: every draw is the same, so each of the k entries is the k = 1 term / k
    one = _tok([8])
    _, pk3 = _engine(resample_weight=0.25, resample_tokens=one, resample_k=3, steps=1)
    _, pk1 = _engine(resample_weight=0.25, resample_tokens=one, steps=1)
    r3 = [pk3["curve"]["terms"][n][0] for n in ("resample", "resample2", "resample3")]
    assert r3[0] == r3[1] == r3[2] == pytest.approx(pk1["curve"]["terms"]["resample"][0] / 3, rel=1e-6)


@pytest.mark.parametrize("fill", ["dense", "sparse"])
def test_unweighted_fits_take_the_term(fill):
    _, p0 = _engine(free_amplitude=False, train_fill=fill)
    s1, p1 = _engine(free_amplitude=False, train_fill=fill, resample_weight=0.25, resample_tokens=DONORS)
    assert p1["free_amplitude"] is False and "resample" in p1["curve"]["terms"]
    assert p1["curve"]["loss"] != p0["curve"]["loss"]
    assert (p1["resample_fill_calls"] or 0) == (12 if fill == "sparse" else 0)


def test_refuses_undefined_settings():
    with pytest.raises(ValueError, match="resample_tokens"):
        _engine(resample_weight=0.25)
    with pytest.raises(ValueError, match="multi-term floor"):
        _engine(resample_weight=0.25, resample_tokens=DONORS, mask_floor_source="negctx", train_fill="dense")
    with pytest.raises(ValueError, match="objective='pos'"):
        _engine(resample_weight=0.25, resample_tokens=DONORS, objective="maximise", train_fill="dense")
    with pytest.raises(ValueError, match="resample_k"):
        _engine(resample_weight=0.25, resample_tokens=DONORS, resample_k=0)
    with pytest.raises(ValueError, match="resample_weight"):
        _engine(resample_weight=-0.1, resample_tokens=DONORS)


# ------------------------------------------------------------------------------------------- the donor patcher
def _codes(inf, bank, rows):
    return capture_donor_codes(inf, bank, _tok(rows), [SITE], batch=4)


def test_capture_and_pool_return_the_natural_codes():
    inf, bank = _model()
    codes = _codes(inf, bank, range(8, 14))
    acts, idx = codes[SITE]
    assert acts.shape == (6, 4, K) and acts.dtype == torch.float32 and acts.device.type == "cpu"
    ta, ti = bank.encode(inf.x[8:14], "resid", 0)
    assert torch.equal(acts, ta.float()) and torch.equal(idx, ti)
    pool = DonorPool(codes, D_SAE, torch.device("cpu"), seed=3)
    with pytest.raises(RuntimeError, match="before draw"):
        pool.get(SITE)
    pool.draw(5)
    want = sparse_topk_to_dense(ta, ti, D_SAE, dtype=torch.float32)[pool.rows]
    assert torch.equal(pool.get(SITE), want) and pool.get((9, "attn")) is None and pool.n_draws == 1


def _donor_patcher(bank, pool, theta, psi, **kw):
    th = {SITE: theta.clone().requires_grad_(True)}
    am = {} if psi is None else {SITE: psi.clone().requires_grad_(True)}
    return LearnedMaskPatcher(bank, th, 1, "resid", bank.W[3], torch.zeros(()), amps=am,
                              code_dtype="stream", bin_threshold=0.5, donors=pool, **kw)


def _gates(seed=1, n_mem=20):
    g = torch.Generator().manual_seed(seed)
    member = torch.zeros(D_SAE, dtype=torch.bool)
    member[torch.randperm(D_SAE, generator=g)[:n_mem]] = True
    theta = torch.where(member, torch.full((D_SAE,), 200.0), torch.full((D_SAE,), -200.0))   # m exactly 1 / 0
    alpha = 0.5 + 1.5 * torch.rand(D_SAE, generator=g)
    return member, theta, torch.log(torch.expm1(alpha))


@pytest.mark.parametrize("weighted", [True, False])
def test_donor_sparse_fill_code(weighted):
    """Binary gates: members at alpha x live; the donor's non-member entries > 0, largest first, up to
    K - #active members; the rest 0 (073 ResamplePatcher's Top-K budget fill)."""
    inf, bank = _model()
    pool = DonorPool(_codes(inf, bank, range(8, 14)), D_SAE, torch.device("cpu"), seed=2)
    member, theta, psi = _gates()
    p = _donor_patcher(bank, pool, theta, psi if weighted else None, train_fill="sparse", sparse_grad="ste")
    x = inf.x[:4].clone()
    pool.draw(4)
    donor = pool.get(SITE)
    cap = {}
    orig = LM.sparse_code

    def spy(code, H, g):
        y = orig(code, H, g)
        cap["code"] = y.detach()
        return y
    LM.sparse_code = spy
    try:
        p.transform(*SITE, x)
    finally:
        LM.sparse_code = orig
    ta, ti = bank.encode(x, "resid", 0)
    dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    alpha = LM._amp_of(psi).detach() if weighted else torch.ones(D_SAE)
    want = torch.zeros_like(dense)
    for b in range(4):
        for t in range(4):
            live = dense[b, t] * alpha
            want[b, t, member] = live[member]
            budget = max(K - int(((live != 0) & member).sum()), 0)
            cand = [(float(donor[b, t, j]), j) for j in range(D_SAE) if not member[j] and donor[b, t, j] > 0]
            for v, j in sorted(cand, reverse=True)[:budget]:
                want[b, t, j] = v
    assert torch.allclose(cap["code"], want, atol=1e-6)
    assert p.sparse_calls == 1
    H = donor_keep_mask(dense, member, alpha, donor, K)
    assert bool(H[..., member].all())


def test_donor_dense_fill_and_no_floor_cache():
    """train_fill="dense": code = m alpha dense + (1 - m) donor; the donor floor is re-read every forward."""
    inf, bank = _model()
    pool = DonorPool(_codes(inf, bank, range(8, 14)), D_SAE, torch.device("cpu"), seed=4)
    member, theta, psi = _gates(3)
    theta = theta.clamp(-2, 2)                                                       # soft gates
    p = _donor_patcher(bank, pool, theta, psi)
    x = inf.x[:4].clone()
    outs = []
    for _ in range(3):
        pool.draw(4)
        donor = pool.get(SITE)
        ta, ti = bank.encode(x, "resid", 0)
        dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
        m, a = torch.sigmoid(theta), LM._amp_of(psi)
        want = x + bank.decode(dense * (m * a - 1.0) + (1.0 - m) * donor, "resid", 0, add_bias=False)
        got = p.transform(*SITE, x)
        assert torch.equal(got, want)
        outs.append(got.detach())
    assert not p._const_cache                                                        # never cached
    assert not (torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2]))


def test_forward_preact_draws_before_each_forward():
    inf, bank = _model()
    pool = DonorPool(_codes(inf, bank, range(8, 14)), D_SAE, torch.device("cpu"))
    member, theta, psi = _gates()
    p = _donor_patcher(bank, pool, theta, psi, train_fill="sparse")
    for n in (1, 2):
        LM._forward_preact(inf, p, PT[:3], grad=False)
        assert pool.n_draws == n and int(pool.rows.shape[0]) == 3
    plain = LearnedMaskPatcher(bank, {SITE: theta.clone()}, 1, "resid", bank.W[3], torch.zeros(()))
    LM._forward_preact(inf, plain, PT[:3], grad=False)                               # no donors: no draw
    assert pool.n_draws == 2


# ------------------------------------------------------------------------------------------- discovery hop wiring
def test_hop_passes_the_callers_donors_only_when_on(monkeypatch):
    """ablation_gradient: resample_weight 0 passes no resample kwarg; > 0 passes method._resample_donors (and
    refuses without them)."""
    from circuit.discovery import ablation_gradient as AG
    import circuit.instrument.learned_mask as lm_mod
    from config import config
    seen = {}

    def fake_run(inference, bank, **kw):
        seen.clear()
        seen.update(kw)
        return {}, {"loss_initial": 0.0, "loss_final": 0.0}
    monkeypatch.setattr(lm_mod, "run_learned_mask", fake_run)
    M = object.__new__(AG.AblationGradientDiscovery)
    M.position_aware = False
    M.sae_bank = FakeBank()
    M.inference = None
    M.probe_batch_size = 4
    M._floor_neg_tokens = NT
    M._stash_amplitudes = lambda prov: None
    log = SimpleNamespace(note=lambda *a, **k: None)
    cfg = config.discovery.learned_mask
    monkeypatch.setattr(cfg, "mask_floor_source", "triple")
    monkeypatch.setattr(cfg, "train_means_source", "fit")
    M._run_mask_hop(1, "resid", 3, PT, torch.zeros(8, dtype=torch.long), log)
    assert not any(k.startswith("resample") for k in seen)
    monkeypatch.setattr(cfg, "resample_weight", 0.25)
    monkeypatch.setattr(cfg, "resample_k", 2)
    with pytest.raises(ValueError, match="_resample_donors"):
        M._run_mask_hop(1, "resid", 3, PT, torch.zeros(8, dtype=torch.long), log)
    M._resample_donors = DONORS
    M._run_mask_hop(1, "resid", 3, PT, torch.zeros(8, dtype=torch.long), log)
    assert seen["resample_weight"] == 0.25 and seen["resample_k"] == 2 and seen["resample_tokens"] is DONORS
    assert seen["resample_seed"] == cfg.resample_seed == LM.RESAMPLE_SEED
    assert M.last_mask_provenance["resample_pool"] == "contrast"
