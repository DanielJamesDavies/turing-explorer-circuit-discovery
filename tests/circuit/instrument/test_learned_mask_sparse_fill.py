"""Protocol v2 (2026-10-25): the learned mask's A / C training terms under the pass rule's sparsity-preserving fill
(learned_mask train_fill="sparse", sparse_grad "ste" | "hard"; ported from experiment 090's sp_engine.py), and the
scorer's A / C means as the training means (train_means_source="eval_ref": protocol_contexts.eval_reference +
AblationGradientDiscovery._eval_ref_means). CPU only, no model.

  - with binary gates the training code equals the scorer's patched code (CircuitOnlyPatcher._respect_topk_fill)
    exactly, ties and more-than-K active members included, and the site transforms agree;
  - "ste" is forward-identical to "hard" and passes the gradient straight through; "hard" blocks dropped entries;
  - train_fill="dense" (default) is the historical forward bit for bit, and the Z patcher (no floor) is untouched;
  - the engine wiring: sparse reaches the C and A patchers only, is recorded in the provenance, and refuses
    configurations it is not defined for;
  - eval_reference returns the scorer's reference rows; _eval_ref_means returns the scorer's means on them.
"""
from types import SimpleNamespace

import pytest
import torch

from circuit.instrument import learned_mask as LM
from circuit.instrument.learned_mask import (
    LearnedMaskPatcher, run_learned_mask, sparse_code, sparse_keep_mask)
from eval.ablation_faithfulness import CircuitOnlyPatcher
from sae.dense import sparse_topk_to_dense

D_MODEL, D_SAE, K = 16, 64, 8
B, T = 3, 5
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


def _setup(seed=1, soft=False, n_mem=20, ties=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, D_MODEL, generator=g)
    floor = torch.rand(D_SAE, generator=g) * 0.5
    floor[torch.rand(D_SAE, generator=g) < 0.3] = 0.0               # dead-in-context latents: mean exactly 0
    if ties:
        floor = (floor * 10).round() / 10                            # many exact ties among nonzero means
    perm = torch.randperm(D_SAE, generator=g)
    member = torch.zeros(D_SAE, dtype=torch.bool)
    member[perm[:n_mem]] = True
    alpha = 0.5 + 1.5 * torch.rand(D_SAE, generator=g)
    psi = torch.log(torch.expm1(alpha))
    if soft:
        theta = torch.randn(D_SAE, generator=g) * 2
        theta[member] = theta[member].abs() + 0.1
        theta[~member] = -theta[~member].abs() - 0.1
    else:
        theta = torch.where(member, torch.full((D_SAE,), 200.0), torch.full((D_SAE,), -200.0))  # m exactly 1 / 0
    return x, floor, member, psi, theta


def _patcher(bank, theta, psi, floor, binarize="none", **kw):
    th = {SITE: theta.clone().requires_grad_(True)}
    am = {SITE: psi.clone().requires_grad_(True)}
    p = LearnedMaskPatcher(bank, th, 5, "resid", torch.zeros(D_MODEL), torch.zeros(()), amps=am,
                           code_dtype="stream", floors=None if floor is None else {SITE: floor},
                           binarize=binarize, bin_threshold=0.5, **kw)
    return p, th, am


def _scorer(bank, member, alpha_eff, floor):
    keep = sorted(member.nonzero().flatten().tolist())
    sv = torch.ones(D_SAE)
    sv[keep] = alpha_eff[keep]
    return CircuitOnlyPatcher(bank=bank, keep_indices={SITE: set(keep)}, in_scope={SITE}, seed_layer=-1,
                              seed_kind="", seed_latent_idx=0, site_means={SITE: floor}, respect_topk=True, topk=K,
                              keep_scales={SITE: sv}), torch.tensor(keep, dtype=torch.long), sv


def _scorer_code(bank, x, member, alpha_eff, floor, dense=None):
    """The scorer's patched code (respect_topk fill) on the same inputs."""
    P, kt, sv = _scorer(bank, member, alpha_eff, floor)
    if dense is None:
        ta, ti = bank.encode(x, *SITE[::-1])
        dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    vals = dense[:, :, kt] * sv[kt]
    return P._respect_topk_fill(dense, floor.to(dense.dtype), kt, vals), dense


# ------------------------------------------------------------------------------------------- fill == scorer's fill
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("grad", ["hard", "ste"])
def test_training_fill_code_equals_scorer_fill(seed, grad):
    """Binary gates: the training pass's sparsity-preserving code == the scorer's patched code, exactly (with ties)."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(seed)
    alpha_eff = LM._amp_of(psi)
    want, dense = _scorer_code(bank, x, member, alpha_eff, floor)
    m = torch.sigmoid(theta)
    assert torch.equal(m, member.float())
    code = m * alpha_eff * dense + (1.0 - m) * floor
    H = sparse_keep_mask(dense, m > 0.5, alpha_eff, floor.to(dense.dtype), K)
    got = sparse_code(code, H, grad)
    assert torch.equal(got, want)
    assert bool(H[..., member].all())                                        # members are never evicted
    n_act = ((dense != 0) & member).sum(-1)
    assert bool(((got != 0).sum(-1) <= torch.maximum(n_act, torch.full_like(n_act, K))).all())


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("grad", ["hard", "ste"])
def test_training_transform_equals_scorer_transform(seed, grad):
    """The whole site transform (engine: x + decode(code_sp - dense, no bias); scorer: decode(patched) + error)."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(seed)
    p, _, _ = _patcher(bank, theta, psi, floor, train_fill="sparse", sparse_grad=grad)
    got = p.transform(*SITE, x)
    P, _, _ = _scorer(bank, member, LM._amp_of(psi).detach(), floor)
    want = P.transform(*SITE, x)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
    assert p.sparse_calls == 1
    # and it is not the dense fill
    pd_, _, _ = _patcher(bank, theta, psi, floor)
    assert not torch.allclose(pd_.transform(*SITE, x), want, atol=1e-3)


def test_more_active_members_than_K():
    """Budget clamps at 0: every active member is kept, no mean fill at that position (the scorer's rule)."""
    g = torch.Generator().manual_seed(9)
    x, floor, member, psi, theta = _setup(5, n_mem=40)
    alpha_eff = LM._amp_of(psi)
    dense = torch.rand(B, T, D_SAE, generator=g) * (torch.rand(B, T, D_SAE, generator=g) < 0.4)
    dense[0, 0, member] = 1.0                                         # 40 active members > K at (0, 0)
    want, _ = _scorer_code(FakeBank(), x, member, alpha_eff, floor, dense=dense)
    m = member.float()
    code = m * alpha_eff * dense + (1.0 - m) * floor
    got = sparse_code(code, sparse_keep_mask(dense, member, alpha_eff, floor, K), "hard")
    assert torch.equal(got, want)
    assert int((got[0, 0] != 0).sum()) == 40 and bool((got[0, 0, ~member] == 0).all())


# ------------------------------------------------------------------------------------------- gradients
def test_ste_passes_the_gradient_straight_through_and_hard_masks_it():
    g = torch.Generator().manual_seed(3)
    code = torch.randn(B, T, D_SAE, generator=g)
    H = torch.rand(B, T, D_SAE, generator=g) < 0.3
    up = torch.randn(B, T, D_SAE, generator=g)
    out = {}
    for grad in ("ste", "hard"):
        c = code.clone().requires_grad_(True)
        y = sparse_code(c, H, grad)
        (y * up).sum().backward()
        out[grad] = (y.detach(), c.grad.clone())
    assert torch.equal(out["ste"][0], out["hard"][0])                      # same forward
    assert torch.equal(out["ste"][0], code * H.float())
    assert torch.equal(out["ste"][1], up)                                  # ste: d y / d code = identity
    assert torch.equal(out["hard"][1], up * H.float())                     # hard: kept entries only
    with pytest.raises(ValueError, match="sparse_grad"):
        sparse_code(code, H, "dense")


def test_hard_blocks_and_ste_passes_gradients_of_dropped_entries():
    """Soft gates through the patcher: a non-member never in the kept set gets NO gradient under hard, and a
    gradient under ste (forward identical to hard)."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(4, soft=True)
    outs, grads = {}, {}
    for grad in ("hard", "ste"):
        p, th, am = _patcher(bank, theta, psi, floor, train_fill="sparse", sparse_grad=grad)
        y = p.transform(*SITE, x)
        (y ** 2).sum().backward()
        outs[grad], grads[grad] = y.detach(), th[SITE].grad.clone()
    assert torch.equal(outs["hard"], outs["ste"])
    ta, ti = bank.encode(x, *SITE[::-1])
    dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    H = sparse_keep_mask(dense, member, LM._amp_of(psi), floor, K)
    never = ~H.reshape(-1, D_SAE).any(0)
    assert int(never.sum()) > 0
    assert bool((grads["hard"][never] == 0).all())
    live = never & ((dense != 0).reshape(-1, D_SAE).any(0) | (floor != 0))
    assert bool((grads["ste"][live] != 0).any())


def test_ste_gradient_equals_the_dense_code_gradient_at_the_sparse_point():
    """Straight-through: the parameter gradient is the dense code's Jacobian applied to the upstream gradient taken
    at the sparse forward (so with a linear readout it equals the dense fill's gradient exactly)."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(2, soft=True)
    w = torch.randn(D_MODEL, generator=torch.Generator().manual_seed(7))
    g = {}
    for fill in ("dense", "sparse"):
        p, th, am = _patcher(bank, theta, psi, floor, train_fill=fill, sparse_grad="ste")
        (p.transform(*SITE, x) @ w).sum().backward()                          # linear readout: upstream grad constant
        g[fill] = (th[SITE].grad.clone(), am[SITE].grad.clone())
    assert torch.allclose(g["dense"][0], g["sparse"][0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(g["dense"][1], g["sparse"][1], atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------------------------------- default path
def test_dense_default_is_the_historical_forward():
    """train_fill="dense" (explicit or default) == the historical code m*alpha*dense + (1-m)*floor, bit for bit."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(2, soft=True)
    for fl in (floor, None):
        a = _patcher(bank, theta, psi, fl, "anneal")[0].transform(*SITE, x)
        b = _patcher(bank, theta, psi, fl, "anneal", train_fill="dense", sparse_grad="hard")[0].transform(*SITE, x)
        assert torch.equal(a, b)
    ta, ti = bank.encode(x, *SITE[::-1])
    dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    m = torch.sigmoid(theta)
    alpha = LM._amp_of(psi)
    want = x + bank.decode(dense * (m * alpha - 1.0) + (1.0 - m) * floor, "resid", 0, add_bias=False)
    assert torch.equal(_patcher(bank, theta, psi, floor)[0].transform(*SITE, x), want)


def test_zero_patcher_untouched_by_sparse():
    """The Z term's patcher (floors=None) is identical under train_fill="sparse"."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(3, soft=True)
    for grad in ("hard", "ste"):
        a = _patcher(bank, theta, psi, None, "anneal", train_fill="sparse", sparse_grad=grad)[0]
        b = _patcher(bank, theta, psi, None, "anneal")[0]
        assert torch.equal(a.transform(*SITE, x), b.transform(*SITE, x))
        assert a.sparse_calls == 0


def test_patcher_refuses_bad_settings_and_pinned_codes():
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(1)
    with pytest.raises(ValueError, match="train_fill"):
        _patcher(bank, theta, psi, floor, train_fill="topk")
    with pytest.raises(ValueError, match="sparse_grad"):
        _patcher(bank, theta, psi, floor, sparse_grad="soft")
    p = LearnedMaskPatcher(bank, {SITE: theta}, 5, "resid", torch.zeros(D_MODEL), torch.zeros(()),
                           floors={SITE: floor}, pins={SITE: floor * 2}, train_fill="sparse")
    with pytest.raises(ValueError, match="without pins"):
        p.transform(*SITE, x)


# ------------------------------------------------------------------------------------------- gate-only (unweighted)
def _gate_patcher(bank, theta, floor, **kw):
    th = {SITE: theta.clone().requires_grad_(True)}
    p = LearnedMaskPatcher(bank, th, 5, "resid", torch.zeros(D_MODEL), torch.zeros(()), code_dtype="stream",
                           floors=None if floor is None else {SITE: floor}, bin_threshold=0.5, **kw)
    return p, th


@pytest.mark.parametrize("grad", ["ste", "hard"])
@pytest.mark.parametrize("soft", [False, True])
def test_gate_only_sparse_equals_weighted_sparse_at_alpha_exactly_one(grad, soft):
    """An unweighted (gate-only) sparse fit is the free-amplitude sparse fit with every alpha exactly 1 and the
    amplitudes frozen: same forward and same gate gradient, bit for bit. (alpha = 1 exactly via the signed
    amplitude, alpha = psi = 1; softplus^-1(1) does not round-trip exactly.)"""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(6, soft=soft)
    LM._SIGNED_AMP[0] = True
    try:
        pw, thw, amw = _patcher(bank, theta, torch.ones(D_SAE), floor, train_fill="sparse", sparse_grad=grad)
        amw[SITE].requires_grad_(False)                                          # frozen amplitudes
        assert torch.equal(LM._amp_of(amw[SITE]), torch.ones(D_SAE))
        yw = pw.transform(*SITE, x)
    finally:
        LM._SIGNED_AMP[0] = False
    pu, thu = _gate_patcher(bank, theta, floor, train_fill="sparse", sparse_grad=grad)
    yu = pu.transform(*SITE, x)
    assert torch.equal(yu, yw)
    w = torch.randn(D_MODEL, generator=torch.Generator().manual_seed(1))
    ((yw @ w) ** 2).sum().backward()
    ((yu @ w) ** 2).sum().backward()
    assert torch.equal(thu[SITE].grad, thw[SITE].grad)
    assert pu.sparse_calls == 1


def test_gate_only_sparse_equals_the_scorers_unweighted_fill():
    """Binary gates, no amplitudes: the gate-only training code == the scorer's respect_topk code at alpha = 1."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(2)
    ones = torch.ones(D_SAE)
    want, dense = _scorer_code(bank, x, member, ones, floor)
    cap = {}
    orig = LM.sparse_code

    def spy(code, H, g):
        y = orig(code, H, g)
        cap["code"] = y.detach()
        return y
    LM.sparse_code = spy
    try:
        p, _ = _gate_patcher(bank, theta, floor, train_fill="sparse")
        got = p.transform(*SITE, x)
    finally:
        LM.sparse_code = orig
    assert torch.equal(cap["code"], want)
    P, _, _ = _scorer(bank, member, ones, floor)
    assert torch.allclose(got, P.transform(*SITE, x), atol=1e-5, rtol=1e-5)
    pd_, _ = _gate_patcher(bank, theta, floor)
    assert not torch.allclose(pd_.transform(*SITE, x), got, atol=1e-3)            # not the dense fill


def test_gate_only_dense_default_unchanged_and_unit_alpha_cached():
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(3, soft=True)
    a = _gate_patcher(bank, theta, floor)[0].transform(*SITE, x)
    b = _gate_patcher(bank, theta, floor, train_fill="dense")[0].transform(*SITE, x)
    ta, ti = bank.encode(x, *SITE[::-1])
    dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    want = x + bank.decode((floor - dense) * (1.0 - torch.sigmoid(theta)), "resid", 0, add_bias=False)
    assert torch.equal(a, b) and torch.equal(a, want)
    p, _ = _gate_patcher(bank, theta, floor, train_fill="sparse")
    u1, u2 = p._unit_alpha(SITE, dense), p._unit_alpha(SITE, dense)
    assert u1 is u2 and torch.equal(u1, torch.ones(D_SAE))
    z = _gate_patcher(bank, theta, None, train_fill="sparse")[0]                   # Z: no floor, untouched
    assert torch.equal(z.transform(*SITE, x), _gate_patcher(bank, theta, None)[0].transform(*SITE, x))
    assert z.sparse_calls == 0


# ------------------------------------------------------------------------------------------- engine wiring
class _Inference:
    """Two-site pipe: the patcher's transform at (0, 'resid'), then the seed tap at (1, 'resid'). tokens[0, 0]
    selects the stored stream (0 = activating, 1 = contrast)."""

    def __init__(self, x_pos, x_neg):
        self.streams = {0: x_pos, 1: x_neg}

    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, patcher=None, activations_callback=None, **kw):
        x = self.streams[int(tokens[0, 0])][: tokens.shape[0]].clone()
        if activations_callback is not None:
            activations_callback(0, (x, x, x))
            return None
        if patcher is not None:
            x = patcher.transform(0, "resid", x)
            patcher.transform(1, "resid", x)


def _engine(**kw):
    g = torch.Generator().manual_seed(11)
    x_pos = torch.randn(8, 4, D_MODEL, generator=g) + 0.5
    x_neg = torch.randn(8, 4, D_MODEL, generator=g)
    inf, bank = _Inference(x_pos, x_neg), FakeBank()
    pt = torch.zeros(8, 4, dtype=torch.long)
    nt = torch.ones(8, 4, dtype=torch.long)
    args = dict(objective="pos", sites=[SITE], seed_layer=1, seed_kind="resid", seed_latent_idx=3, pos_tokens=pt,
                pos_argmax=torch.zeros(8, dtype=torch.long), neg_tokens=nt, steps=12, lr=0.1, l1_lambda=1e-3,
                batch_size=4, log_every=0, mask_floor_source="triple", free_amplitude=True, binarize="anneal",
                triple_floor_weight=0.25, dual_floor_weight=0.25)
    args.update(kw)
    return run_learned_mask(inf, bank, **args)


def test_engine_default_equals_explicit_dense():
    s0, p0 = _engine()
    s1, p1 = _engine(train_fill="dense", sparse_grad="ste")
    assert s0 == s1 and p0["curve"] == p1["curve"] and p0["amp_kept"] == p1["amp_kept"]
    assert p0["train_fill"] == "dense" and p0["sparse_grad"] is None and p0["sparse_fill_calls"] is None


def test_engine_sparse_runs_on_the_floored_terms_only():
    calls = []
    orig = LM.LearnedMaskPatcher.__init__

    def spy(self, *a, **k):
        orig(self, *a, **k)
        calls.append((self.floors is not None and bool(self.floors), self.train_fill))
    LM.LearnedMaskPatcher.__init__ = spy
    try:
        s, p = _engine(train_fill="sparse", sparse_grad="ste")
    finally:
        LM.LearnedMaskPatcher.__init__ = orig
    assert p["train_fill"] == "sparse" and p["sparse_grad"] == "ste" and p["sparse_fill_calls"] > 0
    assert any(f and tf == "sparse" for f, tf in calls)                     # the C / A patchers
    assert all(tf == "dense" for f, tf in calls if not f)                   # Z and the no-floor helpers
    _, pd_ = _engine()
    assert p["curve"]["loss"] != pd_["curve"]["loss"]                       # the fill really changed the terms


def test_engine_unweighted_sparse_runs_and_dense_unweighted_is_unchanged():
    """Unweighted (free_amplitude=False) fits take train_fill="sparse" at alpha = 1; the dense default is unchanged."""
    s0, p0 = _engine(free_amplitude=False)
    s1, p1 = _engine(free_amplitude=False, train_fill="dense")
    assert s0 == s1 and p0["curve"] == p1["curve"]
    s2, p2 = _engine(free_amplitude=False, train_fill="sparse")
    assert p2["train_fill"] == "sparse" and p2["sparse_fill_calls"] > 0 and p2["free_amplitude"] is False
    assert p2["curve"]["loss"] != p0["curve"]["loss"]
    assert p2["curve"]["terms"]["zero"][0] == p0["curve"]["terms"]["zero"][0]     # Z term: same fill at step 0


def test_engine_refuses_sparse_where_undefined():
    with pytest.raises(ValueError, match="no pins"):
        _engine(train_fill="sparse", free_amplitude=False, objective="pin",
                pin_values={SITE: torch.ones(D_SAE)})
    with pytest.raises(ValueError, match="mean floor"):
        _engine(train_fill="sparse", mask_floor_source="zero")
    with pytest.raises(ValueError, match="train_fill"):
        _engine(train_fill="topk")
    with pytest.raises(ValueError, match="sparse_grad"):
        _engine(sparse_grad="none")


def test_engine_precomputed_means_reach_the_patchers():
    """eval_ref's means enter through floor_means (C) / pos_floor_means (A): the floors the patchers train against."""
    seen = []
    orig = LM.LearnedMaskPatcher.__init__

    def spy(self, *a, **k):
        orig(self, *a, **k)
        if self.floors:
            seen.append(self.floors[SITE])
    fC, fA = torch.full((D_SAE,), 0.25), torch.full((D_SAE,), 0.5)
    LM.LearnedMaskPatcher.__init__ = spy
    try:
        _, p = _engine(train_fill="sparse", floor_means={SITE: fC}, pos_floor_means={SITE: fA})
    finally:
        LM.LearnedMaskPatcher.__init__ = orig
    assert p["floors_precomputed"] is True
    assert any(torch.equal(f, fC) for f in seen) and any(torch.equal(f, fA) for f in seen)


# ------------------------------------------------------------------------------------------- per-position means
# Experiment 107 (2026-11-01): fixed-length task prompts can take PER-POSITION A / C means [T, d_sae] (SFC's ablation
# convention). The training fill (sparse_keep_mask) and the scorer's fill (_respect_topk_fill) each gain a 2-D branch;
# the [d_sae] paths are untouched (every test above).
def _floor_tp(seed=7):
    g = torch.Generator().manual_seed(seed)
    f = torch.rand(T, D_SAE, generator=g) * 0.5
    f[torch.rand(T, D_SAE, generator=g) < 0.3] = 0.0
    return f


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_per_position_scorer_fill_is_the_one_d_fill_at_each_position(seed):
    """No ties: position t of the [T, d] fill == the [d] fill with floor[t] on that position alone."""
    bank = FakeBank()
    x, _, member, psi, _ = _setup(seed, ties=False)
    alpha_eff = LM._amp_of(psi).detach()
    ftp = _floor_tp(seed)
    got, dense = _scorer_code(bank, x, member, alpha_eff, ftp)
    # one encoding for both (re-encoding a single position rounds the matmul differently, ~1e-6)
    want = torch.cat([_scorer_code(bank, None, member, alpha_eff, ftp[t], dense=dense[:, t:t + 1])[0]
                      for t in range(T)], dim=1)
    assert torch.equal(got, want)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("ties", [False, True])
def test_per_position_training_fill_equals_scorer_fill(seed, ties):
    """Binary gates: the training pass's per-position sparsity-preserving code == the scorer's, exactly."""
    bank = FakeBank()
    x, _, member, psi, theta = _setup(seed, ties=ties)
    ftp = _floor_tp(seed)
    if ties:
        ftp = (ftp * 10).round() / 10
    alpha_eff = LM._amp_of(psi)
    want, dense = _scorer_code(bank, x, member, alpha_eff, ftp)
    m = torch.sigmoid(theta)
    code = m * alpha_eff * dense + (1.0 - m) * ftp
    got = sparse_code(code, sparse_keep_mask(dense, m > 0.5, alpha_eff, ftp.to(dense.dtype), K), "hard")
    assert torch.equal(got, want)
    # and the whole site transform, engine vs scorer
    p, _, _ = _patcher(bank, theta, psi, ftp, train_fill="sparse", sparse_grad="ste")
    P, _, _ = _scorer(bank, member, alpha_eff.detach(), ftp)
    assert torch.allclose(p.transform(*SITE, x), P.transform(*SITE, x), atol=1e-5, rtol=1e-5)


def test_per_position_dense_fill_broadcasts():
    """train_fill="dense" with a [T, d] floor: code = m alpha dense + (1 - m) floor[t] at position t."""
    bank = FakeBank()
    x, _, member, psi, theta = _setup(2, soft=True)
    ftp = _floor_tp(2)
    ta, ti = bank.encode(x, *SITE[::-1])
    dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    m, alpha = torch.sigmoid(theta), LM._amp_of(psi)
    want = x + bank.decode(dense * (m * alpha - 1.0) + (1.0 - m) * ftp.unsqueeze(0), "resid", 0, add_bias=False)
    assert torch.allclose(_patcher(bank, theta, psi, ftp)[0].transform(*SITE, x), want, atol=1e-6)


def test_per_position_length_mismatch_refused():
    bank = FakeBank()
    x, _, member, psi, _ = _setup(1)
    bad = torch.rand(T + 1, D_SAE)
    with pytest.raises(ValueError, match="positions"):
        _scorer_code(bank, x, member, LM._amp_of(psi).detach(), bad)
    ta, ti = bank.encode(x, *SITE[::-1])
    dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    with pytest.raises(ValueError, match="positions"):
        sparse_keep_mask(dense, member, LM._amp_of(psi), bad, K)


@pytest.mark.parametrize("per_position", [False, True])
def test_full_width_k_keeps_every_kept_value(per_position):
    """k = d_sae (a ReLU dictionary): the budget exceeds the non-kept count. The scorer's fill must never overwrite a
    kept value with its mean (the [d] path's cap, 2026-11-01), and it equals the plain mean fill: kept at their
    values, every other latent at its mean; the training fill agrees."""
    bank = FakeBank()
    x, floor, member, psi, _ = _setup(3, ties=False)
    fl = _floor_tp(3) if per_position else floor
    alpha_eff = LM._amp_of(psi).detach()
    P, kt, sv = _scorer(bank, member, alpha_eff, fl)
    P.topk = D_SAE
    ta, ti = bank.encode(x, *SITE[::-1])
    dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    vals = dense[:, :, kt] * sv[kt]
    got = P._respect_topk_fill(dense, fl.to(dense.dtype), kt, vals)
    want = (fl.unsqueeze(0) if per_position else fl.view(1, 1, -1)).expand_as(dense).clone()
    want[:, :, kt] = vals
    assert torch.equal(got, want)
    m = member.float()
    code = m * alpha_eff * dense + (1.0 - m) * (fl.unsqueeze(0) if per_position else fl)
    tr = sparse_code(code, sparse_keep_mask(dense, member, alpha_eff, fl, D_SAE), "hard")
    assert torch.equal(tr, want)


# ------------------------------------------------------------------------------------------- eval_ref means
def _record(n_held=16, v2=None):
    g = torch.Generator().manual_seed(5)

    def pool(n):
        return dict(pos=torch.randint(2, 100, (n, 64), generator=g), tgt=torch.randint(2, 100, (n, 64), generator=g),
                    arg=torch.randint(0, 64, (n,), generator=g))
    strong = dict(pool(48 + n_held), train=list(range(48)), held=list(range(48, 48 + n_held)))
    mid = dict(pool(32), held=list(range(16, 32)))
    rec = dict(strong=strong, mid=mid, neg=torch.randint(2, 100, (64, 64), generator=g),
               D_train=list(range(32)), B_mid=list(range(16)))
    if v2 is not None:
        rec["neg_held_v2"] = torch.randint(2, 100, (v2, 64), generator=g)
    return rec


@pytest.mark.parametrize("split", ["explicit", "as_run"])
@pytest.mark.parametrize("v2", [None, 9])
def test_eval_reference_is_the_scorers_reference(split, v2):
    """A rows = the 48 strongest-train rows, C rows = the 48 contrast-train rows, in the scorer's order; and the
    scorer's own slicing (wcm_eval.eval_split / contrast_split on eval_probe) agrees."""
    from circuit import protocol_contexts as PC
    from eval.wcm_eval import contrast_split, eval_split
    rec = _record(v2=v2)
    a, c = PC.eval_reference(rec, "cpu", split=split, holdout_frac=0.25)
    assert torch.equal(a, rec["strong"]["pos"][rec["strong"]["train"]])
    assert torch.equal(c, rec["neg"][:48])
    pd_ = PC.eval_probe(rec, "strong", "cpu", 0.25)
    assert torch.equal(a, pd_.pos_tokens[:eval_split(int(pd_.pos_tokens.shape[0]), pd_.metadata, split)])
    assert torch.equal(c, pd_.neg_tokens[:contrast_split(int(pd_.neg_tokens.shape[0]), pd_.metadata)])


def test_eval_reference_as_run_shrinks_with_a_short_held_out_set():
    """as_run (the 15k run's 3:1 rule on the eval probe) takes fewer reference rows when the held-out set is short,
    exactly as the as-run scorer does; explicit always takes the 48."""
    from circuit import protocol_contexts as PC
    rec = _record(n_held=8)
    a_run, _ = PC.eval_reference(rec, "cpu", split="as_run", holdout_frac=0.25)
    a_exp, _ = PC.eval_reference(rec, "cpu", split="explicit", holdout_frac=0.25)
    assert int(a_exp.shape[0]) == 48 and int(a_run.shape[0]) == 56 - 14


def test_eval_ref_means_are_the_scorers_means():
    """_eval_ref_means == the scorers' calls on the same rows: A = collect_site_anchors(ref rows)[0] (means_tr),
    C = collect_site_means(contrast-train rows) (means_neg); bit for bit."""
    from circuit.discovery.ablation_gradient import AblationGradientDiscovery
    from eval.floors import collect_site_anchors, collect_site_means
    g = torch.Generator().manual_seed(2)
    x_pos, x_neg = torch.randn(6, 4, D_MODEL, generator=g), torch.randn(6, 4, D_MODEL, generator=g)
    inf, bank = _Inference(x_pos, x_neg), FakeBank()
    M = object.__new__(AblationGradientDiscovery)
    M.inference, M.sae_bank = inf, bank
    pos_ref, neg_ref = torch.zeros(6, 4, dtype=torch.long), torch.ones(6, 4, dtype=torch.long)
    sites = [SITE, (0, "mlp")]
    cfg = SimpleNamespace(mask_floor_source="triple")
    with pytest.raises(ValueError, match="_train_means_ref"):
        M._eval_ref_means(sites, cfg)
    M._train_means_ref = (pos_ref, neg_ref)
    out = M._eval_ref_means(sites, cfg)
    means_tr, _ = collect_site_anchors(inf, bank, pos_ref, set(sites), torch.zeros(6, dtype=torch.long),
                                       pin_position_specific=False)
    means_neg = collect_site_means(inf, bank, neg_ref, set(sites))
    for s in sites:
        assert torch.equal(out["pos_floor_means"][s], means_tr[s])
        assert torch.equal(out["floor_means"][s], means_neg[s])
    assert not torch.equal(out["pos_floor_means"][SITE], out["floor_means"][SITE])
    assert set(M._eval_ref_means(sites, SimpleNamespace(mask_floor_source="dual"))) == {"floor_means"}
    with pytest.raises(ValueError, match="contrast-floored"):
        M._eval_ref_means(sites, SimpleNamespace(mask_floor_source="posctx"))
