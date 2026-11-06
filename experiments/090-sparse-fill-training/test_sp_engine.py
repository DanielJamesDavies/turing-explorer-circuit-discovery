"""CPU tests for 090's patched engine (no model, no GPU): the training-pass sparsity-preserving fill equals the scorer's.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/090-sparse-fill-training/test_sp_engine.py -q
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sp_engine  # noqa: E402

D_MODEL, D_SAE, K = 16, 64, 8
B, T = 3, 5
SITE = (0, "resid")


class FakeBank:
    """Top-K SAE: encode = top-k of relu(x W^T); decode = code W_dec^T (+ bias)."""
    d_sae, k = D_SAE, K

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(D_SAE, D_MODEL, generator=g)
        self.Wd = torch.randn(D_MODEL, D_SAE, generator=g)
        self.bd = torch.randn(D_MODEL, generator=g)

    def encode(self, x, kind, layer):
        return torch.relu(x @ self.W.T).topk(K, dim=-1)

    def decode(self, code, kind, layer, add_bias=True):
        return code @ self.Wd.T + (self.bd if add_bias else 0)


@pytest.fixture(scope="module")
def mod():
    return sp_engine.install()


@pytest.fixture(scope="module")
def orig():
    """The unpatched src engine, exec'd under its own name."""
    m = types.ModuleType("lm_orig_090")
    m.__file__ = str(sp_engine.SRC)
    m.__package__ = "circuit.instrument"
    exec(compile(sp_engine.SRC.read_text(encoding="utf-8"), "<orig>", "exec"), m.__dict__)
    return m


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


def _engine_patcher(M, bank, theta, psi, floor, binarize="none"):
    th = {SITE: theta.clone().requires_grad_(True)}
    am = {SITE: psi.clone().requires_grad_(True)}
    p = M.LearnedMaskPatcher(bank, th, 5, "resid", torch.zeros(D_MODEL), torch.zeros(()), amps=am,
                             code_dtype="stream", floors=None if floor is None else {SITE: floor},
                             binarize=binarize, bin_threshold=0.5)
    return p, th, am


def _scorer(bank, member, alpha_eff, floor):
    from eval.ablation_faithfulness import CircuitOnlyPatcher
    keep = sorted(member.nonzero().flatten().tolist())
    sv = torch.ones(D_SAE)
    sv[keep] = alpha_eff[keep]
    return CircuitOnlyPatcher(bank=bank, keep_indices={SITE: set(keep)}, in_scope={SITE}, seed_layer=-1,
                              seed_kind="", seed_latent_idx=0, site_means={SITE: floor}, respect_topk=True, topk=K,
                              keep_scales={SITE: sv}), torch.tensor(keep, dtype=torch.long), sv


def _scorer_code(bank, x, member, alpha_eff, floor, dense=None):
    """The scorer's patched code (respect_topk fill) on the same inputs."""
    from sae.dense import sparse_topk_to_dense
    P, kt, sv = _scorer(bank, member, alpha_eff, floor)
    if dense is None:
        ta, ti = bank.encode(x, *SITE[::-1])
        dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    vals = dense[:, :, kt] * sv[kt]
    return P._respect_topk_fill(dense, floor.to(dense.dtype), kt, vals), dense


def test_edits_apply_once_and_compile():
    src = sp_engine.patched_source()
    compile(src, "<patched>", "exec")
    for _, repl in sp_engine.EDITS:
        assert src.count(repl) == 1
    assert "SPFILL = {" in src


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["hard", "ste"])
def test_training_fill_code_equals_scorer_fill(mod, seed, mode):
    """Binary gates: the training pass's sparsity-preserving code == the scorer's patched code, exactly (with ties)."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(seed)
    alpha_eff = mod._amp_of(psi)
    want, dense = _scorer_code(bank, x, member, alpha_eff, floor)
    m = torch.sigmoid(theta)
    assert torch.equal(m, member.float())
    code = m * alpha_eff * dense + (1.0 - m) * floor                     # the soft code, written as the code
    H = mod.sp_keep_mask(dense, m > 0.5, alpha_eff, floor.to(dense.dtype), K)
    got = mod.sp_code(code, H, mode)
    assert torch.equal(got, want)
    # the engine builds the code as delta + dense (delta = dense * (m * alpha - 1) + (1 - m) * floor): equal up to
    # rounding only (the engine's own "algebraic, not bitwise" form)
    code_eng = (dense * (m * alpha_eff - 1.0) + (1.0 - m) * floor) + dense
    assert torch.allclose(mod.sp_code(code_eng, H, mode), want, atol=1e-6, rtol=1e-6)
    # members are never evicted, and each position keeps at most max(K, #active members) nonzero entries
    assert bool(H[..., member].all())
    n_act = ((dense != 0) & member).sum(-1)
    assert bool(((got != 0).sum(-1) <= torch.maximum(n_act, torch.full_like(n_act, K))).all())


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_training_transform_equals_scorer_transform(mod, seed):
    """The whole site transform (engine: x + decode(code_sp - dense, no bias); scorer: decode(patched) + error)."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(seed)
    sp_engine.set_mode(mod, "hard")
    p, _, _ = _engine_patcher(mod, bank, theta, psi, floor)
    got = p.transform(*SITE, x)
    P, _, _ = _scorer(bank, member, mod._amp_of(psi).detach(), floor)
    want = P.transform(*SITE, x)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
    sp_engine.set_mode(mod, "dense")


def test_more_active_members_than_K(mod):
    """Budget clamps at 0: every active member is kept, no mean fill at that position (the scorer's rule)."""
    g = torch.Generator().manual_seed(9)
    x, floor, member, psi, theta = _setup(5, n_mem=40)
    alpha_eff = mod._amp_of(psi)
    dense = torch.rand(B, T, D_SAE, generator=g) * (torch.rand(B, T, D_SAE, generator=g) < 0.4)   # ~26 active / pos
    dense[0, 0, member] = 1.0                                         # 40 active members > K at (0, 0)
    want, _ = _scorer_code(FakeBank(), x, member, alpha_eff, floor, dense=dense)
    m = member.float()
    code = m * alpha_eff * dense + (1.0 - m) * floor
    got = mod.sp_code(code, mod.sp_keep_mask(dense, member, alpha_eff, floor, K), "hard")
    assert torch.equal(got, want)
    assert int((got[0, 0] != 0).sum()) == 40 and bool((got[0, 0, ~member] == 0).all())


def test_dense_mode_is_bitwise_production(mod, orig):
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(2, soft=True)
    sp_engine.set_mode(mod, "dense")
    for fl in (floor, None):
        a = _engine_patcher(mod, bank, theta, psi, fl, "anneal")[0].transform(*SITE, x)
        b = _engine_patcher(orig, bank, theta, psi, fl, "anneal")[0].transform(*SITE, x)
        assert torch.equal(a, b)


def test_zero_patcher_untouched_by_sp_mode(mod, orig):
    """The Z term's patcher (floors=None) is identical to production under every mode."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(3, soft=True)
    for mode in ("hard", "ste", "curriculum"):
        sp_engine.set_mode(mod, mode, switch=0)
        a = _engine_patcher(mod, bank, theta, psi, None, "anneal")[0].transform(*SITE, x)
        b = _engine_patcher(orig, bank, theta, psi, None, "anneal")[0].transform(*SITE, x)
        assert torch.equal(a, b)
    sp_engine.set_mode(mod, "dense")


def test_hard_blocks_and_ste_passes_gradients_of_dropped_entries(mod):
    """Soft gates: a non-member that is never in the kept set gets NO gradient through this term under hard, and a
    gradient under ste (forward identical to hard)."""
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(4, soft=True)
    outs, grads = {}, {}
    for mode in ("hard", "ste"):
        sp_engine.set_mode(mod, mode)
        p, th, am = _engine_patcher(mod, bank, theta, psi, floor)
        y = p.transform(*SITE, x)
        (y ** 2).sum().backward()
        outs[mode], grads[mode] = y.detach(), th[SITE].grad.clone()
    sp_engine.set_mode(mod, "dense")
    assert torch.equal(outs["hard"], outs["ste"])
    from sae.dense import sparse_topk_to_dense
    ta, ti = bank.encode(x, *SITE[::-1])
    dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=x.dtype)
    H = mod.sp_keep_mask(dense, member, mod._amp_of(psi), floor, K)
    never = ~H.reshape(-1, D_SAE).any(0)
    assert int(never.sum()) > 0
    assert bool((grads["hard"][never] == 0).all())
    live = never & ((dense != 0).reshape(-1, D_SAE).any(0) | (floor != 0))
    assert bool((grads["ste"][live] != 0).any())


def test_curriculum_switch(mod):
    bank = FakeBank()
    x, floor, member, psi, theta = _setup(1, soft=True)
    ys = {}
    for mode, step in (("dense", 0), ("hard", 0), ("curriculum", 199), ("curriculum", 200)):
        sp_engine.set_mode(mod, mode, switch=200)
        mod.SPFILL["step"] = step
        ys[(mode, step)] = _engine_patcher(mod, bank, theta, psi, floor)[0].transform(*SITE, x).detach()
    sp_engine.set_mode(mod, "dense")
    assert torch.equal(ys[("curriculum", 199)], ys[("dense", 0)])
    assert torch.equal(ys[("curriculum", 200)], ys[("hard", 0)])
    assert not torch.equal(ys[("dense", 0)], ys[("hard", 0)])
