"""CPU tests for 079's patched engine (no model, no GPU).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/079-resample-training/test_rs_engine.py -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rs_engine  # noqa: E402

D_MODEL, D_SAE, K = 8, 32, 4
SITE = (0, "resid")


class FakeBank:
    """Top-K SAE: encode = top-k of relu(x W^T + b); decode = code W_dec^T (+ bias)."""
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


def test_edits_apply_once_and_compile():
    src = rs_engine.patched_source()
    compile(src, "<patched>", "exec")
    for _, repl in rs_engine.EDITS:
        assert src.count(repl) == 1
    assert "RESAMPLE = {" in src


def test_install_and_weight_zero_builds_nothing():
    mod = rs_engine.install()
    assert mod.RESAMPLE == {"weight": 0.0, "make": None}
    assert mod.SILENT == {"weight": 0.0, "fn": None}
    assert sys.modules[rs_engine.MODNAME] is mod


# ----------------------------------------------------------------------------------------------- silent positions
def test_silent_penalty_zero_when_not_lifted():
    pre = torch.tensor([[-1.0, 0.5, 2.0]])
    floor = torch.tensor([[0.0, 0.5, 3.0]])          # pre <= floor everywhere
    silent = torch.ones(1, 3, dtype=torch.bool)
    assert float(rs_engine.silent_penalty(pre, silent, floor, 1.0)) == 0.0


def test_silent_penalty_value_and_masking():
    pre = torch.tensor([[3.0, 2.0, 10.0, 1.0], [0.0, 4.0, 0.0, 0.0]])
    floor = torch.zeros(2, 4)
    floor[0, 1] = 1.0                                # natural pre-activation 1 there: only the lift above it counts
    silent = torch.tensor([[True, True, False, True], [True, True, True, False]])   # [0, 2] is a firing / anchor
    # row 0: excess (3, 1, -, 1) -> (9 + 1 + 1) / 3 ; row 1: (0, 4, 0) -> 16 / 3 ; mean, x weight 0.5
    want = 0.5 * ((11 / 3 + 16 / 3) / 2)
    got = float(rs_engine.silent_penalty(pre, silent, floor, 0.5))
    assert abs(got - want) < 1e-6


def test_silent_penalty_scales_like_the_anchor_error():
    """The engine divides ALL of data_loss by mean(target^2) once. So the silent term must scale as c^2 when the
    target's scale is multiplied by c, exactly like the anchor squared error, for the normalised total to be scale-free.
    (The 2026-10-18 bug normalised it a second time: it scaled as c^0 and dominated weak targets.)"""
    g = torch.Generator().manual_seed(5)
    pre, floor = torch.randn(3, 8, generator=g) * 2, torch.rand(3, 8, generator=g)
    tgt = torch.rand(3, generator=g) + 1
    silent = torch.rand(3, 8, generator=g) > 0.3
    for c in (0.1, 10.0):
        s1 = float(rs_engine.silent_penalty(pre, silent, floor, 1.0))
        sc = float(rs_engine.silent_penalty(c * pre, silent, c * floor, 1.0))
        a1, ac = float(((pre[:, 0] - tgt) ** 2).mean()), float(((c * pre[:, 0] - c * tgt) ** 2).mean())
        assert abs(sc / s1 - c ** 2) < 1e-4 * c ** 2
        assert abs((sc / s1) - (ac / a1)) < 1e-4 * c ** 2


def test_silent_penalty_has_gradient_only_where_charged():
    pre = torch.tensor([[3.0, -1.0, 5.0]], requires_grad=True)
    silent = torch.tensor([[True, True, False]])
    rs_engine.silent_penalty(pre, silent, torch.zeros(1, 3), 1.0).backward()
    g = pre.grad[0]
    assert g[0] > 0 and g[1] == 0 and g[2] == 0      # lifted silent / below floor / not silent


def test_parse_arms():
    import importlib
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    fs = importlib.import_module("fit_score")
    assert fs.parse_arms(["W0", "RS1", "RS0.25_S0.25", "W0_S1"]) == [
        ("W0", 0.0, 0.0), ("RS1", 1.0, 0.0), ("RS0.25_S0.25", 0.25, 0.25), ("W0_S1", 0.0, 1.0)]
    assert [a for a, _, _ in fs.parse_arms(["RS"])] == ["RS%g" % w for w in fs.WEIGHTS]


def _floors(n=5, T=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    acts = torch.rand(n, T, K, generator=g)
    idx = torch.stack([torch.randperm(D_SAE, generator=g)[:K] for _ in range(n * T)]).reshape(n, T, K)
    return {SITE: (acts, idx)}


def test_donor_floors_densify_the_drawn_rows():
    codes = _floors()
    f = rs_engine.DonorFloors(codes, D_SAE, "cpu", seed=3)
    f.draw(4)
    d = f.get(SITE)
    assert d.shape == (4, 3, D_SAE)
    for b, r in enumerate(f.rows.tolist()):
        ref = torch.zeros(3, D_SAE).scatter_(-1, codes[SITE][1][r], codes[SITE][0][r])
        assert torch.equal(d[b], ref)
    assert f.get((9, "attn")) is None and bool(f)


def _patcher(theta_val, amp_val):
    mod = rs_engine.install()
    floors = rs_engine.DonorFloors(_floors(), D_SAE, "cpu", seed=3)
    make = rs_engine.make_factory(mod, floors)
    thetas = {SITE: torch.full((D_SAE,), float(theta_val))}
    # softplus(psi) = amp_val
    psi = torch.log(torch.expm1(torch.tensor(float(amp_val))))
    amps = {SITE: torch.full((D_SAE,), float(psi))}
    p = make(FakeBank(), thetas, 5, "resid", torch.zeros(D_MODEL), torch.zeros(()), {}, amps, "fp32", "none", 0.5)
    return p, floors


def test_open_gate_unit_amplitude_is_identity():
    p, floors = _patcher(40.0, 1.0)                 # m = 1, alpha = 1: code == dense, no donor contribution
    p.on_forward(2)
    x = torch.randn(2, 3, D_MODEL)
    assert torch.allclose(p.transform(*SITE, x), x, atol=1e-5)


def test_closed_gate_lands_on_the_donor_code():
    p, floors = _patcher(-40.0, 1.0)                # m = 0: code == donor code (the resample fill)
    p.on_forward(2)
    x = torch.randn(2, 3, D_MODEL)
    bank = FakeBank()
    ta, ti = bank.encode(x, *SITE[::-1])
    dense = torch.zeros(2, 3, D_SAE).scatter_(-1, ti, ta)
    donor = floors.get(SITE)
    want = x + bank.decode(donor - dense, None, None, add_bias=False)
    assert torch.allclose(p.transform(*SITE, x), want, atol=1e-5)


def test_donors_are_redrawn_every_forward():
    p, floors = _patcher(-40.0, 1.0)
    p.on_forward(2); r1 = floors.rows.clone()
    p.on_forward(2); r2 = floors.rows.clone()
    assert floors.n_draws == 2 and not torch.equal(r1, r2)
