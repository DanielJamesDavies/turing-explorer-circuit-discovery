"""095 CPU tests (no model) for rs2_engine:
  1. the patched source applies every anchor once, compiles and installs; switches exist
  2. with every switch off, the A patcher (native v2 sparse fill) and the Z patcher are bit-identical to the unpatched
     src module's
  3. donor patcher, sp_donor off: code = m*alpha*dense + (1-m)*donor (079 / 091 raw R)
  4. donor patcher, sp_donor on, binary gates: patched code == 073 ResamplePatcher's fill exactly; backward is
     straight-through (gradients equal to the raw-R gradients)
  5. the drop_floor edit is in the training loop source

  wsl: PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/095-v2-resample-appendix/test_rs2_engine.py
"""
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rs2_engine  # noqa: E402
import rs_engine  # noqa: E402


class FakeBank:
    def __init__(self, d=16, D=64, k=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.d_sae, self.k = D, k
        self.We = torch.randn(D, d, generator=g) / d ** 0.5
        self.be = torch.randn(D, generator=g) * 0.1
        self.Wd = torch.randn(d, D, generator=g) / D ** 0.5
        self.bd = torch.randn(d, generator=g) * 0.1

    def encode(self, x, kind, layer):
        pre = torch.relu(x @ self.We.T.to(x.dtype) + self.be.to(x.dtype))
        v, i = pre.topk(self.k, dim=-1)
        return v, i

    def decode(self, code, kind, layer, add_bias=True):
        out = code @ self.Wd.T.to(code.dtype)
        return out + self.bd.to(code.dtype) if add_bias else out


SITE = (2, "mlp")


class IdBank(FakeBank):
    """Decoder = identity on a d = D space, so transform(x) - x == code - dense exactly."""

    def __init__(self, D=24, k=5, seed=3):
        super().__init__(d=D, D=D, k=k, seed=seed)
        self.Wd = torch.eye(D)
        self.bd = torch.zeros(D)


def test_install():
    src = rs2_engine.patched_source()
    for anchor, repl in rs2_engine.EDITS:
        assert repl in src
    mod = rs2_engine.install("095test")
    assert set(mod.SPEED) == {"drop_floor", "sp_donor"} and hasattr(mod, "RESAMPLE") and hasattr(mod, "SILENT")
    assert hasattr(mod, "sp_keep_mask_donor") and hasattr(mod, "_sp_delta_donor") and hasattr(mod, "sparse_code")
    assert "SPEED[\"drop_floor\"]" in src and "terms = [t_ for t_ in terms if t_[4] is not patcher]" in src
    print("install OK")
    return mod


def test_off_identical(mod):
    """Every switch off: A-type (floored, train_fill sparse, ste) and Z-type patchers == the unpatched src module."""
    sys.modules.pop(rs2_engine.MODNAME, None)
    import circuit.instrument as pkg
    orig = importlib.import_module(rs2_engine.MODNAME)
    torch.manual_seed(1)
    bank = FakeBank()
    D, B, T, d = bank.d_sae, 3, 5, 16
    th = torch.randn(D) * 3
    ps = torch.randn(D) * 0.3
    fA = torch.rand(D) * 0.5
    x = torch.randn(B, T, d)
    outs, grads = [], []
    for m_ in (orig, mod):
        th_, ps_ = th.clone().requires_grad_(), ps.clone().requires_grad_()
        for floors, fill in ((None, "dense"), ({SITE: fA}, "sparse"), ({SITE: fA}, "dense")):
            p = m_.LearnedMaskPatcher(bank, {SITE: th_}, 5, "resid", torch.zeros(d), torch.zeros(()), amps={SITE: ps_},
                                      code_dtype="stream", floors=floors, binarize="anneal", bin_threshold=0.5,
                                      train_fill=fill, sparse_grad="ste")
            p.temperature = 0.3
            y = p.transform(2, "mlp", x)
            outs.append(y.detach())
            y.sum().backward()
        grads.append((th_.grad.clone(), ps_.grad.clone()))
    n = len(outs) // 2
    for a, b in zip(outs[:n], outs[n:]):
        assert torch.equal(a, b)
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])
    pkg.learned_mask = mod
    sys.modules[rs2_engine.MODNAME] = mod
    print("switches off == unpatched src (Z, A-sparse, A-dense): bit-identical OK")


def _donor_setup(mod, sp_donor, binary=True):
    torch.manual_seed(2)
    bank = IdBank()
    D, B, T, K = bank.d_sae, 4, 7, bank.k
    x = torch.randn(B, T, D)
    donor_x = torch.randn(B, T, D)
    dta, dti = bank.encode(donor_x, "mlp", 2)
    codes = {SITE: (dta.clone(), dti.clone())}
    floors = rs_engine.DonorFloors(codes, D, "cpu", seed=0)
    floors.rows = torch.arange(B)                              # donor row i for training row i (fixed for the test)
    kt = torch.tensor([1, 5, 9, 17, 20])
    th = torch.full((D,), -8.0)
    th[kt] = 8.0
    if not binary:
        th = th + torch.randn(D) * 0.5
    psi = torch.randn(D) * 0.3
    th, psi = th.requires_grad_(), psi.requires_grad_()
    make = rs_engine.make_factory(mod, floors)
    p = make(bank, {SITE: th}, 5, "resid", torch.zeros(D), torch.zeros(()), {}, {SITE: psi}, "stream", "anneal", 0.5)
    p.temperature = 0.05
    mod.SPEED["sp_donor"] = sp_donor
    y = p.transform(2, "mlp", x)
    mod.SPEED["sp_donor"] = False
    ta, ti = bank.encode(x, "mlp", 2)
    dense = torch.zeros(B, T, D).scatter_(-1, ti, ta)
    donor = torch.zeros(B, T, D).scatter_(-1, dti, dta)
    return dict(bank=bank, x=x, y=y, dense=dense, donor=donor, dta=dta, dti=dti, kt=kt, th=th, psi=psi, K=K,
                alpha=mod._amp_of(psi).detach(), p=p)


def test_donor_raw(mod):
    s = _donor_setup(mod, sp_donor=False, binary=False)
    m = torch.sigmoid(s["th"].detach() / 0.05)
    want = m * s["alpha"] * s["dense"] + (1 - m) * s["donor"]
    got = s["y"].detach() - s["x"] + s["dense"]
    diff = float((got - want).abs().max())
    assert diff < 1e-5, diff
    print("donor raw R (sp_donor off): max |diff| %.2g OK" % diff)


def test_donor_sp(mod):
    s = _donor_setup(mod, sp_donor=True)
    dense, kt, alpha, K = s["dense"], s["kt"], s["alpha"], s["K"]
    dta, dti = s["dta"], s["dti"]
    # 073 ResamplePatcher fill (strict_eval.py, its logic verbatim)
    patched = torch.zeros_like(dense)
    patched[:, :, kt] = dense[:, :, kt] * alpha[kt]
    valid = (dta > 0) & ~torch.isin(dti, kt)
    budget = (K - (patched != 0).sum(-1)).clamp(min=0)
    order = dta.masked_fill(~valid, float("-inf")).argsort(-1, descending=True)
    dta_s, dti_s, val_s = dta.gather(-1, order), dti.gather(-1, order), valid.gather(-1, order)
    rank = val_s.long().cumsum(-1) - 1
    use = val_s & (rank < budget.unsqueeze(-1))
    patched.scatter_add_(-1, dti_s, dta_s * use.to(dta_s.dtype))
    got = s["y"].detach() - s["x"] + dense
    diff = float((got - patched).abs().max())
    assert diff < 1e-6, diff
    assert int(use.sum()) > 0 and mod.DONOR_STATS["calls"] >= 1
    # straight-through: gradients equal the raw-R gradients
    w = torch.randn_like(s["y"])
    (s["y"] * w).sum().backward()
    g_sp = (s["th"].grad.clone(), s["psi"].grad.clone())
    r = _donor_setup(mod, sp_donor=False)
    (r["y"] * w).sum().backward()
    assert torch.allclose(g_sp[0], r["th"].grad, atol=1e-6) and torch.allclose(g_sp[1], r["psi"].grad, atol=1e-6)
    print("donor sp fill == 073 ResamplePatcher fill (max |diff| %.2g, %d donor entries filled); ste grads == raw OK"
          % (diff, int(use.sum())))


def main():
    mod = test_install()
    test_off_identical(mod)
    test_donor_raw(mod)
    test_donor_sp(mod)
    print("ALL OK")


if __name__ == "__main__":
    main()
