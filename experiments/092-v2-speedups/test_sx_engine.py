"""092 CPU tests (no model): both patched sources compile and install; CombinedPatcher == the three separate patchers
(forward and gradients, dense and ste fills); the donor sparsity-preserving keep mask == 073 ResamplePatcher's fill.

  wsl: PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/092-v2-speedups/test_sx_engine.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sx_engine  # noqa: E402
import sp_engine  # noqa: E402


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


def test_combined(mod, mode):
    torch.manual_seed(1)
    bank = FakeBank()
    D, B, T, d = bank.d_sae, 3, 5, 16
    site = (2, "mlp")
    th = torch.randn(D) * 3
    th[:10] = 5.0
    ps = torch.randn(D) * 0.3
    fC, fA = torch.rand(D) * 0.5, torch.rand(D) * 0.5
    sp_engine.set_mode(mod, mode)

    def make(floors, th_, ps_):
        p = mod.LearnedMaskPatcher(bank, th_, 5, "resid", torch.zeros(d), torch.zeros(()), amps=ps_,
                                   code_dtype="stream", floors=floors, binarize="anneal", bin_threshold=0.5)
        p.temperature = 0.3
        return p

    xs = [torch.randn(B, T, d) for _ in range(3)]
    th1, ps1 = th.clone().requires_grad_(), ps.clone().requires_grad_()
    TH1, PS1 = {site: th1}, {site: ps1}
    sep = [make(None, TH1, PS1), make({site: fC}, TH1, PS1), make({site: fA}, TH1, PS1)]
    out_sep = torch.cat([p.transform(2, "mlp", x) for p, x in zip(sep, xs)], 0)
    w = torch.randn_like(out_sep)
    (out_sep * w).sum().backward()
    th2, ps2 = th.clone().requires_grad_(), ps.clone().requires_grad_()
    TH2, PS2 = {site: th2}, {site: ps2}
    comb_p = [make(None, TH2, PS2), make({site: fC}, TH2, PS2), make({site: fA}, TH2, PS2)]
    cp = mod.CombinedPatcher(comb_p, B)
    out_c = cp.transform(2, "mlp", torch.cat(xs, 0))
    (out_c * w).sum().backward()
    r = dict(mode=mode, fwd=float((out_c - out_sep).abs().max()), g_theta=float((th1.grad - th2.grad).abs().max()),
             g_amp=float((ps1.grad - ps2.grad).abs().max()), scale=float(out_sep.abs().max()))
    print("combined", r)
    assert r["fwd"] < 1e-5 and r["g_theta"] < 1e-5 and r["g_amp"] < 1e-5, r
    # seed tap
    x = torch.randn(3 * B, T, d)
    y = cp.transform(5, "resid", x)
    assert y is x and cp.seed_pre.shape == (3 * B, T)
    sp_engine.set_mode(mod, "dense")


def test_donor_mask(mod):
    """sp_keep_mask_donor * code at binary gates == 073 ResamplePatcher's patched code."""
    torch.manual_seed(2)
    bank = FakeBank(D=64, k=6)
    B, T, d, D, K = 4, 7, 16, 64, 6
    x = torch.randn(B, T, d)
    ta, ti = bank.encode(x, "mlp", 0)
    dense = torch.zeros(B, T, D).scatter_(-1, ti, ta)
    dta, dti = bank.encode(torch.randn(B, T, d), "mlp", 0)              # donor's top-k code
    donor = torch.zeros(B, T, D).scatter_(-1, dti, dta)
    kt = torch.tensor([1, 5, 9, 17, 33, 40, 41, 50])
    member = torch.zeros(D, dtype=torch.bool)
    member[kt] = True
    alpha = torch.ones(D)
    alpha[kt] = torch.rand(len(kt)) + 0.5
    # 073 ResamplePatcher fill (strict_eval.py lines 97-127, verbatim logic)
    patched = torch.zeros_like(dense)
    patched[:, :, kt] = dense[:, :, kt] * alpha[kt]
    valid = (dta > 0) & ~torch.isin(dti, kt)
    budget = (K - (patched != 0).sum(-1)).clamp(min=0)
    order = dta.masked_fill(~valid, float("-inf")).argsort(-1, descending=True)
    dta_s, dti_s, val_s = dta.gather(-1, order), dti.gather(-1, order), valid.gather(-1, order)
    rank = val_s.long().cumsum(-1) - 1
    use = val_s & (rank < budget.unsqueeze(-1))
    patched.scatter_add_(-1, dti_s, dta_s * use.to(dta_s.dtype))
    # training analogue at binary gates
    m = member.float()
    code = m * alpha * dense + (1.0 - m) * donor
    H = mod.sp_keep_mask_donor(dense, member, alpha, donor, K)
    got = code * H.float()
    diff = float((got - patched).abs().max())
    nfill = int((use).sum())
    print("donor mask: max |diff| %.3g, filled entries %d, members active %d" % (diff, nfill,
                                                                                   int((patched[:, :, kt] != 0).sum())))
    assert diff == 0.0, diff
    assert nfill > 0


def main():
    s1 = sx_engine.source_v2()
    mod = sx_engine.install(s1, "092v2")
    for k in ("truncate", "combine", "grab", "drop_floor"):
        assert k in mod.SPEED
    test_combined(mod, "dense")
    test_combined(mod, "ste")
    test_combined(mod, "hard")
    s2 = sx_engine.source_v2r()
    mod2 = sx_engine.install(s2, "092v2r")
    assert hasattr(mod2, "RESAMPLE") and hasattr(mod2, "_sp_delta_donor")
    test_donor_mask(mod2)
    print("ALL OK")


if __name__ == "__main__":
    main()
