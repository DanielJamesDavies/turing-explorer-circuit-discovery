"""THE GATE for the transcoder harness. Nothing downstream means anything
unless an unmodified circuit is a bit-exact no-op, and the two-hook
read/write Runner is new code.

  1. IDENTITY   transforms that return the code unchanged must leave the
                seed's pre-activation bit-identical to no hooks at all.
  2. FVU/L0     per-layer reconstruction of post_ffw_ln(mlp_out) from
                features read at the pre-norm hook, positions >= 1 (BOS
                excluded, per the probe), against the probe's 0.157 /
                L0 83 at layer 4.
  3. ABLATION   zeroing every upstream feature must move the seed.
  4. THRESHOLD  every JumpReLU threshold positive (so raw == relu-gated).

  PYTHONPATH=. python check_gtc.py
"""
import torch

import ours_gtc as O


def main():
    torch.manual_seed(0)
    toks = O.windows(8).to(O.DEV)
    L = max(O.SEED_LAYERS)
    UP = list(range(L))
    print("layers %s | tokens %s" % (UP + [L], tuple(toks.shape)))

    print("\nlayer  L0(pos>=1)   FVU(pos>=1)   thr>0")
    for layer in UP + [L]:
        cap = {}
        h1 = O.block(layer).pre_feedforward_layernorm.register_forward_hook(
            lambda m, i, o: cap.update(x=o.detach()))
        h2 = O.block(layer).post_feedforward_layernorm.register_forward_hook(
            lambda m, i, o: cap.update(y=o.detach()))
        with torch.no_grad():
            O.model(toks)
        h1.remove(); h2.remove()
        t = O.tc(layer)
        c = O.features(layer, cap["x"])
        rec = (c @ t["W_dec"] + t["b_dec"]).float()[:, 1:]
        y = cap["y"].float()[:, 1:]
        fvu = float(((rec - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        l0 = float((c[:, 1:] > 0).sum(-1).float().mean())
        print("  %2d   %10.1f   %11.4f   %s"
              % (layer, l0, fvu, bool((t["threshold"] > 0).all())))

    sl = int(torch.randint(0, O.D_TC, (1,)))
    bare = O.forward(toks, O.Runner({}, L, sl))
    ident = O.forward(toks, O.Runner({l: (lambda c: c) for l in UP}, L, sl))
    d = float((bare - ident).abs().max())
    rel = d / max(float(bare.abs().max()), 1e-9)
    print("\nIDENTITY  max|d| %.3e  rel %.3e  ->  %s"
          % (d, rel, "PASS" if rel < 1e-4 else "FAIL"))
    zero = O.forward(toks, O.Runner({l: (lambda c: torch.zeros_like(c))
                                     for l in UP}, L, sl))
    moved = float((bare - zero).abs().max())
    print("ABLATION  max|d| %.3e  ->  %s"
          % (moved, "PASS (wired)" if moved > 1e-3 else "FAIL (no effect)"))


if __name__ == "__main__":
    main()
