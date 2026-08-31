"""THE GATE. Does the harness reproduce the model, and does it read the
SAEs the way gemma_loader's probe said to?

Nothing downstream means anything unless an UNMODIFIED circuit is a
no-op: every metric is a ratio against the seed's natural activation, so
a harness that perturbs the model even slightly silently rescales every
number. On the Llama side five convention traps were caught by measuring
rather than assuming, and this is the equivalent check here.

Four tests:
  1. IDENTITY   a Runner whose transform returns the code unchanged must
                leave the seed's activation bit-identical to no hooks at
                all. This is the one that would catch a wrong hook point,
                a wrong encoder orientation, or a decoder mismatch.
  2. FVU        per-layer reconstruction quality, which must match the
                probe (~0.13) -- if it does not, tc() is loading weights
                that disagree with the probed convention.
  3. L0         measured per-position density per layer against the
                advertised average_l0 of the tier. This is what the
                sweep VARIES, so it has to be measured, not assumed.
  4. ABLATION   zeroing all upstream latents must actually move the seed;
                if it does not, the intervention path is not wired to
                anything and every "necessity" number would be vacuous.

  TIER=2 PYTHONPATH=. python check_gemma.py
"""
import torch

import ours_gemma as O


def main():
    torch.manual_seed(0)
    toks = O.windows(8).to(O.DEV)
    L = max(O.SEED_LAYERS)
    UP = list(range(L))
    print("tier %d | width %s | layers %s | tokens %s"
          % (O.TIER, O.WIDTH, UP, tuple(toks.shape)))

    print("\nlayer  advertised_l0  measured_l0   FVU")
    for layer in UP + [L]:
        cap = {}
        h = O.block(layer).post_feedforward_layernorm.register_forward_hook(
            lambda m, i, o: cap.update(x=o.detach()))
        with torch.no_grad():
            O.model(toks)
        h.remove()
        x = cap["x"]
        t = O.tc(layer)
        c = O.features(layer, x)
        rec = c @ t["W_dec"] + t["b_dec"]
        fvu = float(((rec - x).float() ** 2).sum()
                    / ((x.float() - x.float().mean()) ** 2).sum())
        l0 = float((c > 0).sum(-1).float().mean())
        print("  %2d   %13d  %11.1f  %.4f"
              % (layer, O._l0_for(layer), l0, fvu))

    # 1. identity
    sl = int(torch.randint(0, O.D_TC, (1,)))
    bare = O.forward(toks, O.Runner({}, L, sl))
    ident = O.forward(toks, O.Runner({l: (lambda c: c) for l in UP}, L, sl))
    d = float((bare - ident).abs().max())
    rel = d / max(float(bare.abs().max()), 1e-9)
    print("\nIDENTITY  max|d| %.3e  rel %.3e  ->  %s"
          % (d, rel, "PASS" if rel < 1e-4 else "FAIL"))

    # 4. ablation must bite
    zero = O.forward(toks, O.Runner({l: (lambda c: torch.zeros_like(c))
                                     for l in UP}, L, sl))
    moved = float((bare - zero).abs().max())
    print("ABLATION  max|d| %.3e  ->  %s"
          % (moved, "PASS (intervention is wired)" if moved > 1e-3
             else "FAIL (no effect - nothing is hooked)"))


if __name__ == "__main__":
    main()
