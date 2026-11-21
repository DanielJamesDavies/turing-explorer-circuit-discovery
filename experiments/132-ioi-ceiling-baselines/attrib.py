"""Exp 132 stage 2a: node attributions on the IOI TRAINING items (GPT-2 small + OAI v5 32k SAEs, the 081 substrate).

Per (site, latent) and per site's error node, three scores from ONE pass over the training items (clean IOI prompt vs
its own ABC counterfactual, metric m = logit(IO) - logit(S) at the anchor, the 081 IOI task metric):

  ig      SFC's node effect (Marks et al. 2025, code circuit.py / attribution.py `_pe_ig`, re-implemented): per site s,
          10-step integrated gradients with only site s moved along the straight line clean -> counterfactual
          (alpha = k/10, k = 0..9, as the code; every other site runs the unmodified model), the gradient averaged
          over the steps, IE = g_f * (f_cf - f_clean) per latent and sum_d g_eps * (eps_cf - eps_clean) for the error.
          Moving x_s = decode(f) + eps jointly IS SFC's joint (f, eps) interpolation: decode is affine.
  atp     attribution patching (Syed et al. 2023; SFC's method='attrib'): the k = 0 gradient only, IE = g * delta.
  adiff   activation difference |f_cf - f_clean| (no gradient).
Aggregation (SFC's rule for non-templatic data, `--aggregation sum`; IOI prompts mix templates and lengths 14-20 so
positions do not line up): each item's effects are SUMMED over its prompt positions, then averaged over items. The
saved node score is that mean (signed); SFC thresholds |mean|. Positions are therefore collapsed natively, so SFC's
native count = (site, latent) pairs + error nodes. Also saved: the mean over items of |sum_pos IE| (atp_abs, ig_abs).

Site space: x_s is the SAE input the 081 engine sees (GPT-2: per-token layer-normalised); an edit is applied by the
site adapter as x + (x_s' - x_s) * scale (model.hf_adapters), as in the scorer.

Checkpoints of the running sums at N_CKPT items (default 100 = SFC's published --num_examples, and every training item).
  env: ATTR_BS (25)  N_CKPT ("100,1600")  IG_STEPS (10)
Writes out/attr/scores_n<N>.pt: {"sites": [...], "<stat>": {site: [d_sae] float32}, "<stat>_err": {site: float},
"n_items": N}; out/attr/meta.json.
"""
import json
import os
import time

import torch

import common132 as K

BS = int(os.environ.get("ATTR_BS", 25))
CKPT = sorted(int(x) for x in os.environ.get("N_CKPT", "100,1600").split(",") if x)
STEPS = int(os.environ.get("IG_STEPS", 10))
OUTD = K.OUT / "attr"
OUTD.mkdir(parents=True, exist_ok=True)
STATS = ("ig", "atp", "adiff", "ig_abs", "atp_abs")


def main():
    eng, B, T = K.setup()
    from config import config
    config.discovery.autocast_bf16 = False          # float32 gradients (the default; set explicitly)
    bank, inf, dev = eng.bank, eng.inf, eng.dev
    D = int(bank.d_sae)
    sites = list(eng.sites)
    k2i = {k: i for i, k in enumerate(eng.kinds)}
    n_tr = min(B["n_items_train"], int(os.environ.get("ATTR_N", 10 ** 9)))   # ATTR_N: smoke only
    toks, anc, val = B["tokens"][:n_tr], B["anchors"][:n_tr], B["valid"][:n_tr]
    ct, ca = B["cf_all"]
    ct, ca, cv = ct[:n_tr], ca[:n_tr], B["cf_valid_all"][:n_tr]
    good, bad = B["good"][:n_tr, 0], B["bad"][:n_tr, 0]
    same = (val == cv).all(1) & (anc == ca)
    K.log("training items %d; clean / cf token-aligned (same valid mask and anchor): %d" % (n_tr, int(same.sum())))
    if not bool(same.all()):
        raise SystemExit("clean and counterfactual prompts are not position-aligned on %d items" % int((~same).sum()))
    sums = {st: {s: torch.zeros(D, dtype=torch.float64, device=dev) for s in sites} for st in STATS}
    esums = {st: {s: 0.0 for s in sites} for st in STATS if st != "adiff"}
    tot_effect = 0.0
    n_done = 0
    t0 = time.time()

    def capture(tk):
        cap = {}

        def hook(layer_idx, activations):
            for kd in eng.kinds:
                if (layer_idx, kd) in cap or (layer_idx, kd) not in set(sites):
                    continue
                cap[(layer_idx, kd)] = activations[k2i[kd]].detach().float().clone()

        with torch.no_grad():
            out = inf.forward(tk, activations_callback=hook, return_activations=False, tokenize_final=False,
                              all_logits=True)
        lg = out[1] if isinstance(out, (tuple, list)) else out
        return cap, lg

    def metric(lg, an, g, b):
        rr = torch.arange(lg.shape[0], device=lg.device)
        z = lg[rr, an.to(lg.device)].float()
        return z[rr, g.to(lg.device)] - z[rr, b.to(lg.device)]

    class Add:
        """y = x_s + delta_s (+ alpha * dx_s at the moved site): additive leaves keep the graph through every site."""

        def __init__(self, leaves, moved=None, shift=None):
            self.leaves, self.moved, self.shift = leaves, moved, shift

        def __call__(self, model):
            from model.hooks import multi_patch
            return multi_patch(model, self.transform)

        def transform(self, layer_idx, kd, x):
            s = (layer_idx, kd)
            y = x
            if self.moved == s:
                y = y + self.shift.to(x.dtype)
            if s in self.leaves:
                y = y + self.leaves[s].to(x.dtype)
            return y if y is not x else None

    def fwd_grad(tk, an, g, b, leaves, moved=None, shift=None):
        p = Add(leaves, moved, shift)
        out = inf.forward(tk, patcher=p, grad_enabled=True, all_logits=True, return_activations=False,
                          tokenize_final=False)
        lg = out[1] if isinstance(out, (tuple, list)) else out
        m = metric(lg, an, g, b)
        ks = list(leaves)
        gr = torch.autograd.grad(m.sum(), [leaves[s] for s in ks], allow_unused=True)
        return {s: (x.detach().float() if x is not None else torch.zeros_like(leaves[s])) for s, x in zip(ks, gr)}, \
            m.detach()

    inf.disable_compile()
    try:
        for s0 in range(0, n_tr, BS):
            s1 = min(n_tr, s0 + BS)
            tk, an, vv = toks[s0:s1], anc[s0:s1], val[s0:s1]
            g_, b_ = good[s0:s1], bad[s0:s1]
            xc, lgc = capture(tk)
            xp, lgp = capture(ct[s0:s1])
            mc, mp = metric(lgc, an, g_, b_), metric(lgp, an, g_, b_)
            tot_effect += float((mp - mc).sum())
            vm = vv.to(dev).float()[..., None]                              # [b, T, 1] real prompt tokens
            # k = 0: every site at once (additive zero leaves): the AtP gradient = IG's first step at every site
            leaves = {s: torch.zeros_like(xc[s], requires_grad=True) for s in sites}
            g0, _ = fwd_grad(tk, an, g_, b_, leaves)
            gsum = {s: g0[s].clone() for s in sites}
            for s in sites:
                dx = xp[s] - xc[s]
                for k in range(1, STEPS):
                    lf = {s: torch.zeros_like(xc[s], requires_grad=True)}
                    gk, _ = fwd_grad(tk, an, g_, b_, lf, moved=s, shift=(k / STEPS) * dx)
                    gsum[s] += gk[s]
            for s in sites:
                l, kd = s
                fc = _dense(bank, xc[s], kd, l, D)
                fp = _dense(bank, xp[s], kd, l, D)
                df = (fp - fc) * vm                                         # [b, T, D]
                ec = xc[s] - bank.decode(fc, kd, l)
                ep = xp[s] - bank.decode(fp, kd, l)
                de = (ep - ec) * vm                                         # [b, T, d]
                for st, gx in (("ig", gsum[s] / STEPS), ("atp", g0[s])):
                    gf = _grad_f(bank, gx, kd, l, D, fc.dtype)              # d m / d f = W_dec g  [b, T, D]
                    ie = (gf * df).sum(1)                                    # [b, D] summed over positions
                    sums[st][s] += ie.double().sum(0)
                    sums[st + "_abs"][s] += ie.abs().double().sum(0)
                    ee = (gx * de).sum((1, 2))                               # [b] error node
                    esums[st][s] += float(ee.sum())
                    esums[st + "_abs"][s] += float(ee.abs().sum())
                    del gf, ie
                sums["adiff"][s] += df.abs().sum(1).double().sum(0)
                del fc, fp, df, ec, ep, de
            if s0 == 0:
                # completeness check (first batch): sum of a site's IG effects (latents + error) vs the exact effect of
                # patching that site alone to its counterfactual value; IG with 10 steps should be close
                for s in [x for x in ((0, "resid"), (5, "attn"), (9, "resid_mid"), (10, "mlp")) if x in sites]:
                    lf = {s: torch.zeros_like(xc[s], requires_grad=True)}
                    _, m1 = fwd_grad(tk, an, g_, b_, lf, moved=s, shift=xp[s] - xc[s])
                    l, kd = s
                    fc = _dense(bank, xc[s], kd, l, D); fp = _dense(bank, xp[s], kd, l, D)
                    gx = gsum[s] / STEPS
                    ie = (_grad_f(bank, gx, kd, l, D, fc.dtype) * (fp - fc) * vm).sum()
                    ee = (gx * ((xp[s] - bank.decode(fp, kd, l)) - (xc[s] - bank.decode(fc, kd, l))) * vm).sum()
                    K.log("check %s: exact patch effect %.3f | IG latents %.3f + error %.3f = %.3f"
                          % (s, float((m1 - mc).sum()), float(ie), float(ee), float(ie + ee)))
            n_done = s1
            del xc, xp, g0, gsum, leaves
            if s0 // BS % 8 == 0:
                K.log("items %d / %d  (%.0fs; mean total effect cf - clean %.3f)" % (n_done, n_tr, time.time() - t0,
                                                                                    tot_effect / n_done))
            if n_done in CKPT or n_done == n_tr:
                save(sums, esums, n_done, sites, tot_effect)
    finally:
        inf.enable_compile()
    (OUTD / "meta.json").write_text(json.dumps(dict(n_items=n_done, ig_steps=STEPS, batch=BS, secs=round(time.time() - t0),
                                                    mean_total_effect=tot_effect / max(1, n_done), ckpt=CKPT), indent=1))
    K.log("DONE %d items %.0fs" % (n_done, time.time() - t0))


def _dense(bank, x, kd, l, D):
    from sae.dense import sparse_topk_to_dense
    with torch.no_grad():
        ta, ti = bank.encode(x, kd, l)
        return sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)


def _grad_f(bank, gx, kd, l, D, dtype):
    f = torch.zeros(gx.shape[:-1] + (D,), device=gx.device, dtype=dtype, requires_grad=True)
    with torch.enable_grad():
        y = bank.decode(f, kd, l)
        (g,) = torch.autograd.grad(y, f, grad_outputs=gx.to(y.dtype))
    return g.float()


def save(sums, esums, n, sites, tot):
    d = dict(sites=sites, n_items=n, mean_total_effect=tot / n)
    for st, v in sums.items():
        d[st] = {s: (x / n).float().cpu() for s, x in v.items()}
    for st, v in esums.items():
        d[st + "_err"] = {s: x / n for s, x in v.items()}
    torch.save(d, OUTD / ("scores_n%d.pt" % n))
    K.log("saved scores_n%d.pt" % n)


if __name__ == "__main__":
    main()
