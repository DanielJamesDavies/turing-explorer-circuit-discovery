"""Exp 132 stage 2b: faithfulness-vs-size curves of the attribution baselines on IOI, scored with the 081 scorer.

Circuits are selected from out/attr/scores_n<N>.pt (TRAINING items only, attrib.py) and scored on held-out items with
eval.wcm_eval.score_task exactly as 081 (scope = all 48 sites, 081's A / C means, alpha = 1: none of these methods has
coefficients). Methods (METHODS, comma list, in this order by default):
  sfc100    SFC node set {(site, latent): |mean IE_ig| > T_N}, IE from 100 training items (SFC's --num_examples),
            T_N over SFC's own sweep t.logspace(-4, 0, 15) (faithfulness.ipynb cell 5) plus 1e-6 .. 5e-5 (ours: the
            native grid stops at sizes far below the 081 circuits). Error nodes: counted (native count) but the 081
            scorer keeps every error live, so they cannot be ablated here (labelled).
  sfc1600   the same with all 1,600 training items (ours: more data than the published default)
  atp       top-k by |mean IE_atp| (1,600 items), k on a 1-2-5 grid 10 .. max
  adiff     top-k by mean |f_cf - f_clean| (1,600 items), same grid
  atpabs    top-k by mean |IE_atp per item| (ours; robust to sign cancellation across items), same grid
  env: HELDOUT_N (score on the first N held-out items: 081's seeded order; unset = all 5,000)
       CURVES (output jsonl name; default curves_h<HELDOUT_N or all>.jsonl)  BUDGET_MIN (stop starting points after
       this many minutes; default 40)  ONLY_KNOBS (method:knob,... rescoring a chosen subset)  MAX_N (largest size, 6e5)
"""
import json
import math
import os
import time

import torch

import common132 as K

METHODS = [m for m in os.environ.get("METHODS", "c081,sfc100,sfc100@am,atp,atp@am,adiff,sfc1600,atpabs").split(",") if m]
BUDGET = float(os.environ.get("BUDGET_MIN", 40)) * 60
MAX_N = int(float(os.environ.get("MAX_N", 6e5)))
HN = os.environ.get("HELDOUT_N")
CURVES = K.OUT / os.environ.get("CURVES", "curves_h%s.jsonl" % (HN or "all"))
ONLY = {tuple(x.split(":")) for x in os.environ.get("ONLY_KNOBS", "").split(",") if x}
SFC_T = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5] + [float(x) for x in torch.logspace(-4, 0, 15).tolist()]
GRID = [int(m * 10 ** e) for e in range(1, 7) for m in (1, 2, 5)]


def C_alphas(d):
    return K.C.alphas_from_json(d)


def load(n):
    n = {100: int(os.environ.get("N_SMALL", 100)), 1600: int(os.environ.get("N_LARGE", 1600))}[n]   # smoke: 25 / 50
    return torch.load(K.OUT / "attr" / ("scores_n%d.pt" % n), weights_only=False)


AM = ("attn", "mlp")


def split(method):
    """"sfc100@am" -> ("sfc100", ("attn", "mlp")): candidates restricted to attn + mlp sites and the ablation scope =
    those 24 sites (resid / resid_mid LIVE; ours, labelled). No suffix: every site, scope 48 (081's scope)."""
    method = method.replace("+inv", "")
    if method.endswith("@am"):
        return method[:-3], AM
    return method, None


def inv_index(d, stat, sites, kinds, D):
    """DIAGNOSTIC (+inv, ours): flat indices of the TRAINING-active latents (081's RND pool, training prompts only)
    whose attribution is exactly 0 on every training item, i.e. latents whose code is identical in the clean and the
    counterfactual prompt wherever they fire (the shared template / first-token features a contrastive attribution can
    never select). Added to every circuit of the method: tests whether the baseline fails for want of them."""
    import gzip
    act = json.loads(gzip.open(K.D081 / "out_v4" / "gpt2-small" / "active" / "ioi.json.gz", "rt").read())
    out = []
    for j, s in enumerate(sites):
        if kinds is not None and s[1] not in kinds:
            continue
        v = act.get("%d/%s" % s) or []
        if not v:
            continue
        v = torch.tensor(v, dtype=torch.long)
        z = v[d[stat][s][v] == 0]
        out.append(z + j * D)
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def flat(d, stat, kinds=None):
    sites = [tuple(s) for s in d["sites"]]
    v = torch.cat([d[stat][s].abs().float() * (1.0 if kinds is None or s[1] in kinds else 0.0) for s in sites])
    return sites, v


def to_alphas(sites, idx, D):
    out = {}
    for j in idx.tolist():
        s = sites[j // D]
        out.setdefault(s, {})[int(j % D)] = 1.0
    return out


def plan(method):
    """[(knob, alphas, native)] in ascending size."""
    D = 32768
    if method == "c081":
        # 081's own v4 circuits, rescored here on the same held-out subset (fitted alphas applied, as 081 scores them)
        out = []
        for arm, lt in (("W", "0.0001"), ("W", "3e-05"), ("U", "1e-06"), ("U", "1e-05")):
            p = K.D081 / "out_v4" / "gpt2-small" / "circuits" / "ioi" / arm / ("%s.json" % lt)
            al = C_alphas(json.loads(p.read_text())["alphas"])
            out.append(("%s_%s" % (arm, lt), al, dict(native=sum(map(len, al.values())))))
        return None, D, out
    if method.endswith("+inv"):
        sites, D, pts = plan(method[:-4])
        base, kinds = split(method)
        d = load(100 if base == "sfc100" else 1600)
        stat = {"sfc100": "ig", "sfc1600": "ig", "atp": "atp", "adiff": "adiff", "atpabs": "atp_abs"}[base]
        inv = inv_index(d, stat, sites, kinds, D)
        out = [(k, torch.unique(torch.cat([idx, inv])), dict(e, n_method=int(idx.numel()), n_inv=int(inv.numel())))
               for k, idx, e in pts]
        out.append(("inv_only", inv, dict(native=int(inv.numel()), n_method=0, n_inv=int(inv.numel()))))
        return sites, D, out
    method, kinds = split(method)
    if method in ("sfc100", "sfc1600"):
        d = load(100 if method == "sfc100" else 1600)
        sites, v = flat(d, "ig", kinds)
        err = d["ig_err"]
        pts, seen = [], set()
        for T in sorted(SFC_T, reverse=True):
            idx = (v > T).nonzero().flatten()
            n = int(idx.numel())
            if n == 0 or n in seen or n > MAX_N:
                continue
            seen.add(n)
            n_err = sum(1 for s in sites if abs(err[tuple(s)]) > T and (kinds is None or s[1] in kinds))
            pts.append(("T%.3g" % T, idx, dict(native=n + n_err, n_err_nodes=n_err, threshold=T)))
        return sites, D, pts
    stat = {"atp": "atp", "adiff": "adiff", "atpabs": "atp_abs"}[method]
    d = load(1600)
    sites, v = flat(d, stat, kinds)
    nz = int((v > 0).sum())
    order = torch.argsort(v, descending=True)
    ks = [k for k in GRID if k <= min(nz, MAX_N)]
    if nz <= MAX_N and nz not in ks:
        ks.append(nz)
    return sites, D, [("k%d" % k, order[:k], dict(native=k, n_nonzero=nz)) for k in ks]


def main():
    t_start = time.time()
    done = set()
    if CURVES.exists():
        for l in open(CURVES):
            r = json.loads(l)
            done.add((r["method"], r["knob"]))
    eng, B, _ = K.setup()
    K.log("held-out items scored: %d -> %s" % (B["n_items_held"], CURVES.name))
    for method in METHODS:
        sites, D, pts = plan(method)
        K.log("%s: %d points, sizes %s" % (method, len(pts), [sum(map(len, p[1].values())) if isinstance(p[1], dict)
                                                              else int(p[1].numel()) for p in pts]))
        for knob, idx, extra in pts:
            if (method, knob) in done or (ONLY and (method, knob) not in ONLY):
                continue
            if time.time() - t_start > BUDGET:
                K.log("budget reached: stopping (resumable)")
                return
            al = idx if isinstance(idx, dict) else to_alphas(sites, idx, D)
            t = time.time()
            kinds = None if method == "c081" else split(method)[1]   # (+inv keeps its base method's scope)
            r = K.score(eng, B, al, None if kinds is None else K.scope_of(eng, kinds))
            rec = dict(method=method, knob=knob, n_heldout=B["n_items_held"], **extra, **K.slim(r))
            K.append(CURVES, rec)
            K.log("%-8s %-10s n=%-7d Z %s A %s C %s nec %s acc %s pass %s (%.0fs)" % (
                method, knob, r["n"], K.f2(r["task_free0"]), K.f2(r["task_freeM_topk"]), K.f2(r["task_freeN_topk"]),
                K.f2(r["task_nec"]), K.f2(r["task_acc_free0"]), r["task_pass"], time.time() - t))
    K.log("DONE")


if __name__ == "__main__":
    main()
