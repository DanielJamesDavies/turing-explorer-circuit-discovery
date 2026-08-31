"""What, if anything, about a seed latent predicts its circuit size?

30 seeds measured at a FIXED lambda=1e-5 (dual, gamma=0.25) across three
components - 10 each at comp 8 (L2-resid, shallow), 25 (L8-mlp, mid) and
32 (L10-resid, deep). The obvious candidate already failed: a_pos looked
perfect on 5 seeds (Spearman -1.00 at comp 32) and collapsed on 10
(R2 0.601 there, 0.048 at comp 25), with leave-one-out error barely beating
"just use the component median" and a WORSE tail.

This sweeps every free feature instead - rarity, magnitude, dispersion,
context structure and coactivation - all from artifacts already on disk, no
extra GPU runs.

METHOD DISCIPLINE. Correlations are computed WITHIN component and reported
per component, never pooled. Pooled analysis reverses the sign here
(a_pos: -0.90/-1.00 within, +0.53 pooled) because deep components have both
larger activations and larger circuits - a textbook Simpson's reversal, and
any pooled fit would learn the inverse of the truth.

With n=10 per component, |rho| > ~0.65 is the rough p<0.05 bar. A feature is
only interesting if it clears that AND keeps its sign in all three
components; one component alone is a coin flip at this sample size.

  PYTHONPATH=src python .../seed_features.py
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
TOTAL_TOKENS = 6060 * 262144.0

rows = [json.loads(l) for l in (HERE / "within_component.jsonl").open()
        if l.strip()]
LS = torch.load(RUN_ROOT / "latent_stats.pt", map_location="cpu", weights_only=False)
CO = torch.load(RUN_ROOT / "top_coactivation.pt", map_location="cpu", weights_only=False)
TOP = torch.load(RUN_ROOT / "top_ctx.pt", map_location="cpu", weights_only=False)

active = LS["active_count"].float()
mean = LS["mean"]
mean_abs = LS["mean_abs"]
m2 = LS["m2"]
seq_count = LS["seq_count"].float()
mean_seq = LS["mean_seq"]
m2_seq = LS["m2_seq"]
co_val = CO["top_values"]          # [36, 40960, 128] PMI-weighted
top_val = TOP["ctx_seq_val"]       # [36, 40960, 64] top-context activations

for r in rows:
    c, l = r["comp_idx"], r["latent"]
    cnt = float(active[c, l])
    r["rate"] = cnt / TOTAL_TOKENS
    r["log_rate"] = math.log(max(r["rate"], 1e-12))
    r["seq_count"] = float(seq_count[c, l])
    r["mean"] = float(mean[c, l])
    r["mean_abs"] = float(mean_abs[c, l])
    # variance of the raw activation stream (Welford m2 over all slots)
    r["std"] = math.sqrt(max(float(m2[c, l]), 0.0) / max(cnt, 1.0))
    r["std_seq"] = math.sqrt(max(float(m2_seq[c, l]), 0.0)
                             / max(r["seq_count"], 1.0))
    r["mean_seq"] = float(mean_seq[c, l])
    # conditional magnitude: average value WHEN active, not diluted by zeros
    r["cond_mean"] = float(mean[c, l]) / max(r["rate"], 1e-12)
    # how far the probe contexts sit above the latent's typical firing
    r["apos_over_cond"] = r["a_pos"] / max(r["cond_mean"], 1e-9)
    # CENSORING GAP: pre-top-k reference vs the post-top-k value actually read.
    # A seed only marginally inside top-k has a large gap.
    r["censor_gap"] = r["posctx_ref"] - r["a_pos"]
    r["censor_ratio"] = r["a_pos"] / max(r["posctx_ref"], 1e-9)
    # coactivation structure (PMI-weighted, top-128 partners)
    cv = co_val[c, l].float()
    pos = cv[cv > 0]
    r["coact_n_pos"] = float(pos.numel())
    r["coact_sum"] = float(pos.sum()) if pos.numel() else 0.0
    r["coact_max"] = float(pos.max()) if pos.numel() else 0.0
    r["coact_mean"] = float(pos.mean()) if pos.numel() else 0.0
    # concentration: share of total PMI mass held by the top 8 partners
    if pos.numel() >= 8:
        r["coact_top8_share"] = float(pos.topk(8).values.sum() / pos.sum())
    else:
        r["coact_top8_share"] = float("nan")
    # top-context spread: how peaked the latent's best contexts are
    tv = top_val[c, l].float()
    tv = tv[tv > 0]
    r["topctx_n"] = float(tv.numel())
    r["topctx_max"] = float(tv.max()) if tv.numel() else 0.0
    r["topctx_ratio"] = (float(tv.min() / tv.max())
                         if tv.numel() and float(tv.max()) > 0 else float("nan"))

FEATURES = [
    ("rate", "firing rate"),
    ("log_rate", "log firing rate"),
    ("seq_count", "n sequences active"),
    ("a_pos", "a_pos (post-topk)"),
    ("posctx_ref", "posctx pre-act ref"),
    ("censor_gap", "pre-act minus a_pos"),
    ("censor_ratio", "a_pos / pre-act"),
    ("mean", "mean act (all slots)"),
    ("mean_abs", "mean |act|"),
    ("cond_mean", "mean act WHEN active"),
    ("apos_over_cond", "a_pos / cond mean"),
    ("std", "std of activation"),
    ("std_seq", "std across sequences"),
    ("mean_seq", "mean per-sequence"),
    ("coact_n_pos", "n coact partners"),
    ("coact_sum", "total coact PMI"),
    ("coact_max", "max coact PMI"),
    ("coact_mean", "mean coact PMI"),
    ("coact_top8_share", "coact top-8 share"),
    ("topctx_n", "n top contexts"),
    ("topctx_max", "top ctx max act"),
    ("topctx_ratio", "top ctx min/max"),
]


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y)
             if not (isinstance(a, float) and math.isnan(a))
             and not (isinstance(b, float) and math.isnan(b))]
    if len(pairs) < 4:
        return float("nan")
    x = [p[0] for p in pairs]; y = [p[1] for p in pairs]

    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):                      # average ties
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2.0
            i = j + 1
        return r
    a, b = rk(x), rk(y)
    n = len(x)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((v - ma) ** 2 for v in a) * sum((v - mb) ** 2 for v in b))
    return num / den if den > 0 else float("nan")


by_comp = defaultdict(list)
for r in rows:
    by_comp[r["comp_idx"]].append(r)
comps = sorted(by_comp)

print("Spearman(feature, n) WITHIN component. n=%d per component."
      % min(len(v) for v in by_comp.values()))
print("|rho| > ~0.65 is the rough p<0.05 bar at this sample size.\n")
hdr = "%-22s" % "feature" + "".join("  comp %-6d" % c for c in comps) + "  consistent?"
print(hdr); print("-" * len(hdr))
results = []
for key, label in FEATURES:
    rhos = []
    for c in comps:
        pts = by_comp[c]
        rhos.append(spearman([p.get(key, float("nan")) for p in pts],
                             [p["n"] for p in pts]))
    ok = (all(not math.isnan(r) for r in rhos)
          and (all(r > 0 for r in rhos) or all(r < 0 for r in rhos)))
    strong = ok and min(abs(r) for r in rhos) > 0.5
    flag = ("SIGN-CONSISTENT" + (" + all |rho|>0.5" if strong else "")) if ok else ""
    results.append((label, rhos, strong))
    print("%-22s" % label
          + "".join(("  %+9.2f" % r) if not math.isnan(r) else "        n/a"
                    for r in rhos)
          + "  " + flag)

print("\nStrongest sign-consistent features (all |rho| > 0.5):")
hits = [(lab, rh) for lab, rh, s in results if s]
if not hits:
    print("  NONE. No free feature predicts circuit size consistently across")
    print("  all three components - which is itself the finding: circuit size")
    print("  looks idiosyncratic per seed rather than a function of the")
    print("  latent's own statistics.")
else:
    for lab, rh in sorted(hits, key=lambda t: -min(abs(r) for r in t[1])):
        print("  %-22s %s  (min |rho| %.2f)"
              % (lab, ["%+.2f" % r for r in rh], min(abs(r) for r in rh)))

(HERE / "seed_features.json").write_text(json.dumps(
    {"rows": rows,
     "spearman": {lab: rh for lab, rh, _ in results},
     "components": comps}, indent=2, default=float))
print("\nwrote seed_features.json")
