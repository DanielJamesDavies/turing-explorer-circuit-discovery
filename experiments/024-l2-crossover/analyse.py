"""Crossover analysis for the 32 L2-resid abl-mask circuits.

The observed shared-node count is meaningless on its own: circuits are
~10^4 nodes drawn from a live pool of ~2x10^4, so pigeonhole forces
enormous overlap. Everything here is reported against two nulls, both
using the SAME per-seed set sizes as the real circuits:

  N1  uniform over the live pool  — controls for set size alone.
  N2  density-weighted over the live pool (P(latent) proportional to its
      corpus activation frequency) — controls for "universal nodes are
      just the latents that fire on everything".

N2 is the one that matters. If the observed universal core is no bigger
than N2 predicts, the crossover is an activation-density artifact and
there is no seed-independent structure. If it exceeds N2, there is.

  python experiments/024-l2-crossover/analyse.py
"""
import gzip
import json
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).parent
D_SAE = 40960
N_NULL = 200

rows = [json.loads(l) for l in (HERE / "rows.jsonl").open() if l.strip()]
members = {}
with gzip.open(HERE / "members.jsonl.gz", "rt") as fh:
    for l in fh:
        if l.strip():
            r = json.loads(l)
            members[r["latent"]] = {(a, b, c) for a, b, c in r["members"]}
rows = [r for r in rows if r["latent"] in members]
S = len(rows)
print("=== %d L2-resid abl-mask circuits ===" % S)
sizes = [len(members[r["latent"]]) for r in rows]
print("nodes per circuit: min %d  median %d  max %d  mean %.0f"
      % (min(sizes), sorted(sizes)[len(sizes) // 2], max(sizes),
         sum(sizes) / len(sizes)))
print("free0: %s" % ", ".join("%.3f" % r["free0"] for r in rows if
                              r["free0"] is not None))

# ---- live pool + density from the shared corpus draw
d = torch.load(HERE / "corpus_density.pt", weights_only=False)
counts, positions = d["counts"], d["positions"]
sites = sorted(counts)
pool, weight = [], []
for s in sites:
    v = counts[s]
    nz = (v > 0).nonzero(as_tuple=True)[0]
    for i in nz.tolist():
        pool.append((s[0], s[1], i))
        weight.append(float(v[i]))
P = len(pool)
w = torch.tensor(weight, dtype=torch.double)
print("live pool: %d latents over %d sites (%.2f%% of %d scope), "
      "%d corpus positions" % (P, len(sites), 100.0 * P / (len(sites) * D_SAE),
                               len(sites) * D_SAE, positions))

# ---- observed crossover
freq = Counter()
for r in rows:
    for m in members[r["latent"]]:
        freq[m] += 1
hist = Counter(freq.values())
print("\n-- observed: latents appearing in k of %d circuits --" % S)
print("%5s %10s %12s" % ("k", "latents", "cumulative>=k"))
cum = 0
for k in range(S, 0, -1):
    cum += hist.get(k, 0)
    if hist.get(k, 0) or k in (S, S - 1, S // 2, 1):
        print("%5d %10d %12d" % (k, hist.get(k, 0), cum))
universal = {m for m, c in freq.items() if c == S}
print("\nUNIVERSAL (in all %d): %d latents" % (S, len(universal)))

# how much of each circuit is universal core
core_frac = [100.0 * len(universal & members[r["latent"]]) / len(members[r["latent"]])
             for r in rows]
if universal:
    print("  = %.1f%%-%.1f%% of each circuit (mean %.1f%%)"
          % (min(core_frac), max(core_frac), sum(core_frac) / len(core_frac)))
    by_site = Counter((a, b) for a, b, _ in universal)
    print("  by site: %s" % ", ".join(
        "L%d/%s %d" % (a, b, n) for (a, b), n in sorted(by_site.items())))
    dens_all = w / positions
    idx = {m: j for j, m in enumerate(pool)}
    ud = torch.tensor([float(dens_all[idx[m]]) for m in universal if m in idx])
    print("  corpus density of universal nodes: median %.4f  (all live: %.4f)"
          % (float(ud.median()), float(dens_all.median())))
    q = float((dens_all < ud.median()).double().mean())
    print("  -> universal nodes sit at the %.1fth percentile of live density" % (100 * q))

# ---- nulls
g = torch.Generator().manual_seed(7)


def null_run(weighted):
    f = torch.zeros(P, dtype=torch.int16)
    for n in sizes:
        n = min(n, P)
        if weighted:
            pick = torch.multinomial(w, n, replacement=False, generator=g)
        else:
            pick = torch.randperm(P, generator=g)[:n]
        f[pick] += 1
    return f


print("\n-- nulls (%d reps, matched sizes) --" % N_NULL)
for label, wt in (("N1 uniform", False), ("N2 density-weighted", True)):
    u, ge_half = [], []
    reps = N_NULL if not wt else max(20, N_NULL // 10)   # multinomial is slower
    for _ in range(reps):
        f = null_run(wt)
        u.append(int((f == S).sum()))
        ge_half.append(int((f >= (S + 1) // 2).sum()))
    u = torch.tensor(u, dtype=torch.double)
    h = torch.tensor(ge_half, dtype=torch.double)
    print("  %-20s universal %8.1f +- %-8.1f   >=half %9.1f +- %.1f  (%d reps)"
          % (label, float(u.mean()), float(u.std()), float(h.mean()),
             float(h.std()), reps))
obs_half = sum(c for k, c in hist.items() if k >= (S + 1) // 2)
print("  %-20s universal %8d            >=half %9d" % ("OBSERVED",
                                                        len(universal), obs_half))

# ---- pairwise Jaccard, observed vs uniform null
def jac(a, b):
    return len(a & b) / max(len(a | b), 1)


ks = [r["latent"] for r in rows]
js = [jac(members[ks[i]], members[ks[j]])
      for i in range(S) for j in range(i + 1, S)]
js_t = torch.tensor(js)
nf = torch.randperm(P, generator=g)
nulls = [set(torch.randperm(P, generator=g)[:n].tolist()) for n in sizes]
njs = torch.tensor([jac(nulls[i], nulls[j])
                    for i in range(S) for j in range(i + 1, S)])
print("\npairwise Jaccard: observed %.3f +- %.3f (min %.3f max %.3f) | "
      "uniform null %.3f" % (float(js_t.mean()), float(js_t.std()),
                             float(js_t.min()), float(js_t.max()),
                             float(njs.mean())))

out = {"n_circuits": S, "sizes": sizes, "pool": P, "positions": positions,
       "universal": len(universal), "hist": {str(k): v for k, v in hist.items()},
       "jaccard_mean": float(js_t.mean()), "jaccard_null": float(njs.mean())}
(HERE / "crossover_summary.json").write_text(json.dumps(out, indent=2))
if universal:
    with (HERE / "universal_nodes.json").open("w") as fh:
        json.dump(sorted([list(m) for m in universal]), fh)
print("\nwrote crossover_summary.json" + (" + universal_nodes.json" if universal else ""))
