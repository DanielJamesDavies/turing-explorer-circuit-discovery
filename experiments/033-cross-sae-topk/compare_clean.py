"""Does subtracting the CLEAN code (ERROR_MODE=clean) change the TopK results?

Pairs every seed's original `triamp400` row (rows.jsonl) with its re-run
`triamp400_clean` row (rows_clean.jsonl) and reports, per seed and pooled:
members n, ampF0, ampFM, sup, cf_amp, and the band pass (ampF0 and ampFM both
in [0.8, 1.25]) under ONE rule applied identically to both. Seeds with held-out
a_pos < 1.0 are excluded, as in the 033 README. Also: how many fitted nulls
pass the band in each run.

  python experiments/033-cross-sae-topk/compare_clean.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
BAND = (0.8, 1.25)


def load(fn):
    p = HERE / fn
    return [json.loads(l) for l in open(p)] if p.exists() else []


def key(r):
    return (r.get("kind", "resid"), r["layer"], r["latent"])


def passes(r):
    f0, fm = r.get("ampF0"), r.get("ampFM")
    return f0 is not None and fm is not None and BAND[0] <= f0 <= BAND[1] and BAND[0] <= fm <= BAND[1]


orig = {key(r): r for r in load("rows.jsonl") if r.get("arm") == "triamp400"}
clean = {key(r): r for r in load("rows_clean.jsonl") if r.get("arm") == "triamp400_clean"}
both = sorted(k for k in orig if k in clean and orig[k].get("a_pos_ho", 0) >= 1.0)
print("paired seeds (held-out a_pos >= 1.0): %d  (clean rows so far %d)\n" % (len(both), len(clean)))
print("%-18s | %6s %6s | %7s %7s | %7s %7s | %6s %6s | %7s %7s | pass o/c"
      % ("seed", "n", "n_cl", "F0", "F0_cl", "FM", "FM_cl", "sup", "sup_cl", "cf", "cf_cl"))
f = lambda v: "%.3f" % v if isinstance(v, (int, float)) else "-"
for k in both:
    o, c = orig[k], clean[k]
    print("%-18s | %6d %6d | %7s %7s | %7s %7s | %6s %6s | %7s %7s | %s / %s"
          % ("%s L%d #%d" % k, o["n"], c["n"], f(o["ampF0"]), f(c["ampF0"]), f(o["ampFM"]), f(c["ampFM"]),
             f(o["sup"]), f(c["sup"]), f(o["cf_amp"]), f(c["cf_amp"]),
             "Y" if passes(o) else ".", "Y" if passes(c) else "."))
if both:
    med = lambda rs, k: float(np.median([r[k] for r in rs if r.get(k) is not None]))
    O, C = [orig[k] for k in both], [clean[k] for k in both]
    print("\nmedians          original -> clean")
    for k in ("n", "ampF0", "ampFM", "sup", "cf_amp"):
        print("  %-8s %10.3f -> %.3f" % (k, med(O, k), med(C, k)))
    print("  band pass  %d/%d -> %d/%d" % (sum(map(passes, O)), len(O), sum(map(passes, C)), len(C)))
    agree = sum(passes(o) == passes(c) for o, c in zip(O, C))
    print("  per-seed pass verdict agrees on %d/%d" % (agree, len(both)))
    dn = [abs(c["n"] - o["n"]) / max(o["n"], 1) for o, c in zip(O, C)]
    print("  |n_clean - n| / n: median %.3f, max %.3f" % (float(np.median(dn)), max(dn)))
for fn, pref in (("rows.jsonl", "null"), ("rows_clean.jsonl", "null")):
    nulls = [r for r in load(fn) if str(r.get("arm", "")).startswith(pref) and key(r) in set(both)]
    if nulls:
        print("  %-17s nulls: %d draws, %d pass the band, max ampF0 %.3f"
              % (fn, len(nulls), sum(map(passes, nulls)), max(r["ampF0"] for r in nulls if r.get("ampF0") is not None)))
