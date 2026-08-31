"""Does the a_pos correction predict circuit size well enough to skip the probe?

Within a component, n at fixed lambda tracks the seed's natural activation
almost perfectly, and INVERSELY: Spearman(a_pos, n) = -0.90 at comp 8 and
-1.00 at comp 32 (5 seeds each). Hypothesised form:

    n  ~  C_component * a_pos^beta        (beta < 0)

If beta is SHARED across components, only the per-component intercept needs
calibrating and a new seed's lambda costs ZERO extra runs - the 1.0x option.
If beta differs per component, each component needs its own fit, which is
still cheap but not free.

Evaluated by LEAVE-ONE-OUT within each component, against two baselines:

  component median  - what per-component lambda calibration alone achieves
                      (measured spread 2.00x at comp 8, 3.78x at comp 32)
  per-seed probe    - the measured 3.6% max size error, the accuracy ceiling

WATCH FOR SIMPSON'S REVERSAL. Pooled across components the correlation FLIPS
sign (-0.90/-1.00 within, +0.53 pooled) because deep components have both
larger a_pos and larger circuits. Any pooled fit that omits component identity
learns exactly the wrong relationship, so every fit here is within-component.

  PYTHONPATH=src python .../apos_fit.py
"""
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
rows = [json.loads(l) for l in (HERE / "within_component.jsonl").open()
        if l.strip()]
by_comp = defaultdict(list)
for r in rows:
    by_comp[r["comp_idx"]].append(r)


def fit(pts):
    """Least squares on log n = a + beta * log a_pos. Returns (a, beta)."""
    xs = [math.log(max(p["a_pos"], 1e-9)) for p in pts]
    ys = [math.log(max(p["n"], 1)) for p in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    beta = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx
    return my - beta * mx, beta


def r2(pts, a, beta):
    ys = [math.log(max(p["n"], 1)) for p in pts]
    my = sum(ys) / len(ys)
    ss_t = sum((y - my) ** 2 for y in ys)
    ss_r = sum((math.log(max(p["n"], 1))
                - (a + beta * math.log(max(p["a_pos"], 1e-9)))) ** 2
               for p in pts)
    return 1 - ss_r / ss_t if ss_t > 0 else float("nan")


print("PER-COMPONENT FITS   log n = a + beta * log(a_pos)")
print("%-7s %-7s %8s %9s %8s %10s" % ("comp", "seeds", "beta", "R2", "spread",
                                      "median n"))
betas = {}
for comp in sorted(by_comp):
    pts = by_comp[comp]
    if len(pts) < 3:
        print("%-7d %-7d  (too few to fit)" % (comp, len(pts)))
        continue
    a, b = fit(pts)
    betas[comp] = b
    ns = sorted(p["n"] for p in pts)
    print("%-7d %-7d %8.3f %9.3f %7.2fx %10s"
          % (comp, len(pts), b, r2(pts, a, b), ns[-1] / max(ns[0], 1),
             format(ns[len(ns) // 2], ",")))
if len(betas) > 1:
    bs = list(betas.values())
    print("\nbeta across components: %s  -> spread %.2f"
          % (["%.3f" % b for b in bs], max(bs) - min(bs)))
    print("(a tight spread means beta can be fitted ONCE and shared; only the")
    print(" per-component intercept would then need calibrating)")

print("\nLEAVE-ONE-OUT PREDICTION (within component)")
print("%-7s %-9s %10s %10s %9s | %9s" % ("comp", "latent", "actual n",
                                         "predicted", "err", "median-only"))
errs, base_errs, shared_errs = [], [], []
shared_beta = sum(betas.values()) / len(betas) if betas else 0.0
for comp in sorted(by_comp):
    pts = by_comp[comp]
    if len(pts) < 4:
        continue
    for i, held in enumerate(pts):
        train = pts[:i] + pts[i + 1:]
        a, b = fit(train)
        pred = math.exp(a + b * math.log(max(held["a_pos"], 1e-9)))
        e = abs(pred - held["n"]) / held["n"]
        # baseline 1: component median (what per-component lambda gives you)
        med = sorted(p["n"] for p in train)[len(train) // 2]
        be = abs(med - held["n"]) / held["n"]
        # baseline 2: shared beta, intercept from the held-out component's train
        a_s = (sum(math.log(max(p["n"], 1)) for p in train) / len(train)
               - shared_beta * sum(math.log(max(p["a_pos"], 1e-9))
                                   for p in train) / len(train))
        pred_s = math.exp(a_s + shared_beta * math.log(max(held["a_pos"], 1e-9)))
        se = abs(pred_s - held["n"]) / held["n"]
        errs.append(e); base_errs.append(be); shared_errs.append(se)
        print("%-7d %-9d %10s %10s %8.1f%% | %8.1f%%"
              % (comp, held["latent"], format(held["n"], ","),
                 format(int(pred), ","), 100 * e, 100 * be))


def summary(name, xs):
    if not xs:
        return
    xs = sorted(xs)
    print("  %-26s median %6.1f%%  mean %6.1f%%  max %6.1f%%"
          % (name, 100 * xs[len(xs) // 2], 100 * sum(xs) / len(xs), 100 * xs[-1]))


print("\nSIZE-PREDICTION ERROR")
summary("a_pos fit (per-comp beta)", errs)
summary("a_pos fit (shared beta)", shared_errs)
summary("component median only", base_errs)
print("  %-26s     3.6%% max (measured, costs ONE probe run per seed)"
      % "per-seed probe")
print("\nThe a_pos fit is only worth it if it beats 'component median only' by")
print("enough to matter; the probe remains the accuracy ceiling.")
