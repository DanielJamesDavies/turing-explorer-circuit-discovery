"""093 report: WCM protocol v2 vs v1, paired per target (run.py rows), -> out/tables.md.

Pass = the DAN-8 rule (Z, A, C in [0.8, 1.5] on the post-Top-K read with the sparsity-preserving A / C fills, and
necessity >= 0.9). Paired statistics are over the targets with an ok row in both arms: exact McNemar on the
discordant pairs, a percentile bootstrap (10,000 resamples of targets, seed 0) on the paired pass-rate difference and
on the median paired changes, and the Wilcoxon signed-rank test on the paired score changes.

  python experiments/093-v2-validation/report.py        (OUT93 overrides experiments/093-v2-validation/out)
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT93 = Path(os.environ.get("OUT93", str(HERE / "out")))
BANDS = ("L0-4", "L5-7", "L8-11")
KINDS = ("attn", "mlp", "resid")
SCORES = (("Z", "free0_tk"), ("A", "freeM_topk_tk"), ("C", "freeN_topk_tk"), ("nec", "phi_sup_blind_tk"),
          ("A dense", "freeM_dense_tk"), ("C dense", "freeN_dense_tk"))
B_BOOT = 10000
RNG = np.random.default_rng(0)


def load():
    rows = [json.loads(l) for l in open(OUT93 / "rows.jsonl") if l.strip()]
    d = pd.DataFrame(rows)
    err = d[d.get("error").notna()] if "error" in d else d.iloc[:0]
    ok = d[d.get("error").isna()] if "error" in d else d
    ok = ok.drop_duplicates(["seed", "arm"], keep="last")
    return ok, err


def mcnemar(b, c):
    """Exact two-sided McNemar: b = v1 pass / v2 fail, c = v1 fail / v2 pass."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot(x, stat=np.mean):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (None, None, None)
    idx = RNG.integers(0, x.size, size=(B_BOOT, x.size))
    s = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def wilcoxon(x):
    from scipy.stats import wilcoxon as w
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x) & (x != 0)]
    if x.size < 5:
        return None
    return float(w(x).pvalue)


def f2(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else ("%.*f" % (nd, x))


def ci(t, nd=2, pct=False):
    m, lo, hi = t
    if m is None:
        return "-"
    k = 100 if pct else 1
    return "%s [%s, %s]" % (f2(m * k, nd), f2(lo * k, nd), f2(hi * k, nd))


def why_fail(r):
    out = []
    for nm, k in SCORES[:3]:
        v = r.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("%s missing" % nm)
        elif not 0.8 <= v <= 1.5:
            out.append("%s %.2f" % (nm, v))
    v = r.get("phi_sup_blind_tk")
    if v is None or (isinstance(v, float) and np.isnan(v)) or v < 0.9:
        out.append("nec %s" % f2(v))
    return ", ".join(out) or "-"


def main():
    ok, err = load()
    tj = json.loads((HERE / "targets.json").read_text())
    keys = [k for k in (HERE / "targets.txt").read_text().split() if k]
    v1 = ok[ok.arm == "v1"].set_index("seed")
    v2 = ok[ok.arm == "v2"].set_index("seed")
    both = [k for k in keys if k in v1.index and k in v2.index]
    v1, v2 = v1.loc[both], v2.loc[both]
    L = []
    w = L.append
    w("# 093: WCM protocol v2 vs v1, paired (72 stratified TuringLLM targets)\n")
    w("v1 = protocol v1 refitted here; v2 = the A / C training terms under the scorer's sparsity-preserving fill "
      "(straight-through) against the scorer's own A / C means; everything else as v1 (gamma_A = gamma_C 0.25, "
      "lambda 1e-3, rank-keep 3e-3, 400 steps). Both arms: this GPU, the 15k run's as-run context records (CTX_DIR, "
      "explicit eval split), the production scorer on held-out strongest (16) and 056 specificity. The as-run "
      "held-out sets carry the DAN-157 duplicate leakage; both arms see the same records, so it affects them equally "
      "and the paired comparison stands, but the absolute pass rates are on leaky held-out sets.\n")
    w("Targets: 8 per band x kind, random within cells (seed %d), non-vacuous; %d (target, arm) rows ok, %d error rows; "
      "**%d paired targets**.\n" % (tj["seed"], len(ok), len(err), len(both)))
    if not both:
        (OUT93 / "tables.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    p1, p2 = v1.pass_rule.astype(bool), v2.pass_rule.astype(bool)
    b, c = int((p1 & ~p2).sum()), int((~p1 & p2).sum())

    # ---------------------------------------------------------------- pass rates
    w("## Pass rate (DAN-8 rule)\n")
    w("| set | targets | v1 pass | v2 pass | v1 -> v2 lost / gained | McNemar p | v2 - v1 (pp) [95% bootstrap CI] |")
    w("|---|---|---|---|---|---|---|")

    def pass_line(name, idx):
        a1, a2 = p1[idx], p2[idx]
        bb, cc = int((a1 & ~a2).sum()), int((~a1 & a2).sum())
        diff = boot((a2.astype(float) - a1.astype(float)).values)
        w("| %s | %d | %d (%.0f%%) | %d (%.0f%%) | %d / %d | %s | %s |" % (
            name, len(a1), a1.sum(), 100 * a1.mean(), a2.sum(), 100 * a2.mean(), bb, cc, f2(mcnemar(bb, cc), 3),
            ci(diff, 1, pct=True)))
    allidx = pd.Series(True, index=both)
    pass_line("**all**", allidx)
    for bd in BANDS:
        pass_line(bd, v1.band == bd)
    for k in KINDS:
        pass_line(k, v1.kind == k)
    for bd in BANDS:
        for k in KINDS:
            pass_line("%s / %s" % (bd, k), (v1.band == bd) & (v1.kind == k))
    pd1, pd2 = v1.pass_dense.astype(bool), v2.pass_dense.astype(bool)
    w("\nDense-fill rule (A / C dense in place of the sparsity-preserving fills; v1's training semantics): v1 %d, v2 %d "
      "of %d.\n" % (pd1.sum(), pd2.sum(), len(both)))

    # ---------------------------------------------------------------- scores
    w("## Scores: medians per arm and paired changes (v2 - v1)\n")
    w("| score | v1 median | v2 median | median paired change [95% CI] | mean paired change [95% CI] | Wilcoxon p |"
      " v1 in band | v2 in band |")
    w("|---|---|---|---|---|---|---|---|")
    for nm, k in SCORES:
        x1, x2 = v1[k].astype(float), v2[k].astype(float)
        dlt = (x2 - x1).values
        inb = (lambda x: "%d" % ((x >= 0.9).sum())) if nm == "nec" else (lambda x: "%d" % (x.between(0.8, 1.5).sum()))
        w("| %s | %s | %s | %s | %s | %s | %s | %s |" % (nm, f2(x1.median()), f2(x2.median()), ci(boot(dlt, np.median)),
                                                       ci(boot(dlt)), f2(wilcoxon(dlt), 3), inb(x1), inb(x2)))
    w("\n(in band: Z / A / C in [0.8, 1.5]; nec >= 0.9.)\n")

    # ---------------------------------------------------------------- Z check
    dz = (v2.free0_tk.astype(float) - v1.free0_tk.astype(float))
    w("## Is Z hurt?\n")
    w("Paired dZ: median %s, mean %s; targets with Z down by > 0.1: %d, up by > 0.1: %d; Z leaves the band under v2 "
      "(in under v1, out under v2): %d, enters it: %d. Wilcoxon p = %s.\n" % (
          ci(boot(dz.values, np.median)), ci(boot(dz.values)), int((dz < -0.1).sum()), int((dz > 0.1).sum()),
          int((v1.free0_tk.between(0.8, 1.5) & ~v2.free0_tk.between(0.8, 1.5)).sum()),
          int((~v1.free0_tk.between(0.8, 1.5) & v2.free0_tk.between(0.8, 1.5)).sum()), f2(wilcoxon(dz.values), 3)))
    w("| band | targets | v1 Z median | v2 Z median | median dZ | Z down > 0.1 |")
    w("|---|---|---|---|---|---|")
    for bd in BANDS:
        m = v1.band == bd
        w("| %s | %d | %s | %s | %s | %d |" % (bd, m.sum(), f2(v1.free0_tk[m].median()), f2(v2.free0_tk[m].median()),
                                              f2(dz[m].median()), int((dz[m] < -0.1).sum())))
    w("")

    # ---------------------------------------------------------------- size, alpha, cost
    ratio = (v2.n_nodes.astype(float) / v1.n_nodes.astype(float).clip(lower=1)).values
    w("## Size, coefficients, cost\n")
    w("| quantity | v1 | v2 | paired ratio v2 / v1 (median [95% CI]) |")
    w("|---|---|---|---|")
    w("| circuit size (members), median | %s | %s | %s |" % (f2(v1.n_nodes.median(), 0), f2(v2.n_nodes.median(), 0),
                                                            ci(boot(ratio, np.median))))
    for k, lab in (("alpha_med", "alpha median (per-circuit median)"), ("alpha_p90", "alpha p90"),
                   ("alpha_max", "alpha max"), ("frac_alpha_gt2", "share of members with alpha > 2")):
        w("| %s | %s | %s | |" % (lab, f2(v1[k].median()), f2(v2[k].median())))
    tf1, tf2 = v1.t_fit.astype(float), v2.t_fit.astype(float)
    tr = (tf2 / tf1).values
    w("| fit time s, median (total) | %s (%s) | %s (%s) | %s |" % (f2(tf1.median(), 1), f2(tf1.sum(), 0),
                                                               f2(tf2.median(), 1), f2(tf2.sum(), 0),
                                                               ci(boot(tr, np.median))))
    pm1, pm2 = v1.fit_peak_mb.astype(float), v2.fit_peak_mb.astype(float)
    w("| fit peak memory MB, median (max) | %s (%s) | %s (%s) | %s |" % (
        f2(pm1.median(), 0), f2(pm1.max(), 0), f2(pm2.median(), 0), f2(pm2.max(), 0),
        ci(boot((pm2 / pm1).values, np.median))))
    w("\nFit time and memory were measured on a shared GPU (another agent's context builder ran alongside part of "
      "the run), with the two arms of a target fitted back to back in alternating order, so the paired ratio is the "
      "number to read.\n")

    # ---------------------------------------------------------------- lost / gained
    w("## v1 passes that v2 loses, and v1 failures that v2 passes\n")
    w("| target | cell | change | v1 Z / A / C / nec | v2 Z / A / C / nec | v2 fails on | n v1 -> v2 |")
    w("|---|---|---|---|---|---|---|")
    for k in both:
        a, bq = bool(p1[k]), bool(p2[k])
        if a == bq:
            continue
        r1, r2 = v1.loc[k].to_dict(), v2.loc[k].to_dict()
        zacn = lambda r: " / ".join(f2(r.get(x)) for _, x in SCORES[:4])   # noqa: E731
        w("| %s | %s | %s | %s | %s | %s | %d -> %d |" % (k, r1["cell"], "LOST" if a else "gained", zacn(r1), zacn(r2),
                                                        why_fail(r2) if a else "-", r1["n_nodes"], r2["n_nodes"]))
    lost = [k for k in both if p1[k] and not p2[k]]
    reasons = {}
    for k in lost:
        for part in why_fail(v2.loc[k].to_dict()).split(", "):
            reasons[part.split(" ")[0]] = reasons.get(part.split(" ")[0], 0) + 1
    w("\nv1 passes lost by v2: %d of %d; failing score among them: %s.\n" % (
        len(lost), int(p1.sum()), ", ".join("%s %d" % kv for kv in sorted(reasons.items())) or "none"))
    fails2 = [k for k in both if not p2[k]]
    reasons2 = {}
    for k in fails2:
        for part in why_fail(v2.loc[k].to_dict()).split(", "):
            reasons2[part.split(" ")[0]] = reasons2.get(part.split(" ")[0], 0) + 1
    reasons1 = {}
    for k in [k for k in both if not p1[k]]:
        for part in why_fail(v1.loc[k].to_dict()).split(", "):
            reasons1[part.split(" ")[0]] = reasons1.get(part.split(" ")[0], 0) + 1
    w("All failures by failing score (a target can fail on several): v1 %s; v2 %s.\n" % (
        ", ".join("%s %d" % kv for kv in sorted(reasons1.items())),
        ", ".join("%s %d" % kv for kv in sorted(reasons2.items()))))

    # ---------------------------------------------------------------- amplifiers
    w("## Amplifier flags (056 specificity: target_faith_pre >= 0.5 and sibling median >= 0.8 x target)\n")
    w("| ablation | v1 flagged | v2 flagged | v1 -> v2 new / cleared |")
    w("|---|---|---|---|")
    for pi in ("Z", "A", "C", "any"):
        col = "amp_any" if pi == "any" else "amp_" + pi
        if col not in v1 or col not in v2:
            continue
        a1, a2 = v1[col].fillna(False).astype(bool), v2[col].fillna(False).astype(bool)
        w("| %s | %d | %d | %d / %d |" % (pi, a1.sum(), a2.sum(), int((~a1 & a2).sum()), int((a1 & ~a2).sum())))
    a1, a2 = v1.amp_any.fillna(False).astype(bool), v2.amp_any.fillna(False).astype(bool)
    w("\nPassing AND flagged: v1 %d, v2 %d.\n" % (int((p1 & a1).sum()), int((p2 & a2).sum())))

    # ---------------------------------------------------------------- 15k production reference
    prod = pd.DataFrame(tj["production"]).T.loc[both]
    pp = prod.apply(lambda r: bool(all(r[k] is not None and 0.8 <= float(r[k]) <= 1.5
                                       for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk"))
                                   and r["phi_sup_blind_tk"] is not None and float(r["phi_sup_blind_tk"]) >= 0.9),
                    axis=1)
    agree = int((pp == p1).sum())
    w("## Sanity: the 15k production circuits (H100, as-run split) on the same targets\n")
    w("Production pass %d of %d; the v1 refit here agrees with production on %d of %d targets (median |dZ| %s, "
      "|dA| %s, |dC| %s).\n" % (int(pp.sum()), len(both), agree, len(both),
                                f2((v1.free0_tk.astype(float) - prod.free0_tk.astype(float)).abs().median(), 3),
                                f2((v1.freeM_topk_tk.astype(float) - prod.freeM_topk_tk.astype(float)).abs().median(), 3),
                                f2((v1.freeN_topk_tk.astype(float) - prod.freeN_topk_tk.astype(float)).abs().median(), 3)))

    # ---------------------------------------------------------------- per target
    w("## Per target\n")
    w("| target | cell | v1 n | v2 n | v1 Z / A / C / nec | v2 Z / A / C / nec | v1 pass | v2 pass | v1 amp | v2 amp |"
      " fit s v1 / v2 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in both:
        r1, r2 = v1.loc[k].to_dict(), v2.loc[k].to_dict()
        zacn = lambda r: " / ".join(f2(r.get(x)) for _, x in SCORES[:4])   # noqa: E731
        w("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s / %s |" % (
            k, r1["cell"], r1["n_nodes"], r2["n_nodes"], zacn(r1), zacn(r2), "Y" if p1[k] else "n", "Y" if p2[k] else "n",
            "Y" if r1.get("amp_any") else "n", "Y" if r2.get("amp_any") else "n", f2(r1.get("t_fit"), 0),
            f2(r2.get("t_fit"), 0)))
    if len(err):
        w("\n## Error rows\n")
        for _, r in err.iterrows():
            w("- %s %s: %s" % (r.seed, r.arm, r.error))
    (OUT93 / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
