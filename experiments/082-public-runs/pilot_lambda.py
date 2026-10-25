"""LAMBDA PILOT REPORT (DAN-42, pod plan B5 / D2). REPORT-only, CPU: reads a driver MODE=sweep output, recommends, adopts
nothing.

The fits are `driver.py MODE=sweep SWEEP_ARMS=W_<lam>,...` over the pilot targets (targets_<family>_pilot.txt, disjoint
from main). Per WCM lambda and depth band (family_table.py: Pythia thirds L0-1 / L2-3 / L4-5; GPT-2 and TuringLLM
L0-4 / L5-7 / L8-11) this reports, on held-out strongest:
  - median circuit size (p25, p75), and the share of sizes in [10^2, 10^3]
  - the DAN-8 pass rate (merge.passes; vacuous denominators excluded and counted). REPORTED ONLY: never used to choose
  - rejected / error rows and target coverage

PRE-STATED RULE (D2, applied mechanically; sizes only):
  keep lambda = 1e-3 if its POOLED median size (all bands; every fitted held-out-strongest circuit of that arm) lies
  in [100, 1000] (inclusive); otherwise choose the grid lambda whose pooled median size is nearest 494 nodes (TuringLLM's
  median at 1e-3) ON A LOG SCALE (decided 2026-10-24), nearest = smallest |log10(median / 494)|, so 247 and 988 are
  equally near; a tie goes to the lambda nearer 1e-3 in log10, then to the larger lambda (the sparser circuit). Lambdas
  with no fitted circuit are not candidates. 1e-3 must be in the grid.
The recommendation goes to a JSON with "adopted": false; adopting it is Daniel's call (C3), and a lambda other than
1e-3 changes the paper's "applied unchanged" sentence (main.tex L403).

  OUT=experiments/082-public-runs/out_pythia-70m PYTHONPATH=src python experiments/082-public-runs/pilot_lambda.py \
      --family pythia-70m --targets experiments/082-public-runs/targets_pythia-70m_pilot.txt
  -> OUT/pilot_lambda.json (or --json PATH)
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "062-h100-protocol-v1"))
from family_table import band_label, family  # noqa: E402
from merge import passes, vacuous  # noqa: E402  (the DAN-8 rule, single source)

PROD_LAM = 1e-3
TARGET_MEDIAN = 494                 # TuringLLM's median WCM circuit size at lambda = 1e-3 (15k run)
SIZE_RANGE = (100, 1000)            # [10^2, 10^3], inclusive
RULE = ("keep lambda = %g unless its pooled median size leaves [%d, %d]; else the grid lambda whose pooled median size "
        "is nearest %d nodes on a log scale (|log10(median / %d)|; ties: nearer %g in log10, then the larger lambda). "
        "Size only; pass rate is reported, never used." % (PROD_LAM, SIZE_RANGE[0], SIZE_RANGE[1], TARGET_MEDIAN,
                                                           TARGET_MEDIAN, PROD_LAM))


def log_dist(median, target=TARGET_MEDIAN):
    """|log10(median / target)|, rounded so exact log ties (e.g. 247 and 988 around 494) compare equal."""
    return round(abs(math.log10(median / target)), 9) if median > 0 else float("inf")


def read_rows(out):
    rows = []
    for f in sorted(glob.glob(str(Path(out) / "sweep" / "eval.*.jsonl"))):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


def prepare(rows, fam):
    """Split the sweep rows into fitted WCM circuits (plain W_<lam> arms, held-out strongest), bookkeeping, and the
    targets with any row per lambda (fitted, rejected or error; a target without contexts gets no row at all)."""
    if rows.empty:
        return rows, {}, {}
    rows = rows.drop_duplicates(["seed", "arm"], keep="last").copy()
    plain = rows.arm.astype(str).str.fullmatch(r"W_[0-9.e+-]+")
    book = {"ignored_arms": sorted(set(rows.arm[~plain].astype(str)))}
    rows = rows[plain].copy()
    rows["lam"] = rows.arm.str[2:].astype(float)            # skip / error rows carry only the arm name
    seen = {lam: set(g.seed) for lam, g in rows.groupby("lam")}
    bad = pd.Series(False, index=rows.index)
    for col in ("error", "skip"):
        if col in rows:
            bad |= rows[col].notna()
    book["not_fitted"] = {"%g" % lam: dict(g.get("skip", pd.Series(dtype=object)).dropna().value_counts().to_dict(),
                                           error=int(g["error"].notna().sum()) if "error" in g else 0)
                          for lam, g in rows[bad].groupby("lam")}
    fit = rows[~bad].copy()
    if "held" in fit:
        fit = fit[fit.held == "strong"]
    bands = [(band_label(lo, hi), lo, hi) for lo, hi in fam["bands"]]
    fit["layer"] = fit.seed.str.split(".").str[0].astype(int)
    fit["depth"] = fit.layer.map(lambda L: next((b for b, _, hi in bands if L <= hi), bands[-1][0]))
    return fit, book, seen


def stats(g):
    """Size and (reported) pass-rate statistics of one group of fitted circuits."""
    n = g["n"].astype(float)
    vac = vacuous(g)
    ok = g[~vac]
    return dict(circuits=int(len(g)), targets=int(g.seed.nunique()),
                median_size=float(n.median()) if len(n) else None,
                p25_size=float(n.quantile(.25)) if len(n) else None, p75_size=float(n.quantile(.75)) if len(n) else None,
                share_in_1e2_1e3=float(((n >= SIZE_RANGE[0]) & (n <= SIZE_RANGE[1])).mean()) if len(n) else None,
                pass_rate_reported=float(passes(ok).mean()) if len(ok) else None, vacuous=int(vac.sum()))


def choose(pooled, prod_lam=PROD_LAM, target=TARGET_MEDIAN, lo_hi=SIZE_RANGE):
    """The pre-stated rule on {lambda: pooled median size}. Returns (lambda or None, reason)."""
    grid = {lam: m for lam, m in pooled.items() if m is not None and not (isinstance(m, float) and math.isnan(m))}
    if not any(math.isclose(lam, prod_lam) for lam in pooled):
        return None, "lambda %g is not in the pilot grid: the rule cannot be applied" % prod_lam
    prod = next(lam for lam in pooled if math.isclose(lam, prod_lam))
    if prod not in grid:
        return None, "no fitted circuit at lambda %g: the rule cannot be applied" % prod_lam
    if lo_hi[0] <= grid[prod] <= lo_hi[1]:
        return prod, "kept: pooled median size %.0f at lambda %g is inside [%d, %d]" % (grid[prod], prod, *lo_hi)
    best = min(grid, key=lambda lam: (log_dist(grid[lam], target), round(abs(math.log10(lam / prod_lam)), 9), -lam))
    return best, ("pooled median size %.0f at lambda %g is outside [%d, %d]; nearest to %d on a log scale: lambda %g "
                  "(median %.0f, |log10 ratio| %.3f)" % (grid[prod], prod, lo_hi[0], lo_hi[1], target, best, grid[best],
                                                         log_dist(grid[best], target)))


def report(out, fam_name, targets=None, prod_lam=PROD_LAM, target=TARGET_MEDIAN):
    fam = family(fam_name)
    fit, book, seen = prepare(read_rows(out), fam)
    lams = sorted(seen)
    want = [t for t in Path(targets).read_text().split() if t] if targets else None
    union = set().union(*seen.values()) if seen else set()
    per = []
    for lam in lams:
        g = fit[np.isclose(fit.lam, lam)]
        row = dict(lam=lam, pooled=stats(g), by_band={})
        for lo, hi in fam["bands"]:
            b = band_label(lo, hi)
            row["by_band"][b] = stats(g[g.depth == b])
        # a target with a row at another lambda but none here: this arm has not reached it yet
        row["targets_missing"] = int(len(union - seen[lam]))
        per.append(row)
    coverage = None
    if want is not None:
        coverage = dict(listed=len(want), with_rows=len(union & set(want)), not_in_list=sorted(union - set(want)),
                        no_row_at_any_lambda=len(set(want) - union))   # context skips (too_few_contexts / no_contrast)
    # complete: every lambda has a row for every target that has a row anywhere, and the run is over the given list
    complete = bool(per) and all(r["targets_missing"] == 0 for r in per) and (
        coverage is not None and not coverage["not_in_list"] and coverage["with_rows"] > 0)
    pooled = {r["lam"]: r["pooled"]["median_size"] for r in per}
    lam, reason = choose(pooled, prod_lam, target) if per else (None, "no sweep rows under %s" % out)
    paired = None
    if per:
        common = set.intersection(*(set(fit[np.isclose(fit.lam, r["lam"])].seed) for r in per))
        paired = dict(targets=len(common), median_size={"%g" % r["lam"]: float(fit[np.isclose(fit.lam, r["lam"])
                                                                                    & fit.seed.isin(common)].n.median())
                                                        if common else None for r in per})
    return dict(family=fam["name"], out=str(out), rule=RULE, prod_lam=prod_lam, target_median=target,
                size_range=list(SIZE_RANGE), bands=[band_label(lo, hi) for lo, hi in fam["bands"]],
                grid=lams, per_lambda=per, pooled_median_size={"%g" % k: v for k, v in pooled.items()},
                paired_reference=paired, bookkeeping=book, targets_file=str(targets) if targets else None,
                coverage=coverage, complete=complete, recommended_lam=lam, reason=reason, adopted=False)


def fmt(v, spec):
    return "-" if v is None else spec % v


def print_report(r):
    print("lambda pilot: %s, %s" % (r["family"], r["out"]))
    print("rule: " + r["rule"])
    for row in r["per_lambda"]:
        print("\nlambda %g%s" % (row["lam"], "" if not row["targets_missing"]
                                  else "  (%d targets not reached yet)" % row["targets_missing"]))
        for name, s in [("pooled", row["pooled"])] + list(row["by_band"].items()):
            print("  %-7s %4d circuits | size median %s (p25 %s, p75 %s) | in [1e2, 1e3] %s | pass (reported) %s"
                  % (name, s["circuits"], fmt(s["median_size"], "%.0f"), fmt(s["p25_size"], "%.0f"),
                     fmt(s["p75_size"], "%.0f"), fmt(s["share_in_1e2_1e3"] and 100 * s["share_in_1e2_1e3"], "%.0f%%"),
                     fmt(s["pass_rate_reported"] and 100 * s["pass_rate_reported"], "%.0f%%")))
    if r["bookkeeping"].get("not_fitted"):
        print("\nnot fitted (skip / error) by lambda: %s" % r["bookkeeping"]["not_fitted"])
    if r["bookkeeping"].get("ignored_arms"):
        print("ignored arms (not plain W_<lambda>): %s" % r["bookkeeping"]["ignored_arms"])
    if r["coverage"]:
        c = r["coverage"]
        print("coverage: %d listed, %d with rows, %d with no row at any lambda (context skips), %d rows not in the list"
              % (c["listed"], c["with_rows"], c["no_row_at_any_lambda"], len(c["not_in_list"])))
    if r["paired_reference"]:
        print("paired reference (targets fitted at every lambda: %d): %s" % (
            r["paired_reference"]["targets"], r["paired_reference"]["median_size"]))
    print("\nRECOMMENDATION (not adopted): lambda %s; %s%s" % (
        fmt(r["recommended_lam"], "%g"), r["reason"], "" if r["complete"] else "  [pilot INCOMPLETE or no --targets]"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.environ.get("OUT"), help="the sweep run's OUT (default: env OUT)")
    ap.add_argument("--family", default=os.environ.get("FAMILY", "turingllm"))
    ap.add_argument("--targets", help="the pilot target list, for coverage")
    ap.add_argument("--json", help="default OUT/pilot_lambda.json")
    a = ap.parse_args(argv)
    if not a.out:
        ap.error("--out (or env OUT) is required")
    r = report(a.out, a.family, a.targets)
    print_report(r)
    path = Path(a.json) if a.json else Path(a.out) / "pilot_lambda.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=2, default=float) + "\n", encoding="utf-8")
    print("wrote", path)
    return r


if __name__ == "__main__":
    main()
