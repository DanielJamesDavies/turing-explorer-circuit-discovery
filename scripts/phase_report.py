"""WHERE DID THE TIME GO — per-phase breakdown from task-metrics files.

  python scripts/phase_report.py outputs/circuits/task_metrics*.jsonl

Reads the rows the discovery window writes (one per seed x method) and
reports, over the seeds that have a "phases" snapshot:
  * mean seconds per seed for each phase, and its share of seed time
  * the fit's internal split (fwd / bwd / penalty_bwd / opt) — exact only
    on a PHASE_SYNC=1 run, launch-time-only otherwise (stated in output)
  * the same, broken down by seed layer (deep seeds are the cost driver)
No torch needed; runs anywhere.
"""
import glob
import json
import os
import sys
from collections import defaultdict

N_KINDS = 3   # attn / mlp / resid — comp_idx = layer * N_KINDS + kind


def load(paths):
    rows = []
    for pat in paths:
        for f in sorted(glob.glob(pat)):
            for line in open(f):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("phases"):
                    rows.append(r)
    return rows


def agg(rows):
    tot = defaultdict(float)
    for r in rows:
        for k, v in r["phases"].items():
            if k.startswith("_"):
                continue
            tot[k] += v["s"]
    n = max(len(rows), 1)
    seed_total = sum(r.get("total_s", r.get("duration_s", 0.0)) for r in rows)
    return {k: v / n for k, v in tot.items()}, seed_total / n


def print_block(title, rows):
    if not rows:
        return
    means, seed_mean = agg(rows)
    print("\n%s  (%d seeds, mean %.1f s/seed)" % (title, len(rows), seed_mean))
    print("  %-24s %9s %7s" % ("phase", "mean s", "share"))
    order = sorted(means, key=lambda k: -means[k])
    for k in order:
        print("  %-24s %9.2f %6.0f%%" % (k, means[k],
                                         100 * means[k] / max(seed_mean, 1e-9)))
    coarse = sum(means.get(k, 0.0) for k in
                 ("seed.probes", "seed.fit", "seed.assemble", "seed.cf_eval",
                  "seed.eval_negs", "seed.prune_loo", "seed.prune_recurrence",
                  "seed.prune_magnitude", "window.post_analysis",
                  "window.node_presence", "window.consolidate",
                  "window.save_store"))
    print("  %-24s %9.2f %6.0f%%   (untimed remainder)"
          % ("other", seed_mean - coarse,
             100 * (seed_mean - coarse) / max(seed_mean, 1e-9)))
    fit_inner = sum(means.get(k, 0.0) for k in
                    ("fit.fwd", "fit.bwd", "fit.penalty_bwd", "fit.opt"))
    if means.get("seed.fit"):
        print("  fit internals sum %.2f s of seed.fit %.2f s (%s)"
              % (fit_inner, means["seed.fit"],
                 "exact: PHASE_SYNC run" if all(
                     r["phases"].get("_sync", {}).get("s") == 1.0 for r in rows)
                 else "launch-time only (run without PHASE_SYNC=1)"))


def main():
    paths = sys.argv[1:] or ["outputs/circuits/task_metrics*.jsonl"]
    rows = load(paths)
    if not rows:
        sys.exit("no rows with phase snapshots found in %s" % paths)
    print_block("ALL SEEDS", rows)
    fitted = [r for r in rows if r["phases"].get("seed.fit", {}).get("s", 0) > 5]
    print_block("SEEDS WITH A REAL FIT (seed.fit > 5 s)", fitted)
    by_layer = defaultdict(list)
    for r in fitted:
        if r.get("comp_idx") is not None:
            by_layer[r["comp_idx"] // N_KINDS].append(r)
    print("\nPER LAYER (fitted seeds): layer | n | seed mean | fit mean | cf_eval mean")
    for L in sorted(by_layer):
        means, seed_mean = agg(by_layer[L])
        print("  L%-2d | %3d | %7.1f s | %7.1f s | %6.1f s"
              % (L, len(by_layer[L]), seed_mean, means.get("seed.fit", 0),
                 means.get("seed.cf_eval", 0)))


if __name__ == "__main__":
    main()
