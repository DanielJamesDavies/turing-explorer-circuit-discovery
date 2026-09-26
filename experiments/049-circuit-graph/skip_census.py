"""SKIP CENSUS + COST from the per-seed task_metrics: seeds attempted vs
circuits produced by (layer, kind), and seconds per seed by layer.

  python experiments/049-circuit-graph/skip_census.py [DATA_DIR]
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
KINDS = ["attn", "mlp", "resid"]
done, acc, dur = Counter(), Counter(), defaultdict(list)
for p in sorted(glob.glob(str(DATA / "task_metrics.shard*.jsonl"))):
    for ln in open(p):
        r = json.loads(ln)
        L, k = r["comp_idx"] // 3, r["comp_idx"] % 3
        done[(L, k)] += 1
        acc[(L, k)] += r["accepted_circuit_count"]
        if r["accepted_circuit_count"]:
            dur[L].append(r["total_s"])
print("layer | kind  | attempted | circuits | skipped | skip%% | median s (fitted)")
tot_d = tot_a = 0
for L in sorted({l for l, _ in done}):
    for k in range(3):
        d, a = done[(L, k)], acc[(L, k)]
        if not d:
            continue
        tot_d += d; tot_a += a
        print("  L%-2d | %-5s | %9d | %8d | %7d | %4.1f%% | %s"
              % (L, KINDS[k], d, a, d - a, 100 * (d - a) / d, ("%.0f" % np.median(dur[L])) if dur[L] else "-"))
print("TOTAL attempted %d | circuits %d | skipped %d (%.1f%%)" % (tot_d, tot_a, tot_d - tot_a, 100 * (tot_d - tot_a) / tot_d))
print("\nseconds per fitted seed by layer (median / p90):",
      {("L%d" % L): ("%.0f/%.0f" % (np.median(v), np.percentile(v, 90))) for L, v in sorted(dur.items())})
