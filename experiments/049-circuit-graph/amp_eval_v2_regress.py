"""REGRESSION of the v2 amplitude-aware pass against v1's stored rows.

  python experiments/049-circuit-graph/amp_eval_v2_regress.py <v2 jsonl> [<v1 jsonl>]

Compares every field the two passes share (v1 names) seed by seed: max / median abs diff on the raw
activations and on the ratios, and lists the worst seeds. Nulls (F0_perm, F0_rand) use a different RNG
in v2 (per-seed, resumable) and are reported but not expected to match.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
v2p = Path(sys.argv[1])
v1p = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results_full" / "amp_eval_subset.jsonl"
v1 = {}
for ln in open(v1p):
    r = json.loads(ln)
    if "a_pos_ho" in r:
        v1[r["seed"]] = r
v2 = {}
for ln in open(v2p):
    r = json.loads(ln)
    if "a_pos_ho" in r:
        v2[r["seed"]] = r
common = [k for k in v2 if k in v1]
print("v1 scored %d | v2 scored %d | common %d" % (len(v1), len(v2), len(common)))
RAW = ["a_pos_ho", "a_pos_tr", "e0_ho", "eMd_ho", "a_base", "ampF0_ho", "ampF0_tr", "F0a1_ho", "ampFMd_ho", "cf_amp_raw"]
RATIO = ["F0_amp", "F0_amp_tr", "F0_a1", "FMd_amp", "cf_amp", "cf_bare_anch", "sup_anch", "cf_bounded"]
NULLS = ["ampF0_perm", "F0_rand", "F0_perm", "F0_rand_frac"]
print("\n%-14s %10s %10s %10s %8s  worst seed (v1 -> v2)" % ("field", "max|d|", "median|d|", "max rel", "n exact"))
for grp, fields in (("RAW", RAW), ("RATIO", RATIO), ("NULLS (RNG differs)", NULLS)):
    print("-- " + grp)
    for f in fields:
        d, rel, keys = [], [], []
        for k in common:
            a, b = v1[k].get(f), v2[k].get(f)
            if a is None or b is None:
                continue
            d.append(abs(a - b)); rel.append(abs(a - b) / max(abs(a), 1e-6)); keys.append(k)
        if not d:
            continue
        i = int(np.argmax(d))
        print("%-14s %10.4g %10.4g %10.4g %8d  %s (%.4f -> %.4f)" % (f, max(d), float(np.median(d)), max(rel), sum(x == 0 for x in d),
                                                                  keys[i], v1[keys[i]][f], v2[keys[i]][f]))
mism = [k for k in common if v1[k]["n"] != v2[k]["n"]]
print("\nmember-count mismatches: %d" % len(mism))
thin = [k for k in common if v2[k].get("thin")]
print("v2 rows flagged thin among common: %d %s" % (len(thin), thin[:5]))
