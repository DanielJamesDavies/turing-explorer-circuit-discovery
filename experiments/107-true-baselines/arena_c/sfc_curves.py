"""Arena C, check F4: SFC's Fig. 3 from our run of their code (notebook cells 6-7, as numbers instead of a plot).

The notebook averages the four tasks' faithfulness (and completeness) at node counts on a log grid, each task's curve
linearly interpolated (scipy interp1d) between its 15 thresholds, over the range every task covers. We print the same
mean curve and the F4 readings (specs/sfc.md §10):
  features: faithfulness >= 0.5 at <= 200 nodes (paper: about 100)
  neurons:  faithfulness 0.5 reached between about 750 and 3,000 nodes (paper: about 1,500)
  features: completeness <= 0.2 within the first few hundred nodes
Run with any python that has numpy + scipy:  python sfc_curves.py out/sfc/faithfulness/summary_<...>.json
"""
import json
import math
import sys

import numpy as np
from scipy import interpolate


def mean_curve(sub, key):
    datasets = list(sub)
    x_min = max(min(p["n_nodes"] for p in sub[d]["curve"]) for d in datasets) + 1
    x_max = min(max(p["n_nodes"] for p in sub[d]["curve"]) for d in datasets) - 1
    fs = {d: interpolate.interp1d([p["n_nodes"] for p in sub[d]["curve"]], [p[key] for p in sub[d]["curve"]])
          for d in datasets}
    xs = np.logspace(math.log10(x_min), math.log10(x_max), 100)
    return xs, np.array([sum(float(f(x)) for f in fs.values()) / len(fs) for x in xs])


def first_x(xs, ys, level):
    hit = np.nonzero(ys >= level)[0]
    return float(xs[hit[0]]) if len(hit) else None


def main(path):
    s = json.load(open(path))
    report = {}
    for setting, sub in s["results"].items():
        print(f"\n== {setting}")
        for d, r in sub.items():
            pts = " ".join(f"{p['n_nodes']}:{p['faithfulness']:.2f}" for p in r["curve"])
            print(f"  {d:10s} F(M) {r['fm']:.3f} F(∅) {r['fempty']:.3f}  n:faith  {pts}")
        xs, f = mean_curve(sub, "faithfulness")
        xc, c = mean_curve(sub, "completeness")
        rep = {"range": [float(xs[0]), float(xs[-1])], "faith_0.5_at": first_x(xs, f, 0.5),
               "faith_at_100": float(np.interp(100, xs, f)) if xs[0] <= 100 <= xs[-1] else None,
               "faith_at_200": float(np.interp(200, xs, f)) if xs[0] <= 200 <= xs[-1] else None,
               "completeness_at_300": float(np.interp(300, xc, c)) if xc[0] <= 300 <= xc[-1] else None}
        report[setting] = rep
        print("  mean curve:", json.dumps(rep))
    out = path.replace(".json", "_F4.json")
    json.dump(report, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
