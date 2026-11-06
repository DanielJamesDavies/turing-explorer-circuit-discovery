"""Compare two run_ge.py output dirs circuit by circuit (node sets, attributions, leaf parts); prints the max differences.
Used for the A7 smoke: the upstream-only splice (run_ge.upstream) must reproduce the all-24 splice.

  python -I -B check_ge_smoke.py <dir a> <dir b>     (any env with torch)
"""
import sys
from pathlib import Path

import torch


def main(a, b):
    fa = sorted(Path(a, "circuits").glob("*/p*.pt"))
    worst = dict(attr=0.0, parts=0.0, set_mismatch=0, posthoc_n_mismatch=0, files=0)
    for f in fa:
        g = Path(b, "circuits", f.parent.name, f.name)
        if not g.exists():
            print("missing in b:", g)
            continue
        x, y = torch.load(f), torch.load(g)
        worst["files"] += 1
        for hx, hy in zip(x["hier"], y["hier"]):
            if not torch.equal(hx["nodes"], hy["nodes"]):
                worst["set_mismatch"] += 1
                continue
            if hx["attr"].numel():
                worst["attr"] = max(worst["attr"], float((hx["attr"] - hy["attr"]).abs().max()))
            worst["parts"] = max(worst["parts"], max(abs(hx["parts"][k] - hy["parts"][k]) for k in hx["parts"]))
        for px, py in zip(x["posthoc"], y["posthoc"]):
            worst["posthoc_n_mismatch"] += int(px["n"] != py["n"])
            worst["parts"] = max(worst["parts"], max(abs(px["parts"][k] - py["parts"][k]) for k in px["parts"]))
    print(worst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
