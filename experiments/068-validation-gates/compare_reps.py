"""Bitwise comparison of two local golden refits: memberships and fitted coefficients (DAN-74 code-vs-hardware check).

r0 was fitted with the working-tree src (2026-10-09 changes: adapters, src scorer, data pass, selector fix); r_old with
the committed src (= the 15k run's commit 538cbed), on the same GPU. Identical circuits mean the 2026-10-09 changes
do not touch the TuringLLM fit, and the r0-vs-H100 differences in out_golden/report.md are hardware.

  PYTHONPATH=src python experiments/068-validation-gates/compare_reps.py        env: A (r0)  B (r_old)  OUT (./out_golden)
"""
import os
from pathlib import Path

import torch

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUT", str(HERE / "out_golden")))
A, B = os.environ.get("A", "r0"), os.environ.get("B", "r_old")


def members(c):
    out = {}
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind), int(f.index))] = float(md.get("amplitude", 1.0))
    return out


def main():
    da, db = OUT / "fits" / A / "circuits", OUT / "fits" / B / "circuits"
    keys = sorted(p.stem for p in db.glob("*.pt") if (da / p.name).exists())
    same, rows = 0, []
    for k in keys:
        ma = members(torch.load(da / ("%s.pt" % k), weights_only=False))
        mb = members(torch.load(db / ("%s.pt" % k), weights_only=False))
        sa, sb = set(ma), set(mb)
        j = len(sa & sb) / max(1, len(sa | sb))
        dalpha = max((abs(ma[x] - mb[x]) for x in sa & sb), default=0.0)
        ident = sa == sb and dalpha == 0.0
        same += ident
        rows.append((k, len(sa), len(sb), j, dalpha, ident))
    for k, na, nb, j, da_, ident in rows:
        print("%-16s n %4d / %4d  J %.4f  max|dalpha| %.2g  %s" % (k, na, nb, j, da_, "IDENTICAL" if ident else "DIFFERENT"))
    print("\n%s vs %s: %d / %d targets identical (memberships and alpha bitwise)" % (A, B, same, len(keys)))


if __name__ == "__main__":
    main()
