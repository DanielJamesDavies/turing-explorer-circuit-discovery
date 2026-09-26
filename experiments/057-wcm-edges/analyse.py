"""057 follow-up stats from the saved edge matrices (data/<arm>/<target>.pt).

Per target x ablation method:
  direct_share   sum w(u->tgt) / sum A_u: share of the circuit's attribution that reaches the target directly
  route_share    sum_{d,u} |E| / sum_u |A_u|: member-edge mass relative to node mass
  top10_mass     share of sum|E| held by the 10 largest edges; n_half = edges needed for half of sum|E|
Across methods (per target):
  jacc_1e-3      mean pairwise Jaccard of the edge sets at theta 1e-3 (Z-A, Z-C, A-C)
  cons_1e-3      edges above 1e-3 in all three with one sign; union_1e-3
  top_overlap    share of each method's top-20 edges (by |w|) that are in the other methods' top-20, mean

  PYTHONPATH=src python experiments/057-wcm-edges/analyse.py   (env ARM, default rkeep3e3fix)
"""
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ARM = os.environ.get("ARM", "rkeep3e3fix")
PIS = ("Z", "A", "C")


def main():
    rows, lines = [], []
    for p in sorted((HERE / "data" / ARM).glob("*.pt")):
        d = torch.load(p, weights_only=False)
        if len(d["nodes"]) < 2 or all(float(d[pi]["E"].abs().sum()) == 0 for pi in PIS):
            continue
        r = dict(seed=p.stem, n=len(d["nodes"]))
        sets3, tops = {}, {}
        for pi in PIS:
            A, E, Et = (d[pi][k].double() for k in ("A", "E", "Et"))
            e = E.abs().flatten(); tot = float(e.sum())
            srt = torch.sort(e, descending=True).values
            cum = torch.cumsum(srt, 0) / max(tot, 1e-12)
            r["direct_share_" + pi] = float(Et.sum() / A.sum()) if float(A.sum()) != 0 else np.nan
            r["route_share_" + pi] = tot / float(A.abs().sum())
            r["top10_mass_" + pi] = float(cum[9]) if len(cum) > 9 else np.nan
            r["n_half_" + pi] = int((cum < 0.5).sum()) + 1
            tau = 1e-3 * float(A.abs().sum())
            sets3[pi] = (E.abs() >= tau, torch.sign(E))
            tops[pi] = set(torch.argsort(e, descending=True)[:20].tolist())
        js = [float((sets3[a][0] & sets3[b][0]).sum()) / max(1.0, float((sets3[a][0] | sets3[b][0]).sum()))
              for a, b in combinations(PIS, 2)]
        r["jacc_1e-3"] = float(np.mean(js))
        same = (sets3["Z"][1] == sets3["A"][1]) & (sets3["A"][1] == sets3["C"][1])
        r["cons_1e-3"] = int((sets3["Z"][0] & sets3["A"][0] & sets3["C"][0] & same).sum())
        r["union_1e-3"] = int((sets3["Z"][0] | sets3["A"][0] | sets3["C"][0]).sum())
        r["top20_overlap"] = float(np.mean([len(tops[a] & tops[b]) / 20 for a, b in combinations(PIS, 2)]))
        rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows)
    txt = df.round(3).to_string(index=False) + "\n\nmedians:\n" + df.drop(columns="seed").median().round(3).to_string()
    print(txt)
    (HERE / "results" / ("analyse_%s.md" % ARM)).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
