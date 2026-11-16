"""Effective size: per circuit, the fewest members whose summed weight reaches 50% / 90% of the circuit's total.

The circuit records store two per-member numbers and NO contribution measure:
  m      the node's `attribution_score` = the converged mask gate sigmoid(theta) of the member (learned_mask.py
         selection: kept when m > keep_threshold 0.5), so m lies in (0.5, 1];
  alpha  the node's `amplitude`, the fitted coefficient.
Neither is a contribution to the target (activation x effect). Both are reported so the shape is on record; a true
contribution-weighted effective size needs the 084 profile pass (activation x gradient of the target pre-activation at
the anchor, per member, on GPU), which the archives keep only as per-set summaries (harder_nulls/profile*.jsonl).

Population: held-out-strongest non-vacuous main circuits. Writes out/run2026_coef_effective.csv (per model x band),
out/run2026_coef_effective_per_circuit.csv, paper/tables/run2026_coef_effective.tex
  python -I -X utf8 experiments/113-coefficients-shape/c5_effective.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lib113 as C  # noqa: E402

L = C.L


def n_cover(df, col, q):
    """Per seed: the smallest k with the top-k |col| summing to >= q of the total."""
    d = df[["seed", col]].assign(w=df[col].abs()).sort_values(["seed", "w"], ascending=[True, False])
    d["cum"] = d.groupby("seed").w.cumsum()
    d["tot"] = d.groupby("seed").w.transform("sum")
    d["rank"] = d.groupby("seed").cumcount() + 1
    return d[d.cum < q * d.tot - 1e-12].groupby("seed")["rank"].max().reindex(d.seed.unique()).fillna(0).astype(int) + 1


def main():
    per, rows = [], []
    for model in C.MODELS:
        pop = C.strong_population(model)
        df = C.members(model, "main")
        df = df[df.seed.isin(pop.index)]
        out = pd.DataFrame({"n": df.groupby("seed").size()})
        for col in ("m", "alpha"):
            for q in (0.5, 0.9):
                out["%s_n%d" % (col, int(100 * q))] = n_cover(df, col, q)
        out["band"] = pop.loc[out.index, "band"]
        out["model"] = model
        per.append(out.reset_index())
        for b in ["all"] + [x for x, _, _ in L.bands_of(model)]:
            g = out if b == "all" else out[out.band == b]
            r = dict(model=model, band=b, circuits=len(g), n_median=g.n.median())
            for c in [c for c in out.columns if c.startswith(("m_", "alpha_"))]:
                r[c + "_median"] = g[c].median()
                r[c + "_frac_median"] = (g[c] / g.n).median()
            rows.append(r)
    R = pd.DataFrame(rows)
    pd.concat(per).to_csv(C.OUT / (C.PREFIX + "effective_per_circuit.csv"), index=False)
    print(R.round(3).to_string(index=False))
    tex = []
    for r in R.itertuples():
        tex.append([L.LABEL[r.model] if r.band == "all" else "", "all layers" if r.band == "all" else
                    "layers " + r.band[1:].replace("-", "--"), L.tex_int(r.circuits), L.tex_int(r.n_median),
                    "%s / %s" % (L.tex_int(r.alpha_n50_median), L.tex_int(r.alpha_n90_median)),
                    "%.2f / %.2f" % (r.alpha_n50_frac_median, r.alpha_n90_frac_median),
                    "%s / %s" % (L.tex_int(r.m_n50_median), L.tex_int(r.m_n90_median))])
    C.write_table("effective", tex, ["Model", "Depth", "Circuits", "Size", r"$N_{50}$ / $N_{90}$ by $|\alpha|$",
                                     "as share of size", r"$N_{50}$ / $N_{90}$ by gate $m$"], "llrrlll",
                  footer=[r"Medians over circuits. $N_q$: fewest members carrying $q$ of the circuit's summed weight. "
                          r"No contribution measure is stored per member; $|\alpha|$ and the mask gate $m$ are weights, "
                          r"not contributions."], csv_df=R)


if __name__ == "__main__":
    main()
