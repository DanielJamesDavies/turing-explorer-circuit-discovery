"""Symmetric-band sensitivity of the pass rule. Production: Z, A, C in [0.75, 1.5] and necessity >= 0.9 (merge.py).
Alternatives (necessity unchanged): [1/1.5, 1.5] = [0.667, 1.5] and [0.75, 1/0.75] = [0.75, 1.333].
Population: main held-out-strongest rows, non-vacuous (merge.vacuous). CIs: bootstrap over targets; differences
against production: paired bootstrap. The production column is checked against merge.passes.

Writes out/run2026_coef_band.csv, paper/tables/run2026_coef_band.tex
  python -I -X utf8 experiments/113-coefficients-shape/c4_band.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lib113 as C  # noqa: E402

L = C.L
BANDS = [("production [0.75, 1.5]", (0.75, 1.5)), ("symmetric [0.667, 1.5]", (1 / 1.5, 1.5)),
         ("symmetric [0.75, 1.333]", (0.75, 1 / 0.75))]


def passes_band(df, lo, hi):
    inb = ((df[L.HEAD] >= lo) & (df[L.HEAD] <= hi)).all(axis=1)
    return (inb & (df[L.NEC] >= L.NEC_MIN)).values


def main():
    rows, tex = [], []
    for model in C.MODELS:
        ev = C.strong_population(model)
        base = passes_band(ev, *BANDS[0][1])
        assert (base == ev["pass"].values).all(), "production band does not reproduce merge.passes"
        for b in ["all"] + [x for x, _, _ in L.bands_of(model)]:
            g = ev if b == "all" else ev[ev.band == b]
            p0 = passes_band(g, *BANDS[0][1]).astype(float)
            cells = [L.LABEL[model] if b == "all" else "", "all layers" if b == "all" else "layers " + b[1:].replace("-", "--"),
                     L.tex_int(len(g))]
            for name, (lo, hi) in BANDS:
                p = passes_band(g, lo, hi).astype(float)
                ci = C.boot_ci(p)
                dlo, dhi = C.boot_paired_diff(p, p0)
                rows.append(dict(model=model, band=b, rule=name, lo=lo, hi=hi, n=len(g), k=int(p.sum()),
                                 pass_pct=100 * p.mean(), pass_lo=100 * ci[0], pass_hi=100 * ci[1],
                                 diff_pp=100 * (p.mean() - p0.mean()), diff_lo=100 * dlo, diff_hi=100 * dhi))
                c = "%.1f {[}%.1f, %.1f{]}" % (100 * p.mean(), 100 * ci[0], 100 * ci[1])
                if name != BANDS[0][0]:
                    c += " (%+.1f)" % (100 * (p.mean() - p0.mean()))
                cells.append(c)
            tex.append(cells)
        tex.append([r"\midrule"])
    df = pd.DataFrame(rows)
    print(df[df.band == "all"].round(2).to_string(index=False))
    C.write_table("band", tex[:-1], ["Model", "Depth", "Scored", r"$[0.75, 1.5]$ (production)",
                                     r"$[0.667, 1.5]$", r"$[0.75, 1.333]$"], "llrlll",
                  footer=[r"Pass (\%) with 95\% bootstrap CI over targets; in brackets after an alternative band, its "
                          r"difference from production (pp). Necessity $\ge 0.9$ in every rule."], csv_df=df)


if __name__ == "__main__":
    main()
