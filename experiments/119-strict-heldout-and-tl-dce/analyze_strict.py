"""119 #3 analysis: production vs local-training-donor vs local-held-out-donor strict resample rows.

Scoring is 109 t4_robustness.strict() exactly (copied below as score_rows, the only change: rows are passed in, not
found in the archives): Z from the all-position anchor means, A / C / R resample faithfulness rs_*, joined with the
production main row (held-out strongest); pass = Z, A, C (or R) in merge.BAND and production necessity >= NEC_MIN;
vacuous by merge.vacuous on the production row.

  python -X utf8 experiments/119-strict-heldout-and-tl-dce/analyze_strict.py
Writes out/strict_summary.json and out/strict_summary.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402


def in_band(x):
    x = pd.to_numeric(x, errors="coerce")
    return x.notna() & (x >= L.BAND[0]) & (x <= L.BAND[1])


def score_rows(model, rows):
    """t4_robustness.strict(), on given rows. Returns (per-target frame, cells, cond, medians)."""
    d = L.frame(rows, keys=("seed",)).set_index("seed")
    ev = L.main_eval(model)
    p = ev[ev.held == "strong"].set_index("seed")
    d = d.drop(columns=[c for c in ("layer", "kind", "band") if c in d]).join(
        p[L.HEAD + [L.NEC, "vac", "pass", "n", "band", "kind"]], how="inner")
    d = d[~d.vac]
    nec = d[L.NEC] >= L.NEC_MIN
    e = d["ap_Z_empty_anchor_mean"]
    for v in ("wcm", "a1", "rand"):
        if "rs_A_%s" % v not in d:
            continue
        z = (d["ap_Z_%s_anchor_mean" % v] - e) / (d["a_pos"] - e)
        d["rs_pass_%s" % v] = in_band(z) & in_band(d["rs_A_%s" % v]) & in_band(d["rs_C_%s" % v]) & nec
        d["rsR_pass_%s" % v] = in_band(z) & in_band(d["rs_A_%s" % v]) & in_band(d["rs_R_%s" % v]) & nec
    hr = d[d.has_rand.fillna(False).astype(bool)]
    cells = {}
    for lab, x, col in (("production", d, "pass"), ("resample A,C", d, "rs_pass_wcm"),
                        ("resample A,R", d, "rsR_pass_wcm"), ("resample A,C, alpha=1", d, "rs_pass_a1"),
                        ("resample A,C, random null", hr, "rs_pass_rand")):
        cells[lab] = L.pass_cell(x[col].astype(bool))
    cond = L.pass_cell(d[d["pass"]]["rs_pass_wcm"])
    med = {k: float(d["rs_%s_wcm" % k].median()) for k in "ACR"}
    return d, cells, cond, med


def rows_of(path):
    return L.read_rows([path], "rows")


def main():
    out = {}
    md = ["# 119 strict resample: production vs local, training vs held-out C donors", ""]
    for model in ("pythia-70m", "gpt2-small"):
        prod_dir = L.find(model, "out", "strict")
        conf = {"production (H100, train C)": L.read_rows(prod_dir, "rows"),
                "local, train C": rows_of(HERE / "out" / ("strict_%s_train" % model)),
                "local, held-out C": rows_of(HERE / "out" / ("strict_%s_held" % model))}
        res, frames = {}, {}
        md += ["## %s" % L.LABEL[model], "",
               "| config | rows (err) | scored | resample A,C | among prod passes | A,R | A,C alpha=1 | A,C random null |"
               " median rs A / C / R |", "|---|---|---|---|---|---|---|---|---|"]
        for name, rows in conf.items():
            if not rows:
                md.append("| %s | missing | | | | | | | |" % name)
                continue
            d, cells, cond, med = score_rows(model, rows)
            frames[name] = d
            res[name] = dict(rows=len(rows), errors=sum("error" in r for r in rows), scored=len(d),
                             cells=cells, cond=cond, med=med)
            md.append("| %s | %d (%d) | %d | %s | %s | %s | %s | %s | %.3f / %.3f / %.3f |" % (
                name, len(rows), res[name]["errors"], len(d), L.fmt_pass(cells["resample A,C"]), L.fmt_pass(cond),
                L.fmt_pass(cells["resample A,R"]), L.fmt_pass(cells["resample A,C, alpha=1"]),
                L.fmt_pass(cells["resample A,C, random null"]), med["A"], med["C"], med["R"]))
        # paired comparisons on common scored targets
        pairs = {}
        for a, b in (("local, train C", "local, held-out C"), ("production (H100, train C)", "local, train C"),
                     ("production (H100, train C)", "local, held-out C")):
            if a not in frames or b not in frames:
                continue
            fa, fb = frames[a], frames[b]
            ix = fa.index.intersection(fb.index)
            pa, pb = fa.loc[ix, "rs_pass_wcm"].astype(float), fb.loc[ix, "rs_pass_wcm"].astype(float)
            lo, hi = L.boot_paired_diff(pa.values, pb.values)
            dif = {}
            for k in "ACR":
                c = "rs_%s_wcm" % k
                x = (pd.to_numeric(fa.loc[ix, c]) - pd.to_numeric(fb.loc[ix, c])).abs()
                dif[k] = dict(median_abs=float(x.median()), max_abs=float(x.max()))
            pr = (fa.loc[ix, "pass"] == fb.loc[ix, "pass"]).mean()
            pairs["%s vs %s" % (a, b)] = dict(
                n=len(ix), pass_a=100 * pa.mean(), pass_b=100 * pb.mean(), diff_pp=100 * (pa.mean() - pb.mean()),
                diff_lo=100 * lo, diff_hi=100 * hi, a_only=int(((pa == 1) & (pb == 0)).sum()),
                b_only=int(((pa == 0) & (pb == 1)).sum()), rs_absdiff=dif, prod_pass_agree=float(pr))
        md += ["", "Paired (same targets), resample A,C pass:", ""]
        for k, v in pairs.items():
            md.append("- %s: n %d, %.1f vs %.1f%%, diff %+.1f pp [%+.1f, %+.1f]; discordant %d / %d; median |d rs| "
                      "A %.3f C %.3f R %.3f" % (k, v["n"], v["pass_a"], v["pass_b"], v["diff_pp"], v["diff_lo"],
                                                v["diff_hi"], v["a_only"], v["b_only"], v["rs_absdiff"]["A"]["median_abs"],
                                                v["rs_absdiff"]["C"]["median_abs"], v["rs_absdiff"]["R"]["median_abs"]))
        md.append("")
        out[model] = dict(configs=res, pairs=pairs)
    (HERE / "out").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "out" / "strict_summary.json", "w"), indent=1, default=float)
    (HERE / "out" / "strict_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
