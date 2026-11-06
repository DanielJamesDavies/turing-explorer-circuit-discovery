"""TABLE tab:matrix (Table 2, app:method-comparison; DAN-148 + DAN-159): the LaTeX body of the method comparison in
paper/main.tex, from experiments/083-v1-comparison/table2.py's summary (table2.csv), with MIB-style marking.

083 table2.py computes every number (per-target rows, the DAN-8 rule, vacuous exclusion, depth bands); this script
only maps its rows onto the paper's row names and columns and marks the best entries. Paper columns: median size n;
medians of Z, A, C faithfulness and necessity (phi_sup); pass rate (%) by depth band (L0-4, L5-7, L8-11).

MARKING (MIB style), per column over every row of the table: \\textbf = best, \\underline = second best; ties share the
mark. Faithfulness Z / A / C: closest to 1 (smallest |median - 1|). Necessity and pass rates: highest. Ranked at the
printed precision (2 dp, whole percent), so cells that print equal share a mark. Size is not marked (smaller is not
better at unequal faithfulness). The exact-effect block (a subset of targets) is printed but not ranked. A cell with
no defined value is a dash and is never marked; a band with no non-vacuous circuit has no pass rate (dash).

  python -X utf8 paper/tables/matrix_body.py                         # v2: 082/out_turingllm/dan159/table2.csv
  python -X utf8 paper/tables/matrix_body.py --csv experiments/083-v1-comparison/out_smoke/table2.csv --tag smoke

Writes paper/tables/out/<tag>/matrix_body.tex (the rows between the header \\midrule and \\bottomrule, section
headers included) and matrix_rows.csv (the 083 row each paper row came from). 083 rows the paper does not list are
reported, not dropped silently.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import common as K

DEFAULT_CSV = K.ROOT / "experiments" / "082-public-runs" / "out_turingllm" / "dan159" / "table2.csv"
BANDS = ("L0-4", "L5-7", "L8-11")
# (section header, [(paper row label, regex on the 083 row label)])
LAYOUT = [
    ("Natural operating point", [
        ("weighted circuit masking, $\\lambda = 10^{-3}$", r"^WCM, λ 1e-3"),
        ("unweighted circuit masking, $\\lambda = 10^{-3}$", r"^unweighted, λ 1e-3"),
        ("AtP, position union", r"^attribution patching \(AtP\), natural size"),
        ("IG, position union", r"^path attribution \(IG\), natural size"),
        ("hierarchical attribution, bisected", r"^hierarchical attribution"),
    ]),
    ("Truncated to the weighted circuit's size", [
        ("AtP, largest score", r"^attribution patching \(AtP\), top-n by max"),
        ("AtP, mean $|$score$|$", r"^attribution patching \(AtP\), top-n by mean"),
        ("AtP, aggregate", r"^attribution patching \(AtP\), top-n by the aggregate"),
        ("IG, largest score", r"^path attribution \(IG\), top-n by max"),
        ("IG, mean $|$score$|$", r"^path attribution \(IG\), top-n by mean"),
        ("IG, aggregate", r"^path attribution \(IG\), top-n by the aggregate"),
    ]),
    ("Ranked members at the weighted circuit's per-site counts: $\\alpha = 1$ / fitted", [
        ("IG-ranked, $\\alpha = 1$", r"^IG-ranked members .*α = 1"),
        ("IG-ranked, fitted", r"^IG-ranked members .*fitted"),
        ("activation-ranked, $\\alpha = 1$", r"^activation-ranked members \(mean anchor activation\).*α = 1"),
        ("activation-ranked, fitted", r"^activation-ranked members \(mean anchor activation\).*fitted"),
        ("co-activation-ranked, $\\alpha = 1$", r"^activation-ranked members \(co-activation.*α = 1"),
        ("co-activation-ranked, fitted", r"^activation-ranked members \(co-activation.*fitted"),
    ]),
    ("Ranked by exact effect (subset of targets)", [
        ("exact-patching-ranked, $\\alpha = 1$", r"^exact-patching-ranked members .*α = 1"),
        ("exact-patching-ranked, fitted", r"^exact-patching-ranked members .*fitted"),
    ]),
]
NCOL = 9


def frac(cell):
    """'k/n' -> 100 k / n; n = 0 or unparsable -> nan."""
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(cell))
    return 100 * int(m.group(1)) / int(m.group(2)) if m and int(m.group(2)) else np.nan


def collect(t):
    rows, used = [], set()
    for section, items in LAYOUT:
        for label, rx in items:
            hit = t[t.row.str.contains(rx, regex=True)]
            if len(hit) > 1:
                raise SystemExit("row %r matches %d 083 rows: %s" % (label, len(hit), hit.row.tolist()))
            if hit.empty:
                rows.append(dict(section=section, label=label, source=None))
                continue
            r = hit.iloc[0]
            used.add(r.row)
            rows.append(dict(section=section, label=label, source=r.row, n=r.n_median, Z=r.Z, A=r.A, C=r.C,
                             necessity=r.necessity, **{"pass_%s" % b: frac(r["pass_%s" % b]) for b in BANDS}))
    return pd.DataFrame(rows), sorted(set(t.row) - used)


def ranks(values, key):
    """Rank-1 and rank-2 masks (ties share a rank) of a column, by key (lower = better); nan never ranked."""
    v = np.array([key(x) if not K.missing(x) else np.nan for x in values], dtype=float)
    uniq = sorted(set(v[~np.isnan(v)]))
    best = v == uniq[0] if uniq else np.zeros(len(v), bool)
    second = v == uniq[1] if len(uniq) > 1 else np.zeros(len(v), bool)
    return best, second


def mark(text, b, s):
    return "\\textbf{%s}" % text if b else ("\\underline{%s}" % text if s else text)


def body(d):
    # ranked at the printed precision (2 dp scores, whole-percent pass rates), so equal-looking cells share a mark;
    # the exact-effect block covers a subset of targets and is not ranked against the rest
    ranked = ~d.section.str.startswith("Ranked by exact")
    cols = {}
    for c in ["Z", "A", "C", "necessity"] + ["pass_%s" % b for b in BANDS]:
        p = 0 if c.startswith("pass_") else 2
        vals = [round(x, p) if (r and not K.missing(x)) else np.nan for x, r in zip(d.get(c, [np.nan] * len(d)), ranked)]
        cols[c] = ranks(vals, (lambda x: abs(x - 1)) if c in "ZAC" else (lambda x: -x))
    lines, prev = [], None
    for i, r in enumerate(d.to_dict("records")):
        if r["section"] != prev:
            if prev is not None:
                lines.append("\\midrule")
            lines.append("\\multicolumn{%d}{@{}l}{\\emph{%s}} \\\\" % (NCOL, r["section"]))
            prev = r["section"]
        if r.get("source") is None:
            lines.append("%s %s \\\\" % (r["label"], "& %s " % K.DASH * (NCOL - 1)))
            continue
        cells = [K.tex_int(r["n"])]
        for c in ["Z", "A", "C", "necessity"]:
            cells.append(mark(K.tex_num(r[c]), cols[c][0][i], cols[c][1][i]) if not K.missing(r[c]) else K.DASH)
        for b in BANDS:
            c = "pass_%s" % b
            cells.append(mark("%.0f" % r[c], cols[c][0][i], cols[c][1][i]) if not K.missing(r[c]) else K.DASH)
        lines.append("%s & %s \\\\" % (r["label"], " & ".join(cells)))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="083 table2.py's table2.csv")
    ap.add_argument("--tag", default="turingllm-v2")
    args = ap.parse_args()
    src = Path(args.csv)
    if not src.is_absolute():
        src = K.ROOT / src
    t = pd.read_csv(src)
    d, unlisted = collect(t)
    d = d[d.source.notna() | ~d.section.str.startswith("Ranked by exact")]       # the exact block only if it ran
    d = d.reset_index(drop=True)
    dest = K.HERE / "out" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    tex = ["%% generated by paper/tables/matrix_body.py from %s; MIB marking: bold best, underline second" % src.name]
    tex += body(d)
    (dest / "matrix_body.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    d.to_csv(dest / "matrix_rows.csv", index=False)
    print("\n".join(tex))
    missing = d[d.source.isna()].label.tolist()
    if missing:
        print("\nWARNING: paper rows with no 083 row (dashes): %s" % missing)
    if unlisted:
        print("\nNOTE: 083 rows not in the paper table: %s" % unlisted)
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()
