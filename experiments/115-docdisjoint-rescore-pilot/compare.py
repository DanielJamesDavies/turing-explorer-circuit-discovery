"""115 report: sanity check (local re-score on the ORIGINAL held-out sets vs the production rows) and the pilot
comparison (document- / topic-disjoint held-out sets vs the original). CPU only.

Pass rule and vacuous rule: 062/merge.py through 109's lib109 (imported). The paired comparison is new set vs the
LOCAL re-score of the original set (same GPU, same code), so hardware float noise does not enter the deltas; the
production rows are the reference for the sanity check only.

  python -X utf8 experiments/115-docdisjoint-rescore-pilot/compare.py
Writes out/sanity.csv, out/compare.csv, out/report.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402

OUT = HERE / "out"
COLS = L.HEAD + [L.NEC]
SHORT = {"free0_tk": "Z", "freeM_topk_tk": "A", "freeN_topk_tk": "C", L.NEC: "nec"}
SANITY_TOL = 0.06     # 086's measured H100-vs-RTX noise on main Z / A / C (086 README); exact is expected for most


def scored(dirs, keys):
    df = L.frame(L.read_rows(dirs, "eval"))
    if df.empty:
        return df
    df = df[(df.held == "strong") & df.seed.isin(keys)].copy()
    df["vacuous"] = L.vacuous(df).values
    df["passes"] = (L.passes(df) & ~df["vacuous"]).values
    return L.add_site(df, "turingllm").set_index("seed").sort_index()


def build_rows(group):
    p = OUT / ("build_%s.jsonl" % group)
    if not p.exists():
        return None
    return pd.DataFrame([json.loads(x) for x in p.read_text().splitlines() if x]).set_index("seed")


def pct(a, b):
    return "%d/%d = %.0f%%" % (a, b, 100.0 * a / b) if b else "-"


def main():
    keys = [k for k in (HERE / "targets_pilot.txt").read_text().split() if k]
    prod = scored([d / "main" for d in L.find("turingllm", "out")], keys)
    loc = {v: scored([OUT / ("rs_%s" % v) / "main"], keys) for v in ("orig", "doc", "topic")}
    lines = ["# 115 pilot: re-score of fixed production circuits on document-disjoint held-out sets", ""]

    # ---------------------------------------------------------------- sanity
    o = loc["orig"]
    s = pd.DataFrame(index=prod.index)
    for c in COLS + ["a_pos_tk"]:
        s["prod_" + SHORT.get(c, c)] = prod[c]
        s["loc_" + SHORT.get(c, c)] = o[c].reindex(prod.index)
    d = (o[COLS].reindex(prod.index).astype(float) - prod[COLS].astype(float)).abs()
    s["max_abs_diff"] = d.max(axis=1)
    s["prod_pass"], s["loc_pass"] = prod.passes, o.passes.reindex(prod.index)
    s.to_csv(OUT / "sanity.csv")
    n_exact = int((s.max_abs_diff == 0).sum())
    agree = int((s.prod_pass == s.loc_pass).sum())
    lines += ["## Sanity: local re-score on the ORIGINAL held-out sets vs production (T3 rows)", "",
              "- targets re-scored: %d of %d" % (o.index.isin(keys).sum(), len(keys)),
              "- exactly equal on Z / A / C / necessity: %d; max |diff| over all four: median %.2g, max %.2g"
              % (n_exact, s.max_abs_diff.median(), s.max_abs_diff.max()),
              "- pass decision agrees: %d / %d" % (agree, len(s)),
              "- a_pos_tk max |diff|: %.3g" % (o.a_pos_tk.reindex(prod.index) - prod.a_pos_tk).abs().max(), ""]
    sanity_ok = bool(s.max_abs_diff.max() <= SANITY_TOL and agree == len(s))
    lines += ["Sanity verdict: **%s** (tolerance %.2f, 086's measured H100-vs-RTX noise)" %
              ("PASS" if sanity_ok else "FAIL", SANITY_TOL), ""]
    worst = s.sort_values("max_abs_diff", ascending=False).head(5)
    lines += ["| target | max abs diff | prod Z/A/C/nec | local Z/A/C/nec |", "|---|---|---|---|"]
    for k, r in worst.iterrows():
        lines.append("| %s | %.3g | %s | %s |" % (k, r.max_abs_diff,
                     " / ".join("%.3f" % r["prod_" + SHORT[c]] for c in COLS),
                     " / ".join("%.3f" % r["loc_" + SHORT[c]] for c in COLS)))
    lines.append("")

    # ---------------------------------------------------------------- comparison
    rows = []
    for v in ("doc", "topic"):
        n = loc[v]
        b = build_rows(v)
        if n is None or n.empty or b is None:
            lines += ["## %s-disjoint: not yet scored" % v, ""]
            continue
        cmp_ = pd.DataFrame(index=o.index)
        cmp_["variant"] = v
        cmp_["band"], cmp_["kind"] = o.band, o.kind
        for c in COLS:
            cmp_["old_" + SHORT[c]] = o[c]
            cmp_["new_" + SHORT[c]] = n[c].reindex(o.index)
            cmp_["d_" + SHORT[c]] = cmp_["new_" + SHORT[c]] - cmp_["old_" + SHORT[c]]
        cmp_["old_vac"], cmp_["new_vac"] = o.vacuous, n.vacuous.reindex(o.index)
        cmp_["old_pass"], cmp_["new_pass"] = o.passes, n.passes.reindex(o.index).fillna(False).astype(bool)
        cmp_["prod_pass"] = prod.passes.reindex(o.index)
        for c in COLS:
            cmp_["prod_" + SHORT[c]] = prod[c].reindex(o.index)
        for f in ("new_share_contrast", "old_share_contrast", "n_want", "n_held", "below_floor", "n_old_leaky", "n_orig_kept", "overlap_old", "n_scan",
                  "cand_total", "cand_allowed", "peak_train_med", "peak_old_med", "peak_new_med", "scan_rank_med",
                  "scan_rank_max"):
            cmp_[f] = b[f].reindex(o.index)
        rows.append(cmp_)
        nn = len(cmp_)
        p_old, p_new = int(cmp_.old_pass.sum()), int(cmp_.new_pass.sum())
        f_down = int((cmp_.old_pass & ~cmp_.new_pass).sum())
        f_up = int((~cmp_.old_pass & cmp_.new_pass).sum())
        lines += ["## %s-disjoint held-out strongest vs the original (both re-scored here)" % v.capitalize(), "",
                  "**Held-out sets.**",
                  "- candidates allowed by the group clause (orig held + scan, before firing / duplicate checks): median "
                  "%d of %d (min %d)" % (cmp_.cand_allowed.median(), cmp_.cand_total.median(), cmp_.cand_allowed.min()),
                  "- old held-out rows in a forbidden %s group: median %d of 16 (total %d of %d = %.0f%%)"
                  % (v, cmp_.n_old_leaky.median(), cmp_.n_old_leaky.sum(), cmp_.n_want.sum(),
                     100 * cmp_.n_old_leaky.sum() / cmp_.n_want.sum()),
                  "- new sets filled to the original size: %d / %d; too few candidates (short of the original size): "
                  "%d; below floor 8: %d"
                  % (int((cmp_.n_held == cmp_.n_want).sum()), nn, int((cmp_.n_held < cmp_.n_want).sum()),
                     int(cmp_.below_floor.sum())),
                  "- rows sharing a %s group with the target's contrast-TRAINING rows (not excluded by the rule): new "
                  "sets %d of %d rows (median %d per target, %d targets with >= 1); old sets %d of %d"
                  % (v, cmp_.new_share_contrast.sum(), cmp_.n_held.sum(), cmp_.new_share_contrast.median(),
                     int((cmp_.new_share_contrast > 0).sum()), cmp_.old_share_contrast.sum(), cmp_.n_want.sum()),
                  "- overlap with the old set: median %d of 16 rows (the rest from the scan, median scan rank %d, max %d)"
                  % (cmp_.overlap_old.median(), cmp_.scan_rank_med.median(), cmp_.scan_rank_max.max()),
                  "- strength (median peak; new / old, per-target median ratio): %.3f; new / strongest-train: %.3f"
                  % ((cmp_.peak_new_med / cmp_.peak_old_med).median(), (cmp_.peak_new_med / cmp_.peak_train_med).median()),
                  "", "**Pass (non-vacuous and Z, A, C in [0.75, 1.5], necessity >= 0.9).** The sample is fail-enriched "
                  "by design (27 pass / 9 fail in production), so the flip rates conditional on the old verdict are the "
                  "quantities to read, not a pooled pass rate.",
                  "- of the targets passing on the old set (local re-score): %s still pass, so pass -> fail %s"
                  % (pct(int((cmp_.old_pass & cmp_.new_pass).sum()), p_old), pct(f_down, p_old)),
                  "- of the targets failing on the old set: fail -> pass %s" % pct(f_up, nn - p_old),
                  "- (pass counts, for reference: old %d / %d, new %d / %d; production rows %d / %d)"
                  % (p_old, nn, p_new, nn, int(cmp_.prod_pass.sum()), nn),
                  "- vacuous on the new set: %d (old %d)" % (int(cmp_.new_vac.sum()), int(cmp_.old_vac.sum())), "",
                  "| band | n | pass old | pass new | pass->fail | fail->pass | median dZ | dA | dC | dnec |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for band, g in list(cmp_.groupby("band")) + [("all", cmp_)]:
            lines.append("| %s | %d | %d | %d | %d | %d | %+.3f | %+.3f | %+.3f | %+.3f |" % (
                band, len(g), g.old_pass.sum(), g.new_pass.sum(), (g.old_pass & ~g.new_pass).sum(),
                (~g.old_pass & g.new_pass).sum(), g.d_Z.median(), g.d_A.median(), g.d_C.median(), g.d_nec.median()))
        lines += ["", "| kind | n | pass old | pass new | pass->fail | fail->pass |", "|---|---|---|---|---|---|"]
        for kind, g in cmp_.groupby("kind"):
            lines.append("| %s | %d | %d | %d | %d | %d |" % (kind, len(g), g.old_pass.sum(), g.new_pass.sum(),
                                                             (g.old_pass & ~g.new_pass).sum(),
                                                             (~g.old_pass & g.new_pass).sum()))
        lines += ["", "Absolute deltas (new - old): |dZ| median %.3f p90 %.3f; |dA| %.3f / %.3f; |dC| %.3f / %.3f; "
                  "|dnec| %.3f / %.3f" % tuple(x for c in "ZACn" for x in (
                      cmp_["d_" + ("nec" if c == "n" else c)].abs().median(),
                      cmp_["d_" + ("nec" if c == "n" else c)].abs().quantile(0.9))), "",
                  "| target | band | prod Z/A/C/nec (ref) | old Z/A/C/nec | new Z/A/C/nec | prod / old -> new | "
                  "kept / scan | old leaky | new share contrast |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for k, r in cmp_.sort_values(["band", "kind"]).iterrows():
            lines.append("| %s | %s | %s | %s | %s | %s / %s -> %s | %d / %d | %d | %s |" % (
                k, r.band, " / ".join("%.2f" % r["prod_" + SHORT[c]] for c in COLS),
                " / ".join("%.2f" % r["old_" + SHORT[c]] for c in COLS),
                " / ".join("%.2f" % r["new_" + SHORT[c]] for c in COLS),
                "P" if r.prod_pass else "F", "P" if r.old_pass else "F", "P" if r.new_pass else "F",
                r.overlap_old, r.n_scan, r.n_old_leaky, r.new_share_contrast))
        lines.append("")
    if rows:
        pd.concat(rows).to_csv(OUT / "compare.csv")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
