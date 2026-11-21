"""124 REPORT (CPU): the fixed production circuits on the subset, scored on the original held-out sets (orig) and on
the paraphrase-filtered sets (para), both on the same local GPU (DAN-217).

Statistics are 123 report.py's, imported (which takes the pass rule, vacuous rule and bootstrap from 062 merge.py via
109 lib109: 95% percentile bootstrap over targets, B = 4000, seed 20261108):
  paired        targets non-vacuous on both orig and para: pass rates with intervals and n, the paired difference with
                a paired bootstrap interval, transitions pass->fail and fail->pass
  conservative  orig's non-vacuous targets; para vacuous / unscorable / missing counts as a fail
  cells         band x site kind, with band and kind margins, transitions per cell
  mid-band      the same, reported (not in the pass rule)
Plus the rebuild counts (rows dropped, rows refilled, short / below floor / unscorable), the vacuous counts, the
production rows of the subset (reference only: other hardware) and the public-model check (public_check.py).

  python -X utf8 experiments/124-paraphrase-filter-subset/report124.py
  env: BUILD (build_para | build_para_cpu: which build log to summarise; default the GPU one if present)
Writes out/report.md, out/report.json, out/report_cells.csv, out/report_pairs.csv.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import para124 as P  # noqa: E402

C = P.C
sys.path.insert(0, str(C.EXP / "123-docgroup-heldout-rescore"))
sys.path.insert(0, str(C.EXP / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402
import report as R  # noqa: E402  (123 report: paired, cells, rate, fmt, scored)

MODEL = "turingllm"


def variant_rows(v, held="strong"):
    return R.scored(L.read_rows([P.WORK / ("rs_%s" % v) / "main"], "eval"), MODEL, held)


def jsonl_frame(name):
    rows = [r for r in C.read_jsonl(P.WORK / name) if "error" not in r and not r.get("skip")]
    df = pd.DataFrame(rows)
    return df.drop_duplicates("seed", keep="last").set_index("seed") if len(df) else df


def add_band(df):
    df = df.copy()
    df["layer"] = [int(k.split(".")[0]) for k in df.index]
    df["kind"] = [k.split(".")[1] for k in df.index]
    df["band"] = [L.band_of(MODEL, x) for x in df.layer]
    return df


def share(a, b):
    return "%d / %d (%.1f%%)" % (a, b, 100.0 * a / b) if b else "-"


def plan_summary(p):
    out = dict(targets=int(len(p)), old_rows=int(p.n_want.sum()), old_rows_dropped=int(p.n_old_leaky.sum()),
               targets_with_drop=int((p.n_old_leaky > 0).sum()), targets_all_dropped=int((p.n_old_leaky == p.n_want).sum()),
               cand_allowed_median=float(p.cand_allowed.median()), short=int(p.short_certain.sum()),
               below_floor=int(p.below_floor_certain.sum()), unscorable=int(p.unscorable_certain.sum()),
               mid_rows=int(p.mid_held.fillna(0).sum()), mid_dropped=int(p.mid_leaky.fillna(0).sum()),
               contrast_rows=int(p.contrast_held.sum()),
               contrast_para_vs_contrast_train=int(p.contrast_para_vs_contrast_train.sum()),
               contrast_para_vs_activating_train=int(p.contrast_para_vs_activating_train.sum()))
    out["old_rows_dropped_pct"] = 100.0 * out["old_rows_dropped"] / max(1, out["old_rows"])
    return out


def build_summary(b):
    out = dict(targets=int(len(b)), device=str(b.device.iloc[0]) if "device" in b else "?",
               old_rows=int(b.n_want.sum()), old_rows_dropped=int(b.n_old_leaky.sum()),
               targets_with_drop=int((b.n_old_leaky > 0).sum()),
               orig_rows_kept=int(b.n_orig_kept.sum()), rows_refilled=int(b.n_scan.sum()), rows_new_total=int(b.n_held.sum()),
               targets_refilled=int((b.n_scan > 0).sum()), short=int((b.n_held < b.n_want).sum()),
               below_floor=int(b.below_floor.sum()), unscorable=int((b.n_held == 0).sum()),
               unscorable_keys=sorted(b.index[b.n_held == 0]), median_kept=float(b.n_orig_kept.median()),
               median_refilled=float(b.n_scan.median()), unstable=int(b.unstable.sum()) if "unstable" in b else None,
               scan_rank_median=float(b.scan_rank_med.dropna().median()) if b.scan_rank_med.notna().any() else None,
               strength_new_over_old=float((b.peak_new_med / b.peak_old_med).median()),
               mid_rows=int(b.mid_held_old.fillna(0).sum()) if "mid_held_old" in b else None,
               mid_rows_after=int(b.mid_held_new.fillna(0).sum()) if "mid_held_new" in b else None,
               mid_all_dropped_kept=int(b.mid_all_leaky_kept.fillna(False).astype(bool).sum())
               if "mid_all_leaky_kept" in b else 0)
    out["old_rows_dropped_pct"] = 100.0 * out["old_rows_dropped"] / max(1, out["old_rows"])
    out["old_rows_kept_but_rejected_086"] = out["old_rows"] - out["old_rows_dropped"] - out["orig_rows_kept"]
    return out


def by_cell(df, num, den):
    d = add_band(df)
    g = d.groupby(["band", "kind"])[[num, den]].sum()
    return {"%s/%s" % k: share(int(r[num]), int(r[den])) for k, r in g.iterrows()}


def main():
    lines = ["# 124: paraphrase-filtered held-out re-score of the fixed production circuits, TuringLLM subset (DAN-217)",
             "", "Rule: drop a held-out row whose 114 TF-IDF cosine to any of the target's activating training rows is "
             ">= %.2f, on top of the DAN-157 rule; refill from the T1 scan. Pass = non-vacuous and Z, A, C in "
             "[%.2f, %.2f], necessity >= %.1f (062 merge.py). Intervals: 95%% bootstrap over targets, B = %d."
             % (P.THRESH, L.BAND[0], L.BAND[1], L.NEC_MIN, L.BOOT_B), ""]
    js = {}
    sub = P.WORK / "subset.json"
    if sub.exists():
        s = json.loads(sub.read_text())
        js["subset"] = s
        lines.append("- subset: %d targets (seed %d, %s); population %d non-vacuous production targets, production pass "
                     "%.1f%% (sample %.1f%%)" % (s["n"], s["seed"], s["rule"], s["population"],
                                                 s["population_pass_pct"], s["sample_pass_pct"]))
    iv = P.WORK / "idf_verify.json"
    if iv.exists():
        v = json.loads(iv.read_text())
        js["idf_verify"] = v
        lines.append("- vectoriser = 114's: %d held rows of %d 114 records recomputed, max |cos diff| %.2g, threshold "
                     "calls differing %d" % (v["rows"], v["records"], v["max_abs_diff"], v["call_mismatches"]))
    p = jsonl_frame("plan_para.jsonl")
    if len(p):
        ps = plan_summary(p)
        js["plan"] = ps
        lines += ["", "## Paraphrase test alone (CPU plan, before the firing / duplicate rule)", "",
                  "- old held-out strongest rows that are paraphrases (dropped): %s; targets with >= 1: %d / %d; all "
                  "rows dropped: %d" % (share(ps["old_rows_dropped"], ps["old_rows"]), ps["targets_with_drop"],
                                        ps["targets"], ps["targets_all_dropped"]),
                  "- allowed candidates (orig held + scan), median %g; certainly short %d, certainly below floor 8 %d, "
                  "certainly unscorable %d" % (ps["cand_allowed_median"], ps["short"], ps["below_floor"],
                                               ps["unscorable"]),
                  "- mid-band held rows dropped (drop only): %s" % share(ps["mid_dropped"], ps["mid_rows"]),
                  "- contrast held rows (measured, not rebuilt) that paraphrase a contrast-training row: %s; an "
                  "activating training row: %s" % (share(ps["contrast_para_vs_contrast_train"], ps["contrast_rows"]),
                                                   share(ps["contrast_para_vs_activating_train"], ps["contrast_rows"])),
                  "- dropped share by band / kind: %s" % by_cell(p, "n_old_leaky", "n_want")]
    bname = os.environ.get("BUILD") or ("build_para.jsonl" if (P.WORK / "build_para.jsonl").exists()
                                        else "build_para_cpu.jsonl")
    b = jsonl_frame(bname)
    if len(b):
        bs = build_summary(b)
        js["build"] = dict(bs, file=bname)
        lines += ["", "## Rebuild (%s, forward on %s)" % (bname, bs["device"]), "",
                  "- old held rows dropped as paraphrases %s (%d targets with >= 1); old rows kept %d; old rows "
                  "rejected by the 086 rule / stability re-check %d" % (
                      share(bs["old_rows_dropped"], bs["old_rows"]), bs["targets_with_drop"], bs["orig_rows_kept"],
                      bs["old_rows_kept_but_rejected_086"]),
                  "- rows refilled from the T1 scan %d (%d targets; median %g per target; median scan rank %s); "
                  "median peak new / old %.3f" % (bs["rows_refilled"], bs["targets_refilled"], bs["median_refilled"],
                                                  bs["scan_rank_median"], bs["strength_new_over_old"]),
                  "- short of the original size %d; below floor 8 %d; unscorable (0 rows) %d %s" % (
                      bs["short"], bs["below_floor"], bs["unscorable"], bs["unscorable_keys"] or ""),
                  "- mid-band held rows %s -> %s (all-paraphrase sets kept and flagged: %d)" % (
                      bs["mid_rows"], bs["mid_rows_after"], bs["mid_all_dropped_kept"])]
    keys = C.read_targets(P.subset_path()) if P.subset_path().exists() else []
    prod = R.production_rows(MODEL)
    prod = prod[prod.index.isin(keys)]
    pc = R.rate(prod[~prod.vac.astype(bool)]["pass"]) if len(prod) else None
    js["production_subset"] = pc
    lines += ["", "## Scores", "", "- production rows of the subset (H100; reference only): %s" % R.fmt(pc)]
    orig = variant_rows("orig")
    if len(orig) == 0:
        lines += ["", "Not scored yet (no rows in %s)." % (P.WORK / "rs_orig" / "main")]
    else:
        oc = R.rate(orig[~orig.vac.astype(bool)]["pass"])
        js["orig"] = oc
        both = prod.index.intersection(orig.index)
        agree = float((prod.loc[both, "pass"] == orig.loc[both, "pass"]).mean()) if len(both) else float("nan")
        lines += ["- orig re-score (local GPU): %s; vacuous %d; pass decision agrees with production on %.1f%% of %d"
                  % (R.fmt(oc), int(orig.vac.sum()), 100 * agree, len(both))]
        new = variant_rows("para")
        if len(new) == 0:
            lines += ["", "para: not scored yet."]
        else:
            uns = set(json.loads((P.WORK / "prep_para.json").read_text())["excluded_keys"]) \
                if (P.WORK / "prep_para.json").exists() else set()
            pr, s = R.paired(orig, new, uns)
            js["para"] = dict(own=R.rate(new[~new.vac.astype(bool)]["pass"]), paired=s, vacuous=int(new.vac.sum()))
            lines += ["- para re-score: own population %s; vacuous %d" % (R.fmt(js["para"]["own"]), int(new.vac.sum())),
                      "", "**Paired** (%d targets non-vacuous on both): orig %s -> para %s; difference %+.1f pp "
                      "[%+.1f, %+.1f]; pass->fail %d, fail->pass %d" % (
                          s["n_pair"], R.fmt(s["old"]), R.fmt(s["new"]), s["diff_pp"], s["diff_lo"], s["diff_hi"],
                          s["pass_to_fail"], s["fail_to_pass"]), "",
                      "Conservative (orig's %d non-vacuous; para vacuous %d / unscorable %d / missing %d count as "
                      "fail): %s -> %s; difference %+.1f pp [%+.1f, %+.1f]" % (
                          s["n_old"], s["new_vacuous"], s["unscorable"], s["new_missing"], R.fmt(s["cons_old"]),
                          R.fmt(s["cons_new"]), s["cons_diff_pp"], s["cons_diff_lo"], s["cons_diff_hi"]), "",
                      "| band | kind | n | orig pass | para pass | diff pp | pass->fail | fail->pass |",
                      "|---|---|---|---|---|---|---|---|"]
            ce = R.cells(pr, MODEL)
            for _, r in ce.iterrows():
                lines.append("| %s | %s | %d | %.1f [%.1f, %.1f] | %.1f [%.1f, %.1f] | %+.1f [%+.1f, %+.1f] | %d | %d |"
                             % (r.band, r.kind, r.n_pair, r.old_pct, r.old_lo, r.old_hi, r.new_pct, r.new_lo, r.new_hi,
                                r.diff_pp, r.diff_lo, r.diff_hi, r.pass_to_fail, r.fail_to_pass))
            ce.to_csv(P.WORK / "report_cells.csv", index=False)
            pr.reset_index().to_csv(P.WORK / "report_pairs.csv", index=False)
            flips = pr[pr.in_pair & (pr.old_pass != pr.new_pass)]
            if len(flips):
                lines += ["", "Flips: " + ", ".join("%s (%s)" % (k, "p->f" if r.old_pass else "f->p")
                                                    for k, r in flips.iterrows())]
            om, nm = variant_rows("orig", "mid"), variant_rows("para", "mid")
            if len(om) and len(nm):
                _, sm = R.paired(om, nm)
                js["para"]["mid_paired"] = sm
                lines += ["", "Mid-band (reported): paired %d, orig %s -> para %s; difference %+.1f pp [%+.1f, %+.1f]; "
                          "p->f %d, f->p %d" % (sm["n_pair"], R.fmt(sm["old"]), R.fmt(sm["new"]), sm["diff_pp"],
                                                sm["diff_lo"], sm["diff_hi"], sm["pass_to_fail"], sm["fail_to_pass"])]
    pub = P.WORK / "public_check.json"
    if pub.exists():
        pj = json.loads(pub.read_text())
        js["public"] = pj
        lines += ["", "## Public models: the same TF-IDF test, measured only (public_check.py)", ""]
        for m, r in pj["models"].items():
            lines.append("- %s: held-out strongest rows with cos >= %.2f to a training row of the same target %s "
                         "(%d / %d targets with >= 1; best match in the same source document for %s of the hits); "
                         "against another target's training rows %s; mid-band %s"
                         % (m, pj["thresh"], share(r["strong_hit"], r["strong_rows"]), r["targets_with_hit"],
                            r["targets"], share(r.get("strong_hit_same_doc", 0), r["strong_hit"]),
                            share(r["strong_hit_other"], r["strong_rows"]), share(r["mid_hit"], r["mid_rows"])))
        if pj.get("tl_control"):
            t = pj["tl_control"]
            lines.append("- control, TuringLLM subset with the same records-only IDF: %s (corpus IDF: %s)"
                         % (share(t["hit_records_idf"], t["rows"]), share(t["hit_corpus_idf"], t["rows"])))
    P.write_json(js, P.WORK / "report.json")
    (P.WORK / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
