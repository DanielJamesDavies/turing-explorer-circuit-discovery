"""123 REPORT (CPU): pass rates of the fixed production circuits on the document-group-disjoint held-out sets, paired
against the same circuits re-scored on the current (production) held-out sets on the same hardware (DAN-217).

Pass rule and vacuous rule: 062/merge.py through 109's lib109 (imported, not re-implemented). Intervals: 95%
percentile bootstrap over targets (lib109 boot_ci / boot_paired_diff: B = BOOT_B 4000, seed 20261108), with n.

Populations (held-out strongest, the pass rule's set; mid-band reported the same way):
  per variant    every scored row of the variant minus its vacuous rows (the paper's headline definition)
  paired         targets with a non-vacuous row on BOTH orig (the re-score of the production records) and the new
                 variant; diff = mean(new pass) - mean(orig pass) with a paired bootstrap interval; flips both ways
  conservative   orig's non-vacuous targets; a target whose new set is empty (unscorable) or vacuous counts as a FAIL
  production     the archived production rows (109 archives), the numbers the paper quotes (88.0 / 81.9 / 82.4%):
                 recomputed here, and compared with the local orig re-score (reproduction check)
Cells: depth band x site kind (lib109 bands per family), plus band and kind margins.

  python -X utf8 experiments/123-docgroup-heldout-rescore/report.py
  env: MODELS  WORK123 (out/)  BOOT_B
Writes out/report.md, out/report_cells.csv, out/report_pairs.csv, out/report.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "109-pod-run-analysis"))
import common123 as C  # noqa: E402
import lib109 as L  # noqa: E402

MODELS = [m for m in os.environ.get("MODELS", ",".join(C.MODELS)).split(",") if m]
PAPER = {"turingllm": 88.0, "pythia-70m": 81.9, "gpt2-small": 82.4}     # the headline the paper quotes (main.tex:380)


# --------------------------------------------------------------------------------------------- loading
def scored(rows, model, held="strong"):
    """Scored eval rows -> one row per seed with band, kind, vac, pass (lib109 rules)."""
    df = L.frame(rows, keys=("seed", "held"))
    if df.empty:
        return pd.DataFrame(columns=["seed", "band", "kind", "vac", "pass"]).set_index("seed")
    df = df[df.held == held]
    if df.empty:
        return pd.DataFrame(columns=["seed", "band", "kind", "vac", "pass"]).set_index("seed")
    return L.score(L.add_site(df, model)).set_index("seed")


def variant_rows(model, variant, held="strong"):
    return scored(L.read_rows([C.variant_out(model, variant) / "main"], "eval"), model, held)


def production_rows(model, held="strong"):
    return scored(L.read_rows(L.find(model, "out", "main"), "eval"), model, held)


def build_rows(model, group):
    rows = C.read_jsonl(C.WORK / model / ("build_%s.jsonl" % group))
    df = pd.DataFrame([r for r in rows if "error" not in r and not r.get("skip")])
    return df.drop_duplicates("seed", keep="last").set_index("seed") if not df.empty else df


def plan_rows(model, group):
    rows = C.read_jsonl(C.WORK / model / ("plan_%s.jsonl" % group))
    df = pd.DataFrame([r for r in rows if "error" not in r and not r.get("skip")])
    return df.drop_duplicates("seed", keep="last").set_index("seed") if not df.empty else df


# --------------------------------------------------------------------------------------------- statistics (pure)
def rate(p):
    """{k, n, pct, lo, hi} over a boolean series (lib109 pass_cell)."""
    return L.pass_cell(pd.Series(p, dtype=float))


def fmt(c):
    if not c or c["n"] == 0:
        return "-"
    return "%.1f%% [%.1f, %.1f] (%d/%d)" % (c["pct"], c["lo"], c["hi"], c["k"], c["n"])


def paired(old, new, unscorable=()):
    """Paired comparison of two scored frames (index seed; columns vac, pass, band, kind).
    Returns (pairs frame, summary dict). pairs: seeds non-vacuous on old; new_pass_cons = new pass where scored and
    non-vacuous, else False (conservative); in_pair = non-vacuous on both."""
    o = old[~old.vac.astype(bool)]
    pr = pd.DataFrame(index=o.index)
    pr["band"], pr["kind"], pr["old_pass"] = o.band, o.kind, o["pass"].astype(bool)
    nv = new.reindex(o.index)
    pr["new_scored"] = nv["pass"].notna()
    pr["new_vac"] = nv.vac.fillna(False).astype(bool)
    pr["new_pass"] = nv["pass"].fillna(False).astype(bool) & ~pr.new_vac
    pr["unscorable"] = pr.index.isin(list(unscorable))
    pr["in_pair"] = pr.new_scored & ~pr.new_vac
    pr["new_pass_cons"] = pr.new_pass & pr.in_pair
    return pr, summarise_pairs(pr)


def summarise_pairs(pr):
    q = pr[pr.in_pair]
    a, b = q.new_pass.astype(float).values, q.old_pass.astype(float).values
    lo, hi = L.boot_paired_diff(a, b) if len(q) else (float("nan"), float("nan"))
    cons_lo, cons_hi = (L.boot_paired_diff(pr.new_pass_cons.astype(float).values, pr.old_pass.astype(float).values)
                        if len(pr) else (float("nan"), float("nan")))
    return dict(n_old=int(len(pr)), n_pair=int(len(q)), old=rate(q.old_pass), new=rate(q.new_pass),
                diff_pp=100 * (a.mean() - b.mean()) if len(q) else float("nan"), diff_lo=100 * lo, diff_hi=100 * hi,
                pass_to_fail=int((q.old_pass & ~q.new_pass).sum()), fail_to_pass=int((~q.old_pass & q.new_pass).sum()),
                new_vacuous=int(pr.new_vac.sum()), new_missing=int((~pr.new_scored).sum()),
                unscorable=int(pr.unscorable.sum()),
                cons_new=rate(pr.new_pass_cons), cons_old=rate(pr.old_pass),
                cons_diff_pp=100 * (pr.new_pass_cons.mean() - pr.old_pass.mean()) if len(pr) else float("nan"),
                cons_diff_lo=100 * cons_lo, cons_diff_hi=100 * cons_hi)


def cells(pr, model):
    """Paired pass rates by band x kind with the band and kind margins and 'all'."""
    out = []
    bands = [b for b, _, _ in L.bands_of(model)]
    kinds = [k for k in L.KIND_ORDER if k in set(pr.kind)]
    groups = [("all", "all", pr)] + [(b, "all", pr[pr.band == b]) for b in bands] + \
             [("all", k, pr[pr.kind == k]) for k in kinds] + \
             [(b, k, pr[(pr.band == b) & (pr.kind == k)]) for b in bands for k in kinds]
    for b, k, g in groups:
        if len(g) == 0:
            continue
        s = summarise_pairs(g)
        out.append(dict(model=model, band=b, kind=k, n_pair=s["n_pair"], old_pct=s["old"]["pct"],
                        old_lo=s["old"]["lo"], old_hi=s["old"]["hi"], new_pct=s["new"]["pct"], new_lo=s["new"]["lo"],
                        new_hi=s["new"]["hi"], diff_pp=s["diff_pp"], diff_lo=s["diff_lo"], diff_hi=s["diff_hi"],
                        pass_to_fail=s["pass_to_fail"], fail_to_pass=s["fail_to_pass"], unscorable=s["unscorable"]))
    return pd.DataFrame(out)


def unscorable_of(model, variant):
    p = C.WORK / model / ("prep_%s.json" % variant)
    if not p.exists():
        return set()
    ex = json.loads(p.read_text()).get("excluded_keys", {})
    return {k for k, r in ex.items() if r == "unscorable_empty_heldout"}


def _share(a, b):
    return "%d / %d (%.1f%%)" % (a, b, 100.0 * a / b) if b else "-"


def plan_summary(model, group):
    """The CPU plan (TuringLLM): what the group clause alone removes, before the firing / duplicate rule and the fill."""
    p = plan_rows(model, group)
    if p is None or len(p) == 0:
        return None
    out = dict(model=model, group=group, targets=int(len(p)), old_rows=int(p.n_want.sum()),
               old_rows_dropped=int(p.n_old_leaky.sum()), old_rows_dropped_pct=100.0 * p.n_old_leaky.sum() / p.n_want.sum(),
               targets_with_drop=int((p.n_old_leaky > 0).sum()), cand_allowed_median=float(p.cand_allowed.median()),
               short=int(p.short_certain.sum()), below_floor=int(p.below_floor_certain.sum()),
               unscorable=int(p.unscorable_certain.sum()),
               unscorable_keys=sorted(p.index[p.unscorable_certain.astype(bool)]))
    if "mid_held" in p:
        out["mid"] = _share(int(p.mid_leaky.fillna(0).sum()), int(p.mid_held.fillna(0).sum()))
    if "contrast_held" in p:
        n = int(p.contrast_held.fillna(0).sum())
        out["contrast_vs_contrast_train"] = _share(int(p.contrast_leaky_vs_contrast_train.fillna(0).sum()), n)
        out["contrast_vs_activating_train"] = _share(int(p.contrast_leaky_vs_activating_train.fillna(0).sum()), n)
    return out


def rebuild_summary(model, group):
    b = build_rows(model, group)
    if b is None or len(b) == 0:
        return None
    n_want = b.n_want.sum()
    out = dict(model=model, group=group, targets=int(len(b)), old_rows=int(n_want),
               old_rows_dropped=int(b.n_old_leaky.sum()), old_rows_dropped_pct=100.0 * b.n_old_leaky.sum() / n_want,
               targets_with_drop=int((b.n_old_leaky > 0).sum()), median_kept_orig=float(b.n_orig_kept.median()),
               median_new_rows=float(b.n_scan.median()), rows_replaced=int(b.n_scan.sum()),
               short=int((b.n_held < b.n_want).sum()), below_floor=int(b.below_floor.sum()),
               unscorable=int((b.n_held == 0).sum()))
    if "peak_new_med" in b and "peak_old_med" in b:
        out["strength_new_over_old"] = float((b.peak_new_med / b.peak_old_med).median())
    if "mid_held_old" in b:
        out.update(mid_rows=int(b.mid_held_old.fillna(0).sum()), mid_rows_after=int(b.mid_held_new.fillna(0).sum()))
    return out


# --------------------------------------------------------------------------------------------- report
def main():
    lines = ["# 123: document-group-disjoint held-out re-score of the fixed production circuits (DAN-217)", "",
             "Pass = non-vacuous and Z, A, C in [%.2f, %.2f], necessity >= %.1f (062 merge.py). Intervals: 95%% "
             "bootstrap over targets, B = %d." % (L.BAND[0], L.BAND[1], L.NEC_MIN, L.BOOT_B), ""]
    all_cells, all_pairs, js = [], [], {}
    for model in MODELS:
        lines += ["## %s" % L.LABEL[model], ""]
        prod = production_rows(model)
        pc = rate(prod[~prod.vac]["pass"]) if len(prod) else None
        lines.append("- production rows (archives; the paper quotes %.1f%%): %s" % (PAPER[model], fmt(pc)))
        js[model] = dict(production=pc, paper=PAPER[model])
        groups = C.GROUPS_OF[model]
        for g in groups:
            ps = plan_summary(model, g)
            if ps:
                js[model]["plan_%s" % g] = ps
                lines.append("- plan %s (group clause only, CPU): %d targets; old held-out rows in a forbidden group "
                             "%d / %d (%.1f%%; %d targets with >= 1); allowed candidates median %g; certainly short "
                             "%d, certainly below floor 8 %d, certainly unscorable (0 candidates) %d; mid-band held rows "
                             "leaky %s; contrast held rows sharing a group with contrast-train %s / with activating "
                             "train %s" % (g, ps["targets"], ps["old_rows_dropped"], ps["old_rows"],
                                           ps["old_rows_dropped_pct"], ps["targets_with_drop"], ps["cand_allowed_median"],
                                           ps["short"], ps["below_floor"], ps["unscorable"], ps.get("mid", "-"),
                                           ps.get("contrast_vs_contrast_train", "-"),
                                           ps.get("contrast_vs_activating_train", "-")))
            rs = rebuild_summary(model, g)
            if rs:
                js[model]["rebuild_%s" % g] = rs
                lines.append("- rebuild %s: %d targets; old held-out rows dropped %d / %d (%.1f%%; %d targets with >= 1); "
                             "median kept %g + new %g; short %d, below floor 8 %d, unscorable (0 rows) %d%s" % (
                                 g, rs["targets"], rs["old_rows_dropped"], rs["old_rows"], rs["old_rows_dropped_pct"],
                                 rs["targets_with_drop"], rs["median_kept_orig"], rs["median_new_rows"], rs["short"],
                                 rs["below_floor"], rs["unscorable"],
                                 "; median peak new/old %.3f" % rs["strength_new_over_old"]
                                 if "strength_new_over_old" in rs else ""))
        orig = variant_rows(model, "orig")
        if len(orig) == 0:
            lines += ["", "Not scored yet (no rows in %s)." % (C.variant_out(model, "orig") / "main"), ""]
            continue
        oc = rate(orig[~orig.vac]["pass"])
        lines.append("- orig re-score (production records, this hardware): %s" % fmt(oc))
        both = prod.index.intersection(orig.index)
        if len(both):
            agree = (prod.loc[both, "pass"] == orig.loc[both, "pass"]).mean()
            dz = (prod.loc[both, L.HEAD + [L.NEC]].astype(float) - orig.loc[both, L.HEAD + [L.NEC]].astype(float)).abs()
            lines.append("- reproduction: pass decision agrees on %.1f%% of %d targets; max |diff| over Z/A/C/nec median "
                         "%.3g, p90 %.3g" % (100 * agree, len(both), dz.max(axis=1).median(), dz.max(axis=1).quantile(.9)))
        js[model]["orig"] = oc
        lines.append("")
        for v in groups:
            new = variant_rows(model, v)
            if len(new) == 0:
                lines += ["### %s-disjoint: not scored yet" % v, ""]
                continue
            pr, s = paired(orig, new, unscorable_of(model, v))
            pr["model"], pr["variant"] = model, v
            all_pairs.append(pr.reset_index())
            nc = rate(new[~new.vac]["pass"])
            js[model][v] = dict(own=nc, paired=s)
            lines += ["### %s-disjoint held-out strongest" % v.capitalize(), "",
                      "- own population: %s" % fmt(nc),
                      "- paired (%d targets non-vacuous on both): orig %s -> new %s; diff %+.1f pp [%+.1f, %+.1f]; "
                      "pass->fail %d, fail->pass %d" % (s["n_pair"], fmt(s["old"]), fmt(s["new"]), s["diff_pp"],
                                                         s["diff_lo"], s["diff_hi"], s["pass_to_fail"],
                                                         s["fail_to_pass"]),
                      "- conservative (orig's %d non-vacuous; new vacuous %d / unscorable %d / missing %d count as "
                      "fail): %s -> %s; diff %+.1f pp [%+.1f, %+.1f]" % (
                          s["n_old"], s["new_vacuous"], s["unscorable"], s["new_missing"], fmt(s["cons_old"]),
                          fmt(s["cons_new"]), s["cons_diff_pp"], s["cons_diff_lo"], s["cons_diff_hi"]), ""]
            ce = cells(pr, model)
            ce["variant"] = v
            all_cells.append(ce)
            lines += ["| band | kind | n | orig pass | new pass | diff pp | p->f | f->p |", "|---|---|---|---|---|---|---|---|"]
            for _, r in ce.iterrows():
                lines.append("| %s | %s | %d | %.1f [%.1f, %.1f] | %.1f [%.1f, %.1f] | %+.1f [%+.1f, %+.1f] | %d | %d |" % (
                    r.band, r.kind, r.n_pair, r.old_pct, r.old_lo, r.old_hi, r.new_pct, r.new_lo, r.new_hi, r.diff_pp,
                    r.diff_lo, r.diff_hi, r.pass_to_fail, r.fail_to_pass))
            # mid-band (reported, not the pass rule)
            om, nm = variant_rows(model, "orig", "mid"), variant_rows(model, v, "mid")
            if len(om) and len(nm):
                _, sm = paired(om, nm)
                lines += ["", "Mid-band (reported): paired %d targets, orig %s -> new %s; diff %+.1f pp [%+.1f, %+.1f]"
                          % (sm["n_pair"], fmt(sm["old"]), fmt(sm["new"]), sm["diff_pp"], sm["diff_lo"], sm["diff_hi"])]
                js[model][v]["mid_paired"] = sm
            lines.append("")
    if all_cells:
        pd.concat(all_cells).to_csv(C.WORK / "report_cells.csv", index=False)
    if all_pairs:
        pd.concat(all_pairs).to_csv(C.WORK / "report_pairs.csv", index=False)
    C.WORK.mkdir(parents=True, exist_ok=True)
    (C.WORK / "report.json").write_text(json.dumps(js, indent=1, default=float))
    (C.WORK / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
