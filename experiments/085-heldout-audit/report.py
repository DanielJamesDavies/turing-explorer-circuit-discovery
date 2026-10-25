"""Tables for the held-out audit (085): out/audit_<tag>.csv (+ store_values.csv, silence_subset.csv) -> out/report_<tag>.md.

  PYTHONPATH=src .venv/bin/python experiments/085-heldout-audit/report.py [all|subset]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"
TAG = sys.argv[1] if len(sys.argv) > 1 else "all"
CELLS = [(b, k) for b in ("L0-4", "L5-7", "L8-11") for k in ("attn", "mlp", "resid")]


def pct(x):
    return "%.1f%%" % (100 * x) if pd.notna(x) else "-"


def table(d, issues, title):
    """issues: [(label, boolean Series over d)] -> share of targets, overall and per band x kind."""
    lines = ["### %s\n" % title, "| issue | all (n=%d) | %s |" % (len(d), " | ".join("%s %s" % c for c in CELLS)),
             "|---|---|%s" % ("---|" * len(CELLS))]
    for label, m in issues:
        cells = []
        for b, k in CELLS:
            sel = (d.band == b) & (d.kind == k)
            cells.append(pct(m[sel].mean()) if sel.any() else "-")
        lines.append("| %s | %s | %s |" % (label, pct(m.mean()), " | ".join(cells)))
    lines.append("\nCell sizes: %s\n" % ", ".join("%s %s %d" % (b, k, int(((d.band == b) & (d.kind == k)).sum()))
                                                 for b, k in CELLS))
    return lines


def main():
    a = pd.read_csv(OUT / ("audit_%s.csv" % TAG))
    L = ["# 085 held-out audit: %s records\n" % TAG]
    sk = a[a["skip"].notna()] if "skip" in a else a.iloc[:0]
    d = a[a["skip"].isna()].copy() if "skip" in a else a.copy()
    L.append("Records %d; skip records %d (%s); usable %d.\n" % (len(a), len(sk), sk["skip"].value_counts().to_dict(), len(d)))
    B = d.arm == "B"

    # ---------------- counts / structure
    L += table(d, [
        ("thin (no mid-band pool, arm A)", d.thin.astype(bool)),
        ("strongest pool < 64", d.n_S < 64),
        ("held-out strongest < 16", d.n_S_ho < 16),
        ("held-out strongest < 8", d.n_S_ho < 8),
        ("mid-band pool present but held-out mid < 16", B & (d.n_M_ho < 16)),
        ("arm B fit set < 48", B & (d.n_fit < 48)),
        ("contrast < 64", d.n_C < 64),
        ("contrast ids missing (neg_ids None)", ~d.neg_ids_present.astype(bool)),
    ], "1. Counts and structure (share of usable targets)")
    L.append("Held-out strongest rows: total %d, distribution %s.\n" % (
        int(d.n_S_ho.sum()), d.n_S_ho.value_counts().sort_index().to_dict()))

    # ---------------- integrity
    integ = [("id -> tokens hash mismatch (any pool)", (d.filter(like="id_hash_mismatch").fillna(0).sum(1) > 0)),
             ("negative / out-of-vocab token", (d.tok_neg + d.tok_oov) > 0),
             ("token 0 (<unk>) in any row", d.tok_unk0_rows > 0),
             ("<|endoftext|> in any row", d.eot_rows_all > 0),
             ("tgt != pos shifted (any pool)", ~(d.tgt_shift_ok_S.astype(bool) & d.get("tgt_shift_ok_M", pd.Series(True, index=d.index)).fillna(True).astype(bool))),
             ("non-finite peak", (d.peak_nonfinite_S.fillna(0) + d.get("peak_nonfinite_M", 0).fillna(0)) > 0),
             ("strongest-pool row with peak <= 0 (target silent on re-forward)", d.peak_le0_S > 0),
             ("mid-pool row with peak <= 0", d.get("peak_le0_M", pd.Series(0, index=d.index)).fillna(0) > 0),
             ("same corpus id in activating and contrast", d.id_overlap_act_ctr > 0),
             ("contrast row token-identical to an activating row (contradiction)", d.exact_ctr_eq_act_rows > 0),
             ("mid row token-identical to a strongest row", d.exact_M_eq_S_rows > 0)]
    L += table(d, integ, "2. Integrity and sanity (share of usable targets)")
    L.append("Strongest-pool rows with peak <= 0: %d rows in %d targets; mid-pool: %d rows.\n" % (
        int(d.peak_le0_S.sum()), int((d.peak_le0_S > 0).sum()), int(d.get("peak_le0_M", pd.Series(0)).fillna(0).sum())))

    # ---------------- held-out strongest leakage
    n = d.n_S_ho
    hs = [
        ("exact dup of a FIT row (>=1)", d.Sho_exact_FIT > 0),
        ("exact dup of a training-side row (FIT, 48-ref, mid-train, contrast-train)", d.Sho_exact_TRAIN > 0),
        ("anchor-prefix identical to a training-side row (>=1)", d.Sho_apfx_TRAIN > 0),
        ("near-dup (LCR>=32 or 8-gram containment>=0.5) of a training-side row", d.Sho_near_TRAIN > 0),
        ("any of exact / anchor-prefix / near (R2 fails >=1 row)", d.Sho_clean_R2 < n),
        ("exact dup inside the held-out set", d.Sho_exact_within > 0),
        ("near-dup / anchor-prefix dup inside the held-out set", (d.Sho_near_within + d.Sho_apfx_within) > 0),
        ("near-dup of a contrast row", d.Sho_near_C > 0),
        ("same source file as a training-side row", d.Sho_src_TRAIN > 0),
        ("half or more of held-out rows fail R2", d.Sho_clean_R2 <= n / 2),
        ("ALL held-out rows fail R2", d.Sho_clean_R2 == 0),
    ]
    L += table(d, hs, "3. Held-out strongest (PRIMARY): leakage, share of targets with >= 1 affected row")
    rows = lambda c: "%d / %d (%s)" % (int(d[c].sum()), int(n.sum()), pct(d[c].sum() / n.sum()))
    L.append("Row-level shares of held-out strongest rows: exact-FIT %s; exact-TRAIN %s; anchor-prefix-TRAIN %s; "
             "anchor-prefix-TRAIN at the same anchor %s; near-TRAIN %s; same-source-TRAIN %s; exact within held %s; "
             "corpus twin exists %s; peak <= 0 %s.\n" % (
                 rows("Sho_exact_FIT"), rows("Sho_exact_TRAIN"), rows("Sho_apfx_TRAIN"), rows("Sho_apfxsa_TRAIN"),
                 rows("Sho_near_TRAIN"), rows("Sho_src_TRAIN"), rows("Sho_exact_within"), rows("Sho_corpus_mult_gt1"),
                 rows("peak_le0_S_ho")))
    L.append("Distinct source files: strongest pool median %d (p10 %d); fit rows median %d.\n" % (
        d.n_src_S.median(), d.n_src_S.quantile(.1), d.n_src_FIT.median()))
    # leakage union decomposed
    only_apfx = (d.Sho_clean_R0 - d.Sho_clean_R1)
    only_near = (d.Sho_clean_R1 - d.Sho_clean_R2)
    L.append("Rows removed per target (mean): R0 %.2f, + anchor-prefix %.2f, + near-dup %.2f, + same source %.2f.\n" % (
        (n - d.Sho_clean_R0).mean(), only_apfx.mean(), only_near.mean(), (d.Sho_clean_R2 - d.Sho_clean_R3).mean()))
    L.append("Threshold sensitivity (rows with LCR >= t vs a training-side row): %s.\n" % ", ".join(
        "t=%s: %s" % (t, pct(d["Sho_lcr%s_TRAIN" % t].sum() / n.sum())) for t in (16, 24, 48)))

    # ---------------- clean counts
    L.append("### 4. Clean held-out strongest counts (rows left of 16)\n")
    L.append("R0 = drop exact dups of training-side or contrast rows and within-held exact dups; R1 = R0 + anchor-prefix "
             "identical; R2 = R1 + near-dup (LCR >= 32 or 8-gram containment >= 0.5); R3 = R2 + same source file.\n")
    L.append("| rule | median | mean | < 16 | < 12 | < 8 | < 4 | 0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    L.append("| as run | %d | %.1f | %s | %s | %s | %s | %s |" % (n.median(), n.mean(), pct((n < 16).mean()),
             pct((n < 12).mean()), pct((n < 8).mean()), pct((n < 4).mean()), pct((n == 0).mean())))
    for r in ("R0", "R1", "R2sa", "R2", "R3"):
        c = d["Sho_clean_%s" % r]
        L.append("| %s | %d | %.1f | %s | %s | %s | %s | %s |" % (r, c.median(), c.mean(), pct((c < 16).mean()),
                 pct((c < 12).mean()), pct((c < 8).mean()), pct((c < 4).mean()), pct((c == 0).mean())))
    L.append("")
    L += table(d, [("R0 < 8", d.Sho_clean_R0 < 8), ("R1 < 8", d.Sho_clean_R1 < 8), ("R2 < 8", d.Sho_clean_R2 < 8),
                   ("R2 < 4", d.Sho_clean_R2 < 4), ("R2 = 0", d.Sho_clean_R2 == 0),
                   ("in-record unseen strongest candidates (held + unused train), R2-clean vs FIT+contrast, < 16",
                    d.cand_S_R2 < 16),
                   ("same, < 8", d.cand_S_R2 < 8)],
                  "4b. Targets below a usable count, by cell")
    L.append("In-record unseen strongest candidates (R2 vs FIT + contrast; only if the A reference moves to the fit's "
             "own strongest rows): median %d, mean %.1f.\n" % (d.cand_S_R2.median(), d.cand_S_R2.mean()))

    # ---------------- anchors and strength
    L += table(d, [("held-out strongest row with anchor 0", d.anchor0_S_ho > 0),
                   ("ALL strongest rows anchored at 0 (position-0 latent)", (d.anchor0_S_ho == d.n_S_ho) & (d.anchor0_S_tr == d.n_S_tr)),
                   ("held-out strongest row with anchor <= 3", d.anchor_le3_S_ho > 0),
                   ("half+ held-out strongest anchors <= 3", d.anchor_le3_S_ho >= d.n_S_ho / 2)],
                  "5. Anchors (held-out strongest)")
    r_ho = d.peak_med_S_ho / d.peak_med_S_tr
    r_fit = d.peak_med_S_ho / d.peak_med_FIT_S
    r_cl = d.Sho_clean_R2_peak_med / d.peak_med_S_ho
    L.append("Held-out vs train strength (median peak ratio per target): held/48-train median %.3f (IQR %.3f-%.3f); "
             "held/fit-strongest %.3f; R2-clean held / all held %.3f (IQR %.3f-%.3f). Peak ties within the 64 pool: median %d.\n" % (
                 r_ho.median(), r_ho.quantile(.25), r_ho.quantile(.75), r_fit.median(), r_cl.median(),
                 r_cl.quantile(.25), r_cl.quantile(.75), d.peak_ties_S.median()))
    L.append("Anchor positions: held-out strongest median-of-medians %.1f, train %.1f; share of held-out rows at "
             "position 0: %s; <= 3: %s.\n" % (d.anchor_med_S_ho.median(), d.anchor_med_S_tr.median(),
                                               pct(d.anchor0_S_ho.sum() / n.sum()), pct(d.anchor_le3_S_ho.sum() / n.sum())))

    # ---------------- mid-band held-out
    m = d[B & (d.n_M_ho > 0)]
    nm = m.n_M_ho
    L += table(m, [("exact dup of a training-side row", m.Mho_exact_TRAIN > 0),
                   ("anchor-prefix identical to a training-side row", m.Mho_apfx_TRAIN > 0),
                   ("near-dup of a training-side row", m.Mho_near_TRAIN > 0),
                   ("anchor-prefix / exact identical to a held-out STRONGEST row", (m.Mho_apfx_Sho + m.Mho_exact_Sho) > 0),
                   ("R2 fails >= 1 row", m.Mho_clean_R2 < nm), ("R2 < 8", m.Mho_clean_R2 < 8),
                   ("held-out mid row with anchor 0", m.anchor0_M_ho > 0)],
                  "6. Held-out mid-band (reported)")
    L.append("Mid-band rows: exact-TRAIN %s, anchor-prefix-TRAIN %s, near-TRAIN %s; R2 clean median %d. Mid peak / "
             "strongest peak (median per target) %.3f.\n" % (
                 pct(m.Mho_exact_TRAIN.sum() / nm.sum()), pct(m.Mho_apfx_TRAIN.sum() / nm.sum()),
                 pct(m.Mho_near_TRAIN.sum() / nm.sum()), m.Mho_clean_R2.median(), m.mid_peak_over_S.median()))

    # ---------------- contrast
    L += table(d, [("held-out contrast exact dup of a contrast-train row", d.Cho_exact_Ctr > 0),
                   ("held-out contrast near-dup of a contrast-train row", d.Cho_near_Ctr > 0),
                   ("held-out contrast near-dup of an activating row", d.Cho_near_ACT > 0),
                   ("duplicate rows inside the contrast set", d.dup_rows_in_C > 0)],
                  "7. Contrast held-out")

    # ---------------- store values
    sv = OUT / "store_values.csv"
    if sv.exists():
        s = pd.read_csv(sv)
        slope = (s.v48 - s.v64) / 16
        v128 = (s.v64 - 64 * slope) / s.v64
        L.append("### 8. What the stores hold beyond the 64 strongest\n")
        L.append("top_ctx.pt: 64 slots per latent (ranked by MEAN activation over the window; filled median %d). Unique "
                 "contents among them: median %d (p10 %d). mid_ctx.pt: a 64-slot reservoir in mu + [0.5, 1.5] sigma of "
                 "the same mean score: median mid score / top rank-1 %.3f, / top rank-64 %.3f. No other store holds "
                 "activating contexts (top_coactivation = latent partners; logit_ctx = output tokens; neg_ctx = silent "
                 "neighbours). Store mean-score decay v64/v1 median %.3f; v48->v64 linear extrapolation puts rank 128 at "
                 "median %.3f x v64 (IQR %.3f-%.3f).\n" % (
                     s.top_filled.median(), s.top_unique.median(), s.top_unique.quantile(.1), s.mid_med_over_v1.median(),
                     s.mid_med_over_v64.median(), s.decay_v64_v1.median(), v128.median(), v128.quantile(.25),
                     v128.quantile(.75)))
    tw = OUT / "twin_ranks_subset.csv"
    if tw.exists():
        t = pd.read_csv(tw)
        L.append("Split mechanics (subset): %d held-out rows have an exact twin in strongest-train; the twin has the SAME "
                 "peak in %s of pairs and sits at the ADJACENT rank in %s (rank distance <= 2: %s). Blocks of 4 hold out "
                 "one rank each, so a tied pair almost always straddles the split.\n" % (
                     len(t), pct(t.same_peak.mean()), pct((t.dist == 1).mean()), pct((t.dist <= 2).mean())))
    sc = OUT / "silence_subset.csv"
    if sc.exists():
        c = pd.read_csv(sc)
        L.append("### 9. Forward-pass checks (subset, %d targets)\n" % len(c))
        L.append("Contrast rows not silent on re-forward: %d of %d rows (%d targets). Best pre-activation rank of the "
                 "target at any position in a contrast row: min over all %d, median of per-target medians %.0f; "
                 "contrast rows where the target is within 2K = 256 of entering Top-K: %d. Strongest peaks re-computed: "
                 "max |diff| median %.3g, max %.3g; anchors changed %d rows (held-out %d); peak <= 0 now %d rows.\n" % (
                     int(c.ctr_not_silent.sum()), int(c.ctr_n.sum()), int((c.ctr_not_silent > 0).sum()),
                     int(c.ctr_rank_best_min.min()), c.ctr_rank_best_med.median(), int(c.ctr_within_2k.sum()),
                     c.strong_peak_maxabsdiff.median(), c.strong_peak_maxabsdiff.max(), int(c.strong_anchor_changed.sum()),
                     int(c.strong_held_anchor_changed.sum()), int(c.strong_peak_le0_now.sum())))
    # ---------------- as-run pass rate by how much of the held-out set is clean (context, not a re-score)
    import glob
    ev = {}
    for f in glob.glob(str(HERE.parent / "062-h100-protocol-v1" / "out_full" / "out" / "main" / "eval.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r and r.get("held") == "strong":
                ev[r["seed"]] = r
    if ev:
        HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
        e = pd.DataFrame(list(ev.values()))
        e = e[~e["vacuous_tk"].fillna(False).astype(bool)]
        ok = e[HEAD].notna().all(axis=1)
        e["passed"] = ok & ((e[HEAD] >= 0.8) & (e[HEAD] <= 1.5)).all(axis=1) & (e["phi_sup_blind_tk"] >= 0.9)
        j = d.merge(e[["seed", "passed"]], left_on="key", right_on="seed")
        j["bin"] = pd.cut(j.Sho_clean_R2, [-1, 0, 3, 7, 11, 16], labels=["0", "1-3", "4-7", "8-11", "12-16"])
        g = j.groupby("bin", observed=False).passed.agg(["mean", "size"])
        L.append("### 10. As-run pass rate (held-out strongest) by R2-clean held-out count\n")
        L.append("Overall %s (n=%d). %s. Between-target comparison only (different latents); the paired answer needs the "
                 "re-score.\n" % (pct(j.passed.mean()), len(j), "; ".join("%s clean: %s (n=%d)" % (b, pct(r["mean"]), r["size"])
                                                                         for b, r in g.iterrows())))
    (OUT / ("report_%s.md" % TAG)).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
