"""111 a2: production specificity (main/spec.shard*.jsonl), per model: target, sibling and control restoration under
Z, A, C (pre-activation read, medians); latents lifted above the unmodified Top-K cut by the circuit and by the empty
circuit; the target's Top-K share; and the share of circuits flagged as amplifiers (main.tex app:specificity and
src/eval/wcm_specificity.amplifier_flag: under at least one of Z / A / C, target_faith_pre >= 0.5 and median sibling
>= 0.8 x target), overall, by site kind, by depth band and among passing circuits, with bootstrap CIs over targets.

Population: targets with a non-vacuous held-out strongest eval row (the headline population); spec rows are scored on
the held-out strongest contexts. A target with no spec row is reported (none expected).

Writes paper/tables/run2026_specificity.tex, out/run2026_specificity.csv (per model x pi medians),
       out/run2026_amplifiers.csv (flag shares by group), out/a2.json

  python -X utf8 experiments/111-failure-anatomy/a2_specificity.py
"""
import numpy as np
import pandas as pd

import lib111 as X

L = X.L


def main():
    med_rows, amp_rows, tex = [], [], []
    summary = {}
    for model in L.MODELS:
        lab = L.LABEL[model]
        strong, _ = X.strong_mid(model)
        sp = X.spec_frame(model)
        if strong.empty or sp.empty:
            L.not_available("%s spec rows" % model)
            continue
        s = strong[~strong.vac].set_index("seed")
        t = X.per_target_spec(sp)
        missing = int((~s.index.isin(t.index)).sum())
        sp = sp[sp.seed.isin(s.index)]
        for pi in ("Z", "A", "C"):
            g = sp[sp.pi == pi]
            med_rows.append(dict(model=lab, pi=pi, n=len(g), target_median=g.target_faith_pre.median(),
                                 sibling_median=g.sibling_faith_median.median(),
                                 control_median=g.control_faith_median.median(),
                                 lifted_circuit_median=g.switched_on_circuit.median(),
                                 lifted_empty_median=g.switched_on_empty.median(),
                                 in_topk_circuit_mean=100 * g.in_topk_circuit.mean(),
                                 in_topk_clean_mean=100 * g.in_topk_clean.mean(),
                                 n_siblings_median=pd.to_numeric(g.n_siblings, errors="coerce").median(),
                                 no_sibling_scored=int((pd.to_numeric(g.n_siblings_scored, errors="coerce") == 0).sum()),
                                 amp_pct=100 * g.apply(X.amplifier_flag, axis=1).mean()))
        s = s.join(t[["amp_any", "amp_Z", "amp_A", "amp_C"]], how="left")
        s["amp_any"] = s.amp_any.fillna(False).astype(bool)
        groups = [("all", s), ("passing", s[s["pass"]]), ("failing", s[~s["pass"]])] + \
                 [(L.KIND_LABEL[k], s[s.kind == k]) for k in L.KIND_ORDER if (s.kind == k).any()] + \
                 [(b, s[s.band == b]) for b, _, _ in L.bands_of(model)]
        cells = {}
        for name, g in groups:
            c = X.rate(g.amp_any)
            cells[name] = c
            amp_rows.append(dict(model=lab, group=name, n=c["n"], amp_k=c["k"], amp_pct=c["pct"], lo=c["lo"],
                                 hi=c["hi"], amp_Z_pct=100 * g.amp_Z.fillna(False).astype(bool).mean(),
                                 amp_A_pct=100 * g.amp_A.fillna(False).astype(bool).mean(),
                                 amp_C_pct=100 * g.amp_C.fillna(False).astype(bool).mean()))
        pass_noflag = 100 * (s["pass"] & ~s.amp_any).mean()
        mr = {r["pi"]: r for r in med_rows if r["model"] == lab}
        summary[model] = dict(n=len(s), missing_spec=missing, amp_pct=cells["all"]["pct"],
                              amp_passing_pct=cells["passing"]["pct"], pass_and_unflagged_pct=pass_noflag)
        print("%-11s n %d (no spec row %d) | target Z/A/C %.2f/%.2f/%.2f | siblings %.2f/%.2f/%.2f | control "
              "%.2f/%.2f/%.2f | lifted A,C %.0f/%.0f vs empty %.0f/%.0f | target in Top-K A,C %.0f%%/%.0f%% | amplifier "
              "%s; passing %s; pass and unflagged %.1f%%" % (
                  lab, len(s), missing, *(mr[p]["target_median"] for p in "ZAC"),
                  *(mr[p]["sibling_median"] for p in "ZAC"), *(mr[p]["control_median"] for p in "ZAC"),
                  mr["A"]["lifted_circuit_median"], mr["C"]["lifted_circuit_median"], mr["A"]["lifted_empty_median"],
                  mr["C"]["lifted_empty_median"], mr["A"]["in_topk_circuit_mean"], mr["C"]["in_topk_circuit_mean"],
                  X.fmt(cells["all"]), X.fmt(cells["passing"]), pass_noflag))
        print("            by kind: " + "; ".join("%s %s" % (n, X.fmt(cells[n])) for n, _ in groups[3:]))
        if tex:
            tex.append([r"\midrule"])
        for j, pi in enumerate("ZAC"):
            r = mr[pi]
            tex.append([lab if j == 0 else "", pi, L.tex_int(r["n"]),
                        "%s / %s / %s" % (L.tex_num(r["target_median"]), L.tex_num(r["sibling_median"]),
                                          L.tex_num(r["control_median"])),
                        "%s / %s" % (L.tex_int(r["lifted_circuit_median"]), L.tex_int(r["lifted_empty_median"])),
                        "%.0f" % r["in_topk_circuit_mean"], "%.1f" % r["amp_pct"],
                        L.fmt_pass(cells["all"], True) if j == 0 else "",
                        L.fmt_pass(cells["passing"], True) if j == 0 else ""])
    pd.DataFrame(med_rows).to_csv(X.OUT / "run2026_specificity_by_pi.csv", index=False)
    L.write_table("specificity", tex, ["Model", r"$\pi$", "$n$", "Target / siblings / control (median)",
                                       "Lifted: circuit / empty (median)", r"Target in Top-K (\%)",
                                       r"Amplifier under $\pi$ (\%)", r"Amplifier, any $\pi$ (\%)",
                                       r"Amplifier among passing (\%)"], "llrllrrll",
                  note="pre-activation restoration (target form) on held-out strongest anchors; lifted = latents "
                       "switched on above the unmodified Top-K cut (median over anchors, then targets); amplifier = "
                       "target >= 0.5 and median sibling >= 0.8 x target (src/eval/wcm_specificity.amplifier_flag)",
                  csv_df=pd.DataFrame(amp_rows))
    pd.DataFrame(amp_rows).to_csv(X.OUT / "run2026_amplifiers.csv", index=False)
    L.write_json("a2", summary)


if __name__ == "__main__":
    main()
