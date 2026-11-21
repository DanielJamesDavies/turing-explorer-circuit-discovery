# Supplementary material (paper/supplement/)

Created 2026-11-16 for DAN-219 (decision 7: Daniel will release a public codebase, so production-detail tables and
figures that only repeat per-band or per-cell detail may move out of the paper). Each file holds the float **verbatim**
as it stood in `paper/main.tex`, with its source comments, plus a header saying where it came from. The paper keeps one
pointer sentence to "the supplementary material in the code release" in place of each.

The files are LaTeX snippets, not standalone documents: `\runtable` and `\runfigure` are defined in the preamble of
`paper/main.tex`, and the data files (`paper/tables/run2026_*.tex`, `paper/figures/run2026_*.pdf`) stay where they are.

| File | Was in main.tex | Label | Pointer left in the paper |
|---|---|---|---|
| `run2026_unweighted_pythia_scatter.tex` | inside the float `tab:run2026-pythia-unweighted` (app:production) | (none; the table stays) | caption of `tab:run2026-pythia-unweighted` |
| `run2026_anatomy.tex` | app:production, "How circuits fail" | `tab:run2026-anatomy` | "How circuits fail", first sentence |
| `run2026_failure_anatomy_figure.tex` | app:production, "How circuits fail" | `fig:run2026-failure-anatomy` | same sentence |
| `run2026_near_threshold.tex` | app:production | `tab:run2026-near-threshold` | prose numbers kept in "How circuits fail" |
| `run2026_induce.tex` | app:production, "Circuit sizes, sufficiency to induce, training dynamics" | `tab:run2026-induce` | same paragraph |
| `run2026_training_curves.tex` | same paragraph | `fig:run2026-training-curves` | same paragraph |
| `run2026_stable_core_bands.tex` | app:underdetermination, "A stable core at scale" | `tab:run2026-stable-core-bands` | "By depth band (...)" sentence |
| `run2026_nulls_detail.tex` | app:nulls, "Harder random circuits" | `tab:run2026-nulls-detail` | "The nulls also fail on the contexts they were fitted on" sentence |

None of these labels is cited from the body; every number the body or appendix prose quotes from them is still stated
in the prose.
