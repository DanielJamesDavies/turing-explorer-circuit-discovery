# 102 diagnosis: why circuits fail (diag-102)

Scope: CPU only; no refits. Panels: 095's 72 targets × arms v2 / v2CR / v2R, 100's 18, and the v1 15k run
(held-out sets have duplicate-context leakage, so these numbers are for power on features only). Floor 0.75.
Tables: `out/q1q2.md`, `out/q1_15k.md`, `out/q3.md`, `out/q4.md`; scripts `diag_*.py`.
Counts match the brief, except v2R production: 38/72 here (brief: 37).
(This report was written by the coordinator from the agent's final message, because the agent's own write of
report.md was blocked.)

## Findings

1. **Failures are fit-side, not generalisation.** Only Z has a train-split score; for the other terms the fit's
   training-loss curve stands in.
   - Held-out Z failures that also fail Z on train: v2 7/10, v2CR 15/18, v2R 13/15.
   - v1 15k: 83% of the 2,233 Z failures also fail on train (attention 85–96%).
   - Held-out minus train Z is ≈ 0 for passes and failures alike.
   - Terms that fail on held-out already had higher training loss (AUC 0.70–0.79; v2CR A 0.73, C 0.79, donor C 0.72).
2. **The largest production failure mode is the Top-K cut.** The pre-activation is restored, then rivals at the
   target's own site push it out of the Top-K.
   - v2CR Z failures with pre-activation Z in band: 15/18 (v2 9/10; v1 15k 86% of 2,233).
   - 10 of v2CR's 24 production failures pass every term on the pre-activation read.
   - Rank-keep (3e-3) is about 300× weaker than the Z term.
3. **R costs production through the cut.** 7 of the 10 targets that pass under v2 and fail under v2CR lose Z at the
   cut with pre-activation Z ≥ 0.87. They are mostly L0–7 residual targets with clean rank 1–6 (e.g. 4.resid.25998:
   Z 0.99 → 0.57, pre-activation 0.98).
4. **Clean rank predicts failure strongly.**
   - 72 panel: rank_clean ≥ 32 passes v2CR production 1/12 and resample 0/12, against 47/60 and 22/60 below 32.
   - v1 15k: rank 64–128 → 1% pass; 32–64 → 16%; rank 1 → 72%.
5. **Context diversity predicts failure, but as a marker of target difficulty.**
   - v1 15k by anchor diversity: low 73% pass, high 17–26%.
   - Within-cell AUCs: tr_anchor_top1 0.66, held-out anchor diversity 0.69, rank_clean 0.70.
   - Diverse targets fail Z on train as often as on held-out (38% vs 39%), so they are harder to fit, not
     under-sampled.
   - Unseen held-out anchors give OR 0.61 (p 1e-15), largely confounded with difficulty.
   - The contrast fallback predicts nothing (AUC 0.45–0.52). Leakage doesn't help pass.
6. **How predictable pass/fail is.**
   - 72 panel, cross-validated logistic: v2CR production AUC 0.70 (permutation p 0.01); v2 production 0.73 from layer +
     kind alone; v2CR resample 0.61 (not significant).
   - v1 15k, GBM: layer + kind 0.85, all no-refit features 0.92.
   - 45 features per outcome; the top features survive BH (q 0.017–0.03).
7. **Resample is binding on donor C, and many failures are close.**
   - Donor C is the largest deficit in 19/50 v2CR failures and the only failing term in 13.
   - 26/50 pass production and fail only on donors; 17/50 are within 0.10 of the floor.
   - A +0.05 shift in donor A/C would take resample from 22 to 30.
8. **Refit noise.**
   - A noise-only refit pair (062 v1 vs 093 v1, same records) flips 10/72 production passes (14%), with Jaccard 0.60.
   - Best-of-2 oracle: +4 to +6 of 72. Flips are residual/MLP targets; attention collapses never flip.
   - Differences under about ±6–9 of 72 are noise.

Bookkeeping:
- 11.attn.12206, 2.attn.16773 and 5.resid.35667 have an undefined term but `vacuous_tk` False, so they fail in every
  arm.
- 7.attn.38584 has 9 strongest contexts.

## Ranked arms
1. **v2CR + stronger rank-keep** (3e-2, 1e-1; or `cut` mode with margin). No extra forwards. Expected +3 to +7
   production.
2. **R 0.5–1.0, paired with arm 1.** Expected resample +5 to +10.
3. **Best-of-2 restarts** on a train-split criterion. Expected +2 to +4, at 2× cost.
4. **Two-stage gain refit**, including the rank hinge.
5. **λ 2.5e-4**, or for deep bands only. Size 2.9×.
6. **Worst-term hinge.** Uncertain.

Not supported:
- **Diversity-based context selection, deduplication, more contexts:** at most +2.
- **Contrast-set fixes:** no effect.
- **A pre-activation term:** the pre-activation is already right.
- **Greedy add:** no help in 056.

Report rank ≥ 32 as its own stratum.
