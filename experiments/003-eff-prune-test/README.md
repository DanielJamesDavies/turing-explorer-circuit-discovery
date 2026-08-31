# eff-prune-test (2026-07-23)

*README generated from the scripts' docstrings; the scripts are the record.*

## `runner.py`

Does cutting the low-score tail of a VALIDATED set crater free0?

  4 seeds (L2/L8/L9/L10) x abl-ig_mean PA
      x {+rec2+mag, +rec2+mag+eff-p50, +rec2+mag+eff-p90} ONLY.

The direct probe of C3 ("closure is collective; individual contributions sit at
the measurement floor"): magnitude bisection keeps the smallest FUNCTIONAL
prefix of the |attribution| ranking; the chained effect-threshold forks then
cut the bottom 50% / 90% of that validated set BY ITS OWN |score| distribution
(threshold_mode="pctl"). Same ranking, so the delta is pure stopping rule, and
the two cut depths give a dose-response:

  * free0 craters  -> the validated set's low-score members are collectively
                      load-bearing (C3 holds, now measured on a pruned set).
  * free0 survives -> the tail was dispensable; C3's "individually-invisible
                      but needed" claim weakens and a cheap threshold suffices.

Percentile cuts replace the first attempt's absolute T=0.1 (Marks et al.'s
node default), which sits ~2.3x ABOVE our maximum member score (L2: max
0.043, p50 7e-5) and deletes the whole circuit — the scale-mismatch record is
archived in abs-t0.1-2026-07-23/. Percentiles are scale-free, so every seed's
cut lands inside its own distribution by construction. The resolved absolute
cut and the quantiles are still logged per row.

All forks descend from ONE discovery and ONE rec2+mag prune (clones of the
pruned circuit), so the comparison shares everything upstream — rows within a
seed are same-discovery and directly comparable; rerunning a seed rediscovers
(pooled_abs_threshold subsample noise), so never mix rows across launches.

Evaluation is the standard fixed-anchor matrix (free0/freeM/pinMC + cf + faith
+ anchors + raw a_c), same geometry as the negctx grid.

Per-seed process isolation; resume-safe. Launch via launch.sh (never inline
`for i in ...` through wsl bash -lc — the outer shell eats $i).

  SEED_IDX=0..3 PYTHONPATH=src python experiments/003-eff-prune-test/runner.py

## Result files

`rows_s0.jsonl`, `rows_s1.jsonl`, `rows_s2.jsonl`, `rows_s3.jsonl`
