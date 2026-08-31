# Dual-floor mask + pre-top-k negative filtering

**Re-run 2026-07-30 on the FIXED normaliser and post-H1 numerics.** The
original 2026-07-29 rows are in `superseded_broken_normaliser_and_pre_H1/`
with a WHY.md; do not merge them with these.

4 seeds (L2/L5/L8/L10, stratified indices 2/5/8/10), 11 arms + 2 matched
controls = 52 rows, no failures. 400 steps, lr 0.05 constant, lambda 1e-4,
AdamW wd 0.05, keep_threshold 0.5, bf16 codes, NPA, no magnitude/recurrence
prune, 64 probes. Eval negatives FIXED per seed (random + rank-based
pre-top-k filter) so every arm is graded on one exam.

## What the dual floor is for

A mask's m=0 state IS some eval's fill, so a single-floor mask is always
scored on home turf: zero-floor gets free0 0.985 / freeN 0.405 at L2, while
negctx-floor gets free0 0.203 / freeN 1.015. Each wins its own metric. The
NEUTRAL metrics are the k-sparse ones (freeN_topk, freeM_topk) - both floors
are dense, so neither family owns a k-sparse fill.

Dual scores ONE mask under BOTH floors every step, so it should beat both
specialists on the neutral metrics while losing to each on that specialist's
home metric.

## Result: that prediction holds at 4/4 seeds

At MATCHED node count, against both truncated specialists:

| seed | n | dual freeN_tk | zero@matched | negctx@matched |
|---|---|---|---|---|
| L2 | 569 | **0.9451** | 0.4764 | 0.4244 |
| L5 | 2,458 | **0.4474** | 0.2353 | 0.0 |
| L8 | 8,741 | **0.4236** | 0.0 | 0.0 |
| L10 | 10,201 | **0.3937** | 0.0023 | 0.0 |

| seed | dual freeM_tk | zero@matched | negctx@matched |
|---|---|---|---|
| L2 | **0.9114** | 0.58 | 0.3651 |
| L5 | **0.5434** | 0.36 | 0.0 |
| L8 | **0.9553** | -0.8706 | -1.2041 |
| L10 | **0.3242** | -0.059 | -0.5917 |

free0 (zero-floor's HOME metric) goes the other way at 2 of 4, which is the
expected half of the same prediction.

Standout: at L8 `dual/store` reaches freeN_topk **0.9242 on 8,838 nodes**,
matching zero-floor's 0.9218 on **54,634** - same neutral score, one sixth
the size.

## lambda=1e-4 over-prunes dual at depth

L8 and L10 land at 6k-10k nodes against zero-floor's 55k-108k. The lambda
sweep (`../009-dual-floor/`) puts the depth knee at lambda ~3e-6,
where the TRAINED (not truncated) L10 comparison favours dual outright:
n 85,032 vs 107,500, free0 1.0072 vs 1.0486, freeN_topk 1.0649 vs 0.8586.

## Caveats

* The matched controls are TRUNCATED circuits, not circuits TRAINED to that
  size, and truncation is brutal - zero-floor@matched at L2 overshoots to
  free0 **5.3565**. "Dual beats truncated specialists" is weaker than "dual
  beats specialists trained to that size". The L10 lambda=3e-6 row above is
  currently the only genuine trained-size comparison.
* Sizes are unmatched between families at natural lambda, and every
  fill-based metric is size-confounded (see mask-profile / eval-metric
  notes), so read the matched rows, not the natural-size ones.

## Controls that behaved

* **negctx-only stays dead at depth** - free0 EXACTLY 0.0 and negative
  freeM_topk at L5/L8/L10 - so the normaliser fix did not simply inflate
  everything.
* The pre-top-k rank filter's effect remains small and mixed at this lambda.
