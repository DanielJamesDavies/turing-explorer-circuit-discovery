# binarize-sweep (2026-07-31)

*README generated from the scripts' docstrings; the scripts are the record.*

## `anneal_hold.py`

Follow-up: 200-step anneal with a reach-then-hold schedule.

The compressed (200-step) anneal matched the 400-step metrics but ended with
2-2.7x the membership churn - the schedule spent all 200 steps DESCENDING and
hit the floor temperature only at the last step, leaving no time at final
sharpness to settle. anneal_reach_frac=0.7 reaches T=0.05 at step 140 and
holds it for the remaining 60.

Per seed: one run at the already-calibrated 200-step lambda (from
anneal_steps.jsonl's -cal arms), wd=0.10, reach=0.7. Judged against the
stored anneal-400 reference rows on metrics AND end-of-run flips.

  PYTHONPATH=src python experiments/011-binarize-sweep/anneal_hold.py

## `anneal_steps.py`

Can a 200-step anneal match the 400-step anneal?

Anneal is the first mode with a natural endpoint - the gate freezes as T->0 -
so the question is whether compressing the whole schedule into half the steps
reaches the same place. Two knobs must move with the step count:

  lambda: sparsity pressure is steps*lr*lambda, so 200 steps at the same
     lambda is HALF the pressure. Arm A doubles lambda (2e-5) to hold the
     product; if its size misses the 400-step reference by >5%, arm B re-runs
     at a probe-corrected lambda (soft-gate exponent 0.759, checked not
     trusted).
  wd: the house rule is steps*lr*wd ~ 1.0 (m_kept calibration breaks
     silently otherwise), so the 200-step arms use wd=0.1.

The T schedule itself needs no adjustment - it is defined over `steps`, so a
200-step run anneals 1.0 -> 0.05 twice as fast by construction.

FULL flip trajectories are saved this time (Sweep 0 kept only summaries), so
the freeze curves can be compared directly: if the 400-step run's flips hit
~zero well before the end, the tail steps were provably idle.

  PYTHONPATH=src python experiments/011-binarize-sweep/anneal_steps.py

## `anneal_truncate.py`

What does stopping an anneal-400 run at step 200 actually give you?

Truncation keeps the 400-run's LOW per-step pressure (lambda 1e-5 - the
low-churn regime) but forfeits half the total pressure (circuit ~2x large)
and stops with the gate half-soft (T = 0.05^(200/399) ~ 0.22, so the
binary-aligned forward that motivates anneal has not happened yet).

One training run per seed; membership snapshotted at steps 200 and 400 from
the optimiser's own parameters and both evaluated. Bit-determinism makes the
step-200 snapshot IDENTICAL to a truncated run.

  PYTHONPATH=src python experiments/011-binarize-sweep/anneal_truncate.py

## `runner.py`

SWEEP 0: training-time binarisation - none vs ste vs anneal.

Motivation (keep_threshold sweep, 2026-07-30): the soft mask converges to
genuinely FRACTIONAL members - 79% of L8's membership has m in (0.5, 0.9),
and 8%/25%/20% of membership (L2/L8/L10) sits within +-0.05 of the cut - and
any post-hoc binarisation is lossy, asymmetrically (raising the cut is
catastrophic, lowering it helps). ste/anneal make TRAINING see the binary
semantics the evals execute (the TopK-SAE property, adapted to a global
membership that still needs gradients for non-members).

PRE-REGISTERED PREDICTIONS:
  1. ste/anneal collapse the near-cut mass to ~0 (the gate is binary or
     near-binary by the end, so theta has no reason to sit at the boundary).
  2. At MATCHED n, ste/anneal beat none on the binary evals.
  If (2) fails, fractional membership is load-bearing structure - not just
  lasso shrinkage - and survival-pressure reporting becomes the primary
  framing rather than harder training.
  Risk watch: STE's biased gradient can oscillate members at the boundary -
  per-step membership flip counts are recorded.

Protocol per seed: none @ lambda=1e-5 (baseline, defines n_target) ->
per-mode probe @ 1e-5 -> per-mode run at
lambda = 1e-5 * (n_probe/n_target)^(1/0.759). The exponent is a soft-gate
quantity, so per-mode sizes are checked in the output rather than trusted;
if the probe already lands within 2% of target, it doubles as the final.

  PYTHONPATH=src python experiments/011-binarize-sweep/runner.py

## Result files

`anneal_steps.jsonl`, `anneal_truncate.jsonl`, `rows.jsonl`
