# Lambda calibration + the prior graveyard (2026-07-30)

One question drove this folder: **how should the mask's sparsity be set per
seed, and can anything cheaper than measurement do it?** The answer settled
into one positive result, one descriptive finding, and a run of cleanly
refuted priors.

All runs: dual floor, gamma=0.25, 400 steps unless stated, seeds
comp8/17043 (L2-resid), comp25/4085 (L8-mlp), comp32/36965 (L10-resid),
plus 10-seed within-component samples at comps 8/25/32.

## THE POSITIVE RESULT: one-probe power-law calibration

n follows a power law in lambda: n ~ lambda^-0.759 (per-seed -0.656..-0.892).
One probe run at lambda=1e-5 fixes the curve; the lambda hitting any target
size is lambda_probe * (n_probe/n_target)^(1/0.759). END-TO-END out-of-sample
size errors: **-1.1% / -2.4% / -3.2% / +3.6%** across L2/L5/L8/L10
(one_probe_calibration.jsonl). Cost: 2 runs per seed for size-matched
comparison (probe + final), 1 run if any size is acceptable. CAVEATS: the
exponent was fitted on these four seeds (transfer to new seeds untested
beyond the within-component data); it is a "at 400 steps" quantity; and it
shifts under any pricing change (weighted arms mis-matched by 5-10% using
the flat exponent).

## THE DESCRIPTIVE FINDING: breadth grows circuits, strength shrinks them

Within components (10 seeds x 3 comps, Spearman vs n at fixed lambda):
seq_count +0.53/+0.64/+0.85 (only feature sign-consistent AND |rho|>0.5
everywhere); firing rate similar; a_pos and posctx pre-act NEGATIVE
(-0.79/-0.22/-0.84); coactivation concentration negative. Broad-and-weak
features need many explainers; narrow-and-strong few. POOLED, the a_pos
correlation FLIPS SIGN (+0.53) - Simpson's reversal via depth; never pool.
As a lambda predictor it is useless: best free model 24-29% median size
error vs the probe's 3.6% (seed_features.json, apos_fit.py).

## STEPS ARE A SPARSITY DIAL, NOT A CONVERGENCE KNOB

n still falling at step 1000 at every seed (L8 -27% per 200 steps at the
end). Sparsity pressure accrues as steps*lr*lambda without bound - there is
no fixed point. Quality: refinement at L2 (metrics IMPROVE to 1.01/0.99 as n
falls 40%), erosion at L10 (free0 0.69->0.40), pathology at L8
(freeM_topk -2.5 -> -52.8: the k-sparse fill budget grows as n shrinks and
the fill itself drives this seed negative - a third documented fill-metric
failure mode, hitting SMALL circuits). 400 steps + calibrated lambda is an
operating point, not an optimum; the honest calibration target would be the
product steps*lr*lambda. Graphs: steps_curve_*.png.

## THE PRIOR GRAVEYARD (all refuted by measurement, same day)

1. **grad-scaled lambda** (lambda = c*q99|dL/dtheta|): ANTI-correlated -
   L2 has 20x less gradient than L10 but needs ~4x more lambda; transferring
   c gave a 14x-oversized circuit at L2.
2. **lambda ~ 1/n_sites**: product spans 3.86x, NON-monotone (peaks at L5).
3. **a_pos / feature-based lambda prediction**: 24-29% vs probe 3.6%; adding
   features overfits (3-feature LOO max error 511%).
4. **active-set theta init** (probe-active high, rest low): no speedup is
   POSSIBLE (Adam updates all params in parallel - the uniform-init burn-in
   walks deadweight concurrently with real optimisation, costing nothing but
   meaningless early n readouts; my "20-25% wasted compute" claim was wrong);
   quality equal-or-worse everywhere, catastrophic freeM_topk -4.0 at L8;
   the probe-active set is 50-59% of the dictionary anyway. init_ab.jsonl.
5. **gradient-weighted per-site pricing** (site_weights_ab.jsonl): the C3
   motivation (flat lambda kills collectively-load-bearing diffuse tails)
   predicted "helps most at depth". Measured at matched size: L2 flat 2/3,
   L8 weighted 2/2 usable, **L10 flat 3/3 decisively** (free0 dev 0.47 vs
   0.31) - refuted at exactly the seed it was for. v1 also produced a
   pathology worth remembering: q99-over-all-latents returned EXACTLY 0 for
   compact-active-set sites -> weight 0 -> latents FREE -> 59k-member flood,
   with geometric-mean normalisation propagating one degenerate site into
   250x prices everywhere else (site_weights_ab_broken_weights.jsonl).

## TWO ENGINEERING LESSONS (recurred twice each today)

* **Unbounded scales degenerate at their extremes**: the dual normaliser
  annihilated its own loss term (ratio 1.9e8 at L10); the zero site-weight
  made latents free. Prices, weights and normalisers need bounds BY
  CONSTRUCTION (the shared mean(target^2) normaliser; the [0.25,4] clamp).
* **Quantiles over mostly-zero tensors measure sparsity, not scale**: q99
  over all 40,960 latents when <1% carry gradient returns 0; always reduce
  over the NONZERO support.

## STANDING CONCLUSION

The flat per-latent price survives every alternative tested. The mask's data
gradient already allocates members across sites in the way the loss wants
(the L10 U-shape is demand, not pricing artefact), and per-seed measurement
(the one-probe rule) beats every free predictor by 7x. Engine additions kept,
all default-off: theta_init_mode="active", site_lambda_weights - documented
negative results with tests pinning their semantics.
