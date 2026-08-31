# learned-mask-spike (2026-07-24)

*README generated from the scripts' docstrings; the scripts are the record.*

## `abl_mask_heldout.py`

Held-out free0 for abl-mask: does closure generalise to unseen probes?

The mask_negctx gate collapsed 0.44 -> 0.055 from train to held-out negatives.
abl-mask's holdout DATA LOSS tracked train closely, which I took as evidence
it generalises — but every free0 in the lambda sweeps was computed on ALL 64
probes, 48 of which the optimiser trained on. This measures free0 per slice.

  train    the probes the optimiser saw (provenance n_train_pos)
  holdout  the probes it did not (provenance n_holdout_pos)
  fresh    additional posctx sequences beyond the 64, if the store has them

Each slice gets its OWN anchors: a_pos(slice) and a_e0(slice), so free0 is
computed exactly as it would be if that slice were the whole probe set.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/abl_mask_heldout.py

## `cf_sweep.py`

Lambda sweeps for the cf-hosted mask modes (contrast + negctx), one seed.

Each objective gets the eval its loss actually targets, alongside free0:

  mask_contrast — selectivity: free0 on posctx (closure) AND the kept set's
    behaviour on negctx (circuit-only on neg tokens at the seed's would-be
    anchors). A selective circuit reconstructs firing on posctx and does NOT
    fire the seed on negctx (a_negfire ~= natural neg ~= 0).

  mask_negctx — gate opening ON THE NATURAL STREAM: the selected edits are
    zeroed out of an otherwise-complete keep set (keep-all is identity), so
    this is the ceteris-paribus knockout, learned. gate_recovery =
    (p_gate - p_neg_nat) / (target - p_neg_nat), measured in PRE-ACTIVATION
    (uncensored). The residual gap decomposes silence into
    suppression-gated vs drive-absent.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/cf_sweep.py

## `determinism_check.py`

Is the ~17% membership churn caused by bf16, or is it run-to-run anyway?

dtype_check.py found fp32 vs stream(bf16) share only ~0.83 Jaccard, with
shared members' m differing by up to 0.34. That is far too large to be
rounding, so either (a) bf16 genuinely changes the answer, or (b) the mask
optimum is non-unique / chaotic and ANY two runs disagree that much
regardless of dtype. Those have opposite implications and the previous
script cannot distinguish them, because it never repeated a dtype.

This runs each dtype TWICE and prints the full pairwise Jaccard matrix.
Read it as:
  within-fp32 ~= within-stream ~= across  -> churn is run-to-run, not dtype
  within-* high, across low               -> bf16 really does change selection

Nothing here is seeded deliberately: the point is to measure the natural
variation the pipeline actually has, not to suppress it.

  PYTHONPATH=src python experiments/008-learned-mask-spike/determinism_check.py

## `dtype_check.py`

Does code_dtype="stream" (bf16) change WHICH latents get selected?

code_dtype="stream" keeps the cached SAE codes in the model's native bf16
instead of promoting them to fp32, saving ~1 GB of peak VRAM. That is only
an acceptable default if it does not move the selection. Membership is a
threshold crossing at m = 0.5, so a latent sitting exactly on the boundary
can flip on pure numerical noise; the question is whether flips are RARE
AND MARGINAL (fine) or widespread (not fine).

Reports the symmetric difference of the two selections and, for any latent
that flips, how far its m sits from the 0.5 threshold.

  PYTHONPATH=src python experiments/008-learned-mask-spike/dtype_check.py

## `gradobs_runner.py`

Gradient/value observability, and the test behind the "learned circuit" idea.

Daniel's hypothesis: many members carry only MINOR negative gradients that do
not mean much individually; collectively they should not be cut. And — the
testable part — after REDUCING the inhibitors, their gradients may flip
POSITIVE, which is what would make an iterative/learned circuit (with a
learning rate) sensible rather than a one-shot sign split.

Measurements, all at the seed's probe positions:

  A. score distribution by sign (from the saved circuit) — are negatives
     systematically smaller than positives, or comparable?
  B. gradient at the NATURAL state: d(seed pre-act)/d(latent), per member,
     via SAEGraphInstrument's detached anchors (exactly the discovery signal).
  C. gradient at the INHIBITORS-ZEROED state — the iteration-2 gradient.
     THE KEY NUMBER: what fraction of members whose gradient was negative at
     the natural state have a POSITIVE gradient once the inhibitors are down?
     High -> the sign is state-dependent, a learned/iterative method is
     motivated. Low -> the sign is intrinsic, one-shot selection is right.
  D. gradient x value (the contribution), by sign, both states.

Writes per-member arrays to grads_{TAG}.pt for offline plotting.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/runner.py

## `heldout_runner.py`

Held-out gate recovery: does the learned gate generalise to unseen negatives?

BLOCKING for the negctx/inject family. mask_negctx and mask_inject optimise
against a set of negative contexts; a learned method can memorise them in a
way one-shot attribution cannot. The v2 inject sweep made the risk concrete —
gate recovery 0.87 on TRAIN negatives with holdout data loss 10.9 — but that
holdout number is a loss, not a recovery, so it can't be compared with the
0.34 the mask_negctx sweep reported.

This measures the SAME quantity on both slices, externally:

  gate_recovery(slice) = (p_gate(slice) - p_nat(slice)) / (a_pos - p_nat(slice))

where p_gate is the ceteris-paribus knockout — every latent kept at its
natural value EXCEPT the learned edits, on the natural stream, measured in
PRE-ACTIVATION at each sequence's would-be-firing anchor. p_nat is that same
slice's untouched pre-activation. The optimiser's own split is read from
provenance (n_train_neg / n_holdout_neg) rather than re-derived.

A third slice, FRESH negatives retrieved independently (different selection
seed via neg_mode), tests generalisation beyond the probe set entirely — the
strongest version, since the holdout slice still comes from the same
retrieval.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/runner.py

## `hubprune_runner.py`

Hub prune: what happens when members firing >10% of tokens are removed?

The mass-by-rate analysis found ~0.5% of members (firing rate > 10% — the
quasi-always-on "hub" latents) carry ~10% of attribution mass, ~20x their
count share. They are also the members whose pinning/injection is most
off-natural. This removes them from the saved rec2+mag circuit and re-runs
the eval matrix, against two references:

  full      — the circuit as saved
  no-hub    — members with rate > HUB_RATE removed
  rand-ctl  — same NUMBER of members removed at random (seed 42)

Metrics: free0, freeM_dense, pinMC_dense (mean-fill pins — the overshoot
metric), and cf/cfa with details (injection overshoot). If the hub latents
drive pinned/injected overshoot, no-hub should pull pinMC and cf toward 1.0
with little free0 cost; if free0 craters instead, hubs are load-bearing.

  SEED_TAG=L10 PYTHONPATH=src python experiments/008-learned-mask-spike/runner.py

## `inject_sweep.py`

Lambda sweep for cf-mask_inject on one seed: the full learned heir of the
original counterfactual question, with its built-in decomposition.

Per point, from provenance (train negatives, pre-activation units):
  p_gate_only    recovery from the learned edits alone (deltas off)
  p_inject_only  recovery from the learned injection alone (mask natural)
  p_both         the joint intervention (the trained state)
plus n split by role and the recovery fractions vs (target - p_neg_nat).

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/inject_sweep.py

## `inject_sweep_v2.py`

mask_inject v2: sweep inject_lambda on its OWN scale, log concentration.

v1 shared one lambda between the two levers and found a diffuse
sub-threshold delta blanket that reached the target exactly with ZERO
selected latents, abandoning the gate entirely. v2 prices delta separately
and reports its concentration, so the row shows whether "recovery" came from
a sparse population or a blanket.

l1_lambda (the gate's price) is PINNED at the value mask_negctx used for its
best gate (1e-3, rec_gate 0.34); only inject_lambda moves. The interpretable
regime is where injection can no longer trivially reach the target.

  SEED_TAG=L8 EXCLUDE=0 PYTHONPATH=src python experiments/008-learned-mask-spike/inject_sweep_v2.py

## `lambda_nesting.py`

Is the lambda sweep NESTED, or does it jump between basins?

L8 lambda=1.6e-3 is an outlier: holdout DATA loss 5.06 against 1.88-2.34
everywhere else, and free0_hold 0.4953 against ~1.0. Its higher-lambda
neighbour (3.2e-3) is back on trend. So the run at 1.6e-3 optimised WORSE,
rather than 3.2e-3 finding something special.

If sparsification were a smooth trajectory, each higher-lambda circuit would
be close to a SUBSET of the one below it: raising the penalty should remove
members, not swap them. Containment = |A & B| / |B| with B the smaller
(higher-lambda) set answers that directly:

  containment ~ 1.0 everywhere  -> smooth nested pruning, 1.6e-3 just
                                   overshot along the same path
  containment dips at 1.6e-3    -> that run left the path into a different
                                   basin, i.e. a rugged landscape

Runs are bit-deterministic, so these reproduce the sweep rows exactly.

  SEED_TAG=L8 PYTHONPATH=src python .../lambda_nesting.py

## `lambda_sweep.py`

Lambda sweep: the learned mask's size/faithfulness Pareto curve on one seed.

lambda is a per-latent price (penalty is a sum), so sweeping it traces the
whole curve in independent runs. Each point: run the engine directly (skip
assembly — we want scores + provenance incl. the holdout loss), evaluate
free0 of the kept set, log everything.

Reference curve: attribution top-K free0 at the same seed (direct-drivers
run) — the one-shot ranking the mask must beat at matched size.

  SEED_TAG=L2 PYTHONPATH=src python experiments/008-learned-mask-spike/lambda_sweep.py

## `lr_sweep.py`

Learning-rate sweep with the decay product held constant.

Decay is schedule-coupled: total shrinkage is exp(-steps*lr*wd), so comparing
learning rates at FIXED wd would confound lr with decay strength. This sweep
sets wd = TARGET_PRODUCT / (steps * lr) at every point, so each run sees the
same total decay (calibrated ~1.0, which holds kept-member m near 0.75) and
lr is the only variable.

Existing data (both wd=0, so decay-free and not comparable to these):
  L8 400/0.1  -> 42,918 members, held-out free0 0.853
  L8 400/0.05 -> 57,268 members, held-out free0 0.934

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/lr_sweep.py

## `mask_prune_forks.py`

Do the existing post-hoc prunes shrink a MASK circuit further?

The mask already selects by optimisation, so rec2/mag are being asked to
improve on a set that was chosen jointly rather than by ranking. Two things
under test:

  1. can rec2 (fires in >= 2 sequences) and mag (free0 bisection) cut the
     mask's 82k L8 circuit without losing closure?
  2. does the mask's m value work as a RANKING for magnitude bisection?
     m is compressed into (0.5, 1.0] by the keep threshold, so its ordering
     may carry much less information than attribution scores do.

free0 is reported per probe slice (train / held-out), since the mask's own
optimisation saw only the train slice.

  SEED_TAG=L8 LAMBDA=1e-4 PYTHONPATH=src python experiments/008-learned-mask-spike/mask_prune_forks.py

## `negctx_free0.py`

free0 of the mask_negctx GATE sets, measured rather than asserted.

The gate latents are suppressive-role edits; evaluated as a free0 keep-set on
posctx (keep only them, zero everything else) the expectation from the
inhibitor-only knockouts is ~0 — but that was an analogy, not a measurement.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/negctx_free0.py

## `per_site_counts.py`

Where does the hot-budget pruning actually remove members?

The 3.11x budget takes L10 from 108,068 members to 26,450. A single average
per site hides the shape of that: pruning could be uniform, or it could be
concentrated in the shallow sites (far from the seed) or the deep ones (near
it). This dumps per-site membership for both arms side by side.

Runs are bit-deterministic, so these reproduce the sweep rows exactly.

  ARMS is not used here; both arms are hardcoded to match the sweep.
  SEED_TAG=L10 PYTHONPATH=src python .../per_site_counts.py

## `preactactonly_runner.py`

The activators-only question, measured UNCENSORED (seed pre-activation).

The post-top-k read used by free0 is floored at 0: a seed that falls out of its
SAE's top-k reads exactly 0.000 however far below the cutoff it sits. In the
free0-actonly run that censored 70 of 95 rows, so "are the inhibitors really
activators?" was unanswerable — a 0.000 is consistent with both "removing them
removed drive" and "removing them left the seed just below threshold".

Pre-activation (w_seed . x + b_seed) is continuous and signed, and is the SAME
quantity discovery optimises (the ig_mean "drive" objective), so this also
closes a discovery/evaluation metric mismatch.

Per (seed, arm), all in pre-activation units:
  p_pos     natural run (no intervention)          — the target
  p_e0      empty circuit, everything zeroed       — the floor
  p_all     all members (both signs)               — standard free0's state
  p_act     activators only, inhibitors zeroed     — Daniel's proposal
  p_inh     inhibitors only
and the normalised phi = (p - p_e0) / (p_pos - p_e0) for each, which is free0's
formula computed on an uncensored measurement.

DECISIVE READ: if p_act > p_all, the inhibitor members were suppressing (their
removal raises drive) — Daniel's model holds. If p_act < p_all, they were net
contributing drive at that seed.

  SEED_IDX=0..9 PYTHONPATH=src python experiments/008-learned-mask-spike/runner.py

## `probe_sweep.py`

Probe-count sweep: is the learned mask data-limited?

The sparse-circuit generalisation penalty measured on 64 probes (~12% train ->
held-out) is the signature of a data-limited fit. Unlike lr, probe count has
no coupled budget: steps, lr, lambda and wd all stay fixed, so more probes
means strictly more distinct data behind the same optimisation.

The headline metric is the GAP (free0_train - free0_holdout), not free0
itself: if the mask is overfitting its probes, the gap should shrink as the
probe pool grows. Circuit size and calibration are reported alongside because
more data may also change what the optimiser considers necessary.

Note on epochs: batch and steps are fixed, so 400 steps x batch 4 = 1600
sequence-visits regardless. With 48 training probes that is ~33 passes over
the data; with 192 it is ~8. So this sweep trades repetition for coverage,
which is exactly the trade the overfitting hypothesis predicts should help.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/probe_sweep.py

## `schedule_sweep.py`

Does an lr SCHEDULE beat a constant lr at matched budget?

Both budgets scale with sum(lr) — sparsity is sum(lr)*lambda, decay is
sum(lr)*wd — so a cosine/linear decay to zero halves sum(lr) for the same
peak. Peak lr is therefore DOUBLED (0.05 -> 0.1) so every arm sees
sum(lr) = 20, identical lambda and wd budgets, and the only difference is
the SHAPE of the schedule.

Rationale for expecting a difference: membership is a threshold crossing at
m = 0.5, so with constant lr a latent oscillating near the boundary has its
inclusion decided by wherever the final step left it. Decay freezes
membership progressively instead.

lr_min_frac = 0: at lr = 0 the AdamW update is exactly zero (both the
gradient and the decay term scale with lr), so a zero floor costs nothing
and makes the budget match exact.

Constant is RE-RUN here rather than reused from earlier sweeps: the code has
changed since (bf16 codes, empty_cache), so old rows are not comparable.

  PYTHONPATH=src python experiments/008-learned-mask-spike/schedule_sweep.py

## `spike.py`

Phase-0 spike: abl-mask end-to-end on L2.

Kill criterion from the plan: if the mask's circuit does not beat the
same-size attribution baseline on free0, stop and rethink. Reference points
(L2, saved circuits): abl-ig_mean rec2mag = 0.965 @ 15,590 members; the
direct-drivers K-sweep gave attribution top-K free0 of 0.008 @ 256 / 0.47 @
4,096.

  PYTHONPATH=src python experiments/008-learned-mask-spike/spike.py

## `wd_sweep.py`

Weight-decay sweep at fixed lambda: is AdamW's edge real or a crossover?

The Adam-vs-AdamW comparison at 400/0.05 showed a monotone trend in
(free0_AdamW - free0_Adam) across lambda: -0.149, -0.043, +0.007, +0.002,
-0.002. The single positive sits where the trend crosses zero, so it may be
an artifact rather than a benefit.

Mechanism under test: decoupled decay pulls theta -> 0, i.e. m -> 0.5, the
keep threshold. That HURTS sparsification (it drags L1-suppressed latents
back over the line) but could HELP by countering sigmoid saturation — a
latent at theta=+8 has gradient ~3e-4 and is effectively frozen, so decay
keeps membership revisable late in training.

  interior optimum in wd  -> the anti-saturation effect is real, keep AdamW
  monotone decline from 0 -> the +0.007 was the crossover, Adam is the default

CONTROL: AdamW at weight_decay=0 IS Adam. The wd=0 row must reproduce the
existing Adam/400/0.05/lambda=1e-4 row (57,268 members, free0_all 0.9751); if
it does not, the two runs differ for some other reason and the whole
comparison is void.

  SEED_TAG=L8 PYTHONPATH=src python experiments/008-learned-mask-spike/wd_sweep.py

## Result files

`abl_rows.jsonl`, `cf_rows.jsonl`, `gradobs_rows.jsonl`, `heldout_rows.jsonl`, `hubprune_rows.jsonl`, `inject_rows.jsonl`, `inject_v2_rows.jsonl`, `lambda_rows.jsonl`, `lr_rows.jsonl`, `mask_prune_rows.jsonl`, `negctx_free0_rows.jsonl`, `rows.jsonl`, `rows_s0.jsonl`, `rows_s1.jsonl`, `rows_s2.jsonl`, `rows_s3.jsonl`, `rows_s4.jsonl`, `rows_s5.jsonl`, `rows_s6.jsonl`, `rows_s7.jsonl`, `rows_s8.jsonl`, `rows_s9.jsonl`, `schedule_rows.jsonl`, `wd_rows.jsonl`
