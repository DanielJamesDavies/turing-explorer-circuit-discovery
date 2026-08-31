# objective='maximise' — an unbounded driver objective (2026-08-05)

Daniel: "is there a way to change the objective from loss from the value
at that sequence to just maximise the seed value?"

## What the existing objective does, for contrast

`pos` (abl-mask) targets `_at(pos_nat, pos_argmax)` — the seed's natural
PRE-activation, **one target per probe sequence**, squared error. It
reproduces, and because the error is symmetric it penalises overshoot
exactly as hard as undershoot. Measured per-sequence spread of those
targets (cv): L2/386 **0.403**, L2/2927 0.120, L8/1991 0.118, L8/3005
**0.054** — probe sequences are selected as POSITIVE contexts, so they
sample the top of the seed's range and the vector is tighter than one
might expect. Substituting a scalar mean would cost ~14% relative
squared error at L2/386 but only ~0.3% at L8/3005.

## What was built

`objective="maximise"`: no target at all.

    loss = -(mean seed pre-activation) / max_scale  +  lambda * sum(m)

`max_scale` is the mean natural target — ONE scalar, not a per-sequence
ratio (per-sequence division blows up wherever a target is near zero,
and the spread above is small). After dividing, holdout loss reads -1.0
for "at natural on average", -2.0 for "twice natural".

Unbounded by construction: no fixed point, so membership is whatever
satisfies *value gained per latent > lambda*. It is the unbounded limit
of the existing `raise` objective (which targets `raise_gamma * natural`).
7 new tests (`TestMaximiseObjective`), suite 100 -> 106.

The test that captures the difference: in a toy where a driver
(latent 0 -> +e0) and a suppressor (latent 5 -> -e0) are BOTH present on
the positive stream, `pos` **keeps** the suppressor — it needs it or the
seed overshoots natural — while `maximise` **drops** it. Same data, same
sites, opposite membership.

## Results — 4 seeds at L2 resid, 4 at L8 resid, lambda swept

| panel | lambda | n | mult | free0 | cf | sup | exploded |
|---|---|---:|---:|---:|---:|---:|---|
| L2 | 1e-5 | 27,409 | 5.9 | 5.95 | 1.03 | 1.00 | 0/4 |
| L2 | 1e-3 | 1,652 | 3.3 | 3.22 | 0.90 | 1.00 | 0/4 |
| L2 | 1e-2 | 124 | 7.2 | 0.48 | 0.28 | 0.77 | 2/4 |
| L2 | 1e-1 | 28 | 17.6 | 0.00 | 0.03 | 0.54 | 4/4 |
| L8 | 1e-5 | 86,707 | 2.6 | 2.24 | 1.37 | 1.00 | 0/4 |
| L8 | 1e-3 | 1,888 | 0.8 | 0.77 | 0.88 | 1.00 | 0/4 |
| L8 | 1e-2 | 838 | 7,437 | 505 | 0.00 | 0.49 | 3/4 |
| L8 | 1e-1 | 1,245 | 18,773 | 54 | 0.00 | 0.37 | 3/3 |

`mult` = achieved multiple of natural PRE-activation (what it optimises).

### R1 — it produces genuine driver objects, and lambda<=1e-3 is safe

8/8 seeds at both 1e-5 and 1e-3: cf 0.88-1.37, sup 1.00. The objective
drives the seed to 2-8x natural at L2 while remaining sufficient AND
necessary.

**Best operating point is lambda=1e-3: ~1,650 nodes at L2, ~1,890 at
L8**, cf ~0.9, sup 1.00 on both. The L8 number is the interesting one —
a sufficient-and-necessary object at **0.18% of scope**, where `pos`
needs ~20,000 nodes for its closure circuit. Maximise finds drivers
roughly **10x smaller than closure circuits at depth**, which is the
driver/closure size gap measured with ONE method instead of two.

At depth these drivers are explicitly NOT closure objects: L8 seeds
3005/4468 at lambda=1e-3 read cf 1.05-1.07 and sup 1.00 with free0 only
0.22-0.39. They can CAUSE the seed in a context where it is absent and
the seed dies without them, but they cannot REBUILD it from a zeroed
stream.

### R2 — unbounded maximisation goes off-manifold at lambda>=1e-2

Zeroing nearly all of upstream is far off-manifold (the documented L10
pre-activation explosion, ~1.8e5). Maximise finds and exploits it:
L2 mult 12-30x, **L8 mult 8,900-27,500x** with free0 in the hundreds.
These circuits are causally worthless (cf 0.00) while scoring
spectacularly on every value metric.

**Diagnosing it needs both columns, and the naive guards fail:**

- "free0 == 0" misses it — L8 degenerate rows read free0 **164, 766,
  1,091**, not 0. Whether the seed survives top-k in the wreckage is
  incidental.
- "cf ~ 0" over-flags — L8/4068 at lambda=1e-3 has cf 0.129 with mult
  2.02, which is a legitimate closure-without-drive object, not a
  pathology.

The working discriminator is the PAIR:

    exploded          mult >> 10  AND  cf ~ 0
    legit non-driver  mult ~ 1-5  AND  cf low
    healthy driver    mult 1-8    AND  cf 0.8-1.5, sup ~ 1.0

This is the argument for the bounded `raise` (fitted gamma) as the
production form, with unbounded `maximise` kept as a diagnostic.

### R3 — "prune to free0 ~ 1" works at L2, not at L8

Daniel's follow-up: since maximise overshoots, can extra sparsity land
it on free0 ~ 1 with a SMALLER circuit than `pos`?

`pos` free0 FALLS with lambda (0.985 -> 0.897 -> 0.399), so its free0~1
point is its loosest lambda: ~2,432 nodes at L2. `maximise` crosses 1.0
from ABOVE, and at L2 the crossing sits between lambda 1e-3 (free0 3.22,
n=1,652) and 1e-2 (free0 0.48, n=124) — a few hundred nodes, potentially
4-6x smaller than `pos` at matched fidelity.

At L8 there is no headroom: maximise reaches only 1.04-2.24x at the
loosest lambda and is already BELOW natural by 1e-3 (0.77). Its free0~1
crossing is at ~50-87k nodes, WORSE than pos's ~20k.

Same shape as the compressibility result in
`experiments/025-logit-endpoint`: the shallow seeds have slack,
the deep ones do not.

### R4 — RETRACTED AS STATED. See R6.

**The comparison below is matched on free0, which is the one metric that
rewards what `maximise` does.** R6 re-scores the same objectives on
freeM_dense (the SFC analogue) and the ranking REVERSES: maximise never
exceeds 0.61 at any size, while pos reaches 0.994 at 942 nodes. The
"7-42x smaller at matched fidelity" claim does not survive the change of
floor and should not be quoted. What follows is kept because the
mechanism and the stability observation are still informative.

### R4 — as measured on free0 (bisect_to_natural.py, L2, 4 seeds)

log10(lambda) bisected against free0 = 1.0, 7 steps, both objectives:

| seed | pos n | maximise n | ratio | free0 pos/max | cf pos/max |
|---|---:|---:|---:|---|---|
| 386 | 10,000 | 322 | **31.1x** | 0.991 / 0.994 | 1.52 / 0.61 |
| 2927 | 4,762 | 112 | **42.5x** | 0.984 / 1.032 | 0.95 / 0.81 |
| 7019 | 2,072 | 290 | **7.1x** | 0.990 / 0.999 | 1.10 / 0.53 |
| 7490 | 801 | 477 | 1.7x | 0.980 / **0.934** | 0.97 / 0.83 |

**3/4 seeds: 7-42x smaller at matched-or-better closure fidelity (mean
26.9x).** Seed 7490 is NOT a like-for-like win — its maximise circuit is
less faithful (0.934 vs 0.980) — and should not be counted.

Mechanism: `pos` penalises overshoot and undershoot symmetrically, so it
recruits latents that TRIM the seed back toward natural as well as ones
that drive it. `maximise` only rewards driving and lets sparsity do the
trimming — you land on natural by REMOVING contributors rather than by
adding compensators. Same endpoint, far fewer nodes.

**The stability may matter more than the ratio.** pos's bisected sizes
span 10,000 / 4,762 / 2,072 / 801 — a **12x spread** across four seeds of
one layer for one target. maximise's span 322 / 112 / 290 / 477 (4x).
Some of the headline ratio is pos being erratic rather than maximise
being good, and 7490 is precisely the seed where pos happened to land
well. The defensible claim: *driving-then-pruning reliably lands on
100-500 nodes at free0 ~ 1, where reproducing directly lands anywhere
from 800 to 10,000.*

Cost: counterfactual drive. cf 0.53-0.83 against pos's 0.95-1.52 on the
same seeds. `sup` is ~1.0 for both throughout, so necessity is
unaffected — these compact circuits reconstruct the seed as well from a
zeroed stream but are weaker at CAUSING it against a live competing
context.

Caveat on the ratio: pos's bisection chased 1.000 exactly and so ran to
looser lambda than its own sweep points. Against pos's best SWEEP points
(2,488 / 2,590 / 2,665 / 1,986 at free0 0.98-0.99) the reductions are
7.7x / 23x / 9.2x / 4.2x — smaller, still substantial, and the fairer
number to quote.

### R5 — L9 bisection: the gain does NOT survive depth

Same protocol at L9 resid (comp_idx 29, 29 upstream sites), 4 seeds:

| seed | pos n @ free0 | maximise n @ free0 | cf p/m | verdict |
|---|---|---|---|---|
| 1283 | 11,087 @ 0.968 | 1,481 @ 0.912 | 1.10/0.81 | 7.5x smaller, 0.056 less faithful |
| 1639 | 411 @ **1.000** | 17,401 @ **-7.024** | 0.88/0.96 | **maximise FAILED** |
| 2062 | 44,953 @ 0.942 | 65 @ **0.000** | 1.20/0.02 | **maximise FAILED** |
| 2766 | 45,445 @ 0.958 | 79,301 @ 0.951 | 1.02/0.97 | **pos 1.7x smaller** |

**maximise: 1 better, 1 worse, 2 failed** — against L2's 3/4 wins at
7-42x. So R4 is a SHALLOW-SEED result, not a general improvement to
circuit discovery.

**The mechanism is not what R3 predicted.** R3 argued from the L8 coarse
grid that the free0=1 crossing would sit at a LARGE node count at depth.
Bisection shows something different: **at depth there is often no
crossing at all.** free0 does not pass smoothly through 1 as lambda
rises — it jumps from negative or zero straight to explosive. Seed 1639
ran to the loosest lambda in range without ever exceeding 1.0 (bisection
had nothing to converge on, hence n=17,401 at free0 -7.02); seed 2062's
closest-to-1 row was 0.000. The R3 reasoning was wrong even though its
conclusion (no gain at depth) held.

Note also pos's own instability at L9: 411 / 11,087 / 44,953 / 45,445,
a **110x spread** across four seeds of one layer for one target, worse
than L2's 12x. Two of the four ran to the loosest lambda without reaching
free0=1.0, so those are lower bounds on what pos would need.

L8 was never bisected (only swept on the coarse grid), so R3's specific
L8 prediction remains untested.

## R6 — the floor panel: maximise EXPLOITS the zero floor (freem_sweep.py)

Daniel: "how does maximise do with freeM_dense?" — i.e. on SFC's own
metric (non-members at the task mean, members RECOMPUTED so causal
self-support is still required, dense). L2, 4 seeds, both objectives,
5 lambdas, every circuit scored on all four floors plus cf/sup.

| obj | lambda | n | freeM_dense | free0 | free0/freeM |
|---|---|---:|---:|---:|---:|
| maximise | 1e-5 | 27,409 | 0.613 | 5.946 | **9.7x** |
| maximise | 1e-4 | 7,878 | 0.374 | 4.390 | **11.7x** |
| maximise | 1e-3 | 1,652 | 0.307 | 3.221 | **10.5x** |
| maximise | 3e-3 | 808 | 0.249 | 1.883 | 7.6x |
| pos | 1e-5 | 2,432 | 1.033 | 0.985 | 1.0x |
| **pos** | **1e-4** | **942** | **0.994** | 0.981 | 1.0x |
| pos | 1e-3 | 463 | 0.681 | 0.897 | 1.3x |
| pos | 3e-3 | 285 | 0.384 | 0.841 | 2.2x |

**maximise never exceeds freeM_dense 0.61 at any size.** pos reaches
0.994 at 942 nodes. The ranking is the exact reverse of R4's.

Per-circuit free0/freeM_dense ratio: **pos median 1.17** (range
0.37-3.23), **maximise median 10.66** (range 4.19-58.75). For a circuit
selected by REPRODUCING the seed the two floors agree; for one selected
by MAXIMISING it, free0 reports ~11x more favourably than SFC's metric.

**Mechanism.** maximise wins free0 by DELETING SUPPRESSORS, and deletion
only helps when "deleted" means set-to-zero. Under mean-ablation a
non-member is not removed — it sits at its typical value — so the
suppression is still present and the strategy buys nothing. The overshoot
is a property of the EVALUATION SEMANTICS, not of the circuit's causal
role. Sharpest single instance: seed 2927 maximise reads free0 8.24 and
pin0 **0.000** on the same circuit.

### R6a — WHICH mean-floor convention? dense is architecture-inappropriate

freeM_dense and freeM_topk disagree enormously at depth on IDENTICAL
circuits (L9 seed 1639, n=809: dense **0.051**, topk **1.460** — 29x).
At L2 they agree closely. The reason is in
`ablation_faithfulness.py`'s own docstring: dense mean-field ablation is
"SFC-standard but runs the model on a stream it would never produce."

**Our SAEs are top-k = 128.** Dense mean-filling sets all 40,960 latents
per site to their means — a code ~320x denser than anything the model
emits. SFC used L1/ReLU SAEs, where dense mean-filling is far less
unnatural. So freeM_dense is a LITERAL TRANSCRIPTION of SFC's convention
onto an architecture it was not designed for; **freeM_topk is the closer
functional analogue for a top-k SAE.**

Recommendation: report **freeM_topk as primary** (architecture-correct),
**freeM_dense alongside** (what SFC literally does, what reviewers will
expect), and **do not headline free0** — it inverts rankings (R6),
flatters small circuits (R6d), and its denominator can go NEGATIVE
(R6e).

### R6e — free0's denominator can invert at depth

L9 seed 1639: empty-circuit baseline under the ZERO floor is **14,248**
against a_pos 21.1 (the off-manifold explosion, appearing in the BASELINE
itself). free0's denominator a_pos - a_e0 is therefore **-14,227**.
The fixed point survives (free0 = 1 still means a_c = a_pos) but every
value away from 1 is sign-inverted: the "free0 = -7.02" reported for
maximise in R5 means the circuit drove the seed to ~10^5, NOT below
baseline. Under the mean floor the same seed's baseline is 0.000 — mean
ablation stays on-manifold where zero ablation does not.

**Any free0 reading on a seed with a_e0 >> 0 is unusable except at
exactly 1.0.** This should be checked per seed before free0 is quoted.

### R6b — concise all-metric circuits DO exist, from `pos`

Circuits passing freeM_dense in [0.8,1.25] AND cf>0.7 AND sup>0.85:
**6 of 30 rows, every one of them `pos`, none maximise.**

| seed | lambda | n | freeM_d | free0 | cf | sup |
|---|---|---:|---:|---:|---:|---:|
| 2927 | 1e-4 | **747** | 1.040 | 0.989 | 1.000 | 0.988 |
| 386 | 1e-4 | **863** | 0.917 | 0.998 | 1.055 | 1.000 |
| 7019 | 1e-4 | **1,151** | 1.051 | 0.980 | 1.041 | 1.000 |
| 386 | 1e-5 | 2,488 | 0.905 | 0.987 | 1.276 | 1.000 |
| 2927 | 1e-5 | 2,590 | 1.097 | 0.991 | 0.930 | 1.002 |
| 7019 | 1e-5 | 2,665 | 1.111 | 0.980 | 1.075 | 1.000 |

So the 10^2-10^3 concise-circuit hypothesis HOLDS at L2 — ~750-1,150
latents satisfying free0, freeM_dense, freeM_topk, pin0, cf and sup
simultaneously — but the credit goes to the EXISTING objective at a
better lambda, not to a new one.

### R6f — L9 panel (freem_c29.jsonl), medians over 4 seeds

Medians, not means — one seed (1639) has explosive outliers that make
means meaningless.

| obj | lambda | n | freeM_dense | freeM_topk | cf |
|---|---|---:|---:|---:|---:|
| pos | 1e-5 | 14,295 | 0.635 | **0.945** | 1.137 |
| **pos** | **1e-4** | **4,402** | 0.430 | **0.828** | 0.985 |
| pos | 1e-3 | 1,260 | 0.014 | 0.716 | 0.941 |
| pos | 3e-3 | 648 | 0.000 | 0.240 | 0.833 |
| maximise | 1e-5 | 41,227 | 0.021 | 3.228 | 1.214 |
| maximise | 1e-4 | 7,237 | 0.000 | 1.724 | 0.922 |
| maximise | 1e-3 | 1,692 | 0.000 | 0.086 | 0.937 |

**Typical L9 all-metric circuit: ~4,400 nodes** (freeM_topk 0.83,
cf 0.99) — 10^3-10^4, not 10^2-10^3. The 535-node circuit on seed 1639
is the OUTLIER, and it is also the seed with the broken free0 baseline;
per-seed the range is 535 to >17,920.

**maximise fails freeM_dense outright at depth** (0.00-0.02 at every
lambda, 4/4 seeds) and its freeM_topk is erratic (3.23 / 1.72 / 0.09 /
0.07 / 1.16) because the underlying circuits explode. Its L2 apparent
advantage does not survive to L9 under any mean floor.

**Per-seed baseline pathologies, all four L9 seeds different:**
1283 all baselines 0; 1639 zero-floor baseline **14,248** (free0
denominator negative); 2062 meanTK leak 1.14; 2766 meanD 0.81 / meanTK
3.28. The documented 23-30% posctx leak did NOT reproduce — measured
0%, 0%, 4.5%, 14%. Every normalisation must be per-seed.

**cf does not catch the explosion at depth.** Seed 1639's maximise rows
read cf 0.92-0.96 while driving the seed to ~10^5. The R2 discriminator
(mult >> 10 AND cf ~ 0) works at L2 and FAILS at L9. The only reliable
guard is the absolute multiple of natural activation, which every
normalised metric inherits and therefore cannot flag.

### R6c — the house lambda is ~2.6x too loose at L2

lambda=1e-4 matches lambda=1e-5 on every metric (freeM_d 0.994 vs 1.033,
free0 0.981 vs 0.985, cf 1.034 vs 1.097, sup 0.993 vs 0.995) at **942
nodes against 2,432**. This is judged on the metric the recipe was tuned
for, so it is a straightforward correction to the frozen default, not a
matter of taste. It is also 4/4 seeds.

### R6d — free0 systematically flatters small circuits

pos, mean over seeds: at n=463 free0 0.897 vs freeM_dense 0.681; at
n=285, 0.841 vs 0.384; at n=113, 0.399 vs 0.123. The gap widens as
circuits shrink, so **every small-circuit size claim in this project that
rests on free0 alone is optimistic** and should be re-checked on
freeM_dense before being quoted against published work.

## R7 — THE FLOOR TEST: an unbounded objective exploits ITS OWN floor

Daniel: "what if a different step count or learning rate would be better
for maximise?" — a fair challenge, since the house recipe was calibrated
for `pos`. Investigating it found a bigger confound than lr/steps.

**First: `maximise` was barred from the dual floor for no reason that
applied to it.** The guard was `objective != "pos"` — a whitelist of one
— and its stated rationale is about negctx/contrast/inject, which
already carry a negative-context term. `maximise` puts its loss on the
POSITIVES exactly like `pos`. It inherited the exclusion because it was
added after that line was written. Now enabled (DUAL_OK = pos, maximise);
the restriction's real targets stay barred and are tested.

**Second, and worse: I chose `mask_floor_source="zero"` for maximise in
every earlier runner without reasoning about it** — the one floor it
exploits. So R6's comparison confounded OBJECTIVE with TRAINING ABLATION
SEMANTICS.

Three arms, L2, same seeds/lambdas/hyperparameters, only the training
floor differing (means over seeds, at lambda=1e-4):

| arm | n | freeM_dense | freeM_topk | free0 | cf |
|---|---:|---:|---:|---:|---:|
| **pos** (dual) | 805 | **0.978** | **1.041** | **0.993** | 1.028 |
| max/zero | 8,677 | 0.422 | 1.114 | **5.119** | 0.936 |
| max/mean | 9,218 | **3.265** | 1.827 | 0.733 | 1.283 |

**Training on the mean floor did not fix the exploit — it MOVED it.**
Zero-trained maximise overshoots free0 (5.1x) and undershoots freeM_dense
(0.42). Mean-trained maximise overshoots freeM_dense (3.3x) and drops
free0 to 0.73. Sharpest single pair, seed 2927 at lambda=1e-5:
max/zero reads free0 **8.24** / pin0 **0.000**; max/mean reads free0
**0.000** / freeM_dense **5.36**. Exact mirror images.

**So R6's framing ("maximise exploits THE ZERO FLOOR") was too specific.
The correct claim is: an UNBOUNDED objective exploits ITS OWN floor.**
No floor choice fixes it, because the failure is in the LOSS SHAPE, not
the ablation semantics.

`pos` is the only arm near 1.0 on all three floors at once, and the
reason is not the dual floor — it is that **`pos` has a TARGET**. A
bounded reproduce-this-value loss cannot overshoot any floor, because
overshoot is penalised identically to undershoot. An objective with no
fixed point runs to whatever ceiling its training floor allows.

**Consequence for the hyperparameter question:** no lr, step count or
floor rescues an objective with no fixed point. The bounded `raise`
(targets raise_gamma * natural) is the production form of this idea;
unbounded `maximise` is a DIAGNOSTIC — it is an excellent instrument for
finding what a floor convention is blind to, which is what it did here.

**Also measured (toy test, TestMaximiseDualFloor):** the dual floor does
NOT remove the deletion incentive. Dropping a suppressor still raises the
seed in the zero-floored term (weight 1.0), is neutral in the
negctx-floored term, and is rewarded by L1. And `dual_floor_weight`
defaults to **0.25**, so the negctx term is outweighed 4:1 — every "dual
floor" result in this project is weighted heavily toward the zero floor.

## Caveats

- 4 seeds per panel, resid only.
- `maximise` runs the single ZERO floor (dual is pos-only by
  construction), so it differs from `pos` in floor semantics as well as
  objective.
- The lambda grid is coarse (4 points, decade spacing); the degenerate
  threshold is somewhere in 1e-3..1e-2 and is not resolved.
- `mult` is measured on the discrete kept-set (circuit-only, preact),
  not on the soft mask the optimiser saw. Under `binarize="anneal"` the
  final mask is near-binary so these should agree, but it is not checked.

## Files

- `runner.py` — lambda sweep, both panels (`COMP_IDX` env: 8 = L2, 26 = L8)
- `bisect_to_natural.py` — bisects lambda to free0 = 1.0 for pos vs maximise
- `rows.jsonl` (L2), `rows_c26.jsonl` (L8), `bisect*.jsonl`
