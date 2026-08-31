# Isolating the training floor, and what the arms actually disagree about

Two questions after the maximise/freeM work showed circuit identity is
objective- AND floor-dependent:

1. **Is `pos/dual` actually the right default?** Every `pos` result in
   this project used `mask_floor_source="dual"`, and dual has never been
   isolated FOR POS. It matters because `dual_floor_weight = 0.25`, so
   dual is weighted 4:1 toward its zero-floored term — it might be
   contributing nothing.
2. **Do the arms disagree on the METRICS, or on the LATENTS?** If they
   select nearly the same latents, this whole thread is about evaluation
   convention. If they select different ones, each objective finds a
   genuinely different subgraph.

L2 resid, 4 seeds, 6 arms x 4 lambdas, full metric panel, membership
archived. `ALL-PASS` = free0, freeM_dense and freeM_topk all within
[0.8, 1.25] AND cf > 0.7 — i.e. faithful under every floor convention at
once, not just the one it trained on.

## What `dual` is

**zero + negctx**, not posctx+negctx. `FLOORS_NEEDING_NEGATIVES =
("negctx", "dual")`, so its floor means come from `neg_tokens`. Two
patchers over the SAME thetas — `patcher_zero` (floors=None) at weight
**1.0** and `patcher` (negctx floor) at weight **0.25** — same data
(positive probes), same target (natural posctx pre-activation), gradients
from both accumulating into one parameter set.

## R1 — `pos/dual` is the right default, by 20-28x

Means over 4 seeds:

| arm | lambda | n | free0 | freeM_dense | freeM_topk |
|---|---|---:|---:|---:|---:|
| pos/zero | 1e-4 | 2,719 | **0.991** | **0.529** | 0.906 |
| pos/posctx | 1e-4 | 3,224 | **0.542** | **0.985** | 0.851 |
| pos/negctx | 1e-4 | 3,881 | 0.681 | 1.006 | 0.869 |
| **pos/dual** | 1e-4 | **942** | **0.981** | **0.994** | **1.075** |

**`pos` on a SINGLE floor specialises exactly like `maximise` does.**
Trained on zero -> wins free0, fails freeM (0.53). Trained on a mean
floor -> wins freeM, free0 collapses (0.54, and to 0.019 at lambda=3e-3).
Only dual holds both.

Smallest ALL-PASS circuit per arm:

| arm | smallest ALL-PASS | passes |
|---|---:|---|
| **pos/dual** | **747** | 8 |
| pos/posctx | 3,224 | 5 |
| pos/negctx | 3,301 | 5 |
| pos/zero | **never** | 0 |
| max/zero | never | 0 |
| max/mean | never | 0 |

**`pos/zero` never passes at any size** — freeM_dense never exceeds 0.66
even at 10,576 nodes. The zero floor alone cannot produce a circuit that
survives mean-ablation, however many latents it is given. That is
stronger than "less efficient".

The six smallest ALL-PASS circuits are ALL `pos/dual` (747-2,665) before
any single-floor arm appears. **So the 0.25-weighted negctx term buys a
20-28x size reduction at equal faithfulness.** It is load-bearing.

**This RE-DERIVES the documented rationale for dual.** The design comment
in `learned_mask.py` already says: *"a single-floor mask learns whatever
its own floor rewards… the negctx-floored mask reaches freeN 0.66-1.06
while its free0 is EXACTLY 0.0 at L5 and L8"*. What is new here is the
SIZE dimension and the ALL-PASS framing.

## R2 — the arms disagree on the LATENTS, and the FLOOR matters more than the OBJECTIVE

Membership Jaccard at matched size (each pair compared at the lambda
settings whose n are closest, so size cannot drive the number). The
uniform random null for sets of this size drawn from the same live pool
is **~0.004** (measured in 024-l2-crossover).

| pair | jaccard | contain | what differs |
|---|---:|---:|---|
| **max/mean vs max/zero** | **0.131** | 0.243 | **floor only** |
| max/zero vs pos/zero | 0.195 | 0.346 | objective only |
| max/zero vs pos/dual | 0.240 | 0.419 | both |
| pos/negctx vs pos/zero | 0.329 | 0.553 | floor only |
| pos/posctx vs pos/zero | 0.374 | 0.567 | floor only |
| pos/dual vs pos/zero | 0.457 | 0.645 | floor only |
| max/mean vs pos/dual | 0.484 | 0.674 | both |
| pos/negctx vs pos/posctx | 0.509 | 0.708 | floor only |
| **max/mean vs pos/posctx** | **0.539** | 0.752 | **objective only** |
| **pos/dual vs pos/negctx** | **0.602** | 0.789 | floor only |

Two conclusions:

**(a) MECHANISM, not just measurement.** Every pair is 30-150x above the
random null, so there IS a shared core — but Jaccard 0.13-0.60 means
**40-87% of members differ**. Containment 0.24-0.79 rules out nesting
(the same object at two resolutions). These are genuinely different
latent sets, not one circuit viewed through different metrics.

**(b) The TRAINING FLOOR determines membership more than the OBJECTIVE
does.** The single lowest agreement in the table is the SAME objective
under two floors (`max/mean` vs `max/zero`, **0.131**), and one of the
highest is two DIFFERENT objectives under the same mean floor
(`max/mean` vs `pos/posctx`, **0.539**). Changing `maximise`'s floor
changes its membership more than swapping the objective for `pos` does.

More generally the mean-floored arms (pos/posctx, pos/negctx, pos/dual,
max/mean) cluster at 0.46-0.60 with each other regardless of objective,
while the zero-floored arms (pos/zero, max/zero) are the outliers,
agreeing with everything at 0.13-0.46. **Zero-ablation selects an
idiosyncratic set; mean-ablation selects a common one.**

## R3 — `triple` (zero + negctx + posctx) beats dual by ~2x

Built 2026-08-05 on Daniel's suggestion; `triple_floor_weight` 0.25 on
the third term. Full 8-arm panel, smallest ALL-PASS circuit:

| arm | smallest ALL-PASS | passes |
|---|---:|---:|
| **pos/triple** | **394** | **11/16** |
| pos/dual | 747 | 8/16 |
| pos/negctx | 3,301 | 5/16 |
| pos/posctx | 4,067 | 5/16 |
| pos/zero | never | 0/16 |
| max/zero, max/mean, max/triple | never | **0/48** |

Per-seed dual -> triple: 863->492, 747->394, 1151->621, 1007->1027 —
**wins 3/4 by 1.75-1.90x**, ties the fourth.

**The whole gain is at lambda=1e-3**, where dual fails 0/4 and triple
passes 3/4 at the same size (freeM_dense 0.842 vs 0.681, free0 0.944 vs
0.897). So the third floor does NOT find better latents at a given
lambda — it **extends the usable sparsity range by ~2x**.

Two mechanism notes:

- **`max/triple` overshoots ALL THREE floors at once** (free0 5.7,
  freeM_d 3.2, freeM_tk 2.8 at 1e-5). Adding ablation semantics gives an
  unbounded objective more places to run to, not fewer — the cleanest
  statement yet that the failure is the LOSS SHAPE.
- **Sparsity constrains the mean floors but barely touches free0.**
  Seed 7019, max/triple at 1e-3: freeM_dense 0.842 and freeM_topk 1.203
  are both IN BAND while free0 sits at 3.06. Under zero-ablation the
  members are the only thing present, so pruning removes drive AND
  competition together and their ratio hardly moves; under mean-ablation
  the background is always there, so pruning reduces the circuit's
  relative contribution monotonically.

**ATTRIBUTION — none of this was the objective.** `pos` is unchanged in
every winning arm. From the house recipe (pos/dual @1e-5, n=2,432) to the
best setting (pos/triple @1e-3, n~400-620) is **4-6x**, split about
evenly between **lambda (2.6x)** and **floor (2x)**. The two experiments
that DID change the objective — `maximise` and `logit` — both failed.
The leverage is in intervention semantics and calibration.

**Caveat specific to triple:** it trains on posctx, so **freeM_dense is
IN-SAMPLE** and a third of the ALL-PASS criterion is self-graded. The
held-out columns do independently favour it (free0 0.944, cf 0.909 at
1e-3 vs dual's 0.897 / 0.828). And posctx is the floor the codebase
excluded for self-crediting at depth — 0% leak measured at L2, UNTESTED
at L9.

## R4 — L9: triple buys RELIABILITY, not compactness (rows_c29.jsonl)

pos/dual vs pos/triple at L9 resid (29 upstream sites), 4 seeds:

| arm | lambda | n | free0 | freeM_dense | freeM_topk | pass |
|---|---|---:|---:|---:|---:|---|
| pos/dual | 1e-5 | 14,295 | 0.923 | 0.590 | 0.948 | 1/4 |
| pos/dual | 1e-4 | 4,402 | 0.846 | 0.478 | 0.761 | 1/4 |
| pos/dual | 1e-3 | 1,260 | 0.561 | 0.016 | 0.655 | 0/4 |
| pos/triple | 1e-5 | 15,313 | 0.913 | **0.959** | 0.999 | **3/4** |
| pos/triple | 1e-4 | 4,998 | 0.833 | 0.801 | 0.663 | 0/4 |
| pos/triple | 1e-3 | 1,517 | 0.585 | 0.311 | 0.577 | 0/4 |

**Seeds with at least one ALL-PASS: triple 3/4 (1283, 2062, 2766), dual
1/4 (1283 only). But dual's single success is 3x SMALLER: 5,078 vs
14,388.**

So the L2 result does NOT carry over unchanged. At L2 triple was both
more reliable AND smaller (394 vs 747). At L9 it is more reliable and
LARGER. The third floor buys coverage at depth, compactness at shallow.

**Mechanism: triple rescues exactly the metric dual leaves
unconstrained.** freeM_dense means — dual 0.590 / 0.478 / 0.016 / -0.001
against triple 0.959 / 0.801 / 0.311 / 0.089 across the lambda grid
(1.6x, 1.7x, **19x**, n/a). Per seed the pattern is sharper still: where
dual is already fine the third floor changes nothing (seed 1283 at 1e-5:
1.054 -> 0.983, i.e. 0.9x), where dual FAILS it recovers 3-29x (seed
1639 at 1e-5: 0.074 -> 0.953, **13x**). It is insurance against a
specific failure mode, not a general improvement.

**Caveat — the posctx leak is real on one L9 seed.** Seed 2766's empty
baselines: meanDense **0.814** (3.5% of a_pos), meanTopK **3.281**
(14.1%). That is the self-crediting the design comment warns about, and
triple TRAINS on that floor. Its ALL-PASS there (freeM_dense 0.947 vs
dual's 0.596) is therefore partly self-graded — though 0.596 -> 0.947 is
far larger than a 3.5% baseline shift explains. The other three L9 seeds
measured 0.0% dense leak.

**Also: seed 1639's free0 is unusable** — its zero-floor empty baseline
is 14,248 against a_pos 21.1, so the free0 denominator is negative and
only free0 == 1.0 is interpretable. Neither arm passes there and the
ALL-PASS criterion requires free0, so that seed contributes nothing
either way.

## What this means for the recipe

- **Keep `pos/dual`.** It is the only configuration producing circuits
  faithful under every convention at readable size.
- **Retune lambda: 1e-4, not the house 1e-5.** 942 nodes vs 2,432 with
  identical ALL-PASS status, 4/4 seeds.
- **Never report a circuit on the floor it trained on alone.** The
  held-out floors are the validation set; a single-floor number is
  in-sample by construction.

## Caveats

- L2 resid only, 4 seeds. The L9 equivalent is unrun and depth has
  reversed conclusions repeatedly in this thread.
- ALL-PASS uses an arbitrary [0.8, 1.25] band and cf > 0.7. Tightening
  it would shrink every arm's pass count; the ORDERING is what matters.
- The matched-size overlap compares each pair at the lambdas whose n are
  closest, which is not the same as truncating both by a shared ranking
  (membership is unordered, so that is not available).
- `pos/posctx` trains on a floor the codebase deliberately excluded (it
  self-credits at depth). Measured on these 8 seeds the posctx leak is
  0.0% on 7 and 3.5% on 1 — far below the 23-30% the design comment
  records for L8/L9 — so that exclusion may deserve re-examination, but
  it is not settled here.

## Files

- `runner.py` — 6 arms x 4 lambdas, full panel, archives membership
- `overlap.py` — Jaccard / containment, raw and matched-size
- `rows.jsonl`, `members.jsonl.gz`

## R5 — the posctx weight is a smooth, monotone dial (`triple_floor_weight`)

Daniel: "could we adjust the posctx part to be slightly less, to do an
in-between?" It is already a parameter — `triple_floor_weight`, default
0.25, and weight 0 reproduces dual exactly (asserted in
`TestTripleFloor::test_third_term_actually_contributes`).

L2 seed 2927 at lambda=1e-3, every metric rising with the weight (see
the caveat below — the trend is consistent in aggregate but not strictly
monotone on every seed):

| arm | w | n | free0 | freeM_dense | cf | pass |
|---|---:|---:|---:|---:|---:|---|
| pos/dual | 0 | 335 | 0.960 | 0.773 | 0.885 | |
| pos/tri.05 | 0.05 | 341 | 0.938 | 0.780 | 0.936 | |
| **pos/tri.10** | 0.10 | **363** | 0.981 | **0.834** | 0.943 | **Y** |
| pos/triple | 0.25 | 394 | 0.951 | 0.913 | 1.038 | Y |
| pos/tri.50 | 0.50 | 411 | 1.029 | 0.935 | 0.994 | Y |

freeM_dense climbs 0.773 -> 0.935 while n grows 335 -> 411: a 23% size
cost for 21 points of faithfulness, band crossed at **w = 0.10**. Note
dual is only 8% smaller than tri.10 yet FAILS — the extra nodes buy the
pass rather than being wasted.

**The load-bearing observation is `cf`.** cf is HELD OUT — it is not a
floor and nothing trains on it — and it rises monotonically with the
posctx weight too (0.885 -> 1.038 on this seed; 0.688 -> 0.804 on seed
386). So the third floor improves the CIRCUIT, not merely the metric it
self-grades. That was the main objection to triple and this is direct
evidence against it.

**Caveat on monotonicity:** the trend holds in aggregate and on the L2
seeds, but inverts occasionally. L9 seed 1639 at lambda=1e-4 runs
freeM_dense 0.197 (dual) -> 0.320 (w.05) -> **0.776 (w.10) -> 0.682
(w.25)** — the last step goes BACKWARDS. Read the dial as "consistently
increasing with occasional inversions", not as strictly monotone.

**Practical:** w=0.10 is the better default for MINIMUM SIZE (it takes
the two smallest ALL-PASS circuits in the whole panel — 363 and 488,
beating w=0.25's 394 and 492), while w=0.25-0.50 gives more freeM_dense
MARGIN at ~5-25% more nodes. Choose by whether you need to clear the
band or to sit comfortably inside it.

No turnover by w=0.50, so the 0.25 default is conservative; w=1.0 and
2.0 are queued to find where the posctx term stops paying. Expect it
somewhere past 1.0, where posctx starts to dominate the zero term (1.0)
and the arm degenerates toward the single-floor `pos/posctx` behaviour
that reads free0 0.02.

## R6 — the optimal posctx weight is DEPTH-DEPENDENT (and seed-dependent)

Sweeping `triple_floor_weight` at both depths.

**L2 — smallest ALL-PASS per seed:**

| arm | w | 386 | 2927 | 7019 | 7490 |
|---|---|---:|---:|---:|---:|
| pos/dual | 0 | 863 | 747 | 1,151 | 1,007 |
| pos/tri.05 | .05 | 816 | 758 | 1,146 | 1,004 |
| **pos/tri.10** | .10 | **488** | **363** | **581** | 1,015 |
| pos/triple | .25 | 492 | 394 | 621 | 1,027 |
| pos/tri.50 | .50 | 512 | 411 | 641 | **645** |

w=0.10 wins 3/4; w=0.50 wins the fourth and by a lot (645 vs ~1,010 for
everything else). 7490 is the seed that resisted every other
intervention today — a HEAVIER posctx term is the first thing to move it.
w=0.05 is barely better than dual, so the threshold between 0.05 and 0.10
is sharp.

**L9 — the optimum moves DOWN.** Seed 1283 at lambda=1e-4:

| arm | n | free0 | freeM_dense | pass |
|---|---:|---:|---:|---|
| pos/dual | 5,078 | 0.821 | 1.016 | Y |
| **pos/tri.05** | **5,062** | **0.936** | 1.008 | **Y** |
| pos/tri.10 | 5,076 | **0.755** | 1.032 | |
| pos/triple (.25) | 5,169 | **0.789** | 0.991 | |

**Both w=0.10 and w=0.25 fail on FREE0, not on freeM_dense.** free0
degrades monotonically with the posctx weight (0.936 -> 0.755 -> 0.789)
while freeM_dense stays ~1.0 throughout.

**Mechanism.** The third floor TRADES zero-floor faithfulness for
mean-floor faithfulness. At L2 free0 has slack (dual already reads
0.90-0.98) so the trade is cheap and w=0.10-0.50 pays. At L9 free0 is
already marginal (dual manages 0.821) so there is nothing to give away,
and only the lightest weight is affordable. **Shallow seeds have free0
slack to trade; deep ones do not.**

So there is no single best weight: **w=0.05 at depth, w=0.10 shallow,
w=0.50 for seeds that refuse to compress.** Every weight still beats
dual, so the third floor is worth having regardless — the weight only
sets how much.

## Session-level result (L2, 188 rows, 48 ALL-PASS)

Same seeds, same ALL-PASS bar, start to finish:

    house recipe   pos/dual   lambda=1e-5   n = 2,432
    best measured  pos/tri.10 lambda=1e-3   n =   363     6.7x

Decomposed, three independent changes, **none of them the objective**:

    lambda 1e-5 -> 1e-4              2.6x
    dual -> triple floor             1.9x
    posctx weight 0.25 -> 0.10       1.08x

## R6b — L9 weight sweep, FINAL (supersedes R4's coverage-vs-size trade)

| arm | w | 1283 | 1639 | 2062 | 2766 |
|---|---|---:|---:|---:|---:|
| pos/dual | 0 | 5,078 | – | – | – |
| **pos/tri.05** | .05 | **5,062** | – | 18,531 | 14,654 |
| **pos/tri.10** | .10 | 17,116 | – | **18,356** | **14,369** |
| pos/triple | .25 | 16,701 | – | 19,142 | 14,388 |

Coverage / smallest / total passes: dual **1 seed** / 5,078 / 2;
tri.05 **3 seeds** / **5,062** / 4; tri.10 2 seeds / 17,116 / 2;
tri.25 3 seeds / 14,388 / 3.

**R4's "coverage vs compactness" trade was not real — it was the wrong
weight.** tri.05 matches full triple's 3-seed coverage AND beats dual's
smallest circuit, which no other arm does. Nothing passes on 1639 at any
weight (broken free0 denominator, so it cannot pass by construction).

**But the optimal weight is SEED-dependent at both depths** — 0.05 on
1283, 0.10 on 2062 and 2766; 0.10 on three L2 seeds, 0.50 on the fourth.
So the defensible statement is a RANGE, not a value:
**w ~ 0.05-0.10 at depth, ~0.10-0.50 shallow**, tuned per seed if the
size matters.

**Per-seed spread dominates depth.** Seed 1283 passes at 5,062 while
2062 and 2766 need 14,000-18,500 — a 3.7x spread within one layer at the
same weight. "L9 needs ~15k" is really "most L9 seeds need ~15k and one
needs 5k", the same wide per-seed variance every experiment in this
thread has shown.

## R7 — NO TURNOVER: w=2.0 is the best measured, and the 0.25 default is ~8x too low

Probing above 0.50 to find where the posctx term stops paying. It does
not stop.

| arm | w | 386 | 2927 | 7019 | 7490 | seeds covered |
|---|---|---:|---:|---:|---:|---:|
| pos/tri.10 | .10 | 488 | 363 | 581 | – | 3 |
| pos/tri.50 | .50 | 512 | 411 | 641 | **645** | 4 |
| pos/tri1.0 | 1.0 | – | 302* | 670 | – | 2 |
| **pos/tri2.0** | **2.0** | **473** | **341** | **527** | 771 | **4** |

(*tri1.0 on 2927 reaches n=302 at lambda=3e-3 — the smallest PASSING
circuit anywhere in the panel, but tri1.0 covers only 2 seeds.)

**w=2.0 passes 7 of 8 configurations and covers all four seeds** — the
only arm to do so — and wins 3/4 on size. Only 7490 still prefers 0.50.

**PREDICTION FAILED.** I expected w>1.0 to degenerate toward the
single-floor `pos/posctx` arm, which reads free0 **0.02**. It does not:
at w=2.0 the posctx term DOMINATES the loss (2.0 against zero's 1.0 and
negctx's 0.25) yet free0 stays **0.83-0.97**. So keeping the zero term
PRESENT AT ALL — even outweighed 2:1 — is qualitatively different from
dropping it. That is a sharper version of "single floors specialise"
than anything else in this panel: the zero term's job is not to dominate,
it is to exist.

**Revised session-best: 2,432 -> 341 = 7.1x** (or 302 if one is willing
to use a 2-seed-coverage setting). The posctx weight is the
second-largest contributor after lambda, and the 0.25 default was ~8x too
low.

**The upper bound is still unmeasured** — w=2.0 has not turned over, so
w=4-8 should be probed before treating 2.0 as optimal.

## R8 — the turnover is on `cf` (HELD OUT), not on the floors

Probing w=4 and w=8. L2 seed 386 at lambda=3e-3, cf held out throughout:

| w | n | free0 | freeM_dense | **cf** | pass |
|---|---:|---:|---:|---:|---|
| .05 | 371 | 0.874 | 0.514 | 0.601 | |
| .10 | 365 | 0.929 | 0.564 | 0.608 | |
| .25 | 395 | 0.884 | 0.645 | 0.618 | |
| .50 | 417 | 0.829 | 0.788 | 0.621 | |
| 1.0 | 447 | 0.709 | 0.821 | 0.663 | |
| **2.0** | 473 | 0.880 | 0.972 | **0.709** | **Y** |
| 4.0 | 484 | 0.888 | **0.996** | 0.601 | |

**freeM_dense rises monotonically all the way to w=4.0 (0.514 -> 0.996)
while cf PEAKS at w=2.0 and then drops.** The optimum is a genuine
interior maximum on the HELD-OUT metric. Tuning on the floors alone would
have pushed the weight to 4+ and produced a circuit that reconstructs the
seed beautifully and drives it ~15% worse.

**This is the concrete payoff of the held-out discipline.** Every floor
is improvable by training harder on floors; only cf, which nothing
trains on, shows the cost.

**CAVEAT — the cf turnover is SEED-SPECIFIC, not a law.** On seed 2927
w=4.0 passes comfortably (lambda=3e-3: n=372, cf **0.949**;
lambda=1e-2: n=282, cf **0.898**). So "cf peaks at w=2" holds on 386 and
not on 2927. What generalises is that cf is the metric that eventually
binds, not the particular weight at which it does.

**It is a (lambda, w) RIDGE, not a point.** w=8.0 passes at lambda=1e-3
(n=659, cf 0.889) but fails at 3e-3 (cf 0.699) and 1e-2 (cf 0.598) — at
looser lambda the circuit is big enough to absorb a heavy posctx term,
at tighter lambda it is not.

## FINAL L2 leaderboard (209 rows, 60 ALL-PASS)

| arm | seed | lambda | n | free0 | freeM_d | cf |
|---|---|---|---:|---:|---:|---:|
| **pos/tri1.0** | 2927 | 3e-3 | **302** | 0.994 | 0.810 | 0.866 |
| pos/tri2.0 | 2927 | 3e-3 | 341 | 0.968 | 0.901 | 0.943 |
| pos/tri.10 | 2927 | 1e-3 | 363 | 0.981 | 0.834 | 0.943 |
| pos/triple | 2927 | 1e-3 | 394 | 0.951 | 0.913 | 1.038 |
| pos/tri2.0 | 386 | 3e-3 | 473 | 0.880 | 0.972 | 0.709 |

**302 nodes** satisfies free0, freeM_dense, freeM_topk AND cf at once —
**8.0x** below the 2,432 this session started from.

**But the top seven slots are all seed 2927**, the most compressible
seed. At the same settings the per-seed range is ~300-770. So the claim
is "the most compressible L2 seed is 302, typical is 400-600", NOT "L2
circuits are ~300 nodes".


## R9 — lambda and the posctx weight trade off along a DIAGONAL

Seed 2927, the full (lambda, w) grid. `PASS` = ALL-PASS.

```
w        l1=1e-3      l1=3e-3      l1=1e-2
0.05     341          222             -
0.10     363 PASS     230             -
0.25     394 PASS     252          142
0.50     411 PASS     281             -
1.0      430 PASS     302 PASS        -
2.0      467 PASS     341 PASS        -
4.0      528 PASS     372 PASS     282 PASS
8.0      618 PASS     424 PASS        -
```

**Higher lambda requires higher w to survive.** At 1e-3 anything from
w=0.10 up passes; at 3e-3 you need w >= 1.0; at 1e-2 only w=4.0 makes it.
Sparsity pressure and posctx pressure have to be raised TOGETHER.

**This is why every earlier sweep found lambda=1e-2 uniformly fatal** —
they held w fixed (at 0 or 0.25) and moved lambda alone, which walks off
the ridge instead of along it. The smallest circuits live in the
tight-lambda / high-w corner and are only reachable by moving both dials.

Note the 142-node cell at (1e-2, w=0.25): the smallest circuit anywhere
in the panel, but it FAILS. It is a marker of how far the corner extends,
not a result.

**Best measured: n=282** (2927, lambda=1e-2, w=4.0 — free0 0.859,
freeM_dense 0.808, freeM_topk 0.990, cf 0.898). **8.6x** below the 2,432
this session started from, at a lambda where every earlier arm collapsed.


## R10 — PRACTICAL RECIPE: sweep lambda and w JOINTLY

Full L2 weight table (218 rows, 68 ALL-PASS), smallest passing circuit
per seed:

| arm | w | 386 | 2927 | 7019 | 7490 |
|---|---|---:|---:|---:|---:|
| pos/dual | 0 | 863 | 747 | 1,151 | 1,007 |
| pos/tri.05 | .05 | 816 | 758 | 1,146 | 1,004 |
| pos/tri.10 | .10 | 488 | 363 | 581 | 1,015 |
| pos/triple | .25 | 492 | 394 | 621 | 1,027 |
| pos/tri.50 | .50 | 512 | 411 | 641 | **645** |
| pos/tri1.0 | 1.0 | – | 302 | 670 | – |
| **pos/tri2.0** | 2.0 | **473** | 341 | **527** | 771 |
| pos/tri4.0 | 4.0 | 577 | **282** | 576 | – |
| pos/tri8.0 | 8.0 | 659 | 313 | – | – |

Per-seed optimum: **w=2.0 on two seeds, 4.0 on one, 0.50 on one.** No
single value works; the useful range is 0.5-4.0, far above the 0.25
default. Every seed improves **1.6-2.6x over dual**.

**RECOMMENDED PROCEDURE.** Do not fix either dial. Sweep
**w in {0.5, 2, 4} x lambda in {1e-3, 3e-3, 1e-2}** and take the smallest
ALL-PASS cell. Nine runs per seed, ~5 min at L2, and it finds circuits
2-3x smaller than any fixed setting. At depth shift the grid down:
**w in {0.05, 0.1, 0.5} x lambda in {1e-5, 1e-4}** (L9 optima were
w=0.05-0.10).

**ALWAYS score cf.** It is the only metric nothing trains on, and it is
what eventually binds (R8) — a floors-only selection walks past the
optimum.


## R11 — the weight response is NON-MONOTONE per seed

Seed 7490 (the seed that resisted every other intervention), smallest
ALL-PASS by weight:

    w      .05    .10    .25    .50    1.0    2.0    4.0    8.0
    n     1004   1015   1027    645      -    771    827      -

A sharp minimum at **w=0.50**, worse on BOTH sides. So "higher weight
helps stubborn seeds" — which I inferred when 0.50 first unlocked this
seed — is wrong. The response is non-monotone with a genuine interior
optimum that has to be SEARCHED for; it cannot be reached by climbing.

Combined with R9's (lambda, w) diagonal, the parameter surface is:
non-monotone in w, coupled to lambda, and with per-seed optima spanning
0.5-4.0. **There is no setting to reason your way to. Sweep the grid.**


## R12 — CORRECTION: seed 1639's free0 band is VACUOUS, not unsatisfiable

I twice wrote that L9 seed 1639 "cannot pass by construction" because its
zero-floor empty baseline is 14,248 against a_pos 21.1. **That is the
wrong direction.** free0 = (a_c - 14248) / (21.1 - 14248), so the
ALL-PASS band [0.8, 1.25] maps to a_c roughly in [-3,500, +2,900] — an
enormous range. The criterion is nearly **VACUOUS** on that seed, so
passing is EASIER there, not impossible. Its earlier failures were on
freeM_dense, not free0.

Consequence: **1639's ALL-PASS rows must be excluded from L9 summaries**
(e.g. tri.50 @1e-4 n=2,426, tri4.0 @1e-4 n=4,535) — one third of the
criterion carries no information there. Any seed with a_e0 >> 0 needs
the same treatment; check the empty-circuit baseline before trusting an
ALL-PASS.


## R13 — the L2 (lambda, w) DIAGONAL DOES NOT TRANSFER TO DEPTH

R9 found that at L2 tight lambda becomes survivable if w is raised with
it, and that the smallest circuits live in that corner. The obvious
worry was that L9's larger circuits were an unexplored-region artifact:
we had only tested w <= 0.25 there, and only at loose lambda.

Tested directly — w in {0.5, 2, 4} x lambda in {1e-4, 1e-3, 3e-3}, 4 L9
seeds, added to the existing w <= 0.25 grid. Seed 1639 excluded (R12).

| arm | w | 1283 | 2062 | 2766 |
|---|---|---:|---:|---:|
| pos/dual | 0 | 5,078 | – | – |
| **pos/tri.05** | .05 | **5,062** | 18,531 | 14,654 |
| pos/tri.10 | .10 | 17,116 | **18,356** | **14,369** |
| pos/triple | .25 | 16,701 | 19,142 | 14,388 |
| pos/tri2.0 | 2.0 | 5,498 | – | – |
| pos/tri.50, tri4.0 | | – | – | – |

**Every L9 ALL-PASS is at lambda 1e-5 or 1e-4. Nothing passes at 1e-3 or
3e-3 at ANY weight.** The corner is closed at depth.

**Why.** The third floor buys mean-floor faithfulness by SPENDING
zero-floor faithfulness (R6). At L2 free0 sits at 0.88-0.99 and can
afford it. At depth it cannot: seed 1283 at lambda=1e-3 reads
freeM_dense **0.885 (w=2.0) / 0.951 (w=4.0)** — perfectly good — while
free0 collapses to **0.33 / 0.30**. Raising w cannot unlock tight lambda
at depth because the currency is exactly what depth is short of.

**So the depth gap is REAL, not a search failure.** After a 7-weight x
4-lambda grid, L9 circuits floor at **5,062-18,356 nodes** against L2's
**282-645** — a 10-30x depth gap that survives the search that shrank L2
by 8.6x. Earlier in this thread I suspected the large deep-circuit
numbers were mostly search failure; for the floor/weight axis at least,
they are not.

Caveats: per-seed spread at L9 is still 3.6x (5,062 vs 18,356) and 1283
is an outlier in kind, not degree. **L8 has had NO floor work at all** —
that is the run that would say whether this is a smooth depth trend or
something specific to L9.


## R14 — free_amplitude: a new engine mode, a caught exploit, and a conceptual fork (in progress)

Daniel: "a mode where the mask can switch latents to 0, allow natural,
or ELEVATED — free range essentially." Built as `free_amplitude=True`:
alpha = softplus(psi) per latent on top of the gate, psi initialised at
softplus^-1(1) so step 0 reproduces gate-only exactly. Bounded via the
pos target (unlike `maximise`). 7 new tests, suite 118 -> 125.

**The test suite caught a reparameterisation exploit before any GPU
run:** L1 prices the gate m but the signal is m*alpha, so the optimiser
pushed m BELOW the membership threshold (cheap) and inflated alpha so
m*alpha ~ 1 — loss near zero, membership EMPTY, nothing priced. Fixed by
charging (1-m)*|alpha-1| at the same lambda, which makes honest
membership strictly cheaper than the sub-threshold route. Regression
test pins it.

**The runs surfaced a conceptual fork rather than a verdict.** On the
standard panel every amp arm underperforms its gate-only counterpart —
but the amp_stats show why: at lambda=1e-3 the mask keeps ~200 latents
with 42% ELEVATED (median alpha 1.05, p90 1.65). It genuinely used the
free range, and the standard eval then scores the SET at natural values,
discarding exactly what was learned. Supporting signal: pin0 (kept
latents at clean values — the closest existing metric to the amplitude
semantics) reads 0.87-0.99 on circuits whose free0 reads 0.06-0.29.

So the mode forces the question: is a circuit a SET (evaluate at natural
values — the field's convention and every metric here) or a set WITH
COEFFICIENTS (evaluate with alpha applied)? `amp_rescore.py` scores the
decision cells both ways; its output decides whether the ~200-node amp
circuits are real under their own semantics.


## R15 — free_amplitude VERDICT: 16/16 cells faithful on BOTH floors at 99-752 nodes

`amp_rescore.py`, all 4 seeds x 4 decision cells, each circuit scored
with its learned alphas applied (kept latents at alpha * live value,
non-members at the floor):

| setting | n range | ampF0 | ampFMd | nat free0 (same sets) |
|---|---|---|---|---|
| dual @1e-4 | 603-752 | 0.97-1.03 | 1.05-1.18 | 0.47-0.55 |
| dual @1e-3 | 124-229 | 0.92-1.01 | 0.94-1.18 | 0.06-0.39 |
| tri.10 @1e-3 | 129-225 | 0.99-1.03 | 1.04-1.11 | 0.11-0.28 |
| **tri2.0 @3e-3** | **99-165** | 0.95-1.06 | 0.99-1.06 | 0.04-0.19 |

**Every cell — 16 of 16 — lands in [0.8, 1.25] on BOTH floor conventions
simultaneously.** Median ampF0 1.009, median ampFMd 1.071. The smallest
is **99 nodes** (seed 7490, tri2.0 @3e-3: ampF0 1.056, ampFMd 1.006 — on
the seed that resisted every gate-only intervention, whose best gate-only
ALL-PASS was 645).

Amplitudes are moderate: median alpha ~1.0-1.07 everywhere, p90 1.2-2.5.
Compensation, not explosion — the pos target keeps it bounded exactly as
designed.

**Verdict.** As a bare SET the amp circuits are poor (nat free0
0.04-0.55) — the standard panel is the wrong lens for them. As a set
WITH COEFFICIENTS they are the most faithful compact objects measured in
this project: both floors at once at 99-229 nodes, where the best
gate-only circuits need 394-645 for the same band and the session's
starting recipe needed 2,432. **~4x below the best gate-only result,
~24x below the starting recipe** — at the cost of a richer circuit
definition (membership + per-latent coefficient vector).

**Open before this is a headline:** cf and sup have not been evaluated
under amp semantics (spawned as a follow-up task); the mode is untested
at depth; and the coefficient vector must be reported as part of the
object — a 99-node amp circuit is NOT comparable to a 99-node set.


## R16 — THE NULL DEFENDS ampF0: random sets with fitted amplitudes CANNOT fake faithfulness

Daniel's challenge: "is ampF0 a scientifically defendable eval?" The
worry is degrees of freedom — alpha is fitted to reproduce the seed, and
the eval asks whether it reproduces the seed. The decisive null
(`amp_null.py`): draw RANDOM same-size sets from the LIVE pool, freeze
the gates open on them (support= mechanism, lambda=0 so nothing prunes),
fit ONLY the amplitudes with the identical objective/floor/steps/lr, and
score identically.

| depth | seed | discovered | random x3 |
|---|---|---:|---|
| L9 | 1283 | **1.011** (n=283) | 20,648 / 19,812 / 18,016 |
| L9 | 2062 | **0.979** (n=596) | 1,037 / 1,104 / 1,206 |
| L2 | 386 | **1.006** (n=206) | 20.9 / 21.2 / 20.8 |

**Every random draw fails; every discovered circuit sits at ~1.0.** The
margin varies by seed — L2/2927 is the narrowest (discovered 0.922 vs
random 3.6-3.9, a ~4x miss rather than orders of magnitude) — but no
draw anywhere lands within the band, and random's ampFMd is EXACTLY 0.0
on every draw at both depths, so on the mean floor the separation is
categorical everywhere.

**Why random EXPLODES rather than reading 0 — and why that is the strong
version of the null.** ampF0 reads the PRE-activation, which has no
top-k censoring. Zeroing ~10^5-10^6 live latents guts the residual
stream; LayerNorm renormalises the gutted stream so whatever the random
latents inject is amplified enormously downstream, and gradients through
causally-uncoupled latents are chaotic, so the fit wanders (alpha_max
5.7-15) with no minimum near the target. So the amplitude mechanism has
enormous POWER over the seed (10^1-10^4 x) and still cannot hit 1.0:
reaching the target per-sequence requires CONTROL — latents whose effect
on the seed is systematic — which only selected circuits provide. If
random had read ~0 the null would be trivially satisfied ("amplitudes
can't conjure signal"); instead it shows they can move the seed hugely
and STILL can't fake faithfulness.

**Verdict: ampF0 survives its null.** The 99-1,590-node amp circuits at
both depths are genuine selection, not fitting artifact. Remaining
before headline use: cf/sup under amp semantics, and the framing
obligation that a set+coefficients is a different object from a set.


## NAMING (settled with Daniel, 2026-08-05)

The winning configuration — `objective="pos"` + `mask_floor_source=
"triple"` + `free_amplitude=True` — is the **TRI-AMP MASK** (codebase
name). The object it produces is a **WEIGHTED CIRCUIT**: membership set
plus per-latent coefficient vector, reported together always. "abl-mask"
remains the name of the gate-only pos/dual ancestor. Deliberately no
"abl-" in the new name: the multi-floor training means it is not tied to
one ablation semantics, which is its point.


## R17 — cf/sup for tri-amp weighted circuits (the last panel column)

Design: `sup` unchanged (removal does not involve coefficients);
`cf_bare` = the standard evaluator on the membership; `cf_amp` = members
set to alpha_i * pin_i in the LIVE negctx stream, alphas exactly as
trained, NO refit on the negatives (held-out by construction); drive
null = random same-size sets from the nonzero-pin pool with
identically-fitted alphas.

| depth | seed | n | cf_bare | cf_amp | sup | random cf_amp x2 |
|---|---|---:|---:|---:|---:|---|
| L2 | 386 | 217 | 0.643 | **1.253** | 0.989 | 0.051 / 0.216 |
| L2 | 2927 | 142 | 0.550 | **0.896** | 1.002 | 0.500 / -0.047 |
| L2 | 7019 | 225 | 0.605 | **1.171** | 1.000 | 0.017 / 0.047 |
| L9 | 1283 | 242 | 0.286 | 0.247 | 1.023 | 0.011 / 0.028 |
| L9 | 2062 | 623 | **0.951** | 1.501 | 1.000 | -0.020 / 0.057 |
| L9 | 2766 | 653 | **0.898** | 1.377 | 0.996 | **0.895** / -0.038 |

**Necessity: clean everywhere.** sup 0.99-1.02 on 6/6 — the weighted
circuits are necessary at both depths, full stop.

**Drive at L2: the coefficients ARE the drive.** cf_bare ~0.6 on all
three; the trained alphas lift it to 0.90-1.25 — a causal task they were
never fitted on. Held-out transfer, the strongest evidence yet that the
coefficient vector is part of the mechanism rather than a reconstruction
trick.

**Drive at L9: heterogeneous, and honest reporting required.**
- 2062/2766: the BARE membership already drives (0.90-0.95) and alphas
  overshoot (1.38-1.50). Drive was never missing.
- 1283: neither bare nor amp drives (0.25-0.29) despite perfect
  reconstruction (ampF0 1.01) and necessity — a genuine
  closure-without-drive weighted circuit. The 10x margin over its null
  says the 0.25 is real, just small.

**The drive null is HEAVY-TAILED and must be reported per-seed with more
draws.** 10 of 12 draws sit in [-0.05, 0.22], but two land high (0.50 on
L2/2927; 0.895 on L9/2766 — that one MATCHES the discovered bare cf, so
on that seed generic injection mass drives the seed and cf carries no
selection information). Draw-to-draw variance within one seed spans the
whole range (0.895 vs -0.038). Two draws per seed is not enough to
estimate the null; treat these as existence proofs of the tail, and run
>=10 draws before quoting any per-seed cf margin.

**Panel status for the tri-amp weighted circuit, after R15-R17:**
reconstruction (both floors) VALIDATED with null at both depths;
necessity VALIDATED 6/6; drive VALIDATED at L2 (with the alphas doing
the work), MIXED at L9 (2/3 already-driving, 1/3 closure-only), drive
null needs more draws. That is the full evidence state for the paper.

## R18 — the amp-cf overshoot is NOT under-training: two-phase amplitude
## training buys nothing consistent (amp_twophase.py, amp_twophase_c8.jsonl)

Question (Daniel, 2026-08-06): the alphas train in the same 400 joint
steps as the gates — are they under-trained, and would a second,
amplitude-only phase on the frozen membership calibrate them better
(ampF0/ampFMd closer to 1, cf_amp overshoot shrinking)?

Design: per seed at the decision cell, phase 1 = standard tri-amp
discovery (400 joint). Phase 2 = freeze the membership (support=,
theta=+40, lambda=0, binarize off) and train amplitudes ALONE for
{100, 200, 400, 800} steps from alpha=1 — the engine has no psi
warm-start, so phase 2 is a refit on the fixed set, which is exactly
the null's fitting procedure (a feature: same fitter, no confound).
Scoring verbatim from amp_null.py / amp_cfsup.py.

L2 (c8, triple w=0.10 @ 1e-3), cf_amp with alpha_p90 in parens:

  seed  joint400        amp100          amp200          amp400          amp800
  386   1.253 (1.56)    1.370 (1.67)    1.391 (1.74)    1.383 (1.92)    1.341 (1.92)
  2927  0.896 (1.67)    0.980 (1.62)    0.995 (1.93)    1.060 (1.99)    1.058 (2.10)
  7019  1.171 (1.87)    1.157 (1.86)    1.143 (1.97)    1.162 (2.15)    1.176 (2.16)

ampF0/ampFMd: within a few percent of 1 in EVERY arm — reconstruction
was already converged at 400 joint steps. All movement is noise-level
and directionless.

Findings:

1. **No arm beats joint-400 across seeds.** 386 is best left alone
   (every refit overshoots harder, 1.25 -> 1.34-1.39); 2927 looks
   best at short refits (0.98-0.995) but that is a crossover through 1
   on its way up; 7019 barely moves. The drive miscalibration is
   seed-dependent structure, not a shared under-training artifact.
2. **alpha_p90 inflates monotonically with amplitude-training budget**
   (1.56-1.87 at joint -> 1.92-2.16 at 800) while reconstruction stays
   pinned at ~1. The objective (posctx reconstruction under floors)
   does not constrain the drive axis; more training drifts along that
   unconstrained direction rather than calibrating it. This is the
   mechanism behind the overshoot, and it is a property of the
   OBJECTIVE, not the step count.
3. **Determinism check passed**: the repeated phase-1 rows across the
   two runs are bit-identical (386: 0.9912/1.0353/1.2529/1.5598 twice).

Practical consequence: keep the single-phase 400-joint schedule. If
drive calibration ever matters enough to fix, the fix is a drive-aware
term in the objective (e.g. a negctx-injection penalty), not more
amplitude training.

L9 (c29, triple w=0.05 @ 1e-3, seeds 1283/2062/2766, 400/800 arms):

  seed  arm         n     ampF0    ampFMd   cf_amp   a_p90
  1283  joint400    242   0.897    0.790    0.247    2.03
  1283  amponly400  242   1.036    1.125    0.369    2.03
  1283  amponly800  227   1.054    1.215    0.352    2.21
  2062  joint400    623   0.961    0.966    1.501    2.18
  2062  amponly400  623   0.977    1.026    1.387    2.32
  2062  amponly800  620   0.939    0.812    1.359    2.36
  2766  joint400    653   1.039    1.263    1.377    2.04
  2766  amponly400  653   0.944    1.038    1.559    2.08
  2766  amponly800  642   1.022    1.414    1.654    2.40

L9 confirms the L2 verdict, with three depth-specific additions:

4. **Still no consistent gain.** 2062 improves slightly (1.50 ->
   1.36-1.39), 2766 gets WORSE (1.38 -> 1.56-1.65), and 1283's
   closure-without-drive persists (0.25 -> 0.35-0.37: the refit lifts
   drive a little but nowhere near driving — the missing drive is in
   the MEMBERSHIP, not the coefficients).
5. **Long amplitude-only training hurts holdout reconstruction.** At
   800 steps ampFMd degrades badly on 2 of 3 seeds (2062: 0.97 ->
   0.81; 2766: 1.26 -> 1.41) — the refit overfits the zero-floor term
   at the expense of the mean-floor one.
6. **The refit silently prunes via amplitude**: n drops at 800 steps
   (242->227, 623->620, 653->642) as some members' alphas collapse to
   ~0 with no L1 pressure at all. Membership is only frozen at the
   gate level; the amplitude can still delete.

VERDICT (both depths): the amp-cf overshoot is a property of the
objective's blind spot, not the training budget. Single-phase 400
joint remains the house schedule; two-phase is retired.

## R19 — the step-count response surface: steps is a sparsity dial that
## SPENDS drive calibration (amp_stepsweep.py, amp_stepsweep_c8/c29.jsonl)

Motivation (Daniel, 2026-08-06): steps is the main compute lever (a
tri-amp step is 3 floored forwards); map evals/size/time over the
joint budget. Full independent discoveries per arm at the decision
cells; scoring as R15-R17. Wall-clock is linear at ~0.06 s/step (L2:
7s at 100 steps, 27s at 400, 102s at 1600).

L2 (steps 50-1600): at 50 steps NOTHING prunes (n = the full 327,680
support; onset of pruning is between 50 and 100). From 100 up:

  seed   metric   100     200     400     800     1600
  386    n        612     327     217     125     105
         cf_amp   1.206   1.266   1.253   1.208   1.125
  2927   n        420     236     142     87      48
         cf_amp   1.065   0.911   0.896   0.804   0.584
  7019   n        644     377     225     162     117
         cf_amp   1.150   1.167   1.171   1.215   1.350

L9 (steps 100-800):

  seed   metric   100     200     400     800
  1283   n        1378    472     242     210
         cf_amp   0.877   0.570   0.247   0.451
  2062   n        2216    936     623     515
         cf_amp   1.180   1.387   1.501   1.650
  2766   n        1916    932     653     499
         cf_amp   1.174   1.307   1.377   1.490

ampF0 is ~1 at EVERY budget from 100 up, both depths (ampFMd is
noisier at L9 low budgets, 1.25-1.64 at 100-200 steps). alpha_p90
inflates monotonically with steps everywhere (~1.2-1.4 at 100 ->
2.5-3.3 at the top), extending R18's amplitude drift to the joint
trajectory.

Findings:

1. **Reconstruction converges by 100 steps.** Everything after that
   buys sparsity, not fidelity. n(steps) roughly halves per doubling
   through 400, then decelerates hard (105 vs 125 for 2x compute).
2. **Drive calibration is best at the SHORTEST budget and decays
   monotonically with steps** in 5 of 6 seeds. At 100 steps every
   L9 seed drives near 1 (0.88 / 1.18 / 1.17); by 800 the survivors
   overshoot 1.5-1.65.
3. **R17's "closure-without-drive" on 1283 is a TRAINING-BUDGET
   ARTIFACT.** Its 1,378-node 100-step circuit DRIVES (0.88). The
   drive-carrying members exist and are progressively pruned
   (0.88 -> 0.57 -> 0.25 across 100/200/400 steps) because posctx
   reconstruction under floors does not need them — the objective's
   blind spot acting on MEMBERSHIP, not just amplitudes. R17's
   per-seed heterogeneity is partly a step-budget story.
4. Efficiency recipe: faithful-but-big at 100-200 steps for 2-4x less
   compute; smallest sets at 800+ at the cost of inflated alphas and
   degraded/dying drive. 400 is a reasonable middle and is now a
   MEASURED choice, not a frozen convention.

Caveat: (lambda, w) stayed at the 400-step calibration throughout, so
low-step arms are also low-pruning-pressure arms. The budget
equivalence test (steps=100 at 2-8x lambda vs steps=400 at 1e-3,
including membership Jaccard) is running as amp_fastlam.py.

## R20 — budget equivalence: steps=100 at higher lambda is 4x faster,
## better drive-calibrated, but finds a DIFFERENT circuit
## (amp_fastlam.py, amp_fastlam_c8.jsonl, amp_fastlam_members_c8.jsonl)

Test (Daniel): can steps=100 with raised lambda reproduce the
steps=400 @ 1e-3 circuit at 1/4 the compute? lambda in {2,4,8}e-3 vs
the 400-step reference, memberships saved for Jaccard. L2, 3 seeds.

  seed  arm            n     ampF0   ampFMd  cf_amp  a_p90  secs
  386   ref s400@1e-3  217   0.99    1.04    1.25    1.56   29.4
        s100@2e-3      466   0.99    0.94    1.07    1.38    9.5
        s100@4e-3      331   0.93    0.88    1.03    1.32    7.4
        s100@8e-3      260   1.02    0.80    1.03    1.40    7.5
  2927  ref s400@1e-3  142   1.02    1.10    0.90    1.67   29.0
        s100@2e-3      310   1.07    1.17    1.10    1.25    8.1
        s100@4e-3      201   1.03    1.09    0.83    1.33    8.1
        s100@8e-3      112   0.80    0.84    0.73    1.38    8.6
  7019  ref s400@1e-3  225   0.99    1.11    1.17    1.87   29.4
        s100@2e-3      484   0.92    1.04    1.11    1.32    7.9
        s100@4e-3      276   0.99    1.04    1.05    1.39    7.3
        s100@8e-3      159   0.96    0.99    1.06    1.48    7.1

Membership vs reference: Jaccard 0.40-0.53 everywhere. At 2e-3 the
short-budget circuit CONTAINS 91% of the reference (it is roughly
"ref + extras"); at 8e-3 containment falls to 56-76% — at matched
size, a QUARTER TO A HALF of the reference members are swapped out.

Findings:

1. **The (steps x lambda) product under-delivers**: 4x lambda at 1/4
   steps gives n ~1.4-1.5x the reference, not parity; parity needs
   ~6-8x lambda. Pruning pressure is sublinear in lambda (the anneal
   schedule, which stretches with steps, does part of the pruning
   work).
2. **Short-budget circuits keep the drive advantage even at high
   lambda**: cf_amp 1.03-1.11 in 7 of 9 arms (vs 0.90-1.25 for the
   references), and alpha_p90 stays 1.25-1.48 (vs 1.56-1.87). The
   drive/alpha degradation of R19 tracks STEPS, not sparsity.
3. **But it is a different circuit.** Jaccard ~0.5 at matched size
   means steps-pruning and lambda-pruning remove DIFFERENT latents.
   Consistent with R19.3: step pruning eats drive-carriers; lambda
   pruning at short budgets keeps them (see cf_amp).
4. **Failure edge**: 2927 @ 8e-3 breaks (ampF0 0.80, cf 0.73) —
   over-pruned below its floor at this budget. The usable lambda
   ceiling is seed-dependent.
5. **Efficiency verdict**: s100 @ 4e-3 is the value point — ~4x
   faster wall-clock (7-9s vs 29s/seed), all metrics in band, better
   drive and alpha calibration, at ~1.4x the node count. If
   compactness is the goal, 400 steps remains the way to get it; if
   fidelity-per-second (or drive) is the goal, short-and-sharp wins.

Open: whether the same holds at L9 (where drive decay with steps is
steepest), and whether ~6x lambda at 100 steps reaches parity without
hitting the 2927-style floor.

R20 addendum — L9 (amp_fastlam_c29.jsonl): the short-and-sharp recipe
only HALF transfers to depth.

  seed  arm            n     ampF0   ampFMd  cf_amp  a_p90  secs
  1283  ref s400@1e-3  242   0.90    0.79    0.25    2.03   88
        s100@2e-3      1269  0.99    1.35    0.95    1.46   24
        s100@4e-3      617   0.93    1.45    0.70    1.42   23
        s100@8e-3      481   0.78    1.16    0.89    1.43   23
  2062  ref s400@1e-3  623   0.96    0.97    1.50    2.18   86
        s100@2e-3      1562  0.99    1.59    1.35    1.45   23
        s100@4e-3      1046  0.94    1.31    1.57    1.57   23
        s100@8e-3      531   0.86    0.00    1.89    1.43   24
  2766  ref s400@1e-3  653   1.04    1.26    1.38    2.04   87
        s100@2e-3      1284  0.90    1.69    1.29    1.50   25
        s100@4e-3      855   1.10    1.07    1.61    1.58   23
        s100@8e-3      574   0.96    0.06    1.79    1.53   23

- **What transfers**: 3.7x wall-clock (23-25s vs 86-88s), low alpha
  inflation (p90 1.4-1.6 vs 2.0-2.2), and the 1283 rescue (s100@2e-3
  drives at 0.95 with n=1269 where the ref sits at 0.25).
- **What does NOT**: at L9, drive decays with SPARSITY however it is
  bought — raising lambda at 100 steps walks cf_amp away from 1 just
  like raising steps did (2062: 1.35 -> 1.57 -> 1.89; 2766: 1.29 ->
  1.61 -> 1.79). R19.3 refines to: PRUNING eats drive-carriers at
  depth; short budgets simply haven't pruned as much yet.
- **Size parity is not reachable at 100 steps at L9**: the 8e-3 arms
  break (ampFMd 0.00/0.06 — the mean-floor reconstruction collapses
  to the empty-circuit baseline; 1283 ampF0 0.78). Jaccard vs ref
  0.17-0.49, ref-coverage 0.57-0.92 falling with lambda, as at L2.
- **L9 value point**: s100 @ 2e-3 — 3.7x faster, best-in-class drive,
  ~half the nodes of s100 @ 1e-3, but still 2-2.6x the reference n.
  ampFMd is systematically high (1.35-1.69) on short-budget L9
  circuits; quote it alongside.

Combined R19+R20 verdict on steps: compactness (long budgets or high
lambda) and drive calibration are in TENSION at depth, full stop. The
practical schedule menu: 400 joint @ 1e-3 for the smallest closure
sets; 100 @ 4e-3 (L2) / 100 @ 2e-3 (L9) for 4x-faster, drive-faithful,
larger circuits. Choose per claim, and report which was used.

## R21 — the no-zero control: amplitudes CANNOT substitute for the zero
## term (amp_pnfloor.py, amp_pnfloor_c8/c29.jsonl; "pn" floor added to
## the engine 2026-08-06, 130 tests green)

Question (Daniel): with free amplitudes, does the zero floor still earn
its place — training can set alpha->0, so is "member at alpha 0" not
the same as "excluded under zero fill"? The confusion the test resolves:
alpha exists only for MEMBERS; the floor defines what NON-members do.
No alpha setting can show the mask what its set does ALONE.

Mode: mask_floor_source="pn" = negctx promoted to the primary 1.0 slot
+ posctx at the triple weight, NO zero term. Same seeds, cells, scoring
as the triple joint400 rows (amp_stepsweep). Results (pn vs triple):

  seed   pn: n / ampF0 / ampFMd / cf     triple: n / ampF0 / ampFMd / cf
  L2 386   132 / 1.53   / 1.06 / 1.46      217 / 0.99 / 1.04 / 1.25
  L2 2927  122 / 2.74   / 1.13 / 0.91      142 / 1.02 / 1.10 / 0.90
  L2 7019  174 / 1.06   / 1.11 / 1.08      225 / 0.99 / 1.11 / 1.17
  L9 1283  192 / 1681.1 / 1.05 / 0.39      242 / 0.90 / 0.79 / 0.25
  L9 2062  261 / 36.0   / 0.99 / 1.39      623 / 0.96 / 0.97 / 1.50
  L9 2766  224 / 97.6   / 0.97 / 1.07      653 / 1.04 / 1.26 / 1.38

Findings:

1. **The zero term is load-bearing, catastrophically so at depth.**
   ampF0 without it: 36-1,681 at L9, 1.5-2.7 at L2 (2 of 3 out of
   band). The zero floor stays.
2. **With amplitudes the failure signature INVERTS vs gate-only R1**:
   negctx-trained gates collapsed (free0 = 0.0 — members alone can't
   reach top-k); negctx-trained amplitudes EXPLODE (ampF0 >> 1). The
   alphas are calibrated against the negctx background; delete the
   background and the amplified set massively overdrives. Same blind
   spot, opposite sign, and far more dangerous — a collapse fails the
   band loudly, an explosion can masquerade as "drive".
3. **The escape case proves the mechanism**: L2/7019 passes (1.06)
   and its negctx barely excites anything (a_base 0.051) — its negctx
   floor IS nearly a zero floor. The zero term is redundant exactly
   when the negctx floor degenerates to it.
4. **pn circuits are smaller (122-261 vs 142-653)** — without the
   zero term the background carries reconstruction and L1 prunes
   members that zero-sufficiency needs. Small-by-leaning-on-the-floor,
   the same genus as the maximise exploit (R7-R9).
5. **ampFMd ~1 everywhere (0.97-1.13)**: the trained semantics is won,
   the untrained one is lost — the cleanest yet of the now-six
   demonstrations of "you win the metric whose semantics you train".

Verdict: zero + negctx (+ posctx) all pull their weight; triple stands
as the production floor. "pn" stays in the engine as the documented
ablation control.
