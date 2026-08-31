# Inhibitor-mask — the "raise" objective (2026-08-02)

Daniel's proposal, built and validated: the MIRROR of abl-mask. Where
"pos" says *keep the seed at natural, pay for every latent you KEEP*
(smallest sufficient support), "raise" says *push the seed ABOVE natural
on POSCTX, pay for every latent you SILENCE* (smallest effective BRAKE
set).

    L = mse( pre_pos(m), gamma * a_pos ) + lambda * sum(1 - m)

The L1-on-edits is what creates the tension Daniel identified: a latent
with no effect on the seed costs lambda to silence and buys nothing, so
it is evicted — the dictionary's inert majority never enters. An
ACTIVATOR would LOWER the seed if silenced, so it is penalised twice and
is never recruited. Only genuine brakes survive.

Engine: `objective="raise"` + `raise_gamma` (validated > 1) in
run_learned_mask; scores delivered NEGATIVE (inhibitors), as for negctx.
5 new unit tests incl. the three-way tension check (suppressor found,
inert latent NOT recruited, driver NOT recruited); mask suite 90 green.
Runs: 8 seeds x 3 lambdas, gamma 1.5, early stopping built in (every 25
steps, best snapshot kept — the cf-mask dynamics lesson applied from the
start). rows_v1/v2.jsonl, summary_tables.md, members_*.jsonl.gz.

## VERDICT: it works, bidirectionally, on every seed

**Silencing the found set RAISES the seed by +10% to +56% (median +0.39
at lambda 1e-3), while a size-matched RANDOM silencing moves it by
0.000 on 23 of 24 arms** (the one exception is -0.004). The tension
holds: the objective returns 6-116 latents out of ~1.4M, not the inert
99%.

**The mirror test passes: firing them harder SILENCES the seed.**
Amplifying the found members to 4x their own posctx values drives the
seed to ZERO (drop = 1.000) in 11 of 24 arms and >= 0.71 in 20 of 24
(median 0.94-0.98 by lambda). Two independent directions — remove them
and the seed rises, drive them and the seed dies — on sets discovered
by a single objective that only ever saw the first direction.

| lambda | median n | median raise | median rand | median drop x2 | median drop x4 |
|---:|---:|---:|---:|---:|---:|
| 3e-4 | 58 | +0.440 | 0.000 | +0.332 | +0.957 |
| **1e-3** | **34** | **+0.389** | **0.000** | +0.416 | +0.937 |
| 3e-3 | 23 | +0.302 | 0.000 | +0.308 | +0.979 |

RECOMMENDED OPERATING POINT: lambda 1e-3, gamma 1.5, early-stopped —
~34 latents, +39% raise, near-total silencing under 4x amplification.

## Method note: phi-sup is the WRONG instrument here (v1 -> v2 fix)

v1 read phi-sup and got -0.03 to -0.38 on every seed. That is not a
failure: phi-sup injects inhibitors at their NEGCTX values, and these
latents are already PRESENT and braking on posctx, so setting them to a
(lower) negctx level RELEASES the brake and raises the seed — the same
phenomenon with the sign flipped. The correct mirror for a
present-and-braking population is AMPLIFICATION (v2's drop_b2/drop_b4),
which is what the table above reports. Kept in the rows for the record.

## FINDING — attribution signs disagree with causal braking on ~half

Only 23-77% of the learned brakes are restoration's attribution-signed
INHIBITORS; the rest are signed ACTIVATORS. Splitting the raise by
label (T3, lambda 1e-3) shows both halves brake, and **the
"activator"-labelled half brakes HARDER on 5 of 8 seeds** (L4-mlp: 0.228
from the R_act half vs 0.063 from the R_inh half). Consistent with the
2026-07-24 gradient-observability finding (sign is state-dependent to
near-independence) and with D2.2's warning that "inhibitor" is a
functional label, not an intrinsic property. An intervention-native
definition of inhibitor is therefore NOT redundant with the
attribution-sign one — it is a different, causally validated object.

## Structure

- Brake sets are SMALL and roughly depth-flat (13-116 at 1e-3, no clear
  growth with depth) — like drivers, unlike closure.
- attn seeds resist 2x amplification (L3 -0.11, L9 -0.03) but fall at 4x
  (0.76, 0.77): their brakes need to be driven hard, consistent with
  attention seeds being the suppression-dominated kind (D2.2).
- L0 is the weakest cell (raise +0.10 to +0.20, drop4 0.27-0.48): a
  2-site seed has little upstream to brake with.

## Caveats

- gamma = 1.5 fixed; the achievable raise is not a frontier yet (a gamma
  sweep would give the "how far can it be pushed" curve).
- raise/drop are measured on the seed's POST-top-k activation, so part
  of a raise can be top-k competition (silencing a neighbour lets the
  seed in) rather than direct suppression — a real causal effect on the
  measured quantity, but a different mechanism worth separating with a
  pre-activation read.
- One split per seed; no resampling error bars (D4.2-style split-half
  would say how identifiable a brake set is).

## FULL EVAL MATRIX (eval_matrix.jsonl / .py — lambda 1e-3 arm, 8 seeds)

Every standing eval, plus a size-matched random control on every column:

| metric | inhibitor circuit (median) | random control |
|---|---:|---:|
| free0 | **0.000** | 0.000 |
| freeN_topk | 0.003 | 0.006 (matched — see note) |
| cf | **0.000** | 0.000 |
| sup | -0.151 | 0.000 |
| **raise** (silence -> seed up) | **+0.354** | +0.000 |
| **drop x4** (amplify -> seed down) | **+0.819** | +0.000 |

NOTE on freeN: the three deep seeds show freeN 0.09-0.21 for the brake
set, but the RANDOM control scores 0.11-0.26 on the same seeds — that
is the negctx floor's own baseline, not circuit content. Read as zero.

**The brake circuit scores ZERO on every eval the project already had,
while being strongly, bidirectionally causal on its own two.** This is
not a failure — it is the sharpest statement yet of the law the campaign
keeps rediscovering ("you win the metric whose semantics you train"),
and it now has THREE mutually blind objects on the same seeds:

| object | free0 | cf | raise/drop |
|---|---|---|---|
| abl-mask closure (D3.6) | 0.7-1.0 | ~0 | untested |
| AMPC / cf-mask drive | 0.000 | ~1.0 | untested |
| inhibitor-mask brake | 0.000 | 0.000 | +0.35 / +0.82 |

METHODOLOGICAL CONSEQUENCE: the standing eval suite is INCOMPLETE — it
has no column that can see suppression-side structure. Scored on the
existing matrix alone, this algorithm looks like it produces garbage
(every column 0.000, indistinguishable from random); only the two
brake-native interventions reveal that it is finding real, specific,
causally-validated structure. Any future "generalist" claim must
therefore be judged on a matrix that includes a suppression column.

## BRAKE vs CLOSURE OVERLAP (Daniel's question, 2026-08-02)

Are the learned brakes among the latents abl-mask SETS TO ZERO, or among
those it KEEPS? Theory predicts KEEPS: abl-mask's objective is MSE to
NATURAL, so silencing a brake pushes the seed ABOVE natural — also a
loss. It must retain brakes, not prune them.

Confirmed, decisively (lambda 1e-3 arm, vs the D3.6 abl-mask member
lists on the same seeds):

| seed | brakes | inside abl-mask's KEPT set | base rate | enrichment |
|---|---:|---:|---:|---:|
| L0 resid | 13 | 77% | 0.79% | 98x |
| L2 resid | 20 | 70% | 0.58% | 121x |
| L3 attn | 34 | 97% | 1.66% | 58x |
| L4 mlp | 32 | 78% | 0.80% | 98x |
| L6 resid | 35 | 57% | 0.89% | 64x |
| L8 resid | 70 | 77% | 1.37% | 56x |
| L9 attn | 50 | 62% | 1.47% | 42x |
| L11 resid | 53 | 55% | 0.94% | 58x |
| **median** | | **73%** | **0.92%** | **61x** |

**Median 73% of brakes are KEPT by abl-mask, against a 0.9% base rate —
42-121x enrichment.** The closure circuit is therefore NOT a support
set: it is a BALANCE set, containing both the latents that drive the
seed and the latents that hold it down, because reproducing the natural
level requires both. This also explains structurally why abl-mask is
role-blind (a scalar keep-probability cannot distinguish the two
populations it is obliged to retain) and why the inhibitor-mask is not
redundant with it: same members, opposite functional role, and only the
brake-native objective can tell them apart.

The ~27% of brakes OUTSIDE the closure are the ones abl-mask could
afford to drop — candidates for "redundant brake" (their removal is
compensated) vs the retained "load-bearing brake". Untested split.

## REMOVING THE HIDDEN BRAKES FROM THE CLOSURE (ablmask_minus_brakes.*)

Daniel's follow-up: abl-mask delivers every member as a SUPPORT, so the
brakes sit in its activator set mislabelled. What happens to the closure
circuit's evals if we delete exactly those members? Balance-set theory
predicts OVERSHOOT (free0 > 1). Result: **it depends on depth, and the
deep answer is the opposite of the prediction.**

| seed | MF free0 | MF-brakes | MF-random (control) | removed |
|---|---:|---:|---:|---:|
| L0 resid | 0.971 | **1.087** (overshoot) | 0.957 | 10 |
| L2 resid | 0.955 | **1.050** (overshoot) | 0.940 | 15 |
| L3 attn | 0.873 | **1.503** (overshoot) | 0.857 | 37 |
| L4 mlp | 0.867 | **0.261** (collapse) | 0.823 | 28 |
| L6 resid | 0.855 | **0.000** (collapse) | 0.803 | 20 |
| L8 resid | 0.959 | **0.101** (collapse) | 0.859 | 54 |
| L9 attn | 0.882 | **0.000** (collapse) | 0.912 | 32 |
| L11 resid | 0.883 | **0.000** (collapse) | 0.683 | 28 |

The random control removes the SAME NUMBER of members and barely moves
free0 (0.68-0.91) — so both effects are specific to the brake set.

**SHALLOW (L0-L3): brakes behave as brakes.** Deleting 10-37 of them
pushes the circuit ABOVE natural — free0 1.05-1.50, exactly the
balance-set prediction. L3-attn overshoots by 50%.

**DEEP (L4-L11): the SAME latents are load-bearing SUPPORTS.** Deleting
20-54 members out of 7k-16k collapses the circuit to free0 0.00-0.26 —
the seed falls out of its SAE's top-k entirely (a_circuit reads exactly
0.000 at L6/L9/L11; note the post-top-k read is CENSORED there, so the
magnitude below the cutoff is unknown, but the direction is unambiguous
and the random control does not do it).

**ROLE IS CONTEXT-DEPENDENT, NOT INTRINSIC.** The same latent brakes the
seed in the NATURAL stream (silence it -> seed rises 35%) and supports
the seed in the ABLATED stream (delete it from the circuit -> seed dies).
Both measurements are causal, held-out, and control-checked; they simply
interrogate different states. This is the circuit-membership-level
version of the 2026-07-24 gradient-observability finding (sign is
state-dependent to near-independence), and it is the strongest evidence
yet that "activator" and "inhibitor" are properties of a
(latent, context, intervention) TRIPLE rather than of a latent.

Consequences: (a) any role label in the paper must name its state;
(b) the closure circuit is not merely a balance set at depth — it is a
tightly-coupled set whose members' apparent sign flips with what else is
alive; (c) the ~27% of brakes OUTSIDE the closure and the deep collapse
together suggest closure membership is chosen for JOINT sufficiency, not
per-latent contribution.

Follow-up worth doing: rerun the deep cells with the PRE-activation read
(preact=True) to uncensor the collapse magnitude.

## !! PREACT CORRECTION to the section above (ablmask_minus_brakes_preact) !!

The post-top-k read CENSORS at 0.000, and two of the five "deep
collapses" were that artefact. Raw pre-activations (natural in parens):

| seed | MF | MF-brakes | natural | post-top-k said | TRUTH |
|---|---:|---:|---:|---|---|
| L0 resid | 2.09 | 2.33 | 2.16 | overshoot | overshoot |
| L2 resid | 6.00 | 6.59 | 6.25 | overshoot | overshoot |
| L3 attn | 0.64 | 1.11 | 0.74 | overshoot | overshoot |
| L4 mlp | 8.56 | 2.56 | 9.88 | collapse | genuine reduction |
| L6 resid | 12.5 | **33.0** | 14.69 | free0 = 0.000 | **2.2x OVERSHOOT** |
| L8 resid | 14.4 | 4.56 | 15.0 | collapse | genuine reduction |
| L9 attn | 4.69 | 0.16 | 5.31 | collapse | genuine collapse |
| L11 resid | 32.0 | **552** | 36.25 | free0 = 0.000 | **15x OVERSHOOT** |

REVISED TALLY: 5/8 overshoot (L0, L2, L3, L6, L11) — the balance-set
prediction — and 3/8 genuinely reduce (L4-mlp, L8-resid, L9-attn). The
context-dependent-role claim SURVIVES but on 3 seeds, not 5; the
previous section overstated it.

**FINDING A — free0 has a FALSE-NEGATIVE MODE.** At L11 a circuit
driving its seed 15x ABOVE natural scores free0 = 0.000, identical to a
circuit with no effect. The seed simply loses the top-k race against
everything else that blew up off-manifold. Any deep free0 ~ 0 must be
re-read pre-act before being called a failure (candidates for re-reading
include D3.6's deep-resid MS10 "hard zeros").

**FINDING B — but normalised pre-act is DEGENERATE at depth.** free0_pre
reads 0.99-1.00 for EVERY arm at L6/L8/L11 including random controls,
because the empty-circuit pre-activation is wildly off-manifold and
dominates the denominator (the same explosion the dual-floor normaliser
arc measured: norm_zero 3.3e10 at L10). So pre-act cannot simply replace
free0 as a ratio.

**CONSEQUENCE: report BOTH.** Post-top-k answers "does the seed still
register in the model's own sparse code"; RAW pre-activation vs natural
answers "what happened to the drive". Neither alone is honest at depth.
