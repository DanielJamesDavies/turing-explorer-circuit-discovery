# cf-mask v2 (D3.2-lite) — 4-seed probe (2026-08-02)

Engine: three new default-off knobs in run_learned_mask (all unit-tested,
mask suite 82 green): scale_normalize (data term / target_act^2, delta
priced per unit of target — the diagnosed MI dead-at-defaults fix),
delta_init (warm start at AMPC's K=64 intervention: psi =
softplus^-1(alpha* x pin) for top direct-mass latents), suppress_weight
(D3.2's posctx-suppression term; built, off in this probe). Run: 4 seeds
(the three AMPC failure cells + L11 healthy control) x 5 arms (warm at
lambda_inj {3e-4, 3e-3, 3e-2}, cold at middle, warm + 2 seed-adjacent
sites excluded). Eval: held-out store negctx, BINARISED intervention
(member inhibitors -> 0, member activators -> +learned delta), cf on
D2.3's convention. rows.jsonl (20 rows).

## Headline results (held-out cf; AMPC K=64 reference in parens)

| seed | best cf-mask arm | AMPC | verdict |
|---|---|---:|---|
| 35/6599 L11 (control) | **1.014** (warm-l1, 53 latents, dir1k 1.0) | 1.01 | MATCHES AMPC with learned per-latent values |
| 13/30053 L4-mlp | 0.70 (warm-l0) | 0.96 | no crack |
| 25/10628 L8-mlp | 0.02 (warm-l0) | 1.01 | far worse — lambda knee missed |
| 27/6859 L9-attn | 0.28 (warm-l1) | 0.00 | ties the seed's historical ceiling |

## Verdicts

1. **Warm-start is load-bearing.** Cold arms produce nothing usable on
   any seed (cf 0.005 / 0.0 / 0.14 / EMPTY); the L11 cold arm is the v1
   degeneracy caught red-handed — train target reached entirely by a
   diffuse sub-threshold blanket (d_sum 16k, p_inject 37.5) with ZERO
   members kept. Warm-started arms keep 21-61 members, 62-100% inside
   direct-mass top-1024 (mechanism, not fabrication).
2. **Method validated on the control**: L11 warm-l1 matches AMPC
   exactly (1.014 vs 1.01), all-mechanism membership, overshoot at the
   cheaper lambda (1.26) corrected by pricing. And warm-x2 (both
   seed-adjacent sites EXCLUDED) still scores 0.99 — L11's drive is
   fully mediated; the "wire" is not needed.
3. **No failure cell cracked.** L4-mlp best 0.70 < 0.96; L8-mlp <= 0.02
   despite reaching target on TRAIN with pure-mechanism members — the
   train->held-out gap plus sub-threshold delta mass stripped by
   binarisation. The working lambda is seed-dependent even after scale
   normalisation (L4 wants 3e-4, L11 wants 3e-3; L8's knee was missed by
   the grid): the per-seed lambda calibration arc (probe-style, as for
   the closure mask) is REQUIRED before this competes with AMPC on mlp.
4. **L9-attn: the identity certificate is now SHARP.** The most
   expressive intervention we can build — per-latent learned removals
   AND unbounded injections — reaches the target ON TRAIN (p_inject
   5.19 vs 5.31; the first method ever to do so there) yet transfers at
   only <= 0.28 held-out, exactly the seed's historical ceiling.
   Context-specific drive is fabricable; TRANSFERABLE drive does not
   exist in upstream latent space. That is a stronger and cleaner claim
   than "no method found a driver", and it is the case-study sentence.
5. **The diagnostics work as designed**: dir1k separates mechanism from
   fabrication (warm 0.62-1.0 vs cold 0.0), concentration stats expose
   the blanket, and the exclusion ablation cleanly measures mediated
   drive (L11: 0.99 without the adjacent sites).

## Bottom line

As a driver-FINDER the cf-mask does not currently beat AMPC anywhere
(ties on the control at 30x the compute). Its unique value today: the
L9-attn transferability certificate, the p_gate/p_inject decomposition,
and per-latent values where uniform alpha is wrong — worth revisiting
only after a per-seed lambda_inj calibration arc, and worth citing now
for the certificate.

## Caveats

- The warm-x2 arms at L4/L8/L9 ran at the MIDDLE lambda (already past
  those seeds' knees) — the exclusion ablation is only clean at L11.
- Binarised eval drops sub-threshold deltas by design (matches how all
  arms are scored); a soft-eval column would separate the soft/hard gap
  from true non-transfer at L8-mlp.
- Single split, 4 seeds, one warm-start recipe (K=64); alpha* for
  L9-attn warm start is the censored 8.0.

## Closure-eval addendum (rows2.jsonl — free0 / freeN_topk added)

Deterministic rerun (all cf values reproduced exactly; members now
archived per arm). The learned cf-circuits are DEAD as closure objects:
free0 <= 0.019 and freeN_topk <= 0.16 on every arm of every seed —
including the L11 arm that matches AMPC on cf (1.014 cf / 0.000 free0).
Expected and structural: these are 30-300-latent driver sets, and
closure at these depths needs 10^4-10^5 members; ablating everything
else destroys the seed's natural computation regardless of how well the
kept members can DRIVE it. The mirror of the D3.6 masks (free0 0.7-1.0,
cf incidental). Fourth clean demonstration of "you win the metric whose
semantics you train" — and the baseline any future tri-objective
(pos + inject + suppress on shared gates; Daniel's generalist vision,
the (P2*) intersection object) would have to beat on BOTH columns at
once.

## !! RETRACTION (2026-08-02, from 020-cfmask) !!

The headline numbers in this file are STEP-400 readings of runs whose cf
decays monotonically after converging around step 100-250. The
instrumented reruns show every "failure" here is an over-training
artefact:

| seed | reported (step 400) | ACTUAL BEST (step) | AMPC |
|---|---:|---|---:|
| L4-mlp | 0.703 | **1.006 (250)** | 0.96 |
| L8-mlp | 0.022 | **1.038 (200)** | 1.01 |
| L9-attn | 0.278 | **0.635 (125)** | 0.00 |
| L11-resid | 1.014 | 1.014 (399) | 1.01 |

RETRACTED: "no failure cell cracked" — L4-mlp and L8-mlp both MATCH
AMPC at their peaks, and L9-attn more than doubles its historical
ceiling. RETRACTED: L11's "match" is not skill — it is the same decay
curve crossing 1.0 as the run happens to end.
REVISED (L9-attn): the certificate is not "transferable drive does not
exist" but "the seed cannot be DRIVEN, only UNGATED" — cf 0.635 is
reached by removing ~700 suppressors (gate half), not by injection,
independently reproducing D2.2's suppression-gated attn finding.
The closure-eval addendum below STANDS: free0 = 0.0 at every step of
every run, from step 0 onward.
