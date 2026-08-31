# cf-mask training dynamics — why cf wins and free0 never moves (2026-08-02)

Six instrumented 400-step runs (engine gained a diagnostics-only
`step_hook`; 3 new tests, mask suite 85 green). Per-step: losses, lr,
temperature, theta/psi gradient norms (all / members / warm-start),
mean m, membership size + composition, add/remove churn, delta mass
(total, max, warm, SUB-THRESHOLD). Every 25 steps: held-out cf, free0,
and alignment of the LIVE circuit against direct-mass top-1024,
restoration PA activators/inhibitors, the D3.6 abl-mask closure circuit
for the same seed, and the seed's coactivation neighbours. Runs: the 4
best cf-mask arms + L11 cold (degeneracy control) + L11 pos objective
(the closure contrast). trace_*.jsonl (2,400 rows), summary_tables.md,
members_*.jsonl.gz, summary_*.json.

## FINDING 1 — the probe's headline numbers were an ARTEFACT OF STEP 400

cf decays MONOTONICALLY with training in every warm run (T1). Every
"failure" in 020-cfmask was a run measured on the far side of its
own peak:

| run | best cf (step) | reported final (step 400) |
|---|---|---|
| L4-mlp | **1.006 (250)** | 0.703 |
| L8-mlp | **1.038 (200)** | 0.022 |
| L9-attn | **0.635 (125)** | 0.278 |
| L11-resid | 1.014 (399) | 1.014 |

**L8-mlp — the cell I reported as "far worse than AMPC, no crack" —
reaches cf 1.038 mid-training.** L4-mlp reaches 1.006. Both MATCH AMPC
(0.96 / 1.01) at their peaks. L11's apparent "match" is coincidence: it
is on the same decay curve (3.81 -> 1.01) and merely crosses 1.0 as the
run ends. RETRACTION: "no failure cell cracked" is wrong — two of three
crack with early stopping, and the third (L9-attn) more than doubles.

## FINDING 2 — the mechanism of the decay: L1 starves a converged circuit

Data loss reaches ~1e-4 by step ~100 and theta gradients fall two orders
of magnitude (T6: 9e-4 -> 1.8e-5); membership stops churning entirely on
the mlp/resid runs (T3: 10-18 churn steps out of 400). From then on the
only live pressure is the delta L1 penalty, which keeps shrinking the
injected values of an already-correct circuit — cf follows the delta
mass down (L8-mlp: d_sum 19.2k -> 11.9k -> 5.7k as cf 1.17 -> 1.08 ->
0.02). Members are also dropped: L8-mlp 62 -> 31, and the collapse is a
CLIFF (cf 1.04 -> 0.32 between steps 200 and 300 as 11 members go),
i.e. deep-mlp drive has a critical mass, not a graceful degradation.

## FINDING 3 — free0 is ZERO AT EVERY STEP OF EVERY RUN

Not a late casualty of over-training: free0 = 0.0000 at step 0 (already
at the warm-start intervention, cf ~1.1-3.8) and at every snapshot
after. The cf-mask never passes near a closure solution — the two
objectives do not trade off along this trajectory, they are disjoint.
The contrast run makes it exact: L11-pos (closure objective, same seed,
same engine) walks 1,433,600 -> 13,428 members with 1.44M adds and
1.43M removes, ending at alignment 1.00 with the D3.6 mask and 0.06
with direct-mass; the cf runs sit at 1.00 direct-mass and (L11) 1.00
maskMF but 53 members. **Closure and drive are not two points on one
axis; they are different objects the same engine finds by different
routes.**

## FINDING 4 — L9-attn fires by REMOVING SUPPRESSION, not by injecting

A phase transition at step ~75: membership explodes 49 -> 777 as the
GATE half recruits ~700 inhibitor removals, and cf jumps 0.00 -> 0.45 ->
0.635 (T5: alignment with direct-mass collapses 1.00 -> 0.10 exactly
there, because the new members are edits, not injected activators).
The seed has no injectable driver — but it has a removable suppressor
set. This independently reproduces D2.2's finding that attention seeds
are suppression-gated (sup collapses 0.9 -> 0.2 when inhibitors are
dropped) from a completely different method. The revised L9-attn claim:
NOT "nothing can make it fire" but "nothing can DRIVE it — it must be
UNGATED", and even that transfers at only ~0.64.

## FINDING 5 — the cold degeneracy, caught step by step

L11-cold: **0 members at every one of the 400 steps** while delta mass
sits at 26k-16k, 100% of it sub-threshold. The optimiser reaches the
train target immediately with a diffuse blanket and never nucleates a
member. Warm runs carry 84-99% sub-threshold delta mass too (T4) — the
blanket is ALWAYS there; warm-starting just also plants above-threshold
members that survive binarisation. This is why warm-start is
load-bearing and why the binarised eval is essential (a soft eval would
score the blanket as success — the v1 trap).

## FINDING 6 — membership provenance is stable and mechanism-pure

The warm-started circuits never wander: alignment with direct-mass stays
0.95-1.00 for the whole run on L4/L8/L11 (T5), and with the abl-mask
closure circuit 0.93-1.00 — i.e. the drivers are a tiny SUBSET of the
closure membership (containment, D4.3's question, answered in passing:
these ~50 drivers all live inside the ~13k closure). Coactivation
alignment is 0.00-0.17 throughout: coactivation neighbours are NOT the
mechanism, on any run, at any step — the strongest form yet of the
"coact is descriptive, not causal" finding.

## Immediate consequences

1. **Early stopping / cf-tracking is mandatory** for this objective —
   or a schedule that stops the delta L1 once data loss converges. The
   cheapest fix: hold out cf every N steps and keep the argmin |cf-1|
   snapshot (all machinery now exists in this runner).
2. The lambda_inj calibration arc is REFRAMED: the problem is not only
   the price level but that a fixed price applied for 400 steps keeps
   eating a converged solution. Consider decaying lambda_inj to 0 after
   convergence, or pricing only until the data term stops improving.
3. 020-cfmask's README needs the retraction above; the L9-attn
   "identity certificate" must be restated as an UNGATING result with a
   0.635 ceiling, not 0.28.
