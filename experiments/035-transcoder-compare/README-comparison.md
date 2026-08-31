# circuit-tracer vs tri-amp on the same model and the same features

2026-08-12. Llama-3.2-1B + EleutherAI TopK skip-transcoders (131,072
latents, k=32, 16 layers), 6 seeds at L4/L6, 48 train windows each.

Both sides read the SAME converted weights and were verified to compute
the SAME feature activations before any comparison was run
(`check_agreement.py`: all 16 layers, 2.4e-7 to 1.3e-6, identical top-k
support). So differences here are differences in METHOD, not convention.

## 1. Agreement: do the two methods pick the same latents?

At matched size (their top-n, where n is our tri-amp size):

| seed | n | overlap | % of ours | vs chance in THEIR pool |
|---|---|---|---|---|
| L4 111451 | 45 | 34 | 75.6% | 2.15x |
| L4 100791 | 71 | 49 | 69.0% | 1.24x |
| L4 70641 | 152 | 69 | 45.4% | 0.99x |
| L6 35569 | 81 | 52 | 64.2% | 1.52x |
| L6 101282 | 50 | 34 | 68.0% | 2.61x |
| L6 18986 | 132 | 56 | 42.4% | 2.44x |

Median 66% of our nodes appear in their matched-size set. Against a draw
from the full live pool (71k-126k latents) that is overwhelming — chance
expects 0.02-0.3 nodes. Two independent methods largely agree on which
latents are involved.

The weaker reading matters too: against their OWN candidate pool (the
128-758 features they assign nonzero attribution), enrichment is only
1.84x median, and L4/70641 sits at 0.99x — exactly chance. Their
attribution already narrows to a small shortlist; within it, our
selection is only moderately enriched.

## 2. Function: do their sets pass the exact-forward exams?

All arms scored through the IDENTICAL code path (their sets are fed to
the same `score()` closure as our own arms — not a reimplementation).
Certified by a `selfcheck` arm that re-scores our own members through
that path and reproduces every `triamp400` row exactly, all 6 seeds.

Their method returns a ranking with no per-latent coefficients, so it is
rendered at alpha = 1.0 — a membership-only circuit, directly comparable
to our `gate400`.

| arm | n (med) | ampF0 med [max] | ampFM | sup | cf_amp | ALL-PASS |
|---|---|---|---|---|---|---|
| triamp400 | 76 | 0.847 [1.425] | 0.929 | 1.00 | 0.872 | 2/6 |
| gate400 | 126 | 0.917 [0.957] | 0.697 | 1.00 | 0.634 | 1/6 |
| theirs (matched) | 76 | **0.000 [0.000]** | 0.000 | 1.00 | 0.118 | 0/6 |
| theirs_full | 192 | **0.000 [0.000]** | 0.000 | 1.00 | 0.585 | 0/6 |
| null | 76 | 0.000 | 0.000 | **0.00** | 0.000 | 0/6 |

**THE TWO CRITERIA COME APART.** Read against the null, their sets are
plainly not random: `sup` is 1.00, identical to ours, where the null is
0.00. Ablating their nodes genuinely silences the seed. But keeping only
their nodes reconstructs NOTHING — ampF0 is 0.000 on every seed, max
0.000, under both fills.

Necessary, but never sufficient. Ours is both.

`theirs_full` is the fairness control, and it is the one that makes the
result solid: scoring only their top-n could have been a budget
artifact, so we also scored their ENTIRE ranking (128-758 nodes). Given
every node their method proposes, faithfulness is still 0.000. It buys
real interventional control (cf_amp rises 0.118 -> 0.585) but no
faithfulness. So this is not "their ranking needs more nodes" — it is a
faithfulness gap that more nodes do not close.

## Reading this fairly

Attribution graphs are not designed to return a sufficient generative
set; they answer "what influences this feature", not "what set
reproduces it". The honest claim is therefore NOT that their method is
wrong at its own goal — it is that **interventional control and
faithfulness are separable properties, and a method optimised for one
does not deliver the other for free.** That is the paper's one-object /
two-criteria framing, demonstrated across methods rather than asserted,
and it is the same home-turf principle we have applied to ourselves
throughout: you win the metric whose semantics you train.

Note also the direction of our own result: `gate400` (membership only,
the same rendering as theirs) reaches ampF0 0.917. The difference is not
the alpha coefficients — it is which latents are selected.

## Cost asymmetry (a finding in its own right)

circuit-tracer chooses which features get adjacency ROWS by influence on
the top LOGITS. Our seed is an internal mid-layer feature, so at a
4,096-node budget it gets NO ROW AT ALL. Reaching it needs a 16,384-node
graph — 32,256 active features, adjacency (17482 x 17482) — to extract
the single row we want. That is the concrete price of logit-rooting when
the question is about an internal feature.

## Caveats

* 6 seeds, 2 depths, one model, one SAE family.
* Their nodes are rendered at alpha=1.0 because their method emits no
  coefficients. That is the faithful rendering of their output, and
  `gate400` is the matched control for it.
* Positions: we accumulate their edge weight across all positions where
  the seed fires, then rank. A different aggregation could reorder.

## Reproduce

```
# their side (venv-ct); ~23s per attribution, ~1.9h for 6 seeds
TC_DIR=$HOME/tc_llama_folded LAZY_ENC=1 BATCH=256 \
  MAX_FEATURE_NODES=16384 ../../dev-notes/data/venv-ct/bin/python theirs_llama.py
# score their sets + the self-check through OUR pipeline
SCORE_THEIRS=1 SELF_CHECK=1 PYTHONPATH=. python ours_llama.py run
# overlap + chance baselines
PYTHONPATH=. python compare.py
```

`prefold_weights.py` bakes the RMS fold into W_enc on disk in bf16 so the
encoders can be loaded lazily; without it the 16k-node graph does not fit
in 16 GB. See README-engineering.md for why that was necessary.
