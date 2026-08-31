# GemmaScope JumpReLU: first validation outside the top-k family

2026-08-12. Gemma-2-2B + google/gemma-scope-2b-pt-mlp, width 16k,
JumpReLU, tier 2 of the L0 ladder (average_l0 ~85-133 per layer — the
same operating point as DeepMind's canonical release). 6 seeds (3 at L4,
3 at L6), 48 train / 16 held-out windows, 400 steps, the exact harness
of the Llama run with only the model/feature layer changed.

## Why this experiment

Every dictionary the method had been validated on was top-k-like, and
the one ReLU/L1 bank tried had a vacuous null. JumpReLU is the third
activation family (per-latent learned thresholds; no fixed L0, no
sub-threshold shrinkage tail) and the standard public SAE suite.

## Conventions: probed, not assumed — and the probe caught two traps

1. **Gemma-2 is not Llama.** FOUR layernorms per block;
   `post_attention_layernorm` is on the ATTENTION path. These SAEs
   reconstruct the MLP's residual contribution =
   `post_feedforward_layernorm`'s OUTPUT, uncentred, gate `raw`.
   Measured FVU 0.127 with L0 84.2 against the advertised average_l0 of
   85 (the L0 match is what identifies the convention; the runner-up
   gave L0 215). Assuming Llama's layout gives FVU 0.69.
2. **The wrong-tensor bug, caught and falsified.** The first port left
   three read-side hooks encoding the layernorm's INPUT. Everything
   downstream looked like a discovered breakdown: anchor support 15.5%,
   circuits of 9,280 nodes, nulls scoring 1.08 — a convincing fake of
   the ReLU/L1 pathology. Rerunning THE SAME SEED (L6/5287, windows
   re-derived under the corrected read) flipped every symptom: anchor
   support 0.94%, n=447, nulls 0.000. Worth keeping because it shows
   how a convention bug can perfectly impersonate a scientific finding.

## Result (6 healthy seeds; the weak-seed 5287 rerun kept separate)

| arm | n (med) | ampF0 med [min,max] | ampFM | sup | cf |
|---|---|---|---|---|---|
| tri-amp | 82 | **1.05** [0.61, 1.25] | 0.99 | 0.87 | 0.317 |
| gate-only | 119 | 0.98 [0.53, 1.16] | 0.69 | 0.90 | 0.155 |
| plain null | 82 | 0.00 [-0.01, 0.06] | 0.01 | 0.00 | 0.000 |
| support-matched null | 82 | **0.36** [0.09, 0.76] | 0.57 | **-0.03** | 0.018 |

Anchor support 0.33-1.82% (median 0.60%): at its standard operating
point JumpReLU is in the LOW-support regime, same side as TopK. What
separates the working cases from the vacuous ReLU/L1 one is per-position
density relative to the live pool, not the activation function.

**Tri-amp replicates**: ~1.0 reconstruction under both fills, 6/6 seeds,
plain nulls dead. Tri-amp again beats gate-only on mean-fill (0.99 vs
0.69) at ~30% fewer nodes.

**The support-matched null is NOT dead here — and that is the
informative part.** Median ampF0 0.36, best draw 0.76 (L6/14044);
narrowest ours-vs-null gap 0.649 vs 0.595 (L6/5011). Anchor-firing
JumpReLU latents with fitted amplitudes can regress a fair way toward a
target — graded activations carry more usable signal than TopK's sharp
survivors. But the same sets have NO necessity (sup -0.03 vs our 0.87)
and no drive (0.018 vs 0.317). On this dictionary, faithfulness alone is
partially spoofable by a support-matched random set; the JOINT
faithfulness+necessity test is what separates real circuits. That is the
one-object/two-criteria framing demonstrated on a third architecture —
if validation had used reconstruction only, JumpReLU would have weakened
the claim.

## Caveats

* cf_amp is low for OUR circuits too (median 0.317 vs 0.87 on Llama),
  and sup spreads 0.75-1.00. Plausibly Gemma routes more of a feature's
  input through attention / the residual than our MLP-sites-below
  universe covers. Unresolved; do not quote Gemma drive numbers without
  this caveat.
* 6 seeds, 2 depths, one width (16k), one tier. The tier-2 choice
  matches the canonical release's operating point.
* Support-matched nulls were fitted with the same amplitude machinery as
  ours (support-restricted, lambda 0), 3 draws/seed.
* Measured L0 on wikitext runs 1.1-1.6x the advertised average_l0
  (trained-distribution label); we report measured density.

## Files

* `gemma_loader.py` — list/probe/fetch; `gemma_convention.json` is the
  probe's verdict
* `ours_gemma.py` — the harness (patched copy of ours_llama.py)
* `check_gemma.py` — identity gate (max|d| exactly 0), FVU + L0 vs
  advertised, ablation-bites test
* `ours_gemma_rows_t2.jsonl`, `ours_gemma_members_t2.jsonl` — data
* `panel_scan.log`, `panel_run.log`, `panel_supnull.log`, `run_t2*.log`
* SAE weights cached at `$HOME/gemmascope` (native ext4 — /mnt/x reads
  at 216 MB/s and barely caches; see the Llama engineering notes)
