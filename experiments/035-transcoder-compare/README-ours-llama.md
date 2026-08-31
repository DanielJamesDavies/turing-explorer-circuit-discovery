# Our tri-amp side on Llama-3.2-1B + EleutherAI TopK skip-transcoders

2026-08-11. Second cross-architecture replication, and the setup that
also serves as the substrate for the circuit-tracer comparison (both
sides run on the *same* model and the *same* transcoder features — see
`check_agreement.py`, which passes on all 16 layers at 2.4e-7–1.3e-6
with identical top-k support).

* model: `unsloth/Llama-3.2-1B` (ungated mirror of meta-llama's)
* features: `EleutherAI/skip-transcoder-Llama-3.2-1B-131k`,
  TopK k=32, 131,072 latents, 16 layers, skip connection
* seeds: 6 (3 each at L4, L6), 4 upstream sites each
* protocol: 48 train / 16 held-out windows, BOS prepended, position 0
  excluded from anchor selection (circuit-tracer zeroes it by design)
* 400 steps, lambda 1e-3, 3 size-matched nulls per seed

## Result

| arm | n (med) | ampF0 med [min,max] | ampFM | sup | cf_amp | ALL-PASS |
|---|---|---|---|---|---|---|
| triamp400 | 76 | 0.85 [0.26, 1.43] | 0.93 | 1.00 | 0.87 | 2/6 |
| gate400 | 126 | 0.92 [0.60, 0.96] | 0.70 | 1.00 | 0.63 | 1/6 |
| null (18 draws) | 76 | **0.00 [0.00, 0.03]** | 0.00 | 0.00 | 0.00 | 0/18 |

**The null is dead: 0 of 18 draws reach ampF0 > 0.05.** Size-matched
random sets drawn from the same live pool reconstruct nothing, ablate
nothing (sup median 0.00) and drive nothing (cf 0.00). This is the clean
separation the dense-SAE arm could not produce, and it is the first time
we have it in a different model family.

Anchor support is 3.5–5.9% (median 4.8%) over live pools of 71k–126k, so
this is **not** the degenerate-support regime: the null has plenty of
live latents to draw from and still fails completely. That distinguishes
this from the dense-SAE pathology, where nulls scored ~1.0 because
anchor support was 6.5–37% and the fill made the null's members the only
nonzero entries.

Tri-amp reaches **higher faithfulness under mean-fill (0.93 vs 0.70) and
higher drive (0.87 vs 0.63) with 40% fewer nodes (76 vs 126)** than the
membership-only gate. The tri-amp advantage replicates outside the home
SAE.

## Honest caveats

* ALL-PASS rates are low in absolute terms (2/6, 1/6). The strict band
  requires both floors in [0.8, 1.25] *and* sup > 0.9; sup is 1.00
  nearly everywhere, so the failures are the amplitude bands (one seed
  undershoots at 0.26, one overshoots at 1.43). 400 steps on a 131k
  dictionary across 4 sites is a tighter budget than the home setting.
* 6 seeds is a small panel. Treat the tri-amp-vs-gate margin as
  suggestive; the null result is the robust part (18/18, unanimous).

## Engineering notes

The run initially spilled into shared GPU memory (15.5/16.0 GB dedicated
plus 14.6 GB shared, ~5x slowdown — a single 400-step fit had not
finished in 9 minutes). Cause: dense `[B, T, 131072]` code tensors when
TopK k=32 means only 32 entries per position are nonzero, a 4096x waste,
across 4 sites x 3 floor terms x autograd graphs.

Fixed by casting the stream to bf16 (`DTYPE`, precedented here: the home
pipeline's bf16 code stream was free0-neutral) and halving the fit batch
(`FIT_BS`). Parameters and the loss stay fp32 — only the injected
residual crosses in bf16. A 400-step fit now takes ~40 s, a >13x
speedup, and the resident set is 14.4 GB with no spill.

The proper fix is a sparse code representation (32 nonzeros, not
131,072). Not attempted: it is a substantial rewrite of the mask
machinery for a run that now completes in ~35 minutes.

`STEPS` is env-overridable so a cheap smoke pass can exercise every arm
and scoring path in minutes — added after two dtype seams were each
found by a separate 15-minute crash.

## Files

* `ours_llama.py` — this side (`scan` then `run`)
* `ours_llama_rows.jsonl` — scored rows, one per (seed, arm)
* `ours_llama_members.jsonl` — tri-amp membership + fitted alphas
* `smoke_rows_6step.jsonl` — 6-step plumbing pass, **not results**
