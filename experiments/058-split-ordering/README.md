# 058 — Is the train / held-out split ordered? (DAN-12 / DAN-15)

2026-09-24. Read-only check (`check_split.py`; `results/check_split.md`, `.jsonl`). Targets: the 16 pilot targets
plus 200 sampled from the 15,046 production seeds, with (d) on 64 of them. No errors.

## How the split is built (code)

- `ProbeDatasetBuilder.build_for_latent` concatenates the top contexts then the mid-band contexts, deduplicates,
  and caps the list at 64.
- The top store is maintained with `torch.topk` (descending).
- The loader preserves order.
- The engine and every evaluator use contexts[:48] for training and contexts[48:] as held-out.
- The close contrast selector returns candidates ranked by max cosine to the activating set, and consumers slice
  it the same way.

## Results

**(a) The store is sorted.** 216 of 216 targets have their stored top contexts in non-increasing order of the
store's score. The distributed merge did not scramble them.

**(b) …but the store's score is not the anchor activation, so the activating-context bias is small.**
- List position is only weakly related to the target's peak activation: Spearman median −0.11 on the sample and
  −0.12 on the pilot. The store ranks by its own per-sequence score, while evaluation reads the peak at the
  anchor.
- Engine split, held-out / train mean activation:

  | Targets | Median | IQR | Held-out weaker |
  |---|---|---|---|
  | Sample | 0.977 | 0.937–1.013 | 133 / 200 |
  | Pilot | 1.000 | 0.962–1.078 | 8 / 16 |

  Only 4 of 200 have every held-out context below the weakest training context.
- So held-out is about 2% weaker on average, and systematically so in two thirds of targets. It is not a gross
  bias.

**The stratified split as first designed is biased too.** Ranking by anchor activation and taking the LAST of
each block of 4 gives held-out / train 0.981 (IQR 0.974–0.987). The variance is much lower, but the held-out
context is always the weakest in its block, so the split is still about 2% low by construction.
- **Fix:** rotate the held-out offset through the blocks (block b gives up position b mod 4), or draw it with a
  seed per block. The expected ratio is then about 1.0, with the same low variance.

**(c) Mid-band contexts almost never enter.**
- Only 501 of 15,046 production targets (3.3%) have fewer than 64 valid top contexts; the median is 64.
- Across the bank, 5.7% of latents that ever fired have fewer than 64.
- So "activating contexts" are the 64 strongest by the store's score for about 97% of targets.
- Paper L177 ("strongest-activating sequences together with a reservoir drawn from the middle of its activation
  distribution") is wrong for most targets and must be corrected (or mid-band contexts included on purpose).

**(d) The close contrast contexts are strictly ranked, so held-out is always the less similar end.**
- 64 of 64 are monotone, with Spearman −1.00.
- Held-out is less similar on 64 of 64: max cosine 0.874 (train) vs 0.861 (held-out), as medians.
- The gap is small in absolute terms, but it holds every time: held-out contrast contexts are slightly easier.

## Decision (Daniel, 2026-09-24; recorded on DAN-12 / DAN-15, both Done; spec in DAN-75)

- **Activating contexts are "the 64 strongest"** by the store's score, with no mid-band contexts. Targets with
  fewer than 64 stored top contexts use what they have.
- **Stratified split with rotating offsets, and the two strongest contexts always train:**
  - rank by anchor activation and cut into 16 blocks of 4;
  - block b (0-based) holds out position (b + 2) mod 4, giving held-out ranks 3, 8, 9, 14, 19, 24, 25, 30, …;
  - each within-block position is held out exactly 4 times.
- **Contrast contexts** use the same rule, ranked by similarity.
- **Implementation:** one shared split function (engine incl. `floors_train_only`, eval pass, 056, 057). Freeze
  the DAN-74 golden set after this change.

## Decision input for DAN-12 / DAN-15 / DAN-75

- **Pre-v1 and pilot results.** Held-out activating contexts were about 2% weaker and held-out contrast contexts
  slightly less similar. This is a mild, conservative-to-neutral bias. It doesn't invalidate the 055–057
  comparisons, since every arm shared it.
- **For DAN-75.** Adopt the **stratified split with rotating offsets**, free since everything is refit:
  - activating contexts ranked by the target's anchor activation (the probe builder already computes the
    argmax in a forward pass, so the values come free);
  - contrast contexts ranked by similarity;
  - one shared function used by the engine, the eval pass, 056 and 057.
- **Paper.** State the split exactly and fix the "reservoir from the middle" description.
