# The coact arc: how strong can a correlational baseline get?

2026-08-12. Synthesis of a three-bank investigation; the per-bank data
lives in `035-transcoder-compare/` (Llama TopK skip-transcoders),
`037-gemmascope/` (Gemma-2 JumpReLU), and this dir (home bank,
TuringLLM k=128/40,960). In the paper as appendix
"Co-Activation with Fitted Amplitudes" (`app:coact-amp`), plus a
qualifying sentence in `sec:coact-results`.

## The ladder (Daniel's framing: "support-matched null sounds like a
## discovery algorithm")

1. **Plain null** — random same-size sets, fitted amplitudes: dead on
   every bank (paper's 124- and 146-draw results; 18/18 on Llama).
2. **Support-matched null** — random among anchor-firing latents,
   matched on firing count: dead on TopK (max 0.054); NOT dead on
   JumpReLU (median 0.36, best 0.76) but zero necessity (sup -0.03).
3. **Greedy co-activation (`coact`)** — top-n by summed anchor
   activation, size-matched to the discovered weighted circuit:
   * `coact_raw` (alpha=1): fails everywhere, DIRECTIONALLY —
     0.08-0.51 shallow, 7.6-9.1x OVERSHOOT deep (sign-blind selection
     harvests excitation, misses inhibitors).
   * `coact_amp` (fitted alphas, identical budget): the strongest
     baseline in the programme. Zero-fill ~1 on all three banks.

## The verdict (best vs best, home bank, triple floor)

| cell | discovered F0 / FMd | coact_amp F0 / FMd |
|---|---|---|
| L2 386 (n=206) | 1.01 / 0.94 | 1.00 / **1.45** |
| L2 2927 (n=124) | 0.92 / 1.18 | 0.99 / **1.30** |
| L2 7019 (n=229) | 0.98 / 1.15 | 0.95 / **2.47** |
| L9 1283 (n=233) | 1.04 / 1.15 | **3.09** / 0.90 |
| L9 2062 (n=602) | 0.92 / 0.86 | **0.79** / 1.07 |

**Discovered: 5/5 cells hold both fill bands. coact_amp: 0/5.**
At L2 the fitted alphas calibrate against one background and overshoot
the other (alphas 6-10x). At L9 no coefficient assignment reconciles the
sign-blind selection with all three floors at once — it overshoots
zero-fill on one seed and undershoots on the other.

Cross-bank: on Llama, coact_amp F0 med 0.811 / FMd 0.777 / sup 1.00
(ours 0.847 / 0.929 / 1.00); on Gemma F0 1.06 but sup 0.71 vs our 0.87
and FMd overshoots. Single scores come within reach; the joint suite
separates on every bank, on a different criterion each time.

## What this changes

* `sec:coact-results` ("correlational proposals fail outright") is now
  QUALIFIED: without coefficients they fail outright; with coefficients
  they fail calibration. One-sentence pointer added in the body.
* Reconstruction alone is spoofable by a calibrated correlational set.
  The two-criteria/multi-floor discipline is not decoration — it is the
  discriminator, demonstrated adversarially on three banks.
* Mechanistic nugget: circuits need brakes as well as accelerators. The
  deep raw overshoot (7.6-9.1x) shows the zeroed complement is
  net-inhibitory; the learned mask retains inhibitors because
  calibration forces it to.

## Tide check (Gemma, `coact_tide.py` there)

Neither method reconstructs by raising the layer wholesale: median
latent shift ~-0.1, ~40-47% of latents up, seed rank preserved where
reconstruction succeeds (5->3, 8->6), lost where it fails (15->459).
Both methods splash a handful of bystanders by +8-16. Seeds naturally
rank 5-15 of ~60-170 active latents at their anchors.

## Files

* `coact_home.py` — dual-floor arms at L2/L9 (c8/c29 jsonl)
* `coact_triple.py` — best-vs-best at L9 (coact_triple_c29.jsonl)
* logs: c8.log, c29.log, triple29.log
