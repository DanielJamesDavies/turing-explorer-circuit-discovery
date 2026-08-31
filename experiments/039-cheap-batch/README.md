# Cheap-analysis batch (holiday questions 1, 3, 4, 5, 7)

2026-08-21. Four analyses on existing artifacts; ~1 GPU-hour total.

## A. Amplitude distribution (`alpha_dist.py`)

| bank / arm | n | p10 | med | p90 | max | <0.1 | <0.5 | 0.9–1.1 | >2 |
|---|---|---|---|---|---|---|---|---|---|
| Llama triamp | 531 | 0.75 | 0.90 | 1.99 | 4.67 | 0% | 1.7% | 7.7% | 9.8% |
| Gemma triamp | 972 | 0.78 | 1.27 | 2.23 | 3.81 | 0% | 0.4% | 16.9% | 15.6% |
| Gemma coact_amp | 525 | 0.26 | 0.97 | 2.81 | 5.21 | 0.2% | **24.6%** | 7.6% | 23.0% |

* Ours is unimodal near 1, **zero mass at alpha~0** — the mask does NOT
  keep members switched off; inhibition is handled by exclusion, not by
  zero-amplitude membership (answers Q3's first half).
* The baseline's signature is crush-and-crank: a quarter of its alphas
  below 0.5 plus a fat >2 tail. The distribution alone distinguishes a
  discovered circuit from a regressed one.

## B. Sign audit + fragility hypothesis (`sign_audit.py`, Llama, held-out)

Single-latent zeroing, otherwise-natural model, 1% threshold:

| seed | members act / inh / neutral | background (100 live non-members) |
|---|---|---|
| L4 100791 | 60 / 1 / 10 | 0 / 0 / 100 |
| L4 111451 | 39 / 0 / 6 | 0 / 0 / 100 |
| L4 70641 | 66 / 3 / 83 | 0 / 0 / 100 |
| L6 101282 | 43 / 0 / 7 | 0 / 0 / 100 |
| L6 18986 | 64 / 5 / 63 | 0 / 0 / 100 |
| L6 35569 | 73 / 1 / 7 | 0 / 0 / 100 |

* **The strong fragility hypothesis is refuted at the single-latent
  level**: activators outnumber inhibitors ~35:1 inside circuits, and
  the strongest single inhibitor lifts the seed just +2-3% (vs
  activators at -86 to -100%).
* But inhibition is real and COLLECTIVE: zeroing the whole complement
  raises deep seeds up to 9.1x (the coact_raw overshoot). No single
  brake matters; the SUM of brakes is enormous. The elephant argument
  survives in aggregate form: each "not-an-elephant" signal is
  negligible alone. Mirrors the collective-faithfulness finding on the
  excitatory side.
* Background: 600/600 random live latents are individually neutral —
  single-deletion robustness is total; seeds are fragile only to
  coordinated removal.
* Neutral MEMBERS are common (up to 83/152): within-circuit redundancy,
  consistent with collective faithfulness and 16% membership
  non-uniqueness.

## C. Interchangeability first pass (`substitutes.py`, Gemma)

Of 326 our-members absent from the coact set, only **2%** have a
same-layer coact member within cos>0.5 of their decoder direction;
median best-match cosine 0.04-0.08 = the random-control level.

**The 32-47% overlap is NOT hiding redundancy-cluster relabelling.**
The two sets reconstruct the same seed with genuinely different
machinery — the strong non-identifiability reading. "The circuit" for a
feature is under-determined by faithfulness+drive even up to
direction-level substitution; necessity+calibration is where the sets
separate.

## D. Selectivity (Q5, from logged rows)

Circuits are 0.04-0.53% of their live pools (>99.5% of live latents
excluded), across both banks.

## Follow-ups these results sharpen

* Inhibitor-circuit campaign (holiday Q6): single-latent audits cannot
  find the brakes because braking is collective — a LEARNED minimal
  inhibitor set (increase-only mask over the posctx floor) is exactly
  the right tool, with its own null suite (maximise lesson applies).
* The alpha-distribution table belongs next to the coact appendix as a
  one-line contrast; the sign audit feeds the fragility discussion.
