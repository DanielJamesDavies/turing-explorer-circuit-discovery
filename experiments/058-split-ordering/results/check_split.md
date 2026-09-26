# Split ordering check (DAN-12 / DAN-15)

Targets: 16 pilot + 200 sampled from the 15046 production seeds; 0 errors.

## (a) Stored top-context order

- monotone non-increasing: 216 / 216 targets (median violations 0)

## (b) Target activation at its anchor, engine split (first 48 train / last 16 held-out)

- **pilot (n=16):** held-out / train mean activation, median 1.000 (IQR 0.962–1.078); held-out weaker on 8 / 16; position-vs-activation Spearman median -0.12; every held-out context below the weakest training context on 0 / 16
  - stratified split instead: held-out / train median 0.984 (IQR 0.974–0.988)
- **sample (n=200):** held-out / train mean activation, median 0.977 (IQR 0.937–1.013); held-out weaker on 133 / 200; position-vs-activation Spearman median -0.11; every held-out context below the weakest training context on 4 / 200
  - stratified split instead: held-out / train median 0.981 (IQR 0.974–0.987)

## (c) How often mid-band contexts enter (fewer than 64 valid top contexts)

- production seeds: 501 / 15046 (3.3%) have < 64; median valid 64
- whole bank, latents that ever fired: 81523 / 1433446 (5.7%) have < 64

## (d) Close contrast contexts: max cosine to the activating set, first 48 vs last 16

- n=64: train 0.874 vs held-out 0.861 (medians of per-target means); held-out less similar on 64 / 64; monotone 64 / 64; Spearman median -1.00