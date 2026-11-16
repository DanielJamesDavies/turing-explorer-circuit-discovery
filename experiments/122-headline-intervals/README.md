# 122 headline intervals (DAN-163)

## Question

Does every pass rate or proportion in the main body of `paper/main.tex` (before `\label{end-of-main}`) have an
interval and an n, and where does each come from?

## Method

CPU only. `intervals_122.py` computes the intervals no 109 / 111 / 112 table had. It uses 95% percentile
bootstrap over targets, B = 4,000, seed 20261108 (`lib109.boot_ci` / `boot_paired_diff`), on existing per-target rows.
Zero-pass nulls get the exact one-sided 95% upper bound 1 − 0.05^(1/n).

    python -X utf8 experiments/122-headline-intervals/intervals_122.py   ->  out/intervals_122.csv / .json

## New intervals (this experiment)

| Number in body | Value [95% CI] | n | Source |
|---|---|---|---|
| Random nulls pass for none, upper bound | 0/112 → ≤ 2.64%; 0/126 → ≤ 2.35%; 0/128 → ≤ 2.31%; pooled 0/366 → ≤ 0.82% | 112 / 126 / 128 | 112 `nulls_detail_meta.json` (matches 112's 2.6 / 2.3%) |
| Deep-band size ratio U/W, smallest passing (13.8×) | 13.76 [8.73, 19.81] | 42 | 109 `run2026_tl_matched_targets.csv` |
| Ratio at L0-4 / L5-7 / all (5.1, 7.9, 6.0×) | 5.05 [4.57, 5.55] / 7.94 [4.95, 10.45] / 5.95 [5.24, 7.65] | 61 / 39 / 142 | same |
| Pythia size ratio U/W (2.3×) | 2.28 [2.04, 2.44] | 127 | 109 `run2026_unweighted_pythia_targets.csv` |
| Production gives up 5.8 points vs λ = 5e-4 (paired) | +5.78 [1.16, 10.40]; by band +5.8 [−1.4, 14.5], +8.9 [2.2, 17.8], +3.4 [−5.1, 11.9] | 173 (69/45/59) | 109 archives (T4 sweep + T3 main, `t6_tl_sweep.load`) |
| WCM 83.2% vs best unweighted 75.1% (paired) | +8.09 [1.16, 15.03]; by band +4.3 [−5.8, 14.5], +8.9 [−4.4, 22.2], +11.9 [−1.7, 23.7] | 173 | same |
| Plateau span: λ = 5e-5 (93.1%) vs 5e-4 (89.0%) (paired) | +4.05 [0.58, 8.09] | 173 | same |

The plateau claim: the pass rate stays in 89–93% from 2.5e-5 to 5e-4, but the best arm against the 5e-4 arm differs
significantly in a paired test (+4.0 [0.6, 8.1]; that arm was picked as the best, so the test is optimistic). "Holds at
89–93%" is a true range statement. It is not a flat plateau, and the tl_sweep caption now says so.

## Existing intervals (already in cited tables; checked against the CSV)

| Body number | Interval | n | Table / source |
|---|---|---|---|
| Pass 88.0 / 81.9 / 82.4% | [87.2, 88.7] / [78.5, 85.2] / [79.1, 85.5] | 7,464 / 508 / 511 | tab:run2026-headline (109 run2026_headline.csv) |
| Mid-band 53.5 / 73.6 / 66.8% | [52.3, 54.8] / [69.6, 77.4] / [62.6, 70.9] | 6,228 / 504 / 506 | same |
| Depth 82.3 / 90.7 / 91.2 (TL), 78.4 (Py L4-5), 65.7 (GPT L8-11) | [80.8, 83.8] … [58.3, 72.6] | per band | tab:run2026-groups, fig:run2026-headline |
| Kind 77.5 (TL attn), 64.8 (Py attn), 76.7 (GPT MLP) | [75.8, 79.3], [57.2, 72.3], [69.0, 83.7] | 2,083 / 159 / 129 | tab:run2026-groups |
| Null-panel production 83–86% | 85.7 [78.6, 92.0], 82.5 [75.4, 88.9], 82.8 [75.8, 89.1] | 112 / 126 / 128 | tab:run2026-controls |
| Donor test 69.2 / 66.1 / 41.1% | [68.1, 70.2] / [62.0, 70.3] / [36.8, 45.4] | 7,464 / 508 / 511 | tab:run2026-resample |
| No-R difference 5.2 pp | [−0.6, 11.6] | 173 | tab:run2026-tl-noR (in text already) |
| Amplifiers 29–34% | 29.0 [28.0, 30.0], 33.7 [29.5, 37.8], 31.3 [27.4, 35.4] | 7,464 / 508 / 511 | tab:run2026-specificity (111) |
| α = 1: 1.8 / 13.8 / 3.3% | [1.5, 2.1] / [10.8, 16.9] / [1.8, 4.9] | as headline | tab:run2026-alpha1 |
| Sweep 89–93%, 83.2%, 65.9%, unweighted peak 75.1% | per arm | 173 | tab:run2026-tl-sweep |
| Deep band 64.4 / 76.3 / 89.8% (intro) | [52.5, 76.3] / [64.4, 86.4] / [81.4, 96.6] | 59 | tab:run2026-tl-sweep (bands) |
| Donor sweep ≤ 9% unweighted, 66–83% weighted | per arm | 173 | tab:run2026-resample |
| Pythia unweighted 59.8 vs 83.5%; L4-5 37.5 vs 87.5% | [51.2, 68.5] / [77.2, 89.8]; [25.0, 52.1] / [77.1, 95.8] | 127; 48 | tab:run2026-pythia-unweighted |
| Stable core 44–72% | 43.8 [28.1, 62.5], 65.0 [56.9, 73.2], 71.9 [56.2, 87.5] | 32 / 123 / 32 | tab:run2026-stable-core |
| Framework 72-target panel 0 → 24% | difference CI +14 to +33 | 72 | app:resample text |

Not intervalled, on purpose: the case-study values (one target), coefficient-shape quantiles (descriptive of the
coefficient distribution, not pass rates), medians of sizes and scores, exclusion counts, and the 30-target pilot
(reported only, never used).

No mismatches were found between body numbers and their sources.
