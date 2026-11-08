# 109 pod-run analysis: paper tables and figures from the 2026-11-07/08 production run

## Question

What did the production pod run (WCM v4, λ = 1e-3, on TuringLLM 8k / Pythia-70m 512 / GPT-2 small 512) produce?
Every scripted table and figure here reads the downloaded result archives. No number is typed in by hand.

## Inputs

Archives (md5-checked) in `../Runs/pod-results-2026-11-07/`. `unpack.sh` unpacks each one into
`out/archives/<archive name>/`. It skips the `*-stores` data-pass tarballs, which no table needs, and leaves `out*`
gitignored.

| Archive | Used for |
|---|---|
| `pub-pythia-70m-S10-results`, `pub-gpt2-small-S10-results` | everything public: `experiments/082-public-runs/out_<model>/{main,alpha1,sweep,harder_nulls,strict,donor,donor_unweighted,stable_core}`, `pilot_lambda.json`, `targets_<model>_*.txt`, manifest |
| `pub-turingllm-T2-records` | `out_turingllm/lists/` (8k list, sweep190 etc., quarantine.json), `records/build.shard*` (skip reasons) |
| `pub-turingllm-T3-main` | `094-tl-v2-records/out_v2/{main,alpha1}` |
| `pub-turingllm-T1-scan` | unpacked; no table reads it |

Later archives are picked up automatically once `unpack.sh` has unpacked them. Every reader globs all
`pub-turingllm-*` folders, and a later archive wins the keep-last. Until then the scripts print "not yet available":
T4 sweep (`out_v2/sweep`), T5 stable core (`out_v2/stable_core`), T6 harder nulls (`out_turingllm/harder_nulls`),
T10 without-R (`out_v2_noR/sweep`), T11 donor (`out_v2/donor/main`) and T12 donor (`out_v2/donor/sweep`,
`out_v2_noR/donor`). T7 (083 baselines) was skipped in this run.

## Definitions (reused, not re-implemented)

- Pass rule and vacuous set: `062-h100-protocol-v1/merge.py` `passes` / `vacuous` (v2 rule). Pass = Z = `free0_tk`,
  A = `freeM_topk_tk` and C = `freeN_topk_tk` all in [0.75, 1.5], and necessity `phi_sup_blind_tk` >= 0.9.
  Induce = `phi_cf_alpha_blind_tk`, reported only.
- Depth bands: `062/family_table.py`. TuringLLM and GPT-2 use L0-4 / L5-7 / L8-11; Pythia uses L0-1 / L2-3 / L4-5.
- Rows: error and skip rows are dropped, and the last row per (seed, held[, arm]) is kept. Held-out strongest is
  primary; mid-band is reported.
- CIs: 95% percentile bootstrap over targets (B = 4000, seed 20261108). Differences use a paired bootstrap.
- Reused summarisers:
  - 084 `report.py` `frame` / `table_quantiles` for the nulls;
  - 073 `report.py` definitions for the resample test;
  - `src/eval/donor_test.py` `resample_pass` for the donor test, recomputed from the stored prod row and checked
    against the stored flag (100% agreement);
  - 062 `stable_core.py` / `rescore_alpha1.py` REPORT logic;
  - `082/pilot_lambda.py` `report()`, re-run on the rows and checked against the pod's `pilot_lambda.json`
    (matches).

## Scripts

Run them from the repo root with `python -X utf8 experiments/109-pod-run-analysis/<script>`, or run everything with
`run_all.py`, which takes about 30 s on the CPU and writes `out/run_all.log`.

| Script | Reads | Writes |
|---|---|---|
| `unpack.sh [RESULTS_DIR]` | `pub-*.tgz` + `.md5` | `out/archives/<name>/` |
| `lib109.py` | (shared: locations, row reader, pass rule import, bootstrap, LaTeX writer) | |
| `t1_headline.py` | main/eval, main target lists | `paper/tables/run2026_headline.tex`, `run2026_headline_groups.tex`; `out/run2026_headline*.csv/.json`; `paper/figures/run2026_pass_by_band.pdf/.png` |
| `t2_exclusions.py` | target lists, 082 manifests, main/status, main/contexts, main/eval; TL records/build + quarantine.json | `run2026_exclusions.tex`; `out/run2026_exclusions.csv`, `out/run2026_excluded_targets.csv` (one row per excluded target with its reason) |
| `t3_alpha1.py` | alpha1/alpha1, main/eval | `run2026_alpha1.tex`, `out/run2026_alpha1.csv` |
| `t4_robustness.py` | harder_nulls/nulls, strict/rows, donor*/rows, stable_core/stable_core, main/eval | `run2026_nulls.tex`, `run2026_resample.tex` (073 + donor), `run2026_stable_core.tex`; `out/run2026_nulls*.csv`, `out/run2026_resample.csv`, `out/run2026_stable_core.csv` |
| `t5_unweighted_pythia.py` | Pythia main/eval + sweep/eval arm U_0.001, `targets_pythia-70m_unweighted.txt` | `run2026_unweighted_pythia.tex`, `out/run2026_unweighted_pythia*.csv`, `paper/figures/run2026_unweighted_pythia.pdf/.png` |
| `t6_tl_sweep.py [--model M] [--test]` | TL `out_v2/sweep/eval` (W_0.001 falls back to the T3 main row), `targets_tl8k_sweep190.txt`, `out_v2_noR/sweep/eval` | `run2026_tl_sweep.tex`, `run2026_tl_sweep_bands.tex`, `run2026_tl_matched.tex`, `run2026_tl_noR.tex`, CSVs, `paper/figures/run2026_tl_sweep_curve.pdf/.png`. The figure has one row: pass rate vs median size per band with a 95% CI band, and only the production point labelled. In `tl_matched`, a pass-rate level is marked † when it lies inside the 95% CI of a method's best arm; the footer gives that arm's maximum, its CI and whether the curve is monotone. **Pending T4 / T10.** `--test` writes to `out/test/` (checked on Pythia's sweep rows) |
| `t8_triad.py` | main/eval, alpha1/alpha1, harder_nulls/nulls (draw 0, held; TuringLLM nulls = T6, drawn when present) | `paper/figures/run2026_triad.pdf/.png`, `out/run2026_triad.csv`: the 062 `triad.py` figure (fig:coef-triad) redrawn for 2026. One row per model; (a) pass rate by band and kind, (b) Z / A / C median + IQR, for fitted / α = 1 / anchor-active null / contribution-matched null. Fixed bar slots, so a missing condition leaves a gap and the panel says "pending" |
| `t7_pilot.py` | out_<model>/sweep/eval (W arms), `targets_<model>_pilot.txt`, `pilot_lambda.json` | `run2026_pilot_lambda.tex`, `out/run2026_pilot_lambda.csv/.json` |

Each `.tex` file is a bare booktabs `tabular` with no float, so it can be `\input` inside a table environment. Some
tables end with footnote-size note rows under the bottom rule. Row labels are readable text; the code arm names
stay in the CSVs. Figures are sized for the 5.5 in text width (text at 7–8.5 pt when included at `\linewidth`). Nothing
here edits `paper/main.tex`.

## Results (2026-11-08, archives T1-T3 + public S10)

**Headline (held-out strongest; mid-band beside it).** All three sanity anchors match exactly.

| Model | Targets | Fitted | Vacuous | Pass strongest [95% CI] | Pass mid-band | Size median (IQR) | Induce |
|---|---|---|---|---|---|---|---|
| TuringLLM | 8,000 | 7,786 | 322 | 6565/7464 = 88.0 [87.2, 88.7] | 3335/6228 = 53.5 [52.3, 54.8] | 609 (403-954) | 0.78 |
| Pythia-70m | 512 | 509 | 1 | 416/508 = 81.9 [78.5, 85.2] | 371/504 = 73.6 [69.6, 77.4] | 294 (132-653) | 0.88 |
| GPT-2 small | 512 | 512 | 1 | 421/511 = 82.4 [79.1, 85.5] | 338/506 = 66.8 [62.6, 70.9] | 371 (208-666) | 0.81 |

**Pass on held-out strongest by depth band.**
- TuringLLM: 90.7 / 91.2 / 82.3.
- Pythia: 89.0 / 80.6 / 78.4.
- GPT-2: 90.3 / 92.3 / 65.7.

**Pass on held-out strongest by site kind.**
- TuringLLM: residual 95.7, MLP 88.4, attention 77.5.
- Pythia: 92.7 / 86.1 / 64.8.
- GPT-2: residual 82.4, residual-mid 84.7, MLP 76.7, attention 85.8.

**Exclusions.**
- TuringLLM: 8,000 targets. 214 had no record: 205 `too_few_contexts` (fewer than 8 distinct strongest contexts on
  the deduplicated corpus; n_top 1-7) and 9 `no_contrast` (0-2 contrast contexts). Then 0 quarantined, 0 fit errors
  and 322 vacuous, leaving 7,464 scored. Mid-band: 358 thin targets have no mid pool and 1,200 mid rows are vacuous,
  leaving 6,228 scored.
- Pythia: 3 fit errors. All three are `explicit split needs both a train and a held-out slice`: their contexts rows
  have 59 training and 0 held-out strongest contexts. The strongest pools are runs of near-adjacent window ids, so
  the grouped split put everything in training. 1 vacuous.
- GPT-2: 0 final errors. 86 CUDA OOM rows were all retried to ok. 1 vacuous.

**α = 1 (same members, coefficients set to 1).** Pass drops on every model; the bracket is the paired CI of the
drop.
- TuringLLM: 88.0% to 1.8%, a drop of 86.1 pp [85.4, 86.9].
- Pythia: 81.9% to 13.8%; the survivors are almost all in L0-1 (46.5%).
- GPT-2: 82.4% to 3.3%.
- No circuit passes at α = 1 without passing fitted.

**Harder nulls (draw 0, held-out strongest).** On both public models, anchor-active and contribution-matched random
sets pass 0% (0/126 Pythia, 0/128 GPT-2), fitted or at α = 1. The production circuits pass 82.5% / 82.8% on the
same targets; the paired gap is +82.5 pp [76.2, 88.9] and +82.8 pp [75.8, 89.1]. The `wcm` arm agrees with the main
pass on 100% of targets.

**Resample and donor tests.** Pythia / GPT-2.
- 073 resample (A, C): 66.7% [58.7, 74.6] / 43.0% [34.4, 51.6]; the random null passes 0%.
- Donor test, main circuits: 66.1% [62.0, 70.3] / 41.1% [36.8, 45.4].
- Donor test among production passes: 80.8% / 49.9%.
- Pythia unweighted circuits under the donor test: 6.3%.

**Stable core (32 targets).**
- Jaccard between the two fits: 0.51 (Pythia) / 0.39 (GPT-2).
- The core alone passes 71.9% / 43.8%.
- Of the targets where both fits pass, the core passes 20/20 (Pythia) and 14/21 (GPT-2).

**Pythia, WCM vs unweighted at λ = 1e-3 (127 paired targets).**
- WCM: 106/127 = 83.5% [77.1, 89.8], median size 318.
- Unweighted: 76/127 = 59.8% [51.2, 68.5], median size 826.
- WCM leads by +23.6 pp [15.7, 31.5]. The per-target size ratio U/W is 2.28 (IQR 1.70-3.04), and the unweighted
  circuit is the larger one for 98% of targets.
- By band the WCM lead is +16.1 / +2.1 / +50.0 pp. This matches the quick check in the brief.

**λ pilot.** The pod's `pilot_lambda.json` is reproduced from the rows. Pooled median size at 1e-3 is 328 for Pythia
and 428 for GPT-2, both inside [100, 1000], so both kept 1e-3.

## Caveats and choices

- **Two rules differ from the older reports.** The nulls (084) and resample (073) scripts use the current floor 0.75
  (merge.BAND), while 084/073 `report.py` still use 0.8. The vacuous set is merge's v2 rule everywhere, where older
  reports used the v1 `vacuous_tk` flag. Both changes keep these tables consistent with the headline.
- **Size statistic.** "Size" is the median over the non-vacuous held-out-strongest circuits (paper_main's
  population). The CSV also gives the median over all fitted circuits.
- **Mid-band population.** It is its own population: its own vacuous exclusion, and no row for thin targets.
- **Bootstrap vs Wilson.** CIs are bootstrap intervals over targets, as the brief asked. `paper/tables/main_result.py`
  uses Wilson intervals instead.
- **073 resample is public only.** The TuringLLM chain has no 073 stage, so it is "not yet available" by design.
- **Stable core.** Fit 1 is the production row. Targets whose fit-1 row is vacuous are dropped (none were). A core
  that is missing or vacuous counts as a fail.
- **Matched-faithfulness reading (a) in `t6_tl_sweep.py`.** It interpolates in log size along each method's curve. A
  size marked ≤ means even the smallest arm already reached the level, so the size is an upper bound and the ratio is
  a bound. Reading (b) uses per-target smallest circuits: the smallest passing circuit, and merge.py's worst-of-3
  deviation <= 0.3. On Pythia's mixed-population sweep, `--test` only exercises the code; those numbers are not
  results.
