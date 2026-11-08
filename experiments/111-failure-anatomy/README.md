# 111 failure anatomy and remaining production-run numbers (DAN-183)

## Question

Which part of the pass rule do the failing production circuits break, by depth band and site kind? The scripts here
also fill the other `\pending` placeholders in `paper/main.tex` that the production archives can answer: the
pre-activation pass rate, the near-threshold stratum, specificity and amplifiers, training curves, fit time,
vacuous rows by term, held-out set sizes, induce >= 0.5, and the stable core by band.

## Inputs and rules

- Inputs are the unpacked archives under `../109-pod-run-analysis/out/archives/`. They are read through `lib109`,
  which `lib111.py` imports with `OUT109` pointed at `111/out`, so nothing is written into 109. `lib111` also patches
  the table provenance line so that it names 111.
- The pass rule, the vacuous set (merge v2), the depth bands, and the bootstrap (B = 4000, seed 20261108, over
  targets) are 109's.
- The pre-activation pass rate is `paper/tables/common.pre_passes` (protocol §12.1). Z is free0_raw_pre / a_pos_pre.
  Necessity is (a_pos_pre - phi_sup_blind_raw_pre) / a_pos_pre. A and C use freeM_topk_pre and freeN_topk_pre. It is
  scored on the same non-vacuous set as the activation read.
- The amplifier flag and the near-threshold stratum come from `src/eval/wcm_specificity` (`amplifier_flag`,
  `target_specificity`).
  - Amplifier: under some π in {Z, A, C}, target_faith_pre >= 0.5 and the median sibling is >= 0.8 x target. This is
    the main.tex app:specificity definition.
  - Near-threshold: rank_clean >= K/2, where rank_clean is the median clean rank at the held-out strongest anchors,
    uncapped. K is 128 for TuringLLM, 32 for GPT-2 and 16 for Pythia.
- Outcome classes follow `062 paper_main.outcome` and are exclusive, in order: pass, undefined, necessity < 0.9, Z out
  of band, otherwise A and/or C only. The per-test flags (low or high per term) are non-exclusive.
- The training-curve statistics follow `062 curves.py` (LAST = 100 steps; within 2x of final size; settled vs
  still-shrinking halves within each band). They use the production bands, not curves.py's curve_bands.

## Scripts (`python -X utf8 experiments/111-failure-anatomy/run_all.py`, ~40 s CPU, log in `out/run_all.log`)

| Script | Writes |
|---|---|
| `a1_anatomy.py` | `paper/tables/run2026_anatomy.tex`, `run2026_near_threshold.tex`, `run2026_vacuous_terms.tex`, `run2026_induce.tex`; `paper/figures/run2026_failure_anatomy.pdf/.png`; `out/run2026_anatomy_{outcomes,tests,combos}.csv`, `run2026_mid_deviation.csv`, `run2026_vacuous_terms*.csv` |
| `a2_specificity.py` | `run2026_specificity.tex`; `out/run2026_specificity_by_pi.csv`, `run2026_amplifiers.csv` |
| `a3_curves.py` | `paper/figures/run2026_training_curves.pdf/.png`; `out/run2026_training_curves.csv`, `run2026_a3.json` |
| `a4_runinfo.py` | `run2026_fit_time.tex`, `run2026_heldout_sizes.tex`; `out/run2026_fit_segments.csv`, `run2026_heldout_sizes_groups.csv` (band, kind and band x kind) |
| `a5_stable_bands.py` | `run2026_stable_core_bands.tex`, `out/run2026_stable_core_bands.csv`. It asserts that the pooled numbers equal 109 `t4_robustness.stable` |

## Results (2026-11-08)

The population is held-out strongest, non-vacuous. The headline pass rates reproduce 109 exactly: 88.0 / 81.9 / 82.4.

**Failure anatomy.** Exclusive outcome shares of all scored circuits.

| Model | Fails | A/C only | Z | Necessity | Mean-only share of fails |
|---|---|---|---|---|---|
| TuringLLM | 899 | 4.7% | 4.7% | 2.6% | 38.9% |
| Pythia-70m | 92 | 1.4% | 9.3% | 7.5% | 7.6% |
| GPT-2 small | 90 | 5.5% | 8.6% | 3.5% | 31.1% |

- **TuringLLM by band.** "A/C only" grows with depth: 1.6 / 3.3 / 9.3%. Z fails go 1.9 / 4.9 / 7.7%. Necessity fails
  sit almost entirely at L0-4 (5.8%, against 0.6% deeper).
- **Layer 0.** Necessity fails 29.6% (TuringLLM), 18.8% (Pythia) and 45.2% (GPT-2), which is the embedding bypass.
- **Which tests fail.** Fails are almost all on the low side. Any score above 1.5 appears in 4% of TuringLLM fails
  (A > 1.5 in 3%) and in none of the public models' fails.
- **Attention.** The lowest-passing kind on TuringLLM (77.5%) and Pythia (64.8%), but 85.8% on GPT-2, where MLP is
  lowest (76.7%).

**Pre-activation pass rate.**
- All targets: 75.7 [74.7, 76.7] / 73.4 [69.5, 77.2] / 81.2 [77.9, 84.5].
- Near-threshold stratum: 597 targets on TuringLLM (8.0%, all of them attention), 38 on Pythia (7.5%) and 3 on GPT-2.
- On the activation read the stratum passes 59.6 / 28.9 / 66.7%; on the pre-activation read 33.5 / 5.3 / 66.7%.
- TuringLLM 2x2 split of the stratum (both / pre-activation only / activation only / neither): 21 / 12 / 38 / 28%.

**Specificity.**
- Target restoration under Z / A / C has medians of TuringLLM 0.92 / 1.00 / 0.98, Pythia 0.97 / 1.00 / 1.00 and
  GPT-2 0.92 / 0.94 / 0.93. Siblings are restored to 0.36 / 0.46 / 0.46, 0.45 / 0.53 / 0.52 and 0.32 / 0.32 / 0.30.
  Controls are restored to 0.17 / 0.23 / 0.24, 0.11 / 0.11 / 0.12 and 0.00 / 0.00 / 0.00.
- Latents lifted above the cut under A / C (median): TuringLLM 501 / 491 against 4 / 4 for the empty circuit, Pythia
  48 / 50 (10 / 10), GPT-2 61 / 59 (29 / 29).
- Amplifiers (any π): 29.0 [28.0, 30.0] / 33.7 [29.5, 37.8] / 31.3 [27.4, 35.4]%.
  - Among passing circuits: 30.4 / 35.1 / 30.6%.
  - TuringLLM by kind: MLP 44.8, residual 26.8, attention 11.3.
  - Pass and unflagged: 61.2 / 53.1 / 57.1%.

**Training curves.**
- No gate falls below the keep threshold before median step 37.
- Circuits are within 2x of their final size by median step 95 / 96 / 125.
- Shrink over the last 100 steps: median 3.2 / 2.8 / 5.2%. The share still shrinking by more than 5% is 24 / 24 / 51%.
- Within the deep band the still-shrinking half passes less often:
  - TuringLLM L8-11: 76.8 against 87.9% (difference CI [8.1, 13.9]).
  - Pythia L4-5: 61.1 against 95.8%.
  - At L0-4 the sign reverses (TuringLLM, GPT-2).

**Fit time.** H100 SXM, fits share GPUs under MPS.

| Model | Median t_fit | Summed t_fit | Main-stage wall clock |
|---|---|---|---|
| TuringLLM | 149 s | 336 process-h | 12.9 h |
| Pythia-70m | 78 s | 11.5 process-h | 0.6 h |
| GPT-2 small | 174 s | 27.5 process-h | 1.5 h |

**Vacuous rows by term.**

| Model and held-out set | Vacuous | Z / necessity | A | C |
|---|---|---|---|---|
| TuringLLM, strongest | 322 | 103 | 219 | 2 |
| TuringLLM, mid-band | 1,200 | 160 | 1,063 | 36 |

- The terms overlap: 217 of the 322 TuringLLM strongest rows are vacuous on A alone.
- The public models have 1 strongest and 5 or 6 mid-band vacuous rows, all from A.

**Held-out sizes.**
- Targets with fewer than 16 held-out strongest contexts: 383 / 240 / 165. Fewer than 8: 262 / 11 / 11. Pythia
  counts include the 3 unfitted targets with 0.
- Targets with fewer than 64 contrast contexts: TuringLLM 6 fitted (minimum 6) plus 9 with no record (0-2 contrast
  contexts). None on the public models.
- TuringLLM strongest pools come from the corpus scan: 99.999% of rows (4 store rows).

**Induce >= 0.5.** 73.9 [72.9, 74.9] / 72.0 [68.1, 75.8] / 66.1 [62.0, 70.3]%; among passing circuits 76.7 / 75.7 /
67.0%.

**Stable core by band.**
- The core-alone pass rate falls with depth: TuringLLM 83.0 / 84.8 / 30.2%, GPT-2 57.1 / 57.1 / 18.2%.
- Pythia has 32 targets and is flat (75 / 67 / 75%).

## Caveats

- **GPT-2 main ran in two launches.** 356 fits ran at 6 per GPU and were stopped by the chain restart. The other 156
  ran at 4 per GPU. The first launch's log was overwritten, so its layout comes from `chain.events`.
- **Wall clock.** Taken from shard-log timestamps. The archives do not record the GPU type, which comes from the pod
  notes.
- **Data-loss curve.** loss - penalty includes the resample term R and the rank-keep part.
