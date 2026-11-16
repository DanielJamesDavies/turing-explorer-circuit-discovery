# 113 coefficients shape: the fitted α of the WCM v4 production circuits (CPU only)

## Question

What do the fitted coefficients α of the production circuits look like (DAN-44)? And four related checks: off-manifold
scaling, per-context spread, a symmetric pass band, effective size, and the overlap between weighted and unweighted
circuits.

## Inputs

- The production archives unpacked by 109 (`../109-pod-run-analysis/out/archives/`). All locations, row readers, the
  pass rule and the vacuous rule come from `lib109.py`, which imports 062 `merge.py` (`passes`, `vacuous`).
  - Main circuits: TuringLLM T3 `out_v2/main/circuits` (7,786); Pythia and GPT-2 `out_<model>/main/circuits` (509 /
    512).
  - Sweep circuits: TuringLLM T4 `out_v2/sweep/circuits/<arm>`, 185 targets × 13 arms.
- Corpus active counts (`latent_stats.active_count`), used for the frequency deciles:
  - TuringLLM: the local store `outputs/latent_stats.pt`.
  - Public models: the pod stores. `latent_stats.pt` and `top_ctx.pt` were extracted from
    `Runs/pod-results-2026-11-07/pub-<model>-S10-stores.tgz` into `out/stores/`.
  - The local `outputs_public/` stores are the 2,048-sequence smoke pass, so they are not used.
  - Check: on all 366 harder-null panel targets (TuringLLM 112, Pythia 126, GPT-2 128), the members' frequency-decile
    histograms reproduce the 084 profiles' `set_wcm.fdec_hist` exactly (`out/run2026_coef_freq_check.csv`).
- TuringLLM SAE decoders: `models/TuringLLM/SAE/sae-<kind>/sae_<kind>_layer_<L>.pth` (`decoder.weight`, 1024 ×
  40,960). These load on the CPU.

## Definitions

- **Population.** Main circuits whose held-out-strongest eval row is non-vacuous (merge.vacuous): TuringLLM 7,464,
  Pythia 508, GPT-2 511.
  - Every member counts; the seed node does not.
  - The member counts equal the eval rows' `n` for all 8,807 circuits (`build_members.py`).
- **α.** The member node's `amplitude`, the engine's converged coefficient rounded to 4 dp.
  - "α ≈ 1" means |α − 1| < 0.01, i.e. left at its initial value.
- **Layer distance.** Target layer minus member layer. A distance of 0 means a member at an earlier site of the same
  layer.
- **Frequency decile.** Defined per site over the latents with a nonzero corpus count; this is 084
  `nulls_core.freq_deciles`, imported. A value of −1 means the latent never fired in the store.
- **CIs.**
  - Pass rates and per-target means: 95% percentile bootstrap over targets (4,000 resamples, seed 20261114).
  - Pooled member shares: a target-cluster bootstrap of sum(k) / sum(n).
  - Band differences: a paired bootstrap.

## Scripts

Run them from the repo root with `python -I -X utf8 experiments/113-coefficients-shape/<script>`. `run_all.py` runs all
of them, takes about 6 minutes on a first run, and writes `out/run_all.log`.

| Script | Does | Writes |
|---|---|---|
| `lib113.py` | Member cache, frequency store, cluster bootstrap, table writer (wraps lib109) | |
| `build_members.py` | Reads every circuit once and checks member counts against eval `n` | `out/members_<model>_<arm>.parquet` |
| `c1_alpha.py` | Item 1: histogram, per-circuit quantiles, shares, breakdowns | `run2026_coef_alpha_summary.tex`, `run2026_coef_alpha_breakdown.tex`, `paper/figures/run2026_coef_alpha_hist.pdf/.png`, `out/run2026_coef_alpha_*.csv`, `out/run2026_coef_freq_check.csv` |
| `c23_stored.py` | Items 2 and 3: lists what the archives store | `out/run2026_coef_stored.json` |
| `c4_band.py` | Item 4: symmetric bands | `run2026_coef_band.tex`, `out/run2026_coef_band.csv` |
| `c5_effective.py` | Item 5: N50 / N90 | `run2026_coef_effective.tex`, `out/run2026_coef_effective*.csv` |
| `c6_overlap.py` | Item 6: WCM vs unweighted overlap and decoder cosine | `run2026_coef_overlap.tex`, `out/run2026_coef_overlap*.csv` |
| `inspect_pt.py` | Prints the structure of a `.pt` file (helper) | |

## Results

### 1. Coefficient distribution (DAN-44)

Source: `out/run2026_coef_alpha_summary.csv` and `out/run2026_coef_alpha_by_*.csv`.

| Model | Members | Median α | Per-circuit median (p10 / p90 / max), median over circuits | α < 1 | α > 1 | α ≈ 1 | α > 2 | α < 0.4 | max |
|---|---|---|---|---|---|---|---|---|---|
| TuringLLM | 5,494,269 | 1.108 | 1.11 (0.79 / 2.23 / 4.37) | 31.2% [30.9, 31.5] | 67.9% [67.5, 68.2] | 17.0% | 14.1% | 2.4% | 11.6 |
| Pythia-70m | 248,561 | 1.006 | 1.01 (0.71 / 1.79 / 3.63) | 42.2% [41.5, 42.8] | 57.0% [56.4, 57.7] | 18.9% | 6.0% | 2.2% | 6.5 |
| GPT-2 small | 256,315 | 1.020 | 1.07 (0.68 / 2.21 / 4.69) | 40.7% [39.5, 42.0] | 58.3% [57.0, 59.6] | 15.6% | 12.1% | 2.9% | 10.6 |

- **Shape.** The distribution has a sharp peak at α = 1 and a right tail. It is not bimodal, and there are almost no
  off-switches: α < 0.1 is very rare, the minimum α is about 0.008, and α < 0.4 covers 2–3% of members.
- **Most circuits amplify more members than they damp.** The share of circuits with a majority of members at α > 1 is
  94% (TuringLLM), 93% (Pythia) and 91% (GPT-2).
- **By site kind.** TuringLLM residual members are amplified most (median 1.17, 70.7% > 1). Attention members are
  amplified least (median 1.03, 62.2% > 1). On GPT-2 and Pythia the order is reversed: attention and MLP members are
  amplified more than residual ones.
- **By layer distance.**
  - TuringLLM: amplification peaks at distance 3–4 (71.7% > 1) and is lowest at distances 1 and 8–11 (63–64%).
  - GPT-2: flat at about 59–60% out to distance 3–4, then falls to 53% at 8–11.
  - Pythia: lowest at distance 0 (48%) and about 57–60% beyond.
- **By frequency decile.** This is the clearest pattern. The top corpus-frequency decile holds 75% of TuringLLM members
  (4.10 M of 5.49 M) and 65% of GPT-2 and 63% of Pythia members.
  - In that decile, α is near 1: medians 1.071 (TuringLLM), 1.000 (GPT-2) and 1.003 (Pythia). The share above 1 is
    64.4%, 50.7% and 55.0%.
  - Rarer members are amplified more:
    - TuringLLM deciles 0–8: 76–79% above 1, medians 1.15–1.25.
    - GPT-2 deciles 0–8: 70–74% above 1.
    - Pythia: only 57–62% above 1.
  - 28 GPT-2 members (20 circuits) never fired in the 4 M-sequence store.

### 2. Off-manifold check

**Not computable from the archives. Stopped here, as briefed.**

- What is stored:
  - Circuits store only α and the gate m per member (`out/run2026_coef_stored.json`).
  - Eval rows store means, not member activations.
  - The ctx records store only the target's own activations (`strong.peak`).
- A reference for each latent's natural maximum does exist: the per-latent top-64 sequence maxima in `top_ctx.pt`
  (`ctx_seq_val[comp, latent, 0]`).
  - For TuringLLM it is the local `outputs/top_ctx.pt`; for the public models it is in `out/stores/`.
  - No corpus 99th percentile is stored. `latent_stats` has only count, mean and second moments.
- **Pass needed.** For each circuit:
  - Run the model on its 16 held-out-strongest (and 48 training) contexts.
  - Read every member's post-Top-K code at the anchor (or at every position up to the anchor, which is the actual
    edit).
  - Compare α × a with `top_ctx` max[member]. For a 99th-percentile reference, also add a streaming quantile sketch
    to the data pass.
  - This is one forward pass per circuit, the same as wcm_eval's `pins_tr` capture.
- **Bound from α alone.** A member with α ≤ 1 can never be pushed above its natural range. Only the α > 1 members
  (68% / 57% / 58%) can be, and only those with α > 2 (14% / 6% / 12%) can exceed the maximum from an activation below
  half of it.

### 3. Per-context spread of faithfulness

**Not stored.**

- Each eval row stores Z, A and C only as ratios of means over the 16 held-out contexts:
  (mean circuit read − mean empty read) / (mean natural − mean empty), as in `wcm_eval.score_circuit`.
- The only per-context-like field is `a_pos_blocks`: four 16-context block means of the natural target activation over
  all 64 strongest contexts (train and held-out), kept as a split-ordering diagnostic. It contains no circuit reads.
- A per-context spread needs a rerun of `score_circuit` that keeps `read_target`'s per-sequence values
  (`per_seq=True` is already supported for the natural tap).

### 4. Symmetric band

Source: `out/run2026_coef_band.csv`. The production column reproduces the headline exactly (6,565 / 7,464, 416 / 508,
421 / 511).

| Model | [0.75, 1.5] production | [0.667, 1.5] | [0.75, 1.333] |
|---|---|---|---|
| TuringLLM | 88.0 [87.2, 88.7] | 92.0 [91.4, 92.6], +4.0 pp [3.6, 4.5] | 87.5 [86.8, 88.3], −0.4 pp [−0.6, −0.3] |
| Pythia-70m | 81.9 [78.5, 85.2] | 87.0 [83.9, 90.0], +5.1 pp [3.1, 7.1] | 81.9 (identical) |
| GPT-2 small | 82.4 [79.1, 85.7] | 87.7 [84.7, 90.6], +5.3 pp [3.5, 7.2] | 82.4 (identical) |

- Failures sit on the low side. The upper cap is almost never what fails a circuit: tightening it to 1.333 costs only
  32 TuringLLM circuits and none on Pythia or GPT-2.
- The low-side gain is largest in the deep band: TuringLLM L8–11 +7.8 pp, GPT-2 L8–11 +12.0 pp.

### 5. Effective size

**No contribution measure is stored per member.**

- The node's `attribution_score` is the converged mask gate sigmoid(θ), which lies in (0.5, 1] for kept members
  (`learned_mask.py` selection). The only other per-member number is α. Neither is a contribution.
- N50 / N90 by those weights are recorded for completeness (`out/run2026_coef_effective.csv`). They are near-uniform:
  - By |α|: 32–36% of members carry 50% of the weight and 79–84% carry 90%.
  - By m: 41–44% and 87–89%.
  - These shares are the same in every model and band.
- A true contribution-weighted effective size needs the 084 profile contribution per member (activation × gradient of
  the target pre-activation at the anchor). The archives keep that only as per-set summaries
  (`harder_nulls/profile*.jsonl`: `contrib_median`, deciles), so it needs a GPU pass. **Do not quote these N50/N90 as
  contribution concentration.**

### 6. Weighted vs unweighted overlap (TuringLLM sweep)

Source: `out/run2026_coef_overlap.csv` and `out/run2026_coef_overlap_targets.csv`.

- **Primary pairing:** the smallest passing WCM circuit vs the smallest passing unweighted circuit (109 t6 reading
  (b)). 142 of the 173 scored targets have both.
  - The WCM circuit usually comes from the sparsest arm (λ = 4e-3 for 103 targets).
  - Median sizes are 284 (WCM) and 1,793 (unweighted).
  - Share of WCM members inside the unweighted circuit: mean 0.81 [0.79, 0.83], median 0.85. Only 3.5% of WCM circuits
    are strict subsets, and 29% have at least 90% of their members inside.
  - Jaccard: mean 0.13 [0.12, 0.15].
- **Same λ = 1e-3:** 173 targets.
  - WCM inside unweighted: 0.59 [0.57, 0.62].
  - Jaccard: 0.26 [0.24, 0.27].
  - No WCM circuit is a subset.
- **Near-duplicates.** 1.60 M unweighted-only members; 100 k of them sit at sites with no WCM member, and 421 k were
  sampled (at most 500 per target and site).
  - Largest same-site decoder cosine to a kept WCM member: median 0.072. For random latents of the same site the
    median is 0.068; random pairs give 0.003.
  - Above 0.3: 1.1% vs 0.1% for random latents. Above 0.5: 0.1% vs 0.0%. Above 0.7: 0 in either.
  - **Unweighted-only members are not near-duplicates of kept WCM members.** They are slightly more aligned than
    random latents of the same site, but almost none share a direction.

## Caveats

- **α is the converged coefficient only.** Members with α ≈ 1 may simply not have moved (the 084 `alpha_inert`
  notion). The 15–19% at |α − 1| < 0.01 is an upper bound on "untouched".
- **"Smallest passing" picks the sparsest arm that passes.** For WCM that is often W_0.004, below production size.
  The λ-matched pairing is reported beside it for that reason.
- **The cosine compares decoder directions at the same site only.** Cross-site duplicates, for example the same
  feature carried in resid L and attn L+1, are not tested.
