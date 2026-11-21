# 134 Members with fitted coefficient α < 1: what are they?

Daniel, 2026-11-19: "nodes where amplitudes are < 1 ... I feel like these are interesting cases".

The coefficient is α = softplus(ψ), so it is never negative. A member at α < 1 is a latent the fit kept but turned
*down* rather than removing. The production members are 31% (TuringLLM), 42% (Pythia-70m) and 41% (GPT-2 small) below
1. The paper reports only the medians and shares (app:weighted-details).

## Step 0: is the tail a decay artefact? No (CPU, 113's member parquets)

- AdamW decays ψ toward 0, which is α = ln 2. A member with no gradient would sit at about α ≈ 0.79 at v4's budget
  (Σlr ≈ 21.4 × wd 0.05).
- No model has a spike there. The tail below 1 is smooth, and no member is negative.
- About 19-23% of members sit within 0.02 of 1, the barely-trained ones.

## Step 1: census (`s1_census.py` → `out/s1_census.md`)

- **The gate m decides it.** α < 1 is almost entirely low-gate members:
  - m < 0.6: median α 1.000, 46-57% below 1;
  - m 0.6-0.8: 24-38% below 1;
  - m ≥ 0.8: ≤ 6%;
  - m ≥ 0.95: none.
- **Depth and kind matter little.** By site kind, layer distance, target band and target kind, the shares below 1 vary
  only within a few points. One exception: α < 0.5 is about twice as common at layer distance 0-1 (~10% / 6-7%) as at
  distance 6+ (~3%).
- **Off-term pilot (131, GPT-2 small, 9 targets).** The off term does not systematically push members below 1:
  - the share below 1 changes by a median of +2.3 points at w 4 (6 up / 3 down);
  - the share below 0.5 changes by −0.2 points;
  - between any two arms, about 30% of shared members fall by more than 0.1 and about 30% rise by more than 0.1. This
    happens even at w 0.25 and on the selective controls, so it is mostly refit churn.
  - So "the off term works by turning members down" is **not** supported.

## Step 2: reproducibility (`s2_stability.py` → `out/s2_stability.md`; TuringLLM, 126 targets with a second production fit)

| fit-1 α | members | kept in fit 2 | of kept: fit-2 α < 1 | of kept: median fit-2 α |
|---|---|---|---|---|
| < 0.5 | 3,675 | 54% | **91%** | 0.49 |
| 0.5-0.8 | 8,378 | 55% | **72%** | 0.86 |
| 0.8-0.95 | 5,975 | 55% | 53% | 1.00 |
| 0.95-1.05 | 24,072 | 38% | 34% | 1.02 |
| ≥ 2 | 12,542 | 58% | 5% | 2.10 |

The same holds for low-gate members alone (m < 0.6: 93% / 74%). **Strongly reduced members (α < 0.8) are a reproducible
property of the target.** Of the kept ones, those below 0.5 come back at almost the same value. The band 0.8-0.95 is
fit noise. The population worth studying is **α < 0.8**: about 7-9% of members, a few dozen per circuit.

## Step 3 (GPU, planned for the night of 2026-11-19; after arena D's lrcheck): what do they do?

See the agent brief / `s3_*.py`. Questions:
1. Over-driven by the ablation (circuit-only value ≫ natural value)?
2. A brake (negative first-order contribution to the target)?
3. Redundant (decoder near-duplicates of other members)?
4. Active on silent contrast contexts (selectivity)?

Then the causal test: set the α < 0.8 members to 1, or remove them, and rescore (pass rule, silent firing, donor test).

## Step 3: design (written 2026-11-19, before any GPU run)

**Panel** (`s3_panel.py` → `out/s3_panel.json`, `s3_targets.txt`; seed 20261119). GPT-2 small, 11-07 production (WCM v4
main, 512 circuits):
- source rows: exp 130 `out/per_target.csv` (held-out strongest, non-vacuous), pass = True;
- ≥ 10 members with α < 0.8 ("low") and ≥ 10 with α in [0.95, 1.05] ("control"), from 113's member cache;
- bands L0-4, L5-7, L8-9; L10-11 skipped (16 GB card). L8-9 only circuits with n ≤ 1000;
- 3 bands × 4 kinds = 12 cells, up to 3 targets per cell drawn at random from the sorted candidates;
- queue order: round 1 (one per cell), round 2, round 3 (cell order shuffled). **Panel = the first 30** (24 = 2 per
  cell, + 6 of round 3); 6 extras run only if time allows. Any prefix is stratified.

**Reads, per member** (`s3_run.py` phase `reads`; ours). Rows: the 16 held-out strongest rows at their anchors, and the
48 contrast-train rows at their would-be anchors (argmax of the target's natural pre-activation, as the scorer picks
held-out contrast anchors). Runs: the unmodified model, and the circuit-only runs Z / A / C exactly as the scorer builds
them (members at α × re-encoded live value, everything else at the fill, SAE errors live). Each run does one backward
from the target's pre-activation at the anchor through a zero leaf added to every member site's stream.
1. **Over-driven**: od_raw = (live value in the circuit run, before α, summed over positions ≤ anchor) / (natural value,
   same positions); od_val = the same with α × live. Undefined when the natural value is 0 (reported as a share).
2. **Brake**: first-order contribution = Σ_{t ≤ anchor} value × (∂pre/∂stream · decoder column), natural and circuit
   runs, / the mean natural target pre at the held-out anchors. Brake = negative. Checked against a central finite
   difference on the CPU test.
3. **Redundant**: max decoder cosine to any other member at the same site.
4. **Selectivity**: natural value at the contrast would-be anchors / natural value at the held-out anchors; and the
   share of all contrast positions where the member is on vs the share of held-out positions ≤ anchor.
Comparison: pooled members, and per target (median over low − median over control), sign counts over targets.

**Causal arms (no refit)**, scored with the production path (062 `driver.Runner.score` = `wcm_eval.score_circuit`,
held-out strongest, the production records):
- (a) production α; (b) every low member at α = 1; (c) every low member removed; (d) the same number of random control
  members removed (`np.random.default_rng([20261119, crc32(key)])`; all controls if fewer, flagged).
- **Dose-response sweep** (added on Daniel's request): low members' α × c for c ∈ {0 (= c), 0.5, 1 (= a), 1/α (= b), 2};
  the matched control set of (d) at c = 0 (= d) and c = 2.
- Read per arm: the pass rule (Z, A, C in [0.75, 1.5], nec ≥ 0.9; vacuous = fail), median Z / A / C / nec, and from the
  donor test (`src/eval/donor_test.donor_target`, pools A / C / R as production, the production random pool, 8 draws,
  variant fit): SF_C (pool-C silent firing, mean over draws, as exp 130/131) and recovery A / C / R.
- Paired against (a) with sign counts; sensitivity = |Δ| of low vs matched control at the same multiplier.

**Order** (each phase finished for every target before the next): reads → arms (a-d) → sweep → donor. STOP_EPOCH cuts
the queue from the end, so the sweep goes before (b)-(d) would and the donor test is the first thing lost.

**CPU test** (2026-11-19, `2.resid_mid.7803`, 2 held-out + 2 contrast rows in the reads, 1 donor draw; log
`out/s3_cputest/cpu_test.log`, report `out/s3_cputest/s3_report.md`): all four phases ran (915 s on CPU).
- Arm (a) reproduces the pod's 11-07 values: max |diff| Z 0.0002, A 0.0008, C 0.0021, nec 0; SF_C 0.0011.
- Gradient × activation matches a central finite difference (scale one member's decoded contribution by 1 ± 0.05) on
  the 4 largest contributions: 15.72 vs 15.74, 13.00 vs 13.00, 6.63 vs 6.63, 4.03 vs 4.05.
- A rerun finds 0 to do (resume works); `DRY_RUN=1 bash s3_job.sh` sees lrcheck waiting and the lock absent.

## Step 3: results (local RTX 5070 Ti, 2026-11-19 17:44-18:54)

The GPU was freed early: lrcheck ran 17:04-17:42, then this job. Every one of the 36 queued targets (the 30-target
panel plus the 6 extras, all drawn by the same rule) finished all four phases. There were 0 errors and 0 watchdog
trips. Peak allocated memory was 11.6 GB; the card showed ~14.1 GB used. Seconds per target (median): reads 11,
arms 19, sweep 14, donor 68. Full tables: `out/s3_report.md`. All numbers are over 36 targets: 3,473 low members
(α < 0.8) and 4,027 control members (α in [0.95, 1.05]).

**1. Over-driven? No, the opposite.** In the circuit-only runs the low members are *starved*. Their re-encoded live
value, before α, summed over positions ≤ anchor, is a median of **0.05-0.12 × their natural value** on the held-out
rows (Z / A / C). Controls are at 0.56-0.63×. The per-target difference is negative on 30-32 of 36 targets. After α,
low members sit at 0.03-0.05× natural (controls 0.56-0.63×). Only 14-15% of low members end up above their natural
value, against 35-36% of controls (36 of 36 targets lower). The pattern is the same on the contrast rows. Low members
are also less often naturally silent at the anchors (7% vs 23%). The lower the α, the stronger the starvation: median
od_raw (A) is 0.008 at α < 0.3, 0.025 at 0.3-0.5 and 0.145 at 0.5-0.8.

**2. Brake? In the unmodified model, partly yes.**
- 55% of low members have a negative first-order contribution to the target's pre-activation at the held-out
  anchors, against 35% of controls (33 of 36 targets higher).
- Their median |contribution| is about 8× the controls' (2.0e-3 vs 2.5e-4 of the natural target pre; 33 of 36).
- Net per target: the sum over a target's low members is a median **−0.21** of the natural target pre. For controls
  it is **+0.16** (low lower on 29 of 36).
- Contrast rows: 58% vs 36% negative; net −0.13 vs +0.02.
- **Circuit runs:** the contributions are ~1e-6 because the members are starved (read 1). There the sign split is
  close to even (29% vs 27% negative), so the brake exists in the natural model but is mostly not carried into the
  circuit-only runs.

**3. Redundant? No.** The max decoder cosine to another member at the same site is 0.225 for low members vs 0.221 for
controls. The share above 0.5 is 12% vs 13% (low lower on 25 of 36 targets).

**4. Selective? Less than controls.** The natural value at the contrast rows' would-be anchors over the value at the
held-out anchors is a median **0.74 for low members vs 0.15 for controls** (low higher on 29 of 36). 62% vs 35% are
above 0.5. Low members fire more often everywhere: 3.1% vs 1.1% of contrast positions, and 4.2% vs 1.5% of held-out
positions ≤ anchor. The contrast/held-out firing ratio is about equal (46% vs 48% fire more on contrast). So low
members are dense, non-selective latents, active on both kinds of context.

**Causal arms (no refit; pass = production rule)**:

| arm | pass | median Z / A / C | median dev | median SF_C | rec A / C | resample pass |
|---|---|---|---|---|---|---|
| (a) production α | 35/36 | 0.93 / 0.94 / 0.94 | 0.088 | 0.190 | 0.82 / 0.75 | 17/36 |
| (b) low → α = 1 | 31/36 | 0.92 / 0.92 / 0.91 | 0.125 | 0.040 | 0.67 / 0.63 | 12/36 |
| (c) low removed | 25/36 | 0.88 / 0.86 / 0.85 | 0.182 | 0.046 | 0.66 / 0.58 | 5/36 |
| (d) matched control removed | 19/36 | 0.82 / 0.84 / 0.85 | 0.241 | 0.134 | 0.70 / 0.64 | 6/36 |
| low α × 0.5 | 34/36 | 0.91 / 0.92 / 0.92 | 0.117 | 0.195 | 0.78 / 0.71 | 13/36 |
| low α × 2 | 27/36 | 0.91 / 0.89 / 0.86 | 0.203 | 0.033 | 0.66 / 0.61 | 9/36 |
| control α × 2 | 31/36 | 0.92 / 0.92 / 0.91 | 0.153 | 0.155 | 0.77 / 0.73 | 14/36 |

Here dev = max |Z−1|, |A−1|, |C−1|. Paired against (a), per target:
- **(b)** costs 5 passes and gains 1. dev rises on 28 of 36. SF_C falls (median −0.016; 26 down / 8 up).
- **(c)** costs 10 passes and gains none. dev rises on 30 of 36; donor recovery falls (A −0.14, C −0.16; 33 of 36).
- **(d)**, removing the same number of near-1 members, costs more: 16 passes lost, dev +0.081 on 32 of 36.

**The low members are load-bearing, but per member less so than ordinary members.** At removal, low members move Z,
A, C and dev less than the matched controls (dev 13 more / 22 less). At × 2 they move them more (dev 23 / 13).

**Dose-response.** The pass count peaks at production: c = 0 / 0.5 / 1 / 1/α / 2 gives 25 / 34 / 35 / 31 / 27. So
the fitted α is a local optimum of the pass rule in both directions.

**Silent firing (surprise).** SF_C is **not monotone in c**: 0.046 / 0.195 / 0.190 / 0.040 / 0.033. Removing the low
members cuts SF_C about as much as *raising* them to α = 1 or × 2. Halving them leaves it unchanged. Matched controls
at × 2 barely move it (0.155), and at removal they move it less than low members do (|ΔSF_C| larger for low members
on 26 of 36 targets at both doses). The production α of the low members is near the setting that *maximises*
off-context firing, but the scores here do not say why. A guess, not tested: at their production value the low
members cancel just enough of the background to let the target through on silent contexts. Both removing them and
strengthening them (they brake in the natural model, read 2) restore the off state.

**Summary.** α < 0.8 members are not over-driven or redundant. They are dense, unselective latents that brake the
target in the full model. The ablation starves them, and the fit keeps them at a tuned low gain. Faithfulness and
pass are best at that gain. Silent firing on contrast contexts is highest near it.

**Caveats**
- **Rescore noise.** Arm (a) is a rescore on a different GPU (TF32) from the pod's 11-07 H100 scores. Median |diff|
  ≤ 0.004, but max 0.095 (A, `6.mlp.11731`, whose production A was 0.78, at the band edge: it fails here). So (a)
  passes 35/36, not 36/36, and 1-2 target pass flips between arms are within this noise.
- **Short control pools.** On 6 targets there were fewer controls than low members, so arm (d) removed all controls,
  fewer members than (c) (flagged "short" in the per-target table).
- **Units of read 2.** First order only: a central finite difference matched it on the CPU test target. Contributions
  are over positions ≤ anchor and normalised by the mean natural target pre on the held-out rows.
- **Deep band.** L8-9 only, circuits with n ≤ 1000; L10-11 are not covered.
- **Per-member medians.** The over-drive ratios are per member; members with zero natural mass are excluded from the
  ratio and reported as a share.
- **Single model.** GPT-2 small only; TuringLLM and Pythia are not tested.

**Commands** (WSL, repo root):

    .venv/bin/python -I experiments/134-sub1-coefficients/s3_panel.py
    # CPU test
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src HF_HUB_OFFLINE=1 OUT134=experiments/134-sub1-coefficients/out/s3_cputest \
      KEYS=2.resid_mid.7803 LIMIT_HO=2 LIMIT_CTR=2 DONOR_DRAWS=1 FD_CHECK=1 .venv/bin/python -X utf8 -u \
      experiments/134-sub1-coefficients/s3_run.py
    # GPU (after writing sub1-134 into .gpu-lock; the job refuses otherwise and never removes the lock)
    STOP_EPOCH=$(date -d 'tomorrow 07:45' +%s) bash experiments/134-sub1-coefficients/s3_job.sh \
      > experiments/134-sub1-coefficients/out/s3/job.out 2>&1
    # report (CPU)
    PYTHONPATH=src .venv/bin/python -B experiments/134-sub1-coefficients/s3_report.py

Outputs: `out/s3/rows.jsonl` (one row per target × phase), `out/s3/members/<key>.npz` (member arrays),
`out/s3/logs/`, `out/s3/watch.csv`, `out/s3_report.md`.

**Reused, not ours**: driver.Runner and its setup (as exp 131 `run.py`), `wcm_eval` scorer and patchers,
`eval.floors` fills, `donor_test`, the watchdog / trip logic of exp 131 `queue.sh`. Ours: `s3_panel.py`, `s3_run.py`
(the member reads, the gradient patchers, the arm variants), `s3_job.sh`, `s3_report.py`.

## Step 3b: Pythia-70m (local RTX, 2026-11-20 02:37-02:58)

Same scripts, switched by `MODEL134=pythia-70m` (default `gpt2-small`). With the default, the GPT-2 outputs are
unchanged: a rerun of the panel gives the same queue and rule, and a rerun of the report is byte-identical. Pythia
writes `s3py_targets.txt`, `out/s3py_panel.json`, `out/s3py/`, `out/s3py_report.md`. Circuits: the 11-07 production
archive `109.../archives/pub-pythia-70m-S10-results/.../out_pythia-70m` (WCM v4 main), the same archive that 113's
`members_pythia-70m_main.parquet` and 130's per_target.csv read.

**Design changes, and the model differences that matter.**
- **Panel.** Pythia has 6 layers and sites attn / mlp / resid (no resid_mid), so there are 3 bands (L0-1, L2-3,
  L4-5) × 3 kinds = 9 cells. Each cell gets 4 draws (seed 20261119; round 4 shuffled), 36 targets in all: the
  30-target panel plus 6 extras. There is no n cap; the deep circuits run up to n = 1384. All cells had ≥ 8
  candidates.
- **SAEs.** EleutherAI sparsify Top-K, **k = 16** (not ReLU), 32k latents, with **no input normalisation**. GPT-2's
  are OAI v5 Top-K k = 32 on the layer-normed stream. So for Pythia the site tensor the reads perturb and
  differentiate is the model-space stream itself. The reads code is unchanged.
- **Parallel residual.** attn and mlp both read the layer input. Members at layer-0 attn / mlp (and same-layer
  attn / mlp in general) therefore never see an upstream edit. Their circuit-run live value equals the natural one
  exactly: **36% of low and 49% of control members** with natural mass, against 3% / 6% on GPT-2. This dilutes read 1
  (below).
- **Contexts.** The production records are in the same format, so nothing changed there.

**Checks.**
- **CPU test** (`2.resid.32507`, 2 + 2 rows, 1 donor draw; `out/s3py_cputest/`): all 4 phases ran in 1,123 s. A
  rerun finds nothing to do.
  - Finite difference vs gradient × activation, 4 largest contributions: 3.654 vs 3.655, 3.303 vs 3.303, 2.111 vs
    2.112, 1.755 vs 1.756.
- **GPU run:** 36 / 36 targets × 4 phases, 0 errors, 0 trips, peak 4.3 GB. Median seconds per target: reads 3,
  arms 5, sweep 4, donor 19.
- **Arm (a) vs the pod's 11-07 values:** median |diff| ≤ 0.0004. Max |diff|: Z 0.017, A 0.038, C 0.039, nec 0,
  SF_C 0.0077. (a) passes 36 / 36, so no target flips at (a).

**Results** (36 targets: 2,292 low members, 4,787 control members; GPT-2 values in brackets).
1. **Over-driven / starved: the starvation does NOT replicate.**
   - Median od_raw is 1.000 for both groups, because of the exact ties above.
   - Over members whose value the edit changed, median od_raw is **0.55 for low vs 0.47 for controls** [GPT-2:
     0.065 vs 0.51]. Low is higher on 24 of 36 targets [GPT-2: lower on 30 of 35].
   - No trend with α: od_raw A is 0.97 / 1.00 / 1.00 in the α bins < 0.3 / 0.3-0.5 / 0.5-0.8.
   - What does replicate: low members are less often naturally silent (11% vs 27% [7% vs 23%]; 31 of 36 targets).
     After α they sit below natural (od_val 0.50 vs 1.00), but that follows mechanically from α < 0.8.
2. **Brake: replicates.**
   - In the natural model, 50% of low members are negative vs 30% of controls [55% vs 35%]; low higher on 33 of 36.
   - Median |contribution| 2.8e-3 vs 3.1e-4 of the natural target pre (about 9× [8×]).
   - Net per target: **−0.10 vs +0.18** [−0.21 vs +0.16]; low lower on 33 of 36.
   - Contrast rows: 56% vs 36% negative, net −0.10 vs +0.004.
   - Unlike GPT-2, the sign split partly survives into the circuit runs (39-42% vs 25% negative [29% vs 27%]),
     because the members are not starved there.
3. **Redundant: no, or barely.** Max decoder cosine 0.241 vs 0.221 [0.225 vs 0.221]. Share above 0.5 is 14% vs 8%
   [12% vs 13%]; low higher on 23 of 36.
4. **Selective: less than controls, replicates.**
   - sel **0.53 vs 0.135** [0.74 vs 0.15]; low higher on 25 of 36. Share above 0.5: 51% vs 30%.
   - Firing on contrast positions 3.0% vs 0.7%, on held-out positions ≤ anchor 3.8% vs 0.95%. The ratio between
     the two is about equal (44% vs 47% fire more on contrast).

**Causal arms (GPT-2 in brackets).**

| arm | pass | median dev | median SF_C | rec A / C | resample pass |
|---|---|---|---|---|---|
| (a) production | 36/36 [35] | 0.092 | 0.012 | 0.93 / 0.89 [0.82 / 0.75] | 29/36 [17] |
| (b) low → 1 | 30/36 [31] | 0.158 | 0.0074 | 0.91 / 0.87 [0.67 / 0.63] | 23/36 [12] |
| (c) low removed | 23/36 [25] | 0.197 | 0.0053 | 0.77 / 0.74 | 11/36 [5] |
| (d) matched ctl removed | 22/36 [19] | 0.144 | 0.0100 | 0.79 / 0.78 | 15/36 [6] |

Paired against (a):
- **(b)** loses 6 passes and gains 0; dev rises on 29 of 36. SF_C falls on 31 of 36.
- **(c)** loses 13 / gains 0; dev rises on 30 of 36; recovery A falls on 35 of 36 (−0.15).
- **(d)** loses 14 / gains 0; dev rises on 31 of 36.
- **Unlike GPT-2,** removing the low members moves Z / A / C more than removing matched controls does (C: 27 of 36
  targets), and costs about as many passes (13 vs 14 [10 vs 16]). Per member, Pythia's low members are at least as
  load-bearing as near-1 members. At × 2 they again move the scores more (dev 21 / 14 [23 / 13]).

**Dose-response** at c = 0 / 0.5 / 1 / 1/α / 2:
- pass **23 / 31 / 36 / 30 / 28** [25 / 34 / 35 / 31 / 27]: the peak is at production, as on GPT-2.
- SF_C **0.0053 / 0.013 / 0.012 / 0.0074 / 0.0068** [0.046 / 0.195 / 0.190 / 0.040 / 0.033]. The shape is the same
  non-monotone one: both removing and raising cut SF_C, and halving leaves it. The level is ~15× lower. Pythia's
  production SF_C is small on most targets; four L4-5 / L2-3 attn targets carry 0.16-0.94.
- Matched controls at × 2 barely move SF_C (0.0089). |ΔSF_C| is larger for low members on 26 of 36 targets at
  removal and 24 of 36 at × 2.

**Summary, two models.**
- **Replicates:** the brake in the natural model, low selectivity / dense firing, no redundancy, production α as
  the pass optimum in both directions, and the non-monotone silent firing.
- **Does not replicate:** the starvation in the circuit-only runs. That was GPT-2-specific, and it is the
  "the ablation starves them" clause of the step-3 summary. On Pythia, low members keep about the same share of
  their natural value as controls do.
- **Differs:** on Pythia, the per-member load is larger for low members than for controls at removal, not smaller.

**Caveats.**
- Different GPU and dtype path from the pod (max |diff| 0.04 on A / C).
- Pythia's SF_C levels are small (median 0.012), so its non-monotone shape rests on small absolute changes. The sign
  counts are clear, though: 31 and 34 of 36 targets go down under (b) and (c).
- In read 1, 36-49% of members are exactly untouched because of the parallel residual. The "changed only"
  comparison above is computed from `out/s3py/members/*.npz` (inline analysis, not in `s3py_report.md`).

**Commands** (WSL, repo root): `MODEL134=pythia-70m .venv/bin/python -I .../s3_panel.py`; CPU test as step 3, with
`MODEL134=pythia-70m OUT134=.../out/s3py_cputest KEYS=2.resid.32507`; GPU:
`MODEL134=pythia-70m S134_LOCK_TAG=sub1py-1120 STOP_EPOCH=... bash .../s3_job.sh > .../out/s3py/job.out 2>&1`.
