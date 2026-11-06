# 104: WCM_VERSION=v3 (the final production algorithm): native port, bit-identity, cross-model check, smoke, budget

1 November 2026 (agent prepod). Daniel approved the final production algorithm on 2026-11-01: Z + A + C + R_C at
λ 1e-3 (v2cr) with **R weight (γ_R) 1.0** and **rank-keep weight 1e-1**, = exp 102's arm `r1.0_rk1e-1` (check panel:
production 48 → 54, resample 19 → 39 of 72).

## Status (2026-11-01)

| part | what | status |
|---|---|---|
| code | `WCM_VERSION=v3` in the 062 driver; per-version rank-keep (`WCM_RANK`); tests | **done** |
| A | bit-identity of native v3 with 102's `r1.0_rk1e-1` | **done: 3 / 3 bit-identical** |
| docs | `chain.sh` default v3; runbook "2026-11-01" line; protocol §12.3 final settings | **done** |
| B | cross-model check (Pythia-70m, GPT-2 small) | **done** 03:51-07:12 (WSL clock, GMT): resample gain carries over, production not clearly hurt (below) |
| D | clean TuringLLM timing panel, then the pod budget | **done** (panel 07:12-08:17; budget below) |
| C | full chain smoke on all three models under v3 (fresh WORK `104/smoke_chain`) | **done** 08:17-09:18: rc 0, 0 error rows |

The GPU was released at 09:19 (WSL clock). No 104 job runs after it.

GPU queue after B: `run_queue.sh` (waits for `run_cross.sh`, then `run_timing_tl.sh`, then the smoke `PHASE=tl`,
then `PHASE=all`); events in `logs/queue.events`. Restart after a stop:
`nohup bash experiments/104-v3-cross-model/run_queue.sh > experiments/104-v3-cross-model/logs/queue.out 2>&1 &`
(every step resumes; restart `run_cross.sh` first if B was not finished).

## Code (part "code")

`experiments/062-h100-protocol-v1/driver.py`:
- `WCM_VERSIONS["v3"]` = v2cr's table with `resample_weight` 1.0. `WCM_RANK` = {v1, v2, v2cr: 3e-3, v3: 1e-1}.
- `RANK = WCM_RANK[WCM_VERSION]`. RANK is the sweep grid's default and the reference for the `_r<weight>` arm-name
  suffix. So under v3, `W_0.001` means rank-keep 1e-1 and `W_0.001_r0.003` means the old weight; under v2,
  `W_0.001_r0.1` names the rank-keep-only variant. The version block moved above the sweep grid for this.
- `Runner.method(..., rank=None)`: None = the version's weight. This applies to the main fit, every sweep arm
  (weighted or unweighted) and every companion that calls `R.method(GAMMA, LAM, ...)`.
- R overrides (`R_WEIGHT` / `R_POOL` / `R_K`) stay v2cr-only; v3 refuses them.
- `OUT/WCM_VERSION` = v3, `OUT/WCM_R` = {1.0, contrast, 1}. v2cr and v3 refuse each other's OUTs.
- v1 / v2 / v2cr are unchanged: same tables, rank-keep 3e-3, same arm names.

`tests/test_wcm_version_driver.py`: +5 tests:
- the v3 table and the per-version RANK and grid;
- v3 refuses R overrides;
- arm names follow the version's rank-keep weight;
- `Runner.method` (engine builder stubbed, CPU) leaves rank-keep 1e-1 + R 1.0 + sparse fill on main, unweighted and
  every default sweep arm under v3, and rank-keep 3e-3 under v2 / v2cr;
- the v3 OUT marker and the v2cr / v3 cross refusals.

The old "unknown version" test used `v3`; it now uses `v9`. 10 / 10 pass (WSL).

## Part A: bit-identity (done 2026-11-01)

`run_bitident.sh` ran the 062 driver, `MODE=main`, `WCM_VERSION=v3`, on 102's records (`062/out_full/out/ctx`,
explicit split) for 3 check-panel targets, one per band (`targets_bitident.txt`). `compare_bitident.py` compared them
with `102/out/check/circuits/r1.0_rk1e-1` (-> `out/bitident_compare.json`).

| target | members (v3 / 102) | members + α | max abs α diff | loss curve, every term curve | held-out strongest scores |
|---|---|---|---:|---|---|
| 2.attn.11852 | 320 / 320 | identical | 0 | identical | 131 compared, max diff 0 |
| 6.mlp.36361 | 702 / 702 | identical | 0 | identical | 131 compared, max diff 0 |
| 9.resid.22205 | 1452 / 1452 | identical | 0 | identical | 121 compared, max diff 0 |

There is no float noise. Fit times: 40 / 98 / 137 s, against 102's 33 / 89 / 131 s. The CPU test suite ran at the
same time, so these times are not clean.

## Resume commands (WSL, repo root; every step skips finished work)

```
bash experiments/104-v3-cross-model/run_bitident.sh                     # A: 3 TuringLLM fits under v3 (~5 min + load)
PYTHONPATH=src .venv/bin/python experiments/104-v3-cross-model/compare_bitident.py
nohup bash experiments/104-v3-cross-model/run_cross.sh > experiments/104-v3-cross-model/logs/run_cross.out 2>&1 &   # B
PYTHONPATH=src .venv/bin/python experiments/104-v3-cross-model/report_cross.py
bash experiments/104-v3-cross-model/run_smoke_chain.sh                  # C: chain smoke, all three models, v3
```
Progress: `experiments/104-v3-cross-model/logs/run_cross.events`, `logs/fit_<model>_<v>.log`, `logs/donor_<model>_<v>.log`.

## Part B design (frozen 2026-11-01 before any 104 fit)

- **Question:** do the v3 settings help, or at least not hurt, on the public models?
- **Arms:** `v2` (no R, rank-keep 3e-3) vs `v3` (R 1.0, rank-keep 1e-1), both λ 1e-3, 400 steps, the 062 driver,
  MODE=main (SKIP_SPEC=1), each arm its own version-marked OUT on the same records (CTX_DIR); then
  `src/eval/donor_test.py` on both (8 draws, variants fit / a1 / pinned, one R pool per model shared by the arms).
- **Targets** (`select_targets.py`, rule in its docstring): ~24 per model, 8 per depth band, kinds dealt round robin,
  drawn (seed 20261101) from 082's preliminary frame lists (`082/prelim/<model>/targets_<model>_main.txt`); records
  built from the local 069 smoke stores (the stores the 082 prelim used; no data pass). Eligible = strongest pool,
  ≥ 8 held-out strongest, ≥ 32 contrast contexts. Bands: GPT-2 L0-4 / L5-7 / L8-11 (as TuringLLM); Pythia (6 layers)
  L0-1 / L2-3 / L4-5 (equal thirds; L0 has resid only).
- **Report** (`report_cross.py`): production and resample pass, size, fit time per arm, paired; bootstrap 10,000,
  seed 20261101, 95% percentile CI of the v3 − v2 pass-rate difference; floor 0.75 (protocol §6 / §12.3).
- **Stop rule:** if v3's production CI lies entirely below 0 on a model, stop after the write-up and flag it to
  Daniel (no production change is triggered by this experiment).

## Part B results (2026-11-01, 03:51-07:12 WSL clock: 3 h 20 min of GPU, under the 4 h stop)

The run had 0 error rows (fits and donor) and 0 vacuous rows. The donor rows' `pass_prod` equals the eval rows' pass
on every target. Tables: `out/tables.md`, `out/report.json`.

**Targets.**
- Pythia: 40 records tried to fill 24 slots. 15 were thin (< 8 held-out strongest). 1 record build failed (`5.attn.15219`:
  `TypeError: keywords must be strings` inside the record builder, on this smoke-store record; logged, counted
  ineligible, not investigated).
- GPT-2: 45 tried, 21 thin.
- No slot was passed between kinds.
- The records are the smoke stores': no mid band, 8-16 held-out strongest, and 14 / 17 of 24 have the full 48
  training contexts.

| model | n | arm | production | resample | d production [95% CI] (w/l) | d resample [95% CI] (w/l) | median size | size x v2 | median fit s | fit x v2 |
|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|
| Pythia-70m | 24 | v2 | 11 | 0 | - | - | 321 | 1.00 | 27.8 | 1.00 |
| Pythia-70m | 24 | v3 | 10 | 5 | -4 pp [-17, +8] (1/2) | **+21 pp [+4, +38] (5/0)** | 747 | 1.91 | 38.5 | 1.42 |
| GPT-2 small | 24 | v2 | 12 | 0 | - | - | 484 | 1.00 | 104.7 | 1.00 |
| GPT-2 small | 24 | v3 | 11 | 7 | -4 pp [-21, +12] (2/3) | **+29 pp [+12, +46] (7/0)** | 764 | 1.96 | 151.3 | 1.45 |

By band (production v2 → v3 / resample v2 → v3):

| model | band | n | production | resample |
|---|---|---:|---|---|
| Pythia | L0-1 | 8 | 5 → 6 | 0 → 4 |
| Pythia | L2-3 | 8 | 2 → 2 | 0 → 0 |
| Pythia | L4-5 | 8 | 4 → 2 | 0 → 1 |
| GPT-2 | L0-4 | 8 | 6 → 5 | 0 → 3 |
| GPT-2 | L5-7 | 8 | 4 → 5 | 0 → 4 |
| GPT-2 | L8-11 | 8 | 2 → 1 | 0 → 0 |

**Reading.**
- **The resample gain carries over.** v2 circuits pass the donor test on **no** public target (median fit recovery
  A / C: Pythia 0.05 / 0.06, GPT-2 0.01 / 0.01). v3 passes it on 5 / 24 and 7 / 24, with no loss: the CIs lie above
  0, and median recovery is about 0.5 on both models.
- **Production is not clearly hurt, but it is not helped either.**
  - The arms differ by -1 target on each model, and the CIs straddle 0. So the stop rule (production CI entirely
    below 0) is not met.
  - v3 loses production passes on deep targets: Pythia L4-5 4 → 2, GPT-2 L8-11 2 → 1. It gains them in the shallow /
    mid bands.
  - Two v3 losses fall far below the floor (GPT-2 `7.resid_mid.14465` Z 1.12 → 0.28; Pythia `5.resid.24122` Z 0.84 →
    0.44, necessity 0.83). TuringLLM (102) gained production at every depth, so this is a difference from TuringLLM.
  - Daniel should know this before the pod. At n = 24 per model it is within noise (102 measured ~14% production
    flips between refits).
- **Circuits are about 1.9-2.0× larger on the public models** (TuringLLM: ×1.33).
- **Fit time ×1.42 / ×1.45** (paired, all 24; on the full-size records ×1.41 / ×1.44).
- **Scoring time is unchanged:** median eval 1.0 → 1.1 s and 4.1 → 4.35 s; donor 8.0 → 8.3 s and 33.0 → 33.4 s per
  target.
- *Caveats:*
  - thin smoke-store records (no mid band, 8-16 held-out strongest);
  - one seed;
  - n = 24 per model, so a difference of 1-2 targets is noise.

## Part C: chain smoke under v3, all three models (2026-11-01 08:17-09:18)

`run_smoke_chain.sh` ran as 103's smoke: `LIMIT=2`, 20 steps, P = 1, GPU 0, `WORK=104/smoke_chain` (fresh).
`WCM_VERSION` was unset, so the chain's own default (v3) was under test. It ran `PHASE=tl` (public tracks stopped at
S3), then `PHASE=all`. Pre-marked done, not under test, as in 103:
- TuringLLM T0-T2, which read 082 `smoke_v3`'s fresh records and lists, and T5-T8;
- Pythia S5 and the S6 α = 1 / nulls / stable core / 073 arms;
- GPT-2 S8 and the same S9 arms.

**Result: the final phase finished rc 0, with 18 jobs and 0 error rows.** Every stage was archived and md5-verified
(19 verified copies). S10 (C4) packed both models' results and stores.
- The `PHASE=tl` call returned rc 1 only because S10 cannot pack while the public tracks are held at S3. Its log
  ends exactly as 103's.

| track | jobs (rows, errors) |
|---|---|
| TuringLLM | T3 main 2/0, α = 1 2/0; T4 W_0.001 2/0, W_0.004 2/0, U_0.001 2/0; T10 noR 2/0; T11 donor 2/0; T12 donor 6/0 + noR 2/0 |
| Pythia | S4 records; S6 main 8/0, unweighted 2/0, donor 8/0, donor_unweighted 2/0 |
| GPT-2 | S7 records; S9 main 6/0, donor 6/0 |

**The v3 marker is on every output.**
- All 57 eval / donor rows carry `wcm_version` v3.
- `out_turingllm/out_v2`, `out_pythia-70m` and `out_gpt2-small` are marked `v3` with
  `WCM_R` = {1.0, contrast, 1}.
- The T10 OUT is marked `v2`, and its rows carry rank-keep 0.003.
- Every v3 sweep row carries rank-keep 0.1, the unweighted `U_0.001` included. Its train rows carry R 1.0, 48 donors,
  the sparse fill and the `resample` term.

## Part D: the v3 pod budget (2026-11-01; `cost_v3.py` -> `out/cost_v3.json`)

### What was measured

**TuringLLM panel** (`run_timing_tl.sh`).
- 12 targets, one per layer, from 102's check panel, on the 15k run's records.
- The idle local GPU ran T3's exact job (fit + eval strong + mid + specificity, CURVE_STRIDE 1) under v1, v2 and
  v3, then T11's donor job on the v2 and v3 circuits.
- Nothing else ran on the PC: no test suite, and no other GPU process (the script refuses to start otherwise).

| target | pod v1 fit / eval s (per process, 062) | local fit v1 / v2 / v3 s | local eval v1 / v3 s | spec v1 / v3 s | donor v2 / v3 s |
|---|---|---|---|---|---|
| 0.mlp.9455 | 19 / 1.1 | 11.8 / 14.5 / 16.1 | 2.0 / 1.3 | 0.4 / 0.4 | 5.2 / 6.8 |
| 1.mlp.25912 | 31 / 1.9 | 14.8 / 19.3 / 25.5 | 2.5 / 2.5 | 0.5 / 0.5 | 7.0 / 7.5 |
| 2.resid.12046 | 42 / 2.3 | 23.0 / 28.1 / 41.0 | 3.5 / 3.5 | 0.6 / 0.7 | 11.1 / 10.6 |
| 3.attn.29509 | 46 / 1.0 | 24.4 / 31.5 / 46.0 | 1.9 / 1.9 | 0.7 / 0.7 | 13.0 / 13.7 |
| 4.mlp.21782 | 60 / 2.9 | 33.3 / 41.6 / 62.9 | 4.6 / 4.3 | 0.9 / 0.9 | 16.7 / 17.1 |
| 5.resid.4573 | 72 / 3.0 | 41.0 / 52.8 / 80.2 | 5.4 / 5.7 | 1.0 / 1.0 | 20.2 / 20.4 |
| 6.attn.38069 | 57 / 0.8 | 40.5 / 51.8 / 76.7 | 1.4 / 1.5 | 0.5 / 0.5 | 15.1 / 15.9 |
| 7.mlp.8115 | 89 / 4.6 | 52.1 / 66.8 / 101.3 | 6.9 / 7.2 | 1.2 / 1.2 | 25.8 / 27.5 |
| 8.resid.5512 | 100 / 5.1 | 60.7 / 78.5 / 117.9 | 7.9 / 8.1 | 1.4 / 1.4 | 30.1 / 31.1 |
| 9.attn.13911 | 103 / 4.6 | 62.8 / 81.5 / 122.5 | 8.3 / 8.3 | 1.5 / 1.5 | 31.2 / 32.4 |
| 10.mlp.6565 | 118 / 6.2 | 71.2 / 92.5 / 139.7 | 9.1 / 9.4 | 1.6 / 1.6 | 35.4 / 36.2 |
| 11.resid.13585 | 133 / 5.7 | 79.3 / 104.0 / 156.4 | 10.5 / 10.4 | 1.8 / 1.8 | 40.3 / 41.0 |

**Measured findings.**
- **Training:** the v3 / v2 fit-time ratio is **1.50** (median; 1.49 total), and v3 / v1 is **1.92**. That is more
  than the 1.4 assumed so far (101's 1.40 × 102's 0.97 predicted 1.36). The public models agree: ×1.41 (Pythia) and
  ×1.44 (GPT-2) on full-size records.
- **Scoring is unchanged by v3:** eval ×1.00, specificity ×1.00, donor ×1.03, though circuits are ×1.3-2 larger. So
  the old practice of applying the fit ratio to whole stages would have overstated only the scoring part. That part
  is small for the fit stages: 1.9 of T3's 54 GPU-h.
- **Local → pod ratio for TuringLLM, measured from history, separately for training and scoring.** The same 12
  targets' v1 per-process times in 062's 15k pod run (8 × H100, MPS P = 6) give, against the local v1 times:
  - training: pod per-process / local = 1.69, so one H100 does **r_fit = 3.5** local-GPU units of fit work;
  - scoring: 0.60, so **r_score = 10.0**.
  - The plan's TuringLLM r = 2.4 came from an older ~35 s local figure. For these 12 targets the local v1 fit +
    eval + spec measured here sums to 591 s, against 918 s per process on the pod (= 153 H100-s at P = 6).

### Method

- **TuringLLM fit stages:** per target, 062's pod v1 per-process seconds at that layer (mean over the 15k), with the
  fit part × the measured v3 / v1 ratio and eval / spec × 1.00, divided by 6 (P = 6 per H100). Summed over each
  stage's real target list (8k, sweep190, stable128, nulls126).
  - This uses no local → pod factor at all: the pod speed is the 062 measurement.
  - Summed per process, it slightly overstates 062's own 12.6 GPU-s per target (13.25 s; +5%, conservative).
- **TuringLLM donor stages (T11 / T12):** local donor seconds (a linear fit in the layer, v3 circuits) / r_score 10.0.
- **Public tracks:** local v3 seconds measured in part B (fit on the full-size records; eval strong; donor) and part
  C (specificity), times the plan's counts per stage. The fit part and the score part are each divided by r.
  - r has no pod measurement, so it is **estimated: r = 2.0 (1.5-2.7)**, as in the plan.
  - "TL ratios" = TuringLLM's measured 3.5 / 10.0 applied instead, an optimistic reading.
- **Unchanged from the plan:** T1 / T2 / α = 1 / data passes. T7 (083) uses its README's local-seconds estimate, so
  it is marked E.

### Stage table (rebuilds `dev-notes/2026-10-23-pod-plan-public-models.md` §0v2.4 for v3; M = measured, E = estimated)

| stage | v2 plan GPU-h | **v3 GPU-h** | basis | M / E |
|---|---:|---:|---|---|
| T1 scan | 2.8-4.2 | 3.5 | unchanged | E (plan) |
| T2 gate + build + verify | 10.6 | 10.6 | unchanged | E (plan) |
| T3 main, 8,000 | 35.8 | **54.3** | fit 52.4 + score 1.9; pod v1 per layer × v3 / v1 fit 1.92 | M |
| T3 α = 1 | 1.3 | 1.3 | score only | E (plan) |
| T4 sweep, 190 × 12 new fits | 10.2 (-15) | **15.0** (-22 if weak-λ arms run ×1.5) | as T3, strong eval only | M |
| T5 stable core, 128 × 2 fits | 2.2 | 1.7 | 2 fits + 2 strong evals | M |
| T6 nulls, 316 + 126 fits, ~1,900 scores | 3.0 | 3.1 | null fits costed as full fits | M (fits) / E (counts) |
| T7 DAN-159 (083) | 26 | **15.9** (28 at r = 2) | 56.5 local-h (incl. +110 s v3 support fits per target) / r_fit 3.5 | E (local-s) / M (r) |
| T10 without-R arm, 190 v2 fits | - | 0.8 | pod v1 × v2 / v1 1.29 | M |
| T11 donor test, 8k main | - | 5.0 | local donor s / r_score 10.0 | M |
| T12 donor test, 190 × 13 arms + noR | - | 1.2 | shared per launch + per arm | M |
| **TuringLLM** | **~92** | **112.4** | | |
| Pythia-70m (local 21.3 h = fit 18.5 + score 2.8) | 6.4 | **10.7** (r 1.5: 14.2; 2.7: 7.9; TL ratios 5.5) | local v3 s / r | M local / E r |
| GPT-2 small (local 53.4 h = fit 43.7 + score 9.6) | 18.4 | **26.7** (r 1.5: 35.6; 2.7: 19.8; TL ratios 13.3) | local v3 s / r | M local / E r |
| data passes | 2.3 | 2.3 | unchanged | E (plan) |
| **busy total** | **~120** | **~152** (142-165) | | |

### GPU-h, wall time and cost

Wall time = busy / 8 GPUs × 1.05 (lock hand-overs, shard tails), plus 0.75 h bootstrap and 0.3 h packing. Cost = wall
× 8 GPUs × price. The cap is $550.

| variant | GPU-h | wall h | cost at $2.7 / 3.1 / 3.5 | range (public r 2.7 .. 1.5, price 2.7 .. 3.5) | GPU-h with TL ratios on public |
|---|---:|---:|---|---|---:|
| full | 152 | 21.0 | $454 / **521** / 588 | $426-634 | 134 |
| (a) `TL_DAN159=0` | 136 | 18.9 | $409 / **469** / 530 | $381-576 | 118 |
| (b) `DONOR_VARIANTS=fit` | 148 | 20.4 | $441 / **506** / 572 | $415-616 | 130 |
| (c) public tracks local | 112 | 15.8 | $341 / **392** / 443 | $341-443 | 112 |
| (a) + (b) | 132 | 18.3 | $396 / **454** / 513 | $369-557 | 114 |
| (a) + (c) | 97 | 13.7 | $296 / **340** / 384 | $296-384 | 97 |
| (b) + (c) | 109 | 15.4 | $333 / **382** / 431 | $333-431 | 109 |
| (a) + (b) + (c) | 93 | 13.3 | $287 / **330** / 373 | $287-373 | 93 |

**What changed vs the previous estimate (~120 GPU-h, ~16.8 h, ~$415):**
- **Up by +36 GPU-h:** v3 fits cost 1.50× v2, not 1.0×. T3 +18.5, T4 +4.8, public +12.6.
- **Up by +7 GPU-h:** the new donor and noR stages (T10-T12), now measured.
- **Down by about 10 GPU-h:** T7, at the measured TuringLLM r_fit 3.5 instead of the plan's r = 2.
- **Unchanged, and now measured:** scoring is unchanged by v3, and is ≤ 4% of every fit stage.
- **Net:** ~152 GPU-h, ~21 h, **~$521 central**.
  - That is under the $550 cap at the central price, but not at the top of the price range ($588), nor at the slow
    end of the public r range ($634).
  - With `TL_DAN159=0` the central figure is $469, and the range reaches $576 only at both slow ends together.

**Notes on the variants:**
- (a) runs 083 locally afterwards: ~56.5 local-h, about 2.4 days of the local GPU.
- (b) loses the α = 1 and pinned donor diagnostics. Only `fit` enters `resample_pass`. It saves only ~4 GPU-h,
  because the donor stages are cheap on the pod (r_score 10).
- (c) moves the public tracks to the local GPU:
  - ~75 local GPU-h (Pythia 21.3, GPT-2 53.4), about 3.1 days serial;
  - plus the public data passes (S1-S3), which need the 4M-window corpora on this PC; not measured;
  - on the pod it needs `FIT_GPUS=0,1,2,3,4,5,6,7` (with no public model, the data GPUs 0 / 1 never join the fit
    pool, so the TuringLLM track would run on 6 of the 8 paid GPUs).

**The remaining big uncertainty** is the public models' local → H100 ratio r. No pod run has measured it; only the
pod can.
- TuringLLM's is now measured from history: 3.5 for training, 10 for scoring.
- If the public models behave like that, the full plan is ~134 GPU-h (~18.6 h, ~$461 central).
- At the plan's r = 2 it is 152 GPU-h; at r = 1.5 it is 165.
- Measure it at C3 (the λ pilots) and re-project then.

**Smaller caveats:**
- The public fit times come from smoke-store records (full 48 training contexts, but fewer held-out). Pod records
  also carry a mid band, so eval is costed as strong + mid = 2 × strong.
- The public donor bases on the pod are 16 + 16 rather than the smokes' 8-16 + 16 (≤ +15% on the donor rows).
- The v3 / v1 ratio is taken to transfer to the H100 under MPS.
- The weak-λ sweep arms are not scaled.
- T7's local seconds are 083's own estimate.
