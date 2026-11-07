# 102: raising production and resample pass rates (DAN-187 track G, agent arms-102)

Plan: `PLAN.md`. Base algorithm = 095's v2CR (Z + A + C + R_C, lambda 1e-3, 400 steps, R weight 0.25, donors = the
target's 48 training contrast contexts redrawn per forward, sparsity-preserving donor fill, no silent term). Pass floor
0.75. `diag/` belongs to agent diag-102 (CPU diagnosis); everything else here is track G.

## Frozen before any phase-1 result was read

### Check panel (frozen; not fitted until phase 3)

`check_targets.txt` / `check_targets.json`, drawn by `select_check.py` (refuses to redraw once the json exists).

- **Rule:** exp 093's population rule verbatim (`093-v2-validation/select_targets.py`): every target with a
  held-out-strongest eval row in `062-h100-protocol-v1/out_full/out/main` (no error / skip, last row wins), kind attn /
  mlp / resid, `vacuous_tk` False, a context record in `062/out_full/out/ctx` and a production circuit; minus 093 / 095's
  72 dev targets and 099 / 100's 18 panel targets; then `random.Random(20261031).sample(sorted(cell), 8)` per depth band
  (L0-4 / L5-7 / L8-11) x kind cell. No score is used. 72 targets, disjoint from dev and from the 18 (asserted).
- **Seed:** 20261031 (093 used 20261026).
- **Records:** the same source as dev (the 15k run's as-run records, `062/out_full/out/ctx`, explicit eval split), so
  dev and check are scored on the same kind of record. Every check target has its record by construction (the
  population requires it); nothing needed building. Deviation from PLAN.md, which said "from 094's deduplicated
  records": 094's rebuilt 8k records are not built locally (only `094/out/smoke10`), and the brief asks for 093 / 095's
  design. The DAN-157 duplicate leakage therefore applies to both panels alike (paired comparisons; absolute rates are
  on leaky held-out sets).

| cell | population | targets |
|---|---:|---|
| L0-4/attn | 1268 | 3.attn.29509 3.attn.17813 3.attn.5076 2.attn.11852 4.attn.30128 1.attn.35263 3.attn.10409 3.attn.18614 |
| L0-4/mlp | 2256 | 4.mlp.21782 2.mlp.10205 1.mlp.25912 0.mlp.9455 0.mlp.20539 2.mlp.26898 2.mlp.12249 0.mlp.37079 |
| L0-4/resid | 2258 | 2.resid.12046 4.resid.21103 1.resid.33347 3.resid.26238 3.resid.19867 2.resid.14946 0.resid.8392 0.resid.24667 |
| L5-7/attn | 1148 | 7.attn.18720 5.attn.7375 7.attn.17162 7.attn.32113 7.attn.5010 6.attn.38069 7.attn.25165 6.attn.6889 |
| L5-7/mlp | 1353 | 6.mlp.36361 6.mlp.33977 7.mlp.8115 6.mlp.3800 6.mlp.33079 5.mlp.28662 7.mlp.28183 7.mlp.6645 |
| L5-7/resid | 1355 | 6.resid.15364 5.resid.4573 5.resid.18637 6.resid.900 5.resid.24138 6.resid.21207 6.resid.36318 5.resid.30420 |
| L8-11/attn | 1647 | 9.attn.13911 11.attn.16424 9.attn.22192 9.attn.8570 10.attn.6771 10.attn.31179 10.attn.29997 8.attn.38035 |
| L8-11/mlp | 1810 | 10.mlp.6565 10.mlp.24907 10.mlp.37647 8.mlp.224 9.mlp.6047 11.mlp.15224 9.mlp.6959 10.mlp.26069 |
| L8-11/resid | 1809 | 9.resid.22205 10.resid.13036 11.resid.13585 9.resid.40186 8.resid.5512 11.resid.36128 11.resid.40719 8.resid.2461 |

### Selection rule (verbatim, stated before any result)

> An arm is **eligible** if on dev its production pass is not lower than the base's (paired bootstrap 95% CI of the
> difference not entirely below 0). The **winner** maximises min(production rate, resample rate); ties within 2 targets
> are broken by fit time. Confirmation is on the check panel.

Operational details fixed with it: rates are over the 72 dev targets minus vacuous rows (`vacuous_tk`), floor 0.75;
production pass = Z (`free0_tk`), A (`freeM_topk_tk`), C (`freeN_topk_tk`) in [0.75, 1.5] and necessity
(`phi_sup_blind_tk`) >= 0.9 on held-out strongest; resample pass = production pass AND 095's donor recoveries `rs_A`,
`rs_C` in [0.75, 1.5]. Paired bootstrap: 10,000 resamples of targets, seed 20261031, percentile CI of (arm - base)
pass-rate difference. "Ties within 2 targets": arms whose min(production, resample) count is within 2 targets of the
best go to fit time (median per-target fit-time ratio vs the in-run base timing, below).

### best3 criterion (stated before any result)

`best3` = per target, the seed among {0 = base, 1 = seed1, 2 = seed2} with the **lowest final training objective**:
the mean of the total training loss (all data terms Z / C / A / R as weighted in the fit, plus the penalty) over the
logged curve points at steps >= 300 (stride-20 grid: 300, 320, 340, 360, 380 and the last step 399), read from the
fit's `train.jsonl` row (095's row for seed 0). Training split only: it never reads a held-out context. (The engine's
own `holdout_data_loss` is NOT used: `training_probe` pads the training probe with held-out mid-band contexts, so that
number touches held-out data.) Ties (equal criterion) go to the lower seed.

## Engine and code

- `p102_engine.py`: 100's `e4_engine` (= 095's `rs2_engine` + four default-off E4 switches) exec'd in memory on a
  **frozen** copy of `learned_mask.py`: 099's source snapshot, sha256 `b9a6142b...`, the exact file 095 / 100 ran. Why:
  during this run the live `src/circuit/instrument/learned_mask.py` (and `config.py`, `ablation_gradient.py`, 062
  `driver.py`) are being edited by agent r-port (101, the native R port); its edits to the other files are additive and
  default-off (`resample_weight` 0 under `WCM_VERSION=v2`), but `learned_mask.py` itself no longer equals 095's text.
  Nothing in 095 / 100 / src is edited; the 102 addition is one default-off switch (`P102["amp_init"]`, warm-started
  gains for the two-stage arm). `test_p102_engine.py` (CPU): the layering reproduces `e4_engine.patched_source()` on the
  frozen file exactly, and 102 adds only the switch.
- `run.py`: per (target, arm) fit + score, resumable, `--start-by / --end-by`, `rows.jsonl`. Every arm is scored by the
  same code: production (`driver.Runner.score`, held-out strongest), 095's pre-activation pass, 095's resample scoring
  (073 `strict_eval.score_target(c_held=True)`: `rs_A`, `rs_C`, `rs_R`), size, alpha stats, fit time. Each process
  records the hashes of every importable source file (`out/phase1/sources/`) and which differ from 100's last manifest.
- `select_check.py`: the check panel draw.
- `report.py [--out out/check]`: tables (rates, paired CIs, bands, rank_clean stratum, medians, flags, selection rule).
- `phase1.sh` (dev, `ARMS=...`), `check.sh` (check panel, base fitted); logs in `logs/`; rows / fits / train curves /
  circuits in `out/phase1/` and `out/check/`.

### Phase-1 arms (dev = 095's 72 targets)

| arm | what changes vs base |
|---|---|
| `base` | 095's v2CR circuits, reused (see the check below), re-scored here |
| `seed1`, `seed2` | the R donor generator seeded 20261017 + 1 / + 2. This is the fit's only randomness: the engine is full-batch (deterministic micro-batch cycle), uniform theta init, no dropout; 095's refits are bit-identical |
| `best3` | derived, no fit: per target the seed with the best training criterion (above) |
| `lamdepth` | lambda 1e-3 / 5e-4 / 2.5e-4 for L0-4 / L5-7 / L8-11 (L0-4 = base rows) |
| `twostage` | base membership fixed (support = base members, lambda 0, binarize none, theta_init 40), gains warm-started at the base alphas, 200 more steps under all terms incl. R; fit time = this stage only (report adds the base's) |
| `r0.5`, `r1.0` | R weight 0.5 / 1.0, C kept |
| `rk3e-2`, `rk1e-1` | (added by the coordinator, from diag-102: 15/18 v2CR Z failures restore the pre-activation and lose at the Top-K cut) rank-keep weight 3e-2 / 1e-1 (production 3e-3), mode `keep`, delta 0 as in production. The engine also has a `cut` mode with margin `rank_delta`; not used |
| `r1.0_rkBEST` | R weight 1.0 + the better of rk3e-2 / rk1e-1 by the selection-rule score on dev (decided after both finish; arm name `r1.0_rk3e-2` or `r1.0_rk1e-1`) |

Run order (coordinator): base re-score -> rk3e-2 -> rk1e-1 -> r1.0 -> r1.0_rkBEST -> lamdepth -> twostage ->
r0.5 -> seed1 -> seed2 (`phase1.sh`, one GPU process at a time: every fit peaks at 8-9.6 GB, so a second concurrent fit
would not fit on the 16 GB card next to r-port's jobs).

**Revised plan (coordinator, given ~2 h per 72-target arm):** dev = base -> rk3e-2 -> rk1e-1 -> r1.0_rkBEST
only. Plain r1.0, lamdepth, twostage, r0.5, seed1 / seed2 and best3 are dropped from this run (diag-102 measured refit
noise at ~14% production flips; they move to a later session; the code for them stays in `run.py`). When r1.0_rkBEST
finishes, the selection rule is applied on dev to {base, rk3e-2, rk1e-1, r1.0_rkBEST} and the decision is written here
with its numbers **before** any check target is fitted. Then the check panel: base (v2CR, fitted here:
`--base-source fit`), then the dev winner, then (time permitting, `--end-by`) the runner-up; if the winner is the base,
the best non-base eligible arm is the second check arm. Check tables: `out/check/tables.md`, same format, paired vs
base, by band and by the rank_clean >= 32 stratum. Without seed1, the selection rule's fit-time tie-break uses the
median per-target ratio vs 095's base fit times (same GPU).

**rkBEST decision (selection-rule score on dev, n = 72 each):** rk1e-1 min(prod, rs) = 33 (prod 57, rs 33)
vs rk3e-2 27 (prod 56, rs 27); both eligible; 6 apart (> the 2-target tie band), so **rkBEST = rk1e-1** and the arm
run is `r1.0_rk1e-1`.

**Dev decision (the pre-stated rule on {base, rk3e-2, rk1e-1, r1.0_rk1e-1}, n = 72 each, floor 0.75; written
before any check target was fitted; `out/phase1/tables.md` §6):**

| arm | production | resample | min | d production vs base [paired 95% CI] (w/l) | eligible |
|---|---:|---:|---:|---|---|
| base (v2CR) | 48 | 22 | 22 | - | yes |
| rk3e-2 | 56 | 27 | 27 | +11 pp [+1, +21] (11/3) | yes |
| rk1e-1 | 57 | 33 | 33 | +12 pp [+4, +22] (11/2) | yes |
| r1.0_rk1e-1 | 54 | 44 | 44 | +8 pp [-1, +18] (10/4) | yes |

Winner: **r1.0_rk1e-1** (score 44; no other arm within 2). Runner-up: **rk1e-1** (33). Check panel order: base (v2CR,
fitted), r1.0_rk1e-1, rk1e-1, with `--end-by 2026-11-01T00:30` (`check.sh`).

## Results (dev ~6 h 16 min, check ~5 h 44 min; all arms complete, n = 72 each, 0 errors, no rejected fits)

Tables: `out/phase1/tables.md` (dev), `out/check/tables.md` (check). Floor 0.75; paired vs base; CI = paired bootstrap
95%; w/l = targets the arm passes and base fails / the reverse. Fit time = median per-target ratio vs base (dev: 095's
base fit times; check: this run's base fit).

| panel | arm | production | resample | pre-act | d production [CI] (w/l) | d resample [CI] (w/l) | size x base | fit time x base |
|---|---|---:|---:|---:|---|---|---:|---:|
| dev | base (v2CR) | 48 | 22 | 51 | - | - | 1.00 | 1.00 |
| dev | rk3e-2 | 56 | 27 | 57 | +11 pp [+1, +21] (11/3) | +7 pp [-3, +18] (10/5) | 1.03 | 1.04 |
| dev | rk1e-1 | 57 | 33 | 58 | +12 pp [+4, +22] (11/2) | +15 pp [+6, +26] (14/3) | 1.03 | 0.99 |
| dev | **r1.0_rk1e-1** | 54 | **44** | 55 | +8 pp [-1, +18] (10/4) | **+31 pp [+19, +42] (23/1)** | 1.30 | 1.05 |
| check | base (v2CR, fitted) | 48 | 19 | 57 | - | - | 1.00 | 1.00 |
| check | **r1.0_rk1e-1** | 54 | **39** | 57 | +8 pp [+0, +18] (9/3) | **+28 pp [+17, +39] (21/1)** | 1.33 | 0.97 |
| check | rk1e-1 (runner-up) | 55 | 25 | 59 | +10 pp [+0, +19] (10/3) | +8 pp [+0, +18] (9/3) | 1.03 | 0.97 |

- **The dev winner replicates on check.** R weight 1.0 + rank-keep 1e-1 doubles the resample pass (19 -> 39 of 72;
  dev 22 -> 44), losing it on one target per panel, and does not lower production (+6 targets on both panels).
  min(production, resample) 39 vs 19 (check), 44 vs 22 (dev). Pre-activation pass unchanged on check (57 = 57).
  Cost: circuits x1.3 larger (median 702 vs 572 on check), no extra fit time.
- **Rank-keep alone (1e-1) is the production lever**: +7 / +9 production passes (dev / check), size and fit time
  unchanged; its resample gain is smaller (+11 dev, +6 check). 3e-2 sits between base and 1e-1 on dev.
- **By depth (check):** the winner gains resample passes in every band (L0-4 9 -> 15, L5-7 7 -> 13, L8-11 3 -> 11) and
  production +1 / +3 / +2. Dev L8-11 resample 3 -> 9.
- **Clean rank >= 32 stays hard:** dev 1 -> 3 production passes of 12 and 0 -> 0 resample; check 5 -> 6 of 16 and 1 -> 4
  resample. Nearly all of the gain is in rank_clean < 32 (check resample 18 -> 35 of 56).
- Differences of a few targets are within refit noise (diag-102: ~14% production flips between refits); the winner's
  resample gain (+20-22 targets, 1 loss per panel) is far outside it, its production gain (+6) is at its edge (CI
  lower ends -1 / +0).
- Flagged targets: dev 11.attn.12206, 2.attn.16773, 5.resid.35667 and check 1.resid.33347, 3.resid.26238,
  9.resid.40186 have undefined Z / necessity reads (vacuous_tk False) in every arm and count as failures everywhere.
- Not run here (coordinator's revised plan): plain R 1.0, depth-scaled lambda, two-stage, R 0.5, extra seeds, best3.

Flagged targets (diag-102), counted as failures wherever a read is undefined, consistently in every arm, and listed in
the tables: 11.attn.12206, 2.attn.16773, 5.resid.35667 (an undefined term with `vacuous_tk` False); 7.attn.38584 (only 9
strongest contexts).

### Base reuse check (phase 0)

`run.py --check-base 2.attn.22342,8.resid.4054` (`out/phase1/check.jsonl`, `logs/check_base.log`): v2CR refits through
`p102_engine` are **bit-identical** to 095's circuits on both (same 279 / 895 members, max |d alpha| 0; trained terms
zero, floor, pos, resample). 095's v2CR circuits are therefore the `base` arm, re-scored by this run's code.

Fit-time baseline: 095's base fit times were measured with no other GPU job; this run shares the GPU (r-port),
so fit-time ratios are reported against `seed1` (the same algorithm as base, fitted in this run) as well as against
095's times.
