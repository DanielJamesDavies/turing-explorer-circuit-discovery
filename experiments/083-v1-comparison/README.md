# 083: the v1 method comparison (DAN-159, DAN-43, DAN-77)

Built 2026-10-23, re-scoped 2026-10-24 to Daniel's DAN-159 decisions. Plan: `dev-notes/2026-10-23-pod-plan-public-models.md`
§4.2. **Nothing here has run on the 190 targets yet**; the smoke runs (below) are CPU only.

## Design: curve-first

The primary output is the **faithfulness-vs-size figure** (DAN-77): every method at its genuine sizes, per depth band,
pass rate (DAN-8) and median A against median circuit size.

| Series | Sizes | Source |
|---|---|---|
| WCM | all eight prices, λ 2.5e-5 .. 4e-3 | 062 circuits (`out_full/out/sweep`, `out_{bands,depth,lowlam}`), re-scored by `rescore_ref.py` |
| unweighted circuit masking | all five prices, λ 1e-5 .. 1e-3 | 062 circuits, re-scored |
| path attribution (IG, `abl_ig`) | the target's WCM size at every WCM price, and its natural size | `attr_baselines.py` |
| attribution patching (AtP, `atp`) | same | `attr_baselines.py` |
| Ge hierarchical attribution | every τ bisection point (target: the W_0.001 size) | `ge_hier.py` |
| attribution members + fitted coefficients (DAN-43) | the W_0.001 per-site counts | `attr_fitted.py` |
| activation-ranked members (mean anchor activation; co-activation variant), α = 1 and fitted | the W_0.001 per-site counts | `attr_fitted.py` |
| exact-patching-ranked members (10-target subset), α = 1 and fitted | the W_0.001 size | `exact_patching.py` |

**Table 2 is the summary** (`table2.py`): each method at its natural operating point (WCM and unweighted at the
**same price**, λ 1e-3; IG and AtP at their natural position-union size; Ge at its bisected size), then IG / AtP
truncated at the W_0.001 size under each ranking, and DAN-43 at α = 1 and fitted. Unweighted is compared at the same
λ (Daniel's choice over size matching); the nearest-size match is printed only as a note (no unweighted price on the
grid is as small as WCM: the nearest is U_0.001 for 99% of the 190 targets, at a median 1.85× the WCM size).

Dropped on 2026-10-24: the restoration row, the contrastive `ig_mean` row and the `ctr_ig_c` variant.

## What it resolves in the paper

| Output | `main.tex` |
|---|---|
| the DAN-77 figure (`dan77-faithfulness-vs-size.pdf`, `dan77_points.csv`) | DAN-77; the main-text "Attribution at matched size" paragraph (L469) can cite it |
| Table 2 under protocol v1 on the 190 sweep targets (`table2.md`, `.tex`) | L469, L847 (`app:method-comparison`) |
| the SFC-aggregate and max rankings side by side | L872 caption ("the aggregate ranking of prior work") and L879 |
| DAN-43 rows | the missing middle row (attribution members, fitted coefficients) |

## The estimators

All discovery runs on each target's v1 **training** contexts (the first `n_train` rows of the probe
`driver.Runner.fit` injects, asserted per target), along the path from every upstream site's mean over those contexts
(the A ablation value) to its natural code, all sites together, through `ig_baseline.InterpolatedCodeInstrument`
(unchanged); target = the target's pre-activation at each context's anchor.

- **IG (`abl_ig`)**: gradients at 10 left-Riemann points × (natural − mean): `integrated_baseline_scores`' estimator
  (tested equal on the collapsed reduction).
- **AtP (`atp`)**: one gradient at the clean point (α = 1, where the instrument reproduces the unedited stream) ×
  (natural − mean): the first-order estimate of mean-ablating each latent (Nanda 2023; Syed et al. 2023; Kramár et
  al. 2024), with SFC's pass-through gradients. **Not the library's `local` ablation mode**: that scores
  activation × grad(pre²), i.e. zero ablation of active latents only, so it is not the mean-ablation AtP asked for. Both
  arms therefore use the same reduction code (`attr_baselines.ig_reduce`, `alphas=[1.0]` for AtP); tested: AtP = IG
  exactly on a linear metric, and AtP = a direct autograd clean-point gradient on the nonlinear one.
- **Rankings** (each target's top `KEEP_TOP` = 65,536 entries; the largest WCM circuit is 37,170): `max` = largest
  |attribution| at any context and position (031's), `sfc` = |mean over contexts of the summed signed attribution|,
  the **aggregate ranking of prior work (SFC)**, and `agg` = mean summed |attribution| (kept in the CSV; the figure
  shows max and sfc).
- **Natural size**: 031's position union (per batch of 4, one pooled 50th-percentile cut on |attribution|, each
  context's own causal prefix).
- **Ge**: 034's `GeHierInstrument` / `hier_run` / log-space τ bisection unchanged (bracket 1e-8..1e2, 7 steps, ±10%);
  every point saved and scored (n ≤ 20,000); resumes mid-bisection.
- **DAN-43**: support = the W_0.001 circuit's per-site counts filled from the top of the **IG max** ranking at each
  site; coefficients fitted as `062/random_null.py` does (support-restricted engine call, λ 0, θ₀ 40, no
  binarisation, production config through `Runner.fit`). IG is kept as the source because AtP is its one-point
  linearisation; `FIT_FROM=abl_ig:max,atp:max` adds AtP as a second source (+2.6 local-h) if the AtP rows beat IG.

- **Activation-ranked members + fitted coefficients** (Daniel, 2026-10-24; `attr_fitted.py`, `FIT_FROM=act:mean,act:coact`):
  the no-gradient heuristic. At each upstream site, latents ranked by their mean post-Top-K activation at the
  target's anchor over the training contexts (`act:mean`, primary), or by the Pearson correlation of that anchor
  activation with the target's own across contexts (`act:coact`, the co-activation variant; latents never active
  at an anchor are not ranked). Top n_s per site at the W_0.001 per-site counts, coefficients fitted exactly as
  DAN-43, scored at α = 1 and fitted. The pre-v1 analogue (activation-ranked members with fitted amplitudes) passed
  **0/128**.

Every score is `driver.Runner.score` on held-out strongest. No 031 `SetPatcher` or 034 `AmpCircuitPatcher` (both left
non-member sites live).

## Exact activation patching: ground truth for AtP and IG (`exact_patching.py`)

On 10 targets (seed 20261024: one per depth band × site kind, the tenth from L8-11; written to `exact/targets.txt`),
on the training contexts, the exact effect of each candidate latent on the target's pre-activation at the anchor:
Δ = mean over contexts of [pre(that one latent set to its training-context mean at every position, everything else
live) − pre(clean)]; edit x_s ← x_s + W_dec[:, i] (mean_i − c_i), the SAE error untouched, the forward stopped at the
target's site, (candidate, context) pairs batched 64 rows per forward. This is the quantity AtP linearises (AtP ≈ −Δ).
- **Candidates**: every latent active at an upstream site at some position ≤ the anchor of some training context
  (`CAND=all`) when there are at most 6,000; otherwise the union of the top-2,000 by AtP, the top-2,000 by IG (both
  by |SFC aggregate|) and a seeded random 1,000 of the remaining active latents (`CAND=subset`). The stats row records
  the mode, the candidate count and the pool size.
- **Reports** (`exact/rows.shard*.jsonl`, `REPORT=1` → `exact/summary.md`): Spearman ρ of AtP and IG (SFC aggregate
  and max) with −Δ, signed and on |·|; top-n overlap with the exact top-n by |Δ| at the W_0.001 size; the
  exact-ranked circuit (global top-n by |Δ|) scored at α = 1 and with fitted coefficients (the DAN-43 fit). Its rows
  also appear in Table 2 (ground-truth section) and as DAN-77 points.

## Held-out data (DAN-157)

Contexts are read only through `driver.Runner.contexts`, so the driver's `CTX_DIR` (the rebuilt records, e.g.
`062/out_full/out/ctx_v2`; forces `EVAL_SPLIT=explicit`) applies to every stage. With `CTX_DIR` set, outputs go to
`out_<CTX_DIR name>/` and `table2.py` defaults to `REF=local`, i.e. WCM / unweighted rows only from the re-scores on
the rebuilt sets (the 062 H100 rows and the DAN-17 random rows were scored on the as-run sets and are then left out).
Caches (rankings, Ge points, fitted circuits) are filed under a hash of each target's training contexts, so they are
reused only when the training rows are identical; rows carry `ctx_dir` and `ctx_hash`.

## Files and commands

| Script | Rows | Cache |
|---|---|---|
| `rescore_ref.py` | `ref/rows.shard*.jsonl` (seed, arm, ctx_dir) | |
| `attr_baselines.py` | `attr/rows.shard*.jsonl` (seed, method, ranking, size_arm, ctx_dir) | `rankings/<arm>/<key>.<hash>.pt` |
| `attr_fitted.py` | `fitted/rows.shard*.jsonl` (seed, method, ranking, variant, ctx_dir) | `fitted/circuits/` |
| `ge_hier.py` | `ge/rows.shard*.jsonl` (seed, point, ctx_dir) | `ge/points/<key>.<hash>.pt` |
| `exact_patching.py` | `exact/rows.shard*.jsonl` (seed, variant, ctx_dir); `REPORT=1` → `exact/summary.md` | `exact/effects/<key>.<hash>.pt` |
| `run_local.sh` / `run_pod.sh` | the queue on one machine / every stage through `062/launch.sh` (`SCRIPT=`, `SHARD=i/N`) | |
| `table2.py` | figure, `dan77_points.csv`, `table2.{md,csv,tex}`, `table2_rows.csv` (every row with `file:line`) | |
| `cmp_common.py` | pure helpers (rankings, truncation, sizes, pass rule, reference merge, Ge replay); unit-tested | |

```bash
# pod (proposal; agree each command with Daniel): every stage through 062/launch.sh, SHARD=i/N per process
CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx_v2 K=8 P=4 bash experiments/083-v1-comparison/run_pod.sh
# one machine (resumable; re-run the same line after any interruption)
CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx_v2 bash experiments/083-v1-comparison/run_local.sh
# smoke (CPU, 1 shallow target, IG 2 steps, bisection 2 steps, 4 fit steps, sizes W_0.001 and W_0.004)
SMOKE=1 bash experiments/083-v1-comparison/run_local.sh
```

## Runtime

Per target, in local-RTX-seconds (from the 031 / 034 / 062 measurements; this suite does the same forward and
backward work):

| Stage | per target |
|---|---|
| ref: 13 re-scores (8 W + 5 U) × ~5 s | 65 s |
| attr: IG 87 s + AtP ~12 s + 48 truncation scores (8 sizes × 2 arms × 3 rankings) + 2 natural-size scores | ~380 s |
| fitted: 3 sources (IG max, activation mean, co-activation) × (400-step support fit + 2 scores) | ~145 s |
| ge: 8 sweeps (034 mean 335 s) + ~7 scored points | ~370 s |
| **total** | **~960 s → 190 targets: ~51 local-h** |
| exact patching, 10 targets, subset candidates (≤ 5,000 × 48 contexts ≈ 240k patched rows, est. ~0.75 ms/row on a GPU) + IG/AtP + 2 scores + 1 fit | ~5 min per target → ~1 local-h |

**8×H100 (pod)**: at the plan's r = 2.0 (one H100 under MPS ≈ 2 local processes; range 1.5–2.7), 51 / (8 × 2) ≈
**3.2 h wall (2.4–4.3 h), ~26 GPU-h, ≈ $80 at the plan's ~$25/h for 8 GPUs**, plus ~0.2 h for the exact stage,
start-up and stage tails. P = 4 per GPU (a ~13 GB bank per process; Ge holds a full-depth graph per chunk).
`SCORE_POINTS=final` saves ~0.2 h; `RANKINGS=max,sfc` saves ~1/6 of the attr scoring.

**Exact patching cost (measured, 1 target, CPU)**: `0.resid.4723` (2 upstream sites): pool 26,160 active latents → subset
of 3,181 candidates (the AtP and IG top-2,000 overlap heavily) + random 1,000; 152,688 patched rows in 2,002 s =
13 ms/row on CPU with the forward stopped at layer 0. `CAND=all` on that target would be 1.26M rows (4.6 h CPU); on
a deep target the pool is ~10^5 latents (~5M rows, ~1-2 h per target on an H100 at the estimated rate), so the
subset rule is the default and `CAND=all` is not recommended. **The other 9 targets have not been run** (waiting on
the estimate, as asked).

A CPU run of the 190 is not feasible (10 IG steps, 400 fit steps, up to 35 sites, ~60 scores per target at 54–166 s
each on CPU).

## Smoke evidence (CPU, 2026-10-24, `out_smoke/`, `CTX_DIR` = the as-run ctx dir to exercise the CTX_DIR path)

One shallow target, `0.resid.4723` (39 WCM members), IG 2 steps, Ge 2 bisection steps, 4 fit steps, sizes W_0.001
and W_0.004; every stage ran end to end (ref, attr, fitted, exact, ge, table, exact report), resumed through four OOM
kills (other agents' 13–19 GB jobs on the shared WSL) without recomputing cached work; 0 error rows.

| row (W_0.001 size, 39) | Z / A / C | nec | pass |
|---|---|---|---|
| WCM (local re-score; H100 row 0.989 / 1.129 / 1.006) | 0.99 / 1.13 / 1.01 | 1.00 | yes |
| unweighted λ 1e-3 (63 latents) | 1.04 / 0.99 / 0.97 | 1.00 | yes |
| IG top-n: max / mean\|attr\| / SFC | 0.79/0.77/0.71, 0.84/0.88/0.81, 0.89/0.87/0.81 | 1.00 | no / yes / yes |
| AtP top-n: max / mean\|attr\| / SFC | 0.77/0.75/0.70, 0.86/0.88/0.81, 0.92/0.89/0.84 | 0.94–1.00 | no / yes / yes |
| IG / AtP natural size (19,728 / 20,265) | 1.00 / 1.00 / 1.00 | 1.00 | yes |
| Ge, τ bisected (37) | 0.86 / 0.86 / 0.80 | 0.97 | yes |
| IG-ranked per-site, α = 1 → fitted (DAN-43) | 0.79/0.77/0.71 → 1.06/1.08/0.99 | 1.00 | no → yes |
| activation-ranked (mean), α = 1 → fitted | 0.75/0.68/0.66 → 1.01/0.94/0.91 | 0.91 | no → yes |
| activation-ranked (co-activation), α = 1 → fitted | 0.21/0.17/0.11 → 0.38/0.21/0.15 | 0.56 | no |
| exact-patching-ranked, α = 1 → fitted | 0.93/0.89/0.84 → 1.14/1.12/1.06 | 1.00 | yes → yes |

Exact patching on that target: Spearman with −Δ, SFC aggregate: **AtP 0.986, IG 0.931** (|·|: 0.985 / 0.940); top-39
overlap with the exact top-39: AtP 0.97, IG 0.95; the max reduction tracks the exact effect much less (ρ 0.67 / 0.59,
overlap 0.77 / 0.74). One shallow target with 2 IG steps is not evidence about the comparison; it shows the rows are
sane.

Unit tests: `tests/analysis/test_083_v1_comparison.py`, 34 tests (rankings, truncation, reductions and chunk
invariance, IG = `integrated_baseline_scores`, AtP = IG on a linear metric and = the clean-point autograd gradient on
a nonlinear one, the natural union's per-chunk anchors, size resolution, reference merge, training-context hash,
nearest size, the DAN-8 table logic, the Ge replay, activation / co-activation scores, the exact-patch edit and the
stratified target draw). Full suite: 1,739 passed, 3 skipped.
