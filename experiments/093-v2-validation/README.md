# 093: protocol v2 vs v1, paired, on 72 stratified TuringLLM targets

## Question

090 showed on 20 hand-picked targets (12 chosen to fail) that training A and C under the scorer's sparsity-preserving
fill lifts the pass rate. Does the ported v2 (native in `src/`, against the scorer's own means) hold up on an unbiased
sample, and does it cost Z?

`paper/protocol.md` §12.1 states the validation rule, fixed before the run. v2 stays production unless, on this
sample:
- its paired pass rate is lower than v1's (the paired bootstrap 95% CI entirely below 0); or
- Z falls systematically (paired median ΔZ < −0.05).

## Design

**Targets** (`select_targets.py`, seed 20261026, CPU; `targets.txt`, `targets.json`):
- 72 targets: 8 per depth band (L0–4 / L5–7 / L8–11) x site kind (attn / mlp / resid), drawn at random within each
  cell;
- population: every 062 15k target with a held-out-strongest eval row, a non-vacuous denominator, a context record and
  a production circuit;
- not enriched for failures: the draw ignores every score. `targets.json` also stores each target's production row.

**Arms** (`run.py`). Both are refitted here through `driver.Runner.fit`, and differ only in the driver's
`WCM_VERSIONS` table (`062/driver.py`):

| arm | `train_fill` | `sparse_grad` | `train_means_source` |
|---|---|---|---|
| `v1` | dense | ste | fit (the engine's own means) |
| `v2` | sparse | ste | eval_ref (the scorer's A / C means) |

- Everything else is v1: γ_Z 1, γ_A = γ_C 0.25, λ 1e-3, rank-keep 3e-3, 400 steps.
- One process runs both arms. Per target the two fits run back to back, in alternating order, switching
  `Runner.wcm_version` and the three engine fields.
- **No engine patch**: v2 is the native `src/` option (`src/config.py` `train_fill`, `sparse_grad`,
  `train_means_source`; `learned_mask.py` `sparse_keep_mask`, `sparse_code`).
- **Contexts:** the 15k run's as-run records (`CTX_DIR=062/out_full/out/ctx`, explicit eval split).
- **Scoring:** `driver.Runner.score` (production scorer) on held-out strongest 16, and `driver.Runner.spec` (056
  specificity, amplifier flags).

**Pre-run checks** (`check_v2.py`, on 2.mlp.4180 and 6.mlp.23423; `check_repro.log`, `check_means.log`,
`out_check/`):
- `CHECK=repro`: native v2 with `train_means_source="fit"` reproduces 090's `sp_ste` circuit exactly (identical
  members and node metadata, α max diff 0.0, identical held-out scores).
- `CHECK=means`: the `eval_ref` means equal the scorer's A and C means bit for bit. v1's own fit A mean does not
  (`v1_fitA_equals_scorer: false`), its C mean does. 090's site check also passes with the native code: training code
  equals the scorer's at every upstream site, for both `ste` and `hard`.

## Files and commands

| File | What |
|---|---|
| `select_targets.py` | the stratified draw (`targets.txt`, `targets.json`) |
| `check_v2.py` | the repro and means checks (`CHECK=repro|means`) |
| `run.py` | both arms, fit + score + spec, resumable per (target, arm); env `ARMS`, `LIMIT`, `OUT93`, `STEPS` |
| `report.py` | `out/tables.md`: paired pass rates (exact McNemar, bootstrap CI, 10,000 target resamples, seed 0), Wilcoxon on score changes, size / α / cost, amplifier flags, the production sanity check |
| `out_smoke/` | a short smoke run (not a result) |

```
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 CHECK=repro .venv/bin/python -u experiments/093-v2-validation/check_v2.py 2.mlp.4180 6.mlp.23423
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 CHECK=means .venv/bin/python -u experiments/093-v2-validation/check_v2.py 2.mlp.4180 6.mlp.23423
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/093-v2-validation/run.py
python experiments/093-v2-validation/report.py
```

## Results

`out/` is gitignored, so the outputs are not tracked. The numbers below are from `out/tables.md` (144 (target, arm)
rows, 0 errors, 72 paired targets) on the author's machine. The run was resumed once (`out/run.log`: 144, then 53
(target, arm) pairs to do).

**Pass rate** (DAN-8 rule, post-Top-K read, held-out strongest):

| set | v1 | v2 | lost / gained | McNemar p | v2 − v1, pp [95% CI] |
|---|---|---|---|---|---|
| **all** | 31/72 (43%) | **50/72 (69%)** | 2 / 21 | 0.000 | **+26.4 [15.3, 38.9]** |
| L0-4 | 17/24 | 19/24 | 2 / 4 | 0.688 | +8.3 [−12.5, 29.2] |
| L5-7 | 12/24 | 20/24 | 0 / 8 | 0.008 | +33.3 [16.7, 54.2] |
| L8-11 | 2/24 | 11/24 | 0 / 9 | 0.004 | +37.5 [16.7, 58.3] |
| attn | 3/24 | 12/24 | 0 / 9 | 0.004 | +37.5 [16.7, 58.3] |
| mlp | 14/24 | 18/24 | 1 / 5 | 0.219 | +16.7 [0.0, 37.5] |
| resid | 14/24 | 20/24 | 1 / 7 | 0.070 | +25.0 [4.2, 45.8] |

The only cell where v2 is lower is L0-4 / mlp (7 → 6 of 8). Under the dense-fill rule (v1's training semantics), v1
passes 37 and v2 2 of 72: each arm fits the fill it was trained under.

**Is Z hurt?** No:
- paired ΔZ median 0.00 [−0.02, 0.02], mean +0.03 [−0.02, 0.09], Wilcoxon p 0.488;
- 7 targets have Z down by more than 0.1 and 11 up;
- Z leaves the band on 3 targets and enters it on 6.

**Other scores** (median v1 → v2):
- A 1.01 → 0.96, C 0.89 → 0.90, necessity 1.00 → 1.00;
- targets in band: A 38 → 62, C 39 → 60;
- among the failures, A fails 34 → 10 and C 33 → 12, while Z fails 16 → 13 and necessity 1 → 4.

**Size and cost:**
- circuit size, median 506 → 465 members (paired ratio 0.91 [0.89, 0.96]);
- α p90 1.70 → 1.56, share of members with α > 2 0.06 → 0.03;
- fit time 1.27x [1.26, 1.28] (median 48.6 s → 60.8 s; totals 3,513 s → 4,524 s);
- peak memory 1.00x (median about 7.05 GB).

Another GPU job shared the card for part of the run, so the paired ratio is the cost figure to read.

**Amplifier flags** (056, any ablation): 22 → 17 of 72. Passing and flagged: 12 in both arms.

**Sanity:** the 15k production circuits pass on 33 of these 72. The v1 refit here agrees with production on 60 of 72
targets (median |ΔZ| 0.058, |ΔA| 0.075, |ΔC| 0.059; different GPU).

**Against the pre-stated rule:** the paired CI is entirely above 0 and the median ΔZ is 0.00, so v2 stays production.

**Where it is used:**
- `paper/protocol.md` §12.1 (validation of protocol v2);
- `paper/main.tex` line 274 (comment) and lines 1767-1768 (comment: "31 -> 50 of 72, +26.4 pp [15.3, 38.9], paired
  median dZ 0.00");
- 095 reuses these 72 targets and the `v2` circuits (`out/circuits/v2`) for the resample appendix;
- `protocol.md` §12 records that moving the band floor from 0.8 to 0.75 takes this sample's v2 from 50 to 55 of 72
  (sensitivity only). The numbers here are at the 0.8 floor.

## Status and caveats

- Current code path. v2 runs natively from `src/` and `062/driver.py`, with no text patch, so this experiment does not
  have the "anchor not found" problem of 089–092. Later production versions (v3, v4; `paper/protocol.md` §0) add R,
  the lr schedule and the censored data term on top of v2, and were not validated here.
- Scored on the 15k run's **as-run held-out sets**, which carry the duplicate-context leakage (DAN-157). Both arms see
  the same records, so the paired comparison stands, but the absolute pass rates are on leaky sets.
- 8 targets per cell: per-cell intervals are wide (see the table).
