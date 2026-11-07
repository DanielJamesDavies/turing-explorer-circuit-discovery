# 091: the donor-resample term R (and the silent-position term S) on top of v2, small test

## Question

079 found that adding a donor-resample training term R to protocol-v1 WCM raises the resample pass rate (the 073
test). v2 (090's `sp_ste`: A and C trained under the scorer's sparsity-preserving fill, straight-through) already
fixes many v1 failures. Does R still help on top of v2, and does 079's silent-position term S add anything?

## Design (`run.py`)

**Targets** (`out/targets.json`): 8 of 090's 20, `random.Random(20261025)`:

| group | targets |
|---|---|
| fail89 (4) | 11.mlp.16843, 5.mlp.29712, 8.attn.20556, 8.mlp.7480 |
| deep attention, near-threshold (1) | 6.attn.4595 |
| deep attention, far (1) | 11.attn.13198 |
| controls passing under both 090 `dense` and `sp_ste` (2) | 7.resid.25654, 9.resid.33488 |

**Versions.** Everything else is production: 400 steps, λ 1e-3, γ_Z 1, γ_A = γ_C 0.25, rank-keep 3e-3. One fit per
target and arm.

| version | what |
|---|---|
| `dense` | 090's v1 circuits, reused (not refitted) |
| `sp_ste` | 090's v2 circuits, reused |
| `v2R` | `sp_ste` + R, weight 0.25 |
| `v2R_S` | `v2R` + S, weight 0.25 (079's `RS0.25_S0.25`) |

- **R donors** are the target's 48 **training** contrast contexts (079 `DONORS=contrast`). There is one donor per
  training row, redrawn at every forward (`DonorFloors`, seed 20261017). `donor_tokens` asserts that they equal the C
  term's contrast rows and the scorer's contrast-train split.
- R is applied exactly as in 079: raw donor code, no Top-K re-imposed. 092 and 095 later added a sparsity-preserving
  donor fill.

**Engine (`engine.py`).** `src/` is not edited. One in-memory patch of `learned_mask.py` carries both
experiment-local patches: 090 `sp_engine` (the v2 fill) and 079 `rs_engine` (R and S). Their anchors do not
overlap. One edit is re-derived: 090's edit 1 skips the donor patcher (`is_resample`), so R stays exactly as in 079.

**Equivalence check** (`CHECK=1`, 11.mlp.16843, 6 steps; `out/check.jsonl`):
- the combined patch with w_R = w_S = 0 under `ste` equals 090 `sp_engine` `ste`: same circuit and same loss curve;
- the combined patch with the dense fill and w_R = w_S = 0.25 equals 079 `rs_engine` `RS0.25_S0.25`: same circuit and
  same loss curve;
- `v2R_S` differs from both.

**Scoring**, held-out strongest 16, all four versions, on the 15k run's as-run records:
- the production scorer (`driver.Runner.score`): rule A / C under the sparsity-preserving fill, dense A / C,
  necessity, and the rule pass (Z, A, C in [0.8, 1.5], necessity ≥ 0.9);
- the resample test, 073 `strict_eval.score_target(c_held=True)`: rs A (strongest-train donors), rs C (the 16
  **held-out** contrast contexts, never a training donor), rs R (073's random pool). Resample pass = Z, rs A, rs C in
  [0.8, 1.5] and necessity ≥ 0.9 (079's definition).

`dense` and `sp_ste` re-scored here match 090's rows exactly (max |diff| 0.000 on Z / A / C / necessity).

## Files and commands

| File | What |
|---|---|
| `engine.py` | the combined in-memory patch (090 + 079) |
| `run.py` | target pick, fits, scoring; resumable; env `CHECK=1` (+ `CHECK_STEPS`), `LIMIT`, `ARMS`, `SCORE_ONLY=1` |
| `report.py` | `out/tables.md` |
| `smoke.log` | a short-step smoke on 11.mlp.16843 (n 1,392,640 latents after so few steps; not a result) |
| `run.log` | the full run (2,052 s for fits + scoring) |

```
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 CHECK=1 .venv/bin/python -u experiments/091-v2-resample-small/run.py
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/091-v2-resample-small/run.py
python experiments/091-v2-resample-small/report.py
```

`run.py` imports `../073-strict-eval/strict_eval.py`, `../090-sparse-fill-training/sp_engine.py`, and
`../079-resample-training/rs_engine.py`, and reads 090's `out/` circuits and rows.

## Results

`out/` is gitignored, so the outputs are not tracked. The numbers below are from `out/tables.md` on the author's
machine (one GPU, the same as 090's).

Medians per version (8 targets):

| version | n | Z | A | C | rule pass | rs A | rs C (held) | rs R | resample pass | fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 652 | 0.84 | 0.63 | 0.79 | 2/8 | 0.18 | 0.03 | 0.00 | 0/8 | 60 |
| sp_ste | 596 | 0.93 | 1.01 | 0.93 | 7/8 | 0.08 | 0.01 | 0.00 | 0/8 | 78 |
| v2R | 704 | 0.85 | 1.03 | 0.98 | 4/8 | 0.82 | 0.72 | 0.52 | 2/8 | 110 |
| v2R_S | 967 | 0.86 | 1.02 | 0.90 | 3/8 | 0.81 | 0.45 | 0.26 | 2/8 | 105 |

Paired against `sp_ste`:

| version | size ratio | Δ rs A | Δ rs C | Δ rs R | rule pass +/− | resample pass +/− | fit time | peak memory |
|---|---|---|---|---|---|---|---|---|
| v2R | 1.22x | +0.77 | +0.69 | +0.51 | +0 / −3 | +2 / −0 | 1.38x | 1.06x |
| v2R_S | 1.64x | +0.70 | +0.37 | +0.26 | +0 / −4 | +2 / −0 | 1.34x | 1.06x |

Reading:
- **R on top of v2 lifts the resample scores a lot, but costs rule passes.** The resample pass goes 0/8 → 2/8, both
  fail89 MLP targets (11.mlp.16843, 5.mlp.29712). The rule pass goes 7/8 → 4/8.
- **The three rule losses are all Z below 0.8:** 8.mlp.7480 0.78, 6.attn.4595 0.25, and 9.resid.33488 0.70, a passing
  control.
- **S adds nothing here:** the same resample-pass count, lower rs C and rs R than `v2R`, rule pass 3/8, and circuits
  1.64x `sp_ste`'s size.

**Where it is used:**
- `paper/protocol.md` §12.1 lists R and S under "not included" in v2 (exps 091–092).
- 095 drops the silent term on this evidence and re-runs the R comparison on 72 targets for `app:resample`.
- 092 reuses these 8 targets and compares against `v2R`.
- `paper/main.tex` does not cite 091.

## Status and caveats

- **Superseded** by 095 (72 targets, R under a sparsity-preserving donor fill, on the ported v2 engine).
- **`engine.py` no longer applies to the current `src/`.** It reuses 090's edit 1, whose anchor (the v1 floor branch
  of `LearnedMaskPatcher.transform`) occurs 0 times in the current `src/circuit/instrument/learned_mask.py`, so
  `patched_source()` raises "anchor found 0 times". 079's six `rs_engine` anchors still match once each (checked by
  reading the file, not by running). 095's `rs2_engine.py` is the port onto the current `src/`.
- 8 targets, one fit each. 092 later found that any kernel-shape change moves single-target passes (membership
  Jaccard about 0.3–0.7 between mathematically identical refits), so single flips here are not evidence.
- Held-out sets are the 15k run's as-run sets, which carry the duplicate-context leakage (DAN-157). The comparison is
  paired.
- The v2 here is 090's `sp_ste`, which trains against the engine's own means, not the scorer's
  (`train_means_source = "eval_ref"` in production v2).
