# 131 The off term O: train circuits to stay off on silent contexts (GPT-2 small, local GPU)

## Question

Exp 130 found that most passing production WCM v4 circuits are not selective. On the target's 16 held-out
verified-silent contrast contexts, with members at fitted α × their live value and every other latent taken from
another silent context (the donor test's pool C), passing circuits switch the target on at a median 45% (TuringLLM),
7% (GPT-2 small) and 1% (Pythia) of positions. The full model fires at 0% there, the empty circuit at about 0.1%, and α = 1
at about 0%. So the fitted gains on partly-active members cause the firing. No training term reads the target on a
silent context: Z, A, C and R all read it at the anchors of activating rows.

Does adding an "off" term O, which trains the circuit to also stay off on the contrast contexts, make the circuits
selective? What does it cost in the production test (Z, A, C, nec_b), in circuit size, and in the resample test?

## Design (written before any GPU result)

**The O term** (`off_engine.py`) is a fifth data term in the v4 engine, over the same thetas, amplitudes and gate as
Z, C, A and R.
- **Rows.** The target's 48 *training* contrast contexts: `eval_reference`'s C rows = `rec["neg"][:48]`, the same rows
  R uses as donors. The 16 held-out contrast rows read by the scorer and the donor test are never used (test (c)).
- **Members** sit at α × their live value on that contrast context. This is the transform's own `m·α·dense`.
- **Non-members** take the natural sparse code of *another* training contrast row at the same position, under the
  sparsity-preserving donor fill (`donor_keep_mask`: at binary gates this equals 073's `ResamplePatcher`, the donor
  test's fill; tested in 095). The donor row is redrawn at every forward from its own generator (seed 20261118) and is
  never the row itself.
- **Penalty, at every position** of those rows:
  `O = w_O / dual_norm · mean_{rows, positions} relu(pre − max(natural pre, 0))²`.
  Here pre is the target's pre-activation under this ablation, natural pre is the full model's on the same row
  (computed once per fit), and `dual_norm` is the engine's shared normaliser: mean natural anchor pre² over the
  activating training rows (the one Z / C / A / R divide by). Only a lift above the natural value (and above 0) is
  charged. This is 079's S formula and normalisation, moved from the silent positions of activating rows onto the
  contrast contexts themselves.
- **Batching.** The engine's own micro-batch slicing (rows s..s+micro_bs, cycling through all 48). That is one extra
  forward and backward per micro-batch, so the O arm costs about 1.25× the v4 fit time (Z, C, A and R are the other
  four). Peak VRAM is unchanged: each term is backwarded on its own.

**Why the donor fill, not the C-style mean fill.** The read we want to move is pool-C silent firing, and the donor
fill is that read's training twin: same kind of base (a silent contrast context), same member values (α × live), same
background (another silent contrast context's real code, Top-K budget kept). A mean fill (the C term's floor, applied
on contrast rows) would train against a smoothed, off-manifold background that the test never shows. The donor fill is
also already in the engine (R) and tested. The cost is noise: one donor per row per step. The mean fill is the natural
follow-up if O turns out too noisy to train.

**Mechanics.** `off_engine.install()` execs `src/circuit/instrument/learned_mask.py` with five text edits, each
required to match exactly once (079 / 095's mechanics). The edits:
1. build the off patcher;
2. add its term after R, weight `w_O / dual_norm`;
3. name it `off` in the curve;
4. route its forwards to the off loss in `data_loss`;
5. keep its anneal temperature in step.
Nothing in `src/`, 062, 082 or 107 is edited. With `OFF["weight"] = 0` no patcher or term is built: the production
engine, bit-identical (test (a)).

**Arms** (each target fitted in every arm, same records, same seeds; paired):
| arm | what |
|---|---|
| `v4` | the production WCM v4 fit (λ 1e-3, 400 steps, R 1.0, rank-keep 0.1, geometric lr ×3.15, censor 0.05), refitted here through the patched engine at w_O = 0: the in-run baseline |
| `o1` | v4 + O, w_O = 1.0 |
| `o0.25` | v4 + O, w_O = 0.25 |
| `o4` | v4 + O, w_O = 4 (extra, last) |

Every arm runs through 062 `driver.Runner` (FAMILY gpt2-small, WCM_VERSION v4, the production records and train arm),
with `DEEP_BS=1` (every local fit since 126). The pod ran the config's 4. It is the same for all arms, so the comparison
stays paired. It costs some speed and bit-identity with the pod's fits.

**Eval, per (target, arm)** (`run.py`):
- **Production scorer.** `Runner.score(c, rec, "strong")`, held-out strongest. Pass = `wcm_eval.passes` (Z, A, C in
  [0.75, 1.5], nec_b ≥ 0.9); vacuous counts as a fail in the paired tables (flagged).
- **Donor test.** `src/eval/donor_test.donor_target` with pools as production: A = the 48 strongest-train rows, C = the
  16 held-out contrast rows, R = the production run's random pool, copied from the pod archive. 8 draws. Variants
  `fit` and `a1` (production also ran `pinned`, which is not needed here). Silent firing SF = share of the 16 × 64
  silent positions where the post-Top-K target is > 0. Resample pass = `donor_test.resample_pass`.
- **Magnitude read (new).** From the same forwards (`donor_test.run_full` is wrapped to keep its outputs), per pool and
  variant, including the empty circuit: the post-Top-K target at the silent positions, divided by the natural mean
  anchor activation of the 16 held-out activating bases. Reported: mean, p95, p99, max, and the mean over the firing
  positions. The row also holds the full model's own read at those positions (`natural_silent_max`, which must be 0)
  and a check that the capture reproduces the row's SF.
- **Selective pass** = production pass AND SF pool C (fit) ≤ 0.01; also reported at ≤ 0.05. Circuit size n.

**Report** (`report.py` → `out/report.md`):
- per arm: pass, selective pass, SF_C median / IQR, magnitude, n, Z / A / C / nec, resample pass, fit time;
- by group (hi / sel / fail);
- paired deltas against v4 with sign counts and lost / gained pass flips;
- per-target table;
- a sanity block: in-run v4 vs the 11-07 production values;
- the run's checks.

**Doubts, flagged before the run:**
- **O charges sub-threshold lifts.** SF counts only positions where the target enters the Top-K, but O charges any
  rise of the pre-activation above natural, even one that stays below the cut. So O is stricter than the read it
  targets, and it may cost Z / A / C more than needed (pulling α down on members that are also active at the
  anchors). A Top-K-gated version would charge only where the target would enter: `relu(pre − floor)` × a soft gate on
  pre − τ_K, the censor term's gate at every position. It is not built; it is the follow-up if O costs too much. The
  pre-activation hinge was chosen to match S and the brief.
- **Training and test pools differ.** The training contrast rows (48) and the test's silent rows (16) are different
  contexts from the same "close" selector. O fits only the 48, so held-out SF measures generalisation. Overfitting to
  the 48 is possible: a large O weight could buy low training lift without held-out selectivity. The magnitude read on
  held-out rows is what counts.
- **What O asks for is narrower than "selective".** It wants members to stay off from their own silent-context state.
  It does not ask the circuit to encode the gate on contexts unlike the close contrast set (pool R, random text, is
  reported but not trained).
- **The baseline is a refit.** The in-run v4 baseline differs from production by GPU, DEEP_BS and kernel noise (092:
  refits move single-target passes). Read 1-2 target differences as noise. The panel is small (16) by design.
- **Deep band is L8-9 only** (16 GB VRAM; L10-11 did not fit in exp 121). The worst production cases (L10-11, large
  attention circuits) are not covered.
- **One donor per row per step adds noise to O's gradient.** O's loss is not a smooth function of the gates.

## Panel (`select_panel.py`, seed 20261118; `out/panel.json`, `targets.txt`)

The panel has 16 GPT-2 small targets from exp 130 `out/per_target.csv` (11-07 production, held-out strongest,
non-vacuous): one per cell, deep band L8-9. The queue order is the order below. Production values:

| # | cell | target | n | Z / A / C / nec | pass | SF_C | φ_ind |
|---|---|---|---|---|---|---|---|
| 1 | hi / L0-4 / attn | 4.attn.25180 | 701 | 0.98 / 0.94 / 0.92 / 1.00 | P | 0.722 | 0.00 |
| 2 | hi / L5-7 / attn | 7.attn.12712 | 823 | 0.79 / 0.86 / 0.79 / 1.00 | P | 0.938 | 0.21 |
| 3 | hi / L8-9 / attn | 9.attn.30928 | 708 | 0.91 / 0.87 / 0.92 / 1.00 | P | 0.801 | 0.00 |
| 4 | sel / L0-4 / resid | 1.resid.10656 | 232 | 0.99 / 1.00 / 1.01 / 0.91 | P | 0.009 | 1.65 |
| 5 | hi / L0-4 / mlp | 3.mlp.15959 | 1073 | 0.94 / 1.01 / 0.93 / 1.00 | P | 0.337 | 1.38 |
| 6 | hi / L5-7 / resid | 5.resid.6963 | 194 | 0.92 / 0.95 / 0.95 / 1.00 | P | 0.238 | 1.32 |
| 7 | fail / L5-7 / attn | 6.attn.28217 | 948 | 0.91 / 0.71 / 0.90 / 1.00 | F | 0.855 | 0.00 |
| 8 | hi / L8-9 / resid | 8.resid.11283 | 420 | 0.88 / 0.94 / 0.94 / 1.00 | P | 0.705 | 1.15 |
| 9 | sel / L0-4 / mlp | 1.mlp.22405 | 421 | 0.91 / 0.93 / 0.96 / 1.00 | P | 0.002 | 0.56 |
| 10 | hi / L0-4 / resid_mid | 3.resid_mid.15558 | 525 | 0.95 / 1.06 / 1.02 / 1.00 | P | 0.530 | 1.15 |
| 11 | hi / L5-7 / mlp | 6.mlp.4271 | 1592 | 0.83 / 0.88 / 0.82 / 1.00 | P | 0.291 | 0.00 |
| 12 | sel / L5-7 / resid_mid | 6.resid_mid.21305 | 246 | 1.01 / 1.01 / 1.01 / 1.00 | P | 0.003 | 1.56 |
| 13 | hi / L8-9 / mlp | 9.mlp.6797 | 688 | 0.85 / 0.81 / 0.79 / 1.00 | P | 0.690 | 0.68 |
| 14 | fail / L8-9 / resid | 9.resid.15615 | 1370 | 0.77 / 0.67 / 0.47 / 1.00 | F | 0.563 | 1.10 |
| 15 | hi / L8-9 / resid_mid | 9.resid_mid.22889 | 215 | 0.91 / 0.89 / 0.91 / 1.00 | P | 0.887 | 1.08 |
| 16 | sel / L8-9 / mlp | 8.mlp.28791 | 161 | 1.05 / 1.04 / 1.04 / 1.00 | P | 0.002 | 0.87 |

Cell groups:
- **hi** (10): passing, SF_C ≥ 0.2.
- **sel** (4): passing, SF_C ≤ 0.01. These are the controls; O should cost them nothing.
- **fail** (2): failing, SF_C ≥ 0.2.

## CPU tests (`test_off_engine.py`, one real target 1.mlp.22405, 4 steps, CPU)

All passed on 2026-11-18 (359 s; `out_cputest/20261118_175530/test_results.json`; log `out/logs/cpu_test.log`):
- **(+)** Every anchor applies exactly once to today's src.
- **(a)** At w_O = 0, the patched engine matches the unpatched src engine bit for bit: identical circuit (members and
  coefficients), identical loss curve, identical per-term curves (zero, floor, pos, resample).
  Caveat: at 4 steps nothing is pruned yet (n = 196,608 in both), so this checks the forward and backward and the loss
  numbers, not the selection.
- **(b)** At w_O = 1 the terms are zero, floor, pos, resample and off.
  - The off curve is 5.3e-6 at step 0 and then 0.00 to 6 decimals (4 steps: gates still about 1, α about 1).
  - On the first off forward, the gradient of O reaches the gates (|∇θ| 4.3e-4) and the coefficients (|∇ψ| 9.9e-3).
- **(c)** The off tokens equal `eval_reference`'s C rows, which equal the R donors and `rec["neg"][:48]`.
  - 0 of the 48 rows equal a held-out contrast row.
  - The rows used were indices 0-15 (4 steps × batch 4; max index 15 < 48).
  - Donor rows were e.g. [6, 15, 46, 27] for base rows [0, 1, 2, 3]; 0 self-draws in 4 draws.
- **(d)** The full model's post-Top-K target on the 16 verified-silent held-out contrast contexts is 0 at every
  position (max 0.0). The capture reproduces the donor row's own SF exactly (diff 0.0).
- **Natural pre-activation.** On this target's 48 contrast-train rows it is > 0 at only 0.26% of positions (max 0.87).
  So the O floor is 0 almost everywhere, and O charges the positive part of pre. Lifts that stay below 0 are free.
  This narrows the first doubt above: O charges a pre-activation that becomes positive but stays below the Top-K cut.

**End-to-end CPU smoke of `run.py` + `report.py`.** 1.mlp.22405, arms v4 and o1, 3 steps, 1 donor draw; log
`out/logs/cpu_smoke_run.log`, output `out_cputest/smoke_run/`.
- Both rows were written: production score, donor test, magnitude read, fits and train rows.
- The engine config was printed as v4: sparse fill, eval_ref means, R 1.0 / contrast / 1, censor 0.05, rank 0.1, lr
  0.1575 geometric, anneal, deep_bs 1.
- The report renders every table, and its checks block is clean.
- A rerun's plan shows 0 to do: resume works.
- CPU time per (target, arm) was about 4 min: fit 75 s, the rest scoring and the donor test.
- The dry run of `queue.sh` lists the 64 (target, arm) fits with estimates. Today it says a real start would be refused:
  the 081 queue holds `.gpu-lock` and the GPU (13.4 GB).

## Files

| file | what |
|---|---|
| `off_engine.py` | the in-memory engine patch (5 anchored edits + the off patcher / pool / loss) |
| `run.py` | fits + production scorer + donor test + magnitude read; resumable per (target, arm); `DRY_RUN=1` plan with estimates |
| `queue.sh` | the GPU queue: lock, idle / RAM checks, watchdog (per-fit trips skip the target), STOP_EPOCH, phases, final report |
| `report.py` | `out/report.md` |
| `select_panel.py` | the panel (`out/panel.json`, `targets.txt`) |
| `test_off_engine.py` | CPU tests (a)-(d) |
| `out/rows.jsonl`, `fits.jsonl`, `train.jsonl`, `circuits/<arm>/` | results; `current.json`, `tripped.txt`, `queue.log`, `watch.csv`, `logs/` from the queue |

## Commands (WSL, repo root)

    # CPU tests (no GPU; ~6 GB RAM)
    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -X utf8 experiments/131-off-term/test_off_engine.py
    # the plan
    DRY_RUN=1 STOP_EPOCH=$(date -d 'tomorrow 07:45' +%s) bash experiments/131-off-term/queue.sh
    # the night (after the 081 queue has released .gpu-lock)
    STOP_EPOCH=$(date -d 'tomorrow 07:45' +%s) nohup bash experiments/131-off-term/queue.sh > experiments/131-off-term/out/queue.out 2>&1 &
    # the report (any time; CPU)
    .venv/bin/python -I experiments/131-off-term/report.py

## Estimate

The estimate is in `run.py` `EST_FIT`. Seconds per v4 fit by layer on this RTX 5070 Ti come from exp 121's local GPT-2
fits (median per layer): L1 60, L3 125, L5 205, L6 220, L7 260, L8 300, L9 450 (L9 was memory-bound there, at 600-770 s
under 121's train patch). L4 is interpolated. Then:
- **O arm:** ×1.25 (one extra forward and backward per micro-batch over 4 terms).
- **Scoring + donor test + magnitude:** about 90 s per (target, arm) (pod: score 5 s, donor 28 s).
- **Start-up:** about 2 min per launch.

| phase | fits | estimate |
|---|---|---|
| `v4,o1` | 32 | 92 + 109 = ~200 min |
| `o0.25` | 16 | ~109 min |
| `o4` | 16 | ~109 min |
| total | 64 | **~7.0 h** |

The window from ~21:30 (when the 081 queue frees the GPU) to 07:45 is 10.25 h, so all four arms should fit with about
3 h to spare.
- **Main uncertainty:** `DEEP_BS=1` on L5+ targets, which have more than 21 sites. It runs 4 micro-forwards of batch 1
  per step, where 121 ran one of batch 4, and the slowdown is unmeasured on GPT-2. At 1.5× on the deep fits the total
  is about 9 h. Then `o4`, the last phase, may be cut by STOP_EPOCH; the cut is resumable.
- **STOP_EPOCH rule:** `run.py` does not start a (target, arm) whose estimate would end after STOP_EPOCH, and the queue
  kills everything at STOP_EPOCH + 15 min.

## Results

Not run yet.

## Deep slice (2026-11-20)

Linear DAN-222's deep-target arm, approved by Daniel on 2026-11-20. The off-term appendix (paper `app:off-term`) says the
pilot had 9 targets on one model and one seed, none beyond layer 6, where production silent firing is highest. This
slice asks whether O still cuts silent firing on **deep** targets (L8-11) without losing passes. It does not change the
pod: the production recipe carries no off term.

### Design (written before any GPU result)

**Why the pilot had no deep targets.** The pilot's seven L7-9 targets were not out of memory. The queue's watchdog
killed each one 1-3 min into its v4 fit, when `nvidia-smi` memory.used passed 15,300 MiB (15,512-15,829 MiB seen;
`out/queue.log`). Exp 121 showed GPT-2 L8-9 fits finishing on this card at full VRAM, memory-bound (300-770 s), and L10-11
fits spilling to host memory (29 min, killed). So this slice raises the GPU watchdog above the card's 16,303 MiB
(`N131_GPU_MAX_MIB=16400`, i.e. off) and guards with time instead: a per-(target, arm) cap `N131_FIT_MAX_S=1800`
(a trip puts the target in `tripped.txt`, skipped in every arm; at most 12 trips), the RAM watchdog (< 3 GB) as before,
and the night's hard end.

**Targets** (`select_deep.py`, seed 20261120, from exp 130 `out/per_target.csv`: 11-07 production, held-out strongest,
non-vacuous, **passing only**). 12 cells stratified by site kind: 9 `hi` (production SF_C ≥ 0.2: attn ×3, mlp ×2,
resid ×2, resid_mid ×2) and 3 `sel` controls (SF_C ≤ 0.05: mlp, resid, resid_mid). Two panels are drawn, because
L10-11 is decided by a timed fit:
- `out_deep/panel_L8-11.json`: bands alternate L8-9 / L10-11 within each kind (6 targets at L10-11).
- `out_deep/panel_L8-9.json`: every cell at L8-9.

Queue order, production values (n, SF_C):

| # | L8-11 panel | n | SF_C | L8-9 panel | n | SF_C |
|---|---|---|---|---|---|---|
| 1 | hi 10.mlp.5635 | 295 | 0.645 | hi 9.mlp.19676 | 217 | 0.519 |
| 2 | hi 9.attn.24133 | 295 | 0.979 | hi 9.attn.24133 | 295 | 0.979 |
| 3 | sel 9.resid_mid.29737 | 235 | 0.001 | sel 9.resid_mid.29737 | 235 | 0.001 |
| 4 | hi 11.resid.22616 | 623 | 0.799 | hi 8.resid.29279 | 357 | 0.465 |
| 5 | hi 9.resid_mid.22889 | 215 | 0.887 | hi 9.resid_mid.22889 | 215 | 0.887 |
| 6 | hi 10.attn.27546 | 1222 | 0.616 | hi 8.attn.17680 | 627 | 0.892 |
| 7 | sel 10.mlp.27800 | 1165 | 0.001 | sel 9.mlp.31090 | 180 | 0.000 |
| 8 | hi 8.resid.9265 | 697 | 0.688 | hi 8.resid.27749 | 1034 | 0.655 |
| 9 | hi 10.resid_mid.12302 | 907 | 0.524 | hi 8.resid_mid.31397 | 456 | 0.222 |
| 10 | hi 8.attn.5856 | 356 | 0.290 | hi 8.attn.11727 | 738 | 0.860 |
| 11 | sel 8.resid.29870 | 202 | 0.001 | sel 9.resid.22051 | 214 | 0.003 |
| 12 | hi 8.mlp.30479 | 707 | 0.683 | hi 8.mlp.30479 | 707 | 0.683 |

**The L10-11 test.** The first L8-11 target, `10.mlp.5635`, is fitted alone in arm v4 into `out_deep/` (cap 20 min
for fit + score + donor test). Rule, fixed now: if it finishes with fit time ≤ 900 s, the L8-11 panel runs (that row is
its first v4 row); otherwise the L8-9 panel runs, and L10-11 is reported as not covered.

**Arms.** `v4` (the in-run refit control, w_O = 0), `o1`, `o4` (the pilot's best range), in one phase, target-major:
all three arms of target 1, then target 2, ... so any prefix of the queue is a set of complete targets. Phase 2 is `o0.25`
on the same targets, only if time is left.

**Same as the pilot.** `run.py` unchanged: 062 `driver.Runner`, WCM v4 (λ 1e-3, 400 steps, R 1.0, rank-keep 0.1, lr
0.1575 geometric, censor 0.05), DEEP_BS 1, O's donor generator seed 20261118, production scorer and pass rule, donor
test (8 draws, variants fit / a1, pools A / C / R as production), resample pass, the magnitude read. Selective pass =
pass AND SF_C ≤ 0.01 (also ≤ 0.05 reported).

**Size and time.** Aim: 8-12 targets × 3 arms. Each L8-9 target is about 3 × (300-770 s fit + ~90 s eval), i.e.
20-45 min. The queue lists all 12 and the night's clock cuts it: no new (target, arm) after ~07:30 (`STOP_EPOCH` 07:36
with `run.py`'s per-arm estimate), everything killed at 07:56 (`N131_HARD_GRACE=1200`), lock released by 08:00. Only
complete targets (all of v4, o1, o4) enter the paired read; a cut target is reported as cut.

**Decision rule (DAN-222).** Adopt only if selective passes rise, pass does not fall, and median size cost ≤ 15% (median
paired n ratio ≤ 1.15). This slice reports what the rule says on its targets; one model and one seed is not a decision.

**Doubts.**
- Memory-bound fits run on a card at full VRAM; their times are slow and noisy, but the fit itself is the same
  computation (no CPU fallback inside the engine; spilling only moves memory).
- In-run v4 differs from production by GPU / DEEP_BS / noise (sanity block in the report).
- 12 targets at most: read 1-2 target differences as noise.

**Code changes for this slice** (pilot behaviour unchanged by default): `select_deep.py` (new); `report.py --panel`
(the panel file and its group names); `queue.sh` `N131_LOCK_OWNER` (the caller holds `.gpu-lock`), `N131_FIT_MAX_S`
(time-cap trip), `N131_PANEL`, and `TARGETS` passed through to `run.py`. CPU smoke: `out_deep_cputest/`.

### Deep slice: CPU smoke and the L10-11 test

- **CPU smoke** (2026-11-20 00:55-01:40; `out_deep_cputest/`, log `logs/cpu_smoke.log`): `9.attn.24133`, arms v4 and
  o1, 3 steps, 1 donor draw, through `run.py` with `KEYS` / `OUT131`. Both rows written; v4 terms zero / floor / pos /
  resample, o1 adds off; off rows = R donors = `rec["neg"][:48]`, 0 in the held-out set, 0 self-draws; full model 0 at
  the silent positions; capture = row SF. `report.py --panel` renders. CPU time 27 + 18 min (fit 615 / 313 s).
- **L10-11 test** (GPU, lock `offdeep-1120` taken 02:58; `out_deep/logs/l10_test.log`): `10.mlp.5635` v4 fit **291 s**,
  peak allocated 14.8 GB (`nvidia-smi` 15.9 GB, ~172 W: not spilling), total 344 s. By the rule above the **L8-11
  panel runs**. That row (in-run v4: n 299, Z 0.81 A 0.48 C 0.46, fails; production passed at 0.81 / 0.86 / 0.84) is
  the panel's first v4 row: a refit pass flip, of the kind the sanity block counts.
- Queue launched 03:08 (`out_deep/queue.log`), `TARGETS=out_deep/targets_L8-11.txt`, phases `v4,o1,o4` then `o0.25`.

### Deep slice results

All numbers from `out_deep/report.md` (`report.py --out out_deep --panel out_deep/panel_L8-11.json`). Phase
`v4,o1,o4` finished **12 / 12 targets × 3 arms** (03:08-06:20, 0 watchdog trips, 0 error rows); 6 targets at L10-11.
Every check is clean (off rows never held out, = R donors, 0 self-draws, full model 0 at the silent positions,
capture = row SF).

| arm | N | pass | selective ≤ 0.01 / ≤ 0.05 | SF_C median (IQR) | mag_C mean / p95 | n median | resample pass | fit s median |
|---|---|---|---|---|---|---|---|---|
| v4 | 12 | 10 | 2 / 3 | 0.576 (0.212-0.739) | 0.478 / 0.909 | 438 | 4 | 219 |
| o1 | 12 | 9 | 2 / 2 | 0.196 (0.066-0.425) | 0.116 / 0.605 | 591 | 3 | 288 |
| o4 | 12 | 9 | 2 / 4 | 0.048 (0.006-0.083) | 0.022 / 0.093 | 869 | 3 | 288 |
| o0.25 | 12 | 9 | 2 / 2 | 0.368 (0.178-0.698) | 0.286 / 0.838 | 631 | 3 | 288 |

Phase `o0.25` (time allowed) also finished all 12 targets (06:20-07:32); everything was off the GPU at 07:32 and
`.gpu-lock` was released then.

Paired vs v4 (median delta [lower / equal / higher]):

| arm | pass | selective ≤ 0.01 | selective ≤ 0.05 | resample | dSF_C | dZ / dA / dC | n ratio | fit-time ratio |
|---|---|---|---|---|---|---|---|---|
| o0.25 | 10 → 9 (lost 2, gained 1) | 2 → 2 | 3 → 2 (lost 1) | 4 → 3 | −0.043 [9− / 0 / 3+] | −0.026 / −0.033 / −0.019 | 1.07 | 1.31 |
| o1 | 10 → 9 (lost 3, gained 2) | 2 → 2 | 3 → 2 (lost 1) | 4 → 3 | −0.248 [11− / 0 / 1+] | −0.012 / −0.014 / −0.018 | 1.05 | 1.31 |
| o4 | 10 → 9 (lost 2, gained 1) | 2 → 2 | 3 → 4 (gained 1) | 4 → 3 | −0.520 [11− / 0 / 1+] | −0.007 / −0.012 / −0.051 | 1.24 | 1.31 |

Group hi (9): SF_C median 0.680 → 0.320 (o1) → 0.071 (o4), falling on 9 / 9 targets in both arms; pass 7 → 7 → 6;
n ratio 1.30 (o1), 1.52 (o4); no hi target becomes selective at ≤ 0.01, one does at ≤ 0.05 (8.resid.9265, o4).
Group sel (3): firing stays ≤ 0.04; o1 loses one pass (10.mlp.27800: Z 0.73), o4 keeps all 3.

**Reading.**
- **O still cuts silent firing at depth, and more steeply in w than in the pilot:** median SF_C 0.58 → 0.05 at
  w_O = 4 (pilot, L0-7: 0.25 → 0.02); the magnitude read falls with it (mean 0.48 → 0.02 of the natural anchor
  activation). This holds at L10-11 too (e.g. 10.attn.27546 0.58 → 0.007).
- **Unlike the pilot, it costs passes and size.** Pass falls 10 → 9 in both arms, with flips both ways (o4 lost
  11.resid.22616 at Z 0.72 and 10.attn.27546 at C 0.69, gained 10.resid_mid.12302). One fit degenerated:
  10.mlp.5635 at o4 gives Z = A = C = 0.00 (n 272): O drove the circuit to switch the target off everywhere. Median
  size grows 24% at o4 (52% on the hi group), against 7% in the pilot.
- The in-run v4 baseline itself flipped 2 of 12 production passes (10.mlp.5635, 10.resid_mid.12302; sanity block),
  so 1-2 target pass differences are within refit noise.
- **DAN-222 rule** (selective passes rise, pass does not fall, median size cost ≤ 15%): **o0.25 fails** (selective
  2 → 2, pass 10 → 9); **o1 fails** (selective
  2 → 2, pass 10 → 9); **o4 fails** (selective ≤ 0.01 2 → 2, ≤ 0.05 3 → 4; pass 10 → 9; n ratio 1.24). On this slice
  the rule says do not adopt. One model, one seed, 12 targets: not a decision.
