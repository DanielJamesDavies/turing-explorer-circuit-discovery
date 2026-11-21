# 127 Norm rescale: are WCM's fitted amplitudes mostly norm repair?

## Question

In a circuit-only run (non-members ablated; WCM v4 censored reads Z / A / C) the member sites keep only about half
of their natural latent-reconstruction norm. Exp 118 §4 measured ‖x̂_edit‖ / ‖x̂_nat‖ at the anchor on the 16
held-out rows: median 0.53 (Z) at α = 1 and 0.91 with the fitted α. Does α pass the circuit **because** it puts that
norm back? Or do the per-member weights carry structure that a norm match does not supply?

Constants are already covered. 118 job 3 (README §2) refitted one scalar per site and one per circuit on the
pass-rule objective itself, against production α, on 54 targets:

| arm | pass |
|---|---|
| production α | 85.2% |
| α = 1 | 3.7% |
| per-site scalar | 37.0% (L8-11: 0 / 18) |
| per-circuit scalar | 33.3% |

So 127 does not refit constants. Its new test is the strongest form of norm repair: an **oracle** that matches the
natural norm at every site, row and position.

## Arms

All arms go through 118's reduced scorer (`run118.Ctx`: 049 `V.read`, the engine `CircuitOnlyPatcher` with
`keep_scales`, the same contexts, means and empty baselines). Each arm is scored on Z, A and C, on the held-out rows
(**ho**, 16) and on the training rows (**home**, the 48 strongest-training rows).

- Pass rule: 062 merge (Z, A, C ∈ [0.75, 1.5] and necessity ≥ 0.9).
- Necessity and vacuity come from 118's full production score of the same circuits, which is α-independent.
- home uses the ho necessity, since no training-row necessity exists.
- xdoc is not available: TuringLLM records have no per-window document split.

**Main arms (run first):**

| arm | what |
|---|---|
| `alpha1` | Every member amplitude is 1. Reproduces 118's α = 1 variant. |
| `fitted` | Production α. Reproduces 118's `red_fitted`. |
| `oracle` | α = 1. Then, at every member site and every (row, position), the site's latent reconstruction x̂_edit = decode(patched code) is scaled so that ‖s·x̂_edit‖ = ‖x̂_nat‖ of the same row and position in a natural run. This uses natural information, so it is an upper bound on what norm repair can do. |
| `oracle_lat` | As `oracle`, but only the latent part W_dec·z is scaled; the decoder bias b is kept: ‖s·W z + b‖ = ‖x̂_nat‖ (positive root; s = max(0, −(Wz·b)/‖Wz‖²) when no root exists). This is the bias-free reading of "scale the reconstruction". |
| diagnostic | Measured on every arm and every fill, on the ho rows, at the anchor. See below. |

**Diagnostic.** Measured at three places:

- every RMSNorm input (`norm_1` / `norm_2` of each block) after the earliest member site and before the target's read;
- the target's own read norm (attn targets: `norm_1` of their layer; mlp and resid targets: `norm_2`);
- the target SAE's input x at the seed site.

At each point it records:

- the cosine of the post-norm vector to the natural one;
- ‖post‖ / ‖post_nat‖ and ‖pre‖ / ‖pre_nat‖;
- the member-site reconstruction norm ratio. For `alpha1` and `fitted` this ratio is 118 §4's 0.53 / 0.91: a
  reproduction check.

RMSNorm removes scale, so a norm repair can only help through the **direction** of the residual stream it produces.
The post-norm cosine is the number to read.

**Reference arm (run after the main ones):**

| arm | what |
|---|---|
| `rescale_site` | α = 1, one scalar per (circuit, member site): the median, over the training rows (anchor) and the three fills, of ‖x̂_nat‖ / ‖x̂_edit‖. It is applied unchanged to ho and is shared across fills, as α is. |

- Calibration takes `N127_SITE_ITERS` [3] passes. Each pass measures every site under the previous pass's scalars
  upstream, so a downstream scalar sees rescaled upstream sites.
- The scalar is matched to the norm, not optimised: a cheap reference row. 118 job 3's refitted per-site and
  per-circuit scalars are reported beside it, paired on the same targets.
- `rescale_global` (one scalar per circuit) was dropped, because 118 job 3 already refitted it (33%).

## Hook point and SAE error

The rescale is defined in `run127.py` as a subclass (`Resc`) of 118's `TapCO`, which is the engine's
`CircuitOnlyPatcher` plus the seed tap. Nothing in `src/` is changed.

- The engine's `transform` returns decode(patched) + error, where error = x − decode(encode(x)) of the live
  (already edited) stream.
- The engine preserves that error in all three reads (Z, A, C; the default `keep_error_sites = None`).
- `Resc` calls that `transform` unchanged and recomputes the same error the same way (as 118's capture did). It
  takes x̂_edit = out − error and returns s·x̂_edit + error. For `oracle_lat` it returns s·(x̂_edit − b) + b + error.
- **So the error term is never scaled, in any read, and is the same error the pipeline preserves.**
- In Z, x̂_edit = W·(α ⊙ members) + b. In A and C it also includes the scorer's Top-K mean fill of non-members.
  `oracle` scales the whole vector, including b.
- A scalar on this vector is not the same as scaling every α uniformly, because b, the A / C fill and the error are
  not α-scaled. That difference is why it is implemented on the vector and not as α.
- `alpha1` and `fitted` return the engine output itself, so they stay bit-identical to 118. The capture there only
  reads.
- The run reports ‖b‖ / ‖x̂_nat‖ at member sites (`bdec_share_med`), so the size of the bias question is visible.

## Targets

- **Main: 118 job 3's 54 targets** (`118/targets_fit54.txt`, 6 per band × kind cell), so the results pair with 118
  job 3's table.
- After that, the other 162 of 118's 216 (`targets_tl.txt` minus the 54) run as an extension.
- All are production WCM v4 TuringLLM circuits:
  - circuits from the T3 archive;
  - context records from T2 `ctx_fresh`;
  - both read only, through `run118`'s paths.

## How to run (WSL, repo root)

```
DRY_RUN=1 bash experiments/127-norm-rescale/run_night.sh          # the plan; changes nothing
STOP_EPOCH=$(date -d 'tomorrow 08:00' +%s) nohup bash experiments/127-norm-rescale/run_night.sh \
    > experiments/127-norm-rescale/out/run_night.out 2>&1 &
```

**Stages,** in order:

1. `main54`: alpha1, fitted, oracle, oracle_lat and the diagnostic.
2. `analyze1`: writes `out/report.md`. The main answer is in place at this point.
3. `site54`: rescale_site.
4. `main162`.
5. `site162`.
6. `analyze2`.

**Safety and control:**

- **Refuses to start** when `.gpu-lock` exists, when the GPU holds more than `N127_GPU_IDLE_MIB` [2500] MiB, or when
  `out/ABORTED` exists.
- **Lock:** writes its own `.gpu-lock` ("pid $$") and removes it at exit only while the lock still names this pid.
- **Watchdog:** stops the run if the GPU goes above 15300 MiB or host RAM drops below 4 GB (exit 9, `out/ABORTED`).
- **`STOP_EPOCH`:** no new stage after it. It is passed through to `run127.py --stop-epoch`, which starts no new
  target after it and exits 7.
- **Resume:** each stage is resumable. A stage counts as done (`out/state/<stage>.done`) only when every target has
  an ok row. Error rows are retried on a rerun.
- **Selecting stages:** `N127_STAGES` / `N127_SKIP` pick which stages run.

**One script directly:**

```
PYTHONPATH=src .venv/bin/python -X utf8 experiments/127-norm-rescale/run127.py --targets FILE --out DIR --arms ...
```

Then `python -I -X utf8 experiments/127-norm-rescale/analyze.py`.

**`out/report.md`, in order:**

1. Pass rates per arm, on ho and on home, with paired Δ vs fitted (bootstrap CIs over targets), the share of the
   α = 1 → fitted gap, by depth band, and the failure side.
2. The downstream-read table per fill.
3. The oracle scale distribution, the rescale_site scalars and the decoder-bias share.

## What each outcome means

| outcome | reading |
|---|---|
| `oracle` ≈ `fitted` (pass and downstream cosine) | α is mostly norm repair: once the site norm is put back, which members carry it hardly matters. Then even `rescale_site`, or a better fitted constant, should approach fitted, and 118 job 3's 37% is fitter-limited. |
| `oracle` ≪ `fitted` (e.g. near `alpha1`, or below the per-site refit) | Even a perfect, per-position, natural-information norm match does not pass. The per-member weights carry structure (the direction of the reconstruction), not just its norm. The post-norm cosine should then be lower for `oracle` than for `fitted`. |
| `oracle` in between | Norm explains part of the gap. The "share of the α = 1 → fitted gap" column quantifies it, and the band table says where it holds (118: per-site constants fail every L8-11 target). |
| `rescale_site` ≈ `fitted` | α is mostly norm repair. |
| `rescale_site` ≪ `fitted` | The per-latent weights carry structure. |
| `oracle` ≫ `rescale_site` | Norm repair needs context-specific gains, which constants cannot give. |
| `oracle_lat` vs `oracle` | Says whether scaling the decoder bias matters. |

The pass rule is two-sided (Z, A, C ∈ [0.75, 1.5]), so an oracle can also fail high. The report counts failures
below and above.

## Runtime (local RTX 5070 Ti)

118 job 1 timings on the same 54 targets (reduced scorer only):

- 10 no-refit variants × 3 fills × 16 rows took 4.3 s per target (median; the build included);
- the whole of job 1 took 37 min for 216 targets.

A main unit here is about 13 such read-equivalents:

- 3 arms + oracle_lat, × 3 fills, × (16 + 48) rows;
- a natural pass over 64 rows;
- one extra encode per member site in the rescaled arms.

Estimates:

| | per target | 54 targets | all stages, 216 targets |
|---|---|---|---|
| main | ~15–30 s (estimated) | ~15–30 min | |
| rescale_site | ~10–20 s | ~10–20 min | |
| total | | | ~1.5–2.5 h |

The runner's dry run prints its own plan at the pessimistic 30 s / 20 s per target. These are guesses from 118's
timings; the GPU was not used while building this experiment, so the first night's `out/main54/rows.jsonl`
(`t_arm`, `t_total` per target) is the real measurement. Memory per target is the same as 118 job 1, which ran
these targets on this card at EVAL_BS 16.

## Results (GPU night 2026-11-17 01:55-03:13, all 216 targets, 0 failed stages; `out/report.md`)

α = 1 and fitted reproduce 118 job 3 exactly on the 54 (3.7% and 85.2%). The report's automatic reproduction line
reads "nan" (the per-target diff did not join); the pass rates and medians match.

| held-out pass % | all 216 | job-3 54 | L0-4 / L5-7 / L8-11 (216) |
|---|---|---|---|
| fitted α (production) | 81.5 | 85.2 | 86 / 90 / 68 |
| α = 1 | 2.8 | 3.7 | 8 / 0 / 0 |
| oracle norm match, per row and position | 6.9 | 9.3 | 21 / 0 / 0 |
| oracle, latent part only (bias kept) | 11.6 | 18.5 | 29 / 6 / 0 |
| site scalar, median train ratio | 8.3 | 9.3 | 25 / 0 / 0 |
| 118 job 3 refitted per-site scalar | | 37.0 | (54: 72 / 39 / 0) |

- **Norm repair explains little of α.** Even the oracle, which restores every site's natural norm on every row and
  position, closes 5-11% of the α = 1 → fitted gap (18% at best on the 54), and passes no target in L8-11.
- **α changes direction, not size.** Target SAE input, Z run: cosine to natural 0.03 at α = 1, 0.005 for the oracle,
  0.31 for fitted (paired α = 1 − fitted −0.21 [−0.24, −0.18]); the norm ratio is 1.25 / 1.40 / 1.04. Fitted α
  points the target's input towards the natural one; matching norms leaves it orthogonal (and overshoots its norm).
- The CPU test's "fitted lowers the cosine" (two layer-0 targets) does not hold at scale: over 216 targets fitted
  raises it at every read point and fill.
- Reading: the coefficients are per-latent structure, not a norm correction. With 118 job 3 (a refitted per-site
  scalar reaches 37%), per-site or per-context scaling cannot stand in for them, least of all for deep targets.

## Status

Built 2026-11-17; GPU run done 2026-11-17 (above).

**CPU test** (2026-11-17, while exp 126 held the GPU). The full TuringLLM setup needs about 13 GB of host RAM on the
CPU (36 SAEs of 335 MB each, plus about 4.6 GB of stores). The test therefore ran through a scratch shim. The shim
loaded only the layer-0 SAEs and skipped the `top_coactivation` and `logit_ctx` stores, which the reduced scorer
does not read. It is not part of the experiment. Peak RSS was about 6 GB, and each target took about 7–8 min on the
CPU.

Held-out Z / A / C per arm (rows in `out/cpu_test*/rows.jsonl`, report in `out/cpu_test/report.md`):

| target | sites | α = 1 | fitted | oracle | rescale_site |
|---|---|---|---|---|---|
| 0.mlp.23604 | 1 | 0.54 / 0.86 / 0.87 | 1.01 / 1.18 / 1.14 | 0.39 / 0.75 / 0.76 | 0.38 / 0.75 / 0.77 |
| 0.resid.8256 | 2 | 0.61 / 0.65 / 0.62 | 1.05 / 1.11 / 1.07 | 1.01 / 0.99 / 0.96 | 0.93 / 1.03 / 0.96 |

- **Reproduction of 118 (GPU):** max |diff| was 0.004 for α = 1 and 0.010 for fitted on 0.mlp.23604, and
  0.003 / 0.003 on 0.resid.8256. The differences are CPU vs GPU numerics.
- **The oracle matches the norm:** the member-site reconstruction ratio after the oracle is 1.000 exactly.
- **oracle_lat** agrees with oracle on 0.mlp.23604, where the bias share is 0.004. On 0.resid.8256 the bias share is
  0.18.
- **rescale_site calibration converges.** On the 2-site target the scalars are 1.04 (0.attn) and 1.62 (0.mlp).
- **The two targets point in opposite directions.**
  - On 0.mlp.23604, α = 1 already keeps 0.96 of the natural norm. Raising it to natural (×1.04) **lowers** Z
    (0.54 → 0.39), while fitted α also raises the norm (×1.04) and passes.
  - On 0.resid.8256 (α = 1 keeps 0.79), the oracle passes.
- In both cases the downstream post-norm cosine to natural is slightly **lower** for fitted than for α = 1 (0.91 vs
  0.92; 0.89 vs 0.95). Passing does not require the residual stream to look more natural overall.

## Files

- `run127.py`: the scorer (arms, rescale patcher, diagnostics). It imports `run118` read-only.
- `run_night.sh`: the night runner.
- `analyze.py`: writes `out/report.md` and `out/pass_{ho,home}.csv`.
- `out/<stage>/rows.jsonl`: one row per target. `out/cpu_test/` is the CPU smoke test; `analyze.py` ignores it.
