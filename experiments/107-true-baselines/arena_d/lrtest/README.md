# Arena D lr-budget test (agent `dlrtest`; built and checked on CPU, the GPU part not yet run)

## Question

Arena D's WCM (Gemma-2-2B, task-level logit-difference objective through experiment 081's task engine) runs on the pod
pinned to `WCM_VERSION=v2` with arena C's behavioural `V3_LOGIT`:
- Z + A + C + R;
- no rank-keep;
- geometric lr at peak 0.05, no warm-up;
- no censored term.

See `paper/protocol.md` §0.6.1 and `../pod_stage_d.sh`. Would it benefit from v4's pieces?

## Which v4 pieces apply to arena D

| v4 piece | applicable here? | why |
|---|---|---|
| lr budget: geometric, peak ×3.15 (0.1575), warm-up 0.1 | **yes, tested (C1)** | `learned_mask` takes `lr`, `warmup_frac` and `weight_decay` as kwargs for every objective. The wrapper sets them on top of the production kwargs. No engine edit. |
| censored-activation term (`censor_temp` 0.05) | **no: C2 skipped** | `run_learned_mask` raises for `censor_temp` when objective ≠ `pos` ("censor_temp / rival_weight / rank_zero_mult need objective='pos'"). The term censors the seed latent's pre-activation at its own site's Top-K cut. A logit-difference objective has no seed latent and no Top-K read. Both families are JumpReLU (PLT `activation_function.threshold`, Gemma Scope `threshold`; `wcm_gemma.JumpReLUSAE`) with k = d_sae, so the term would **not** be inert on a latent objective (τ is clamped at the threshold). For the task objective, though, it is undefined, not merely inert. Using it here would need a new behavioural version of the term in `learned_mask` (a post-pod engine change, and a design question first). |
| rank-keep 1e-1 | **no: skipped** | `run_learned_mask` raises for `rank_weight > 0` when objective ≠ `pos`. A behavioural rank-keep (the answer token's vocabulary rank) was tried and removed: it collapsed SVA fits (comment at the check; arena C `wcm_marks.py:70-73`). |
| R (γ_R 1.0) | already in C0 | arena C's V3_LOGIT |

One more v4 detail: v4 keeps AdamW's weight decay at the absolute 0.05 at the ×3.15 peak, so Σlr·wd = 1.07. 081's
`configure()` would instead rescale wd to 1/(400·lr) = 0.0159 at lr 0.1575. **C1 overrides it to 0.05, as v4 does.**

Budgets at 400 steps (re-stated `lr_at`; matches the engine's own `lr_sum` on the tiny CPU fits):

| arm | Σlr | decay budget Σlr·wd | sparsity budget at λ 1e-3 |
|---|---|---|---|
| C0 | 6.35 | 0.318 | 0.0064 |
| C1 | 21.40 | 1.07 | 0.0214 |

## Design (as implemented)

- **Arms.**
  - **C0** = the pod's arena D WCM. This is `wcm_gemma.make_engine` and `wcm_gemma.run` unchanged, with lr 0.05 and wd 0.05.
  - **C1** = C0 + peak ×3.15 (lr 0.1575) + `warmup_frac` 0.1 + wd 0.05.
  - **C0s2 / C1s2** = the same two arms with R-donor seed 20261018 (production 20261017), as a replicate.
- **Mechanism** (`lrtest.py`, no edit under `src/`, `experiments/081-task-circuits/` or the arena D production files):
  - one production `GemmaEngine` per family and process;
  - its `engine_kwargs` is wrapped. The production kwargs are built and asserted unchanged (schedule geometric, rank 0,
    R 1.0 k 1, sparse fill + STE, no censor), then the arm's overrides are applied;
  - the peak lr goes through `wcm_gemma.run(lr=)`;
  - each arm has its own `--out` (`out/runs/<arm>/<res|plt>`), because `run()`'s resume keys do not carry the lr;
  - `V3_LOGIT` is extended per arm for provenance only (`lrtest_*` keys in each circuit JSON's `wcm`), then restored.
- **Setup** = `../run_pilot_d.sh`'s `wcm_res` and `wcm_plt` steps, except 400 steps (`pod_stage_d.sh` DW_STEPS) instead of 50:

  | | `res` | `plt` |
  |---|---|---|
  | family | `gemma-2-2b-gsres-gap6` | `gemma-2-2b-plt` |
  | task | rc | rc, simple, nounpp |
  | data | Caples (1,344 train / 400 held out) | SFC (100 / 40) |
  | mode | `--mode caples` | native |
  | means | per_position, the local means file `../out/caples_means/rc_gap6_bf16_SMOKE.pt` | pooled |

  - gsres is rc only, because the Caples data and means exist for rc only.
  - Both families run in bf16 on the unsloth weights (the 16 GB card). The pod runs fp32 google weights. **Labelled
    local, bf16.**
- **λ:** 1e-3, 1e-4, 1e-2. These are three of the pod's 9 values, 1e-3 first.
- **Measured per fit** (from wcm_gemma's `eval.jsonl` = the 081 engine's own held-out `score_task`):
  - size: `n_latents`;
  - `task_free0` (Z), `task_freeM_topk` (A), `task_freeN_topk` (C), `task_nec`, and 081's `task_pass`;
  - `t_fit`, s/step;
  - peak VRAM: torch `peak_fit_mb` / `peak_score_mb`, and nvidia-smi per stage from `out/gpu_mem.csv`;
  - the fit's own lr provenance (`lr_peak`, `lr_sum`, `warmup_steps`, `weight_decay`).
- **Order and budget.**
  - Stages: `res_main` (6 fits), `plt_main` (18 fits: rc, then simple, then nounpp), `res_seed2` (2 fits).
  - Prior projection: 26 fits ≈ 2.1 GPU-h. The GPU-night pilot (`../run_pilot_d.sh`) measured 0.64 s/step on both families, so ≈ 4.5 min per
    fit + scoring, + ~3 min per stage load.
  - `LRTEST_BUDGET_S` (default 10,800 = 3 h, loads included) is checked before every unit, using the largest measured
    unit time of that family. The first fit's s/step and a fresh projection are logged (`FIRST FIT` in the stage log).
    Units past the budget are dropped from the end of the priority order.

## Pre-stated reading (written before any GPU run)

**v4's lr budget helps if it lowers size at equal or better Z / A / C / nec on both families.** Operationally
(`analyze.py`):
1. **Pair** = C0 vs C1 (and C0s2 vs C1s2) at the same family, task and λ.
2. **Equal or better** means all of:
   - |m − 1| of C1 ≤ that of C0 + 0.02, for m = Z, A, C;
   - nec(C1) ≥ nec(C0) − 0.02.
3. **Pair classes:**
   - **better** = equal or better (2) and n(C1) < n(C0);
   - **worse_faith** = any metric outside that tolerance;
   - **not_smaller** = equal or better, but n(C1) ≥ n(C0);
   - **undecided** = a metric is None (vacuous). Undecided pairs are excluded.
4. **A family helps** if ≥ 2/3 of its decided pairs are better **and** no pair passes 081's rule under C0 but fails it
   under C1.
5. **Overall "helps"** = both families help.

Caveat, stated in advance: a larger Σlr at fixed λ also prices sparsity 3.4× harder, so "smaller" can just be a move
along the λ curve. `analyze.py` therefore also reports, descriptively and outside the verdict, where each C1 fit lies
relative to C0's (log size → worst |m − 1|) curve of the same task. A "helps" verdict with C1 on C0's curve means "same
frontier, cheaper λ", not a better optimiser.

## Files and commands (WSL, repo root, `.venv`)

- `lrtest.py`:
  - `check` (CPU): imports; 081 v2 / PROD_LR; the engine's refusals; JumpReLU families; budgets; tiny CPU fits of
    C0 / C1 / C1s2 through the wrapper, with the overrides read back from the fit's own provenance;
  - `plan` (CPU): units, done state, projection;
  - `run --stage S` (GPU; called by run.sh).
- `run.sh`:
  - the queue entry point. Resumable, and honours `STOP_EPOCH`: no new unit after it, and a hard kill
    `LRTEST_HARD_GRACE` = 600 s later;
  - refuses to start if the card holds > 3000 MiB, if MemAvailable < 4 GB, or if another run.sh holds `out/run.lock`;
  - watchdog every 10 s: it kills the stage when the GPU is > 15,300 MiB or host MemAvailable < 4 GB, and no further
    stage starts;
  - one GPU process at a time;
  - `DRY_RUN=1` = files, parse, commands, `plan` and `analyze` on CPU.
- `analyze.py` (CPU): `out/RESULTS.md` + `out/results.json`.

```
bash experiments/107-true-baselines/arena_d/lrtest/run.sh > experiments/107-true-baselines/arena_d/lrtest/out/run.log 2>&1
DRY_RUN=1 bash experiments/107-true-baselines/arena_d/lrtest/run.sh
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 WCM_VERSION=v2 ./.venv/bin/python experiments/107-true-baselines/arena_d/lrtest/lrtest.py check
```

## CPU checks run

- `bash -n run.sh`: OK.
- `DRY_RUN=1 run.sh`: 0 problems. It printed 26 pending units and a projection of 7,690 s, and `analyze` ran on the
  empty dir.
- `lrtest.py check` (`CUDA_VISIBLE_DEVICES=""`): 0 failures. The tiny fits read back from their own provenance:

  | arm | lr_peak | warmup_steps | wd | lr_sum (= re-stated) | R seed |
  |---|---|---|---|---|---|
  | C0 | 0.05 | 0 | 0.05 | 0.1703 | 20261017 |
  | C1 | 0.1575 | 1 of 10 | 0.05 | 0.6444 | 20261017 |
  | C1s2 | 0.1575 | 1 of 10 | 0.05 | 0.6444 | 20261018 |

  - All three: R 1.0, rank-keep off.
  - `V3_LOGIT` was restored after the arms.

## Post-pod changes this test cannot make

- **The censored term for a behavioural objective:** a new term in `learned_mask` (an engine change). The current one is
  undefined without a seed latent.
- **A behavioural rank-keep:** an engine change. It was tried and removed.
- **If C1 helps:** adopting it for arena D means the arena D WCM gets v4's lr budget. That needs a new behavioural
  version (e.g. V4_LOGIT) in `wcm_gemma.py` / arena C's `wcm_marks.py`, plus 081's v1 / v2 guard. It is **not** a change
  to the 081 engine's versions table, and it is Daniel's decision.
