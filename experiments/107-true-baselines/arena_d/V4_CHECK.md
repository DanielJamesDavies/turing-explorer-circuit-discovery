# Arena D v4 quick check (RESUME note; written 2026-11-16 by agent arena-prep, NOT run)

## What changed (D7, Daniel 2026-11-16: arena D's WCM uses the production version v4)

- `wcm_gemma.py`: new arena D WCM version switch `--wcm v3|v4`, env `ARENA_D_WCM`, default **v3** (so the pilot,
  lrtest and every local run keep their meaning).
  - **v4** = v3 (arena C's `V3_LOGIT`: Z + A + C + R at γ_R 1.0, sparse fill + STE, geometric lr, no rank-keep) plus
    every v4 piece that is defined for a logit-difference objective:
    - peak lr × 3.15 (0.05 → 0.1575; `062 driver.WCM_LR_MULT["v4"]`);
    - warm-up 0.1 (40 of 400 steps);
    - v4's **absolute** AdamW decay 0.05, instead of 081's lr-rescaled 1/(400·lr) = 0.0159.
  - This is exactly lrtest's arm C1.
  - **Not applied** (as before): v4's censored read and rank-keep 1e-1.
    - `run_learned_mask` refuses both for `objective != 'pos'`: there is no seed latent and no Top-K cut in a task
      objective.
    - The behavioural rank-keep collapsed SVA fits (arena C `wcm_marks.py:70-73`).
    - So "v4" for arena D = v4's optimisation budget on the behavioural loss. **Stated in every circuit JSON**
      (`wcm_tag`) and in `WCM_VERSION`.
  - Provenance:
    - every circuit JSON carries `wcm_version` and `lr_peak`;
    - every eval row carries `wcm` (`"v4"`, was the constant `"v3"`), and `fit_stats` gains `lr_peak`, `lr_sum`,
      `warmup_steps` and `weight_decay`.
  - An `--out` whose `WCM_VERSION` holds the other version's fits is refused (the resume keys carry neither the lr nor
    the version).
- `pod_stage_d.sh`: new knob `DW_WCM`, default **v4**, exported to every python step as `ARENA_D_WCM`, and shown in the
  start line.
  - `WCM_VERSION=v2` stays in `BASE`. It is 081 `task_common`'s own fill table: v1 / v2 only, and the chain's v4 would
    make 081 exit at import. `V3_LOGIT`'s sparse fill + STE override it under both versions. It is **not** the arena D
    version; the comments now say so.
- `test_wcm_gemma.py`:
  - `_fit_tiny` takes `wcm`;
  - new CPU test `test_tiny_fit_v4_lr_budget`: env selection; provenance read back from the fit (peak 0.1575, warm-up,
    wd 0.05, R on, no rank / censor); 081 still at v2; the mixed-version `--out` refused.
  - **21 passed, 3 deselected (gpu)** on CPU, WSL, 2026-11-16 (226 s).
- No change: `src/`, 081, arena C's `wcm_marks.py`, `lrtest/`, `chain.sh`. The chain's `arena_d_stage` passes no WCM
  env, so the pod gets `DW_WCM=v4` by default.

## Read this before trusting v4 on arena D: lrtest's verdict on exactly this recipe

`lrtest/out/RESULTS.md` (2026-11-06, local bf16, 13 C0/C1 pairs): **"DOES NOT HELP"** by its pre-stated rule.
- At a fixed λ, v4 gives circuits about 2.5× smaller, but 11 of 13 pairs lose more than 0.02 on some metric. Three PLT
  pairs lose 081's pass.
- Res rc at λ 1e-2 collapses: Z 0.028.
- On the size/faithfulness frontier (descriptive), the 8 points with a comparison split three ways:
  - 4 better (plt rc 1e-3, plt simple 1e-4 and 1e-3, res rc 1e-3);
  - 2 worse (plt nounpp 1e-3, plt rc 1e-4);
  - 2 on C0's curve.

So v4's budget mostly moves the circuits along the λ curve, toward smaller circuits. The 9-λ grid (3e-6 … 3e-2) still
spans the curve. But the largest λ values may now give vacuous circuits, and the matched-size comparison against Caples
and against AG shifts. Daniel's D7 stands; this is the known cost.

## Memory on the pod (fp32 google weights, H100 80 GB)

- v4 adds no tensors: the lr knobs only. R (the largest term) is already in v3.
- Local bf16 peaks of these very fits:
  - torch: res 9.8 GB, plt 10.0 GB;
  - nvidia-smi: 11.5-11.7 GB.
- fp32 roughly doubles that, to about 20-25 GB per job. The queue runs one job per GPU, so the pod has ample room.
- s/step is unchanged by v4. The local measurement was 0.37-0.50 s/step bf16.

## The check (GPU, local 16 GB card, ~15 min; NOT run)

Two tiny fits through the production path at λ 1e-3, 400 steps, bf16, unsloth weights:
1. **res**: gsres, Caples data + Caples mode per_position, with the local means file;
2. **plt**: PLT, SFC data, pooled.

Both are compared against lrtest's C1, which is the same recipe through a wrapper, and C0 (v3) units.

1. Wait until `.gpu-lock` at the repo root is absent (it held `gpu-44` at 02:50 on 2026-11-16). Then write the
   single line `arena-d-v4` into it, with the Write tool.
2. WSL, repo root:

   ```
   bash experiments/107-true-baselines/arena_d/v4_check.sh 2>&1 | tee experiments/107-true-baselines/arena_d/out/v4_check.log
   ```

   The script refuses (exit 3) unless `.gpu-lock` is exactly `arena-d-v4` and the card holds ≤ 3000 MiB. Its steps:
   - 0 = the CPU unit test;
   - 1 = res;
   - 2 = plt;
   - 3 = `v4_check_compare.py` (CPU).

   Outputs go to `arena_d/out/v4_check/{res,plt}/` plus logs.
3. When it has exited (pass or fail): `mcp__agent-gate__delete_path` on `<repo>/.gpu-lock`.

Time:
- unit test about 1 min;
- each family about 3 min to load, about 2.5-3.5 min to fit (146 s res, 188 s plt in lrtest), plus about 15 s to score;
- compare a few seconds.

**About 15 min in all, under 30.**

### Expected output (step 3)

```
res provenance OK: {'wcm': 'v4', 'lr_peak': 0.1575, 'warmup_steps': 40, 'weight_decay': 0.05, 'lr_sum': 21.3957, 'schedule': 'geometric', 'R': 1.0, 'rank': None}
  arm        n      Z      A      C    nec   t_fit  peak_MB
  v4      ~215  ~0.921 ~0.966 ~0.944 ~0.988   ~146    ~9845
  C1       215  0.921  0.966  0.944  0.988     146     9845
  C0       549  0.969  1.008  0.965  1.000     147     9837
  v4 vs C1: ... -> PASS (identical, if the card is deterministic here)
plt provenance OK: {... same provenance ...}
  v4      ~666  ~0.990 ~0.956 ~0.990 ~0.832   ~188    ~9983
  C1       666  0.990  0.956  0.990  0.832     188     9983
  C0      1713  0.951  0.984  1.013  1.081     199     9975
  v4 vs C1: ... -> PASS
v4_check_compare: PASS
```

The step 1 and 2 log lines (`... rc ... lam 0.001: N latents | task Z .. A .. C .. nec ..`) should show the same
numbers.

**Pass** means:
- the provenance is exact;
- the size is within 25% of C1's (C1 vs its R-seed replicate C1s2 is 215 vs 232);
- every metric is within 0.05 of C1's (C1 vs C1s2 differ by ≤ 0.008).

The compare script checks this already against lrtest's C1 dirs: the numbers match, and only the `wcm` label differs
(C1 = v3 + wrapper). A result near **C0** (549 / 1713 latents) instead means v4 did not reach the engine.

If it fails:
- **Provenance mismatch**: `ARENA_D_WCM` / `--wcm` is not reaching `make_engine`. Check `res.log` for the `WCM_VERSION`
  header in `out/v4_check/res/WCM_VERSION`.
- **Sizes off but provenance OK**: GPU nondeterminism beyond the tolerance. Rerun once into a fresh `V4C_OUT`.
- **OOM**: not expected (lrtest peaked at 11.7 GB nvidia-smi on this card).

## Result (RUN 2026-11-16 05:45-05:58, agent arena-d-v4, local RTX 5070 Ti, WSL): **PASS**

The full log is at `out/v4_check.log`, and the per-family logs and fits are in `out/v4_check/{unit.log,res,plt}`. The script exited 0, with 0 failed steps.

| step | time | outcome |
|---|---|---|
| 0 unit test (`-k v4_lr_budget`, CPU) | 05:45:36-05:47:57 | 1 passed, 23 deselected (133.8 s) |
| 1 res (gsres-gap6, Caples per_position, rc, λ 1e-3) | 05:47:57-05:52:43 (286 s) | 215 latents, Z 0.921 / A 0.966 / C 0.944 / nec 0.988; fit 145.8 s, score 9.4 s, peak 9837 MB |
| 2 plt (PLT, SFC pooled, rc, λ 1e-3) | 05:52:43-05:58:39 (356 s) | 666 latents, Z 0.990 / A 0.956 / C 0.990 / nec 0.832; fit 210.8 s, score 1.6 s, peak 9975 MB |
| 3 compare | 05:58:39-05:58:40 | `v4_check_compare: PASS` |

- **Provenance**, the same for both families and exact: wcm v4, lr_peak 0.1575, warm-up 40 steps, wd 0.05, lr_sum 21.3957, geometric schedule, R 1.0, no rank-keep.
- **vs lrtest C1**: identical in both families (size ratio 1.000, every |Δmetric| = 0.0).
- **vs C0 (v3)**: 549 / 1713 latents, so v4 clearly reached the engine.
- **Fit times**: the plt fit took 211 s here against C1's 188 s; the res fit took 146 s, the same as C1. This is timing only.
- The `.gpu-lock` was released at 05:58 via delete_path. Afterwards nvidia-smi showed 198 MiB used, no compute processes from this job, and no `wcm_gemma` process in WSL.
