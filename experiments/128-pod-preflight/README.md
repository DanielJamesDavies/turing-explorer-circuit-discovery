# 128: pod preflight (local night 2026-11-17 -> 11-18)

Purpose: before the 2026-11-21 pod (5x H100, ~125 GB host RAM), exercise today's `run_pod.sh` changes for real at
small scale, time Ge on a GPU, get WCM a1 numbers on the exp 126 OWT corpus, and find out what the host RAM of a deep
WCM fit is used for. Local RTX 5070 Ti (16 GB), WSL (~30 GB RAM), ONE GPU process at a time.

Built and dry-run tested 2026-11-17 (agent `night1117`). **Not launched.**

## How to run (WSL, repo root)

```
DRY_RUN=1 STOP_EPOCH=$(date -d 'tomorrow 07:45' +%s) bash experiments/128-pod-preflight/run_night.sh   # the plan
mkdir -p experiments/128-pod-preflight/out
STOP_EPOCH=$(date -d 'tomorrow 07:45' +%s) N128_SOFT=ge_gpu,a1_fit_deep nohup bash experiments/128-pod-preflight/run_night.sh \
    > experiments/128-pod-preflight/out/run_night.out 2>&1 &
```

`N128_SOFT` is optional; see "Watchdog" below. Any evening launch fits: the plan is ~5.2 h, t7 starts only before
05:30, and every job is killed by 08:15 at the latest.

## Stages (in order; per-stage estimates from measured 107 / 126 / 083 timings)

| stage | what | est. min |
|---|---|---|
| smoke | `arena_a/run_smoke_local.sh` against the new `run_pod.sh`, fresh `SMOKE_WORK=arena_a/out/smoke_pod_1117`, `AA_FROM=P4`, keys 3.tc.13057, 3.z.23269, **1.tc.23034** (a 2nd tc key so `AA_DF_PAR=3` runs 2 Dunefsky processes with work: run_dunefsky.py shards by target), `SMOKE_LAMBDAS=1e-3,1.6e-2`, `AA_WCM_P=1`; P4 P5 P6 P4x P6x P4h P6b P7. Baselines on the CPU | 85 |
| smoke_check | `check_smoke.py` -> `out/smoke_check.md` (CPU, every run): last rc of every stage 0; ok WCM rows for a1/a2 x 2 lambda; "order interleaved" in the driver log; P4x circuits `p0.b100.i5.pt` for both tc keys; P6x score rows of the b100 units + `logs/P6x`; `score/all/ge_recovery.x.s0.jsonl` >= 1 row, 0 errors; >= 2 `.df<s>` logs with work in P4 and P4x; both summary.md; `a2_skipped_L0.txt` | 1 |
| ge_gpu | `run_ge.py` (unchanged) in `~/venvs/ge107gpu`, `--device cuda`, 3.tc.13057 / 7.z2.18144 / 9.tc.17800, K 2, 30 tau (= the 11-05 CPU smoke `out/ge_smoke_up`), one process per target; `ge_gpu_report.py` -> `out/ge_gpu.md` (seconds per circuit GPU vs CPU, sizes over the tau grid, node-set mismatches) | 8 |
| a1_fit_shallow | WCM **a1** lambda 1e-3, CHOSEN.env, `run_pod.sh` P5 only, on 126's 8 owt131k shallow tc targets (1.tc.17037 1.tc.14335 2.tc.15605 2.tc.3416 3.tc.11202 3.tc.9976 4.tc.6575 4.tc.16695) | 33 |
| a1_score_shallow | `score_arena_a.py --scope a1 --selftest` | 8 |
| a1_fit_deep | the same on 126's 8 deep targets whose a2 fit finished (5.tc.1481 5.tc.8088 6.tc.9000 6.tc.24021 7.tc.3375 7.tc.15758 8.tc.22944 8.tc.14821), **one run_pod.sh call per key, ascending depth** (`AA_REDO=P5`; run_pod.sh's own `ORDER=interleaved` would put an L8 fit second) | 70 |
| a1_score_deep | score the deep keys whose fit finished | 8 |
| a1_compare | `compare_a1_a2.py` -> `out/a1_vs_a2.md` (a1 here vs 126's a2 rows, same targets) | 1 |
| ram_l1e-3, ram_l2.5e-4, ram_s100 | one deep a2 fit each on **9.tc.21268** (the first L9 fit, the one that tripped 126's watchdog at 03:28), lambda 1e-3, 2.5e-4, then 1e-3 at `STEPS=100`; `DEEP_BS=1` as in 126. See "RAM probe" | 25 + 25 + 12 |
| ram_report | `ram_report.py` -> `out/ram_report.md` | 1 |
| t7_full | 083 suite = chain T7 (`ref attr fitted`, then `table`) at production settings on one TuringLLM target, 3.attn.11307. See "T7" | 35 |
| t7_summary | `t7_summary.py` -> `out/t7_full/T7_SUMMARY.md` | 1 |

Total ~313 min (~5.2 h). Basis: 11-06 2-target smoke 33 min at 1 lambda (P5 685 s for 3 fits, P6 634 s); 126 a2 fits
2.5 min shallow, 3-5.4 min L5-8 (a1 has about twice the candidate sites: x1.5 guessed); 126's L9 fit ran > 10 min
before the kill; 083 README 750 s per target + ~3 min start-up per stage.

## Safety and control (same conventions as 126 / 127)

- Refuses to start if `.gpu-lock` exists, `out/ABORTED` exists, or the GPU holds > `N128_GPU_IDLE_MIB` [2500] MiB (the
  desktop holds ~1.5 GB). Writes `.gpu-lock` ("pid $$"), removed at exit only if it still names its pid.
- **Watchdog** every `N128_WATCH_S` [10] s (`out/watch.csv`): GPU > 15,300 MiB or MemAvailable < 3 GB kills the job's
  process group, writes `out/ABORTED`, and stops the night (exit 9). Exception: stages listed in `N128_SOFT` (comma
  list, default empty), where only that job is killed and the night goes on. The ram_* stages always behave so (their
  own monitor). **Recommended: `N128_SOFT=ge_gpu,a1_fit_deep`.** 126's watch.csv shows the a2 L7 fits already down to
  6.4 GB MemAvailable and the first L8/L9 at 3.9 GB; a1 has more sites, so an L7/L8 a1 fit may trip 3 GB. Under the
  strict default that trip would end the night before ram_probe and t7. With it soft, the deep loop stops at the first
  trip (deeper keys are not started), the fitted keys are scored, and the night goes on.
- `STOP_EPOCH` (set to 07:45): no new stage / key after it; passed to `run_pod.sh` (soft stop; `AA_HARD_GRACE=1800`)
  and the scorers. Every job is killed at `STOP_EPOCH + N128_HARD_GRACE` [1800 s] = 08:15. t7 starts only before
  `N128_T7_LATEST_EPOCH` [STOP_EPOCH - 2 h 15 = 05:30] and is killed at `N128_T7_KILL_EPOCH` [STOP_EPOCH + 15 min = 08:00].
- Resumable: `out/state/<stage>.done`; inside stages run_pod.sh / the scorers skip finished work. `N128_STAGES` /
  `N128_SKIP` select stages. Logs: `out/night.log`, `out/logs/<job>.log`.
- Exit 0 all done; 1 a stage failed; 7 cut by STOP_EPOCH; 8 refused; 9 watchdog abort.

## Variable names (run_smoke_local.sh vs today's run_pod.sh)

All match: `AA_DUNEFSKY_EXT`, `AA_DUNEFSKY_EXT_K`, `AA_DUNEFSKY_EXT_BRANCHES`, `AA_DUNEFSKY_K`, `AA_DUNEFSKY_BRANCHES`,
`AA_GE_K`, `AA_GE_NTAU`, `AA_HELD_K`, `AA_PP_*`, `AA_DONOR_RECORDS`, `AA_ALLOW_UNTUNED` are read under these names.
`AA_WCM_P` / `AA_DF_PAR` are not set by the smoke script, so the runner passes them through the environment. The dry
run greps run_pod.sh's printed plan and confirms: P4x = `--k 1 --branches 100 --iters 5 --shard 0/3`; P6x writes
`score/all/ge_recovery.x.s0.jsonl`; P5 `ORDER=interleaved P=1`. run_pod.sh was not edited. Note: CHOSEN.env exists, so
the smoke fits are tuned (NEC_W 0.3), not plain v4 (AA_ALLOW_UNTUNED only applies without the file).

## Ge on the GPU (`~/venvs/ge107gpu`)

Buildable offline, and **built** (2026-11-17 16:39, 7.1 GB) by `build_ge107gpu.sh`: a copy of `ge107` (Python 3.10.19,
torch 2.3.1+cpu) whose torch / torchgen / triton / nvidia libs (+ dist-info) come from `~/venvs/tc107gpu` (also Python
3.10.19, torch 2.8.0+cu128, downloaded 11-16 for the Dunefsky GPU test). No pip, no network. Import check: torch
2.8.0+cu128, transformers 4.43.1, numpy 1.26.4. CPU test (`out/cpu_test_ge107gpu`): `run_ge.py` on 3.tc.13057 p0
(3 tau) gives the CPU smoke's a_t 6.968152, G1 gap 3.9e-7 and n_hier[0] 951. The copy's `bin/activate` and
`bin/ge_env.sh` still name ge107: the runner does not source them (it sets PATH / VIRTUAL_ENV and calls
`ge107gpu/bin/python -I` directly, with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`). The pod's ge107 is unchanged (torch
2.3.1+cu121, H100 = sm_90). `pip check` in the copy would flag torch 2.8 against Ge's pins; timing only.

## RAM probe (what the host RAM of a deep fit is for)

`probe_driver.py` runs `driver_tcz.py` -> 062 `driver.py` unchanged, with markers: it wraps
`observability.phases.phase` before `learned_mask.py` imports it, so every optimizer step (`fit.opt`) is counted, and
`driver.Runner.__init__ / contexts / fit / score` with enter / exit markers. Every 5th step and each marker it writes
`status.jsonl`: VmRSS / VmHWM / VmSwap, torch `memory_allocated` / `memory_reserved` / `max_memory_reserved`, the
device total. At set-up end, context load, fit end and steps 1 / 50 / 200 it scans the live CPU tensors (total,
pinned, the 6 largest). The driver itself logs nothing per step (no `logger` reaches the mask engine), hence the hook.

`ram_monitor.py` samples every 2 s (`mon.csv`): process-tree RSS / HWM, RssAnon / RssFile / RssShmem of the largest
process, /proc/meminfo (MemAvailable, Cached, Shmem, AnonPages, Mapped, Unevictable, Mlocked, swap, Committed_AS),
nvidia-smi used / total and per-process (often empty on WSL), plus the driver's last phase / step / CUDA numbers. It
kills its fit at MemAvailable < 3 GB, GPU > 15,300 MiB or after `N128_PROBE_MAX_MIN` [35] min, then the next probe runs.

`ram_report.py` gives per probe RSS and MemAvailable at the end of set-up, after the context load, at steps 1 / 50 /
100 / 200 / 400 and the peaks; "outside-tree" memory (MemAvailable drop minus the tree's RSS growth: what the kernel /
driver holds for no process); the RSS slope per step; and a verdict per hypothesis: **H1** sysmem fallback (torch
reserved more than the card holds, or nvidia-smi full with outside-tree growth), **H2** growth with steps (RSS step 1
-> last step > 2 GB and > 30 % of peak), **H3** up-front (RSS at fit step 1 > 50 % of the peak). The s100 probe
separates the two: same peak at 100 steps = up-front.

Already in 126's `out/watch.csv` (before tonight): during the L5-7 a2 fits MemAvailable fell from 29 GB to 11.7 / 6.4
GB while nvidia-smi showed only 9.4-11.4 of 16 GB used, so the card was not full when the host RAM went: H1 (spill
because the card is full) looks unlikely a priori; tonight's numbers decide.

`DEEP_BS=1` in the probes reproduces 126's conditions (every local fit uses it; the pod runs the config's 4).
`N128_PROBE_DEEP_BS=` (empty) would probe the pod setting, which on 16 GB is expected to spill (driver.py comment).

## T7 (one TuringLLM target at production settings)

Possible without touching production scripts. The 11-16 re-smoke ran the whole chain (`LIMIT=1 ... TL_DAN159_STAGES=
'ref attr fitted exact'`, `WORK=082/smoke_v4_t7`) with smoke budgets (`STEPS=4`, `IG_STEPS=2`, a cut size list),
whose own STEPS=20 fits did not sparsify. Here instead `t7_prep.py` copies the target's **production v4 circuits**
from the 11-07 pod archives (`109-pod-run-analysis/out/archives`: T4 sweep circuits for all 13 arms + their eval
rows, T3 main circuit, the `WCM_VERSION` v4 / `WCM_R` markers) into `out/t7_full/ref` (15.8 MB; done at build time,
production W_0.001 n = 510), and the runner calls `083/run_pod.sh` (unchanged) per stage exactly as chain.sh's
`t7_stage_body` does (`REF_RUN=OUT=<ref>`, `OUT083=out/t7_full/dan159`, `CTX_DIR=` the T2 archive's ctx_fresh, read
only), with `GPUS=0 P=1` and none of the smoke budget overrides. Target 3.attn.11307 (the re-smoke's, in sweep190).
Reports: per-stage wall time, rows, sizes and Z / A / C / nec (`T7_SUMMARY.md`) and 083's own `table2.md`.

## Files

`run_night.sh` (the runner), `build_ge107gpu.sh`, `probe_driver.py`, `ram_monitor.py`, `ram_report.py`,
`check_smoke.py`, `compare_a1_a2.py`, `ge_gpu_report.py`, `t7_prep.py`, `t7_summary.py`. Outputs in `out/`.
CPU tests (build day): `out/cpu_test_ge107gpu/` (Ge in the new env), `out/cpu_test_probe/` (probe_driver +
ram_monitor + ram_report on a 6-step CPU fit), `out/cpu_tests/` (check_smoke on the 11-06 smoke dir, compare,
t7_summary on the 11-16 re-smoke output, ge_gpu_report).
