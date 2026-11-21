# SMOKE_1120: local GPU smoke of the 2026-11-21 pod scripts (agent `dryrun`, 2026-11-20 00:51 to 02:27)

Daniel approved this run (relayed by the coordinator, from 00:50). Local RTX 5070 Ti 16 GB under WSL (~30 GB RAM, which
three other agents' CPU jobs shared). Job 1 of `out/gpu_queue_1120/`. `.gpu-lock` (`smoke-1120`) was taken at 00:51
and released at 02:27 with delete_path, and `1_smoke.done` was written. No network, no HF token read (everything
ran `HF_HUB_OFFLINE=1` from the local cache). No pod script was edited. Outputs: `out/smoke_1120/`. The CPU dry run
is in `DRYRUN_1119.md`. Its headline still stands: the documented launch line sets `AA_LAMBDAS` and drops the 1.6e-2 arm.

## 1. Verdict

| stage | verdict | evidence |
|---|---|---|
| (b) arena A, `run_pod.sh` P5 (first end-to-end run of the learned_mask.py donor trim) | **PASS** | 2 keys (3.tc.11202 shallow, 9.tc.21268 deep L9) x default lambdas `2.5e-4,1e-3,4e-3,1.6e-2` x scopes a1 + a2: **16 / 16 fits ok, 0 errors**; `WCM v4`, tuning `AA_TUNE_CONFIG=nec0.3 NEC_W=0.3`, `ORDER=interleaved` (`arena_a/wcm/*/main/status.shard0.jsonl`, `aa2.out`) |
| arena A host RAM per worker | **PASS** (just over the 3.5 GB budget) | peak RSS per driver_tcz process **3.24-3.60 GiB**, 9 processes (`rss_report_arena_a.md`). 15 on the pod ≈ 54 GiB (was ~27 GiB per deep fit before the trim) |
| arena A kill + resume | **PASS** | SIGTERM to run_pod.sh's process group at 01:11:43, during the a1_l1e-3 deep fit (the shallow one had finished): rc 143, no process left (ps). Same command again: a1_l2.5e-4 "every target fitted" (skipped); a1_l1e-3 refit only 9.tc.21268 (`2/2 9.tc.21268 ok`), one status row per key, no duplicate. fits_done's row-count bug (run_pod.sh:657-663) did **not** bite (1 ok row of 2 keys -> relaunched) |
| (a) chain T7 through chain.sh (one TuringLLM target 3.attn.11307 at v4, production 083 budgets) | **PASS** | `TL_TRACK=1 STOP_AFTER=S0 TL_STOP_AFTER=T7 TL_DAN159_STAGES="ref attr fitted exact"` on the scratch-restored state (R3 applied): T0-T6 skipped, T7 ref 13 rows, attr 78, fitted 6, exact 3, table rendered, **0 error rows**; T8 stopped by TL_STOP_AFTER; chain rc 0 (`t72.out`) |
| chain T7 kill + resume | **PASS, with a bug** | SIGTERM to the chain's process group during attr (12 of 78 rows). **The 083 worker survived** (own process group from chain.sh:434 `setsid`; seen in ps 7 s later, re-parented to init, 7.6 GiB RSS). I killed it by hand, then relaunched. The relaunch re-ran ref (3.3 min start-up, 0 new rows) and finished attr: 78 rows, **78 unique** (method x ranking x size_arm), no duplicates. See bug 1 |
| (c) arena D, one fit, DW_WCM v3 | **PASS (closest path, labelled)** | `pod_stage_d.sh` itself cannot run here: P0 needs `google/gemma-2-2b` fp32 and the PLT transcoders, neither cached, and HF is offline. fp32 Gemma-2-2B + bank + grads also does not fit 16 GB (arena_d README:313: ~15.5 GiB for Caples alone). So: P3 job `wcm_d2_native_pp`'s exact command and BASE env (`WCM_VERSION=v2 ARENA_D_WCM=v3` via the env), but **`GEMMA_HF_ID=unsloth/gemma-2-2b` (a local copy, not Google's), bf16**, 1 lambda (1e-3). rc 0, `WCM_VERSION` file `arena_d_wcm: v3`, lr_peak 0.05 (v3, not 0.1575): rc 1,350 latents, Z 0.989 / A 1.070 / C 1.068 / nec 1.090 |
| archive check | **PASS** | the chain's own `archive` + `copy_verified` packed `pub-turingllm-T7-dan159.tgz` (53 entries, 4.1 MB) into the run's RESULTS, md5 verified. `md5sum -c` again, unpacked into empty `unpack/`: per-file md5 identical to the live dir. `t7_summary.py` reads the unpacked dir (`T7_SUMMARY_from_unpack.md`), and table2.md / dan77 figure are inside |

## 2. Measurements (local card; H100 numbers in section 4)

Arena A WCM fits (`AA_STEPS=200`, `AA_WCM_P=1`, `DEEP_BS=1`: local-card knobs; the pod runs 400 steps, P=3, DEEP_BS 4):

| arm | 3.tc.11202 fit s (n) | 9.tc.21268 fit s (n) | arm wall (incl. ~2 min model / bank / store set-up per launch) |
|---|---|---|---|
| a1_l2.5e-4 | 155 (673) | 310 (1,921) | 11.0 min |
| a1_l1e-3 | 135 (203) | 197 (1,044), refit after the kill | 5.8 + 5.4 min |
| a1_l4e-3 | 80 (62) | 173 (359) | 6.6 min |
| a1_l1.6e-2 | 77 (17) | 166 (118) | 6.6 min |
| a2_l2.5e-4 | 65 (442) | 126 (891) | 5.4 min |
| a2_l1e-3 | 64 (168) | 118 (456) | 5.3 min |
| a2_l4e-3 | 56 (53) | 105 (29) | 4.9 min |
| a2_l1.6e-2 | 56 (23) | 105 (9) | 4.9 min |

- Eval: 4-15 s per fit. P5 total: 39 min, 01:11-01:51 after the kill.
- Peak VRAM: 7.4 GiB, whole card, desktop ~0.9 GB included.
- Host RAM peak per worker: 3.6 GiB.
- No GIVING UP and no error rows.

Chain T7, one shallow L3 target, 1 process (`TL_083_P=1`; the pod runs 4 per GPU):
- Stage wall times:
  - ref 4.3 min;
  - attr 3.7 min to the kill, then 5.0 min;
  - fitted 5.9 min;
  - exact 5.4 min;
  - table 0.6 min.
- Each 083 stage spends ~3 min starting up (model, SAEs, stores) before its first row, so work per target ≈ 80 + 180 +
  180 + 140 s ≈ 580 s. The plan figure is 750 s plus exact.
- Peak VRAM 8.1 GiB (card).
- Worker RSS: **7.6 GiB** (ps at 01:59 for the attr worker). The sampler missed the T7 workers (their interpreter
  path has a space, which broke its awk match). MemAvailable during T7 also dropped by ~8 GB. This matches exp 128's
  ~8 GB.

Arena D, one D2-WCM fit (bf16 local unsloth, 400 steps):
- wall 281 s: model load ~2 min, fit 143 s, score 8 s;
- peak VRAM 7.4 GiB (card); host RSS 7.6 GiB.
- Tonight's lrcheck_1119 v3 fits, same path, agree: res rc 158 s, PLT within_rc 200-205 s, peak 5.9 / 9.0 GB.

Host contention (not a pod issue): other agents' CPU jobs (exp 131 at 16 GiB, exp 135 at 12-15 GiB x 2, exp 134 at
5 GiB) shared the 30 GB WSL. MemAvailable fell to **11 MB at 01:01:44**, and the kernel OOM-killed another agent's
python process (pid 7282, 15.8 GB anon; `dmesg`). This run's fits were not hit. The queue protocol locks only the GPU,
not host RAM.

## 3. Bugs (file:line + proposed fix, none applied)

1. **The chain leaves job workers running when it is killed** (new, seen tonight).
   - `experiments/082-public-runs/chain.sh:434` starts every fitting job with `setsid "$@" &`, its own process
     group, so the job_errors watcher can stop it. chain.sh has no signal trap.
   - So SIGTERM / SIGINT / SIGHUP to the chain kills the chain, but the 083 / launch.sh / driver processes run on.
     This covers Ctrl-C in tmux, a killed tmux pane or `kill`.
   - Relaunching "the same command" (the documented recovery) then starts a second copy beside them, with 2x VRAM and
     2x host RAM. On a 125 GB host during T7 that is enough to OOM.
   - Proposed fix: in `run_job` (after :435), write the pid to `$STATE/jobs/$job.pgid` and remove it after :439. Add a
     top-level `trap 'for f in "$STATE"/jobs/*.pgid; do kill -TERM -- -"$(cat "$f")"; done; exit 143' TERM INT HUP`
     next to :1315. At start, refuse to run if a recorded pgid is still alive.
   - Saturday workaround (no edit): before any relaunch, check
     `pgrep -af '083-v1-comparison|launch.sh|driver'` and `nvidia-smi`, and kill leftovers first.
2. **T7 RAM.** Measured T7 worker RSS is 7.6 GiB. chain.sh:221 `TL_083_P` default 4 x `FIT_GPUS=1,2,3,4` = 16
   workers ≈ 122 GiB, on a ~125 GB host. The standalone arena A P2 data pass runs at the same time (option B), so
   this **will not fit**. Fix, env only, on the launch line: `TL_083_P=2` (≈ 61 GiB) or 3 (≈ 91 GiB). DRYRUN_1119
   bug 4 is now measured, not estimated.
3. **Resume re-pays each finished 083 stage's start-up.** The relaunch re-ran T7 ref, finding 0 new rows after 3.3
   min. T7 has no per-sub-stage markers (chain.sh:1103-1105). Proposed: `mark turingllm.T7.$s done` after each
   `t7_stage_body` and skip it on rerun. Minor: ~3 min per stage per resume.
4. **run_pod.sh is not space-safe** (local only). Its `run_cmd` strings (run_pod.sh:299-323, `env $BASEENV ...`
   unquoted) split any path with a space. My first attempt with an absolute scratch AA_WORK failed with rc 127. Pod
   paths have no spaces, so this is harmless there. The smoke used repo-relative paths, like the pod's default AA_WORK.
5. Confirmed from DRYRUN_1119: the fake `turingllm.T7.done` arrives with the restore. B3/R3's `mv` is mandatory, as is
   dropping `AA_LAMBDAS` from B4.

## 4. Saturday estimate (5x H100, full target lists)

**Speed factor assumed: one H100 process ≈ 2x one local RTX 5070 Ti process** (range 1.5-3x). This is the factor the
plan already uses (083 README: 40 local-h ≈ 20 H100 GPU-h). These fits are small-batch and latency-bound (Dunefsky
~24% GPU per process; WCM fits at 64 tokens). An H100's advantage therefore shows mostly as more co-tenants per GPU
(P=3-4), not as a faster single process. Not measured tonight. A 2.5 h first-hour check on the pod (fits per hour
from `steps.tsv` / rows) replaces it.

| stage | local work (from tonight) | H100 GPU-h | wall on the pod |
|---|---|---|---|
| setup: quickstart, restore, venvs, arena gates | - | - | 1.5-2 h (plan) |
| T7, 190 targets, ref attr fitted exact + table | 750-900 s / target (tonight 580 s on a shallow target; deep ones cost more) -> 40-48 local-h | 20-24 | 5-6 h on GPUs 1-4 at TL_083_P=4. At **TL_083_P=2** (bug 2), assume ~6-8 h. Arena A P0-P3 runs on GPU 0 meanwhile (option B) |
| arena A P5 WCM: a1 48 + a2 ~22 targets x 4 lambdas, 400 steps | 400-step fit ≈ 1.8x tonight's 200-step fit (probe r3: deep a2 203 s at 400): a1 ~335 s, a2 ~185 s mean -> 4 x (48 x 335 + 22 x 185) s ≈ 22 local-h | ~11 | ~2-3 h on 5 GPUs at P=3 |
| arena A P4 / P4x / P4h / P6 / P6x / P6b (Ge, Dunefsky, scoring) | not run on a GPU tonight | 35-47 (plan's 46-58 total less P5) | ~8-11 h. **Arena A total ~10-14 h; hard cap ARENA_A_MAX_H=16** |
| arena D tasks: 27 D2-WCM + 72 D1-WCM fits + Caples + AG | ~150 s / D2 fit, ~200 s / PLT fit (bf16 local). fp32 on the pod ≈ 2x the work, and the H100 ≈ 2x faster, so ≈ the same | 5 (WCM) + 2-3 (Caples, AG) ≈ 7-9 | ~2-3 h + gate 0.5-1 h. DRYRUN_1119 bug 5: 5 fp32 loads at once |
| **total** | | **~75-95** | **~21-28 h** |

Cost: the pod bills ~$20/h whether its GPUs are busy or idle, so cost = wall x $20 ≈ **$420-560**. The ARENA_A_MAX_H=16
cap bounds the worst case at ≈ 2 + 8 + 16 + 4 = 30 h ≈ $600. Idle GPU-time sits inside this: T7 uses 4 of 5 GPUs,
arena A P0-P3 runs on GPU 0 only, and the arena D gate uses 2. D4 now stops the pod at the end (DRYRUN_1119:
simulated, PASS), so no idle tail is paid after arena D packs.

## 5. What ran

- `out/smoke_1120/run_smoke.sh` (GPU, queue-locked, 03:15 / 03:27 deadlines):
  - stage A = `run_pod.sh` with `AA_GPUS=0 AA_WORK=<smoke>/arena_a AA_KEYS=3.tc.11202,9.tc.21268 AA_FROM=P5 STOP_AT=P5`
    on exp 126's owt131k corpus / stores / targets, `AA_WCM_P=1 DEEP_BS=1 AA_STEPS=200 AA_MAX_H=0`;
  - lambdas, scopes and CHOSEN.env at their defaults;
  - kill test, then the same command again.
- Stage T = `chain.sh` with the B4-style env above: `FIT_GPUS=0`, both `DATA_GPU_*=0`, `TL_083_P=1`, `MPS=0`,
  `STOP_POD=0`, `TL_ROOT=` this repo, and `TL_V2_OUT=` a copy of exp 128's production v4 circuits for 3.attn.11307.
  `TL_CTX_FRESH=` the T2 archive's records (DRYRUN_1119 restore), `TL_LISTS=` a one-target sweep190 list. Kill test,
  then the same command again.
- Two earlier attempts failed in under 2 min on a space in my scratch path (rc 127 / rc 1, no fit started). Their
  dirs were removed with delete_path, and an orphaned 083 process group of that attempt was killed.
- `out/smoke_1120/ad_fit.sh`: the arena D closest-path fit (section 1).
- `out/smoke_1120/rss_report.py` -> `rss_report_arena_a.md` / `rss_report.md` (3 s samples: `rss.csv`, `sys.csv`).
  Archive check: `md5sum -c`, `tar -xzf` into `out/smoke_1120/unpack/`, a per-file md5 diff, and `t7_summary.py`.
- Not done: the Ge / Dunefsky baselines on a GPU (P4 / P4h); arena A P6 scoring; arena A P0-P3 (the 524k-window data
  pass is still unmeasured); an fp32 Google Gemma fit; pod_stage_d.sh's own P0-P2 gate. All are blocked by the local
  16 GB card / cache, or were left out to keep within the 03:15 deadline.
