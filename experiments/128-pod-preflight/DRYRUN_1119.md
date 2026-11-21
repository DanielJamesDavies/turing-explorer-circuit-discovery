# DRYRUN_1119: CPU dry run of the 2026-11-21 pod launch (agent `dryrun`, 2026-11-19 23:30 to 11-20 00:50)

CPU only (`CUDA_VISIBLE_DEVICES=""`), no network, no pod script edited. Scratch output: `out/dryrun_1119/`.
"Documented launch line" = pod plan `agent-planning/2026-11-08-post-paris/01-next-pod-plan.md` section 8.7
(B3 restore + markers, B4 chain launch, B5 standalone arena A P0-P3 and arena D P0-P1). That is the decided Saturday
sequence (option B, D8). `scripts/pod_quickstart.sh` (a8b3a2f) prints an older launch line; see bug 2.

## 1. Verdict

| check | verdict | evidence |
|---|---|---|
| chain + T7 at v4 | **PASS** | `chain_dry_corrected.log`: header `WCM_VERSION=v4`; all 5 T7 083 commands (`ref attr fitted exact table`) carry `WCM_VERSION=v4`, `GPUS=1,2,3,4 P=4`; restored `out_v2/WCM_VERSION` = `v4` (`setup_restore.log`) |
| arena A at v4 | **PASS** | `aa_dry_full.log`: header `WCM v4`; 41 commands `WCM_VERSION=v4` (run_pod.sh:212 `AA_WCM_VERSION` default, independent of the chain's export) |
| arena D at v3 | **PASS** (push caveat, section 2) | `ad_dry_g0.log` / `ad_dry_5.log`: header `WCM v3`; 45 commands `ARENA_D_WCM=v3` (pod_stage_d.sh:107) |
| launch line: AA_LAMBDAS unset | **FAIL (documented line)** | plan 8.7 B4 (01-next-pod-plan.md:376, also R5 :85) exports `AA_LAMBDAS=2.5e-4,1e-3,4e-3`. Its dry run (`chain_dry_b4_documented.log`) prints `lambdas 2.5e-4,1e-3,4e-3`: the 1.6e-2 arm is lost. Without it (`chain_dry_corrected.log`): `2.5e-4,1e-3,4e-3,1.6e-2` |
| arena A defaults take effect (line corrected) | **PASS** | `aa_dry_full.log`: 8 WCM launches (a1, a2 x 4 lambdas), each `GPUS=0,1,2,3,4 P=3 ... ORDER=interleaved`; Dunefsky `--shard s/3` (15 lines); a2 list = `grep -F .tc. ... \| grep -v '^0\.'` (L0 dropped, `a2_skipped_L0.txt`); the driver_tcz.py slot rotation is in the working copy (driver_tcz.py:51-66) |
| AA_TUNE_ENV / AA_ALLOW_UNTUNED | **PASS** | `out/tune/CHOSEN.env` is tracked (a8b3a2f), so the pod clone has it: `WCM tuning choice: AA_TUNE_CONFIG=nec0.3 NEC_W=0.3` in every P5 command. AA_ALLOW_UNTUNED is not reached. If the file were missing, a real P5 refuses (rc 1, run_pod.sh:670-673); only a dry run continues as untuned |
| D4 auto-stop (planned option B) | **PASS** (simulated) | `sim_d4.out`, case `optionB_planned`: real chain.sh run on a copy of the restored state with T7 / arena_a / arena_d marked done, stub `runpodctl`: `S10 not allowed ... no public stage started in this run (D4)` then the stub is called once (`stop pod sim-pod`), rc 0 |
| D4, arenas-only (TL_TRACK=0) | **PASS** (simulated) | case `arenas_only`: same D4 line, stub called once |
| D4 negative controls | **PASS** | `negative_public_start` (a future-dated `pythia-70m.S1.start`): "the pod stays up", 0 calls; `stop_pod_off` (STOP_POD=0): 0 calls |
| D4 with a real arena failure / rc 2 | **NOT RUN** (code read only) | chain.sh:1383-1390: any arena rc != 0 (including rc 2 "waiting for approval" and rc 4 time limit) packs logs and **stops the pod**. By design (D9), but an arena A rc 2 at P0/P1 stops the pod after T7 with arena A unrun |
| RAM, arena A P5 (15 WCM fits) | **PASS** (local probe, DEEP_BS=1) | probe r3 deep 9.tc.21268 (post-trim): RSS 3.20 GiB flat over 400 steps (`out/ram_report.md`); post-trim refit 5.tc.8088 used ~2.9 GB (`out/watch.csv`, a1_trim_identity). 15 x 3.2 = 48 GiB, nothing else runs during arena A. The pod's DEEP_BS=4 (config) is unmeasured |
| RAM, T7 phase | **RISK, likely FAIL** | T7 runs 16 083 processes (4 GPUs x TL_083_P 4). One process locally took ~8 GB (MemAvailable 29.4 -> 21.3 GB in t7_ref / attr / fitted, `out/watch.csv`). 16 x 8 ≈ 130 GB > 125 GB, plus the standalone arena A P2 data pass (unmeasured) on GPU 0 at the same time |
| RAM, arena D | **RISK, unmeasured** | P3 starts one job per GPU at once (pod_stage_d.sh:278, no stagger). arena_d README:313: Caples' fp32 TL conversion peaks at 25-36 GB host. 5 at once could reach 125-180 GB |
| every DRY_RUN / preflight mode, CPU | **PASS** | 11 dry runs rc 0 (`run_dry.out`): chain x4, run_pod.sh x3 (full, STOP_AT=P1, B5 STOP_AT=P3), pod_stage_d.sh x3 (DW_GPUS=0, 0-4, DW_STOP_AFTER=P1), pod_restore.sh DRY (12 archives, md5 checked, stores skipped). `bash -n` and no CR on 8 scripts. AA_HARDSTOP_TEST not rerun (README: passed 11-17) |
| restore + R3 markers | **PASS, and R3 is mandatory** | real pod_restore.sh of the 4 small archives into scratch: restored markers include the fake `turingllm.T7.done` + `.note` and no T3-T6 / T8 / T10-T12 done. Without B3's `mv ...T7.done ...skipped-2026-11-07` T7 is silently skipped; without the T3..T12 markers T3 re-runs. After R3 the dry run shows T0-T6 skipped, **T7 START**, T8 stopped by TL_STOP_AFTER |
| CPU tests | **PASS** | full `tests/`: **1947 passed, 3 skipped**, 12.5 min (the brief's "666" was an older / narrower count); `tests/circuit/instrument`: 299 passed; arena D `test_wcm_gemma.py -m 'not gpu'`: 21 passed |

## 2. Files the pod needs that are not in HEAD, against Daniel's push list

On the push list and needed (all confirmed modified / untracked): `src/circuit/instrument/learned_mask.py` (the donor
trim: without it a deep a2 fit is ~27 GB, not 3.2), `tests/circuit/instrument/test_donor_code_trim.py`,
`082-public-runs/chain.sh` (D4), `arena_a/run_pod.sh` + `driver_tcz.py` + README, `arena_d/pod_stage_d.sh`.
On the list, not on the pod path: experiments 126-134, `lrcheck_1119.sh`, `lrcheck_compare.py`, `paper/main.tex`,
the arena D / 081 READMEs.

**Not on the list, but the pod path reads them:**
- `experiments/107-true-baselines/arena_d/wcm_gemma.py` (M) and `test_wcm_gemma.py` (M). P2 runs
  `pytest test_wcm_gemma.py -m gpu` and P3 runs `wcm_gemma.py run`. HEAD's wcm_gemma.py is plain v3 and ignores
  `ARENA_D_WCM`, so pushing only pod_stage_d.sh still fits v3. But the rows would lack `wcm_version` / `lr_peak`, and
  the version guard would not exist. **Push both or neither.** If test_wcm_gemma.py goes without wcm_gemma.py, the P2
  gate tests new code against old.

**Not needed by the pod** (fine to leave or push): 081 `engine.py`, `run.py`, `task_common.py`, `config_v4_*.json`,
`queue_v4.sh`, `test_081_v4_cpu.py`. Arena D imports 081's Engine under `WCM_VERSION=v2`, where the 081 changes are
inert: the lr multiplier and wd are keyed on v4; `cf_train` defaults False; arena D overrides `engine_kwargs` /
`resample_tokens`. Also not used: arena_d `V4_CHECK.md`, `v4_check.sh`, `v4_check_compare.py` (no pod script references them).

**Modified but probably unintended:** `.gpu-lock` is tracked in HEAD (115a563 committed it, content `gpu-44`) and
deleted in the working copy. No pod script reads it. Commit the deletion and add `/.gpu-lock` to `.gitignore`. Also
modified, not pod-relevant: `paper/build/main.pdf`, `paper/drafts/framework_section.tex`, `083/README.md`,
`118/README.md` / `RESUME.md`. Untracked, not pod-relevant: experiments 124 / 125, `paper/supplement/`.

**Needed on the pod but not from git:** `arena_d/out/fidelity/gemma-fact-dallas-austin.json` (git-ignored `out/`;
fetched by arena D `setup_env.sh FETCH=1`, B2). On the volume, check for `pub-save-*.tgz` before B3: pod_restore
unpacks them last, so their `logs/state` would win. No local copy exists to test that case.

## 3. Bugs and doc errors (none applied)

1. **Documented launch sets AA_LAMBDAS** (headline). `agent-planning/2026-11-08-post-paris/01-next-pod-plan.md:376`
   (B4) and `:85` (R5): drop `AA_LAMBDAS=2.5e-4,1e-3,4e-3` from the export. Corrected B4 export:
   `HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 TL_TRACK=1 TL_ROOT=/root/turing-tl TL_STOP_AFTER=T7
   TL_DAN159_STAGES="ref attr fitted exact" STOP_AFTER=S0 FIT_GPUS=1,2,3,4 DATA_GPU_pythia_70m=1 DATA_GPU_gpt2_small=1
   ARENA_A=1 ARENA_A_GPUS=0,1,2,3,4 ARENA_A_MAX_H=16 AA_DUNEFSKY_EXT=1 ARENA_D=1 ARENA_D_GPUS=0,1,2,3,4 STOP_POD=1`.
   Before launching, check: `env | grep -E '^AA_(LAMBDAS|LAMS)='` prints nothing.
2. **The quickstart's printed launch line is the 11-07 one.** `scripts/pod_quickstart.sh:151,153`: no STOP_AFTER, no
   TL_STOP_AFTER, no arenas. On the restored state its dry run (`chain_dry_quickstart_line.log`) shows: T7 on GPUs
   `2,3,4,0,1`, **S10: START** (S10.done is not in the archives), then tar + `cp ... /workspace/results/` of
   `pub-*-S10-results.tgz`. With `RESTORE_SKIP=S10-stores` the stores archive fails ("nothing to pack"). The
   results tgz would still be re-made and copied over the volume's 11-07 copy. Fix: print the 8.7 B4 line, or a
   note "11-21 pod: use the pod plan 8.7 line", at :150-155. Saturday: do not copy the quickstart's line.
3. **Plan 8.7 is stale on arena D and D4.** `01-next-pod-plan.md:386` ("`arena_d: START` with `WCM v4`") and :293 /
   :451: the dry run now prints `WCM v3`, which is correct. :430-431 ("STOP_POD fires only once D4 lands") is
   outdated: D4 is in chain.sh. Update the grep in :378 (`WCM v4` -> `WCM v3`).
4. **T7 RAM.** chain.sh:221 `TL_083_P` default 4. Env-only fix on the launch line: `TL_083_P=2` (8 processes, about
   65 GB) or 3 (12, about 100 GB). Or check `free -g` on the pod and watch MemAvailable in T7's first 10 min.
5. **Arena D P3 start burst.** pod_stage_d.sh:278 starts all workers at once. Proposed: a per-worker start stagger
   (`sleep $((60 * idx))` before the first job in `q_worker`), or launch with `ARENA_D_GPUS=0,1,2` if host RAM is tight.
6. **Stale labels (cosmetic).** run_pod.sh:1026 prints `lambdas ... (placeholder)` (they are settled now).
   driver_tcz.py:12 docstring says `ORDER=list`. wcm_gemma.py `--wcm` help says "the pod runs v4". The runbook
   `docs/reproducing-the-runs.md` says "P5 ... x 5 λ" and "Arena D's WCM ... WCM_VERSION=v2".
7. Known `fits_done` bug (run_pod.sh:657-663): it counts ok rows, not distinct keys. It is harmless on a fresh pod
   WORK, but a rerun after an OUT holds duplicate rows can skip keys. Make it key-based.

## 4. What ran (all in WSL at the repo root, CUDA_VISIBLE_DEVICES="")

- `out/dryrun_1119/setup_restore.sh`: real `scripts/pod_restore.sh` of the 4 small 11-07 archives (symlinked from
  `Runs/pod-results-2026-11-07`) into `out/dryrun_1119/{pub,root}`, then B3's R3 marker commands verbatim
  (`setup_restore.log`). The 11-16 scratch restore had been deleted.
- `out/dryrun_1119/run_dry.sh` -> `run_dry.out` + one log per mode (section 1 row "every DRY_RUN mode").
- `out/dryrun_1119/sim_d4.sh` -> `sim_d4.out`, `sim_<case>/chain.out`: 4 real chain runs on state copies, stub
  `out/dryrun_1119/stub/runpodctl`, MPS=0. First attempt mis-split TL_ROOT (space in path); its sim dirs were
  removed with delete_path and rerun.
- `pytest tests/circuit/instrument` (`pytest_instrument.log`), `pytest tests` (`pytest_full.log`), `pytest
  arena_d/test_wcm_gemma.py -m 'not gpu'` (`pytest_wcm_gemma_cpu.log`).
- Read-only git: status, diff, show, log, ls-files, check-ignore. No GPU, no `.gpu-lock` touched (absent), no HF
  token read.
