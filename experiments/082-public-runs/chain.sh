#!/bin/bash
# 082 PUBLIC-MODEL STAGE CHAIN (DAN-39 B8): stages S1-S10 of dev-notes/2026-10-23-pod-plan-public-models.md §3.3 for
# Pythia-70m and GPT-2 small on one 8-GPU pod, after scripts/pod_bootstrap_public.sh (S0). Run from anywhere; it cds
# to the repo root. PROPOSAL ONLY until Daniel approves each step on the pod ([[remote-pod-discuss-first]]).
#
#   DRY_RUN=1 TL_TRACK=1 bash experiments/082-public-runs/chain.sh      # print every command, run nothing
#   TL_TRACK=1 STOP_POD=1 nohup bash experiments/082-public-runs/chain.sh > experiments/082-public-runs/logs/chain.out 2>&1 &
#   LIMIT=2 TL_TRACK=1 bash experiments/082-public-runs/chain.sh        # local smoke on the 069 smoke stores (FROM=S4)
#
# FINAL DESIGN (Daniel, 2026-10-24; dev-notes/2026-10-23-pod-plan-public-models.md §2.4):
# Two public tracks run side by side, one per model, each with its own data GPU (DATA_GPU_pythia_70m=0, gpt2_small=1):
#   pythia-70m   S1 corpus -> S2 native check -> S3 data pass -> S4 records + frame -> S5 lambda pilot -> S6 main + arms
#   gpt2-small   S1 corpus -> S2 native check -> S3 data pass -> S7 records + frame -> S8 lambda pilot -> S9 main + arms
# then S10 (pack both).
#   S1  a FIXED, pre-stated corpus: N_SEQ = 4,000,000 windows of 64 tokens per model (Pythia: one pinned Pile file,
#       streamed; GPT-2: apollo OWT parquet files 0-2, ~5.8M windows before exact dedupe), revisions pinned
#   S3  pass 1 + neg_ctx only. Pass 2 (co-activation) and candidates are NOT run: nothing in the protocol reads them
#       (build_record / driver / 084 / 073 read top / mid / neg / seq_repr / latent_stats only); 084's frequency
#       deciles use pass 1's own latent_stats (no separate frequency scan)
#   S4/S7  frame: 512 targets (--frame fired: >= 200 firing windows, hub cap 50% of windows), SEED 20261024; subsets
#       pilot 30 (disjoint), nulls 128 (subsample 32 first), stable 32, unweighted 128 (Pythia only)
#   S5/S8  lambda pilot, 30 x 3 lambdas, then C3 (Daniel adopts the lambda; pilot_lambda.py's pre-stated rule)
#   S6/S9  main -> alpha = 1 -> DAN-181 nulls (084: 128 targets x {anchor, contrib} x 1 draw + a 2nd draw on the
#       first 32; WIN 8; production members IN the anchor pool and OUT of the contrib pool; no DAN-17 refits) ->
#       stable core (32) -> unweighted at the adopted lambda (Pythia only, 128) -> 073 resample on the nulls list.
# PROTOCOL v2 (Daniel, 2026-10-25): every fit of every track runs under WCM_VERSION=v2 (exported below; the driver's
# switch: A and C trained under the sparsity-preserving fill, straight-through, against the scorer's means). The
# driver marks each OUT with its version and refuses to mix, so every v2 OUT is fresh. No R arm, no silent term.
# FINAL (Daniel, 2026-11-01; exps 101, 102, 104): the default is now WCM_VERSION=v3 = v2 + the resample term R_C at
# weight 1.0 with rank-keep 1e-1 on every arm (protocol §12.3). "v2" below names the protocol family; the OUTs are
# marked v3 (OUT/WCM_VERSION, OUT/WCM_R).
# v4 (Daniel, 2026-11-04; exp 107 latent_pilot): the default is now WCM_VERSION=v4 = v3 + the geometric lr schedule
# (peak x3.15, warm-up 0.1) + the censored-activation term; the OUTs are marked v4. WCM_VERSION=v3 still runs v3.
# Third track (TL_TRACK=1), TuringLLM, in its own checkout TL_ROOT (scripts/pod_bootstrap_global.sh ROOT=...): a v2
# REFIT of 8,000 targets (094/targets_tl8k.txt) on FRESH records (094/build_records.py: deduplicated pools from the
# stores + the corpus scan, group-aware split, deduplicated contrast), into TL_V2_OUT:
#   T0 artefact check -> T1 corpus scan (086 scan.py, top 512 for the 8k) -> T2 fresh records: the EARLY GATE first
#   (build + independent verify of the 64 gate targets, 094/targets_tl8k_gate64.txt, spread over the 35 cells; any
#   failure stops the track), then build + verify of the rest; a later failure is QUARANTINED (094 quarantine.py:
#   TL_LISTS = every list minus the quarantined, read by T3-T8; quarantine.tsv packed) -> report -> T3 main fits (strongest + mid + specificity; 8 chunks, the lock per chunk) + alpha = 1
#   -> T4 the 190-target lambda sweep (WCM 8 lambda + unweighted 5, one launch per arm) -> T5 stable core (128) ->
#   T6 DAN-181 nulls (126 panel; DAN-17 references all refit under v2) -> T7 DAN-159 / DAN-43 (083 run_pod.sh, our own
#   baselines only: stages TL_DAN159_STAGES = ref attr fitted, then table; WCM / U references = the T4 sweep, REF_RUN)
#   -> T8 case-study + gallery pack (their 15 circuits come from T3) ->
#   T9 (optional, TL_ASRUN=1) the leakage note: the fixed 15k v1 circuits of the as-run 1,000 subset, scored on the
#   as-run records and on 086-rebuilt held-out sets (WCM_VERSION=v1: scoring only).
#   DONOR-BACKGROUND TEST (protocol v2 §12.3, Daniel 2026-10-31; src/eval/donor_test.py, experiments/103-donor-stage),
#   after every fit stage, so it delays none of them:
#   T10 (WCM_VERSION=v2cr / v3 / v4 only) the without-R appendix arm: the running version without R (TL_NOR_VERSION:
#   v4nr under v4, v2 under v2cr / v3) at the production lambda and the running version's rank-keep (TL_NOR_ARM
#   W_0.001 under v4 / v2cr, W_0.001_r0.1 under v3) on the 190 sweep targets, MODE=sweep into its own OUT TL_NOR_OUT
#   (a version-marked OUT cannot mix versions) -> T11 donor test of the 8k main circuits (8 chunks, the lock per chunk) -> T12 donor test of every sweep
#   arm (WCM and U_, one launch: the shared per-target work is done once) + the T10 arm. The public tracks get the
#   same test as S6/S9 arms after the 073 one: donor (main list x main) and, Pythia, donor_unweighted (U_<lambda>).
#   The R pool (073's random pool on the run's own corpus, filtered against every record of its CTX_DIR) is built by
#   the first donor shard of each model (file lock) from local data only: no upload, no download.
# GPU PACKING. Every multi-GPU stage goes through 062/launch.sh on the fit pool = FIT_GPUS (2..7) plus the data GPU of
# every model whose data pass (S3) is done, one holder at a time (the fit lock). The public tracks have PRIORITY: a
# TuringLLM stage takes the lock only while no public stage is waiting for it (STATE/want.*), and every TuringLLM
# sub-step (scan, build, verify, each re-score job, each 083 stage) takes it separately, so a public pilot or main
# fit waits at most one sub-step. So: data passes on GPUs 0 / 1 at the start, the TuringLLM scan on 2..7 meanwhile,
# then the pilots (C3) and the TuringLLM rebuild / re-score / nulls / 083 fill every wait for Daniel.
#
# CHECKPOINTS
#   C2 (S4/S7)  every record check must pass (records_check.json), else the track stops; the target lists are written
#               by make_targets_public.py and md5-frozen in STATE; every later stage re-checks the md5 before fitting.
#   C3 (S5/S8)  the pilot report (pilot_lambda.py: recommendation, "adopted": false) is printed; the main fit then
#               WAITS until Daniel adopts a lambda:  echo 1e-3 > <STATE>/lam_<model>   (or LAM_pythia_70m=... in env).
#               The fit lock is released while waiting, so the other model's fits can use the GPUs meanwhile.
#   Every stage boundary (and every arm inside S6/S9) writes a tar + md5 to CKPT on the pod disk, copies it to RESULTS
#   (at most 4 copies at once: geesefs ENOMEM) and runs md5sum -c ON THE VOLUME. A stage is marked done only after that.
#   C4 (S10)    per-model results + stores tgz, one md5 file, md5sum -c on the volume, tar -tzf counts.
#   STOP_POD=1  (opt-in, D9) `runpodctl stop pod $RUNPOD_POD_ID` only after S10's verification passed in this run.
#
# RESUME: every stage writes STATE/<model>.<stage>.done; a rerun skips done stages, and the fits resume per target
# (driver / companions). To redo a stage delete its marker. FROM=S4 treats S1-S3 as done (their outputs must exist).
#
# ENV (defaults in brackets)
#   STOP_AFTER [S10]  stop each track before any stage numbered higher (GPT-2's S7 > S5: STOP_AFTER=S5 stops GPT-2
#                     after S3); STOP_AFTER_<model> (e.g. STOP_AFTER_gpt2_small=S7) overrides it per track. The lambda
#                     checkpoints need no STOP_AFTER: the main fit waits for the adopted lambda (C3 above).
#                     FROM [S1; S4 when LIMIT is set]        MODELS ["pythia-70m gpt2-small"]
#   DRY_RUN=1         print the commands only (no file is written, no lock taken)
#   LIMIT=N           smoke: WORK=082/smoke_v2 (a fresh dir: v2-marked OUTs), the 069 smoke stores, target lists cut to their first N, STEPS=SMOKE_STEPS
#                     [20], P=1, GPUs 0, no MPS, local CKPT / RESULTS under WORK; the TuringLLM track on TL_SMOKE_TARGETS
#                     (scan on TL_SMOKE_SHARDS corpus shards, 083 on 1 target with tiny budgets)
#   FRAME [fired]     make_targets_public.py --frame (final: fired)   SEED [20261024]
#   LAM_<model>       the adopted lambda (C3), e.g. LAM_pythia_70m=1e-3; else STATE/lam_<model>; smoke: 1e-3
#   PILOT_ARMS_<model> [pythia: W_0.0003,W_0.001,W_0.003; gpt2: W_0.001,W_0.003,W_0.01]   (D2)
#   UNWEIGHTED_<model> [pythia 1, gpt2 0]: the unweighted arm U_<adopted lambda> on targets_<model>_unweighted.txt
#   NULLS_SUB_N [32; smoke 1] NULLS_SUB_DRAWS [2] NULLS_WIN [8] NULLS_EXCLUDE [contrib] NULLS_REFS [wcm]
#                     NULLS_DAN17_REFIT [0] (084; public: no DAN-17 references); NULLS_SUB_DRAWS_<model> etc. per model
#   SAE_PRELOAD [0]   lazy SAE loading for the public fits (driver / companions: config.model.sae_preload)
#   CTX_DIR / CTX_DIR_<model>  [unset = OUT/ctx] context-record dir passed to the driver and companions
#   P [6] STAGGER [5] FIT_GPUS [2,3,4,5,6,7] DATA_GPU_<model> [0 / 1]
#                     (FIT_GPUS=cpu DATA_GPU_<model>=cpu: CUDA_VISIBLE_DEVICES=cpu hides every GPU, a CPU-only smoke)
#   N_SEQ [4000000] BATCH [2048] DTYPE [bfloat16] (S2 failure -> float32 for that model)  LICENCE_<model>
#   WORK [082] LOGS [WORK/logs] STATE [LOGS/state] CKPT [/root/ckpt] RESULTS [/workspace/results] TAG [pub]
#   DATA_ROOT [data_public] STORE_ROOT [outputs_public]  MPS [1]  STOP_POD [0]  PY [./.venv/bin/python]
#   WCM_VERSION [v4]  (every track; v4 = v3 + geometric lr x3.15, warm-up 0.1, censored term, the production
#                     algorithm, 2026-11-04; v3 = Z + A + C + R_C, R weight 1.0, rank-keep 1e-1, 2026-11-01; v4nr / v2
#                     for T10's without-R arm; v1 only for T9's re-score of the fixed 15k circuits)
#   TuringLLM track: TL_TRACK [0]  TL_STOP_AFTER [T12]  TL_ROOT [/root/turing-tl; smoke: this repo]
#                     TL_PY [TL_ROOT/.venv/bin/python if present, else PY]
#                     TL_OUT [TL_ROOT/062/out_full/out: the 15k v1 run, read only (T9)]
#                     TL_WORK [TL_ROOT/082/out_turingllm; smoke: WORK/out_turingllm]
#                     TL_V2_OUT [TL_ROOT/094/out_v2; smoke: TL_WORK/out_v2] (the refit's driver OUT, v2-marked)
#                     TL_CTX_FRESH [TL_WORK/ctx_fresh] (094 records = every T3-T8 fit's CTX_DIR)
#                     TL_TARGETS_DIR [TL_ROOT/094] (targets_tl8k.txt + subsets, md5 in its manifest)
#                     TL_CORPUS_DIR [TL_ROOT/085/out]  TL_P [P]  TL_MAIN_CHUNKS [8]
#                     TL_SCAN_P [1] TL_BUILD_P [3] TL_VERIFY_P [6] (processes per GPU; verify is CPU-bound, the
#                     builder ~2.6 GB RAM per process)  TL_083_P [4]
#                     TL_SWEEP_ARMS [W_2.5e-05 .. W_0.004 (8), U_1e-05 .. U_0.001 (5); smoke W_0.001,W_0.004,U_0.001]
#                     TL_NULLS_SUB_N [32] (the first 32 of the 126 panel: 3-4 per band x kind)
#                     TL_ASRUN [0] (T9; 1 = run the leakage-note stage)  TL_DAN159 [1] (T7; 0 = skip)
#                     TL_DAN159_STAGES ["ref attr fitted"] (T7's 083 stages before table; "exact" adds the IG / AtP
#                     ground truth; "ge" = our adapted Ge, withdrawn: exp 107 arena A runs the real method)
#                     TL_NOR [1] (T10; 0 = skip; skipped anyway unless WCM_VERSION=v2cr / v3 / v4)
#                     TL_NOR_VERSION [v4nr under v4, v2 under v2cr / v3]
#                     TL_NOR_ARM [W_0.001_r0.1 under v3, W_0.001 under v4 / v2cr]
#                     (under v3: v2's W_0.001 = the previous production algorithm; W_0.001_r0.1 = without R only)
#                     TL_NOR_OUT [TL_ROOT/094/out_v2_noR; smoke TL_WORK/out_v2_noR]
#                     TL_DONOR [1] (T11 + T12; 0 = skip both)  TL_DONOR_MAIN [1] TL_DONOR_SWEEP [1] (one of them)
#                     TL_DONOR_POOL [TL_V2_OUT/donor/random_pool.pt] (shared by T11 / T12 / the T10 arm: one CTX_DIR)
#                     (a switched-off stage is reported SKIPPED and not marked done, so a later rerun runs it)
#   DONOR [1]         the public donor arms (S6/S9 donor, donor_unweighted); DONOR_<model> overrides; 0 = skip
#   DONOR_SCRIPT [src/eval/donor_test.py]  DONOR_P [P] (donor processes per GPU, every track)
#   STOP_POD=1 also requires the TuringLLM track (when on) to have finished and packed (D9: final verified backup)
#   ARENA_A [0]       1 = exp 107 arena A (107/arena_a/run_pod.sh) after S10 + the TuringLLM track (arena_a_stage);
#                     ARENA_A_GPUS [0..7] ARENA_A_ANYWAY [0]; STOP_POD then also waits for it. Arena D (D9) goes after it.
#                     Its time limit (2026-11-06): ARENA_A_MAX_H [run_pod.sh's auto: 1.5 x the estimate = 15.75 h on 8
#                     GPUs, 28.5 h on 4] hours after arena A starts = the HARD stop: every arena A process killed
#                     (SIGTERM, SIGKILL 30 s later), then its P7 packs what finished (rc 4), then arena D / STOP_POD;
#                     the soft stop (no new unit) ARENA_A_HARD_GRACE [3600] s before it. ARENA_A_STOP_EPOCH [none] = an
#                     explicit soft stop instead (hard = it + ARENA_A_HARD_GRACE; with ARENA_A_MAX_H too, the earlier)
#   ARENA_D [0]       1 = exp 107 arena D (107/arena_d/pod_stage_d.sh) after arena A (arena_d_stage); ARENA_D_GPUS
#                     [0..7] ARENA_D_ANYWAY [0] ARENA_D_STOP_EPOCH [none] (its STOP_EPOCH: no new job after it, a hard
#                     kill DW_HARD_GRACE later). STOP_POD waits for both arenas but does NOT require them to succeed:
#                     each packs its own results (arena D packs even after a failure), so a failed arena never keeps
#                     the pod up
#
# DEPENDENCIES (built by other B-items; called by the interfaces the plan describes):
#   062/driver.py FAMILY / DATA / STORES / LAM / STEPS / SAE_PRELOAD / CTX_DIR    062/launch.sh GPUS / LOGDIR / SCRIPT
#   062/rescore_alpha1.py A1_TARGETS, OUT_A1                      062/stable_core.py SC_TARGETS, OUT_SC
#   073/strict_eval.py TARGETS, OUT73, RAND_DIR                   all: SHARD=i/k from launch.sh, FAMILY/DATA/STORES
#   082/make_targets_public.py --family --frame --seed --stores --out -> targets_<model>_{main,pilot,nulls,stable
#                     [,unweighted]}.txt + manifest      082/pilot_lambda.py --family --out --targets -> pilot_lambda.json
#   084/harder_nulls.py TARGETS SUB_N SUB_DRAWS N_DRAWS WIN EXCLUDE_MEMBERS REFS RN_DIR DAN17_REFIT OUT84 (+ driver env)
#   086 scan.py MODE=scan|merge (SHARD over corpus shards, OUT_SCAN, CORPUS_DIR, TARGETS, LIMIT_SHARDS, PARTIAL)
#       rebuild.py MODE=build|verify|report (SCAN, CTX_IN, CTX_OUT, CORPUS_DIR, STORES_DIR, OUT_RB, TARGETS, LIMIT)
#       rescore.py JOBS (main,sweep,alpha1), CTX_DIR, OUT_RS, TARGETS, LIMIT; REPORT=1 with ASRUN_RS, ASRUN_TARGETS
#   094 build_records.py MODE=build|verify|report (TARGETS, OUT_DIR, OUT_RB, SCAN, CORPUS_DIR, STORES_DIR, LIMIT)
#   083/run_pod.sh STAGES (ref,attr,fitted,exact,ge,table), CTX_DIR, OUT083, GPUS, P (+ LIMIT / smoke budgets);
#       REF_RUN (083 cmp_common: the run whose main / sweep are the WCM / U references)
#   062/driver.py WCM_VERSION (v1 | v2 | v2cr | v3 | v4 | v4nr; OUT/WCM_VERSION marker)
#   src/eval/donor_test.py (062 companion) OUT, DONOR_TARGETS, DONOR_ARMS, OUT_DONOR, DONOR_POOL (+ driver env)
set -uo pipefail
shopt -s nullglob

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
E=experiments/082-public-runs
D62=experiments/062-h100-protocol-v1
D69=experiments/069-public-data-pass
D73=experiments/073-strict-eval
PY=${PY:-./.venv/bin/python}
export PYTHONPATH=src

LIMIT=${LIMIT:-}
SMOKE=0; [ -n "$LIMIT" ] && SMOKE=1
DRY_RUN=${DRY_RUN:-0}
MODELS=${MODELS:-"pythia-70m gpt2-small"}
STOP_AFTER=${STOP_AFTER:-S10}
SEED=${SEED:-20261024}
FRAME=${FRAME:-fired}
NULLS_WIN=${NULLS_WIN:-8}
NULLS_REFS=${NULLS_REFS:-wcm}
NULLS_EXCLUDE=${NULLS_EXCLUDE:-contrib}
NULLS_DAN17_REFIT=${NULLS_DAN17_REFIT:-0}
NULLS_SUB_DRAWS=${NULLS_SUB_DRAWS:-2}
SAE_PRELOAD=${SAE_PRELOAD:-0}
D84=experiments/084-harder-nulls
STAGGER=${STAGGER:-5}
N_SEQ=${N_SEQ:-4000000}
DTYPE=${DTYPE:-bfloat16}
TAG=${TAG:-pub}
STOP_POD=${STOP_POD:-0}
JOB_GATE=${JOB_GATE:-16}               # a fitting job FAILS when its first JOB_GATE fits all error (job_errors.py)
JOB_WATCH_EVERY=${JOB_WATCH_EVERY:-30} # seconds between the watcher's looks at a running job's rows
DATA_ROOT=${DATA_ROOT:-data_public}
STORE_ROOT=${STORE_ROOT:-outputs_public}
export WCM_VERSION=${WCM_VERSION:-v4}                      # v4 (2026-11-04; exp 107 latent_pilot): every fit of every track
if [ "$SMOKE" = 1 ]; then
  FROM=${FROM:-S4}; WORK=${WORK:-$E/smoke_v2}; P=${P:-1}; FIT_GPUS=${FIT_GPUS:-0}; MPS=${MPS:-0}
  DATA_GPU_pythia_70m=${DATA_GPU_pythia_70m:-0}; DATA_GPU_gpt2_small=${DATA_GPU_gpt2_small:-0}
  BATCH=${BATCH:-128}; SMOKE_STEPS=${SMOKE_STEPS:-20}; N_AUTO=${N_AUTO:-2}
  CKPT=${CKPT:-$WORK/ckpt}; RESULTS=${RESULTS:-$WORK/results}
  NULLS_SUB_N=${NULLS_SUB_N:-1}; TL_ROOT=${TL_ROOT:-$ROOT}
else
  FROM=${FROM:-S1}; WORK=${WORK:-$E}; P=${P:-6}; FIT_GPUS=${FIT_GPUS:-2,3,4,5,6,7}; MPS=${MPS:-1}
  BATCH=${BATCH:-2048}; N_AUTO=${N_AUTO:-16}
  CKPT=${CKPT:-/root/ckpt}; RESULTS=${RESULTS:-/workspace/results}
  NULLS_SUB_N=${NULLS_SUB_N:-32}; TL_ROOT=${TL_ROOT:-/root/turing-tl}
fi
LOGS=${LOGS:-$WORK/logs}
STATE=${STATE:-$LOGS/state}
TL_TRACK=${TL_TRACK:-0}
TL_STOP_AFTER=${TL_STOP_AFTER:-T12}
if [ -z "${TL_PY:-}" ] && [ -x "$TL_ROOT/.venv/bin/python" ]; then TL_PY=$TL_ROOT/.venv/bin/python; fi   # its own venv
TL_PY=${TL_PY:-$PY}
case $TL_PY in /*) ;; *) TL_PY=$ROOT/${TL_PY#./} ;; esac      # TL stages run from TL_ROOT: absolute interpreter
TL_OUT=${TL_OUT:-$TL_ROOT/$D62/out_full/out}
case $WORK in /*) WORKA=$WORK ;; *) WORKA=$ROOT/$WORK ;; esac
if [ "$SMOKE" = 1 ]; then TL_WORK=${TL_WORK:-$WORKA/out_turingllm}; else TL_WORK=${TL_WORK:-$TL_ROOT/$E/out_turingllm}; fi
TL_P=${TL_P:-$P}
TL_SCAN_P=${TL_SCAN_P:-1}; TL_BUILD_P=${TL_BUILD_P:-3}; TL_VERIFY_P=${TL_VERIFY_P:-6}; TL_083_P=${TL_083_P:-4}
TL_NULLS_SUB_N=${TL_NULLS_SUB_N:-32}
[ "$SMOKE" = 1 ] && { TL_SCAN_P=1; TL_BUILD_P=1; TL_VERIFY_P=1; TL_083_P=1; TL_NULLS_SUB_N=1; }
D86=experiments/086-heldout-rebuild
case $LOGS in /*) LOGA=$LOGS ;; *) LOGA=$ROOT/$LOGS ;; esac
TL_CORPUS_DIR=${TL_CORPUS_DIR:-$TL_ROOT/experiments/085-heldout-audit/out}   # 085 corpus_hash.npy (recomputed at bootstrap)
TL_ASRUN_TARGETS=${TL_ASRUN_TARGETS:-$TL_ROOT/$D86/targets_asrun1000.txt}    # frozen stratified 1,000 (086 subset.py)
TL_SMOKE_TARGETS=$TL_WORK/targets_smoke.txt                                  # smoke: the first LIMIT of the 084 panel
TL_SMOKE_SHARDS=${TL_SMOKE_SHARDS:-6}                                        # smoke: corpus shards scanned
# protocol v2 refit (094): the 8k list + subsets, the fresh records, the v2 OUT
D94=experiments/094-tl-v2-records
TL_TARGETS_DIR=${TL_TARGETS_DIR:-$TL_ROOT/$D94}
if [ "$SMOKE" = 1 ]; then TL_V2_OUT=${TL_V2_OUT:-$TL_WORK/out_v2}; else TL_V2_OUT=${TL_V2_OUT:-$TL_ROOT/$D94/out_v2}; fi
TL_CTX_FRESH=${TL_CTX_FRESH:-$TL_WORK/ctx_fresh}
TL_LISTS=${TL_LISTS:-$TL_WORK/lists}     # T2's quarantine-applied lists (every stage after T2 reads these: tl_use)
TL_MAIN_CHUNKS=${TL_MAIN_CHUNKS:-8}
TL_ASRUN=${TL_ASRUN:-0}    # off by default (Daniel, 2026-10-26: no v1 re-score on the pod under v2)
# T7 (Daniel, 2026-11-03, DAN-206): ON again, trimmed to our own baselines (DAN-159 rescoped, DAN-43): ref (re-score
# the WCM / unweighted sweep on the rebuilt held-out sets), attr (IG / AtP rankings, DAN-43's input), fitted (DAN-43),
# then table. The adapted Ge stage stays out (2026-11-01: other papers' methods only as published, exp 107).
TL_DAN159=${TL_DAN159:-1}
TL_DAN159_STAGES=${TL_DAN159_STAGES:-ref attr fitted}
if [ "$SMOKE" = 1 ]; then TL_SWEEP_ARMS=${TL_SWEEP_ARMS:-W_0.001,W_0.004,U_0.001}; TL_MAIN_CHUNKS=1
else TL_SWEEP_ARMS=${TL_SWEEP_ARMS:-W_2.5e-05,W_5e-05,W_0.0001,W_0.00025,W_0.0005,W_0.001,W_0.002,W_0.004,U_1e-05,U_3e-05,U_0.0001,U_0.0003,U_0.001}; fi
# donor-background test (protocol v2 §12.3; src/eval/donor_test.py)
DONOR=${DONOR:-1}
DONOR_SCRIPT=${DONOR_SCRIPT:-src/eval/donor_test.py}
DONOR_P=${DONOR_P:-$P}
TL_NOR=${TL_NOR:-1}
# the without-R arm changes R alone: the running version without R at its rank-keep (v4 -> v4nr, whose own rank-keep
# is v4's 1e-1; v3 -> v2 at v3's 1e-1; v2cr -> v2 at v2's 3e-3)
if [ "$WCM_VERSION" = v4 ]; then TL_NOR_VERSION=${TL_NOR_VERSION:-v4nr}; else TL_NOR_VERSION=${TL_NOR_VERSION:-v2}; fi
if [ "$WCM_VERSION" = v3 ]; then TL_NOR_ARM=${TL_NOR_ARM:-W_0.001_r0.1}; else TL_NOR_ARM=${TL_NOR_ARM:-W_0.001}; fi
if [ "$SMOKE" = 1 ]; then TL_NOR_OUT=${TL_NOR_OUT:-$TL_WORK/out_v2_noR}; else TL_NOR_OUT=${TL_NOR_OUT:-$TL_ROOT/$D94/out_v2_noR}; fi
TL_DONOR=${TL_DONOR:-1}; TL_DONOR_MAIN=${TL_DONOR_MAIN:-1}; TL_DONOR_SWEEP=${TL_DONOR_SWEEP:-1}
TL_DONOR_POOL=${TL_DONOR_POOL:-$TL_V2_OUT/donor/random_pool.pt}
TAR_Z=(-z); command -v pigz > /dev/null 2>&1 && TAR_Z=(-I pigz)
FINAL_VERIFIED=0

# ------------------------------------------------------------------------------------------------- helpers
say() {
  local line
  line="[$(date +%H:%M:%S)] $*"
  echo "$line"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$LOGS"; echo "$line" >> "$LOGS/chain.events"; }
}
mvar() { local v="${1}_${2//-/_}"; printf '%s' "${!v:-$3}"; }       # mvar NAME model default -> $NAME_<model>
snum() { local s=${1#S}; echo "${s%%.*}"; }
allowed() { [ "$(snum "$1")" -le "$(snum "${2:-$STOP_AFTER}")" ]; }        # allowed STAGE [STOP]
before_from() { [ "$(snum "$1")" -lt "$(snum "$FROM")" ]; }
is_done() { [ -f "$STATE/$1.done" ]; }
mark() { [ "$DRY_RUN" = 1 ] || { mkdir -p "$STATE"; date '+%F %T' > "$STATE/$1.$2"; }; }   # mark KEY done|start
data_gpu() { mvar DATA_GPU "$1" "$([ "$1" = gpt2-small ] && echo 1 || echo 0)"; }
out_dir() { echo "$WORK/out_$1"; }
tlist() { if [ "$SMOKE" = 1 ]; then echo "$WORK/targets_$1_$2.limit.txt"; else echo "$WORK/targets_$1_$2.txt"; fi; }
fit_stages() { if [ "$1" = gpt2-small ]; then echo "S7 S8 S9"; else echo "S4 S5 S6"; fi; }

run() {   # run LOG cmd...   (LOG '-' = this terminal). DRY_RUN prints the command instead.
  local log=$1; shift
  if [ "$DRY_RUN" = 1 ]; then
    printf '    $ %s' "$(printf '%q ' "$@")"
    [ "$log" = - ] && echo || echo ">> $log 2>&1"
    return 0
  fi
  if [ "$log" = - ]; then "$@"; else mkdir -p "$(dirname "$log")"; "$@" >> "$log" 2>&1; fi
}

copy_verified() {   # copy_verified FILE MD5FILE: cp to RESULTS in one of 4 slots, then md5sum -c on the volume
  local f=$1 m=$2 tries=0 slot rc
  if [ "$DRY_RUN" = 1 ]; then
    echo "    $ cp $f $m $RESULTS/ && (cd $RESULTS && md5sum -c $(basename "$m"))   # <= 4 copies at once"
    return 0
  fi
  mkdir -p "$RESULTS" "$STATE"
  while :; do
    for slot in 0 1 2 3; do
      (
        flock -n 9 || exit 75
        # geesefs: no chmod, so cp may complain about modes; the md5 check below is the verdict
        cp --no-preserve=mode,ownership "$f" "$m" "$RESULTS/" 2> /dev/null
        cd "$RESULTS" && md5sum -c --quiet "$(basename "$m")"
      ) 9> "$STATE/copy.$slot.lock"
      rc=$?
      if [ "$rc" = 0 ]; then
        say "  verified on $RESULTS: $(basename "$f") ($(du -h "$f" | cut -f1))"
        echo "$(basename "$f")" >> "$STATE/verified.list"
        return 0
      fi
      if [ "$rc" != 75 ]; then
        tries=$((tries + 1))
        say "  copy / md5 check of $(basename "$f") failed (attempt $tries)"
        [ "$tries" -ge 3 ] && return 1
        sleep 20
      fi
    done
    sleep 5
  done
}

archive() {   # archive NAME PATH...: tar + md5 in CKPT, verified copy on RESULTS
  local name=$1 p rc; shift
  local paths=()
  for p in "$@"; do
    if [ -e "$p" ] || [ "$DRY_RUN" = 1 ]; then paths+=("$p"); fi
  done
  if [ ${#paths[@]} -eq 0 ]; then say "  archive $name: nothing to pack"; return 1; fi
  local tgz=$TAG-$name.tgz
  if [ "$DRY_RUN" = 1 ]; then
    run - tar "${TAR_Z[@]}" -cf "$CKPT/$tgz" "${paths[@]}"
    echo "    $ (cd $CKPT && md5sum $tgz > $TAG-$name.md5); tar -tzf $CKPT/$tgz | wc -l"
    copy_verified "$CKPT/$tgz" "$CKPT/$TAG-$name.md5"
    return 0
  fi
  mkdir -p "$CKPT"
  tar "${TAR_Z[@]}" -cf "$CKPT/$tgz" "${paths[@]}"
  rc=$?
  if [ "$rc" -gt 1 ]; then say "  tar $tgz failed (rc $rc)"; return 1; fi    # rc 1 = a file changed while read (logs)
  (cd "$CKPT" && md5sum "$tgz" > "$TAG-$name.md5") || return 1
  say "  packed $tgz: $(tar -tzf "$CKPT/$tgz" | wc -l) entries, $(du -h "$CKPT/$tgz" | cut -f1)"
  copy_verified "$CKPT/$tgz" "$CKPT/$TAG-$name.md5"
}

stage() {   # stage NUM KEY FUNC ARGS...: STOP_AFTER / FROM gating, done markers. rc 3 = stopped by STOP_AFTER
  local num=$1 key=$2 rc stop; shift 2
  stop=$(mvar STOP_AFTER "${key%%.*}" "$STOP_AFTER")                     # STOP_AFTER_<model> overrides
  if ! allowed "$num" "$stop"; then say "$key: after STOP_AFTER=$stop; this track stops here"; return 3; fi
  if before_from "$num"; then say "$key: before FROM=$FROM, taken as done"; return 0; fi
  if is_done "$key"; then say "$key: done earlier (marker), skipped"; return 0; fi
  say "$key: START"
  mark "$key" start
  "$@"
  rc=$?
  if [ "$rc" = 0 ]; then mark "$key" done; say "$key: DONE"; else say "$key: FAILED (rc $rc)"; fi
  return "$rc"
}

fit_pool() {   # FIT_GPUS + the data GPU of every model (in MODELS) whose data pass (S3) is finished
  local pool=$FIT_GPUS m
  for m in $MODELS; do
    if is_done "$m.S3" || before_from S4; then pool="$pool,$(data_gpu "$m")"; fi
  done
  echo "$pool" | tr ',' '\n' | awk 'NF && !seen[$0]++' | paste -sd, -
}

pool_size() { echo "$1" | tr ',' '\n' | grep -c .; }

with_fit_lock() {   # with_fit_lock FUNC ARGS...: one holder at a time on the fit pool; a public caller registers as
  #                   waiting (STATE/want.<pid>) so the TuringLLM track yields to it (tl_with_fit_lock)
  if [ "$DRY_RUN" = 1 ]; then "$@"; return; fi
  mkdir -p "$STATE"
  local want=$STATE/want.$BASHPID
  : > "$want"
  (
    flock 9
    rm -f "$want"
    "$@"
  ) 9> "$STATE/fit.lock"
  local rc=$?
  rm -f "$want"
  return "$rc"
}

public_waiting() { local w=("$STATE"/want.*); [ ${#w[@]} -gt 0 ]; }       # nullglob: no match = empty array

tl_with_fit_lock() {   # the TuringLLM track's lock: taken only while no public stage waits (public priority);
  #                      after taking it, a public waiter that arrived meanwhile keeps the lock until it is released
  if [ "$DRY_RUN" = 1 ]; then "$@"; return; fi
  mkdir -p "$STATE"
  local n=0
  while :; do
    while public_waiting; do
      [ $((n % 60)) = 0 ] && say "  turingllm: yielding the fit pool to a public stage" >&2
      n=$((n + 1)); sleep 10
    done
    (
      flock 9
      if public_waiting; then exit 75; fi                                        # a public stage queued meanwhile
      "$@"
    ) 9> "$STATE/fit.lock"
    local rc=$?
    [ "$rc" = 75 ] && { sleep 2; continue; }
    return "$rc"
  done
}

gave_up() {   # gave_up DIR: "GIVING UP" lines in DIR/*.log (nullglob-safe; logs are appended across reruns, so
  #           callers compare the count before and after a launch)
  local f=("$1"/*.log)
  [ ${#f[@]} -gt 0 ] || { echo 0; return; }
  cat "${f[@]}" | grep -c "GIVING UP"
}

split_rows() {   # split_rows ENV...: JOB_ROWS = the value of a ROWS=<glob[,glob]> entry (the job's output rows, read by
  #                job_errors.py, never passed on); JOB_ARGS = the other entries
  JOB_ROWS=; JOB_ARGS=()
  local a
  for a in "$@"; do case $a in ROWS=*) JOB_ROWS=${a#ROWS=} ;; *) JOB_ARGS+=("$a") ;; esac; done
}

run_job() {   # run_job LOG JOB ROWS cmd...: run a fitting job (DRY_RUN prints it) under job_errors.py (2026-10-26):
  #             a watcher stops the launch once its first JOB_GATE rows exist and ALL error (the early gate: rc 1,
  #             FAILED line with the first error, the stage is not marked done); otherwise errors stop nothing and
  #             the count + first error go to chain.events, the launch log and LOGS/job_errors.jsonl
  local log=$1 job=$2 rows=$3; shift 3
  if [ -z "$rows" ]; then run "$log" "$@"; return; fi
  local snap=$STATE/jobs/$job.snap live=$STATE/jobs/$job.live.json
  if [ "$DRY_RUN" = 1 ]; then
    run "$log" "$@"
    echo "    $ (job errors) $job: rows $rows | FAILS if its first $JOB_GATE fits all error (watched), else counted"
    return 0
  fi
  mkdir -p "$STATE/jobs" "$(dirname "$log")"
  "$PY" -X utf8 "$E/job_errors.py" snapshot --rows "$rows" --snapshot "$snap" || return 1
  setsid "$@" >> "$log" 2>&1 &                    # its own process group, so the watcher can stop all of it
  local pid=$! wpid rc crc msg
  "$PY" -X utf8 "$E/job_errors.py" watch --rows "$rows" --snapshot "$snap" --pid "$pid" --name "$job" \
    --live "$live" --gate "$JOB_GATE" --every "$JOB_WATCH_EVERY" >> "$log" 2>&1 &
  wpid=$!
  wait "$pid"; rc=$?
  kill "$wpid" 2> /dev/null; wait "$wpid" 2> /dev/null
  msg=$("$PY" -X utf8 "$E/job_errors.py" check --rows "$rows" --snapshot "$snap" --name "$job" --gate "$JOB_GATE" \
    --jsonl "$LOGS/job_errors.jsonl" --live "$live")
  crc=$?
  echo "$msg" >> "$log"
  say "  $msg"
  [ "$crc" = 0 ] || return 1
  return "$rc"
}

launch() {   # launch MODEL NAME ENV...: 062/launch.sh on the current fit pool; 1 if any shard gave up in this launch,
  #           or if the job's early error gate failed (ENV may hold ROWS=<glob>: its output rows, see run_job)
  local F=$1 name=$2 pool gu0; shift 2
  split_rows "$@"
  pool=$(fit_pool)
  local logdir=$LOGS/$F/$name
  local extra=() ctx
  # smoke: LIMIT= (empty) for the jobs, so each fits its whole cut list (the main list holds every subset's picks);
  # the chain's own LIMIT would otherwise cut main to its first LIMIT and leave the subsets without circuits
  [ "$SMOKE" = 1 ] && extra+=(STEPS="$SMOKE_STEPS" LIMIT=)
  ctx=$(mvar CTX_DIR "$F" "${CTX_DIR:-}")
  [ -n "$ctx" ] && extra+=(CTX_DIR="$ctx")                      # unset: the driver's default (OUT/ctx)
  say "  $F $name on GPUs $pool (P=$P)"
  gu0=$(gave_up "$logdir")
  run_job "$LOGS/$F/$name.launch.log" "$F.$name" "$JOB_ROWS" env GPUS="$pool" P="$P" STAGGER="$STAGGER" MERGE=0 \
    LOGDIR="$logdir" PY="$PY" WCM_VERSION="$WCM_VERSION" FAMILY="$F" DATA="$DATA_ROOT/$F" STORES="$STORE_ROOT/$F" \
    OUT="$(out_dir "$F")" SAE_PRELOAD="$SAE_PRELOAD" "${extra[@]}" "${JOB_ARGS[@]}" bash "$D62/launch.sh" || return 1
  if [ "$DRY_RUN" != 1 ] && [ "$(gave_up "$logdir")" -gt "$gu0" ]; then
    say "  $F $name: a shard gave up after 5 attempts (see $logdir)"; return 1
  fi
}

verify_targets() {   # the frozen lists of S4/S7 are unchanged
  [ "$DRY_RUN" = 1 ] && { echo "    $ md5sum -c $STATE/targets_$1.md5"; return 0; }
  if ! (md5sum -c --quiet "$STATE/targets_$1.md5"); then say "  $1: target lists changed since they were frozen"; return 1; fi
}

get_lam() {   # the adopted lambda (C3): env LAM_<model>, else STATE/lam_<model>; waits for Daniel. stdout = lambda
  local F=$1 v n=0
  v=$(mvar LAM "$F" "")
  if [ -n "$v" ]; then echo "$v"; return 0; fi
  if [ "$SMOKE" = 1 ]; then echo 1e-3; return 0; fi
  if [ "$DRY_RUN" = 1 ]; then echo "<lambda-from-C3>"; return 0; fi
  while [ ! -s "$STATE/lam_$F" ]; do
    [ $((n % 30)) = 0 ] && say "C3 $F: waiting for the adopted lambda: echo <lambda> > $STATE/lam_$F" >&2
    n=$((n + 1))
    sleep 60
  done
  tr -d ' \n' < "$STATE/lam_$F"
}

# ------------------------------------------------------------------------------------------------- stages
s1_corpus() {
  local F=$1 args
  local budget=(--max-sequences "$N_SEQ")
  [ "$SMOKE" = 1 ] && budget=(--max-sequences "${SMOKE_SEQS:-2048}" --seqs-per-shard 1024)
  if [ "$F" = pythia-70m ]; then
    # verified 2026-10-23 (HF metadata): 30 train shards train/NN.jsonl.zst of ~11.1 GB, rows {text, meta.pile_set_name},
    # subsets interleaved in the first rows; card licence "other"
    args=(--hf-dataset monology/pile-uncopyrighted --hf-revision 3be90335b66f24456a5d6659d9c8d208c0357119
          --hf-pattern "train/00.jsonl.zst" --doc-mode row --source-name "monology/pile-uncopyrighted train/00"
          --licence "$(mvar LICENCE "$F" "UNCONFIRMED: HF card license 'other' (The Pile minus Books3, BookCorpus2, OpenSubtitles, YTSubtitles, OWT2)")")
  else
    # verified 2026-10-23: 73 parquet files of ~243 MB, 8,824,092 rows x 1,024 int32 gpt2 ids, EOS 50256 after every
    # document, no BOS, rows shuffled; one file = ~120.9k rows = ~1.93M windows of 64, so files 0-2 = ~5.8M windows
    # before exact dedupe, enough for the fixed N_SEQ = 4M (the writer stops there; the manifest records the counts)
    args=(--hf-dataset apollo-research/Skylion007-openwebtext-tokenizer-gpt2
          --hf-revision f02886b54795e8acabceb637ca119f9ae8f19d3f
          --hf-pattern "data/train-0000[0-2]-of-00073.parquet" --token-field input_ids
          --source-name "OpenWebText (Skylion007), apollo-research gpt2-tokenised, 1,024-token rows, files 0-2 of 73"
          --licence "$(mvar LICENCE "$F" "UNCONFIRMED: Skylion007/openwebtext card CC0-1.0; the apollo copy has no card licence")")
  fi
  if [ "$DRY_RUN" != 1 ] && [ -f "$DATA_ROOT/$F/corpus_manifest.json" ]; then
    say "  $F: $DATA_ROOT/$F/corpus_manifest.json exists (the corpus was written); packing it"
  elif [ "$DRY_RUN" != 1 ] && shards=("$DATA_ROOT/$F"/*.npy) && [ ${#shards[@]} -gt 0 ]; then   # nullglob-safe
    say "  $F: $DATA_ROOT/$F holds shards but no manifest (an interrupted run): move it away, then rerun"; return 1
  else
    run "$LOGS/$F/corpus.log" env HF_HUB_OFFLINE=0 "$PY" -X utf8 "$D69/prepare_corpus.py" --family "$F" \
      --out "$DATA_ROOT/$F" "${args[@]}" "${budget[@]}" || return 1
  fi
  archive "$F-S1-corpus" "$DATA_ROOT/$F" "$LOGS/$F"
}

s2_native() {
  local F=$1
  if run "$LOGS/$F/native_check.log" env CUDA_VISIBLE_DEVICES="$(data_gpu "$F")" HF_HUB_OFFLINE=1 \
      "$PY" -X utf8 "$D69/gpu_native_check.py" --family "$F" --data "$DATA_ROOT/$F"; then
    say "  $F: native check passed; data pass dtype $DTYPE"
  else
    say "  $F: native check FAILED; this model's data pass falls back to float32 (see $LOGS/$F/native_check.log)"
    [ "$DRY_RUN" = 1 ] || echo float32 > "$STATE/$F.dtype"
  fi
  archive "$F-S2-native" "$LOGS/$F"
}

data_pass() {   # data_pass MODEL STAGES LOGNAME
  local F=$1 dtype=$DTYPE
  [ -s "$STATE/$F.dtype" ] && dtype=$(cat "$STATE/$F.dtype")
  local extra=()
  [ "$SMOKE" = 1 ] && extra+=(--mid-warmup 4)
  run "$LOGS/$F/$3.log" env CUDA_VISIBLE_DEVICES="$(data_gpu "$F")" HF_HUB_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -X utf8 "$D69/run_data_pass.py" --family "$F" \
    --data "$DATA_ROOT/$F" --out "$STORE_ROOT/$F" --batch-size "$BATCH" --dtype "$dtype" --stages "$2" "${extra[@]}"
}

s3_datapass() {   # pass 1 + neg_ctx only (final 2026-10-24): pass 2 / candidates are read by nothing downstream
  local F=$1
  data_pass "$F" pass1,negctx dp_pass1 || return 1
  [ "$DRY_RUN" = 1 ] || say "  C1 $F: $(grep -h -m1 -iE 'pass1.*(s|sec)' "$LOGS/$F/dp_pass1.log" 2> /dev/null | tail -c 200)"
  archive "$F-S3-stores" "$STORE_ROOT/$F" "$LOGS/$F"
}

records_gate() {   # all_usable_pass, and every all_checks_pass in records_check.json is true (None = unusable record)
  [ "$DRY_RUN" = 1 ] && { echo "    $ (gate) all_usable_pass and every all_checks_pass in $1 are true"; return 0; }
  "$PY" -c '
import json, sys
rep = json.load(open(sys.argv[1]))
if rep.get("all_usable_pass") is False or rep.get("n_usable") == 0:
    print("  record checks: all_usable_pass %s, n_usable %s" % (rep.get("all_usable_pass"), rep.get("n_usable")))
    sys.exit(1)
vals = []
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "all_checks_pass":
                vals.append(v)
            else:
                walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(rep)
bad, ok = vals.count(False), vals.count(True)
print("  record checks: %d pass, %d fail, %d without a usable record" % (ok, bad, len(vals) - ok - bad))
sys.exit(0 if ok and not bad else 1)' "$1"
}

s_records() {   # S4 / S7 (C2): records check, reports, the frozen target frame
  local F=$1 S=$STORE_ROOT/$F Dd=$DATA_ROOT/$F
  local gpu R=$STORE_ROOT/$F
  gpu=$(data_gpu "$F")
  # smoke: the reports go to WORK, so the 069 smoke stores' own report files are not overwritten
  [ "$SMOKE" = 1 ] && { R=$WORK/reports/$F; [ "$DRY_RUN" = 1 ] || mkdir -p "$R"; }
  run "$LOGS/$F/records.log" env CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 "$PY" -X utf8 "$D69/check_records.py" \
    --family "$F" --data "$Dd" --out "$S" --n-auto "$N_AUTO" --report "$R/records_check.json" || return 1
  records_gate "$R/records_check.json" || { say "  C2 $F: record checks FAILED; fits not started"; return 1; }
  run "$LOGS/$F/dup_report.log" env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -X utf8 "$D69/dup_report.py" --family "$F" \
    --data "$Dd" --out "$S" --report "$R/duplicate_report.json" || say "  (dup_report failed; not a gate)"
  if [ "$SMOKE" != 1 ]; then                                 # writes corpus_estimate.json into the store root
    run "$LOGS/$F/estimate_corpus.log" env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -X utf8 "$D69/estimate_corpus.py" \
      --family "$F" --data "$Dd" --out "$S" --real-warmup-seqs 204800 || say "  (estimate_corpus failed; not a gate)"
  fi
  run "$LOGS/$F/targets.log" "$PY" -X utf8 "$E/make_targets_public.py" --family "$F" --frame "$FRAME" \
    --seed "$SEED" --stores "$S" --out "$WORK" || return 1
  [ "$DRY_RUN" = 1 ] || tail -n 70 "$LOGS/$F/targets.log" | grep -E "frame|excluded|dead|WARNING" | tail -n 55
  local names=(main pilot stable nulls) n
  [ "$(unweighted_on "$F")" = 1 ] && names+=(unweighted)
  local lists=("$WORK/targets_${F}_manifest.json")
  for n in "${names[@]}"; do lists+=("$WORK/targets_${F}_$n.txt"); done
  if [ "$SMOKE" = 1 ]; then          # the first LIMIT of each list; main also takes the inside-main subsets' picks,
    for n in "${names[@]}"; do       # so every arm has its production circuit
      if [ "$DRY_RUN" = 1 ]; then echo "    $ head -n $LIMIT $WORK/targets_${F}_$n.txt > $(tlist "$F" "$n")"
      else head -n "$LIMIT" "$WORK/targets_${F}_$n.txt" > "$(tlist "$F" "$n")" || return 1; fi
      lists+=("$(tlist "$F" "$n")")
    done
    if [ "$DRY_RUN" != 1 ]; then
      local sub=()
      for n in "${names[@]}"; do [ "$n" = main ] || [ "$n" = pilot ] || sub+=("$(tlist "$F" "$n")"); done
      cat "$(tlist "$F" main)" "${sub[@]}" | awk 'NF && !seen[$0]++' > "$(tlist "$F" main).tmp" \
        && mv "$(tlist "$F" main).tmp" "$(tlist "$F" main)" || return 1
    fi
  fi
  if [ "$DRY_RUN" = 1 ]; then echo "    $ md5sum ${lists[*]} > $STATE/targets_$F.md5   # frozen (C2)"
  else
    md5sum "${lists[@]}" > "$STATE/targets_$F.md5" || return 1
    local counts=""
    for n in "${names[@]}"; do counts="$counts $(wc -l < "$(tlist "$F" "$n")") $n,"; done
    say "  C2 $F: target lists frozen (seed $SEED, frame $FRAME):$counts md5 in $STATE/targets_$F.md5"
  fi
  local reports=("$R"/*.json)
  [ ${#reports[@]} -eq 0 ] && reports=("$R/*.json")             # dry run before the stores exist
  archive "$F-$(fit_stages "$F" | cut -d' ' -f1)-records" "${reports[@]}" "${lists[@]}" \
    "$STATE/targets_$F.md5" "$LOGS/$F"
}

pilot_body() {
  launch "$1" pilot MODE=sweep TARGETS="$(tlist "$1" pilot)" SWEEP_ARMS="$2" ROWS="$(out_dir "$1")/sweep/eval.shard*.jsonl"
}

s_pilot() {   # S5 / S8 (C3): the lambda pilot, MODE=sweep over the pilot list, then the size-only report
  local F=$1 out arms
  out=$(out_dir "$F")
  arms=$(mvar PILOT_ARMS "$F" "$([ "$F" = gpt2-small ] && echo W_0.001,W_0.003,W_0.01 || echo W_0.0003,W_0.001,W_0.003)")
  verify_targets "$F" || return 1
  with_fit_lock pilot_body "$F" "$arms" || return 1
  run "$LOGS/$F/pilot_lambda.log" env FAMILY="$F" OUT="$out" "$PY" -X utf8 "$E/pilot_lambda.py" --family "$F" \
    --out "$out" --targets "$(tlist "$F" pilot)" || say "  (pilot_lambda.py failed; read the sweep rows by hand)"
  if [ "$DRY_RUN" != 1 ] && [ -f "$out/pilot_lambda.json" ]; then
    say "  C3 $F: $("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("pilot recommends lambda %s (%s); adopted: %s" % (r.get("recommended_lam"), r.get("reason"), r.get("adopted")))' "$out/pilot_lambda.json")"
  fi
  say "  C3 $F: adopt a lambda with: echo <lambda> > $STATE/lam_$F   (the main fit waits for it)"
  archive "$F-$(fit_stages "$F" | cut -d' ' -f2)-pilot" "$out/sweep" "$out/ctx" "$out/pilot_lambda.json" "$LOGS/$F"
}

arm() {   # arm MODEL STAGE NAME REQUIRED ENV...: one sub-step of S6/S9 with its own marker and tar
  local F=$1 num=$2 name=$3 required=$4; shift 4
  local key=$F.$num.$name
  if is_done "$key"; then say "  $key: done earlier, skipped"; return 0; fi
  mark "$key" start
  local out
  out=$(out_dir "$F")
  if launch "$F" "$name" "$@"; then
    local what=("$out/$name")
    case $name in main) what=("$out/main" "$out/ctx") ;; unweighted) what=("$out/sweep") ;; esac
    if archive "$F-$num-$name" "${what[@]}" "$LOGS/$F"; then
      mark "$key" done; say "  $key: DONE"; return 0
    fi
  fi
  say "  $key: FAILED"
  [ "$required" = 1 ] && return 1
  echo "$key" >> "$STATE/failed.list"
  return 2
}

nulls_env() {   # nulls_env OUT84 TARGETS MODEL: the 084 stage's settings (final 2026-10-24), one per line: 128 targets
  #             x {anchor, contrib} x 1 draw, + a 2nd draw on the first NULLS_SUB_N; WIN 8; members in the anchor pool,
  #             out of the contrib pool; public: no DAN-17 reference (REFS wcm, no refit). NULLS_*_<model> override
  printf '%s\n' SCRIPT="$D84/harder_nulls.py" TARGETS="$2" N_TARGETS=0 SUB_N="$NULLS_SUB_N" \
    SUB_DRAWS="$(mvar NULLS_SUB_DRAWS "$3" "$NULLS_SUB_DRAWS")" N_DRAWS=1 WIN="$NULLS_WIN" \
    EXCLUDE_MEMBERS="$(mvar NULLS_EXCLUDE "$3" "$NULLS_EXCLUDE")" REFS="$(mvar NULLS_REFS "$3" "$NULLS_REFS")" \
    DAN17_REFIT="$(mvar NULLS_DAN17_REFIT "$3" "$NULLS_DAN17_REFIT")" OUT84="$1"
}

unweighted_on() { mvar UNWEIGHTED "$1" "$([ "$1" = pythia-70m ] && echo 1 || echo 0)"; }

lam_arm() {   # lam_arm PREFIX LAMBDA -> PREFIX_<lambda in %g> (U_0.001 for 1e-3: the driver's sweep arm names)
  if [ "$DRY_RUN" = 1 ] && [ "${2#<}" != "$2" ]; then echo "$1_<lambda>"; return; fi
  awk -v p="$1" -v l="$2" 'BEGIN { printf "%s_%g\n", p, l + 0 }'
}

main_body() {   # under the fit lock: main -> alpha = 1 -> harder nulls -> stable core -> (unweighted) -> 073 -> donor
  #               (-> donor_unweighted)
  local F=$1 num=$2 lam=$3 out rc=0
  local ne=()
  out=$(out_dir "$F")
  # ROWS= each arm's output rows: run_job's error gate (an arm whose first 16 fits all error FAILS; other errors are
  # counted and logged)
  arm "$F" "$num" main 1 MODE=main TARGETS="$(tlist "$F" main)" LAM="$lam" ROWS="$out/main/status.shard*.jsonl" \
    || return 1
  arm "$F" "$num" alpha1 0 MODE=main SCRIPT="$D62/rescore_alpha1.py" A1_TARGETS="$(tlist "$F" main)" \
    OUT_A1="$out/alpha1" LAM="$lam" ROWS="$out/alpha1/alpha1*.jsonl" || rc=2
  mapfile -t ne < <(nulls_env "$out/harder_nulls" "$(tlist "$F" nulls)" "$F")
  arm "$F" "$num" harder_nulls 0 MODE=main "${ne[@]}" RN_DIR= LAM="$lam" \
    ROWS="$out/harder_nulls/nulls*.jsonl,$out/harder_nulls/profile*.jsonl" || rc=2
  arm "$F" "$num" stable_core 0 MODE=main SCRIPT="$D62/stable_core.py" SC_TARGETS="$(tlist "$F" stable)" \
    OUT_SC="$out/stable_core" LAM="$lam" ROWS="$out/stable_core/stable_core*.jsonl" || rc=2
  if [ "$(unweighted_on "$F")" = 1 ]; then                       # one lambda: the WCM lambda adopted at C3
    arm "$F" "$num" unweighted 0 MODE=sweep TARGETS="$(tlist "$F" unweighted)" SWEEP_ARMS="$(lam_arm U "$lam")" \
      LAM="$lam" ROWS="$out/sweep/eval.shard*.jsonl" || rc=2
  fi
  # 073 resample (D11) on the nulls list; its random variant = the anchor null's draw 0 (<key>.d0.pt, same layout)
  arm "$F" "$num" strict 0 MODE=main SCRIPT="$D73/strict_eval.py" TARGETS="$(tlist "$F" nulls)" \
    OUT73="$out/strict" RAND_DIR="$out/harder_nulls/circuits/anchor" LAM="$lam" ROWS="$out/strict/rows*.jsonl" || rc=2
  # donor-background test (protocol v2 §12.3): every main circuit; Pythia also the unweighted arm. One R pool per
  # model (OUT/ctx: every record of this OUT), built by the first donor shard
  if [ "$(mvar DONOR "$F" "$DONOR")" = 1 ]; then
    arm "$F" "$num" donor 0 MODE=main SCRIPT="$DONOR_SCRIPT" P="$DONOR_P" DONOR_TARGETS="$(tlist "$F" main)" \
      DONOR_ARMS=main OUT_DONOR="$out/donor" DONOR_POOL="$out/donor/random_pool.pt" LAM="$lam" \
      ROWS="$out/donor/rows*.jsonl" || rc=2
    if [ "$(unweighted_on "$F")" = 1 ]; then
      arm "$F" "$num" donor_unweighted 0 MODE=main SCRIPT="$DONOR_SCRIPT" P="$DONOR_P" \
        DONOR_TARGETS="$(tlist "$F" unweighted)" DONOR_ARMS="$(lam_arm U "$lam")" OUT_DONOR="$out/donor_unweighted" \
        DONOR_POOL="$out/donor/random_pool.pt" LAM="$lam" ROWS="$out/donor_unweighted/rows*.jsonl" || rc=2
    fi
  else
    say "  $F.$num donor: switched off (DONOR=0)"
  fi
  return "$rc"
}

s_main() {   # S6 / S9: waits for the adopted lambda (outside the lock), then every arm under the fit lock
  local F=$1 num lam
  num=$(fit_stages "$F" | cut -d' ' -f3)
  verify_targets "$F" || return 1
  lam=$(get_lam "$F") || return 1
  say "  $F: main fit at lambda $lam"
  [ "$DRY_RUN" = 1 ] || echo "$lam" > "$STATE/lam_$F.used"
  with_fit_lock main_body "$F" "$num" "$lam"
}

track_body() {
  local F=$1 s4 s5 s6
  read -r s4 s5 s6 <<< "$(fit_stages "$F")"
  stage S1 "$F.S1" s1_corpus "$F" || return
  stage S2 "$F.S2" s2_native "$F" || return
  stage S3 "$F.S3" s3_datapass "$F" || return
  stage "$s4" "$F.$s4" s_records "$F" || return
  stage "$s5" "$F.$s5" s_pilot "$F" || return
  stage "$s6" "$F.$s6" s_main "$F" || return
}

track() {   # one model
  track_body "$1"
}

s10_pack() {   # C4: per-model results + stores tgz, one md5 file, md5sum -c on the volume, entry counts
  local F out ok=1 names=()
  for F in $MODELS; do
    out=$(out_dir "$F")
    local small=("$STORE_ROOT/$F"/*.json "$STORE_ROOT/$F"/*.log)
    archive "$F-S10-results" "$out" "$LOGS" "${small[@]}" "$DATA_ROOT/$F/corpus_manifest.json" \
      "$WORK/targets_${F}_"* || ok=0
    archive "$F-S10-stores" "$STORE_ROOT/$F" "$DATA_ROOT/$F" || ok=0
    names+=("$TAG-$F-S10-results.tgz" "$TAG-$F-S10-stores.tgz")
  done
  if [ "$DRY_RUN" = 1 ]; then
    echo "    $ (cd $CKPT && md5sum ${names[*]} > $TAG-final.md5) && cp to $RESULTS && md5sum -c there"
    return 0
  fi
  [ "$ok" = 1 ] || return 1
  (cd "$CKPT" && md5sum "${names[@]}" > "$TAG-final.md5") || return 1
  local n
  for n in "${names[@]}"; do echo "$n $(tar -tzf "$CKPT/$n" | wc -l)" >> "$CKPT/$TAG-final.counts"; done
  cp "$CKPT/$TAG-final.counts" "$RESULTS/" 2> /dev/null
  copy_verified "$CKPT/$TAG-final.md5" "$CKPT/$TAG-final.md5" || return 1
  (cd "$RESULTS" && md5sum -c "$TAG-final.md5") || return 1
  FINAL_VERIFIED=1
  say "C4: every final archive verified on $RESULTS ($(tr '\n' ';' < "$CKPT/$TAG-final.counts"))"
}

# ------------------------------------------------------------------------------------------------- TuringLLM track
# The TuringLLM work of the same session (protocol v2, 2026-10-25; dev-note §0.8): the 8k refit on fresh records
# (T0-T8, header) and the optional leakage-note re-score (T9). Runs in TL_ROOT (its own checkout: config.yaml =
# config-h100-triamp.yaml), TuringLLM's default path (bf16, D8), WCM_VERSION=v2 into TL_V2_OUT. Every GPU sub-step
# (the scan, the builder, the verifier, each main chunk, each sweep arm, each 083 stage) takes the fit lock
# separately through tl_with_fit_lock (public priority).
D83=experiments/083-v1-comparison
tl_stage() {   # tl_stage T<n> FUNC ARGS...: TL_STOP_AFTER gating, markers turingllm.T<n> (FROM does not apply)
  local num=$1 key=turingllm.$1 rc; shift
  if [ "${num#T}" -gt "${TL_STOP_AFTER#T}" ]; then say "$key: after TL_STOP_AFTER=$TL_STOP_AFTER; the TuringLLM track stops"; return 3; fi
  if is_done "$key"; then say "$key: done earlier (marker), skipped"; return 0; fi
  say "$key: START"
  mark "$key" start
  "$@"
  rc=$?
  if [ "$rc" = 0 ]; then mark "$key" done; say "$key: DONE"
  elif [ "$rc" = 4 ]; then
    say "$key: SKIPPED by its switch (not marked done; a rerun with the switch on runs it)"
    [ "$DRY_RUN" = 1 ] || { rm -f "$STATE/$key.start"; mark "$key" skipped; }   # not shown as running by progress*
  else say "$key: FAILED (rc $rc)"; fi
  return "$rc"
}

tl_ctx() {   # the records every v2 TuringLLM fit / score reads: the fresh 094 records (T2)
  echo "${TL_CTX_DIR:-$TL_CTX_FRESH}"
}

abspath() { case $1 in /*) echo "$1" ;; *) echo "$ROOT/$1" ;; esac; }

tl_run() {   # tl_run LOG cmd...: run in TL_ROOT (its own checkout and config.yaml); LOG stays under this chain's LOGS
  local log
  log=$(abspath "$1"); shift
  if [ "$DRY_RUN" = 1 ]; then printf '    $ (cd %s) %s>> %s 2>&1\n' "$TL_ROOT" "$(printf '%q ' "$@")" "$log"; return 0; fi
  mkdir -p "$(dirname "$log")"
  (cd "$TL_ROOT" && "$@") >> "$log" 2>&1
}

tl_run_job() {   # tl_run_job LOG JOB ROWS cmd...: tl_run (in TL_ROOT) under run_job's error gate (ROWS absolute)
  local log job=$2 rows=$3
  log=$(abspath "$1"); shift 3
  if [ -z "$rows" ]; then tl_run "$log" "$@"; return; fi
  if [ "$DRY_RUN" = 1 ]; then
    tl_run "$log" "$@"
    echo "    $ (job errors) $job: rows $rows | FAILS if its first $JOB_GATE fits all error (watched), else counted"
    return 0
  fi
  run_job "$log" "$job" "$rows" env -C "$TL_ROOT" "$@"
}

tl_launch() {   # tl_launch NAME ENV...: 062/launch.sh in TL_ROOT on the fit pool (P=TL_P unless ENV sets P); the
  #               defaults OUT=TL_V2_OUT, CTX_DIR=fresh records, WCM_VERSION; a later ENV entry overrides (env order)
  #               ENV may hold ROWS=<glob>: the job's output rows (run_job's error gate)
  local name=$1 pool gu0; shift
  split_rows "$@"
  pool=$(fit_pool)
  local extra=()
  [ "$SMOKE" = 1 ] && extra+=(STEPS="$SMOKE_STEPS" LIMIT="$LIMIT")
  say "  turingllm $name on GPUs $pool"
  gu0=$(gave_up "$LOGA/turingllm/$name")
  tl_run_job "$LOGS/turingllm/$name.launch.log" "turingllm.$name" "$JOB_ROWS" env GPUS="$pool" P="$TL_P" \
    STAGGER="$STAGGER" MERGE=0 LOGDIR="$LOGA/turingllm/$name" PY="$TL_PY" OUT="$TL_V2_OUT" CTX_DIR="$(tl_ctx)" \
    WCM_VERSION="$WCM_VERSION" "${extra[@]}" "${JOB_ARGS[@]}" bash "$D62/launch.sh" || return 1
  if [ "$DRY_RUN" != 1 ] && [ "$(gave_up "$LOGA/turingllm/$name")" -gt "$gu0" ]; then
    say "  turingllm $name: a shard gave up (see $LOGS/turingllm/$name)"; return 1
  fi
}

tl_py() {   # tl_py LOG ENV... SCRIPT: one CPU-side python call in TL_ROOT (merge / report modes)
  local log=$1; shift
  local n=$#
  local env_=("${@:1:$((n - 1))}") script=${!n}
  tl_run "$log" env PYTHONPATH=src OUT="$TL_V2_OUT" WCM_VERSION="$WCM_VERSION" "${env_[@]}" "$TL_PY" -X utf8 "$script"
}

tl_targets() {   # the TuringLLM target list of a stage: the smoke list under LIMIT, else $1 ('' = the script's default)
  if [ "$SMOKE" = 1 ]; then echo "$TL_SMOKE_TARGETS"; else echo "$1"; fi
}

tl_list() {   # tl_list NAME: a frozen 094 list (8k = targets_tl8k.txt, else targets_tl8k_<NAME>.txt); smoke: the smoke list
  if [ "$1" = 8k ]; then tl_targets "$TL_TARGETS_DIR/targets_tl8k.txt"; else tl_targets "$TL_TARGETS_DIR/targets_tl8k_$1.txt"; fi
}

TL_SUBSETS="sweep190 nulls126 case6 gallery9 stable128 gate64"

tl_quarantine() {   # T2: the quarantine (094 quarantine.py apply) from the full verify rows -> TL_LISTS: the 8k and
  #                   every subset minus the quarantined; appended to records_report.log, and LOUD lines to the events
  local s=() n log=$LOGS/turingllm/records_report.log rc
  for n in $TL_SUBSETS; do s+=(--subset "$n=$(tl_list "$n")"); done
  tl_run "$log" env PYTHONPATH=src "$TL_PY" -X utf8 "$D94/quarantine.py" apply --verify "$TL_WORK/records" \
    --targets "$(tl_list 8k)" --gate "$(tl_list gate64)" --out "$TL_LISTS" "${s[@]}"
  rc=$?
  if [ "$DRY_RUN" != 1 ]; then
    local line
    while IFS= read -r line; do say "$line"; done < <(grep -h "T2 quarantine\|WARNING\|gate targets not ok" "$log" | tail -n 8)
  fi
  return "$rc"
}

tl_use() {   # tl_use NAME: the list a stage AFTER T2 reads = the quarantine-applied copy in TL_LISTS (made by T2;
  #            rebuilt from the verify rows when missing, e.g. on a resumed pod)
  local f
  if [ "$1" = 8k ]; then f=$TL_LISTS/targets_tl8k.txt; else f=$TL_LISTS/targets_tl8k_$1.txt; fi
  if [ "$DRY_RUN" != 1 ] && [ ! -f "$f" ]; then tl_quarantine > /dev/null 2>&1; fi
  echo "$f"
}

t0_check() {   # the TuringLLM artefacts are in TL_ROOT (scripts/pod_bootstrap_global.sh ROOT=$TL_ROOT, dev-note §3.8)
  local p miss=()
  local need_root=(models data outputs/top_ctx.pt outputs/mid_ctx.pt outputs/neg_ctx.pt outputs/seq_repr.pt
                   outputs/latent_stats.pt "$D84/targets_turingllm_nulls.txt" "$D62/targets_full.txt"
                   "$D94/build_records.py" "$D86/scan.py" "$D86/rebuild.py")
  local need_abs=("$TL_CORPUS_DIR/corpus_hash.npy" "$TL_TARGETS_DIR/targets_tl8k.txt"
                  "$TL_TARGETS_DIR/targets_tl8k.manifest.json")
  local n
  for n in $TL_SUBSETS; do need_abs+=("$TL_TARGETS_DIR/targets_tl8k_$n.txt"); done
  need_abs+=("$TL_ROOT/$D94/quarantine.py")
  [ "$TL_DONOR" = 1 ] && need_root+=("$DONOR_SCRIPT")
  [ "$TL_ASRUN" = 1 ] && need_abs+=("$TL_OUT/main/circuits" "$TL_OUT/ctx" "$TL_ASRUN_TARGETS")
  if [ "$DRY_RUN" = 1 ]; then
    echo "    $ (check) $TL_ROOT: ${need_root[*]}"
    echo "      ${need_abs[*]}"
    echo "      every 094 list's md5 == targets_tl8k.manifest.json; driver.py has WCM_VERSIONS ($WCM_VERSION);" \
         "$TL_V2_OUT holds no circuits of another version; config.yaml == config-h100-triamp.yaml"
    return 0
  fi
  for p in "${need_root[@]}"; do [ -e "$TL_ROOT/$p" ] || miss+=("$TL_ROOT/$p"); done
  for p in "${need_abs[@]}"; do [ -e "$p" ] || miss+=("$p"); done
  if [ ${#miss[@]} -gt 0 ]; then say "  T0: missing ${miss[*]} (bootstrap TL_ROOT first)"; return 1; fi
  if ! grep -q "\"$WCM_VERSION\": dict(" "$TL_ROOT/$D62/driver.py" || ! grep -q WCM_VERSIONS "$TL_ROOT/$D62/driver.py"; then
    say "  T0: $TL_ROOT/$D62/driver.py has no WCM_VERSION=$WCM_VERSION in its WCM_VERSIONS table"; return 1
  fi
  if [ -f "$TL_V2_OUT/WCM_VERSION" ] && [ "$(tr -d ' \n' < "$TL_V2_OUT/WCM_VERSION")" != "$WCM_VERSION" ]; then
    say "  T0: $TL_V2_OUT is marked $(cat "$TL_V2_OUT/WCM_VERSION"), this chain runs $WCM_VERSION"; return 1
  fi
  if ! "$TL_PY" - "$TL_TARGETS_DIR" << 'EOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
man = json.loads((d / "targets_tl8k.manifest.json").read_text())
bad = [f for f, m in man["files"].items() if hashlib.md5((d / f).read_bytes()).hexdigest() != m]
print("  T0: 094 lists %s (%d files, 8k md5 %s)" % ("CHANGED: %s" % bad if bad else "match their manifest",
                                                  len(man["files"]), man["files"]["targets_tl8k.txt"]))
sys.exit(1 if bad else 0)
EOF
  then return 1; fi
  if ! cmp -s "$TL_ROOT/config.yaml" "$TL_ROOT/config-h100-triamp.yaml"; then
    if [ "$SMOKE" = 1 ]; then say "  T0 (smoke): config.yaml is not config-h100-triamp.yaml; continuing"
    else say "  T0: $TL_ROOT/config.yaml is not the 15k run's config-h100-triamp.yaml"; return 1; fi
  fi
  if [ "$SMOKE" = 1 ]; then
    mkdir -p "$TL_WORK" && head -n "$LIMIT" "$TL_ROOT/$D84/targets_turingllm_nulls.txt" > "$TL_SMOKE_TARGETS" || return 1
    say "  T0 (smoke): targets $(tr '\n' ' ' < "$TL_SMOKE_TARGETS")"
  fi
  say "  T0: 8k list $(wc -l < "$TL_TARGETS_DIR/targets_tl8k.txt") targets; WCM_VERSION $WCM_VERSION; v2 OUT $TL_V2_OUT"
  archive turingllm-T0-check "$LOGS/turingllm" 2> /dev/null || true
}

t1_scan_body() {
  local extra=()
  [ "$SMOKE" = 1 ] && extra+=(LIMIT_SHARDS="$TL_SMOKE_SHARDS")
  tl_launch scan SCRIPT="$D86/scan.py" MODE=scan P="$TL_SCAN_P" OUT_SCAN="$TL_WORK/scan" CORPUS_DIR="$TL_CORPUS_DIR" \
    TARGETS="$(scan_targets)" "${extra[@]}"
}

scan_targets() {   # the scan's targets: the 8k (smoke: the smoke list); T9's as-run 1,000 are inside the 8k
  tl_list 8k
}

t1_scan() {   # corpus scan (086: top 512 per target on the content-deduplicated corpus, the stores' ranking
  #             statistic), then merge: the strongest candidates beyond the stores' 64 slots for the fresh records
  tl_with_fit_lock t1_scan_body || return 1
  local partial=()
  [ "$SMOKE" = 1 ] && partial+=(PARTIAL=1)
  tl_py "$LOGS/turingllm/scan_merge.log" MODE=merge OUT_SCAN="$TL_WORK/scan" CORPUS_DIR="$TL_CORPUS_DIR" \
    TARGETS="$(scan_targets)" "${partial[@]}" "$D86/scan.py" || return 1
  archive turingllm-T1-scan "$TL_WORK/scan" "$LOGS/turingllm"
}

verify_gate() {   # verify_gate DIR N: N records re-checked in DIR/verify.shard*.jsonl, every one ok (no fail, no missing)
  [ "$DRY_RUN" = 1 ] && { echo "    $ (gate) $2 rows in $1/verify.shard*.jsonl, every one ok=true"; return 0; }
  "$TL_PY" -c '
import glob, json, sys
rows = [json.loads(l) for p in glob.glob(sys.argv[1] + "/verify.shard*.jsonl") for l in open(p) if l.strip()]
bad = [r for r in rows if not r.get("ok")]
print("  verify: %d records (expected %s), %d ok, %d failed or missing" % (len(rows), sys.argv[2], len(rows) - len(bad),
                                                                         len(bad)))
for r in bad[:10]:
    print("   ", r.get("seed"), r.get("missing") and "missing" or r.get("failures", [])[:3])
sys.exit(0 if rows and not bad and len({r["seed"] for r in rows}) == int(sys.argv[2]) else 1)' "$1" "$2"
}

n_lines() { if [ "$DRY_RUN" = 1 ] && [ ! -f "$1" ]; then echo "<n>"; else grep -c . "$1"; fi; }

fr_env() {   # fr_env TARGETS OUT_RB: 094 build_records.py's settings, one per line (the fresh v2 records)
  printf '%s\n' SCRIPT="$D94/build_records.py" TARGETS="$1" OUT_DIR="$TL_CTX_FRESH" \
    OUT_RB="$2" SCAN="$TL_WORK/scan/scan_top512.pt" CORPUS_DIR="$TL_CORPUS_DIR" \
    STORES_DIR="$TL_ROOT/outputs"
}

t2_body() {   # t2_body NAME MODE TARGETS OUT_RB P: one build_records.py launch (build skips records already built)
  local e=() rows=()
  mapfile -t e < <(fr_env "$3" "$4")
  [ "$2" = build ] && rows=(ROWS="$4/build.shard*.jsonl")         # verify rows are T2's own gate / quarantine
  tl_launch "$1" "${e[@]}" MODE="$2" P="$5" "${rows[@]}"
}

t2_records() {   # fresh records -> TL_CTX_FRESH (every later fit's CTX_DIR), in two steps (Daniel, 2026-10-26):
  #   1. EARLY GATE: build + verify the 64 gate targets (targets_tl8k_gate64.txt, spread over the 35 cells) first;
  #      any failure or missing record stops the TuringLLM track here (a builder bug costs minutes, not the build)
  #   2. build + verify the rest of the 8k; a later failure QUARANTINES that target (094 quarantine.py: TL_LISTS =
  #      the lists minus the quarantined, read by every later stage; quarantine.tsv packed; nothing stops)
  local log=$LOGS/turingllm/records_report.log e=()
  [ "$DRY_RUN" = 1 ] || mkdir -p "$LOGS/turingllm"
  tl_with_fit_lock t2_body records_gate_build build "$(tl_list gate64)" "$TL_WORK/records" "$TL_BUILD_P" || return 1
  tl_with_fit_lock t2_body records_gate_verify verify "$(tl_list gate64)" "$TL_WORK/records_gate" "$TL_VERIFY_P" \
    || return 1
  if ! tl_run "$log" env PYTHONPATH=src "$TL_PY" -X utf8 "$D94/quarantine.py" gate --verify "$TL_WORK/records_gate" \
      --gate "$(tl_list gate64)"; then
    [ "$DRY_RUN" = 1 ] || grep -h "early gate\|GATE FAIL" "$log" | tail -n 12 | while IFS= read -r l; do say "$l"; done
    say "  T2: the EARLY GATE failed (64 gate targets); the TuringLLM track stops before the full build"
    archive turingllm-T2-gate-failed "$TL_WORK/records_gate" "$TL_WORK/records" "$LOGS/turingllm"
    return 1
  fi
  [ "$DRY_RUN" = 1 ] || say "  T2: $(grep -h "early gate" "$log" | tail -n 1 | sed 's/^ *//')"
  tl_with_fit_lock t2_body records_build build "$(tl_list 8k)" "$TL_WORK/records" "$TL_BUILD_P" || return 1
  tl_with_fit_lock t2_body records_verify verify "$(tl_list 8k)" "$TL_WORK/records" "$TL_VERIFY_P" || return 1
  mapfile -t e < <(fr_env "$(tl_list 8k)" "$TL_WORK/records")
  tl_py "$log" "${e[@]:1}" MODE=report "$D94/build_records.py" || say "  (records report failed; the rows are kept)"
  tl_quarantine || { say "  T2: a gate target is not ok in the full verify; the TuringLLM track stops"; return 1; }
  archive turingllm-T2-records "$TL_CTX_FRESH" "$TL_WORK/records" "$TL_WORK/records_gate" "$TL_LISTS" \
    "$LOGS/turingllm"
}

tl_chunks() {   # the 8k list cut into TL_MAIN_CHUNKS files (line i -> chunk i mod K: every chunk spans every layer);
  #               prints their paths. Deterministic from the frozen list, so a resumed chain cuts the same chunks
  local src k i
  src=$(tl_use 8k); k=$TL_MAIN_CHUNKS
  for ((i = 0; i < k; i++)); do
    local f=$TL_WORK/chunks/main.$i.of$k.txt
    if [ "$DRY_RUN" != 1 ]; then
      mkdir -p "$TL_WORK/chunks"
      awk -v k="$k" -v i="$i" 'NF && (NR - 1) % k == i' "$src" > "$f.tmp" && mv "$f.tmp" "$f" || return 1
    fi
    echo "$f"
  done
}

t3_chunk_body() { tl_launch "main_c$2" MODE=main TARGETS="$1" ROWS="$TL_V2_OUT/main/status.shard*.jsonl"; }
t3_alpha1_body() {
  tl_launch alpha1 MODE=main SCRIPT="$D62/rescore_alpha1.py" A1_TARGETS="$(tl_use 8k)" OUT_A1="$TL_V2_OUT/alpha1" \
    ROWS="$TL_V2_OUT/alpha1/alpha1*.jsonl"
}

t3_main() {   # the v2 main fits of the 8k (fit -> held-out strongest + mid-band scores -> 056 specificity, the driver's
  #             MODE=main), one fit-lock holder per chunk, then alpha = 1 (score only)
  local c j=0 chunks=()
  mapfile -t chunks < <(tl_chunks) || return 1
  [ ${#chunks[@]} -gt 0 ] || return 1
  [ "$DRY_RUN" = 1 ] && echo "    $ (split) $(tl_use 8k) ->${#chunks[@]} chunk lists (line i -> chunk i mod $TL_MAIN_CHUNKS)"
  for c in "${chunks[@]}"; do
    tl_with_fit_lock t3_chunk_body "$c" "$j" || return 1
    j=$((j + 1))
  done
  tl_with_fit_lock t3_alpha1_body || say "  T3: alpha = 1 failed (the main fits are kept; rerun the chain to resume)"
  archive turingllm-T3-main "$TL_V2_OUT/main" "$TL_V2_OUT/alpha1" "$TL_V2_OUT/WCM_VERSION" "$TL_WORK/chunks" \
    "$LOGS/turingllm"
}

t4_arm_body() {
  tl_launch "sweep_$1" MODE=sweep TARGETS="$(tl_use sweep190)" SWEEP_ARMS="$1" ROWS="$TL_V2_OUT/sweep/eval.shard*.jsonl"
}

t4_sweep() {   # the 190-target lambda sweep refit (062's arms: WCM 8 lambda incl. the 3 weak ones, unweighted 5), one
  #              launch per arm; W_0.001 reuses the T3 main circuit (the driver's identical-config rule)
  local a
  for a in ${TL_SWEEP_ARMS//,/ }; do
    tl_with_fit_lock t4_arm_body "$a" || return 1
  done
  archive turingllm-T4-sweep "$TL_V2_OUT/sweep" "$LOGS/turingllm"
}

t5_stable_body() {
  tl_launch stable_core MODE=main SCRIPT="$D62/stable_core.py" SC_TARGETS="$(tl_use stable128)" \
    OUT_SC="$TL_V2_OUT/stable_core" ROWS="$TL_V2_OUT/stable_core/stable_core*.jsonl"
}

t5_stable() {   # stable core on 128 of the 8k (stratified, 094 make_targets.py)
  tl_with_fit_lock t5_stable_body || return 1
  archive turingllm-T5-stable "$TL_V2_OUT/stable_core" "$LOGS/turingllm"
}

t6_nulls_body() {   # the public scheme; the DAN-17 references are ALL refit under v2 (RN_DIR empty: the stored ones
  #                   are v1 fits)
  local ne=()
  mapfile -t ne < <(NULLS_SUB_N=$TL_NULLS_SUB_N NULLS_REFS=wcm,dan17 NULLS_DAN17_REFIT=1 \
    nulls_env "$TL_WORK/harder_nulls" "$(tl_use nulls126)" turingllm)
  tl_launch harder_nulls MODE=main "${ne[@]}" RN_DIR= \
    ROWS="$TL_WORK/harder_nulls/nulls*.jsonl,$TL_WORK/harder_nulls/profile*.jsonl"
}

t6_nulls() {   # DAN-181 on TuringLLM: the frozen 126 panel, against the v2 production circuits (T3)
  tl_with_fit_lock t6_nulls_body || return 1
  archive turingllm-T6-nulls "$TL_WORK/harder_nulls" "$LOGS/turingllm"
}

t7_stage_body() {   # t7_stage_body STAGE: one 083 stage through run_pod.sh on the current fit pool; the WCM / U
  #                   references (and the driver OUT) = the v2 refit (REF_RUN = TL_V2_OUT: T3 main + T4 sweep)
  local s=$1 pool extra=()
  pool=$(fit_pool)
  if [ "$SMOKE" = 1 ]; then                                    # run_local.sh SMOKE=1's budgets, 1 target
    extra+=(LIMIT=1 IG_STEPS=2 BISECT_STEPS=2 STEPS=4 SIZES=W_0.001,W_0.004 REF_ARMS=W_0.001,U_0.001 N_EXACT=1)
  fi
  say "  turingllm dan159 $s on GPUs $pool, CTX_DIR $(tl_ctx), REF_RUN $TL_V2_OUT"
  local gu0
  gu0=$(gave_up "$TL_WORK/dan159/logs")
  local rows=$TL_WORK/dan159/$s/rows*.jsonl                    # every 083 stage writes <stage>/rows[.shard<i>].jsonl
  [ "$s" = table ] && rows=                                     # table2.py: a report, no fits
  tl_run_job "$LOGS/turingllm/dan159_$s.log" "turingllm.dan159_$s" "$rows" env GPUS="$pool" P="$TL_083_P" \
    STAGGER="$STAGGER" STAGES="$s" PY="$TL_PY" WCM_VERSION="$WCM_VERSION" CTX_DIR="$(tl_ctx)" REF_RUN="$TL_V2_OUT" \
    OUT="$TL_V2_OUT" OUT083="$TL_WORK/dan159" TARGETS="$(tl_use sweep190)" "${extra[@]}" bash "$D83/run_pod.sh" \
    || return 1
  if [ "$DRY_RUN" != 1 ] && [ "$(gave_up "$TL_WORK/dan159/logs")" -gt "$gu0" ]; then
    say "  turingllm dan159 $s: a shard gave up (see $TL_WORK/dan159/logs)"; return 1
  fi
}

t7_dan159() {   # DAN-159 / DAN-43 (083 stages TL_DAN159_STAGES, default ref attr fitted: the WCM / U references
  #             re-scored, the IG / AtP rankings, DAN-43's fitted amplitudes) on the 190 sweep targets, then table; lock
  #             per stage. TL_DAN159=0 skips it.
  local s
  if [ "$TL_DAN159" != 1 ]; then return 4; fi
  for s in $TL_DAN159_STAGES; do
    tl_with_fit_lock t7_stage_body "$s" || return 1
  done
  t7_stage_body table || say "  (table2.py failed; the rows are kept)"
  archive turingllm-T7-dan159 "$TL_WORK/dan159" "$LOGS/turingllm"
}

t8_cases() {   # the case-study (6) and random-gallery (9) circuits are T3 main fits (they are in the 8k): check them
  #              and pack them with their fresh records, for the case-study / gallery redo (063 / 072, local)
  #              A quarantined case / gallery target (T2) is reported, not fitted, and not a failure
  local k miss=() quar=() keys=() files=() ok=()
  if [ "$DRY_RUN" = 1 ]; then
    echo "    $ (check) $TL_V2_OUT/main/circuits/<key>.pt for every key of $(tl_use case6) $(tl_use gallery9)" \
         "(quarantined ones: reported)"
    archive turingllm-T8-cases "$TL_V2_OUT/main/circuits/<case6+gallery9>.pt" "$TL_CTX_FRESH/<case6+gallery9>.pt"
    return 0
  fi
  mapfile -t keys < <(cat "$(tl_list case6)" "$(tl_list gallery9)" | awk 'NF && !seen[$0]++')
  mapfile -t ok < <(cat "$(tl_use case6)" "$(tl_use gallery9)")
  for k in "${keys[@]}"; do
    if ! printf '%s\n' "${ok[@]}" | grep -qxF "$k"; then quar+=("$k")
    elif [ -f "$TL_V2_OUT/main/circuits/$k.pt" ]; then files+=("$TL_V2_OUT/main/circuits/$k.pt" "$TL_CTX_FRESH/$k.pt")
    else miss+=("$k"); fi
  done
  local note=""
  [ ${#miss[@]} -gt 0 ] && note="; missing ${miss[*]}"
  [ ${#quar[@]} -gt 0 ] && note="$note; !!! QUARANTINED in T2 (no record, no fit): ${quar[*]}"
  say "  T8: ${#keys[@]} case / gallery targets, $(( ${#files[@]} / 2 )) with a v2 circuit$note"
  [ ${#files[@]} -gt 0 ] || return 1
  archive turingllm-T8-cases "${files[@]}" "$(tl_list case6)" "$(tl_list gallery9)" "$TL_LISTS"
}

asrun_rb_env() {   # T9: 086 rebuild.py over the as-run subset (the fixed 15k v1 circuits' records), from the 8k scan
  printf '%s\n' SCRIPT="$D86/rebuild.py" SCAN="$TL_WORK/scan/scan_top512.pt" CTX_IN="$TL_OUT/ctx" \
    CTX_OUT="$TL_WORK/ctx_v2_asrun1000" CORPUS_DIR="$TL_CORPUS_DIR" STORES_DIR="$TL_ROOT/outputs" \
    OUT_RB="$TL_WORK/asrun_rebuild" TARGETS="$(tl_targets "$TL_ASRUN_TARGETS")"
}

t9_build_body() { local e=(); mapfile -t e < <(asrun_rb_env); tl_launch asrun_build "${e[@]}" MODE=build P="$TL_BUILD_P"; }
t9_verify_body() { local e=(); mapfile -t e < <(asrun_rb_env); tl_launch asrun_verify "${e[@]}" MODE=verify P="$TL_VERIFY_P"; }
t9_score_body() {   # t9_score_body CTX NAME: the v1 circuits (OUT = the 15k run, WCM_VERSION=v1: scoring only)
  tl_launch "asrun_$2" SCRIPT="$D86/rescore.py" JOBS=main OUT="$TL_OUT" WCM_VERSION=v1 CTX_DIR="$1" \
    OUT_RS="$TL_WORK/rescore_asrun1000_$2" TARGETS="$(tl_targets "$TL_ASRUN_TARGETS")"
}

t9_asrun() {   # OPTIONAL (TL_ASRUN=1): the paper's leakage note. The 1,000 as-run circuits (fixed, v1) scored on their
  #              as-run records and on 086-rebuilt held-out sets, same machine: the paired leakage effect
  if [ "$TL_ASRUN" != 1 ]; then return 4; fi
  tl_with_fit_lock t9_build_body || return 1
  tl_with_fit_lock t9_verify_body || return 1
  verify_gate "$TL_WORK/asrun_rebuild" "$(n_lines "$(tl_targets "$TL_ASRUN_TARGETS")")" \
    || { say "  T9: rebuilt as-run records failed verification; not scored"; return 1; }
  tl_with_fit_lock t9_score_body "$TL_OUT/ctx" asrun || return 1
  tl_with_fit_lock t9_score_body "$TL_WORK/ctx_v2_asrun1000" ctx_v2 || return 1
  tl_py "$LOGS/turingllm/asrun_report.log" REPORT=1 OUT="$TL_OUT" WCM_VERSION=v1 CTX_DIR="$TL_WORK/ctx_v2_asrun1000" \
    OUT_RS="$TL_WORK/rescore_asrun1000_ctx_v2" ASRUN_RS="$TL_WORK/rescore_asrun1000_asrun" \
    ASRUN_TARGETS="$(tl_targets "$TL_ASRUN_TARGETS")" "$D86/rescore.py" \
    || say "  (as-run report failed; the rows are kept, rerun REPORT=1 by hand)"
  archive turingllm-T9-asrun "$TL_WORK/asrun_rebuild" "$TL_WORK/rescore_asrun1000_asrun" \
    "$TL_WORK/rescore_asrun1000_ctx_v2" "$LOGS/turingllm"
}

t10_nor_body() {   # the without-R arm: TL_NOR_VERSION (no R) at the production lambda, its own OUT (version-marked)
  tl_launch sweep_noR MODE=sweep TARGETS="$(tl_use sweep190)" SWEEP_ARMS="$TL_NOR_ARM" OUT="$TL_NOR_OUT" \
    WCM_VERSION="$TL_NOR_VERSION" ROWS="$TL_NOR_OUT/sweep/eval.shard*.jsonl"
}

has_r() { [ "$WCM_VERSION" = v2cr ] || [ "$WCM_VERSION" = v3 ] || [ "$WCM_VERSION" = v4 ]; }   # fits with the R term

t10_nor() {   # the appendix ablation (protocol v2 §12.3): the 190 sweep targets fitted WITHOUT R (protocol v2), a paired
  #             comparison with the production sweep's W_0.001 arm (= the T3 main circuits). Only under v2cr / v3 / v4.
  #             Under v4 the arm is v4nr's W_0.001 (v4 without R). Under v3 the default arm W_0.001_r0.1 is v2 at v3's
  #             rank-keep 1e-1, so the pair differs in R alone (Daniel 2026-11-01); TL_NOR_ARM=W_0.001 gives the
  #             previous production algorithm instead (rank-keep 3e-3)
  if [ "$TL_NOR" != 1 ]; then return 4; fi
  if ! has_r; then
    say "  T10: WCM_VERSION=$WCM_VERSION has no R term (the sweep's $TL_NOR_ARM arm is already without R)"; return 4
  fi
  tl_with_fit_lock t10_nor_body || return 1
  archive turingllm-T10-noR "$TL_NOR_OUT/sweep" "$TL_NOR_OUT/WCM_VERSION" "$LOGS/turingllm"
}

donor_env() {   # donor_env OUT_DONOR TARGETS ARMS: the donor CLI's settings, one per line (ROWS = its output rows)
  printf '%s\n' SCRIPT="$DONOR_SCRIPT" P="$DONOR_P" DONOR_TARGETS="$2" DONOR_ARMS="$3" OUT_DONOR="$1" \
    DONOR_POOL="$TL_DONOR_POOL" ROWS="$1/rows*.jsonl"
}

t11_chunk_body() { local e=(); mapfile -t e < <(donor_env "$TL_V2_OUT/donor/main" "$1" main); tl_launch "donor_main_c$2" "${e[@]}"; }

t11_donor_main() {   # the donor-background test of every 8k main circuit (rows TL_V2_OUT/donor/main), one fit-lock
  #                    holder per chunk (T3's chunk lists); the first shard builds the R pool (TL_DONOR_POOL)
  if [ "$TL_DONOR" != 1 ] || [ "$TL_DONOR_MAIN" != 1 ]; then return 4; fi
  local c j=0 chunks=()
  mapfile -t chunks < <(tl_chunks) || return 1
  [ ${#chunks[@]} -gt 0 ] || return 1
  for c in "${chunks[@]}"; do
    tl_with_fit_lock t11_chunk_body "$c" "$j" || return 1
    j=$((j + 1))
  done
  archive turingllm-T11-donor-main "$TL_V2_OUT/donor/main" "$TL_DONOR_POOL" "$LOGS/turingllm"
}

t12_sweep_body() { local e=(); mapfile -t e < <(donor_env "$TL_V2_OUT/donor/sweep" "$(tl_use sweep190)" "$TL_SWEEP_ARMS"); tl_launch donor_sweep "${e[@]}"; }
t12_nor_body() {
  local e=()
  mapfile -t e < <(donor_env "$TL_NOR_OUT/donor/sweep" "$(tl_use sweep190)" "$TL_NOR_ARM")
  tl_launch donor_sweep_noR "${e[@]}" OUT="$TL_NOR_OUT" WCM_VERSION="$TL_NOR_VERSION"
}

t12_donor_sweep() {   # the donor-background test of every sweep arm (WCM + U_, under v2cr with R) in ONE launch (the
  #                     per-target work is shared by the arms), then the T10 without-R arm (its own OUT)
  if [ "$TL_DONOR" != 1 ] || [ "$TL_DONOR_SWEEP" != 1 ]; then return 4; fi
  tl_with_fit_lock t12_sweep_body || return 1
  local what=("$TL_V2_OUT/donor/sweep")
  if is_done turingllm.T10 || { [ "$DRY_RUN" = 1 ] && has_r && [ "$TL_NOR" = 1 ]; }; then
    tl_with_fit_lock t12_nor_body || return 1
    what+=("$TL_NOR_OUT/donor")
  else
    say "  T12: no without-R arm to score (T10 not done)"
  fi
  archive turingllm-T12-donor-sweep "${what[@]}" "$LOGS/turingllm"
}

tl_track() {   # the v2 refit (T0-T8), the optional leakage note (T9), then the donor-background test (T10-T12)
  tl_stage T0 t0_check || return
  tl_stage T1 t1_scan || return
  tl_stage T2 t2_records || return
  tl_stage T3 t3_main || return
  tl_stage T4 t4_sweep || return
  tl_stage T5 t5_stable || return
  tl_stage T6 t6_nulls || return
  tl_stage T7 t7_dan159 || [ $? = 4 ] || return
  tl_stage T8 t8_cases || return
  tl_stage T9 t9_asrun || [ $? = 4 ] || return
  tl_stage T10 t10_nor || [ $? = 4 ] || return
  tl_stage T11 t11_donor_main || [ $? = 4 ] || return
  tl_stage T12 t12_donor_sweep || [ $? = 4 ] || return
  return 0
}

arena_a_stage() {   # ARENA_A=1 (off by default; exp 107 A11): arena A's pod stage, 107/arena_a/run_pod.sh P0-P7, on
  #                   the whole pod AFTER S10 and the TuringLLM track; its own resumable stages, logs and markers under
  #                   its AA_WORK; corpus source = this chain's gpt2-small S1 corpus. Not started when the main chain
  #                   failed (ARENA_A_ANYWAY=1 starts it anyway). run_pod.sh rc 2 = downloads / env builds to approve
  local aa=experiments/107-true-baselines/arena_a/run_pod.sh arc gpus=${ARENA_A_GPUS:-0,1,2,3,4,5,6,7}
  [ "$SMOKE" = 1 ] && gpus=${ARENA_A_GPUS:-0}
  if { [ "$rc" != 0 ] || [ "$tl_rc" != 0 ]; } && [ "${ARENA_A_ANYWAY:-0}" != 1 ]; then
    say "arena_a: not started (main chain rc $rc, TuringLLM rc $tl_rc; ARENA_A_ANYWAY=1 starts it anyway)"; return 1
  fi
  if is_done arena_a; then say "arena_a: done earlier (marker), skipped"; return 0; fi
  say "arena_a: START ($aa on GPUs $gpus)"
  mark arena_a start
  local envs=(AA_GPUS="$gpus" AA_DATA_SRC="$DATA_ROOT/gpt2-small" PY="$PY" CKPT="$CKPT" RESULTS="$RESULTS" TAG="$TAG")
  # its time limit (run_pod.sh TIME LIMIT; unset = its default AA_MAX_H=auto, 15.75 h on 8 GPUs): soft stop, hard kill
  [ -n "${ARENA_A_STOP_EPOCH:-}" ] && envs+=(STOP_EPOCH="$ARENA_A_STOP_EPOCH")
  [ -n "${ARENA_A_MAX_H:-}" ] && envs+=(AA_MAX_H="$ARENA_A_MAX_H")
  [ -n "${ARENA_A_HARD_GRACE:-}" ] && envs+=(AA_HARD_GRACE="$ARENA_A_HARD_GRACE")
  [ "$SMOKE" = 1 ] && envs+=(AA_LIMIT="$LIMIT" AA_N_SEQ=0 AA_STEPS="$SMOKE_STEPS" AA_WORK="$WORK/arena_a")
  if [ "$DRY_RUN" = 1 ]; then
    run - env "${envs[@]}" bash "$aa"
    env "${envs[@]}" DRY_RUN=1 bash "$aa" | sed 's/^/  /'; arc=${PIPESTATUS[0]}
  else
    run "$LOGS/arena_a/run_pod.log" env "${envs[@]}" bash "$aa"; arc=$?
  fi
  case $arc in
    0) mark arena_a done; say "arena_a: DONE" ;;
    2) say "arena_a: WAITING for approval of its downloads / env builds ($LOGS/arena_a/run_pod.log); rerun the chain after" ;;
    4) say "arena_a: STOPPED by its time limit (soft / hard stop; P7 packed what finished; $LOGS/arena_a/run_pod.log; a rerun resumes it)" ;;
    *) say "arena_a: FAILED (rc $arc; $LOGS/arena_a/run_pod.log; a rerun resumes it)" ;;
  esac
  return "$arc"
}

arena_d_stage() {   # ARENA_D=1 (off by default; exp 107 D9): arena D's pod stage, 107/arena_d/pod_stage_d.sh P0-P5, on
  #                   the whole pod AFTER arena A (whatever arena A's result: the two are independent); its own
  #                   resumable stages, logs and markers under its DW_WORK. Not started when the main chain failed
  #                   (ARENA_D_ANYWAY=1 starts it anyway). pod_stage_d.sh rc 2 = downloads / env builds to approve
  local ad=experiments/107-true-baselines/arena_d/pod_stage_d.sh drc gpus=${ARENA_D_GPUS:-0,1,2,3,4,5,6,7}
  [ "$SMOKE" = 1 ] && gpus=${ARENA_D_GPUS:-0}
  if { [ "$rc" != 0 ] || [ "$tl_rc" != 0 ]; } && [ "${ARENA_D_ANYWAY:-0}" != 1 ]; then
    say "arena_d: not started (main chain rc $rc, TuringLLM rc $tl_rc; ARENA_D_ANYWAY=1 starts it anyway)"; return 1
  fi
  if is_done arena_d; then say "arena_d: done earlier (marker), skipped"; return 0; fi
  say "arena_d: START ($ad on GPUs $gpus)"
  mark arena_d start
  local envs=(DW_GPUS="$gpus" PY="$PY" CKPT="$CKPT" RESULTS="$RESULTS" TAG="$TAG")
  [ -n "${ARENA_D_STOP_EPOCH:-}" ] && envs+=(STOP_EPOCH="$ARENA_D_STOP_EPOCH")
  [ "$SMOKE" = 1 ] && envs+=(DW_LAMS=1e-3 DW_STEPS="$SMOKE_STEPS" DW_AG_LIMIT="$LIMIT" DW_WORK="$WORK/arena_d")
  if [ "$DRY_RUN" = 1 ]; then
    run - env "${envs[@]}" bash "$ad"
    env "${envs[@]}" DRY_RUN=1 bash "$ad" | sed 's/^/  /'; drc=${PIPESTATUS[0]}
  else
    run "$LOGS/arena_d/pod_stage_d.log" env "${envs[@]}" bash "$ad"; drc=$?
  fi
  case $drc in
    0) mark arena_d done; say "arena_d: DONE" ;;
    2) say "arena_d: WAITING for approval of its downloads / env builds ($LOGS/arena_d/pod_stage_d.log); rerun the chain after" ;;
    *) say "arena_d: FAILED (rc $drc; $LOGS/arena_d/pod_stage_d.log; a rerun resumes it)" ;;
  esac
  return "$drc"
}

# ------------------------------------------------------------------------------------------------- main
say "082 chain | $(git --no-pager log -1 --format='%h %s' 2> /dev/null) | WCM_VERSION=$WCM_VERSION | MODELS=$MODELS"\
" FROM=$FROM STOP_AFTER=$STOP_AFTER"\
" | WORK=$WORK RESULTS=$RESULTS | P=$P FIT_GPUS=$FIT_GPUS$([ "$SMOKE" = 1 ] && echo " | SMOKE LIMIT=$LIMIT")"\
"$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
for m in $MODELS; do
  case $m in pythia-70m|gpt2-small) ;; *) echo "unknown model $m" >&2; exit 2 ;; esac
done
if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$STATE" "$LOGS" || exit 1
  rm -f "$STATE"/want.*                                  # stale lock-priority flags of an earlier (killed) chain
  if [ "$SMOKE" != 1 ]; then
    mkdir -p "$RESULTS" || { echo "RESULTS=$RESULTS is not writable" >&2; exit 1; }
    if [ "$MPS" = 1 ] && command -v nvidia-cuda-mps-control > /dev/null && ! pgrep -f nvidia-cuda-mps-control > /dev/null; then
      nvidia-cuda-mps-control -d && say "CUDA MPS daemon started"
    fi
  fi
fi

tl_rc=0
tl_wait() {   # the TuringLLM track's result: 0 = finished (or stopped by TL_STOP_AFTER), else failed / not run
  if [ "$TL_TRACK" != 1 ]; then tl_rc=0; return; fi
  if [ -n "${tl_pid:-}" ]; then wait "$tl_pid"; tl_rc=$?; tl_pid=; fi
  [ "$tl_rc" = 3 ] && tl_rc=0
  [ "$tl_rc" = 0 ] || say "TuringLLM track FAILED or incomplete (rc $tl_rc; see $LOGS/chain.events and $LOGS/turingllm)"
}
if [ "$DRY_RUN" = 1 ]; then
  for m in $MODELS; do track "$m"; done                 # sequential, so the printed plan stays readable
  if [ "$TL_TRACK" = 1 ]; then tl_track; tl_rc=$?; fi
else
  pids=()
  if [ "$TL_TRACK" = 1 ]; then                           # its GPU stages queue on the same fit lock
    tl_track &
    tl_pid=$!
  fi
  for m in $MODELS; do
    track "$m" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
fi

if [ "$DRY_RUN" != 1 ] && allowed S10; then
  for m in $MODELS; do
    read -r _ _ s6 <<< "$(fit_stages "$m")"
    if ! is_done "$m.$s6.main"; then
      say "S10 needs $m.$s6.main (the main fit) done; not packing (fix the failed stage, then rerun the chain)"
      tl_wait
      exit 1
    fi
  done
  # arms finished after an earlier pack (a rerun resumed them): pack again (the TuringLLM track packs its own stages)
  if is_done S10 && [ -n "$(find "$STATE" -name '*.done' ! -name 'turingllm.*' -newer "$STATE/S10.done" 2> /dev/null)" ]; then
    rm -f "$STATE/S10.done"
  fi
fi
stage S10 S10 s10_pack
rc=$?
[ "$rc" = 3 ] && { say "stopped after $STOP_AFTER as asked"; rc=0; }
if [ -s "$STATE/failed.list" ]; then say "arms that failed (rerun to resume): $(sort -u "$STATE/failed.list" | tr '\n' ' ')"; fi
tl_wait                                                   # D9: the last verified backup may be the TuringLLM track's
arena_rc=0                                                # post-main exp 107 stages (off by default): arena A, then
if [ "${ARENA_A:-0}" = 1 ]; then arena_a_stage; arena_rc=$?; fi   # then arena D (D9), whatever arena A's result
if [ "${ARENA_D:-0}" = 1 ]; then arena_d_stage || arena_rc=1; fi
if [ "$STOP_POD" = 1 ]; then
  # the arenas do not gate the stop (D9, 2026-11-05): a failed arena must not keep an idle pod up. Each arena packs its
  # own results to RESULTS (arena D also after a failure); a failed arena's logs go to RESULTS best-effort first
  if [ "$rc" = 0 ] && [ "$FINAL_VERIFIED" = 1 ] && [ "$tl_rc" = 0 ]; then
    if [ "$arena_rc" != 0 ]; then
      say "STOP_POD=1: an exp 107 arena failed (rc above); its logs to $RESULTS/$TAG-arena-logs.tgz (best effort), then the stop"
      run - tar -czf "$RESULTS/$TAG-arena-logs.tgz" --ignore-failed-read "$LOGS/arena_a" "$LOGS/arena_d" \
        experiments/107-true-baselines/arena_a/out/pod/logs experiments/107-true-baselines/arena_d/out/pod/logs
    fi
    say "STOP_POD=1, C4 verified and the TuringLLM track (if on) packed: stopping the pod"
    run - runpodctl stop pod "${RUNPOD_POD_ID:-}"
  elif [ "$DRY_RUN" = 1 ]; then
    echo "    $ runpodctl stop pod \$RUNPOD_POD_ID   # only after C4 verified (and the TuringLLM track packed) in this run"
  else
    say "STOP_POD=1 but the final verification did not pass in this run (or the TuringLLM track did not finish): the pod stays up"
  fi
fi
[ "$rc" = 0 ] && [ "$tl_rc" != 0 ] && rc=1
[ "$rc" = 0 ] && [ "$arena_rc" != 0 ] && rc=1
say "chain finished (rc $rc)"
exit "$rc"
