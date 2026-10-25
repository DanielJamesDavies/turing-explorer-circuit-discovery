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
# Third track (TL_TRACK=1), TuringLLM, in its own checkout TL_ROOT (scripts/pod_bootstrap_global.sh ROOT=...):
#   T0 artefact check -> T1 held-out rebuild corpus scan (086 scan.py, all 15k targets) -> T2 rebuild + independent
#   verify + report (ctx_v2; no source-file rule; contrast held-out deduplicated) -> T3 re-score on ctx_v2 (main
#   strongest + mid + specificity, the 190 sweep arms, alpha = 1) + the AS-RUN re-score of the stratified 1,000
#   subset (paired leakage effect) + report -> T4 DAN-181 nulls on ctx_v2 (126 panel, same scheme; DAN-17
#   references: 65 stored + 61 refit) -> T5 DAN-159 (083 run_pod.sh, every stage, on ctx_v2).
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
#   LIMIT=N           smoke: WORK=082/smoke, the 069 smoke stores, target lists cut to their first N, STEPS=SMOKE_STEPS
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
#   TuringLLM track: TL_TRACK [0]  TL_STOP_AFTER [T5]  TL_ROOT [/root/turing-tl; smoke: this repo]
#                     TL_PY [TL_ROOT/.venv/bin/python if present, else PY]
#                     TL_OUT [TL_ROOT/062/out_full/out]  TL_WORK [TL_ROOT/082/out_turingllm; smoke: WORK/out_turingllm]
#                     TL_CTX_REBUILT [TL_OUT/ctx_v2; smoke: TL_WORK/ctx_v2]  TL_CTX_DIR [forces one; default:
#                     TL_CTX_REBUILT once T2 is done, else TL_OUT/ctx]  TL_CORPUS_DIR [TL_ROOT/085/out]  TL_P [P]
#                     TL_SCAN_P [1] TL_BUILD_P [3] TL_VERIFY_P [6] (processes per GPU; verify is CPU-bound, the
#                     rebuild ~2.6 GB RAM per process)  TL_083_P [4]
#                     TL_NULLS_SUB_N [32] (the first 32 of the 126 panel: 3-4 per band x kind)
#   STOP_POD=1 also requires the TuringLLM track (when on) to have finished and packed (D9: final verified backup)
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
#   083/run_pod.sh STAGES (ref,attr,fitted,exact,ge,table), CTX_DIR, OUT083, GPUS, P (+ LIMIT / smoke budgets)
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
DATA_ROOT=${DATA_ROOT:-data_public}
STORE_ROOT=${STORE_ROOT:-outputs_public}
if [ "$SMOKE" = 1 ]; then
  FROM=${FROM:-S4}; WORK=${WORK:-$E/smoke}; P=${P:-1}; FIT_GPUS=${FIT_GPUS:-0}; MPS=${MPS:-0}
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
TL_STOP_AFTER=${TL_STOP_AFTER:-T5}
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
if [ "$SMOKE" = 1 ]; then TL_CTX_REBUILT=${TL_CTX_REBUILT:-$TL_WORK/ctx_v2}   # never 062/out_full/out/ctx_v2 in a smoke
else TL_CTX_REBUILT=${TL_CTX_REBUILT:-$TL_OUT/ctx_v2}; fi                     # 086 rebuild.py's output (the only variant)
TL_ASRUN_TARGETS=${TL_ASRUN_TARGETS:-$TL_ROOT/$D86/targets_asrun1000.txt}    # frozen stratified 1,000 (086 subset.py)
TL_SMOKE_TARGETS=$TL_WORK/targets_smoke.txt                                  # smoke: the first LIMIT of the 084 panel
TL_SMOKE_SHARDS=${TL_SMOKE_SHARDS:-6}                                        # smoke: corpus shards scanned
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

launch() {   # launch MODEL NAME ENV...: 062/launch.sh on the current fit pool; 1 if any shard gave up in this launch
  local F=$1 name=$2 pool gu0; shift 2
  pool=$(fit_pool)
  local logdir=$LOGS/$F/$name
  local extra=() ctx
  [ "$SMOKE" = 1 ] && extra+=(STEPS="$SMOKE_STEPS")
  ctx=$(mvar CTX_DIR "$F" "${CTX_DIR:-}")
  [ -n "$ctx" ] && extra+=(CTX_DIR="$ctx")                      # unset: the driver's default (OUT/ctx)
  say "  $F $name on GPUs $pool (P=$P)"
  gu0=$(gave_up "$logdir")
  run "$LOGS/$F/$name.launch.log" env GPUS="$pool" P="$P" STAGGER="$STAGGER" MERGE=0 LOGDIR="$logdir" PY="$PY" \
    FAMILY="$F" DATA="$DATA_ROOT/$F" STORES="$STORE_ROOT/$F" OUT="$(out_dir "$F")" SAE_PRELOAD="$SAE_PRELOAD" \
    "${extra[@]}" "$@" bash "$D62/launch.sh" || return 1
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
  launch "$1" pilot MODE=sweep TARGETS="$(tlist "$1" pilot)" SWEEP_ARMS="$2"
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

main_body() {   # under the fit lock: main -> alpha = 1 -> harder nulls -> stable core -> (unweighted) -> 073
  local F=$1 num=$2 lam=$3 out rc=0
  local ne=()
  out=$(out_dir "$F")
  arm "$F" "$num" main 1 MODE=main TARGETS="$(tlist "$F" main)" LAM="$lam" || return 1
  arm "$F" "$num" alpha1 0 MODE=main SCRIPT="$D62/rescore_alpha1.py" A1_TARGETS="$(tlist "$F" main)" \
    OUT_A1="$out/alpha1" LAM="$lam" || rc=2
  mapfile -t ne < <(nulls_env "$out/harder_nulls" "$(tlist "$F" nulls)" "$F")
  arm "$F" "$num" harder_nulls 0 MODE=main "${ne[@]}" RN_DIR= LAM="$lam" || rc=2
  arm "$F" "$num" stable_core 0 MODE=main SCRIPT="$D62/stable_core.py" SC_TARGETS="$(tlist "$F" stable)" \
    OUT_SC="$out/stable_core" LAM="$lam" || rc=2
  if [ "$(unweighted_on "$F")" = 1 ]; then                       # one lambda: the WCM lambda adopted at C3
    arm "$F" "$num" unweighted 0 MODE=sweep TARGETS="$(tlist "$F" unweighted)" SWEEP_ARMS="$(lam_arm U "$lam")" \
      LAM="$lam" || rc=2
  fi
  # 073 resample (D11) on the nulls list; its random variant = the anchor null's draw 0 (<key>.d0.pt, same layout)
  arm "$F" "$num" strict 0 MODE=main SCRIPT="$D73/strict_eval.py" TARGETS="$(tlist "$F" nulls)" \
    OUT73="$out/strict" RAND_DIR="$out/harder_nulls/circuits/anchor" LAM="$lam" || rc=2
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
# The TuringLLM work of the same session (dev-note §3.8): 086 held-out rebuild (scan, build, verify, report), the
# re-score on ctx_v2 + the as-run 1,000 subset, 084 nulls and 083 (DAN-159) on ctx_v2. Runs in TL_ROOT (its own
# checkout: config.yaml = config-h100-triamp.yaml), TuringLLM's default path (bf16, D8). Every GPU sub-step takes the
# fit lock separately through tl_with_fit_lock (public priority).
D83=experiments/083-v1-comparison
tl_stage() {   # tl_stage T<n> FUNC ARGS...: TL_STOP_AFTER gating, markers turingllm.T<n> (FROM does not apply)
  local num=$1 key=turingllm.$1 rc; shift
  if [ "${num#T}" -gt "${TL_STOP_AFTER#T}" ]; then say "$key: after TL_STOP_AFTER=$TL_STOP_AFTER; the TuringLLM track stops"; return 3; fi
  if is_done "$key"; then say "$key: done earlier (marker), skipped"; return 0; fi
  say "$key: START"
  mark "$key" start
  "$@"
  rc=$?
  if [ "$rc" = 0 ]; then mark "$key" done; say "$key: DONE"; else say "$key: FAILED (rc $rc)"; fi
  return "$rc"
}

tl_ctx() {   # the records every TuringLLM score reads: the rebuilt held-out sets once T2 is done (else the 15k run's)
  if [ -n "${TL_CTX_DIR:-}" ]; then echo "$TL_CTX_DIR"
  elif [ "$DRY_RUN" = 1 ] || is_done turingllm.T2; then echo "$TL_CTX_REBUILT"
  else echo "$TL_OUT/ctx"; fi
}

abspath() { case $1 in /*) echo "$1" ;; *) echo "$ROOT/$1" ;; esac; }

tl_run() {   # tl_run LOG cmd...: run in TL_ROOT (its own checkout and config.yaml); LOG stays under this chain's LOGS
  local log
  log=$(abspath "$1"); shift
  if [ "$DRY_RUN" = 1 ]; then printf '    $ (cd %s) %s>> %s 2>&1\n' "$TL_ROOT" "$(printf '%q ' "$@")" "$log"; return 0; fi
  mkdir -p "$(dirname "$log")"
  (cd "$TL_ROOT" && "$@") >> "$log" 2>&1
}

tl_launch() {   # tl_launch NAME ENV...: 062/launch.sh in TL_ROOT on the fit pool (P=TL_P unless ENV sets P)
  local name=$1 pool gu0; shift
  pool=$(fit_pool)
  local extra=()
  [ "$SMOKE" = 1 ] && extra+=(STEPS="$SMOKE_STEPS" LIMIT="$LIMIT")
  say "  turingllm $name on GPUs $pool, CTX_DIR $(tl_ctx)"
  gu0=$(gave_up "$LOGA/turingllm/$name")
  tl_run "$LOGS/turingllm/$name.launch.log" env GPUS="$pool" P="$TL_P" STAGGER="$STAGGER" MERGE=0 \
    LOGDIR="$LOGA/turingllm/$name" PY="$TL_PY" OUT="$TL_OUT" CTX_DIR="$(tl_ctx)" "${extra[@]}" "$@" \
    bash "$D62/launch.sh" || return 1
  if [ "$DRY_RUN" != 1 ] && [ "$(gave_up "$LOGA/turingllm/$name")" -gt "$gu0" ]; then
    say "  turingllm $name: a shard gave up (see $LOGS/turingllm/$name)"; return 1
  fi
}

tl_py() {   # tl_py LOG ENV... SCRIPT: one CPU-side python call in TL_ROOT (merge / report modes)
  local log=$1; shift
  local n=$#
  local env_=("${@:1:$((n - 1))}") script=${!n}
  tl_run "$log" env PYTHONPATH=src OUT="$TL_OUT" "${env_[@]}" "$TL_PY" -X utf8 "$script"
}

tl_targets() {   # the TuringLLM target list of a stage: the smoke list under LIMIT, else $1 ('' = the script's default)
  if [ "$SMOKE" = 1 ]; then echo "$TL_SMOKE_TARGETS"; else echo "$1"; fi
}

t0_check() {   # the TuringLLM artefacts are in TL_ROOT (scripts/pod_bootstrap_global.sh ROOT=$TL_ROOT, dev-note §3.8)
  local p miss=()
  local need_root=(models data outputs/top_ctx.pt outputs/mid_ctx.pt outputs/neg_ctx.pt outputs/seq_repr.pt
                   outputs/latent_stats.pt "$D84/targets_turingllm_nulls.txt" "$D62/targets_full.txt"
                   "$D62/targets_sweep.txt" "$D62/out_alpha1/alpha1.jsonl" "$D86/targets_asrun1000.txt"
                   "$D62/out_bands" "$D62/out_depth" "$D62/out_lowlam")
  local need_abs=("$TL_OUT/main/circuits" "$TL_OUT/sweep/circuits" "$TL_OUT/ctx" "$TL_CORPUS_DIR/corpus_hash.npy")
  if [ "$DRY_RUN" = 1 ]; then
    echo "    $ (check) $TL_ROOT: ${need_root[*]}"
    echo "      ${need_abs[*]}; 062 out_random{,_scale}/circuits (DAN-17 references: 65 stored, the rest refit in T4)"
    echo "      config.yaml == config-h100-triamp.yaml (the 15k run's); md5 of $D86/targets_asrun1000.txt"
    return 0
  fi
  for p in "${need_root[@]}"; do [ -e "$TL_ROOT/$p" ] || miss+=("$TL_ROOT/$p"); done
  for p in "${need_abs[@]}"; do [ -e "$p" ] || miss+=("$p"); done
  if [ ${#miss[@]} -gt 0 ]; then say "  T0: missing ${miss[*]} (bootstrap TL_ROOT first)"; return 1; fi
  for p in out_random out_random_scale; do
    [ -d "$TL_ROOT/$D62/$p/circuits" ] || say "  T0: no $D62/$p/circuits: its DAN-17 references will be REFIT in T4"
  done
  if ! cmp -s "$TL_ROOT/config.yaml" "$TL_ROOT/config-h100-triamp.yaml"; then
    if [ "$SMOKE" = 1 ]; then say "  T0 (smoke): config.yaml is not config-h100-triamp.yaml; continuing"
    else say "  T0: $TL_ROOT/config.yaml is not the 15k run's config-h100-triamp.yaml"; return 1; fi
  fi
  if [ "$SMOKE" = 1 ]; then
    mkdir -p "$TL_WORK" && head -n "$LIMIT" "$TL_ROOT/$D84/targets_turingllm_nulls.txt" > "$TL_SMOKE_TARGETS" || return 1
    say "  T0 (smoke): targets $(tr '\n' ' ' < "$TL_SMOKE_TARGETS")"
  fi
  say "  T0: as-run subset md5 $(md5sum < "$TL_ROOT/$D86/targets_asrun1000.txt" | cut -c1-32)" \
    "($(wc -l < "$TL_ROOT/$D86/targets_asrun1000.txt") targets)"
  archive turingllm-T0-check "$LOGS/turingllm" 2> /dev/null || true
}

t1_scan_body() {
  local extra=()
  [ "$SMOKE" = 1 ] && extra+=(LIMIT_SHARDS="$TL_SMOKE_SHARDS")
  tl_launch scan SCRIPT="$D86/scan.py" MODE=scan P="$TL_SCAN_P" OUT_SCAN="$TL_WORK/scan" CORPUS_DIR="$TL_CORPUS_DIR" \
    TARGETS="$(tl_targets "$TL_ROOT/$D62/targets_full.txt")" "${extra[@]}"
}

t1_scan() {   # held-out rebuild corpus scan (086: top 512 per target on the content-deduplicated corpus), then merge
  tl_with_fit_lock t1_scan_body || return 1
  local partial=()
  [ "$SMOKE" = 1 ] && partial+=(PARTIAL=1)
  tl_py "$LOGS/turingllm/scan_merge.log" MODE=merge OUT_SCAN="$TL_WORK/scan" CORPUS_DIR="$TL_CORPUS_DIR" \
    TARGETS="$(tl_targets "$TL_ROOT/$D62/targets_full.txt")" "${partial[@]}" "$D86/scan.py" || return 1
  archive turingllm-T1-scan "$TL_WORK/scan" "$LOGS/turingllm"
}

rb_env() {   # rebuild.py's settings, one per line
  printf '%s\n' SCRIPT="$D86/rebuild.py" SCAN="$TL_WORK/scan/scan_top512.pt" CTX_IN="$TL_OUT/ctx" \
    CTX_OUT="$TL_CTX_REBUILT" CORPUS_DIR="$TL_CORPUS_DIR" STORES_DIR="$TL_ROOT/outputs" OUT_RB="$TL_WORK/rebuild" \
    TARGETS="$(tl_targets "$TL_ROOT/$D62/targets_full.txt")"
}

verify_gate() {   # every rebuilt record passed the independent re-check (verify.shard*.jsonl: no fail, no missing)
  [ "$DRY_RUN" = 1 ] && { echo "    $ (gate) every row of $TL_WORK/rebuild/verify.shard*.jsonl has ok=true"; return 0; }
  "$TL_PY" -c '
import glob, json, sys
rows = [json.loads(l) for p in glob.glob(sys.argv[1] + "/verify.shard*.jsonl") for l in open(p) if l.strip()]
bad = [r for r in rows if not r.get("ok")]
print("  verify: %d records, %d ok, %d failed or missing" % (len(rows), len(rows) - len(bad), len(bad)))
for r in bad[:10]:
    print("   ", r.get("seed"), r.get("missing") and "missing" or r.get("failures", [])[:3])
sys.exit(0 if rows and not bad else 1)' "$TL_WORK/rebuild"
}

t2_build_body() { local e=(); mapfile -t e < <(rb_env); tl_launch rebuild_build "${e[@]}" MODE=build P="$TL_BUILD_P"; }
t2_verify_body() { local e=(); mapfile -t e < <(rb_env); tl_launch rebuild_verify "${e[@]}" MODE=verify P="$TL_VERIFY_P"; }

t2_rebuild() {   # rebuilt records -> TL_CTX_REBUILT (ctx_v2) = CTX_DIR from here; independent verify; report
  tl_with_fit_lock t2_build_body || return 1
  tl_with_fit_lock t2_verify_body || return 1
  local e=()
  mapfile -t e < <(rb_env)
  tl_py "$LOGS/turingllm/rebuild_report.log" "${e[@]:1}" MODE=report "$D86/rebuild.py" || return 1
  if [ "$DRY_RUN" = 1 ]; then verify_gate
  else
    mkdir -p "$LOGS/turingllm"
    verify_gate | tee -a "$LOGS/turingllm/rebuild_report.log" || { say "  T2: verification FAILED; T3-T5 not started"; return 1; }
  fi
  archive turingllm-T2-ctx "$TL_CTX_REBUILT" "$TL_WORK/rebuild" "$LOGS/turingllm"
}

t3_v2_body() {   # every target of each job (smoke: the smoke list)
  local t=()
  [ "$SMOKE" = 1 ] && t+=(TARGETS="$TL_SMOKE_TARGETS")
  tl_launch rescore_ctx_v2 SCRIPT="$D86/rescore.py" JOBS=main,sweep,alpha1 OUT_RS="$TL_WORK/rescore_ctx_v2" "${t[@]}"
}

t3_asrun_body() {   # the as-run records, the frozen stratified 1,000, same machine: the paired leakage effect
  TL_CTX_DIR="$TL_OUT/ctx" tl_launch rescore_asrun SCRIPT="$D86/rescore.py" JOBS=main \
    OUT_RS="$TL_WORK/rescore_ctx_asrun1000" TARGETS="$(tl_targets "$TL_ASRUN_TARGETS")"
}

t3_rescore() {   # re-score the fixed 062 circuits on ctx_v2 (main strongest + mid + spec, sweep arms, alpha = 1),
  #              then the as-run subset, then the report (headline >= 8 clean rows + worst case; paired effect)
  tl_with_fit_lock t3_v2_body || return 1
  tl_with_fit_lock t3_asrun_body || return 1
  tl_py "$LOGS/turingllm/rescore_report.log" REPORT=1 CTX_DIR="$TL_CTX_REBUILT" OUT_RS="$TL_WORK/rescore_ctx_v2" \
    ASRUN_RS="$TL_WORK/rescore_ctx_asrun1000" ASRUN_TARGETS="$(tl_targets "$TL_ASRUN_TARGETS")" "$D86/rescore.py" \
    || say "  (rescore report failed; the rows are kept, rerun REPORT=1 by hand)"
  archive turingllm-T3-rescore "$TL_WORK/rescore_ctx_v2" "$TL_WORK/rescore_ctx_asrun1000" "$LOGS/turingllm"
}

t4_nulls_body() {   # the same scheme as the public models; TuringLLM keeps the DAN-17 references (65 stored + 61 refit)
  local ne=()
  mapfile -t ne < <(NULLS_SUB_N=$TL_NULLS_SUB_N NULLS_REFS=wcm,dan17 NULLS_DAN17_REFIT=1 \
    nulls_env "$TL_WORK/harder_nulls" "$(tl_targets "$TL_ROOT/$D84/targets_turingllm_nulls.txt")" turingllm)
  tl_launch harder_nulls MODE=main "${ne[@]}" RN_DIR="$TL_ROOT/$D62/out_random,$TL_ROOT/$D62/out_random_scale"
}

t4_nulls() {   # DAN-181 on TuringLLM: the frozen 126 panel, on ctx_v2
  tl_with_fit_lock t4_nulls_body || return 1
  archive turingllm-T4-nulls "$TL_WORK/harder_nulls" "$LOGS/turingllm"
}

t5_stage_body() {   # t5_stage_body STAGE: one 083 stage through run_pod.sh on the current fit pool
  local s=$1 pool extra=()
  pool=$(fit_pool)
  if [ "$SMOKE" = 1 ]; then                                    # run_local.sh SMOKE=1's budgets, 1 target
    extra+=(LIMIT=1 IG_STEPS=2 BISECT_STEPS=2 STEPS=4 SIZES=W_0.001,W_0.004 REF_ARMS=W_0.001,U_0.001 N_EXACT=1)
  fi
  say "  turingllm dan159 $s on GPUs $pool, CTX_DIR $(tl_ctx)"
  local gu0
  gu0=$(gave_up "$TL_WORK/dan159/logs")
  tl_run "$LOGS/turingllm/dan159_$s.log" env GPUS="$pool" P="$TL_083_P" STAGGER="$STAGGER" STAGES="$s" PY="$TL_PY" \
    CTX_DIR="$(tl_ctx)" OUT083="$TL_WORK/dan159" TARGETS="$(tl_targets "$TL_ROOT/$D62/targets_sweep.txt")" \
    "${extra[@]}" bash "$D83/run_pod.sh" || return 1
  if [ "$DRY_RUN" != 1 ] && [ "$(gave_up "$TL_WORK/dan159/logs")" -gt "$gu0" ]; then
    say "  turingllm dan159 $s: a shard gave up (see $TL_WORK/dan159/logs)"; return 1
  fi
}

t5_dan159() {   # DAN-159 / DAN-43 / DAN-77 (083, every stage incl. AtP, IG, Ge, DAN-43, activation-ranked, exact
  #             patching, WCM / U references, the DAN-77 sizes) on the 190 sweep targets, ctx_v2; lock per stage
  local s
  for s in ref attr fitted exact ge; do
    tl_with_fit_lock t5_stage_body "$s" || return 1
  done
  t5_stage_body table || say "  (table2.py failed; the rows are kept)"
  archive turingllm-T5-dan159 "$TL_WORK/dan159" "$LOGS/turingllm"
}

tl_track() {   # rebuild FIRST (T1-T3), then the nulls and DAN-159 on the rebuilt sets
  tl_stage T0 t0_check || return
  tl_stage T1 t1_scan || return
  tl_stage T2 t2_rebuild || return
  tl_stage T3 t3_rescore || return
  tl_stage T4 t4_nulls || return
  tl_stage T5 t5_dan159 || return
}

# ------------------------------------------------------------------------------------------------- main
say "082 chain | $(git --no-pager log -1 --format='%h %s' 2> /dev/null) | MODELS=$MODELS FROM=$FROM STOP_AFTER=$STOP_AFTER"\
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
if [ "$STOP_POD" = 1 ]; then
  if [ "$rc" = 0 ] && [ "$FINAL_VERIFIED" = 1 ] && [ "$tl_rc" = 0 ]; then
    say "STOP_POD=1, C4 verified and the TuringLLM track (if on) packed: stopping the pod"
    run - runpodctl stop pod "${RUNPOD_POD_ID:-}"
  elif [ "$DRY_RUN" = 1 ]; then
    echo "    $ runpodctl stop pod \$RUNPOD_POD_ID   # only after C4 verified (and the TuringLLM track packed) in this run"
  else
    say "STOP_POD=1 but the final verification did not pass in this run (or the TuringLLM track did not finish): the pod stays up"
  fi
fi
[ "$rc" = 0 ] && [ "$tl_rc" != 0 ] && rc=1
say "chain finished (rc $rc)"
exit "$rc"
