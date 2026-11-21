#!/bin/bash
# Exp 126 night runner (local RTX 5070 Ti 16 GB, WSL, ONE GPU process at a time): why do WCM v4 a2 circuits of shallow
# transcoder targets overshoot on held-out contexts (README). Scope a2, lambda 1e-3, tuned CHOSEN.env (NEC_W 0.3).
#
#   WSL, repo root:  DRY_RUN=1 bash experiments/126-wcm-overshoot/run_night.sh        # print the plan, change nothing
#                    STOP_EPOCH=$(date -d 'tomorrow 08:00' +%s) nohup bash experiments/126-wcm-overshoot/run_night.sh \
#                        > experiments/126-wcm-overshoot/out/run_night.out 2>&1 &
#
# STAGES, in this order (each skipped when OUT/state/<stage>.done exists; inside a stage run_pod.sh / the scorer skip
# finished fits / rows, so a rerun resumes; a stage whose dependency is not done is skipped):
#   subsets     (CPU) make_ctx_subsets.py: arm B records out/targets_n8, out/targets_n16 (made at build time already)
#   fit_s400    WCM fit, 400 steps (the baseline; = production), smoke records      -> out/fit/s400
#   clip        (CPU) make_clipped.py: arm D, the s400 circuits with alpha clipped to [1/c, c], c = 1.5, 2, 3 -> out/clip
#   score_s400  score_arena_a.py a2: l1e-3 = s400 + c1.5 / c2 / c3 = the clipped copies (one call) -> out/score/s400
#   fit_s200 score_s200 fit_s100 score_s100      arm A (steps), smoke records
#   fit_n8 score_n8 fit_n16 score_n16            arm B (D_train 8 / 16 + B_mid 4 / 8), out/targets_n8 / n16 records
#   fit_s800 score_s800                          arm A, the costliest
#   analyze1    (CPU) analyze.py --alphas -> out/report.md
#   dp_owt      arm C: data_pass_tcz.py on out/corpus_owt131k (131,072 OWT windows, built on the CPU at build time),
#               batch 64 (this card), --mid-warmup 200 (= the pod's 9.8 % of batches), float32 -> out/owt131k/datapass.
#               Not started unless N126_DP_EST_S [5400] s fit before STOP_EPOCH (a data pass cannot stop half-way)
#   targets_owt (CPU) make_targets.py (EXCLUDE = arena A tune_exclude.txt) -> out/owt131k/targets
#   fit_owt score_owt   WCM fit + scoring on the tc targets of layers 0-4 that make_targets picked
#   diversity_owt (CPU) diversity.py on the owt records -> out/diversity_owt131k.md
#   analyze2    (CPU) analyze.py again (final report)
# Fits go through arena A's run_pod.sh (AA_FROM=P5 STOP_AT=P5: P5 only; P6 / P6b never start, so the P6b trap of
# STOP_AT=P6 does not arise); scoring calls score_arena_a.py directly as run_pod.sh p6_worker does (no --donor, no
# Ge recovery: not needed for the overshoot question; N126_DONOR=1 adds the smoke donor test, --donor-records run).
#
# Safety: refuses to start when REPO/.gpu-lock exists, the GPU already holds > N126_GPU_IDLE_MIB [1500] MiB, or
# OUT/ABORTED exists; holds .gpu-lock for the whole run and removes its own lock at exit (the one shell delete).
# Watchdog every N126_WATCH_S [30] s (OUT/watch.csv): GPU memory.used > N126_GPU_MAX_MIB [15300] or host MemAvailable
# < N126_RAM_MIN_GB [4] GB kills the running job's process group, writes OUT/ABORTED and stops (exit 9).
# STOP_EPOCH: no new stage once date +%s >= it (the analyze stages excepted: CPU, ~1 min); passed to run_pod.sh (its
# soft stop: no new target; hard stop N126_HARD_GRACE [1800] s later) and to the scorer (--stop-epoch: no new target).
# N126_STAGES: run only these stages (comma list); N126_SKIP: skip these. Log: OUT/night.log (one line per stage
# start / end), job logs OUT/logs/<stage>.log.
# Exit: 0 every stage done; 1 a stage failed; 7 STOP_EPOCH cut the plan; 8 refused; 9 aborted by the watchdog.
set -uo pipefail
shopt -s nullglob
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
export PATH="$PATH:/usr/lib/wsl/lib"
E=experiments/126-wcm-overshoot
O=$E/out
A=experiments/107-true-baselines/arena_a
PY=./.venv/bin/python
DRY_RUN=${DRY_RUN:-0}
STOP=${STOP_EPOCH:-99999999999}
GPU_MAX=${N126_GPU_MAX_MIB:-15300}
GPU_IDLE=${N126_GPU_IDLE_MIB:-1500}
RAM_MIN_GB=${N126_RAM_MIN_GB:-4}
EVERY=${N126_WATCH_S:-30}
HARD_GRACE=${N126_HARD_GRACE:-1800}
DP_EST_S=${N126_DP_EST_S:-5400}
DONOR=${N126_DONOR:-0}
ONLY=${N126_STAGES:-}
SKIP=${N126_SKIP:-}
LOCK=$ROOT/.gpu-lock
STATE=$O/state
LOGS=$O/logs

# inputs
# 2026-11-16 night 1: the layer-0 tc targets (0.tc.1257, 0.tc.5811) are dropped: in scope a2 (transcoders only) a layer-0
# transcoder has no upstream candidate, the fit errors ("does not require grad") and run_pod.sh's every-target-ok check fails
# the whole stage. no0() below drops layer-0 keys from every fit / score key list (arm B and OWT lists too).
KEYS=1.tc.23034,1.tc.5306,2.tc.20862,2.tc.5850,3.tc.13057,3.tc.19795,4.tc.10851,4.tc.9014
SMOKE_TGT=$A/out/targets_smoke
SMOKE_STO=$A/out/pilot/datapass
SMOKE_DATA=data_public/gpt2-small
TUNE=$A/out/tune/CHOSEN.env
EXCLUDE=$A/tune_exclude.txt
OWT_CORPUS=$O/corpus_owt131k
OWT=$O/owt131k
OWT_STO=$OWT/datapass
OWT_TGT=$OWT/targets
CLIPS="1.5 2 3"
# cost model (minutes; measured 2026-11-16 on this card: shallow a2 fit ~2.5 min at 400 steps, scoring ~5 min per
# 8 circuits; the OWT 131k data pass from the owt32k pass, 813 s at 32k, x4 + the pessimistic est_dp.py fit)
FIT_MIN=2.5; FIT_START=2; UNIT_MIN=0.625; SCORE_START=2; DP_MIN=80; TGT_MIN=45

BASEENV="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ARENA_A_BOS=1 FAMILY=gpt2-small-tcz SAE_PRELOAD=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WCM_VERSION=v4 DEEP_BS=1 NVIDIA_TF32_OVERRIDE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID"
FIT_COMMON="AA_GPUS=0 AA_PACK=0 AA_SCOPES=a2 AA_LAMBDAS=1e-3 AA_FROM=P5 STOP_AT=P5 AA_BASELINE_DEVICE=cpu \
AA_TUNE_ENV=$TUNE AA_N_SEQ=0 DEEP_BS=1 NVIDIA_TF32_OVERRIDE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID AA_HARD_GRACE=$HARD_GRACE"
if [ -n "${STOP_EPOCH:-}" ]; then FIT_COMMON="$FIT_COMMON STOP_EPOCH=$STOP_EPOCH"; else FIT_COMMON="$FIT_COMMON AA_MAX_H=0"; fi

# ------------------------------------------------------------------------------------------------- helpers
ts() { date '+%F %T'; }
say() {
  local line="[n126 $(date +%H:%M:%S)] $*"
  echo "$line"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$O"; echo "[$(ts)] $*" >> "$O/night.log"; }
}
late() { [ "$(date +%s)" -ge "$STOP" ]; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
nlines() { if [ -f "$1" ]; then grep -c . "$1"; else echo 0; fi; }
ncsv() { echo "$1" | tr ',' '\n' | grep -c .; }
no0() { echo "$1" | tr ',' '\n' | grep -v '^0\.' | paste -sd, -; }   # drop layer-0 keys (no a2 upstream; see KEYS)
fmin() { awk -v x="$1" 'BEGIN { printf "%.0f", x }'; }
sel() {   # sel STAGE: in N126_STAGES (if set) and not in N126_SKIP
  { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; } && [[ ",$SKIP," != *",$1,"* ]]
}

JOB=""
job() {   # job NAME CMD...: the command in its own process group, watchdog-guarded; rc = its rc, 9 = aborted
  local name=$1; shift
  local log=$LOGS/$name.log m r why="" rc
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ $*   >> $log"; return 0; fi
  mkdir -p "$LOGS"
  echo "[$(ts)] \$ $*" >> "$log"
  setsid "$@" >> "$log" 2>&1 < /dev/null &
  JOB=$!
  while kill -0 "$JOB" 2> /dev/null; do
    m=$(gpu_mib); r=$(avail_mib)
    echo "$(date +%s),$name,${m:-NA},${r:-NA}" >> "$O/watch.csv"
    if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX MiB"
    elif [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then why="host MemAvailable $r MiB < $RAM_MIN_GB GB"; fi
    if [ -n "$why" ]; then
      say "ABORT $name: $why: killing its process group"
      echo "$(ts) $name: $why" > "$O/ABORTED"
      kill -TERM -- -"$JOB" 2> /dev/null; sleep 15; kill -KILL -- -"$JOB" 2> /dev/null
      break
    fi
    sleep "$EVERY" & wait $!
  done
  wait "$JOB" 2> /dev/null; rc=$?
  JOB=""
  [ -n "$why" ] && return 9
  [ "$rc" = 0 ] || echo "[$(ts)] exit $rc" >> "$log"
  return "$rc"
}

declare -A DEPS=(
  [clip]="fit_s400" [score_s400]="fit_s400 clip" [score_s200]="fit_s200" [score_s100]="fit_s100"
  [fit_n8]="subsets" [score_n8]="fit_n8" [fit_n16]="subsets" [score_n16]="fit_n16" [score_s800]="fit_s800"
  [targets_owt]="dp_owt" [fit_owt]="targets_owt" [score_owt]="fit_owt" [diversity_owt]="targets_owt"
  [fit_owt_l4]="targets_owt" [fit_owt_l16]="targets_owt" [score_owt_small]="fit_owt_l4 fit_owt_l16"
  [df_deep]="targets_owt" [fit_deep]="targets_owt" [score_deep]="fit_deep"
)
declare -A EST=()
FAILS=0
CUT=0
ETA=0
stage() {   # stage NAME FUNC [always]: markers, deps, STOP_EPOCH, night.log; "always" = runs every time (CPU analysis)
  local name=$1 fn=$2 always=${3:-} d rc t0
  sel "$name" || return 0
  if [ -z "$always" ] && [ -f "$STATE/$name.done" ]; then
    say "$name: done earlier ($(cat "$STATE/$name.done"))"; return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    ETA=$(awk -v a="$ETA" -v b="${EST[$name]:-1}" 'BEGIN { print a + b }')
    say "$name: PLAN (~$(fmin "${EST[$name]:-1}") min; cumulative ~$(fmin "$ETA") min)$([ -n "${DEPS[$name]:-}" ] && echo "; needs ${DEPS[$name]}")"
    "$fn"
    return 0
  fi
  for d in ${DEPS[$name]:-}; do
    if [ ! -f "$STATE/$d.done" ]; then say "$name: skipped (dependency $d not done)"; return 0; fi
  done
  if [ -z "$always" ] && late; then say "$name: STOP_EPOCH reached: not started"; CUT=1; return 7; fi
  say "START $name (estimate ~$(fmin "${EST[$name]:-1}") min)"
  t0=$(date +%s)
  "$fn"; rc=$?
  say "END $name rc $rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if [ "$rc" = 0 ]; then mkdir -p "$STATE"; ts > "$STATE/$name.done"
  elif [ "$rc" = 9 ]; then say "watchdog abort: stopping the night (exit 9; read $O/ABORTED, move it away to resume)"; exit 9
  elif [ "$rc" = 7 ] || late; then CUT=1                 # cut by STOP_EPOCH (run_pod.sh then exits 1): resumable
  else FAILS=$((FAILS + 1)); fi
  return 0
}

# ------------------------------------------------------------------------------------------------- units
fit() {   # fit NAME TGT STO DATA KEYS [STEPS]: run_pod.sh P5 (WCM a2 lambda 1e-3, CHOSEN.env), own AA_WORK
  local name=$1 tgt=$2 sto=$3 data=$4 keys steps=${6:-}
  keys=$(no0 "$5")
  local W=$O/fit/$name lam=${LAM:-1e-3}   # LAM (night 2): the lambda; env takes the last AA_LAMBDAS, so it overrides FIT_COMMON's
  local envs="$FIT_COMMON AA_LAMBDAS=$lam AA_WORK=$W AA_TARGETS_DIR=$tgt AA_STORES=$sto AA_DATA_SRC=$data AA_KEYS=$keys${steps:+ AA_STEPS=$steps}"
  if [ "$DRY_RUN" = 1 ]; then
    echo "    \$ env $envs bash $A/run_pod.sh   >> $LOGS/fit_$name.log"
    echo "      run_pod.sh's own dry run of it (P5 lines):"
    env $envs DRY_RUN=1 bash "$A/run_pod.sh" 2>&1 | grep -E 'P5|REFUSED|WCM|launch.sh' | sed 's/^/        /' | cut -c1-1500
    return 0
  fi
  job "fit_$name" env $envs bash "$A/run_pod.sh" || return $?
  # belt and braces: an ok / skip status row for every key (run_pod.sh fits_done)
  local n ok
  n=$(ncsv "$keys")
  ok=$(cat "$W"/wcm/a2_l$lam/main/status.shard*.jsonl 2> /dev/null | grep -c -E '"ok": true|"skip"')
  [ "$ok" -ge "$n" ] || { say "  fit_$name: $ok of $n status rows ok"; return 1; }
}

score() {   # score NAME TGT STO DATA KEYS SPEC N_LABELS: score_arena_a.py a2 -> out/score/NAME
  local name=$1 tgt=$2 sto=$3 data=$4 keys spec=$6 nl=$7
  keys=$(no0 "$5")
  local S=$O/score/$name stop="" donor=""
  [ -n "${STOP_EPOCH:-}" ] && stop=" --stop-epoch $STOP_EPOCH"
  [ "$DONOR" = 1 ] && donor=" --donor --donor-pool $S/random_pool.pt --donor-records run"
  [ -n "${DF:-}" ] && donor="$donor --dunefsky $DF"    # DF (night 2): a run_dunefsky.py out dir, scored beside the WCM spec
  job "score_$name" bash -c "env $BASEENV ARENA_A_SCOPE=a2 DATA=$data STORES=$sto CUDA_VISIBLE_DEVICES=0 ARENA_A_CTX_DIRS=$tgt/ctx OUT=$S $PY -X utf8 -u $A/score_arena_a.py --scope a2 --out $S --wcm $spec --keys $keys --selftest$donor$stop" \
    || return $?
  [ "$DRY_RUN" = 1 ] && return 0
  local want got
  want=$(( $(ncsv "$keys") * nl ))
  got=$(grep -c '"eval_set": "home"' "$S/rows.jsonl" 2> /dev/null)
  [ "${got:-0}" -ge "$want" ] || { say "  score_$name: $got of $want units scored"; late && return 7; return 1; }
}

wcm_dir() { echo "$O/fit/$1/wcm/a2_l${2:-1e-3}"; }

u_subsets() {
  if [ "$DRY_RUN" != 1 ] && [ -f "$O/targets_n8/targets.txt" ] && [ -f "$O/targets_n16/targets.txt" ]; then
    say "  arm B records exist ($(nlines "$O/targets_n8/targets.txt") targets)"; return 0
  fi
  job subsets env PYTHONPATH=src CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -B $E/make_ctx_subsets.py \
    --src $SMOKE_TGT --out $O --keys $KEYS --sizes 8,16
}
keys_of() { if [ -f "$1/targets.txt" ]; then paste -sd, "$1/targets.txt"; else echo "KEYS_OF_$(basename "$1")"; fi; }

u_fit_s400() { fit s400 $SMOKE_TGT $SMOKE_STO $SMOKE_DATA $KEYS; }
u_fit_s200() { fit s200 $SMOKE_TGT $SMOKE_STO $SMOKE_DATA $KEYS 200; }
u_fit_s100() { fit s100 $SMOKE_TGT $SMOKE_STO $SMOKE_DATA $KEYS 100; }
u_fit_s800() { fit s800 $SMOKE_TGT $SMOKE_STO $SMOKE_DATA $KEYS 800; }
u_fit_n8() { fit n8 $O/targets_n8 $SMOKE_STO $SMOKE_DATA "$(keys_of $O/targets_n8)"; }
u_fit_n16() { fit n16 $O/targets_n16 $SMOKE_STO $SMOKE_DATA "$(keys_of $O/targets_n16)"; }
u_clip() {
  local cs
  cs=$(echo $CLIPS | tr ' ' ',')
  job clip env PYTHONPATH=src CUDA_VISIBLE_DEVICES= ARENA_A_SCOPE=a2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    $PY -B $E/make_clipped.py --src "$(wcm_dir s400)" --out $O/clip --c "$cs" --keys $KEYS || return $?
  [ "$DRY_RUN" = 1 ] && return 0
  local c n
  for c in $CLIPS; do
    n=$(ls "$O/clip/c$c/main/circuits/"*.pt 2> /dev/null | wc -l)
    [ "$n" -ge "$(ncsv "$KEYS")" ] || { say "  clip c$c: $n circuits"; return 1; }
  done
}
u_score_s400() {
  local spec="l1e-3=$(wcm_dir s400)" c
  for c in $CLIPS; do spec="$spec,c$c=$O/clip/c$c"; done
  score s400 $SMOKE_TGT $SMOKE_STO $SMOKE_DATA $KEYS "$spec" $((1 + $(echo $CLIPS | wc -w)))
}
u_score_s200() { score s200 $SMOKE_TGT $SMOKE_STO $SMOKE_DATA $KEYS "l1e-3=$(wcm_dir s200)" 1; }
u_score_s100() { score s100 $SMOKE_TGT $SMOKE_STO $SMOKE_DATA $KEYS "l1e-3=$(wcm_dir s100)" 1; }
u_score_s800() { score s800 $SMOKE_TGT $SMOKE_STO $SMOKE_DATA $KEYS "l1e-3=$(wcm_dir s800)" 1; }
u_score_n8() { score n8 $O/targets_n8 $SMOKE_STO $SMOKE_DATA "$(keys_of $O/targets_n8)" "l1e-3=$(wcm_dir n8)" 1; }
u_score_n16() { score n16 $O/targets_n16 $SMOKE_STO $SMOKE_DATA "$(keys_of $O/targets_n16)" "l1e-3=$(wcm_dir n16)" 1; }
u_analyze() {
  job analyze env PYTHONPATH=src CUDA_VISIBLE_DEVICES= ARENA_A_SCOPE=a2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    $PY -B $E/analyze.py --alphas --out $O/report.md
}

dp_done() { [ -f "$1/neg_ctx.pt" ] && [ -f "$1/data_pass_manifest.json" ]; }
u_dp_owt() {
  if [ "$DRY_RUN" != 1 ]; then
    dp_done "$OWT_STO" && { say "  data pass $OWT_STO exists"; return 0; }
    [ -f "$OWT_CORPUS/corpus_manifest.json" ] || { say "  no corpus $OWT_CORPUS (manifest)"; return 1; }
    if [ -e "$OWT_STO" ]; then say "  $OWT_STO exists without a finished pass (interrupted?): move it away, then rerun"; return 1; fi
    if [ $(( $(date +%s) + DP_EST_S )) -gt "$STOP" ]; then
      say "  not started: the data pass (~$DP_EST_S s, N126_DP_EST_S) would not end before STOP_EPOCH"; return 7
    fi
    mkdir -p "$OWT"
  fi
  job dp_owt bash -c "env $BASEENV ARENA_A_SCOPE=a1 DATA=$OWT_CORPUS STORES=$OWT_STO CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY -X utf8 -u $A/data_pass_tcz.py --data $OWT_CORPUS --out $OWT_STO --batch-size 64 --mid-warmup 200 --dtype float32"
}
u_targets_owt() {
  if [ "$DRY_RUN" != 1 ] && [ -f "$OWT_TGT/MANIFEST.json" ] && [ -f "$OWT_TGT/targets.txt" ]; then
    say "  targets $OWT_TGT exist ($(nlines "$OWT_TGT/targets.txt"))"
  else
    job targets_owt bash -c "env $BASEENV ARENA_A_SCOPE=a1 DATA=$OWT_CORPUS STORES=$OWT_STO CUDA_VISIBLE_DEVICES= OUT=$OWT_TGT EXCLUDE=$EXCLUDE $PY -X utf8 -I -B $A/make_targets.py" \
      || return $?
  fi
  [ "$DRY_RUN" = 1 ] && { echo "    \$ awk -F. '\$2 == \"tc\" && \$1 <= 4' $OWT_TGT/targets.txt > $OWT/keys_shallow_tc.txt"; return 0; }
  awk -F. '$2 == "tc" && $1 <= 4' "$OWT_TGT/targets.txt" > "$OWT/keys_shallow_tc.txt"
  say "  owt shallow tc targets: $(paste -sd' ' "$OWT/keys_shallow_tc.txt")"
  [ -s "$OWT/keys_shallow_tc.txt" ] || { say "  make_targets picked no tc target in layers 0-4"; return 1; }
}
owt_keys() { if [ -s "$OWT/keys_shallow_tc.txt" ]; then paste -sd, "$OWT/keys_shallow_tc.txt"; else echo "OWT_SHALLOW_TC_KEYS"; fi; }
u_fit_owt() { fit owt131k $OWT_TGT $OWT_STO $OWT_CORPUS "$(owt_keys)"; }
u_score_owt() { score owt131k $OWT_TGT $OWT_STO $OWT_CORPUS "$(owt_keys)" "l1e-3=$(wcm_dir owt131k)" 1; }
# ---- night 2 (2026-11-17): smaller circuits (larger lambda) on the owt shallow targets; deep owt tc targets
# (layers 5-11) with Dunefsky (their code, ~/venvs/tc107gpu torch 2.8 cu128, prompt 0 = K 1) against WCM lambda 1e-3
DF_PY=$HOME/venvs/tc107gpu/bin/python
DF_DEEP=$O/dunefsky_owt_deep
DF_BR="5 30 60 150"
owt_deep_keys() { if [ -s "$OWT_TGT/targets.txt" ]; then awk -F. '$2 == "tc" && $1 >= 5' "$OWT_TGT/targets.txt" | paste -sd, -; else echo OWT_DEEP_TC_KEYS; fi; }
u_fit_owt_l4() { LAM=4e-3 fit owt131k_l4e-3 $OWT_TGT $OWT_STO $OWT_CORPUS "$(owt_keys)"; }
u_fit_owt_l16() { LAM=1.6e-2 fit owt131k_l1.6e-2 $OWT_TGT $OWT_STO $OWT_CORPUS "$(owt_keys)"; }
u_score_owt_small() {
  score owt131k_small $OWT_TGT $OWT_STO $OWT_CORPUS "$(owt_keys)" \
    "l4e-3=$(wcm_dir owt131k_l4e-3 4e-3),l1.6e-2=$(wcm_dir owt131k_l1.6e-2 1.6e-2)" 2
}
u_df_deep() {   # ascending branch values, so a cut still leaves the small ones; the gate only on the first call
  local nb gate="" keys rc=0
  keys=$(owt_deep_keys)
  for nb in $DF_BR; do
    job "df_deep_b$nb" env PYTHONPATH="$ROOT/third_party/transcoder_circuits" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      "$DF_PY" -B $A/run_dunefsky.py --targets $OWT_TGT --out $DF_DEEP --keys "$keys" --k 1 --branches $nb --iters 5 $gate \
      || { rc=$?; [ "$rc" = 9 ] && return 9; }
    gate="--no-gate"
    late && return 7
  done
  return $rc
}
u_fit_deep() { fit owt131k_deep $OWT_TGT $OWT_STO $OWT_CORPUS "$(owt_deep_keys)"; }
u_score_deep() { DF=$DF_DEEP score owt131k_deep $OWT_TGT $OWT_STO $OWT_CORPUS "$(owt_deep_keys)" "l1e-3=$(wcm_dir owt131k_deep)" 1; }
# deep57 (2026-11-17 03:30): the L8+ a2 fits exceed this PC's host RAM (MemAvailable < 2 GB at the first L8 fit, twice),
# so score the deep targets whose fit finished (ok status rows; L5-7) against Dunefsky on the same keys. No dependency
# on fit_deep; run alone with N126_STAGES=score_deep57.
deep_ok_keys() { cat "$(wcm_dir owt131k_deep)"/main/status.shard*.jsonl 2> /dev/null | grep '"ok": true' | sed 's/.*"seed": "\([^"]*\)".*/\1/' | paste -sd, -; }
u_score_deep57() { DF=$DF_DEEP score owt131k_deep57 $OWT_TGT $OWT_STO $OWT_CORPUS "$(deep_ok_keys)" "l1e-3=$(wcm_dir owt131k_deep)" 1; }

u_diversity_owt() {
  job diversity_owt env PYTHONPATH=src CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -B $E/diversity.py \
    --ctx $OWT_TGT/ctx --kind tc --out $O/diversity_owt131k.md --title "owt131k records (OWT 131,072 windows), tc targets"
}

# ------------------------------------------------------------------------------------------------- estimates
NK=$(ncsv "$KEYS")
NB=9                                                      # arm B: the 9 arm-B records (1.tc.5306 is arm A)
[ -f "$O/targets_n8/targets.txt" ] && NB=$(nlines "$O/targets_n8/targets.txt")
NCL=$(echo $CLIPS | wc -w)
fe() { awk -v n="$1" -v s="$2" -v f="$FIT_MIN" -v z="$FIT_START" 'BEGIN { print z + n * f * s / 400 }'; }
se() { awk -v n="$1" -v u="$UNIT_MIN" -v z="$SCORE_START" 'BEGIN { print z + n * u }'; }
EST=([subsets]=1 [fit_s400]=$(fe $NK 400) [clip]=1 [score_s400]=$(se $((NK * (1 + NCL)))) [fit_s200]=$(fe $NK 200)
     [score_s200]=$(se $NK) [fit_s100]=$(fe $NK 100) [score_s100]=$(se $NK) [fit_n8]=$(fe $NB 400) [score_n8]=$(se $NB)
     [fit_n16]=$(fe $NB 400) [score_n16]=$(se $NB) [fit_s800]=$(fe $NK 800) [score_s800]=$(se $NK) [analyze1]=1
     [dp_owt]=$DP_MIN [targets_owt]=$TGT_MIN [fit_owt]=$(fe 10 400) [score_owt]=$(se 10) [diversity_owt]=1 [analyze2]=1
     [fit_owt_l4]=$(fe 8 400) [fit_owt_l16]=$(fe 8 400) [score_owt_small]=$(se 16)
     [df_deep]=40 [fit_deep]=$(fe 28 400) [score_deep]=$(se 70) [score_deep57]=$(se 30))   # deep: ~2x a shallow fit (guess); 14 x (WCM + 4 Dunefsky)

# ------------------------------------------------------------------------------------------------- preflight
preflight() {   # read-only; prints every problem; rc 1 if any
  local bad=0 f k
  for f in "$PY" "$A/run_pod.sh" "$A/score_arena_a.py" "$A/driver_tcz.py" "$A/data_pass_tcz.py" "$A/make_targets.py" \
           "$TUNE" "$EXCLUDE" "$SMOKE_TGT/targets.txt" "$SMOKE_STO/data_pass_manifest.json" "$SMOKE_STO/neg_ctx.pt" \
           "$SMOKE_DATA/corpus_manifest.json" experiments/035-transcoder-compare/transcoders_ct/meta.json \
           "$HOME/.cache/huggingface/hub/models--gpt2" "$HOME/.cache/huggingface/hub/models--pchlenski--gpt2-transcoders"; do
    [ -e "$f" ] || { say "  missing: $f"; bad=1; }
  done
  for k in ${KEYS//,/ }; do
    grep -qxF "$k" "$SMOKE_TGT/targets.txt" 2> /dev/null || { say "  $k is not in $SMOKE_TGT/targets.txt"; bad=1; }
    [ -f "$SMOKE_TGT/ctx/$k.pt" ] || { say "  no record $SMOKE_TGT/ctx/$k.pt"; bad=1; }
  done
  [ -f "$OWT_CORPUS/corpus_manifest.json" ] || say "  NOTE: no $OWT_CORPUS/corpus_manifest.json: arm C (dp_owt) will fail"
  command -v nvidia-smi > /dev/null || { say "  nvidia-smi not found"; bad=1; }
  return "$bad"
}

# ------------------------------------------------------------------------------------------------- main
say "exp 126 night | $(git --no-pager log -1 --format='%h' 2> /dev/null) | keys $KEYS | stop epoch $([ -n "${STOP_EPOCH:-}" ] && date -d "@$STOP_EPOCH" '+%a %F %H:%M' || echo none) | watchdog GPU > $GPU_MAX MiB / RAM < $RAM_MIN_GB GB every $EVERY s$([ -n "$ONLY" ] && echo " | only $ONLY")$([ -n "$SKIP" ] && echo " | skip $SKIP")$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
preflight || { say "preflight refused"; [ "$DRY_RUN" = 1 ] || exit 8; }
m=$(gpu_mib)
if [ "$DRY_RUN" = 1 ]; then
  say "dry run: GPU now holds ${m:-?} MiB (refuse above $GPU_IDLE); lock $LOCK $([ -e "$LOCK" ] && echo "EXISTS: $(head -c 200 "$LOCK")" || echo absent); ABORTED $([ -f "$O/ABORTED" ] && echo EXISTS || echo absent)"
  [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ] && say "dry run: NOTE the GPU holds $m MiB > $GPU_IDLE: a real start would be REFUSED (N126_GPU_IDLE_MIB)"
else
  [ -e "$LOCK" ] && { say "refused: $LOCK exists: $(head -c 300 "$LOCK")"; exit 8; }
  [ -f "$O/ABORTED" ] && { say "refused: $O/ABORTED exists ($(cat "$O/ABORTED")): read it, move it away to resume"; exit 8; }
  if [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ]; then say "refused: the GPU already holds $m MiB (> $GPU_IDLE, N126_GPU_IDLE_MIB)"; exit 8; fi
  [ -n "$m" ] || say "WARNING: nvidia-smi gave no reading; the GPU watchdog is blind"
  echo "exp 126 run_night.sh (experiments/126-wcm-overshoot) pid $$ started $(ts) on $(hostname); stop epoch ${STOP_EPOCH:-none}" > "$LOCK" \
    || { say "cannot write $LOCK"; exit 8; }
  # the lock is this run's own: removed at exit only while it still names this pid
  trap 'grep -q "pid $$ " "$LOCK" 2> /dev/null && rm -f "$LOCK"' EXIT
  trap '[ -n "$JOB" ] && { kill -TERM -- -"$JOB" 2> /dev/null; sleep 10; kill -KILL -- -"$JOB" 2> /dev/null; }; say "signal: stopped"; exit 143' INT TERM HUP
  mkdir -p "$O" "$STATE" "$LOGS"
fi

stage subsets u_subsets
stage fit_s400 u_fit_s400
stage clip u_clip
stage score_s400 u_score_s400
stage fit_s200 u_fit_s200
stage score_s200 u_score_s200
stage fit_s100 u_fit_s100
stage score_s100 u_score_s100
stage fit_n8 u_fit_n8
stage score_n8 u_score_n8
stage fit_n16 u_fit_n16
stage score_n16 u_score_n16
stage fit_s800 u_fit_s800
stage score_s800 u_score_s800
stage analyze1 u_analyze always
stage dp_owt u_dp_owt
stage targets_owt u_targets_owt
stage fit_owt u_fit_owt
stage score_owt u_score_owt
stage diversity_owt u_diversity_owt
stage analyze2 u_analyze always
# night 2 (2026-11-17)
stage fit_owt_l4 u_fit_owt_l4
stage fit_owt_l16 u_fit_owt_l16
stage score_owt_small u_score_owt_small
stage df_deep u_df_deep
stage fit_deep u_fit_deep
stage score_deep u_score_deep
stage score_deep57 u_score_deep57
stage analyze3 u_analyze always
if [ "$DRY_RUN" = 1 ]; then say "dry run: total ~$(fmin "$ETA") min (~$(awk -v m="$ETA" 'BEGIN { printf "%.1f", m / 60 }') h) for the stages not done"; exit 0; fi
say "night finished: $FAILS failed stage(s)$([ "$CUT" = 1 ] && echo '; STOP_EPOCH cut the plan')"
[ "$FAILS" = 0 ] || exit 1
[ "$CUT" = 0 ] || exit 7
exit 0
