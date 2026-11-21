#!/bin/bash
# Experiment 081, the v4 rerun (2026-11-18, DAN-221; README "v4 rerun 2026-11-18"): IOI and greater-than task circuits
# under behavioural WCM v4 (task_common WCM_VERSIONS["v4"]: v2 + R with the items' own counterfactual prompts as
# donors, geometric lr + 0.1 warm-up at peak x3.15, absolute wd 0.05), local RTX 5070 Ti, ONE GPU process at a time.
# Does v4 recover the published heads (Wang et al. 2022 IOI; Hanna et al. 2023 greater-than)?
#
#   WSL, repo root:  DRY_RUN=1 bash experiments/081-task-circuits/queue_v4.sh          # print the plan, change nothing
#                    STOP_EPOCH=$(date -d 'today 23:30' +%s) nohup setsid bash experiments/081-task-circuits/queue_v4.sh \
#                        > experiments/081-task-circuits/out_v4/queue_v4.out 2>&1 < /dev/null &
#
# STAGES (V4_STAGES picks a subset, comma list), in value order; each is one run.py process (resumable: circuits on
# disk are reloaded, finished eval rows are not redone) and is skipped when OUT/state/<stage>.done exists:
#   ioi_W    IOI, W at the 3 lambdas of config_v4_ioi.json (+ member profiles, + minimal-pair score)
#   gt_W     greater-than, the same with config_v4_greater_than.json
#   ioi_CF   IOI, W_CF (minimal pairs in the training rows) at the middle lambda (V4_IOI_MID)
#   gt_CF    greater-than, W_CF at V4_GT_MID
#   ioi_RND  IOI random null, V4_DRAWS draws at V4_IOI_MID (W reloaded, not refitted)
#   gt_RND   greater-than random null at V4_GT_MID
#   ioi_U    IOI unweighted arm, 3 lambdas (claim 2)
#   gt_U     greater-than unweighted arm
#   report   (CPU-light, always) report.py + figure.py on OUT
# Outputs: experiments/081-task-circuits/out_v4/gpt2-small (eval.jsonl, circuits/, train.jsonl, profiles.jsonl.gz, ...);
# competence from out_v2/probe (the full model's, version-independent).
#
# Safety (the pattern of experiments/128-pod-preflight/run_night.sh): refuses to start when REPO/.gpu-lock exists, the
# GPU already holds > V4_GPU_IDLE_MIB [2500] MiB, or OUT/ABORTED exists; writes its own .gpu-lock ("pid $$ ...") and
# removes it at exit only while it still names this pid. Watchdog every V4_WATCH_S [10] s (OUT/watch.csv): GPU
# memory.used > V4_GPU_MAX_MIB [15300] or host MemAvailable < V4_RAM_MIN_GB [3] GB kills the job's process group,
# writes OUT/ABORTED and stops the queue (exit 9). STOP_EPOCH: no new stage once date +%s >= it, and run.py starts no
# new fit / scoring / profile after it (exit 7); every job is killed at STOP_EPOCH + V4_HARD_GRACE [1800] s.
# Exit: 0 every stage done; 1 a stage failed; 7 STOP_EPOCH cut the plan; 8 refused; 9 aborted by the watchdog.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
export PATH="$PATH:/usr/lib/wsl/lib"
E=experiments/081-task-circuits
O=$E/out_v4
OM=$O/gpt2-small
PROBE=$E/out_v2/probe
PY=./.venv/bin/python
DRY_RUN=${DRY_RUN:-0}
STOP=${STOP_EPOCH:-99999999999}
GPU_MAX=${V4_GPU_MAX_MIB:-15300}
GPU_IDLE=${V4_GPU_IDLE_MIB:-2500}
RAM_MIN_GB=${V4_RAM_MIN_GB:-3}
EVERY=${V4_WATCH_S:-10}
HARD_GRACE=${V4_HARD_GRACE:-1800}
ONLY=${V4_STAGES:-}
IOI_MID=${V4_IOI_MID:-3e-5}
GT_MID=${V4_GT_MID:-3e-5}
DRAWS=${V4_DRAWS:-2}
CF_N=${V4_CF_SCORE_N:-1000}
LOCK=$ROOT/.gpu-lock
STATE=$OM/state
LOGS=$O/logs
if [ -n "${STOP_EPOCH:-}" ]; then HARD_DEADLINE=$((STOP_EPOCH + HARD_GRACE)); else HARD_DEADLINE=99999999999; fi
BASE="MODEL=gpt2-small WCM_VERSION=v4 PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_BS=64 OUT_PROBE=$PROBE OUT=$OM CF_SCORE_N=$CF_N${STOP_EPOCH:+ STOP_EPOCH=$STOP_EPOCH}"

ts() { date '+%F %T'; }
say() {
  local line="[v4q $(date +%H:%M:%S)] $*"
  echo "$line"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$O"; echo "[$(ts)] $*" >> "$O/queue_v4.log"; }
}
late() { [ "$(date +%s)" -ge "$STOP" ]; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
hm() { if [ "$1" -ge 99999999999 ]; then echo none; else date -d "@$1" '+%a %H:%M'; fi; }
inlist() { [[ ",$2," == *",$1,"* ]]; }
sel() { [ -z "$ONLY" ] || inlist "$1" "$ONLY"; }
fmin() { awk -v x="$1" 'BEGIN { printf "%.0f", x }'; }

JOB=""
job() {   # job NAME CMD...: own process group, watchdog-guarded; rc = its rc, 9 = watchdog trip, 7 = hard deadline
  local name=$1; shift
  local log=$LOGS/$name.log m r why="" rc hard=0
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ $*   >> $log"; return 0; fi
  mkdir -p "$LOGS"
  echo "[$(ts)] \$ $*" >> "$log"
  setsid "$@" >> "$log" 2>&1 < /dev/null &
  JOB=$!
  while kill -0 "$JOB" 2> /dev/null; do
    m=$(gpu_mib); r=$(avail_mib)
    echo "$(date +%s),$name,${m:-NA},${r:-NA}" >> "$O/watch.csv"
    if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX MiB"
    elif [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then why="host MemAvailable $r MiB < $RAM_MIN_GB GB"
    elif [ "$(date +%s)" -ge "$HARD_DEADLINE" ]; then why="hard deadline $(hm "$HARD_DEADLINE") reached"; hard=1; fi
    if [ -n "$why" ]; then
      say "KILL $name: $why: killing its process group"
      [ "$hard" = 1 ] || echo "$(ts) $name: $why" > "$OM/ABORTED"
      kill -TERM -- -"$JOB" 2> /dev/null; sleep 15; kill -KILL -- -"$JOB" 2> /dev/null
      break
    fi
    sleep "$EVERY" & wait $!
  done
  wait "$JOB" 2> /dev/null; rc=$?
  JOB=""
  if [ -n "$why" ]; then echo "[$(ts)] killed: $why" >> "$log"; [ "$hard" = 1 ] && return 7; return 9; fi
  [ "$rc" = 0 ] || echo "[$(ts)] exit $rc" >> "$log"
  return "$rc"
}

# ---------------------------------------------------------------------------------------------- estimates (min)
# Basis: out_v2 (v2, same card, float32, EVAL_BS 64): IOI fit 172-177 s, score (5,000 items) 351-364 s, profile 24-49 s;
# greater-than fit 141-148 s, score 212-218 s; ~2 min process start-up + means per run.py process. v4 adds the R term
# (a 4th forward + backward per step: fit x4/3, NOT timed here) and the minimal-pair score on 1,000 items (2 x 0.2 x
# score). Per circuit: IOI W 232 + 352 + 141 + 40 = 765 s; IOI RND 232 + 352; IOI U 232 + 352 + 141. greater-than W 197
# + 216 + 86 + 30 = 529 s; RND 197 + 216; U 197 + 216 + 86.
declare -A EST=([ioi_W]=40 [gt_W]=29 [ioi_CF]=14 [gt_CF]=10 [ioi_RND]=21 [gt_RND]=16 [ioi_U]=38 [gt_U]=27 [report]=2)

FAILS=0
CUT=0
ETA=0
stage() {   # stage NAME FUNC [always]
  local name=$1 fn=$2 always=${3:-} rc t0
  sel "$name" || return 0
  if [ -z "$always" ] && [ -f "$STATE/$name.done" ]; then say "$name: done earlier ($(cat "$STATE/$name.done"))"; return 0; fi
  if [ "$DRY_RUN" = 1 ]; then
    ETA=$(awk -v a="$ETA" -v b="${EST[$name]:-1}" 'BEGIN { print a + b }')
    say "$name: PLAN (~$(fmin "${EST[$name]:-1}") min; cumulative ~$(fmin "$ETA") min = $(date -d "@$(( $(date +%s) + ${ETA%.*} * 60 ))" '+%H:%M') if started now)"
    "$fn"
    return 0
  fi
  if [ -z "$always" ] && late; then say "$name: STOP_EPOCH reached: not started"; CUT=1; return 0; fi
  say "START $name (estimate ~$(fmin "${EST[$name]:-1}") min)"
  t0=$(date +%s)
  "$fn"; rc=$?
  say "END $name rc $rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if [ "$rc" = 0 ]; then mkdir -p "$STATE"; ts > "$STATE/$name.done"
  elif [ "$rc" = 9 ]; then say "watchdog abort: stopping the queue (exit 9; read $OM/ABORTED, move it away to resume)"; exit 9
  elif [ "$rc" = 7 ] || late; then CUT=1
  else FAILS=$((FAILS + 1)); fi
  return 0
}

run() {   # run NAME TASK CONFIG ARMS [extra env]
  local name=$1 task=$2 cfg=$3 arms=$4 extra=${5:-}
  job "$name" env $BASE TASKS=$task CONFIG=$cfg ARMS=$arms $extra $PY -X utf8 -u $E/run.py
}
u_ioi_W() { run ioi_W ioi $E/config_v4_ioi.json W; }
u_gt_W() { run gt_W greater_than $E/config_v4_greater_than.json W; }
u_ioi_CF() { run ioi_CF ioi $E/config_v4_ioi.json W_CF "CF_LAMS=$IOI_MID"; }
u_gt_CF() { run gt_CF greater_than $E/config_v4_greater_than.json W_CF "CF_LAMS=$GT_MID"; }
u_ioi_RND() { run ioi_RND ioi $E/config_v4_ioi.json W,RND "RND_LAMS=$IOI_MID N_DRAWS=$DRAWS"; }
u_gt_RND() { run gt_RND greater_than $E/config_v4_greater_than.json W,RND "RND_LAMS=$GT_MID N_DRAWS=$DRAWS"; }
u_ioi_U() { run ioi_U ioi $E/config_v4_ioi.json U; }
u_gt_U() { run gt_U greater_than $E/config_v4_greater_than.json U; }
u_report() {
  job report env $BASE CUDA_VISIBLE_DEVICES= bash -c "$PY -X utf8 $E/report.py > $OM/report.txt 2>&1; \
$PY -X utf8 $E/figure.py > $OM/figure.log 2>&1; tail -40 $OM/report.txt"
}

preflight() {   # read-only; rc 1 on a problem
  local bad=0 f
  for f in "$PY" "$E/run.py" "$E/report.py" "$E/config_v4_ioi.json" "$E/config_v4_greater_than.json" \
           "$PROBE/competence.json" "$E/tasks/gpt2-small/ioi.json" "$E/tasks/gpt2-small/greater_than.json" \
           "$HOME/.cache/huggingface/hub/models--gpt2"; do
    [ -e "$f" ] || { say "  missing: $f"; bad=1; }
  done
  if [ -f "$OM/WCM_VERSION" ] && [ "$(cat "$OM/WCM_VERSION")" != v4 ]; then say "  $OM holds $(cat "$OM/WCM_VERSION") circuits"; bad=1; fi
  command -v nvidia-smi > /dev/null || { say "  nvidia-smi not found"; bad=1; }
  env $BASE CUDA_VISIBLE_DEVICES= $PY -I -c "import sys; sys.path[:0] = ['src', '$E']; import json, task_common as C; \
assert C.WCM_VERSION == 'v4', C.WCM_VERSION; v = C.WCM_VERSIONS['v4']; \
assert v['resample_weight'] == 1.0 and v['lr_schedule'] == 'geometric' and C.WCM_LR_MULT['v4'] == 3.15; \
ok = json.load(open('$PROBE/competence.json'))['tasks']; \
assert ok['ioi']['competent'] and ok['greater_than']['competent']; print('  preflight: task_common v4 ok', v)" \
    || { say "  task_common v4 / competence check failed"; bad=1; }
  return "$bad"
}

# ---------------------------------------------------------------------------------------------- main
say "081 v4 queue | $(git --no-pager log -1 --format='%h' 2> /dev/null) | stop epoch $(hm "$STOP") (hard $(hm "$HARD_DEADLINE")) | watchdog GPU > $GPU_MAX MiB / RAM < $RAM_MIN_GB GB every $EVERY s | mids ioi $IOI_MID gt $GT_MID | draws $DRAWS | cf score $CF_N$([ -n "$ONLY" ] && echo " | only $ONLY")$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
preflight || { say "preflight refused"; [ "$DRY_RUN" = 1 ] || exit 8; }
m=$(gpu_mib)
if [ "$DRY_RUN" = 1 ]; then
  say "dry run: GPU now holds ${m:-?} MiB (refuse above $GPU_IDLE); lock $LOCK $([ -e "$LOCK" ] && echo "EXISTS: $(head -c 200 "$LOCK")" || echo absent); ABORTED $([ -f "$OM/ABORTED" ] && echo EXISTS || echo absent); MemAvailable $(avail_mib) MiB"
else
  [ -e "$LOCK" ] && { say "refused: $LOCK exists: $(head -c 300 "$LOCK")"; exit 8; }
  [ -f "$OM/ABORTED" ] && { say "refused: $OM/ABORTED exists ($(cat "$OM/ABORTED")): read it, move it away to resume"; exit 8; }
  if [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ]; then say "refused: the GPU already holds $m MiB (> $GPU_IDLE)"; exit 8; fi
  [ -n "$m" ] || say "WARNING: nvidia-smi gave no reading; the GPU watchdog is blind"
  echo "exp 081 queue_v4.sh (experiments/081-task-circuits) pid $$ started $(ts) on $(hostname); stop epoch ${STOP_EPOCH:-none}" > "$LOCK" \
    || { say "cannot write $LOCK"; exit 8; }
  trap 'grep -q "pid $$ " "$LOCK" 2> /dev/null && rm -f "$LOCK"' EXIT
  trap '[ -n "$JOB" ] && { kill -TERM -- -"$JOB" 2> /dev/null; sleep 10; kill -KILL -- -"$JOB" 2> /dev/null; }; say "signal: stopped"; exit 143' INT TERM HUP
  mkdir -p "$OM" "$STATE" "$LOGS"
fi

stage ioi_W u_ioi_W
stage gt_W u_gt_W
stage ioi_CF u_ioi_CF
stage gt_CF u_gt_CF
stage ioi_RND u_ioi_RND
stage gt_RND u_gt_RND
stage ioi_U u_ioi_U
stage gt_U u_gt_U
stage report u_report always
if [ "$DRY_RUN" = 1 ]; then say "dry run: total ~$(fmin "$ETA") min (~$(awk -v m="$ETA" 'BEGIN { printf "%.1f", m / 60 }') h) for the stages not done"; exit 0; fi
say "queue finished: $FAILS failed stage(s)$([ "$CUT" = 1 ] && echo '; STOP_EPOCH cut the plan')"
[ "$FAILS" = 0 ] || exit 1
[ "$CUT" = 0 ] || exit 7
exit 0
