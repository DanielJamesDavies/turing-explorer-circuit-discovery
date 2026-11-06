#!/usr/bin/env bash
# Thursday 2026-11-05 evening/night GPU queue (Daniel: GPU free after 16:00; stop 08:30). ONE GPU process at a time:
#   1 arena D smoke      arena_d/run_pilot_d.sh            (always; it resumes finished steps)
#   2 arena A smoke      arena_a/run_smoke_local.sh        (only once arena_a/SMOKE_READY exists)
#   3 arena A tuning     arena_a/run_tune_a.sh             (only once arena_a/TUNE_READY exists)
# Each stage waits for its READY flag until the stop time; a missing flag skips the stage. Every stage honours STOP_AT
# itself; this wrapper also refuses to start a stage after the stop time.
#   WSL, repo root:
#     nohup setsid bash experiments/107-true-baselines/latent_pilot/run_evening3.sh \
#         > experiments/107-true-baselines/latent_pilot/evening3.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
B=experiments/107-true-baselines
START_AT=${START_AT:-16:00}
export STOP_AT=${STOP_AT:-08:30}
now=$(date +%s)
START=$(date -d "today $START_AT" +%s)
STOP=$(date -d "today $STOP_AT" +%s); [ "$STOP" -le "$now" ] && STOP=$(date -d "tomorrow $STOP_AT" +%s)
say() { echo "[evening3] $(date +%H:%M:%S) $*"; }
late() { [ "$(date +%s)" -ge "$STOP" ]; }
wait_flag() {  # $1 = flag file; returns 1 if the stop time passes first
  while [ ! -e "$1" ]; do late && return 1; sleep 60; done
}
run_stage() {  # $1 = name, $2 = script, $3 = log
  late && { say "skip $1: past stop time"; return; }
  say "start $1"
  bash "$2" > "$3" 2>&1 < /dev/null
  say "done $1 (exit $?)"
}

while [ "$(date +%s)" -lt "$START" ]; do sleep 30; done
say "queue start (stop $STOP_AT)"

mkdir -p $B/arena_d/out/pilot
run_stage "arena D smoke" $B/arena_d/run_pilot_d.sh $B/arena_d/out/pilot/pilot_d.log

if wait_flag $B/arena_a/SMOKE_READY; then
  # run_smoke_local.sh takes its clock stop as STOP_EPOCH (its STOP_AT is a run_pod.sh stage name, set inside)
  export STOP_EPOCH=$STOP SMOKE_GPU_IDLE_MIB=2000
  run_stage "arena A smoke" $B/arena_a/run_smoke_local.sh $B/arena_a/smoke_local.log
else
  say "skip arena A smoke: SMOKE_READY never appeared"
fi

if wait_flag $B/arena_a/TUNE_READY; then
  run_stage "arena A tuning" $B/arena_a/run_tune_a.sh $B/arena_a/tune_a.log
else
  say "skip arena A tuning: TUNE_READY never appeared"
fi
say "DONE"
