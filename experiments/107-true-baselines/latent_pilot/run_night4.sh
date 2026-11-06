#!/usr/bin/env bash
# Friday 2026-11-06 early-morning GPU rerun queue (stop 08:30). ONE GPU process at a time:
#   1 arena D smoke rerun  arena_d/run_pilot_d.sh       once arena_d/RERUN_READY exists (resumes finished steps)
#   2 arena A smoke rerun  arena_a/run_smoke_local.sh   once arena_a/RERUN_READY exists (its one line = env, e.g. AA_FROM=P6)
# A flag that never appears before the stop time skips its stage.
#   WSL, repo root:
#     nohup setsid bash experiments/107-true-baselines/latent_pilot/run_night4.sh \
#         > experiments/107-true-baselines/latent_pilot/night4.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
B=experiments/107-true-baselines
export STOP_AT=${STOP_AT:-08:30}
now=$(date +%s)
STOP=$(date -d "today $STOP_AT" +%s); [ "$STOP" -le "$now" ] && STOP=$(date -d "tomorrow $STOP_AT" +%s)
say() { echo "[night4] $(date +%H:%M:%S) $*"; }
late() { [ "$(date +%s)" -ge "$STOP" ]; }
wait_flag() { while [ ! -e "$1" ]; do late && return 1; sleep 60; done; }
run_stage() {  # $1 = name, $2 = script, $3 = log
  late && { say "skip $1: past stop time"; return; }
  say "start $1"
  bash "$2" > "$3" 2>&1 < /dev/null
  say "done $1 (exit $?)"
}
say "queue start (stop $STOP_AT)"

if wait_flag $B/arena_d/RERUN_READY; then
  run_stage "arena D smoke rerun" $B/arena_d/run_pilot_d.sh $B/arena_d/out/pilot/pilot_d.rerun.log
else
  say "skip arena D rerun: RERUN_READY never appeared"
fi

if wait_flag $B/arena_a/RERUN_READY; then
  for kv in $(head -n1 $B/arena_a/RERUN_READY); do
    case "$kv" in [A-Z_]*=*) export "$kv"; say "env $kv";; *) say "ignored token $kv";; esac
  done
  export STOP_EPOCH=$STOP SMOKE_GPU_IDLE_MIB=2000
  run_stage "arena A smoke rerun" $B/arena_a/run_smoke_local.sh $B/arena_a/smoke_local.rerun.log
else
  say "skip arena A rerun: RERUN_READY never appeared"
fi
say "DONE"
