#!/usr/bin/env bash
# Friday 2026-11-06 evening/night GPU queue (Daniel: GPU free; stop Sat 10:30). ONE GPU process at a time:
#   0 wait for the a2 lambda check (arena_a/run_a2lam.sh) to exit and the card to go idle
#   1 owt32k data pass        arena_a/out/owt32k/run_gpu1.sh                   once its GPU1_READY exists (Daniel: first)
#   2 batch-size experiment   experiments/108-batch-size/run.sh                once experiments/108-batch-size/READY exists
#   3 arena D lr-schedule test experiments/107-true-baselines/arena_d/lrtest/run.sh   once its READY exists
#   4 owt32k WCM fits + scoring arena_a/out/owt32k/run_gpu2.sh                 once its GPU2_READY exists
# Stage 1 waits for its flag only until OW1_GIVEUP_AT [18:00]; if late it runs after stage 3 instead, so it never
# holds up the batch experiment. Other flags wait until the stop time.
# A flag that never appears before the stop time skips its stage. Every stage gets STOP_EPOCH (no new fit after it).
#   WSL, repo root:
#     nohup setsid bash experiments/107-true-baselines/latent_pilot/run_night5.sh \
#         > experiments/107-true-baselines/latent_pilot/night5.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
export PATH="$PATH:/usr/lib/wsl/lib"
B=experiments/107-true-baselines
STOP_AT=${STOP_AT:-10:30}
now=$(date +%s)
STOP=$(date -d "today $STOP_AT" +%s); [ "$STOP" -le "$now" ] && STOP=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$STOP
say() { echo "[night5] $(date '+%a %H:%M:%S') $*"; }
late() { [ "$(date +%s)" -ge "$STOP" ]; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
wait_flag() {  # $1 = flag file, $2 = optional give-up epoch (default: the stop time)
  local until=${2:-$STOP}
  while [ ! -e "$1" ]; do { late || [ "$(date +%s)" -ge "$until" ]; } && return 1; sleep 60; done
}
OW1_GIVEUP=$(date -d "today ${OW1_GIVEUP_AT:-18:00}" +%s)   # stage 1 must not hold up the batch experiment past this
wait_idle() {  # until no a2lam / earlier stage python is on the card (< 3000 MiB used) for 2 checks in a row
  local ok=0
  while [ $ok -lt 2 ]; do
    late && return 1
    m=$(gpu_mib); if [ -n "$m" ] && [ "$m" -lt 3000 ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 30
  done
}
run_stage() {  # $1 = name, $2 = script, $3 = log
  late && { say "skip $1: past stop time"; return; }
  wait_idle || { say "skip $1: card never went idle before the stop time"; return; }
  say "start $1 (GPU $(gpu_mib) MiB)"
  bash "$2" > "$3" 2>&1 < /dev/null
  say "done $1 (exit $?)"
}
say "queue start (stop $(date -d @$STOP '+%a %H:%M'))"

# 0: the a2 lambda check runs from arena_a/run_a2lam.sh (pattern bracketed so this script never matches itself)
while pgrep -f "run_a2la[m]\.sh" > /dev/null; do late && break; sleep 60; done
say "a2 lambda check not running (or past stop time)"

OW=$B/arena_a/out/owt32k
if wait_flag $OW/GPU1_READY "$OW1_GIVEUP"; then
  run_stage "owt32k data pass" $OW/run_gpu1.sh $OW/run_gpu1.console.log
else
  say "skip owt32k data pass for now: GPU1_READY not there by ${OW1_GIVEUP_AT:-18:00} (it then runs after the arena D test)"
  OW1_LATE=1
fi

if wait_flag experiments/108-batch-size/READY; then
  run_stage "batch-size experiment" experiments/108-batch-size/run.sh experiments/108-batch-size/run.console.log
else
  say "skip batch-size experiment: READY never appeared"
fi

if wait_flag $B/arena_d/lrtest/READY; then
  run_stage "arena D lr test" $B/arena_d/lrtest/run.sh $B/arena_d/lrtest/run.console.log
else
  say "skip arena D lr test: READY never appeared"
fi

if [ "${OW1_LATE:-0}" = 1 ] && wait_flag $OW/GPU1_READY; then
  run_stage "owt32k data pass (late)" $OW/run_gpu1.sh $OW/run_gpu1.console.log
fi
if wait_flag $OW/GPU2_READY; then
  run_stage "owt32k WCM fits + scoring" $OW/run_gpu2.sh $OW/run_gpu2.console.log
else
  say "skip owt32k WCM fits + scoring: GPU2_READY never appeared"
fi
say "DONE"
