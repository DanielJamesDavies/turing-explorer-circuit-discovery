#!/bin/bash
# 2026-11-18 evening GPU chain, strictly sequential (one GPU process at a time):
#   wait for the exp 081 v4 queue (its pid) and its .gpu-lock to go -> exp 128 trim check -> exp 131 off-term queue.
# Each runner takes and releases .gpu-lock itself; this script only orders them. Stop epoch: tomorrow 07:45.
#   bash experiments/128-pod-preflight/chain_1118.sh <pid of 081 queue_v4.sh>
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
WAIT_PID=${1:-}
STOP_EPOCH=$(date -d "tomorrow 07:45" +%s)
export STOP_EPOCH
log() { echo "[chain $(date +%H:%M:%S)] $*"; }
wait_free() {
  while [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2> /dev/null; do sleep 30; done
  while [ -e .gpu-lock ]; do sleep 30; done
  sleep 60
}
log "waiting for pid ${WAIT_PID:-none} and .gpu-lock"
wait_free
log "exp 128 trim check START"
N128_STAGES=trim_identity,a1_fit_deep,a1_score_deep,a1_compare,ram_l1e-3,ram_report N128_SOFT=a1_fit_deep \
  bash experiments/128-pod-preflight/run_night.sh > experiments/128-pod-preflight/out/run_trimcheck_1118.out 2>&1
log "exp 128 trim check END rc $?"
WAIT_PID=""
wait_free
if [ "$(date +%s)" -ge "$STOP_EPOCH" ]; then log "stop epoch reached: exp 131 not started"; exit 0; fi
mkdir -p experiments/131-off-term/out
log "exp 131 off-term START"
bash experiments/131-off-term/queue.sh > experiments/131-off-term/out/queue.out 2>&1
log "exp 131 off-term END rc $?"
