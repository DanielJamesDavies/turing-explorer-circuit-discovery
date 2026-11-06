#!/usr/bin/env bash
# Wednesday night 2026-11-04 (Daniel, 23:14: "Please build and run over night"). ONE GPU process at a time; a
# watchdog kills every driver / donor_test / arena A process at STOP_AT (default 08:30). Stages:
#   1  v4 parity: 3 check-panel targets under WCM_VERSION=v4 (the new pod default), to compare with
#      check_geo_x3.15_warmup_censor (the same settings as overrides): membership should match
#   2  the 7 check-panel targets the 08:30 stop cut from check_geo_x3.15_warmup_censor (resumes)
#   3  donor tests of the check panel's warm-up arms
#   4  arena A's GPU pilot (../arena_a/run_pilot_a.sh), only if ../arena_a/PILOT_READY exists when stage 3 ends
# Every step resumes. WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/latent_pilot/run_tonight2.sh > experiments/107-true-baselines/latent_pilot/tonight2.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/latent_pilot
A=experiments/107-true-baselines/arena_a
PY=.venv/bin/python
now=$(date +%s)
STOP=$(date -d "today ${STOP_AT:-08:30}" +%s); [ "$STOP" -le "$now" ] && STOP=$(date -d "tomorrow ${STOP_AT:-08:30}" +%s)
say() { echo "[tonight2] $(date +%H:%M:%S) $*"; }
say "start; hard stop $(date -d @$STOP '+%a %H:%M')"
(
  while [ "$(date +%s)" -lt "$STOP" ]; do sleep 30; done
  say "STOP_AT reached: killing driver / donor_test / arena A processes"
  pkill -TERM -f "experiments/062-h100-protocol-v1/driver.py" || true
  pkill -TERM -f "src/eval/donor_test.py" || true
  pkill -TERM -f "107-true-baselines/arena_a/" || true
) &
late() { [ "$(date +%s)" -ge "$STOP" ]; }
TL="PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx"
GEO="LR_SCHEDULE=geometric LR_PEAK_MULT=3.15"
CHECK=experiments/102-pass-improvement/check_targets.txt
fit() {   # name targets version extra-env...
  local name=$1 targets=$2 ver=$3; shift 3
  late && return
  say "fit $name ($ver)"
  env $TL WCM_VERSION=$ver "$@" MODE=main LAM=1e-3 OUT="$E/out/$name" TARGETS="$targets" CURVE_STRIDE=1 ORDER=list \
    SKIP_SPEC=1 $PY -u experiments/062-h100-protocol-v1/driver.py > "$E/logs_tonight_$name.log" 2>&1 || say "FAILED fit $name"
}
donor() {   # name targets extra-env...
  local name=$1 targets=$2; shift 2
  late && return
  say "donor $name"
  env $TL WCM_VERSION=v3 $GEO "$@" OUT="$E/out/$name" DONOR_TARGETS="$targets" DONOR_ARMS=main DONOR_VARIANTS=fit \
    DONOR_POOL=experiments/073-strict-eval/out/random_pool.pt $PY -u src/eval/donor_test.py \
    > "$E/logs_tonight_donor_$name.log" 2>&1 || say "FAILED donor $name"
}
fit check_v4_parity $E/targets_v4_parity.txt v4
fit check_geo_x3.15_warmup_censor $CHECK v3 $GEO LR_WARMUP=0.1 CENSOR_T=0.05
donor check_geo_x3.15_warmup $CHECK LR_WARMUP=0.1
donor check_geo_x3.15_warmup_censor $CHECK LR_WARMUP=0.1 CENSOR_T=0.05
say "stage 3 done"
if [ -f "$A/PILOT_READY" ] && ! late; then
  say "arena A pilot"
  STOP_EPOCH=$STOP bash "$A/run_pilot_a.sh" > "$A/pilot_a.log" 2>&1 || say "FAILED arena A pilot"
else
  say "skip arena A pilot ($A/PILOT_READY missing or late)"
fi
say "DONE"
kill %1 2> /dev/null
