#!/usr/bin/env bash
# Selection-bias check for the recommended setting (2026-11-03 daytime, Daniel: compute until 14:00). The 48 dev targets
# never used to choose anything (093's 72 dev targets minus adaptive_lambda's 24; targets_dev48.txt), under
#   dev48_geo_x3.15          geometric lr x3.15, fixed λ 1e-3
#   dev48_geo_x3.15_censor   + the censored-activation term (CENSOR_T 0.05)
# both started now as two parallel driver processes (062 driver, MODE=main, v3, 400 steps, full curve, SKIP_SPEC=1).
# Hard stop: at STOP_AT (default 13:45) both drivers are killed by PID; the driver resumes per target. WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/latent_pilot/run_dev48.sh > experiments/107-true-baselines/latent_pilot/dev48.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/latent_pilot
now=$(date +%s); STOP=$(date -d "today ${STOP_AT:-13:45}" +%s); [ "$STOP" -le "$now" ] && STOP=$(date -d "tomorrow ${STOP_AT:-13:45}" +%s)
start() {
  local name=$1; shift
  echo "[dev48] $(date +%H:%M:%S) start $name"
  env "$@" PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=1e-3 \
    LR_SCHEDULE=geometric LR_PEAK_MULT=3.15 OUT="$E/out/$name" CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx \
    TARGETS=$E/targets_dev48.txt CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 .venv/bin/python -u \
    experiments/062-h100-protocol-v1/driver.py > "$E/dev48_$name.log" 2>&1 &
}
start dev48_geo_x3.15
P1=$!
sleep 60
start dev48_geo_x3.15_censor CENSOR_T=0.05
P2=$!
echo "[dev48] driver PIDs $P1 $P2"
while kill -0 $P1 2> /dev/null || kill -0 $P2 2> /dev/null; do
  if [ "$(date +%s)" -ge "$STOP" ]; then
    echo "[dev48] $(date +%H:%M:%S) STOP_AT reached: stopping the drivers"
    kill -TERM $P1 $P2 2> /dev/null
    break
  fi
  sleep 30
done
wait
echo "[dev48] $(date +%H:%M:%S) DONE"
