#!/usr/bin/env bash
# C raise-only on the 72-target check panel (2026-11-03 ~05:40): c_assign.py's λ for the check panel
# (targets_check72_C_*.txt); only the targets C RAISES above 1e-3 are fitted (λ 2e-3: 4, 4e-3: 29), the rest keep
# latent_pilot's check_geo_x3.15 fits (fixed λ 1e-3) when the hybrid is composed. Geometric lr x3.15 as every arm.
# Dev: +1 / -0 vs fixed, passing-in-both size median 0.81x, max 1.00x. Waits for run_check72_B3.sh to exit, then
# runs alongside run_after_B3.sh's B4 arm; no start after 07:45; run_after_B3.sh's watchdog kills drivers at 08:30.
#   nohup setsid bash experiments/107-true-baselines/adaptive_lambda/run_check72_Cup.sh > experiments/107-true-baselines/adaptive_lambda/check72_Cup.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/adaptive_lambda
now=$(date +%s); LAST_START=$(date -d "today 07:45" +%s); [ "$LAST_START" -le "$now" ] && LAST_START=$(date -d "tomorrow 07:45" +%s)
while pgrep -f "run_check72_B3.sh" > /dev/null; do sleep 30; done
for lam in 0.004 0.002; do
  [ "$(date +%s)" -ge "$LAST_START" ] && { echo "[check72_Cup] too late for λ $lam"; continue; }
  echo "[check72_Cup] $(date +%H:%M:%S) start λ $lam"
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=$lam LR_SCHEDULE=geometric \
    LR_PEAK_MULT=3.15 OUT="$E/out/check72_C_l$lam" CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx \
    TARGETS=$E/targets_check72_C_$lam.txt CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 .venv/bin/python -u \
    experiments/062-h100-protocol-v1/driver.py || echo "[check72_Cup] FAILED λ $lam"
  echo "[check72_Cup] $(date +%H:%M:%S) end λ $lam"
done
echo "[check72_Cup] $(date +%H:%M:%S) DONE"
