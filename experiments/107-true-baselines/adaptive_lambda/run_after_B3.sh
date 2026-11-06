#!/usr/bin/env bash
# After the check-panel B3 run (2026-11-03 ~04:50): B3 lost 3 of the first 24 check targets (necessity 0.77 at 24
# latents; held-out A 2.06 overshoot; held-out Z 0.70) while shrinking passing ones to ~0.53x. A conservative variant,
# B4 = the same controller with λ allowed to rise only x3, on the first 36 check targets (check_targets_first36.txt,
# where the losses cluster), into its own OUT. Waits for run_check72_B3.sh to exit; no start after 07:30 (it needs ~1 h);
# a watchdog kills every 062 driver process at 08:30 (Daniel's stop). WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/adaptive_lambda/run_after_B3.sh > experiments/107-true-baselines/adaptive_lambda/after_B3.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/adaptive_lambda
now=$(date +%s)
LAST_START=$(date -d "today 07:30" +%s); [ "$LAST_START" -le "$now" ] && LAST_START=$(date -d "tomorrow 07:30" +%s)
HARD_STOP=$(date -d "today 08:30" +%s); [ "$HARD_STOP" -le "$now" ] && HARD_STOP=$(date -d "tomorrow 08:30" +%s)
(
  while [ "$(date +%s)" -lt "$HARD_STOP" ]; do sleep 30; done
  echo "[after_B3] $(date +%H:%M:%S) 08:30 hard stop: killing driver processes"
  pkill -TERM -f "experiments/062-h100-protocol-v1/driver.py" || true
) &
head -36 experiments/102-pass-improvement/check_targets.txt > $E/check_targets_first36.txt
while pgrep -f "run_check72_B3.sh" > /dev/null; do sleep 30; done
if [ "$(date +%s)" -ge "$LAST_START" ]; then
  echo "[after_B3] $(date +%H:%M:%S) too late to start B4 (needs ~1 h before 08:30)"
else
  echo "[after_B3] $(date +%H:%M:%S) start B4 (up x3) on 36 check targets"
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=1e-3 LR_SCHEDULE=geometric \
    LR_PEAK_MULT=3.15 LAM_ADAPT_EPS=0.10 LAM_ADAPT_UP=3 OUT="$E/out/check36_B4_eps0.10_up3" \
    CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx TARGETS=$E/check_targets_first36.txt \
    CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py \
    || echo "[after_B3] FAILED B4"
  echo "[after_B3] $(date +%H:%M:%S) end B4"
fi
echo "[after_B3] $(date +%H:%M:%S) DONE (watchdog stays until 08:30)"
wait
