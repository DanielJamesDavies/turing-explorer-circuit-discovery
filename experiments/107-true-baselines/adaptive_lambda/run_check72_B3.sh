#!/usr/bin/env bash
# Confirmation (2026-11-03 ~04:20): the dev winner, adaptive λ B3 (band [0.9, 1.2] on Z / A / C training faithfulness,
# λ from 1e-3 within x[1/30, 10], window 25-70%, rate 0.02) on geometric lr x3.15, on 102's frozen 72-target check panel
# (dev24: 19 / 24 vs fixed 18, +1 / -0, passing-in-both size median 0.72x, p90 1.02x). Compared with production v3
# (54 / 72) and the check_geo_x3.15 arm (fixed λ, 58 / 72) of latent_pilot. Stops the dev round-3 script once its
# B3+censor arm has finished (its last arm, censored-only, is skipped to free the GPU). No start after END_BY 08:30;
# the driver itself does not stop mid-arm. WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/adaptive_lambda/run_check72_B3.sh > experiments/107-true-baselines/adaptive_lambda/check72_B3.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/adaptive_lambda
now=$(date +%s); END_BY=$(date -d "today 08:30" +%s); [ "$END_BY" -le "$now" ] && END_BY=$(date -d "tomorrow 08:30" +%s)
(
  until grep -q -E "end dev24_B3_eps0.10_up10_censor" $E/dev24c.log; do sleep 20; done
  pid=$(pgrep -f "bash $E/run_dev24c.sh" | head -1)
  if [ -n "$pid" ]; then
    pg=$(ps -o pgid= -p "$pid" | tr -d ' ')
    echo "[check72_B3] $(date +%H:%M:%S) stopping run_dev24c (pgid $pg) before its censored-only arm"
    kill -TERM -- "-$pg"
  fi
) &
[ "$(date +%s)" -ge "$END_BY" ] && { echo "[check72_B3] END_BY passed"; exit 0; }
echo "[check72_B3] $(date +%H:%M:%S) start"
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=1e-3 LR_SCHEDULE=geometric \
  LR_PEAK_MULT=3.15 LAM_ADAPT_EPS=0.10 LAM_ADAPT_UP=10 OUT="$E/out/check_B3_eps0.10_up10" \
  CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx TARGETS=experiments/102-pass-improvement/check_targets.txt \
  CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py \
  || echo "[check72_B3] FAILED"
echo "[check72_B3] $(date +%H:%M:%S) DONE"
wait
