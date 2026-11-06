#!/usr/bin/env bash
# Adaptive λ on the 24 dev targets, round 3 (2026-11-03 ~02:50). Waits for run_dev24b.sh to exit. Arms (geometric
# lr x3.15, λ start 1e-3, as run_dev24b.sh):
#   dev24_B3_eps0.10_up10          B, band [0.9, 1.2], λ may rise only to x10 (B2 eps 0.10 lost 11.mlp.39113 after a x21
#                                  rise: 737 -> 180 latents, held-out Z 0.70)
#   dev24_B3_eps0.10_up10_censor   + the censored-activation data term (9.attn.38289 failed with Z = 0.00: censoring)
#   dev24_censor                   fixed λ 1e-3 + censored term (the censored term alone, on dev)
# No arm starts after END_BY (default 08:30). WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/adaptive_lambda/run_dev24c.sh > experiments/107-true-baselines/adaptive_lambda/dev24c.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/adaptive_lambda
now=$(date +%s); END_BY=$(date -d "today 08:30" +%s); [ "$END_BY" -le "$now" ] && END_BY=$(date -d "tomorrow 08:30" +%s)
while pgrep -f "run_dev24b.sh" > /dev/null; do sleep 30; done
run_arm() {
  local name=$1; shift
  [ "$(date +%s)" -ge "$END_BY" ] && { echo "[dev24c] END_BY passed before $name"; return; }
  echo "[dev24c] $(date +%H:%M:%S) start $name"
  env "$@" PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=1e-3 \
    LR_SCHEDULE=geometric LR_PEAK_MULT=3.15 OUT="$E/out/$name" CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx \
    TARGETS=$E/targets_dev24.txt CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 .venv/bin/python -u \
    experiments/062-h100-protocol-v1/driver.py || echo "[dev24c] FAILED $name"
  echo "[dev24c] $(date +%H:%M:%S) end $name"
}
run_arm dev24_B3_eps0.10_up10 LAM_ADAPT_EPS=0.10 LAM_ADAPT_UP=10
run_arm dev24_B3_eps0.10_up10_censor LAM_ADAPT_EPS=0.10 LAM_ADAPT_UP=10 CENSOR_T=0.05
run_arm dev24_X_censor CENSOR_T=0.05
echo "[dev24c] $(date +%H:%M:%S) DONE"
