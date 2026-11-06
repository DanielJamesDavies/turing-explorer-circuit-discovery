#!/usr/bin/env bash
# Adaptive λ on 24 dev targets (2026-11-02 night, Daniel: build B and C and run them on some targets). Dev = every third
# of 093's 72 dev targets (targets_dev24.txt; disjoint from 102's check panel). Every arm: 062 driver, MODE=main, v3,
# 400 steps, geometric lr x3.15, full curve, SKIP_SPEC=1, each its own OUT:
#   dev24_base        fixed λ 1e-3 (the reference)
#   dev24_C_l<λ>      option C: c_assign.py's per-target λ (one run per assigned λ, targets_dev24_C_<λ>.txt)
#   dev24_B_eps0.03   option B: adaptive λ from 1e-3, worst-term relative error target 0.03 (rate 0.05, first 70%)
#   dev24_B_eps0.01   option B, eps 0.01
# No arm starts after END_BY (default 08:30). WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/adaptive_lambda/run_dev24.sh > experiments/107-true-baselines/adaptive_lambda/dev24.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/adaptive_lambda
now=$(date +%s); END_BY=$(date -d "today 08:30" +%s); [ "$END_BY" -le "$now" ] && END_BY=$(date -d "tomorrow 08:30" +%s)
run_arm() {
  local name=$1 targets=$2 lam=$3; shift 3
  [ "$(date +%s)" -ge "$END_BY" ] && { echo "[dev24] END_BY passed before $name"; return; }
  echo "[dev24] $(date +%H:%M:%S) start $name (λ $lam, $(wc -l < "$targets") targets)"
  env "$@" PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=$lam \
    LR_SCHEDULE=geometric LR_PEAK_MULT=3.15 OUT="$E/out/$name" CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx \
    TARGETS="$targets" CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py \
    || echo "[dev24] FAILED $name"
  echo "[dev24] $(date +%H:%M:%S) end $name"
}
run_arm dev24_base $E/targets_dev24.txt 1e-3
for f in $E/targets_dev24_C_*.txt; do
  lam=$(basename "$f" .txt); lam=${lam#targets_dev24_C_}
  run_arm "dev24_C_l$lam" "$f" "$lam"
done
run_arm dev24_B_eps0.03 $E/targets_dev24.txt 1e-3 LAM_ADAPT_EPS=0.03
run_arm dev24_B_eps0.01 $E/targets_dev24.txt 1e-3 LAM_ADAPT_EPS=0.01
echo "[dev24] $(date +%H:%M:%S) DONE"
