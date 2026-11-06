#!/usr/bin/env bash
# Adaptive λ on 24 dev targets, v2 (2026-11-02 ~23:30; replaces run_dev24.sh, whose B arms used the first controller's
# eps scale). Dev = every third of 093's 72 dev targets (targets_dev24.txt; disjoint from 102's check panel). Every arm:
# 062 driver, MODE=main, v3, 400 steps, geometric lr x3.15, full curve, SKIP_SPEC=1, each its own OUT:
#   dev24_base               fixed λ 1e-3 (the reference; resumes run_dev24.sh's partial OUT)
#   dev24_C_l<λ>             option C: c_assign.py's per-target λ (one run per assigned λ)
#   dev24_B2_eps0.15         option B: adaptive λ from 1e-3, keep each term's training faithfulness in [0.85, 1.3]
#   dev24_B2_eps0.10         option B, band [0.90, 1.2]
#   dev24_B2_eps0.15_censor  option B eps 0.15 + the censored-activation data term (CENSOR_T 0.05)
# B2 = the controller as of 23:20 (faithfulness band on Z / A / C, window 25%-70%, rate 0.02). No arm starts after
# END_BY (default 08:30). WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/adaptive_lambda/run_dev24b.sh > experiments/107-true-baselines/adaptive_lambda/dev24b.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/adaptive_lambda
now=$(date +%s); END_BY=$(date -d "today 08:30" +%s); [ "$END_BY" -le "$now" ] && END_BY=$(date -d "tomorrow 08:30" +%s)
run_arm() {
  local name=$1 targets=$2 lam=$3; shift 3
  [ "$(date +%s)" -ge "$END_BY" ] && { echo "[dev24b] END_BY passed before $name"; return; }
  echo "[dev24b] $(date +%H:%M:%S) start $name (λ $lam, $(wc -l < "$targets") targets)"
  env "$@" PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=$lam \
    LR_SCHEDULE=geometric LR_PEAK_MULT=3.15 OUT="$E/out/$name" CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx \
    TARGETS="$targets" CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py \
    || echo "[dev24b] FAILED $name"
  echo "[dev24b] $(date +%H:%M:%S) end $name"
}
run_arm dev24_base $E/targets_dev24.txt 1e-3
for f in $E/targets_dev24_C_*.txt; do
  lam=$(basename "$f" .txt); lam=${lam#targets_dev24_C_}
  run_arm "dev24_C_l$lam" "$f" "$lam"
done
run_arm dev24_B2_eps0.15 $E/targets_dev24.txt 1e-3 LAM_ADAPT_EPS=0.15
run_arm dev24_B2_eps0.10 $E/targets_dev24.txt 1e-3 LAM_ADAPT_EPS=0.10
run_arm dev24_B2_eps0.15_censor $E/targets_dev24.txt 1e-3 LAM_ADAPT_EPS=0.15 CENSOR_T=0.05
echo "[dev24b] $(date +%H:%M:%S) DONE"
