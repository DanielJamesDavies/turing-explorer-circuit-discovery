#!/usr/bin/env bash
# Latent check (2026-11-02 evening, Daniel: "Let's go ahead"): the fixes from the 9-target pilot on 102's frozen
# 72-target check panel (experiments/102-pass-improvement/check_targets.txt), against production v3's 54 / 72 there.
# Arms, serial, each its own OUT (062 driver, MODE=main, v3, λ 1e-3, 400 steps, full curve, SKIP_SPEC=1):
#   check_geo_x3.15          geometric lr, peak x3.15 (sum(lr) = constant's)
#   check_geo_x3.15_censor   + the censored-activation data term (CENSOR_T 0.05)
# Caveat: the censored term and the x3.15 multiplier were chosen on 9 / 6 targets OF THIS PANEL, so the check is not
# fully independent for those targets (listed in targets.txt / targets_attn.txt).
# A new arm does not start after END_BY (default 08:30, the overnight stop); the driver is resumable per target. WSL:
#   nohup setsid bash experiments/107-true-baselines/latent_pilot/run_check72.sh > experiments/107-true-baselines/latent_pilot/check72.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/latent_pilot
now=$(date +%s); END_BY=$(date -d "today 08:30" +%s); [ "$END_BY" -le "$now" ] && END_BY=$(date -d "tomorrow 08:30" +%s)
run_arm() {
  local name=$1; shift
  [ "$(date +%s)" -ge "$END_BY" ] && { echo "[check72] END_BY passed before $name"; return; }
  echo "[check72] $(date +%H:%M:%S) start $name"
  env "$@" PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=1e-3 \
    LR_SCHEDULE=geometric LR_PEAK_MULT=3.15 OUT="$E/out/$name" CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx \
    TARGETS=experiments/102-pass-improvement/check_targets.txt CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 \
    .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py || echo "[check72] FAILED $name"
  echo "[check72] $(date +%H:%M:%S) end $name"
}
run_arm check_geo_x3.15
run_arm check_geo_x3.15_censor CENSOR_T=0.05
echo "[check72] $(date +%H:%M:%S) DONE"
