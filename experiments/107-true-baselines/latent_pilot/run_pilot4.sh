#!/usr/bin/env bash
# Latent pilot, part 4 (2026-11-02 ~17:40, Daniel: try (1) the censored-activation data term, (4) a heavier rank term
# inside the zero ablation only, (5) a rival term, and (6) log the censored share). Every arm on top of geometric
# x3.15, λ 1e-3, the 6 attention targets (targets_attn.txt), each its own OUT. The engine now logs, per ablation term,
# the share of anchors with the target outside its site's Top-K (curve "censored") and the rank term in loss units
# (curve "rank_norm"). Arms (env -> learned_mask):
#   cut64        RANK_CUT_K=64 (the part-3 arm, rerun: its first run loaded a mid-edit engine file and crashed)
#   censor0.05   CENSOR_T=0.05      censored-activation data term (soft Top-K gate, T = 0.05 x |clean target|)
#   rankzero10   RANK_ZERO_MULT=10  the version's keep rank term x10 inside the zero ablation's term only
#   rival0.1     RIVAL_W=0.1        rivals rising above both their clean value and the target are charged
#   censor+rival CENSOR_T=0.05 RIVAL_W=0.1
# WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/latent_pilot/run_pilot4.sh > experiments/107-true-baselines/latent_pilot/pilot4.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/latent_pilot
run_arm() {
  local name=$1; shift
  echo "[pilot4] $(date +%H:%M:%S) start $name"
  env "$@" PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=1e-3 \
    LR_SCHEDULE=geometric LR_PEAK_MULT=3.15 OUT="$E/out/geo_x3.15_l1e-3_$name" \
    CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx TARGETS="$E/targets_attn.txt" CURVE_STRIDE=1 \
    ORDER=list SKIP_SPEC=1 .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py || echo "[pilot4] FAILED $name"
  echo "[pilot4] $(date +%H:%M:%S) end $name"
}
run_arm cut64 RANK_CUT_K=64
run_arm censor0.05 CENSOR_T=0.05
run_arm rankzero10 RANK_ZERO_MULT=10
run_arm rival0.1 RIVAL_W=0.1
run_arm censor0.05_rival0.1 CENSOR_T=0.05 RIVAL_W=0.1
echo "[pilot4] $(date +%H:%M:%S) DONE"
