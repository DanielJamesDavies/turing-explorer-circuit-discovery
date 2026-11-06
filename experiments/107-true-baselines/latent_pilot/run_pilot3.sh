#!/usr/bin/env bash
# Latent pilot, part 3 (2026-11-02 ~17:30, Daniel: "could we try something that tries to get it in the top 32").
# The failing attention targets are near-threshold (clean rank 65-117 of k 128 on their own top contexts,
# clean_ranks.py). Under geometric x3.15 their pre-activation is restored under Z (~1.0) and the "keep" rank term is
# ~0 in training, yet the scorer finds them censored (Z activation read 0.00-0.69), on training contexts too: training
# (soft gates, T = 0.05) and scoring (hard circuit) disagree on the rivals. Arms: the rank term in "cut" mode against the
# 32nd / 64th rival (driver RANK_CUT_K), at v3's rank weight 1e-1, on top of geometric x3.15, λ 1e-3. Targets
# (targets_attn.txt): the 4 failing attention targets + 2 passing ones (9.attn.22192, 7.attn.17162). WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/latent_pilot/run_pilot3.sh > experiments/107-true-baselines/latent_pilot/pilot3.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/latent_pilot
for kc in 32 64; do
  echo "[pilot3] $(date +%H:%M:%S) start rankcut $kc"
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=1e-3 LR_SCHEDULE=geometric \
    LR_PEAK_MULT=3.15 RANK_CUT_K=$kc OUT="$E/out/geo_x3.15_l1e-3_cut$kc" \
    CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx TARGETS="$E/targets_attn.txt" CURVE_STRIDE=1 ORDER=list \
    SKIP_SPEC=1 .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py || echo "[pilot3] FAILED $kc"
  echo "[pilot3] $(date +%H:%M:%S) end rankcut $kc"
done
echo "[pilot3] $(date +%H:%M:%S) DONE"
