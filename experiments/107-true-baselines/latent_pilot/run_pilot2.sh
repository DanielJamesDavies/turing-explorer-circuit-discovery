#!/usr/bin/env bash
# Latent pilot, part 2 (2026-11-02 ~16:30, Daniel: geometric circuits are larger, why and how to get around it).
# Geometric's lr integral is ~0.32x constant's (0.05 x 400 x 0.317 = 6.3 vs 20), and a gate must walk theta from
# theta_init 4 across 0 at ~lr per step to be pruned, so fewer latents finish the walk. Arm geo_x3.15_l1e-3: geometric
# with the peak lr x 3.15 (LR_PEAK_MULT), restoring constant's sum(lr) = 20 (and so the sparsity and weight-decay
# budgets) at production λ 1e-3. lr / T then stays at 3.15x constant's starting value (constant reaches 20x by the end).
# Same driver settings and targets as run_pilot.sh. WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/latent_pilot/run_pilot2.sh > experiments/107-true-baselines/latent_pilot/pilot2.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/latent_pilot
for spec in 3.15:1e-3; do
  mult=${spec%%:*}; lam=${spec#*:}
  echo "[pilot2] $(date +%H:%M:%S) start geometric x$mult λ $lam"
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=$lam LR_SCHEDULE=geometric \
    LR_PEAK_MULT=$mult OUT="$E/out/geo_x${mult}_l${lam}" CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx \
    TARGETS="$E/targets.txt" CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 .venv/bin/python -u \
    experiments/062-h100-protocol-v1/driver.py || echo "[pilot2] FAILED $spec"
  echo "[pilot2] $(date +%H:%M:%S) end $spec"
done
echo "[pilot2] $(date +%H:%M:%S) DONE"
