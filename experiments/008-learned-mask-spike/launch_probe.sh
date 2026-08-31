#!/usr/bin/env bash
# Probe-count sweep + the steps-matched companion that settles whether low lr
# is under-trained or just under-sparsified.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate

for t in "$@"; do
  echo "===== PROBE SWEEP $t  $(date +%H:%M:%S) ====="
  SEED_TAG="$t" PYTHONPATH=src \
    python experiments/008-learned-mask-spike/probe_sweep.py 2>&1 \
    | grep -vE "extension loaded|topcoactivation|n_gpus|fused_linear|adopting"
done

# Companion: lr 0.01 with 4x the steps matches lr 0.05 / 400 on BOTH budgets
# (steps*lr*lambda and steps*lr*wd). If it lands at the same size with better
# calibration, low lr was under-trained; if it simply matches, only the
# products matter and lr is pure overhead.
echo "===== STEPS-MATCHED COMPANION (L8, lr 0.01 x 1600 steps)  $(date +%H:%M:%S) ====="
SEED_TAG=L8 LRS=0.01 STEPS=1600 PRODUCT=1.0 PYTHONPATH=src \
  python experiments/008-learned-mask-spike/lr_sweep.py 2>&1 \
  | grep -vE "extension loaded|topcoactivation|n_gpus|fused_linear|adopting"
echo "DONE $(date +%H:%M:%S)"
