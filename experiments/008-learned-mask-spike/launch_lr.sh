#!/usr/bin/env bash
# lr sweep with steps*lr*wd held at 1.0. Usage: launch_lr.sh L8 L2 L10
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
for t in "$@"; do
  echo "===== $t  $(date +%H:%M:%S) ====="
  SEED_TAG="$t" PYTHONPATH=src \
    python experiments/008-learned-mask-spike/lr_sweep.py 2>&1 \
    | grep -vE "extension loaded|topcoactivation|n_gpus|fused_linear|adopting"
done
echo "LR SWEEP DONE $(date +%H:%M:%S)"
