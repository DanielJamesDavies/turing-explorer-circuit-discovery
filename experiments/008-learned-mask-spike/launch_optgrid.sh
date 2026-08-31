#!/usr/bin/env bash
# Optimiser grid on one seed. Usage: launch_optgrid.sh L8
set -u
TAG="${1:-L8}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
run () {
  echo "===== $TAG  opt=$1 steps=$2 lr=$3  $(date +%H:%M:%S) ====="
  SEED_TAG="$TAG" OPT="$1" STEPS="$2" LR="$3" WD="${4:-0.0}" PYTHONPATH=src \
    python experiments/008-learned-mask-spike/lambda_sweep.py 2>&1 \
    | grep -vE "extension loaded|topcoactivation|n_gpus|fused_linear|adopting"
}
run adam 400 0.1
run adam 400 0.05
echo "OPT GRID DONE $(date +%H:%M:%S)"
