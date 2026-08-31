#!/usr/bin/env bash
# Fine weight-decay sweep across seeds: is the calibration optimum stable, or
# does it need per-seed tuning? Range 0.03-0.1 brackets the L8 optimum.
#   free0 -> 1.0 is the target; above 1.0 is overshoot, not improvement.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
for t in "$@"; do
  echo "===== $t  $(date +%H:%M:%S) ====="
  SEED_TAG="$t" WDS=0.03,0.045,0.06,0.08,0.1 LAMBDA=1e-4 PYTHONPATH=src \
    python experiments/008-learned-mask-spike/wd_sweep.py 2>&1 \
    | grep -vE "extension loaded|topcoactivation|n_gpus|fused_linear|adopting"
done
echo "WD FINE SWEEP DONE $(date +%H:%M:%S)"
