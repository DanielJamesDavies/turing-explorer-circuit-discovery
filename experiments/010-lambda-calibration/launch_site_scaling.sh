#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
export PYTHONPATH=src
# Wait for any in-flight calibration run to release the GPU rather than
# contend for VRAM - L10 dual already peaks at ~15.9/16.3 GB, so two jobs
# would spill into WDDM shared memory and cost ~5x.
while pgrep -f "010-lambda-calibration/runner.py" >/dev/null; do sleep 20; done
stdbuf -oL -eL env SEED_IDS="${SEED_IDS:-5,8}" \
  python experiments/010-lambda-calibration/site_scaling.py 2>&1 \
  | grep --line-buffered -vE 'CF-Capture|CFaithfulness|UserWarning|Consider using|mean_m'
