#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
export PYTHONPATH=src
# Wait for the one-probe calibration to release the GPU. L10 dual peaks near
# 15.9/16.3 GB, so a second job would spill into WDDM shared memory (~5x).
while pgrep -f "one_probe_calibration.py" >/dev/null; do sleep 20; done
stdbuf -oL -eL env COMPONENTS="${COMPONENTS:-8,32}" \
  N_PER_COMPONENT="${N_PER_COMPONENT:-5}" \
  python experiments/010-lambda-calibration/within_component.py 2>&1 \
  | grep --line-buffered -vE 'CF-Capture|CFaithfulness|UserWarning|Consider using|mean_m'
