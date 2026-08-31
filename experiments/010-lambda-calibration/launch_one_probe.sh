#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
export PYTHONPATH=src
stdbuf -oL -eL env SEED_IDS="${SEED_IDS:-2,5,8,10}" \
  python experiments/010-lambda-calibration/one_probe_calibration.py 2>&1 \
  | grep --line-buffered -vE 'CF-Capture|CFaithfulness|UserWarning|Consider using|mean_m'
