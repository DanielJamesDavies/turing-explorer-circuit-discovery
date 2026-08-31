#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
export PYTHONPATH=src
stdbuf -oL -eL env PER_COMPONENT="${PER_COMPONENT:-2}" \
  python experiments/010-lambda-calibration/probe_steps.py 2>&1 \
  | grep --line-buffered -vE 'CF-Capture|CFaithfulness|UserWarning|Consider using|mean_m'
