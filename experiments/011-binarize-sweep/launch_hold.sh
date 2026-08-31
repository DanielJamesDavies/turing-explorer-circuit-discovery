#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
export PYTHONPATH=src
stdbuf -oL -eL python experiments/011-binarize-sweep/anneal_hold.py 2>&1 \
  | grep --line-buffered -vE 'CF-Capture|CFaithfulness|UserWarning|Consider using|mean_m'
