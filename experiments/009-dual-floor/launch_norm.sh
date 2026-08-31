#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
export PYTHONPATH=src
# stdbuf + --line-buffered: without both, the filter block-buffers and short
# output never reaches the log, which reads identically to a hung process.
stdbuf -oL -eL python experiments/dual-sweep-2026-07-29/norm_diagnostic.py 2>&1 \
  | grep --line-buffered -vE 'CF-Capture|CFaithfulness|UserWarning|Consider using|mean_m'
