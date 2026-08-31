#!/usr/bin/env bash
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
for t in "$@"; do
  echo "===== $t ====="
  SEED_TAG=$t PYTHONPATH=src python \
    experiments/007-direct-drivers/runner.py 2>&1 \
    | grep -vE "extension loaded|topcoactivation|n_gpus|fused_linear|adopting"
done
