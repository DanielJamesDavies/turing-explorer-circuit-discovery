#!/usr/bin/env bash
# Per-seed processes (allocator isolation); resume-safe. Usage: launch.sh 0 9
set -u
START="${1:-0}"; END="${2:-9}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
for i in $(seq "$START" "$END"); do
  SEED_IDX=$i PYTHONPATH=src python \
    experiments/preact-actonly-2026-07-24/runner.py 2>&1 \
    | grep -vE "extension loaded|topcoactivation|n_gpus|fused_linear|adopting"
done
