#!/usr/bin/env bash
# Sequential per-seed processes (allocator isolation); resume-safe.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
for i in 0 1 2 3; do
  echo "===== SEED_IDX=$i  $(date +%H:%M:%S) ====="
  SEED_IDX=$i PYTHONPATH=src python \
    experiments/006-negctx-floor-grid/runner.py 2>&1 \
    | grep -E "upstream sites|a_pos |free0=|NO CIRCUIT|FAILED|skip|no positive"
done
echo "GRID DONE $(date +%H:%M:%S)"
