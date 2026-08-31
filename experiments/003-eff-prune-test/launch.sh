#!/usr/bin/env bash
# Sequential per-seed processes (allocator isolation); resume-safe.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
for i in 0 1 2 3; do
  echo "===== SEED_IDX=$i  $(date +%H:%M:%S) ====="
  SEED_IDX=$i PYTHONPATH=src python \
    experiments/003-eff-prune-test/runner.py 2>&1 \
    | grep -E "upstream sites|free0=|removed|NO CIRCUIT|FAILED|skip|no positive"
done
echo "EFF TEST DONE $(date +%H:%M:%S)"
