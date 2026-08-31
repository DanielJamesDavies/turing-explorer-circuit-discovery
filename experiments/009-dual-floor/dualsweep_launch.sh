#!/usr/bin/env bash
# Per-seed process isolation: a crash or OOM on a deep seed cannot take the
# run with it. Resume by re-running — completed (seed, arm) pairs are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
export PYTHONPATH=src
for i in ${SEEDS:-2 5 8 10}; do
  echo "################ SEED_IDX=$i ################"
  SEED_IDX=$i python experiments/dual-sweep-2026-07-29/runner.py || \
    echo "!!! seed $i exited non-zero — continuing"
done
