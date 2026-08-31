#!/usr/bin/env bash
# D1 bake-off: one seed per process, sequential, 12 seeds.
set -u
cd "$(dirname "$0")/../../.."
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
    echo "=== SEED_IDX=$i $(date '+%H:%M:%S') ==="
    SEED_IDX=$i PYTHONPATH=src .venv/bin/python \
        experiments/012-driver-bakeoff/runner.py
done
echo "=== ALL SEEDS DONE $(date '+%H:%M:%S') ==="
