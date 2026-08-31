#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
export PYTHONPATH=src
export ARMS=hot
for t in L8 L2 L10; do
  echo "=================== $t ==================="
  SEED_TAG=$t python experiments/008-learned-mask-spike/schedule_sweep.py
done
