#!/usr/bin/env bash
# 119 #3: the four local strict runs (pythia-70m, gpt2-small) x (held-out C donors, training C donors), sequential.
#   wsl: bash experiments/119-strict-heldout-and-tl-dce/run_strict.sh   (from the repo root)
cd "$(dirname "$0")/../.." || exit 1
E=experiments/119-strict-heldout-and-tl-dce
mkdir -p "$E/out"
for m in pythia-70m gpt2-small; do
  for h in 1 0; do
    PYTHONPATH=src HF_HUB_OFFLINE=1 MODEL=$m C_HELD=$h timeout 2400 .venv/bin/python -X utf8 "$E/strict_heldout.py" \
      > "$E/out/strict_${m}_c${h}.log" 2>&1
    echo "$m c_held=$h exit $?" >> "$E/out/strict_runs.status"
  done
done
echo ALLDONE >> "$E/out/strict_runs.status"
