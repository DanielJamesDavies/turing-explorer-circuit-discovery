#!/bin/bash
# 092 queue: wait for 091 to leave the GPU (poll every 60 s), then the step-0 check, then the three tests in order.
# Resumable: run.py skips (target, arm) rows already in out/rows.jsonl and reloads fitted circuits.
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
E=experiments/092-v2-speedups
while pgrep -f "experiments/091-v2-resample-small/run.py" > /dev/null; do sleep 60; done
echo "092 queue: GPU free at $(date)"
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
if [ "${SKIP_CHECK:-0}" != "1" ]; then
  CHECK=1 .venv/bin/python -u $E/run.py > $E/check.log 2>&1
  echo "092 queue: check exit $? at $(date)"
fi
.venv/bin/python -u $E/run.py >> $E/run.log 2>&1
echo "092 queue: run exit $? at $(date)"
