#!/bin/bash
# 092 queue 2: after queue.sh's run exits, the optional t3b arm (Z + A + 091's raw R, no C).
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
E=experiments/092-v2-speedups
until grep -q "run exit" $E/queue.log; do sleep 30; done
echo "092 queue2: start $(date)"
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 ARMS=t3b_ZAR_rawR .venv/bin/python -u $E/run.py >> $E/run.log 2>&1
echo "092 queue2: run exit $? at $(date)"
