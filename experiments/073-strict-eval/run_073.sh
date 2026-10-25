#!/usr/bin/env bash
# 073/074 full run (747 random-null targets). Resumes from out/rows.jsonl.
#   wsl -e bash -lc 'cd <repo> && nohup setsid bash experiments/073-strict-eval/run_073.sh > experiments/073-strict-eval/run.log 2>&1 &'
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
echo "[$(date +%H:%M)] 073 start"
.venv/bin/python -X utf8 -u experiments/073-strict-eval/strict_eval.py
echo "[$(date +%H:%M)] 073 exit $?"
