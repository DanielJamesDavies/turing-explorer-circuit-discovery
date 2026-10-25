#!/usr/bin/env bash
# Local GPU queue, 2026-10-11 night: finish the DAN-64 sweep analysis. Both jobs resume where they stopped; re-run this
# script after a restart.
#   wsl -e bash -lc 'cd <repo> && nohup setsid bash experiments/071-contrast-sweep/queue_2026-10-11.sh > experiments/071-contrast-sweep/queue_2026-10-11.log 2>&1 &'
cd "$(dirname "$0")/../.." || exit 1
D=experiments/071-contrast-sweep
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
echo "[$(date +%H:%M)] queue start"

echo "[$(date +%H:%M)] job 1: set geometry on all 280 targets"
OUT=$D/out $PY -X utf8 -u $D/set_geometry.py > $D/geometry_full.log 2>&1
echo "[$(date +%H:%M)] job 1 exit $?"

echo "[$(date +%H:%M)] job 2: near / mid / far C-mean re-score (random_silent, then close_silent)"
OUT=$D/out $PY -X utf8 -u $D/thirds_rescore.py > $D/thirds.log 2>&1
echo "[$(date +%H:%M)] job 2 exit $?"
OUT=$D/out REPORT=1 $PY -X utf8 $D/thirds_rescore.py > /dev/null 2>&1

echo "[$(date +%H:%M)] queue DONE"
