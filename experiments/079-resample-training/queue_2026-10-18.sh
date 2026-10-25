#!/usr/bin/env bash
# Local GPU queue, 2026-10-18 (one job at a time; a failed job is logged and the queue moves on; every job resumes).
#   1. 079 silent-position rerun (run_silent.sh, ~2 h)
#   2. 076 bag size: 2-target smoke (out_smoke), then 270 targets (30 per band x kind cell)
#   3. 075 reader-recovered / KL: all 747 random-null targets (~35-45 min)
#   wsl -e bash -lc 'cd "<repo>" && nohup setsid bash experiments/079-resample-training/queue_2026-10-18.sh > experiments/079-resample-training/queue_2026-10-18.log 2>&1 < /dev/null & disown; sleep 3'
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
E=experiments
echo "[$(date +%H:%M)] queue start"

echo "[$(date +%H:%M)] job 1: 079 silent rerun"
bash $E/079-resample-training/run_silent.sh > $E/079-resample-training/silent.log 2>&1
echo "[$(date +%H:%M)] job 1 exit $?"

echo "[$(date +%H:%M)] job 2a: 076 smoke (2 targets)"
LIMIT=2 OUT76=$E/076-bag-size/out_smoke $PY -X utf8 -u $E/076-bag-size/bag_size.py > $E/076-bag-size/smoke.log 2>&1
echo "[$(date +%H:%M)] job 2a exit $?"
echo "[$(date +%H:%M)] job 2b: 076 (270 targets)"
LIMIT=270 $PY -X utf8 -u $E/076-bag-size/bag_size.py > $E/076-bag-size/run.log 2>&1
echo "[$(date +%H:%M)] job 2b exit $?"
$PY $E/076-bag-size/report.py > /dev/null 2>&1 && echo "  076 report written"

echo "[$(date +%H:%M)] job 3: 075 (747 targets)"
$PY -X utf8 -u $E/075-reader-recovered/reader_recovered.py > $E/075-reader-recovered/run.log 2>&1
echo "[$(date +%H:%M)] job 3 exit $?"

echo "[$(date +%H:%M)] queue DONE"
