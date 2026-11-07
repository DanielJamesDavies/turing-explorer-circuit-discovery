#!/usr/bin/env bash
# Local GPU queue, 2026-10-09 (Daniel out ~10 h). One job at a time (the 16 GB card holds one model + SAE bank).
# Both jobs are resumable: re-run this script to continue after a kill.
#   wsl -e bash -lc 'cd <repo> && nohup setsid bash experiments/062-h100-protocol-v1/queue_2026-10-09.sh > experiments/062-h100-protocol-v1/queue_2026-10-09.log 2>&1 &'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/062-h100-protocol-v1
export PYTHONPATH=src
PY=.venv/bin/python

while pgrep -f "062-h100-protocol-v1/(driver|random_null|stable_core|rescore_mid).py" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] queue start"

echo "[$(date +%H:%M)] job 1: re-score the 58 padded mid-band rows (protocol.md 3.6)"
OUT=$E/out_full/out $PY -X utf8 -u $E/rescore_mid.py > $E/rescore_mid.log 2>&1
echo "[$(date +%H:%M)] job 1 exit $?"

echo "[$(date +%H:%M)] job 2: stable core, 5 per cell (DAN-18)"
OUT=$E/out_full/out N_PER=5 $PY -X utf8 -u $E/stable_core.py >> $E/stable_core.log 2>&1
echo "[$(date +%H:%M)] job 2 exit $?"

echo "[$(date +%H:%M)] queue DONE"
