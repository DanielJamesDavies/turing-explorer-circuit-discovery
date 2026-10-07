#!/usr/bin/env bash
# Local GPU job queue (2026-10-02, Daniel out for the day). One job at a time: the 16 GB card holds one model + SAE
# bank. Waits for any running driver.py to finish first. A failed job is logged and the queue moves on.
#   wsl -e bash -lc 'cd <repo> && nohup setsid bash experiments/062-h100-protocol-v1/queue_local.sh > .../queue.log 2>&1 &'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/062-h100-protocol-v1
export PYTHONPATH=src
PY=.venv/bin/python

while pgrep -f "062-h100-protocol-v1/driver.py" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] queue start"

echo "[$(date +%H:%M)] job 2: alpha = 1 re-scoring (DAN-20)"
OUT=$E/out_full/out $PY -X utf8 -u $E/rescore_alpha1.py > $E/out_alpha1.log 2>&1
echo "[$(date +%H:%M)] job 2 exit $?"

echo "[$(date +%H:%M)] job 3: hunt case-study validation (DAN-138)"
OUT=$E/out_full/out MEMBERS=experiments/063-case-studies/results_full_deep/members \
  KEYS=10.resid.15497,9.resid.20419,9.resid.17596,8.resid.31095 \
  $PY -X utf8 -u experiments/063-case-studies/validate.py > $E/validate_hunt.log 2>&1
echo "[$(date +%H:%M)] job 3 exit $?"

echo "[$(date +%H:%M)] job 4: deep targets at lambda 2.5e-5 (DAN-131)"
MODE=sweep ORDER=list TARGETS=$E/targets_depth32.txt OUT=$E/out_depth SWEEP_ARMS=W_2.5e-5 \
  $PY -X utf8 -u $E/driver.py >> $E/out_depth/run.log 2>&1
echo "[$(date +%H:%M)] job 4 exit $?"

echo "[$(date +%H:%M)] queue DONE"
