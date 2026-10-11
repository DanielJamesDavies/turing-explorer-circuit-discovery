#!/usr/bin/env bash
# GPU checks for the H100 code, queued behind the stable-core run (2026-10-09). One job at a time.
#   wsl -e bash -lc 'cd <repo> && nohup setsid bash experiments/062-h100-protocol-v1/queue_2026-10-09b.sh > experiments/062-h100-protocol-v1/queue_2026-10-09b.log 2>&1 &'
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src
PY=.venv/bin/python

while pgrep -f "queue_2026-10-09.sh|062-h100-protocol-v1/(driver|random_null|stable_core|rescore_mid).py" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] queue b start"

echo "[$(date +%H:%M)] job 1: src scorer vs experiment scorer on the 15k circuits (DAN-73, 067)"
$PY -X utf8 -u experiments/067-src-scorer-equivalence/check_equivalence.py > experiments/067-src-scorer-equivalence/run.log 2>&1
echo "[$(date +%H:%M)] job 1 exit $?"

# As run on 2026-10-09. The sweep has since moved to experiments/071-contrast-sweep and changed design (2026-10-10:
# close / random / distant disjoint + fused); this line is kept as the record of what ran, not as a launcher.
echo "[$(date +%H:%M)] job 2: contrast-sweep smoke, 8 targets x {store, close_silent} (DAN-64, 065)"
bash experiments/065-contrast-sweep/smoke.sh > experiments/065-contrast-sweep/smoke.log 2>&1
echo "[$(date +%H:%M)] job 2 exit $?"

echo "[$(date +%H:%M)] queue b DONE"
