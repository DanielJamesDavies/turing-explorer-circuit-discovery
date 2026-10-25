#!/usr/bin/env bash
# Experiment 080 stage 1, local GPU: the calibration sweep (targets_calib.txt, 18 under causal x {OUT_W, OUT_U} x
# 3 lr x 5 lambda, 400 steps),
# then its report and recommendation (calib/recommendation.json, NOT adopted). Resumable: re-run after a restart.
# Waits for any other 080 job on the card first. After it finishes, read the report, then adopt explicitly:
#   OUT_CAL=experiments/080-latent-vs-output/calib ADOPT=1 PYTHONPATH=src .venv/bin/python experiments/080-latent-vs-output/calibrate.py
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid bash experiments/080-latent-vs-output/queue_calib.sh > experiments/080-latent-vs-output/queue_calib.log 2>&1 < /dev/null & disown'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/080-latent-vs-output
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "080-latent-vs-output/(run|calibrate).py" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] calibration start"
mkdir -p $E/calib
OUT_CAL=$E/calib $PY -X utf8 -u $E/calibrate.py >> $E/calib/run.log 2>&1
echo "[$(date +%H:%M)] calibration exit $?"
OUT_CAL=$E/calib REPORT=1 $PY -X utf8 $E/calibrate.py > $E/calib/report.txt 2>&1
echo "[$(date +%H:%M)] report -> $E/calib/report.txt; recommendation -> $E/calib/recommendation.json (not adopted)"
echo "[$(date +%H:%M)] queue DONE"
