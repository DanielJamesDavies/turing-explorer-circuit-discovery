#!/usr/bin/env bash
# Experiment 080 stage 2, local GPU: the full run (150 targets: LAT re-scored, OUT_W and OUT_U fitted over the adopted
# lambda grid, every circuit scored on both endpoints), then the report and the figure. Needs config_main.json (written
# by calibrate.py ADOPT=1 from calib/recommendation.json): run.py refuses to start without it. Resumable: re-run after
# a restart. Waits for any other 080 job on the card first.
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid bash experiments/080-latent-vs-output/queue_local.sh > experiments/080-latent-vs-output/queue_local.log 2>&1 < /dev/null & disown'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/080-latent-vs-output
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
if [ ! -f $E/config_main.json ]; then
  echo "no $E/config_main.json: run the calibration (queue_calib.sh) and adopt its recommendation first"; exit 1
fi
while pgrep -f "080-latent-vs-output/(run|calibrate).py" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] main run start"
mkdir -p $E/out
OUT=$E/out $PY -X utf8 -u $E/run.py >> $E/out/run.log 2>&1
echo "[$(date +%H:%M)] main run exit $?"
OUT=$E/out $PY -X utf8 $E/report.py > $E/out/report.txt 2>&1
echo "[$(date +%H:%M)] report exit $? -> $E/out/report.txt"
OUT=$E/out $PY -X utf8 $E/figure.py > $E/out/figure.log 2>&1
echo "[$(date +%H:%M)] figure exit $?"
echo "[$(date +%H:%M)] queue DONE"
