#!/usr/bin/env bash
# Experiment 081 stage 4, local GPU: the main run (every competent task: W and U over the adopted lambda grid, the
# random null at every W lambda, the W_ZA sensitivity arm for generated-contrast tasks, profiles), then the report and
# the figure. Needs probe/<MODEL>/competence.json and the contrast pools (queue_calib.sh) and config_main_<MODEL>.json
# (calibrate.py ADOPT=1): run.py refuses to start without them. Resumable. Waits for any other 081 job first.
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid bash experiments/081-task-circuits/queue_local.sh > experiments/081-task-circuits/queue_local.log 2>&1 < /dev/null & disown'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/081-task-circuits
export MODEL=${MODEL:-gpt2-small}
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
if [ ! -f $E/probe/$MODEL/competence.json ]; then
  echo "no $E/probe/$MODEL/competence.json: run queue_calib.sh (prepare + probe + calibration) first"; exit 1
fi
if [ ! -f $E/config_main_$MODEL.json ]; then
  echo "no $E/config_main_$MODEL.json: run the calibration (queue_calib.sh) and adopt its recommendation first"; exit 1
fi
while pgrep -f "081-task-circuits/(prepare|probe|run|calibrate).py" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M)] main run start (MODEL $MODEL)"
mkdir -p $E/out/$MODEL
$PY -X utf8 -u $E/run.py >> $E/out/$MODEL/run.log 2>&1
echo "[$(date +%H:%M)] main run exit $?"
$PY -X utf8 $E/report.py > $E/out/$MODEL/report.txt 2>&1
echo "[$(date +%H:%M)] report exit $? -> $E/out/$MODEL/report.txt"
$PY -X utf8 $E/figure.py > $E/out/$MODEL/figure.log 2>&1
echo "[$(date +%H:%M)] figure exit $? (paper version: PAPER=1 figure.py, by hand)"
echo "[$(date +%H:%M)] queue DONE"
