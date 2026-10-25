#!/usr/bin/env bash
# Experiment 081 stages 1-3, local GPU, in order: the generated contrast sets (prepare.py: tasks without patch prompts),
# the competence probe over every task file, then the lambda calibration on the competent calibration tasks (their
# training rows only, re-split), then its report and recommendation (calib/<MODEL>/recommendation.json, NOT adopted).
# Resumable: re-run after a restart. Waits for any other 081 job first. Afterwards read probe/<MODEL>/competence.json
# and calib/<MODEL>/report.txt, then adopt explicitly:
#   MODEL=gpt2-small ADOPT=1 PYTHONPATH=src .venv/bin/python experiments/081-task-circuits/calibrate.py
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid bash experiments/081-task-circuits/queue_calib.sh > experiments/081-task-circuits/queue_calib.log 2>&1 < /dev/null & disown'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/081-task-circuits
export MODEL=${MODEL:-gpt2-small}
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "081-task-circuits/(prepare|probe|run|calibrate).py" > /dev/null; do sleep 30; done
mkdir -p $E/probe/$MODEL $E/calib/$MODEL
echo "[$(date +%H:%M)] prepare start (MODEL $MODEL)"
$PY -X utf8 -u $E/prepare.py > $E/probe/$MODEL/prepare.log 2>&1
echo "[$(date +%H:%M)] prepare exit $? -> $E/tasks/$MODEL/*.contrast.json"
tail -12 $E/probe/$MODEL/prepare.log
echo "[$(date +%H:%M)] probe start"
$PY -X utf8 -u $E/probe.py > $E/probe/$MODEL/probe.log 2>&1
echo "[$(date +%H:%M)] probe exit $? -> $E/probe/$MODEL/competence.json"
tail -24 $E/probe/$MODEL/probe.log
echo "[$(date +%H:%M)] calibration start"
$PY -X utf8 -u $E/calibrate.py >> $E/calib/$MODEL/run.log 2>&1
echo "[$(date +%H:%M)] calibration exit $?"
REPORT=1 $PY -X utf8 $E/calibrate.py > $E/calib/$MODEL/report.txt 2>&1
echo "[$(date +%H:%M)] report -> $E/calib/$MODEL/report.txt; recommendation -> $E/calib/$MODEL/recommendation.json (not adopted)"
echo "[$(date +%H:%M)] queue DONE"
