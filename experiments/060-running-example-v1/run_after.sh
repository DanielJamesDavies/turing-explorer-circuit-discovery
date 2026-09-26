#!/usr/bin/env bash
# Follow-ups for the running example, queued after run_overnight.sh (same GPU):
#   - the string-detector control (is 3.resid.35381 a "temperature" token detector?)
#   - re-run case.py so arm Bw is also scored (the first run skipped it: the resume key ignored the arm; fixed)
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
FILTER='warn|extension|n_gpus|artifact stores|bogus|epilogue compiled|CF-Capture|CFaithfulness'
while pgrep -f run_overnight.sh > /dev/null; do sleep 30; done
sleep 10
PYTHONPATH=src $PY -X utf8 experiments/060-running-example-v1/string_control.py 2>&1 | grep -v -i -E "$FILTER" \
  > experiments/060-running-example-v1/results/string_control.log
echo "STRING CONTROL DONE $(date)"
sleep 10
PYTHONPATH=src $PY -X utf8 experiments/060-running-example-v1/case.py 2>&1 | grep -v -i -E "$FILTER" \
  > experiments/060-running-example-v1/results/run2.log
echo "CASE RERUN DONE $(date)"
