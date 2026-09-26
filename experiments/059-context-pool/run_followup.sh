#!/usr/bin/env bash
# 059 follow-ups: E (48 strongest + 16 mid), B2 (B resampled: the noise floor), C800 (C at 800 steps: the budget
# control). Uses the cached contexts from run_all.sh. Resumable.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/059-context-pool
FILTER='warn|extension|n_gpus|artifact stores|bogus|epilogue compiled|CF-Capture|CFaithfulness'
for arm in E B2 C800; do
  sleep 10
  STAGE=fit ARM=$arm PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/fit_$arm.log
  echo "FIT $arm DONE"
done
for arm in E B2 C800; do
  sleep 10
  STAGE=eval ARM=$arm PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/eval_$arm.log
  echo "EVAL $arm DONE"
done
STAGE=summary PYTHONPATH=src $PY -X utf8 $E/pool_test.py > $E/results/summary.log 2>&1
echo "ALL DONE"
