#!/usr/bin/env bash
# 059 full run: build the context pools for the 16 pilot targets, fit arms D/A/B/C, score every arm on both held-out
# sets, summarise. Resumable (fit skips fitted targets, eval skips scored rows).
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/059-context-pool
FILTER='warn|extension|n_gpus|artifact stores|bogus|epilogue compiled|CF-Capture|CFaithfulness'
mkdir -p $E/results
STAGE=build PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/build.log
echo "BUILD DONE"
for arm in D A B C; do
  sleep 10
  STAGE=fit ARM=$arm PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/fit_$arm.log
  echo "FIT $arm DONE"
done
for arm in D A B C; do
  sleep 10
  STAGE=eval ARM=$arm PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/eval_$arm.log
  echo "EVAL $arm DONE"
done
STAGE=summary PYTHONPATH=src $PY -X utf8 $E/pool_test.py > $E/results/summary.log 2>&1
echo "ALL DONE"
