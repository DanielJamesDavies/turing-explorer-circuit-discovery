#!/usr/bin/env bash
# 059 matched-contrast arms (Dm / Em / Cm): the C ablation value averages over as many contrast training contexts as
# the arm has activating training contexts. Waits for run_followup.sh (same GPU, and build_neg128 rewrites the cached
# contexts file). Resumable.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/059-context-pool
FILTER='warn|extension|n_gpus|artifact stores|bogus|epilogue compiled|CF-Capture|CFaithfulness'
while pgrep -f run_followup.sh > /dev/null; do sleep 30; done
sleep 10
STAGE=build_neg128 PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/build_neg128.log
echo "NEG128 DONE"
for arm in Dm Em Cm; do
  sleep 10
  STAGE=fit ARM=$arm PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/fit_$arm.log
  echo "FIT $arm DONE"
done
for arm in Dm Em Cm; do
  sleep 10
  STAGE=eval ARM=$arm PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/eval_$arm.log
  echo "EVAL $arm DONE"
done
STAGE=summary PYTHONPATH=src $PY -X utf8 $E/pool_test.py > $E/results/summary.log 2>&1
echo "ALL DONE"
