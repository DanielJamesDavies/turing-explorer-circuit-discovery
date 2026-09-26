#!/usr/bin/env bash
# 059 equal-weight comparison (Daniel, 2026-09-25): the protocol's 32 + 16 contexts (B and its resample B2) refit with
# gamma_C = gamma_A = 1, lambda = 2e-3, compared against B / B2 (0.25 / 1e-3) on the same held-out sets. Uses the
# cached contexts. Resumable.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/059-context-pool
FILTER='warn|extension|n_gpus|artifact stores|bogus|epilogue compiled|CF-Capture|CFaithfulness'
for arm in Bw B2w; do
  sleep 10
  STAGE=fit ARM=$arm PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/fit_$arm.log
  echo "FIT $arm DONE"
done
for arm in Bw B2w; do
  sleep 10
  STAGE=eval ARM=$arm PYTHONPATH=src $PY -X utf8 $E/pool_test.py 2>&1 | grep -v -i -E "$FILTER" > $E/results/eval_$arm.log
  echo "EVAL $arm DONE"
done
STAGE=summary PYTHONPATH=src $PY -X utf8 $E/pool_test.py > $E/results/summary.log 2>&1
echo "ALL DONE"
