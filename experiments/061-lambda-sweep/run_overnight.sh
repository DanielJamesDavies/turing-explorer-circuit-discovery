#!/usr/bin/env bash
# Overnight local queue (2026-09-25, Daniel away ~12 h). Runs after 059's run_weights.sh finishes (same GPU).
#   2. running example under protocol v1   experiments/060-running-example-v1/case.py
#   3. lambda sweep, WCM vs unweighted     experiments/061-lambda-sweep/sweep.py
#   4. specificity on B/B2/Bw/B2w          experiments/056-specificity/spec059.py
#   5. SAE quality figure                  eval.sae.reconstruction_eval (+CE) then eval.sae.report
# Every stage is resumable and logs to its own folder; a failing stage does not stop the queue.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
FILTER='warn|extension|n_gpus|artifact stores|bogus|epilogue compiled|CF-Capture|CFaithfulness'
while pgrep -f run_weights.sh > /dev/null; do sleep 30; done
sleep 10
mkdir -p experiments/060-running-example-v1/results experiments/061-lambda-sweep/results experiments/056-specificity/results/v059

PYTHONPATH=src $PY -X utf8 experiments/060-running-example-v1/case.py 2>&1 | grep -v -i -E "$FILTER" \
  > experiments/060-running-example-v1/results/run.log
echo "CASE DONE $(date)"
sleep 10
PYTHONPATH=src $PY -X utf8 experiments/061-lambda-sweep/sweep.py 2>&1 | grep -v -i -E "$FILTER" \
  > experiments/061-lambda-sweep/results/run.log
echo "SWEEP DONE $(date)"
sleep 10
PYTHONPATH=src $PY -X utf8 experiments/056-specificity/spec059.py 2>&1 | grep -v -i -E "$FILTER" \
  > experiments/056-specificity/results/v059/run.log
echo "SPEC059 DONE $(date)"
sleep 10
PYTHONPATH=src $PY -X utf8 -m eval.sae.reconstruction_eval --batches 3 --seqs 64 --ce --out analysis-restyled/sae-eval \
  > analysis-restyled/sae-eval-run.log 2>&1
PYTHONPATH=src $PY -X utf8 -m eval.sae.report --run-root outputs --log-root /mnt/x/Projects/AIs/Turing/sae-system \
  --out analysis-restyled/sae-eval >> analysis-restyled/sae-eval-run.log 2>&1
echo "SAE DONE $(date)"
echo "ALL DONE $(date)"
