#!/bin/bash
# Arena C: re-scores the 2026-11-02 daytime runs (wcm_tasks_s1200, warm_path) once their fits stop: the donor test, WCM
# under SFC's evaluator, the curves (figs / summary suffixed with the run's name). Deadline STOP_AT (default 14:30).
# WSL:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_rescore_today.sh > experiments/107-true-baselines/arena_c/out/rescore_today.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-14:30} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
A=experiments/107-true-baselines/arena_c
while pgrep -f 'run_s1200.sh|run_warm_path.sh' > /dev/null; do sleep 30; done
echo "[$(date +%T)] fits finished; re-scoring"
./.venv/bin/python $A/donor_tasks.py --out $A/out/wcm_tasks_s1200 $A/out/warm_path || echo "FAILED donor_tasks"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ~/venvs/sfc107/bin/python $A/score_wcm_sfc.py \
  --wcm_out $A/out/wcm_tasks_s1200 $A/out/warm_path || echo "FAILED score_wcm_sfc"
for O in wcm_tasks_s1200 warm_path; do
  ./.venv/bin/python $A/curves.py --wcm_out $A/out/$O > $A/out/curves_$O.md || echo "FAILED curves $O"
done
echo "[$(date +%T)] DONE"
