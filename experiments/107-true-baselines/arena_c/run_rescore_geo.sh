#!/bin/bash
# Arena C: re-scores the 2026-11-02 daytime runs once the geometric fits stop (geometric: wcm_tasks_geo; cosine:
# wcm_tasks_cos [per-position complete, pooled partial], warm_path_hold, warm_path_cos [both partial]; constant lr:
# wcm_tasks_s1200, warm_path [partial]): the donor test, WCM under SFC's evaluator, the curves (figs / summary suffixed
# with the run's name; tables to out/curves_<run>.md). Replaces run_rescore_final.sh. Deadline STOP_AT (default 14:30).
# WSL:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_rescore_geo.sh > experiments/107-true-baselines/arena_c/out/rescore_geo.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-14:30} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
A=experiments/107-true-baselines/arena_c
RUNS="wcm_tasks_geo wcm_tasks_cos warm_path_hold warm_path_cos wcm_tasks_s1200 warm_path"
while pgrep -f 'run_geometric.sh' > /dev/null; do sleep 30; done
echo "[$(date +%T)] fits finished; re-scoring"
OUTS=""; for O in $RUNS; do [ -f $A/out/$O/eval.jsonl ] && OUTS="$OUTS $A/out/$O"; done
./.venv/bin/python $A/donor_tasks.py --out $OUTS || echo "FAILED donor_tasks"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ~/venvs/sfc107/bin/python $A/score_wcm_sfc.py --wcm_out $OUTS || echo "FAILED score_wcm_sfc"
for O in $OUTS; do
  n=$(basename $O)
  ./.venv/bin/python $A/curves.py --wcm_out $O > $A/out/curves_$n.md || echo "FAILED curves $n"
done
echo "[$(date +%T)] DONE"
