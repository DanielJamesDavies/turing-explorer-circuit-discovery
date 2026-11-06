#!/bin/bash
# Arena C: extend WCM's curve to small circuits (the first comparison's area counted 0 below WCM's smallest fit, 319-1,551
# latents, while SFC's curves reach 10^2). Waits for size_explore.py to exit, then fits λ 3e-3, 1e-2, 3e-2 for the four
# tasks and both means modes into out/wcm_tasks_zacr (the same run: resumable keys include λ), then re-scores: donor test,
# WCM under SFC's evaluator, the curves. One deadline fixed at launch, the next STOP_AT (default 08:30). WSL:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_high_lambda.sh > experiments/107-true-baselines/arena_c/out/high_lambda.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-08:30} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
OUT=experiments/107-true-baselines/arena_c/out/wcm_tasks_zacr
A=experiments/107-true-baselines/arena_c
while pgrep -f size_explore.py > /dev/null; do sleep 30; done
echo "[$(date +%T)] size_explore finished; high-λ fits start"
for M in pooled per_position; do
  for T in simple nounpp within_rc rc; do
    [ "$(date +%s)" -ge "$STOP_EPOCH" ] && break 2
    ./.venv/bin/python $A/wcm_marks.py run --tasks "$T" --lams 3e-3,1e-2,3e-2 --steps 400 --mode sfc --sites noembed \
      --means "$M" --out "$OUT" || echo "FAILED $T $M"
  done
done
echo "[$(date +%T)] re-scoring"
./.venv/bin/python $A/donor_tasks.py --out "$OUT" || echo "FAILED donor_tasks"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ~/venvs/sfc107/bin/python $A/score_wcm_sfc.py --wcm_out "$OUT" || echo "FAILED score_wcm_sfc"
./.venv/bin/python $A/curves.py --wcm_out "$OUT" || echo "FAILED curves"
echo "[$(date +%T)] DONE"
