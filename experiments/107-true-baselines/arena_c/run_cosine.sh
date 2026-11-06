#!/bin/bash
# Arena C: behavioural WCM with a cosine lr schedule [ours, a fitting-procedure variant; diag_anneal.py]. The run's
# constant lr under the gate anneal (m = sigmoid(theta / T), T 1 -> 0.05) moves the gates up to 20x further per step
# late in training: fits blow up late (simple λ 1e-5, 1,200 steps: zero term 0.0015 -> 12.2 at step ~900, Z 0.008);
# cosine (lr 1 -> 0.05, tracking T) trains smoothly (Z / A / C 1.01 / 1.01 / 1.02). Usage:
#   KIND=cold: the main grid, independent fits, 400 steps, all tasks, both means -> out/wcm_tasks_cos; then 1,200 steps
#              on simple + rc, per-position -> out/wcm_tasks_cos_s1200
#   KIND=warm: the λ path (warm_path.py), 400 steps per λ, all tasks, both means -> out/warm_path_cos
# One deadline fixed at launch, the next STOP_AT (default 13:45: re-scoring then ends by 14:30). WSL:
#   KIND=cold nohup setsid bash experiments/107-true-baselines/arena_c/run_cosine.sh > experiments/107-true-baselines/arena_c/out/cos_cold.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-13:45} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
A=experiments/107-true-baselines/arena_c
LAMS=3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2
if [ "${KIND:-cold}" = warm ]; then
  for M in per_position pooled; do
    [ "$(date +%s)" -ge "$STOP_EPOCH" ] && break
    ./.venv/bin/python $A/warm_path.py --tasks simple,within_rc,rc,nounpp --means "$M" --steps 400 --lams $LAMS \
      --lr_schedule cosine --out $A/out/warm_path_cos || echo "FAILED warm $M"
  done
else
  for M in per_position pooled; do
    for T in simple within_rc rc nounpp; do
      [ "$(date +%s)" -ge "$STOP_EPOCH" ] && break 2
      ./.venv/bin/python $A/wcm_marks.py run --tasks "$T" --lams $LAMS --steps 400 --mode sfc --sites noembed \
        --means "$M" --lr_schedule cosine --out $A/out/wcm_tasks_cos || echo "FAILED cold $T $M"
    done
  done
  for T in simple rc; do
    [ "$(date +%s)" -ge "$STOP_EPOCH" ] && break
    ./.venv/bin/python $A/wcm_marks.py run --tasks "$T" --lams $LAMS --steps 1200 --mode sfc --sites noembed \
      --means per_position --lr_schedule cosine --out $A/out/wcm_tasks_cos_s1200 || echo "FAILED cold1200 $T"
  done
fi
echo "[$(date +%T)] DONE"
