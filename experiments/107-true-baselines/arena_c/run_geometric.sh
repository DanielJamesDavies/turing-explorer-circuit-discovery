#!/bin/bash
# Arena C: behavioural WCM with the GEOMETRIC lr schedule [ours, a fitting-procedure variant] = lr * 0.05**progress,
# the anneal's own shape, so the gate step lr / T stays constant (diag_anneal.py, WCM_ADAPTER.md section 8: constant lr
# blows up late, cosine still blows up early on rc; geometric trains smoothly on both). 400 steps, the main λ grid, all
# four tasks, into out/wcm_tasks_geo. MEANS=per_position|pooled (one process per means mode); SEEDS=1 adds the donor-
# seed refits (seeds 1-3, simple + rc, per-position) into out/seed_variance_geometric after the grid. One deadline fixed
# at launch, the next STOP_AT (default 13:45). WSL:
#   MEANS=per_position SEEDS=1 nohup setsid bash experiments/107-true-baselines/arena_c/run_geometric.sh > experiments/107-true-baselines/arena_c/out/geo_pp.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-13:45} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
A=experiments/107-true-baselines/arena_c
LAMS=3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2
M=${MEANS:-per_position}
for T in simple rc within_rc nounpp; do
  [ "$(date +%s)" -ge "$STOP_EPOCH" ] && break
  ./.venv/bin/python $A/wcm_marks.py run --tasks "$T" --lams $LAMS --steps 400 --mode sfc --sites noembed \
    --means "$M" --lr_schedule geometric --out $A/out/wcm_tasks_geo || echo "FAILED geo $T $M"
done
if [ "${SEEDS:-0}" = 1 ] && [ "$(date +%s)" -lt "$STOP_EPOCH" ]; then
  ./.venv/bin/python $A/seed_variance.py --seeds 1,2,3 --tasks simple,rc --lams 1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2 \
    --means per_position --lr_schedule geometric || echo "FAILED seed_geo"
fi
echo "[$(date +%T)] DONE"
