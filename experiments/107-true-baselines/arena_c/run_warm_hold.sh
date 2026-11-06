#!/bin/bash
# Arena C: the λ path with held temperature (warm_path.py --hold): the first λ a cold cosine fit, every later λ starts
# from the previous λ's gates and amplitudes and trains AT T = 0.05 with a constant lr of 0.05 x the run's lr. 400 steps
# per λ, all tasks, per-position then pooled, into out/warm_path_hold. Replaces run_cosine.sh KIND=warm, whose warm fits
# restarted the anneal at T = 1 and re-softened the inherited gates (simple λ 1e-3 / 3e-3 collapsed). One deadline
# fixed at launch, the next STOP_AT (default 13:45). WSL:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_warm_hold.sh > experiments/107-true-baselines/arena_c/out/warm_hold.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-13:45} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
A=experiments/107-true-baselines/arena_c
for M in per_position pooled; do
  [ "$(date +%s)" -ge "$STOP_EPOCH" ] && break
  ./.venv/bin/python $A/warm_path.py --tasks simple,within_rc,rc,nounpp --means "$M" --steps 400 \
    --lams 3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2 --lr_schedule cosine --hold --out $A/out/warm_path_hold \
    || echo "FAILED warm_hold $M"
done
echo "[$(date +%T)] DONE"
