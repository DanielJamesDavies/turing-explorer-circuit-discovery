#!/bin/bash
# Arena C: does the cosine lr remove the fit noise? seed_variance.py --lr_schedule cosine, donor seeds 1-3, simple and
# rc, per-position, λ 1e-5..1e-2 (where the constant-lr refits flipped pass / fail and size 2-3x), into
# out/seed_variance_cosine/. One deadline fixed at launch, the next STOP_AT (default 13:45). WSL:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_seed_cos.sh > experiments/107-true-baselines/arena_c/out/seed_cos.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-13:45} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
A=experiments/107-true-baselines/arena_c
./.venv/bin/python $A/seed_variance.py --seeds 1,2,3 --tasks simple,rc --lams 1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2 \
  --means per_position --lr_schedule cosine || echo "FAILED seed_cos"
echo "[$(date +%T)] DONE"
