#!/bin/bash
# Arena C: the donor-seed variance refits (seed_variance.py), queued behind the main WCM run: waits until
# run_wcm_tasks.sh has exited, then fits seeds 1-3 (pooled means; per-position if time is left). One deadline for
# every process: the next STOP_AT (default 08:30, Daniel 2026-11-02) after launch. Run in WSL:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_seed_variance.sh > experiments/107-true-baselines/arena_c/out/seed_variance.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-08:30} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
while pgrep -f run_wcm_tasks.sh > /dev/null; do sleep 30; done
echo "[$(date +%T)] main run finished; seed refits start"
for M in pooled per_position; do
  [ "$(date +%s)" -ge "$STOP_EPOCH" ] && break
  ./.venv/bin/python experiments/107-true-baselines/arena_c/seed_variance.py --seeds 1,2,3 --means "$M" || echo "FAILED $M"
done
echo DONE
