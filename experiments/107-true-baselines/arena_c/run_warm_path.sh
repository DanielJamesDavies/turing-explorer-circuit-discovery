#!/bin/bash
# Arena C: behavioural WCM along a λ path (warm_path.py: each λ starts from the previous λ's final gates and
# amplitudes), 400 steps per λ, all four tasks, per-position means first, then pooled, into out/warm_path. One deadline
# fixed at launch, the next STOP_AT (default 13:45: fits stop there so the re-scoring ends by 14:30; Daniel 2026-11-02,
# meeting at 15:00). WSL:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_warm_path.sh > experiments/107-true-baselines/arena_c/out/warm_path.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-13:45} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
A=experiments/107-true-baselines/arena_c
for M in per_position pooled; do
  [ "$(date +%s)" -ge "$STOP_EPOCH" ] && break
  ./.venv/bin/python $A/warm_path.py --tasks simple,within_rc,rc,nounpp --means "$M" --steps 400 \
    --out $A/out/warm_path || echo "FAILED warm_path $M"
done
echo "[$(date +%T)] DONE"
