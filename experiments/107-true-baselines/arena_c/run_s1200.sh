#!/bin/bash
# Arena C: the main λ grid refitted at 1,200 steps (3x the run's 400), all four tasks and both means modes, into
# out/wcm_tasks_s1200. Why: the seed refits flip pass / fail and size up to 2-3x at λ >= 3e-5, and size_explore's
# steps1200 variant shrank the smallest passing circuit (simple 895 -> 614, rc 12,441 -> 2,474): the 400-step fits have
# not finished. One deadline fixed at launch, the next STOP_AT (default 14:30: Daniel, 2026-11-02, meeting at 15:00).
# WSL:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_s1200.sh > experiments/107-true-baselines/arena_c/out/s1200.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-14:30} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
OUT=experiments/107-true-baselines/arena_c/out/wcm_tasks_s1200
A=experiments/107-true-baselines/arena_c
for M in per_position pooled; do
  for T in simple within_rc rc nounpp; do
    [ "$(date +%s)" -ge "$STOP_EPOCH" ] && break 2
    ./.venv/bin/python $A/wcm_marks.py run --tasks "$T" --lams 3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2 \
      --steps 1200 --mode sfc --sites noembed --means "$M" --out "$OUT" || echo "FAILED $T $M"
  done
done
echo "[$(date +%T)] DONE"
