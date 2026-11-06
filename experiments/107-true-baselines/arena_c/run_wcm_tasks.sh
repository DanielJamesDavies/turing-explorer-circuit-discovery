#!/bin/bash
# Arena C (DAN-194): WCM v3 on SFC's four SVA tasks, both A / C means modes, the 6-point lambda grid, scope noembed
# (18 sites), SFC's 100 / 40 data. Resumable per circuit (a finished circuit is reloaded, a scored row skipped); no new
# fit starts after STOP_AT (default 07:30). Run in WSL from anywhere:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_wcm_tasks.sh > experiments/107-true-baselines/arena_c/out/wcm_tasks.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-08:30} PYTHONUNBUFFERED=1   # Daniel 2026-11-02: cap 08:30
# one deadline for every process: the next STOP_AT after launch
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
# wcm_tasks_zacr: behavioural WCM = Z + A + C + R (2026-11-02). out/wcm_tasks (v3 with the removed behavioural
# rank-keep) is superseded, so this run writes to its own folder.
OUT=${OUT:-experiments/107-true-baselines/arena_c/out/wcm_tasks_zacr}
LAMS=${LAMS:-3e-6,1e-5,3e-5,1e-4,3e-4,1e-3}
for M in pooled per_position; do
  for T in simple nounpp within_rc rc; do
    ./.venv/bin/python experiments/107-true-baselines/arena_c/wcm_marks.py run --tasks "$T" --lams "$LAMS" \
      --steps 400 --mode sfc --sites noembed --means "$M" --out "$OUT" || echo "FAILED $T $M"
  done
done
echo DONE
