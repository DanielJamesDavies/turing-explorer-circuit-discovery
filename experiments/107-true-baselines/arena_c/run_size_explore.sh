#!/bin/bash
# Arena C: the size diagnostics (size_explore.py), runs alongside the other jobs; one deadline fixed at launch, the
# next STOP_AT (default 08:30, Daniel 2026-11-02). Run in WSL:
#   nohup setsid bash experiments/107-true-baselines/arena_c/run_size_explore.sh > experiments/107-true-baselines/arena_c/out/size_explore.log 2>&1 < /dev/null &
set -uo pipefail
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" || exit 1
export HF_HUB_OFFLINE=1 STOP_AT=${STOP_AT:-08:30} PYTHONUNBUFFERED=1
now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s); [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
export STOP_EPOCH=$stop
./.venv/bin/python experiments/107-true-baselines/arena_c/size_explore.py "$@" || echo FAILED
echo DONE
