#!/bin/bash
# Stage 2b driver (NOT itself a GPU process): runs sweep.py in <= BUDGET_MIN-minute GPU jobs, each under gpujob.sh
# (lock taken and released per job, so exp 133 can interleave), until sweep.py logs DONE or gpujob refuses (07:30).
#   bash experiments/132-ioi-ceiling-baselines/stage2b.sh        (repo root; env passes through: METHODS, HELDOUT_N)
set -uo pipefail
E=experiments/132-ioi-ceiling-baselines
HN=${HELDOUT_N:-500}
TAG=${TAG:-h$HN}
ENV="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HELDOUT_N=$HN CURVES=curves_$TAG.jsonl BUDGET_MIN=${BUDGET_MIN:-38}${METHODS:+ METHODS=$METHODS}${ONLY_KNOBS:+ ONLY_KNOBS=$ONLY_KNOBS}"
[ "$HN" = all ] && { ENV="${ENV/HELDOUT_N=all /}"; unset HELDOUT_N; }
for i in $(seq ${FIRST:-1} ${LAST:-12}); do     # FIRST=LAST=n: one job only (alternating with exp 133 by hand)
  name=sweep_${TAG}_$i
  bash $E/gpujob.sh "$name" env $ENV ./.venv/bin/python -X utf8 -u $E/sweep.py; rc=$?
  [ "$rc" = 8 ] && { echo "refused (late): stop"; exit 8; }
  [ "$rc" = 0 ] || { echo "job $name rc $rc"; [ "$rc" = 9 ] && exit 9; }
  grep -q "\] DONE" "$E/out/logs/$name.log" && { echo "sweep DONE"; exit 0; }
  sleep 30   # leave the GPU free briefly for the other agent
done
