#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
for SPEC in "29 3736" "35 13633" "29 23920"; do
  set -- $SPEC
  COMP=$1 LAT=$2 ARMS=triamp400,mrgamp400 MEMFILE=know_members.jsonl \
    PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python $R/firing_fidelity.py > "$R/firing_c$1_$2.log" 2>&1
done
echo DONE > $R/firing.flag
