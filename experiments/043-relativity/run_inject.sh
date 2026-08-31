#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
for SPEC in "29 3736" "35 13633" "11 18699"; do
  set -- $SPEC
  COMP=$1 LAT=$2 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python $R/inject_profile.py > "$R/inject_c$1_$2.log" 2>&1
done
echo DONE > $R/inject.flag
