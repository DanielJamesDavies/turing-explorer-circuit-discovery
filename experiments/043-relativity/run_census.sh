#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
for SPEC in "29 3736" "35 13633" "29 23920"; do
  set -- $SPEC
  COMP=$1 LAT=$2 TOPK_ACT=200 PYTHONPATH=src \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python $R/sign_census.py > "$R/census_c$1_$2.log" 2>&1
done
echo DONE > $R/census.flag
