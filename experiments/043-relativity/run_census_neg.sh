#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
for SPEC in "29 3736" "35 13633"; do
  set -- $SPEC
  CTX=neg COMP=$1 LAT=$2 TOPK_ACT=200 PYTHONPATH=src \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python $R/sign_census.py > "$R/census_neg_c$1_$2.log" 2>&1
  mv "$R/sign_census_c$1_$2.jsonl" "$R/sign_census_neg_c$1_$2.jsonl" 2>/dev/null
done
echo DONE > $R/census_neg.flag
