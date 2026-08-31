#!/bin/bash
# Regenerate the clobbered pos census jsonl, then cumulative raise
# curves in both frames for the relativity seed.
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
COMP=29 LAT=3736 TOPK_ACT=200 PYTHONPATH=src \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/sign_census.py > $R/census_pos_redo.log 2>&1
COMP=29 LAT=3736 CENSUS=sign_census_c29_3736.jsonl MAXK=48 PYTHONPATH=src \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/cumulative_raise.py > $R/cum_pos.log 2>&1
CTX=neg COMP=29 LAT=3736 CENSUS=sign_census_neg_c29_3736.jsonl MAXK=48 \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/cumulative_raise.py > $R/cum_neg.log 2>&1
echo DONE > $R/cumraise.flag
