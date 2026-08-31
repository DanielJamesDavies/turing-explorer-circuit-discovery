#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
env COMP=29 LAT=3736 ARM=negamp400 MEMFILE=know_members.jsonl STEM="relativ,Einstein" \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/member_token_id.py > $R/tid_c29_negamp.log 2>&1
env COMP=29 LAT=3736 ARM=negamp400 MEMFILE=know_members.jsonl TOPM=25 \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/analyse_members.py > $R/an_c29_negamp.log 2>&1
env COMP=29 LAT=3736 ARM=triamp400 MEMFILE=members.jsonl TOPM=25 \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/analyse_members.py > $R/an_c29_triamp.log 2>&1
echo DONE > $R/negcmp.flag
