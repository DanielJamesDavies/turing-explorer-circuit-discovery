#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
env COMP=29 LAT=3736 ARM=know400 TOPM=25 MEMFILE=know_members.jsonl \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/analyse_members.py > $R/an_c29_know.log 2>&1
env COMP=35 LAT=13633 ARM=know400 TOPM=25 MEMFILE=know_members.jsonl \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/analyse_members.py > $R/an_c35_know.log 2>&1
env COMP=29 LAT=3736 ARM=know400 MEMFILE=know_members.jsonl STEM="relativ,Einstein" \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/member_token_id.py > $R/tid_c29_know.log 2>&1
env COMP=29 LAT=3736 ARM=triamp400 MEMFILE=members.jsonl STEM="relativ,Einstein" \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/member_token_id.py > $R/tid_c29_prod.log 2>&1
echo DONE > $R/analysis.flag
