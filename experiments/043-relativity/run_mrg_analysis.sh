#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
for SPEC in "29 3736" "35 13633" "29 23920"; do
  set -- $SPEC
  env COMP=$1 LAT=$2 ARM=mrgamp400 MEMFILE=know_members.jsonl TOPM=15 TOPK=2 \
    PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python $R/analyse_members.py > "$R/an_mrg_$1_$2.log" 2>&1
done
echo DONE > $R/mrgan.flag
