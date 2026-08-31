#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
B=experiments/044-behaviours
while [ ! -f $B/behav3.flag ]; do sleep 60; done
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python -X utf8 $B/make_knowledge.py > $B/make_kb.log 2>&1 \
  || { echo VERIFY_FAIL > $B/kb.flag; exit 1; }
DATA=knowledge_clusters.pt OUT=knowledge CLUSTERS=0,1,2 LAM=3e-3 \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python -X utf8 $B/behaviour_runner.py > $B/kb.log 2>&1
echo DONE > $B/kb.flag
