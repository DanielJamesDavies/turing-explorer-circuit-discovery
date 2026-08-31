#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
B=experiments/044-behaviours
NWIN=32768 K=100 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python -X utf8 $B/cluster_behaviours.py > $B/cluster2.log 2>&1
CLUSTERS=auto3 LAM=3e-3 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python -X utf8 $B/behaviour_runner.py > $B/behav2.log 2>&1
echo DONE > $B/behav2.flag
