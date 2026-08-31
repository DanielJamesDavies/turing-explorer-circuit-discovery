#!/bin/bash
# Differential-seed circuits, queued behind the injection profiles.
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
while [ ! -f $R/inject.flag ]; do sleep 60; done
DIFF="29:3736-4523" STEPS=400 PYTHONPATH=src \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/diff_runner.py > $R/diff_29.log 2>&1
DIFF="29:4523-3736" STEPS=400 PYTHONPATH=src \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/diff_runner.py > $R/diff_29_rev.log 2>&1
echo DONE > $R/diff.flag
