#!/bin/bash
# Circuits for the token-selected relativity seeds. Seeds chosen by
# activation on the 'relativ' stem in physics contexts, with a
# linguistic-relativity distractor control (see find_seeds.py).
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
SEEDS_JSON='{"29":[3736,4523],"35":[13633],"26":[455],"23":[12639],"32":[8627]}' \
N_NULL=4 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python experiments/043-relativity/runner.py \
  > experiments/043-relativity/circuits.log 2>&1
echo DONE > experiments/043-relativity/circuits.flag
