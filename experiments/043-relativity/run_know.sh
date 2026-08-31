#!/bin/bash
# Full knowledge-circuit runs: relativity (c29/3736, the measured
# echo case) and boson (c35/13633, the knowledge-rich control).
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
SEEDS_JSON='{"29":[3736],"35":[13633]}' \
N_NULL=2 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python experiments/043-relativity/know_runner.py \
  > experiments/043-relativity/know_full.log 2>&1
echo DONE > experiments/043-relativity/know.flag
