#!/bin/bash
# Circuits for the CHAIN links found by the edge audit: the L3
# Einstein's-theory latent (comp 11/18699) and the L6 relativistic-
# corrections latent (comp 20/23753), both top causal edges of the
# c29/3736 relativity-theory seed. Same harness, appends to the same
# rows/members files (done-guard keys on comp/latent).
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
SEEDS_JSON='{"11":[18699],"20":[23753]}' \
N_NULL=4 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python experiments/043-relativity/runner.py \
  > experiments/043-relativity/chain_circuits.log 2>&1
echo DONE > experiments/043-relativity/chain.flag
