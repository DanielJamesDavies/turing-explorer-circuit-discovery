#!/bin/bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
for SPEC in "20 23753" "11 18699"; do
  set -- $SPEC
  COMP=$1 LAT=$2 TOPK_INT=15 PYTHONPATH=src \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python experiments/043-relativity/edge_audit.py \
    > "experiments/043-relativity/audit_c$1_$2.log" 2>&1
done
echo DONE > experiments/043-relativity/chain_audits.flag
