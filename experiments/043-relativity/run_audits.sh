#!/bin/bash
# Full edge audits: boson (c35/13633), then the relativity-theory seed
# (c29/3736). Mean-fill frame, canonical evaluator, held-out probes.
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
for SPEC in "35 13633" "29 3736"; do
  set -- $SPEC
  COMP=$1 LAT=$2 TOPK_INT=20 PYTHONPATH=src \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python experiments/043-relativity/edge_audit.py \
    > "experiments/043-relativity/audit_c$1_$2.log" 2>&1
done
echo DONE > experiments/043-relativity/audits.flag
