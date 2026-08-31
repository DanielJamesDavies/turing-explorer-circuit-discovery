#!/bin/bash
# Chain: (1) knowledge loss on the worst-echo seed c11/18699;
# (2) ECHO_W sweep on c29/3736 (know400 combo arm only);
# (3) edge audits of the know400 circuits (c29, c35).
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity

SEEDS_JSON='{"11":[18699]}' N_NULL=1 \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/know_runner.py > $R/know_c11.log 2>&1

SEEDS_JSON='{"29":[3736]}' N_NULL=1 ARMS=know400 KTAG=_e2 ECHO_W=2e-2 \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/know_runner.py > $R/know_e2.log 2>&1

SEEDS_JSON='{"29":[3736]}' N_NULL=1 ARMS=know400 KTAG=_e5 ECHO_W=5e-2 \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/know_runner.py > $R/know_e5.log 2>&1

for SPEC in "29 3736" "35 13633"; do
  set -- $SPEC
  COMP=$1 LAT=$2 TOPK_INT=15 MEMFILE=know_members.jsonl AUDIT_ARM=know400 \
    PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python $R/edge_audit.py > "$R/audit_know_c$1_$2.log" 2>&1
done
echo DONE > $R/nextmoves.flag
