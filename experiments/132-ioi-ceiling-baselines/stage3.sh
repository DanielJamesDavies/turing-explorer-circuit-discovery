#!/bin/bash
# One GPU job (under gpujob.sh): the SAME-NAMES view (exp 133's held_sn items: names from the training halves of the
# lists, none a training item). (1) ceiling pools on all 998 SN items; (2) baseline curves on the first 500 SN items.
set -uo pipefail
E=experiments/132-ioi-ceiling-baselines
ENV="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HELD_SET=sn"
env $ENV OUT132=$E/out_sn VARIANTS=active_all,active_scope_attnmlp ./.venv/bin/python -X utf8 -u $E/ceiling.py || exit 1
env $ENV HELDOUT_N=500 CURVES=curves_sn500.jsonl BUDGET_MIN=${BUDGET_MIN:-28} \
  METHODS=${METHODS:-c081,atp,atp@am,atp@am+inv,sfc100,sfc100@am} ./.venv/bin/python -X utf8 -u $E/sweep.py || exit 2
