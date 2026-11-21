#!/bin/bash
# One GPU job (run under gpujob.sh): full attributions on the 1,600 training items, a sweep smoke on the smoke
# scores (8 held-out items), then the remaining stage-1 ceiling variants (active pools at resid scopes, oracles).
set -uo pipefail
E=experiments/132-ioi-ceiling-baselines
ENV="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
env $ENV ./.venv/bin/python -X utf8 -u $E/attrib.py || exit 1
env $ENV HELDOUT_N=8 OUT132=$E/out_smoke N_SMALL=25 N_LARGE=50 MAX_N=3000 BUDGET_MIN=6 \
  ./.venv/bin/python -X utf8 -u $E/sweep.py || exit 2
env $ENV VARIANTS=active_scope_resid,active_scope_resid2,oracle_heldout,oracle_union \
  ./.venv/bin/python -X utf8 -u $E/ceiling.py || exit 3
