#!/bin/bash
# 103 equivalence run (WSL, from anywhere): the native donor test (src/eval/donor_test.py, through its CLI) on 3 of
# experiment 100's targets, on 099's audited records, for 100's two lambda-1e-3 arms: wcm_1e-3 (= 099's panel18
# circuits, which 100 reused) and v2cr_1e-3 (100's fits). The R pool is BUILT here by the native path (073's
# random_pool on the local corpus, filtered against the 18 records in CTX_DIR) and compared with 099's panel18 pool by
# equiv.py. Production rows are re-scored (Runner.score). Then: python experiments/103-donor-stage/equiv.py
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
E=experiments/103-donor-stage
O=${O:-$E/out/equiv}
mkdir -p "$O"
env OUT="$O/driver" CTX_DIR=experiments/099-recursive-causal-scrubbing/out/records/ctx_v2 WCM_VERSION=v2 \
  DONOR_TARGETS="${TARGETS:-$E/targets_equiv.txt}" OUT_DONOR="$O" DONOR_POOL="$O/random_pool.pt" \
  DONOR_ARMS="wcm_1e-3=experiments/099-recursive-causal-scrubbing/out/panel18/circuits/{key}.wcm.pt,v2cr_1e-3=experiments/100-scrub-frontier/out/main/circuits/v2cr_1e-3/{key}.pt" \
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u -X utf8 src/eval/donor_test.py
