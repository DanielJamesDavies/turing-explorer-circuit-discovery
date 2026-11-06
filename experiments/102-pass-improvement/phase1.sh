#!/usr/bin/env bash
# 102 phase 1 on dev (WSL, repo root). One GPU process; arms in order (ARMS env). Resumable: rerun after a stop.
# Fits peak at 8-9.6 GB each (095 fits.jsonl), so two concurrent fits would not fit the 16 GB card next to r-port.
# Order (coordinator, 11:55): base re-score -> rk3e-2 -> rk1e-1 -> [decide rkBEST] -> r1.0 -> r1.0_rkBEST -> lamdepth
# -> twostage -> r0.5 -> seed1 -> seed2
#   stage A: ARMS=base,rk3e-2,rk1e-1
#   stage B: ARMS=r1.0,r1.0_rk<best>,lamdepth,twostage,r0.5,seed1,seed2
set -u
cd "$(dirname "$0")/../.."
LOG=experiments/102-pass-improvement/logs/phase1.log
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/102-pass-improvement/run.py \
    --arms "${ARMS:?set ARMS}" --end-by "${END_BY:-2026-11-01T00:30}" >> "$LOG" 2>&1
