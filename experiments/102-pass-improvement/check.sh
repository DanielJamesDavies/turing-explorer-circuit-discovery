#!/usr/bin/env bash
# 102 phase 3: the frozen check panel (check_targets.json, seed 20261031). Base (v2CR) is FITTED here
# (--base-source fit), then the dev winner, then the runner-up, in ARMS order; --end-by stops before a (target, arm)
# whose projected end passes it. Resumable. Run only after the dev decision is written into README.md.
set -u
cd "$(dirname "$0")/../.."
LOG=experiments/102-pass-improvement/logs/check.log
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/102-pass-improvement/run.py \
    --panel check --allow-check --base-source fit --out experiments/102-pass-improvement/out/check \
    --arms "${ARMS:?set ARMS}" --end-by "${END_BY:-2026-11-01T00:30}" >> "$LOG" 2>&1
