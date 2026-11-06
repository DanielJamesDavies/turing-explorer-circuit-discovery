#!/usr/bin/env bash
# 104 part A: bit-identity of the native WCM_VERSION=v3 (062 driver) with exp 102's arm r1.0_rk1e-1 (p102 engine on
# a frozen learned_mask copy). 3 check-panel targets, one per depth band (targets_bitident.txt); the same records as 102
# (062 out_full/out/ctx, explicit split), MODE=main, 400 steps, CURVE_STRIDE 20 (102's), ORDER=list, SKIP_SPEC=1.
# Resumable: the driver skips targets with a status row.
#   wsl (repo root): bash experiments/104-v3-cross-model/run_bitident.sh
set -u
cd "$(dirname "$0")/../.."
E=experiments/104-v3-cross-model
mkdir -p "$E/out" "$E/logs"
echo "[104A] $(date +%H:%M:%S) start"
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 OUT="$E/out/bitident" \
  CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx TARGETS="$E/targets_bitident.txt" \
  CURVE_STRIDE=20 ORDER=list SKIP_SPEC=1 .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py \
  >> "$E/logs/bitident.log" 2>&1
rc=$?
echo "[104A] $(date +%H:%M:%S) end rc=$rc"
exit $rc
