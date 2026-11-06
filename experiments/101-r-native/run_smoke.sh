#!/usr/bin/env bash
# 101: driver smoke + GPU equivalence of the native resample term (WCM_VERSION=v2cr), then the same targets under v2
# for the fit-time ratio. 062 driver MODE=main on 099's audited records (as 100), protocol settings (400 steps, lambda
# 1e-3, gamma 0.25, rank-keep 3e-3), CURVE_STRIDE 20 (100's), ORDER=list. Resumable: the driver skips finished targets
# (status rows); a killed fit is refitted. Refuses to start a version when the GPU is busy (shared GPU rule: > 10 GB
# used or >= 3 compute processes).
#   wsl: bash experiments/101-r-native/run_smoke.sh [v2cr|v2|both]   (from the repo root)
set -u
cd "$(dirname "$0")/../.."
E=experiments/101-r-native
VERS=${1:-both}
[ "$VERS" = both ] && VERS="v2cr v2"
mkdir -p "$E/out"
gpu_ok() {
  local used n
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  echo "[101] GPU: ${used} MiB used, ${n} compute processes"
  [ "${used:-99999}" -le 10240 ] && [ "${n:-9}" -lt 3 ]
}
for v in $VERS; do
  if ! gpu_ok; then echo "[101] GPU busy: not starting $v"; exit 3; fi
  echo "[101] $(date +%H:%M:%S) start $v"
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION="$v" OUT="$E/out/smoke_$v" \
    CTX_DIR=experiments/099-recursive-causal-scrubbing/out/records/ctx_v2 TARGETS="$E/targets_smoke.txt" \
    CURVE_STRIDE=20 ORDER=list .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py \
    2>&1 | tee -a "$E/out/smoke_$v.log"
  echo "[101] $(date +%H:%M:%S) end $v"
done
