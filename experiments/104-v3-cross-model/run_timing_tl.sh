#!/usr/bin/env bash
# 104 part D: clean local timing of the TuringLLM T3 work per target under v1 / v2 / v3 (fit, eval strong + mid,
# specificity: the T3 main job as the pod runs it, CURVE_STRIDE 1), then the donor test on the v2 and v3 main circuits
# (T11's job). 12 targets, one per layer (targets_timing_tl.txt, from 102's check panel), on the 15k run's records
# (a 12-record copy, so the donor R-pool filter reads only these). Refuses to start while another GPU process runs.
# Resumable (driver status rows, donor rows).
#   wsl (repo root): bash experiments/104-v3-cross-model/run_timing_tl.sh
set -u
cd "$(dirname "$0")/../.."
E=experiments/104-v3-cross-model
O=$E/out/timing_tl
T=$E/targets_timing_tl.txt
mkdir -p "$O" "$E/logs"
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
say() { echo "[104D $(date '+%F %T')] $*" | tee -a "$E/logs/timing_tl.events"; }
busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -gt 0 ]; }
if busy; then say "GPU busy: not starting"; nvidia-smi --query-compute-apps=pid,name --format=csv; exit 3; fi
$PY $E/subset_records.py experiments/062-h100-protocol-v1/out_full/out/ctx "$T" "$O/ctx" || exit 1
for V in v1 v2 v3; do
  say "fit $V start"
  OUT=$O/$V CTX_DIR=$O/ctx TARGETS=$T MODE=main WCM_VERSION=$V ORDER=list \
    $PY -X utf8 -u experiments/062-h100-protocol-v1/driver.py >> "$E/logs/timing_tl_fit_$V.log" 2>&1 || { say "fit $V FAILED"; exit 1; }
  say "fit $V end"
done
for V in v2 v3; do
  say "donor $V start"
  OUT=$O/$V CTX_DIR=$O/ctx WCM_VERSION=$V DONOR_TARGETS=$T DONOR_ARMS=main DONOR_POOL=$O/random_pool.pt \
    $PY -X utf8 -u src/eval/donor_test.py >> "$E/logs/timing_tl_donor_$V.log" 2>&1 || { say "donor $V FAILED"; exit 1; }
  say "donor $V end"
done
say "done"
