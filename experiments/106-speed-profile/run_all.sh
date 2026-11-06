#!/usr/bin/env bash
# 106: the whole GPU session, in priority order; every step resumes (profile: one json per target; bench: fits.jsonl
# rows). Local GPU only, one job at a time.
#   wsl (repo root): nohup bash experiments/106-speed-profile/run_all.sh > experiments/106-speed-profile/logs/run_all.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
E=experiments/106-speed-profile
mkdir -p "$E/logs"
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
say() { echo "[106 $(date '+%F %T')] $*" | tee -a "$E/logs/run_all.events"; }
DMON=/usr/lib/wsl/lib/nvidia-smi
$DMON dmon -s um -d 1 -o T > "$E/logs/dmon.$(date +%H%M%S).log" 2>&1 &
DM=$!
trap 'kill $DM 2> /dev/null' EXIT
say "profile base start"
VARIANT= $PY -X utf8 -u $E/profile_fit.py >> "$E/logs/profile_base.log" 2>&1; say "profile base end rc=$?"
say "bench start"
$PY -X utf8 -u $E/bench.py >> "$E/logs/bench.log" 2>&1; say "bench end rc=$?"
say "profile exact start"
VARIANT=early_exit,donor_gpu,rank_nosync TRACE=0 $PY -X utf8 -u $E/profile_fit.py >> "$E/logs/profile_exact.log" 2>&1
say "profile exact end rc=$?"
say "profile batch4 start"
VARIANT=early_exit,donor_gpu,rank_nosync,batch4 TRACE=0 $PY -X utf8 -u $E/profile_fit.py >> "$E/logs/profile_batch4.log" 2>&1
say "profile batch4 end rc=$?"
say "done"
