#!/usr/bin/env bash
# 106 minimal GPU session (~10 min): profile base on the 3 panel targets with short windows, then base vs the exact
# bundle (early_exit + donor_gpu + rank_nosync) on the 3 panel targets, 400 steps, with the equivalence check.
#   wsl (repo root): nohup bash experiments/106-speed-profile/run_min.sh > experiments/106-speed-profile/logs/run_min.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
E=experiments/106-speed-profile
mkdir -p "$E/logs"
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
say() { echo "[106m $(date '+%F %T')] $*" | tee -a "$E/logs/run_all.events"; }
X=early_exit,donor_gpu,rank_nosync
say "bench min start"
VARIANTS="base;$X" $PY -X utf8 -u $E/bench.py >> "$E/logs/bench.log" 2>&1; say "bench min end rc=$?"
say "profile base start"
VARIANT= STEPS_T=40 PROF_WAIT=6 PROF_ACTIVE=3 $PY -X utf8 -u $E/profile_fit.py >> "$E/logs/profile_base.log" 2>&1
say "profile base end rc=$?"
say "done"
