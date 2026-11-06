#!/usr/bin/env bash
# 106 trimmed GPU session (used when the GPU came free late): profile base (shorter windows) -> bench of the three
# priority variants -> the remaining variants while time allows. Every step resumes.
#   wsl (repo root): nohup bash experiments/106-speed-profile/run_quick.sh > experiments/106-speed-profile/logs/run_quick.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
E=experiments/106-speed-profile
mkdir -p "$E/logs"
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
say() { echo "[106q $(date '+%F %T')] $*" | tee -a "$E/logs/run_all.events"; }
/usr/lib/wsl/lib/nvidia-smi dmon -s um -d 1 -o T > "$E/logs/dmon.$(date +%H%M%S).log" 2>&1 &
DM=$!
trap 'kill $DM 2> /dev/null' EXIT
X=early_exit,donor_gpu,rank_nosync
say "profile base start"
VARIANT= STEPS_T=60 PROF_WAIT=8 PROF_ACTIVE=3 $PY -X utf8 -u $E/profile_fit.py >> "$E/logs/profile_base.log" 2>&1
say "profile base end rc=$?"
say "bench A start"
VARIANTS="base;$X;$X,batch4" $PY -X utf8 -u $E/bench.py >> "$E/logs/bench.log" 2>&1; say "bench A end rc=$?"
say "bench B start"
VARIANTS="base;$X;$X,batch4;$X,compile_sp;$X,no_cfeval;$X,triton_topk;early_exit;base#2" \
  $PY -X utf8 -u $E/bench.py >> "$E/logs/bench.log" 2>&1; say "bench B end rc=$?"
say "done"
