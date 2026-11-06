#!/usr/bin/env bash
# Experiment 081, v2 local run (2026-10-27): DIAGNOSTICS after queue_v2_main.sh (not protocol results):
#   1. v2_insample.py: the IOI / greater-than W and U circuits re-scored on 1,000 TRAINING items (training name / noun
#      lists) -> out_v2/gpt2-small/insample.jsonl
#   2. IOI W lambda 3e-4 at 1,200 steps (config_v2_ioi_steps1200.json) -> out_v2/ioi_steps1200 (no profile, no RND)
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && bash experiments/081-task-circuits/queue_v2_diag.sh >> experiments/081-task-circuits/out_v2/queue_diag.log 2>&1'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/081-task-circuits
O=$E/out_v2
export MODEL=gpt2-small WCM_VERSION=v2 PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 OUT_PROBE=$O/probe
export EVAL_BS=${EVAL_BS:-64}
PY=.venv/bin/python
STAGES=${STAGES:-"insample steps1200"}
step() { echo "[$(date +%H:%M:%S)] $*"; }
while pgrep -f "081-task-circuits/((prepare|probe|run|calibrate|v2_insample).py|queue_v2_main.sh)" > /dev/null; do
  sleep 30; done
for S in $STAGES; do
  case $S in
    insample)
      step "in-sample re-score -> $O/$MODEL/insample.jsonl"
      OUT=$O/$MODEL $PY -X utf8 -u $E/v2_insample.py >> $O/$MODEL/insample.log 2>&1
      step "    exit $?"; tail -12 $O/$MODEL/insample.log ;;
    steps1200)
      mkdir -p $O/ioi_steps1200
      step "IOI W 3e-4 at 1200 steps -> $O/ioi_steps1200"
      OUT=$O/ioi_steps1200 TASKS=ioi ARMS=W NO_PROFILE=1 CONFIG=$E/config_v2_ioi_steps1200.json \
        $PY -X utf8 -u $E/run.py >> $O/ioi_steps1200/run.log 2>&1
      step "    exit $?"; grep "lam " $O/ioi_steps1200/run.log | tail -2 ;;
  esac
done
step "queue_v2_diag DONE [$STAGES]"
