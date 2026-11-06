#!/usr/bin/env bash
# Experiment 081, v2 local run (2026-10-27), after queue_v2.sh (smoke, prepare, probe, W calibration): in order
#   1. a v1 control on the calibration split (WCM_VERSION=v1, logit_norm gap): IOI W 3e-4, greater-than W 1e-4 / 3e-4
#      -> out_v2/calib_v1_gap (the same fits as out_v2/calib_gap, dense A / C training fill)
#   2. main fits, per task with its own hand-written grid (config_v2_<task>.json; README "v2 local run"):
#      ioi, greater_than (W, U, RND 3 draws at one W lambda), then the secondary tasks (config_v2_small.json: W, U,
#      RND 2 draws at one W lambda; no W_ZA) -> out_v2/gpt2-small (one OUT; eval rows carry their lambda)
#   3. report + figure
# Resumable (circuits on disk are reloaded; eval rows are not redone). One GPU process at a time.
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && bash experiments/081-task-circuits/queue_v2_main.sh >> experiments/081-task-circuits/out_v2/queue_main.log 2>&1'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/081-task-circuits
O=$E/out_v2
export MODEL=gpt2-small PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 OUT_PROBE=$O/probe EVAL_BS=${EVAL_BS:-64}
PY=.venv/bin/python
OM=$O/$MODEL
SMALL=${SMALL:-gender_pronoun,sva_across_pp,sva_simple,factual_known,blimp_anaphor_number_agreement}
SKIP_V1=${SKIP_V1:-0}
step() { echo "[$(date +%H:%M:%S)] $*"; }
while pgrep -f "081-task-circuits/(prepare|probe|run|calibrate).py" > /dev/null; do sleep 30; done
mkdir -p $OM $O/calib_v1_gap

if [ "$SKIP_V1" != 1 ]; then
  step "v1 control (calibration split) -> $O/calib_v1_gap"
  WCM_VERSION=v1 OUT_CAL=$O/calib_v1_gap LOGIT_NORM=gap CAL_TASKS=ioi LAMS_W=3e-4 LAMS_U= \
    $PY -X utf8 -u $E/calibrate.py >> $O/calib_v1_gap/run.log 2>&1
  WCM_VERSION=v1 OUT_CAL=$O/calib_v1_gap LOGIT_NORM=gap CAL_TASKS=greater_than LAMS_W=1e-4,3e-4 LAMS_U= \
    $PY -X utf8 -u $E/calibrate.py >> $O/calib_v1_gap/run.log 2>&1
  step "    v1 control exit $?"; grep -E "lam " $O/calib_v1_gap/run.log | tail -4
fi

main() {   # $1 tag, $2 TASKS, $3 CONFIG, $4 RND_LAMS, $5 N_DRAWS, $6 ARMS
  step "main [$1] tasks=$2 config=$3 rnd_lams=$4 draws=$5 arms=$6"
  WCM_VERSION=v2 OUT=$OM TASKS=$2 CONFIG=$3 RND_LAMS=$4 N_DRAWS=$5 ARMS=$6 \
    $PY -X utf8 -u $E/run.py >> $OM/run.log 2>&1
  step "    main [$1] exit $?"
  cp $OM/run_config.json $OM/run_config_$1.json
  tail -3 $OM/run.log
}
main ioi ioi $E/config_v2_ioi.json 3e-4 3 W,U,RND
main greater_than greater_than $E/config_v2_greater_than.json 1e-4 3 W,U,RND
main small $SMALL $E/config_v2_small.json 3e-4 2 W,U,RND

step "report + figure"
WCM_VERSION=v2 OUT=$OM $PY -X utf8 $E/report.py > $OM/report.txt 2>&1
step "    report exit $?"
WCM_VERSION=v2 OUT=$OM $PY -X utf8 $E/figure.py > $OM/figure.log 2>&1
step "    figure exit $?"
step "queue_v2_main DONE"
