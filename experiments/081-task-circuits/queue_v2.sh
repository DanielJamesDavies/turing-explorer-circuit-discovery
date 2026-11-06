#!/usr/bin/env bash
# Experiment 081, the v2 local run (2026-10-27): WCM under protocol v2 (WCM_VERSION=v2: the A / C training terms under
# the scorer's sparsity-preserving fill, straight-through; the A / C means are the scorer's by construction) on GPT-2
# small, one GPU, one process at a time. Everything goes to out_v2/ (the v1 smoke in out_smoke/ is untouched).
# Stages (STAGES, in order; each is resumable, re-run after a restart):
#   smoke    smoke.sh under v2 -> out_v2/smoke (ioi + blimp_anaphor_gender_agreement, 12 steps, every stage)
#   prepare  the generated contrast pools (tasks/<MODEL>/<task>.contrast.json; existing files are kept)
#   probe    competence of every task -> out_v2/probe/competence.json
#   calib    calibration on CAL_TASKS (training rows only, re-split 75 / 25), per logit normaliser in NORMS
#            ("gap" = 080's shared zero-gap scale, "gap_term" = each term by its own closed-mask gap) ->
#            out_v2/calib_<norm>/ (LAMS_W / LAMS_U; an empty LAMS_U skips U)
#   main     run.py with CONFIG (config_v2_<MODEL>.json, written by hand from the calibration) -> out_v2/<MODEL>
#   report   report.py + figure.py on out_v2/<MODEL>
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && STAGES="smoke prepare probe" nohup setsid bash experiments/081-task-circuits/queue_v2.sh >> experiments/081-task-circuits/out_v2/queue.log 2>&1 < /dev/null & disown'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/081-task-circuits
O=$E/out_v2
export MODEL=${MODEL:-gpt2-small}
export WCM_VERSION=v2 PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export OUT_PROBE=$O/probe
PY=.venv/bin/python
STAGES=${STAGES:-"smoke prepare probe calib"}
NORMS=${NORMS:-"gap gap_term"}
CAL_TASKS=${CAL_TASKS:-ioi,greater_than}
LAMS_W=${LAMS_W-1e-4,3e-4,1e-3,3e-3}
LAMS_U=${LAMS_U-1e-5,3e-5,1e-4,3e-4}
CONFIG=${CONFIG:-$E/config_v2_$MODEL.json}
mkdir -p "$O"
step() { echo "[$(date +%H:%M:%S)] $*"; }
while pgrep -f "081-task-circuits/(prepare|probe|run|calibrate).py" > /dev/null; do sleep 30; done
step "queue_v2 start: STAGES=[$STAGES] WCM_VERSION=$WCM_VERSION MODEL=$MODEL"

for S in $STAGES; do
  case $S in
    smoke)
      step "smoke -> $O/smoke"
      OUT=$O/smoke bash $E/smoke.sh > $O/smoke.log 2>&1
      step "    smoke exit $?"; tail -16 $O/smoke.log ;;
    prepare)
      step "prepare (generated contrast pools)"
      $PY -X utf8 -u $E/prepare.py >> $O/prepare.log 2>&1
      step "    prepare exit $?"; tail -12 $O/prepare.log ;;
    probe)
      step "probe -> $OUT_PROBE"
      mkdir -p $OUT_PROBE
      $PY -X utf8 -u $E/probe.py > $OUT_PROBE/probe.log 2>&1
      step "    probe exit $?"; tail -24 $OUT_PROBE/probe.log ;;
    calib)
      for N in $NORMS; do
        C_OUT=$O/calib_$N
        mkdir -p $C_OUT
        step "calibration logit_norm=$N tasks=$CAL_TASKS W=[$LAMS_W] U=[$LAMS_U] -> $C_OUT"
        OUT_CAL=$C_OUT LOGIT_NORM=$N CAL_TASKS=$CAL_TASKS LAMS_W=$LAMS_W LAMS_U=$LAMS_U \
          $PY -X utf8 -u $E/calibrate.py >> $C_OUT/run.log 2>&1
        step "    calibration $N exit $?"
        OUT_CAL=$C_OUT REPORT=1 $PY -X utf8 $E/calibrate.py > $C_OUT/report.txt 2>&1
        tail -14 $C_OUT/report.txt
      done ;;
    main)
      step "main run CONFIG=$CONFIG -> $O/$MODEL"
      mkdir -p $O/$MODEL
      OUT=$O/$MODEL CONFIG=$CONFIG $PY -X utf8 -u $E/run.py >> $O/$MODEL/run.log 2>&1
      step "    main exit $?"; tail -6 $O/$MODEL/run.log ;;
    report)
      OUT=$O/$MODEL $PY -X utf8 $E/report.py > $O/$MODEL/report.txt 2>&1
      step "    report exit $? -> $O/$MODEL/report.txt"
      OUT=$O/$MODEL $PY -X utf8 $E/figure.py > $O/$MODEL/figure.log 2>&1
      step "    figure exit $?" ;;
    *) step "unknown stage $S"; exit 1 ;;
  esac
done
step "queue_v2 DONE [$STAGES]"
