#!/usr/bin/env bash
# Experiment 081 local smoke test (one GPU, one process at a time), end to end on 2 GPT-2 tasks at toy settings:
#   ioi (the dataset's own ABC patch prompts as C) and blimp_anaphor_gender_agreement (no patch prompts: the generated
#   close-silent contrast pool as C, plus the W_ZA sensitivity arm). NOT a result: 12 steps at lr 0.5
#   (config_smoke.json: 12 x 4 = 48 training presentations per fit), 2 random draws, and only the first HELDOUT_N=200
#   held-out items (a fixed random subset; the task files hold 5,000 for ioi and 250 for the BLiMP task) with member
#   profiles on the first PROFILE_N=80. The A / C means still cover the whole training / contrast pools (ioi: 1,600
#   prompts each), once per stage.
#   1. the 081 CPU tests
#   2. prepare.py: the BLiMP task's contrast pool (written to tasks/<MODEL>/<task>.contrast.json, the same file the
#      main run uses; deterministic)
#   3. probe.py on the 2 tasks -> OUT/probe
#   4. a code check of calibrate.py: ioi x (1 W + 1 U lambda) x 12 steps, then its REPORT (NOT the calibration stage)
#   5. run.py: W, U at 2 lambdas, RND 2 draws per W lambda, W_ZA for the BLiMP task, profiles (FORCE_TASKS=1)
#   6. report.py and figure.py on the smoke output
# Resumable. Expected ~15-25 min including ~1-2 min model + 48-SAE load per stage (an estimate). Do not run while
# another job holds the GPU.
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && bash experiments/081-task-circuits/smoke.sh'
#   env: MODEL (gpt2-small)  OUT (experiments/081-task-circuits/out_smoke/<MODEL>)  PY (.venv/bin/python)
#        HELDOUT_N (200)  PROFILE_N (80)
cd "$(dirname "$0")/../.." || exit 1
E=experiments/081-task-circuits
export MODEL=${MODEL:-gpt2-small}
O=${OUT:-$E/out_smoke/$MODEL}
PY=${PY:-.venv/bin/python}
T_ORIG=ioi
T_GEN=blimp_anaphor_gender_agreement
export PYTHONPATH=src HF_HUB_OFFLINE=1
export HELDOUT_N=${HELDOUT_N:-200} PROFILE_N=${PROFILE_N:-80}
mkdir -p "$O"
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "1/6 CPU tests"
CUDA_VISIBLE_DEVICES= $PY -m pytest -q -p no:cacheprovider $E 2>&1 | tail -2

step "2/6 prepare ($T_GEN contrast pool)"
TASKS=$T_GEN $PY -X utf8 -u $E/prepare.py >> "$O/prepare.log" 2>&1
step "    exit $?"; tail -3 "$O/prepare.log"

step "3/6 probe ($T_ORIG, $T_GEN) -> $O/probe"
OUT_PROBE=$O/probe TASKS=$T_ORIG,$T_GEN $PY -X utf8 -u $E/probe.py >> "$O/probe.log" 2>&1
step "    exit $?"; tail -6 "$O/probe.log"

step "4/6 calibrate.py code check -> $O/calib"
OUT_PROBE=$O/probe OUT_CAL=$O/calib CAL_TASKS=$T_ORIG FORCE_TASKS=1 LAMS_W=0.001 LAMS_U=0.0001 \
  LRS_W=0.5 LRS_U=0.5 STEPS=12 $PY -X utf8 -u $E/calibrate.py >> "$O/calib.log" 2>&1
step "    exit $?"; tail -3 "$O/calib.log"
OUT_PROBE=$O/probe OUT_CAL=$O/calib REPORT=1 $PY -X utf8 $E/calibrate.py 2>&1 | tail -8

step "5/6 run.py (12 steps, config_smoke.json) -> $O/run.log"
OUT=$O OUT_PROBE=$O/probe CONFIG=$E/config_smoke.json TASKS=$T_ORIG,$T_GEN FORCE_TASKS=1 \
  $PY -X utf8 -u $E/run.py >> "$O/run.log" 2>&1
step "    exit $?"; tail -4 "$O/run.log"

step "6/6 report + figure"
OUT=$O OUT_PROBE=$O/probe $PY -X utf8 $E/report.py > "$O/report.txt" 2>&1
step "    report exit $? -> $O/report.txt"; tail -14 "$O/report.txt"
OUT=$O OUT_PROBE=$O/probe $PY -X utf8 $E/figure.py 2>&1 | tail -2
step "smoke DONE"
