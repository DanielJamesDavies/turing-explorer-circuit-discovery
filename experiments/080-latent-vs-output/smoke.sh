#!/usr/bin/env bash
# Experiment 080 local smoke test (one RTX, one process), end to end on the 2 smoke targets (targets_smoke.txt: an
# L1 mlp target and an L11 resid target, the deepest kind; direct-path (064) low and high):
#   1. the 080 logic tests
#   2. run.py: LAT re-scored + OUT_W and OUT_U at 2 lambdas each, 12 steps at lr 0.5 (config_smoke.json; not a
#      calibrated setting), both scorers on every circuit
#   3. a timing pass: the same arms at 1 lambda, 40 steps (config_smoke_timing.json), for estimate.py
#   4. a code check of calibrate.py: 1 target x 2 fits x 12 steps into OUT/calib, then its REPORT (this is NOT the
#      calibration stage; that is queue_calib.sh)
#   5. report.py (SMOKE=1: per-row listing, LAT re-score vs 062's own eval rows, output denominators), figure.py
#   6. estimate.py: projected calibration and main-run times on this GPU
# Resumable: re-run to continue (finished fits and rows are skipped). Fresh: ~10-20 min.
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && bash experiments/080-latent-vs-output/smoke.sh'
#   env: OUT (experiments/080-latent-vs-output/out_smoke)  OUT_TIMING (..._timing)  PY (.venv/bin/python)
cd "$(dirname "$0")/../.." || exit 1
E=experiments/080-latent-vs-output
O=${OUT:-$E/out_smoke}
OT=${OUT_TIMING:-$E/out_smoke_timing}
PY=${PY:-.venv/bin/python}
export PYTHONPATH=src HF_HUB_OFFLINE=1
mkdir -p "$O" "$OT"
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "1/6 logic tests"
CUDA_VISIBLE_DEVICES= $PY -m pytest -q -p no:cacheprovider $E/test_common.py 2>&1 | tail -2

step "2/6 run.py (12 steps) -> $O/run.log"
OUT=$O TARGETS=$E/targets_smoke.txt CONFIG=$E/config_smoke.json $PY -X utf8 -u $E/run.py >> "$O/run.log" 2>&1
step "    exit $?"; tail -3 "$O/run.log"

step "3/6 timing pass (40 steps) -> $OT/run.log"
OUT=$OT TARGETS=$E/targets_smoke.txt CONFIG=$E/config_smoke_timing.json $PY -X utf8 -u $E/run.py >> "$OT/run.log" 2>&1
step "    exit $?"; tail -2 "$OT/run.log"

step "4/6 calibrate.py code check (1 target x 2 fits x 12 steps) -> $O/calib"
OUT_CAL=$O/calib TARGETS=$E/targets_smoke.txt LIMIT=1 LAMS_OUT_W=0.001 LAMS_OUT_U=0.0001 LRS=0.5 STEPS=12 \
  $PY -X utf8 -u $E/calibrate.py >> "$O/calib.log" 2>&1
step "    exit $?"; tail -2 "$O/calib.log"
OUT_CAL=$O/calib REPORT=1 $PY -X utf8 $E/calibrate.py 2>&1 | tail -12

step "5/6 report + figure"
SMOKE=1 OUT=$O B=200 $PY -X utf8 $E/report.py
OUT=$O B=200 $PY -X utf8 $E/figure.py

step "6/6 runtime estimate"
OUT_SMOKE=$O OUT_TIMING=$OT $PY -X utf8 $E/estimate.py
step "smoke DONE"
