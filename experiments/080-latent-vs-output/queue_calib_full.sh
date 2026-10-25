#!/usr/bin/env bash
# Experiment 080, whole-model scope calibration (2026-10-22): OUT_W_FULL and OUT_U_FULL on targets_calib.txt (18), every
# site maskable (logit_scope "all"), so no layer after the circuit runs live. One lr per arm, the upstream calibration's
# best (calib/report.txt: OUT_W lr 0.05, OUT_U lr 0.02); lambda grids from calibrate.DEFAULT_LAMS (W 5, U 6) -> 198 fits.
# Rows -> calib_full/eval.jsonl, report -> calib_full/report.txt (its LAT-size bracketing warning does not apply to this
# scope: the LAT circuits only see upstream sites). Resumable. Waits for any other 080 job on the card first.
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && bash experiments/080-latent-vs-output/queue_calib_full.sh > experiments/080-latent-vs-output/queue_calib_full.log 2>&1'
cd "$(dirname "$0")/../.." || exit 1
E=experiments/080-latent-vs-output
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
while pgrep -f "080-latent-vs-output/(run|calibrate).py" > /dev/null; do sleep 30; done
mkdir -p $E/calib_full
echo "[$(date +%H:%M)] OUT_W_FULL start"
OUT_CAL=$E/calib_full CAL_ARMS=OUT_W_FULL LRS=0.05 $PY -X utf8 -u $E/calibrate.py >> $E/calib_full/run.log 2>&1
echo "[$(date +%H:%M)] OUT_W_FULL exit $?"
echo "[$(date +%H:%M)] OUT_U_FULL start"
OUT_CAL=$E/calib_full CAL_ARMS=OUT_U_FULL LRS=0.02 $PY -X utf8 -u $E/calibrate.py >> $E/calib_full/run.log 2>&1
echo "[$(date +%H:%M)] OUT_U_FULL exit $?"
OUT_CAL=$E/calib_full REPORT=1 $PY -X utf8 $E/calibrate.py > $E/calib_full/report.txt 2>&1
echo "[$(date +%H:%M)] report -> $E/calib_full/report.txt (not adopted)"
echo "[$(date +%H:%M)] queue DONE"
