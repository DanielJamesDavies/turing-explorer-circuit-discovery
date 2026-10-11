#!/usr/bin/env bash
# Local GPU queue, 2026-10-10. One job at a time; both resumable.
#   1. DAN-74 code-vs-hardware check: refit the 32 golden targets with the COMMITTED src (= the 15k run's 538cbed,
#      exported to OLDSRC with the built native extensions copied in), then compare bitwise with yesterday's
#      working-tree refit r0 (compare_reps.py).
#   2. DAN-36: resume the 033 Pythia panel through the engine on the GPU (70 rows done before WSL stopped).
#   wsl -e bash -lc 'cd <repo> && OLDSRC=<dir>/src nohup setsid bash experiments/068-validation-gates/queue_2026-10-10.sh > experiments/068-validation-gates/queue_2026-10-10.log 2>&1 &'
cd "$(dirname "$0")/../.." || exit 1
G=experiments/068-validation-gates
PY=.venv/bin/python
export HF_HUB_OFFLINE=1
echo "[$(date +%H:%M)] queue start (old src: $OLDSRC)"

echo "[$(date +%H:%M)] job 1: golden refit with the committed src (r_old)"
PYTHONPATH="$OLDSRC" CUDA_VISIBLE_DEVICES=0 PHASES=fit REPS=r_old $PY -X utf8 -u $G/golden_turingllm.py > $G/golden_oldsrc.log 2>&1
echo "[$(date +%H:%M)] job 1 exit $?"
PYTHONPATH=src $PY -X utf8 $G/compare_reps.py > $G/golden_oldsrc_compare.txt 2>&1
tail -1 $G/golden_oldsrc_compare.txt

echo "[$(date +%H:%M)] job 2: resume the 033 Pythia panel, GPU"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 DEVICE=cuda SEEDS=all OUT=$G/out_panel_gpu $PY -X utf8 -u $G/pythia_033_panel.py >> $G/panel_gpu.log 2>&1
echo "[$(date +%H:%M)] job 2 exit $?"

echo "[$(date +%H:%M)] queue DONE"
