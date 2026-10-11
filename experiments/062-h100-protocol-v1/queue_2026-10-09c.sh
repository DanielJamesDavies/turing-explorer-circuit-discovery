#!/usr/bin/env bash
# Validation gates for the H100 code (068), queued behind queue b (2026-10-09). One job at a time; both resumable.
#   wsl -e bash -lc 'cd <repo> && nohup setsid bash experiments/062-h100-protocol-v1/queue_2026-10-09c.sh > experiments/062-h100-protocol-v1/queue_2026-10-09c.log 2>&1 &'
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src HF_HUB_OFFLINE=1
PY=.venv/bin/python
G=experiments/068-validation-gates

while pgrep -f "queue_2026-10-09(b)?\.sh" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M)] queue c start"

echo "[$(date +%H:%M)] job 1: TuringLLM golden set, 32 targets refit twice (DAN-74)"
CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -u $G/golden_turingllm.py > $G/golden.log 2>&1
echo "[$(date +%H:%M)] job 1 exit $?"

echo "[$(date +%H:%M)] job 2: 15k circuits re-scored on unseen held-out contexts only (exact-duplicate leakage)"
OUT=experiments/062-h100-protocol-v1/out_full/out CUDA_VISIBLE_DEVICES=0 \
  $PY -X utf8 -u experiments/062-h100-protocol-v1/rescore_dedup.py > experiments/062-h100-protocol-v1/rescore_dedup.log 2>&1
echo "[$(date +%H:%M)] job 2 exit $?"

echo "[$(date +%H:%M)] job 3: the 033 Pythia panel through the engine, GPU (DAN-36)"
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda SEEDS=all OUT=$G/out_panel_gpu $PY -X utf8 -u $G/pythia_033_panel.py > $G/panel_gpu.log 2>&1
echo "[$(date +%H:%M)] job 3 exit $?"

echo "[$(date +%H:%M)] queue c DONE"
