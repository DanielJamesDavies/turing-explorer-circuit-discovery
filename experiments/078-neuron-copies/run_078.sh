#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
echo "[$(date +%H:%M)] 078 start"
.venv/bin/python -X utf8 -u experiments/078-neuron-copies/neuron_copies.py
echo "[$(date +%H:%M)] 078 exit $?"
