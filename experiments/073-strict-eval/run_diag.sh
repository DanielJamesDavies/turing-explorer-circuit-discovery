#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
.venv/bin/python -X utf8 -u experiments/073-strict-eval/diag.py
