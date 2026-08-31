#!/bin/bash
cd "$(dirname "$0")"
CANDS=chem_candidates.jsonl OUT=chem_seeds.pt \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  ../../.venv/bin/python "$PWD/screen_factual.py" > chemscreen.log 2>&1
echo DONE > chemscreen.flag
