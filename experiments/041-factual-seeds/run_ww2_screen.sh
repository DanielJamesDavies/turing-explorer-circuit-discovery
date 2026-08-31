#!/bin/bash
# Chain: wait for the concept screen to release the GPU, then screen the
# WW2 candidates. Both write their own *_seeds.pt so nothing is mixed.
cd "$(dirname "$0")"
while [ ! -f cscreen.flag ]; do sleep 30; done
CANDS=ww2_candidates.jsonl OUT=ww2_seeds.pt \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  ../../.venv/bin/python "$PWD/screen_factual.py" > ww2screen.log 2>&1
echo DONE > ww2screen.flag
