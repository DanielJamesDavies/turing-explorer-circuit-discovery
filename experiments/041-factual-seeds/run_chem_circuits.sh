#!/bin/bash
cd "$(dirname "$0")/../038-transcoder-compare-gemma"
SCAN_FILE=../041-factual-seeds/run_seeds.pt \
ROWS_TAG=_chem \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  ../../.venv/bin/python "$PWD/ours_gtc.py" run \
  > ../041-factual-seeds/chem_circuits.log 2>&1
echo DONE > ../041-factual-seeds/chem_circuits.flag
