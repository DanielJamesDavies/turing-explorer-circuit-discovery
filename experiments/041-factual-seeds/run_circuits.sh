#!/bin/bash
# Circuits for the VERIFIED concept seeds. Same harness, same arms, same
# null suite as the Gemma arena -- only the seed set differs, and it
# writes to its own rows/members files via ROWS_TAG.
cd "$(dirname "$0")/../038-transcoder-compare-gemma"
SCAN_FILE=../041-factual-seeds/run_seeds.pt \
ROWS_TAG=_fact \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  ../../.venv/bin/python "$PWD/ours_gtc.py" run \
  > ../041-factual-seeds/circuits.log 2>&1
echo DONE > ../041-factual-seeds/circuits.flag
