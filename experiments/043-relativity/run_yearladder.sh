#!/bin/bash
# Circuits for the six year-ladder latents found by completion-flow
# tracing (the production pathway for "...in the year -> 1905").
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
SEEDS_JSON='{"14":[16611],"17":[23392],"20":[23868],"23":[32639],"26":[37742],"29":[11019]}' \
  N_NULL=1 ARMS=triamp400 \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/know_runner.py > $R/yearladder.log 2>&1
for SPEC in "14 16611" "17 23392" "20 23868" "23 32639" "26 37742" "29 11019"; do
  set -- $SPEC
  env COMP=$1 LAT=$2 ARM=triamp400 MEMFILE=know_members.jsonl TOPM=12 \
    PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python $R/analyse_members.py > "$R/an_year_$1_$2.log" 2>&1
done
echo DONE > $R/yearladder.flag
