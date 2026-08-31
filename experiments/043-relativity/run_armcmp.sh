#!/bin/bash
# Two-arm comparison (triamp400 vs negamp400) on 5 diverse L9 seeds,
# then profile analysis of both arms per seed.
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
R=experiments/043-relativity
SEEDS_JSON='{"29":[38310,17115,40651,24084,23920]}' N_NULL=1 \
  ARMS=triamp400,negamp400 \
  PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./.venv/bin/python $R/know_runner.py > $R/armcmp.log 2>&1
for L in 38310 17115 40651 24084 23920; do
  for ARM in triamp400 negamp400; do
    env COMP=29 LAT=$L ARM=$ARM MEMFILE=know_members.jsonl TOPM=15 \
      PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      ./.venv/bin/python $R/analyse_members.py \
      > "$R/an_c29_${L}_${ARM}.log" 2>&1
  done
done
echo DONE > $R/armcmp.flag
