#!/bin/bash
# 100 E4 check confirmation (WSL, repo root): the arms selected by the README's pre-stated rule (report_e4.py), one arm
# at a time in selection order, on the 9 check targets. Usage: bash e4_check.sh <arm> [<arm>]
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
X=experiments/100-scrub-frontier
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
DL="--start-by 2026-10-31T00:30 --end-by 2026-10-31T01:00"
for ARM in "$@"; do
    echo "$(date +%H:%M:%S) start $ARM"
    .venv/bin/python -u $X/run.py --engine e4 --targets $X/targets_check9.txt --arms "$ARM" --out $X/out/e4 $DL \
        >> $X/logs/e4_check9.log 2>&1
    echo "$(date +%H:%M:%S) exit $? $ARM"
done
echo "$(date +%H:%M:%S) E4 CHECK DONE"
