#!/bin/bash
# 100 E4 development runs (WSL, repo root), one arm group at a time in the README's priority order, so a deadline cut
# removes the lowest priorities. Baselines are not refitted (Run A's wcm_1e-3 / v2cr_1e-3 in out/main).
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
X=experiments/100-scrub-frontier
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
DL="--start-by 2026-10-31T00:30 --end-by 2026-10-31T01:00"
for ARMS in "v2cr-w0.1_1e-3,v2cr-w0.5_1e-3,v2cr-w1_1e-3" "v2cr-mixCR_1e-3,v2cr-mixCAR_1e-3" "rsAC_1e-3" \
            "v2cr-k4_1e-3" "v2cr_2.5e-4" "v2cr-cap2_1e-3" "v2cr-s800_1e-3"; do
    echo "$(date +%H:%M:%S) start $ARMS"
    .venv/bin/python -u $X/run.py --engine e4 --targets $X/targets_dev9.txt --arms "$ARMS" --out $X/out/e4 $DL \
        >> $X/logs/e4_dev.log 2>&1
    echo "$(date +%H:%M:%S) exit $? $ARMS"
done
echo "$(date +%H:%M:%S) E4 DEV CHAIN DONE"
