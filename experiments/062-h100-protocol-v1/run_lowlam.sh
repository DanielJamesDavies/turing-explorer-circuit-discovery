#!/usr/bin/env bash
# Figure 3 budget curves (2026-10-05): the three weakest WCM penalties on the sweep targets that out_bands / out_depth
# did not cover, so every sweep target has WCM at all eight penalties (4e-3 .. 2.5e-5). Resumable.
cd "$(dirname "$0")/../.." || exit 1
E=experiments/062-h100-protocol-v1
O=$E/out_lowlam
mkdir -p $O/ctx $O/main/circuits
for k in $(cat $E/targets_lowlam127.txt); do
  cp -n $E/out_full/out/ctx/$k.pt $O/ctx/ 2>/dev/null
done
MODE=sweep ORDER=list TARGETS=$E/targets_lowlam127.txt OUT=$O SWEEP_ARMS=W_0.0001,W_5e-05,W_2.5e-05 PYTHONPATH=src \
  .venv/bin/python -X utf8 -u $E/driver.py >> $O/run.log 2>&1
echo "lowlam exit $?" >> $O/run.log
