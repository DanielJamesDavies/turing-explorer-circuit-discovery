#!/usr/bin/env bash
# After the 063 case-study run finishes, fit the case-study target (9.resid.20419) under unweighted circuit masking
# across the sweep's penalties, for the running-example sentence (weighted vs unweighted on the same target).
cd "$(dirname "$0")/../.." || exit 1
E=experiments/062-h100-protocol-v1
O=$E/out_case
mkdir -p $O/ctx $O/main/circuits
cp -n $E/out_full/out/ctx/9.resid.20419.pt $O/ctx/
cp -n $E/out_full/out/main/circuits/9.resid.20419.pt $O/main/circuits/
while pgrep -f "063-case-studies/case_study.py" > /dev/null; do sleep 30; done
MODE=sweep ORDER=list TARGETS=$E/targets_case.txt OUT=$O \
  SWEEP_ARMS=W_0.001,W_0.0005,W_0.00025,U_0.001,U_0.0003,U_0.0001,U_3e-05,U_1e-05 PYTHONPATH=src \
  .venv/bin/python -X utf8 -u $E/driver.py > $O/run.log 2>&1
echo "sweep exit $?" >> $O/run.log
