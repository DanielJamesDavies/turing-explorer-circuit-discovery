#!/usr/bin/env bash
# After the running driver.py finishes (DAN-131 band arms), smoke-test the DAN-17 random null: 1 target x 1 draw.
cd "$(dirname "$0")/../.." || exit 1
E=experiments/062-h100-protocol-v1
while pgrep -f "$E/driver.py" > /dev/null; do sleep 30; done
OUT=$E/out_full/out OUT_RN=$E/out_random_smoke N_TARGETS=1 N_DRAWS=1 PYTHONPATH=src \
  .venv/bin/python -X utf8 -u $E/random_null.py > $E/random_smoke.log 2>&1
echo "smoke exit $?" >> $E/random_smoke.log
