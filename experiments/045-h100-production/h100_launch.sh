#!/bin/bash
# 8xH100 launcher: one shard per GPU, retry-per-shard (no set -e — see
# PROTOCOL.md harness trap #7), logs per shard. Run from the repo root
# on the H100 image with RUN_ROOT pointing at LOCAL NVMe artifacts.
#
#   RUN_ROOT=/nvme/run-artifacts ARMS=triamp400 ./experiments/045-h100-production/h100_launch.sh
#   # then the neg-amp pass:
#   RUN_ROOT=/nvme/run-artifacts ARMS=<sgnamp400|negsup400> ./experiments/045-h100-production/h100_launch.sh

K=${K:-8}
ARMS=${ARMS:-triamp400}
D=experiments/045-h100-production
mkdir -p "$D/logs"

for i in $(seq 0 $((K - 1))); do
  (
    tries=0
    until CUDA_VISIBLE_DEVICES=$i SEED_SHARD=$i/$K ARMS=$ARMS \
        PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python experiments/045-h100-production/driver.py \
        >> "$D/logs/shard$i.log" 2>&1; do
      tries=$((tries + 1))
      echo "shard $i exited nonzero (attempt $tries)" >> "$D/logs/shard$i.log"
      [ $tries -ge 5 ] && { echo "shard $i GIVING UP" >> "$D/logs/shard$i.log"; break; }
      sleep 30
    done
    echo "shard $i finished" >> "$D/logs/shard$i.log"
  ) &
done
wait
echo "all shards returned; concatenate with:"
echo "  cat $D/rows.shard*.jsonl > $D/rows.jsonl"
