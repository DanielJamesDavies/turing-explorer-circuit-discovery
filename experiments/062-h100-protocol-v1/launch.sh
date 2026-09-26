#!/bin/bash
# Multi-GPU launcher for the protocol-v1 runs: one driver process per GPU (SHARD=i/K), retry per shard (no set -e;
# a shard that dies is restarted and resumes where it stopped), one log per shard. Run from the repo root on the pod.
#
#   MODE=main  K=8 ./experiments/062-h100-protocol-v1/launch.sh      # stage 1: 1,416 targets
#   MODE=sweep K=8 ./experiments/062-h100-protocol-v1/launch.sh      # Figure 4 sweep: 191 targets x 10 arms
#   MODE=both  K=8 ./experiments/062-h100-protocol-v1/launch.sh      # main, then the sweep (reuses main's contexts
#                                                                    # and its WCM lambda 1e-3 circuits)
#   MODE=main TARGETS=experiments/062-h100-protocol-v1/targets_full.txt K=8 ./...launch.sh   # the full 15k
#   (TARGETS / OUT set on this command are inherited by every shard's driver)
# Progress:  tail -f experiments/062-h100-protocol-v1/logs/main.shard*.log
# Summary:   PYTHONPATH=src python experiments/062-h100-protocol-v1/merge.py

K=${K:-8}
MODE=${MODE:-main}
D=experiments/062-h100-protocol-v1
PY=${PY:-./.venv/bin/python}
mkdir -p "$D/logs"

run_mode() {
  local mode=$1
  for i in $(seq 0 $((K - 1))); do
    (
      tries=0
      until CUDA_VISIBLE_DEVICES=$i SHARD=$i/$K MODE=$mode \
          PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          $PY -X utf8 $D/driver.py >> "$D/logs/$mode.shard$i.log" 2>&1; do
        tries=$((tries + 1))
        echo "shard $i exited nonzero (attempt $tries)" >> "$D/logs/$mode.shard$i.log"
        [ $tries -ge 5 ] && { echo "shard $i GIVING UP" >> "$D/logs/$mode.shard$i.log"; break; }
        sleep 30
      done
      echo "shard $i finished" >> "$D/logs/$mode.shard$i.log"
    ) &
    sleep 20          # stagger start-up (model + store loading) so the shards don't all hit the disk at once
  done
  wait
  echo "$mode: all $K shards returned"
}

if [ "$MODE" = "both" ]; then
  run_mode main
  run_mode sweep
else
  run_mode "$MODE"
fi
PYTHONPATH=src $PY -X utf8 $D/merge.py > "$D/logs/merge.log" 2>&1 && echo "summary: $D/out/summary.md"
