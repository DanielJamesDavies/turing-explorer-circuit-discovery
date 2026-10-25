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
#   GPUS=2,3,4,5,6,7 P=6 MODE=main ./...launch.sh   # only these GPUs (K = their count), e.g. beside data passes on 0/1
# Optional env (082 public runs; every default reproduces the 062 behaviour):
#   GPUS     comma list of GPU ids; shard i runs on GPU ${GPUS[i % K]} (default: GPUs 0..K-1)
#   LOGDIR   per-shard logs (default experiments/062-h100-protocol-v1/logs), so concurrent launches don't share files
#   SCRIPT   the per-shard program (default driver.py; any script that takes SHARD=i/N, e.g. a SHARD-aware companion)
#   MERGE=0  skip the closing merge.py call
# Progress:  tail -f experiments/062-h100-protocol-v1/logs/main.shard*.log
# Summary:   PYTHONPATH=src python experiments/062-h100-protocol-v1/merge.py

if [ -n "${GPUS:-}" ]; then
  IFS=',' read -r -a GPU_LIST <<< "$GPUS"
  if [ -n "${K:-}" ] && [ "$K" != "${#GPU_LIST[@]}" ]; then
    echo "launch.sh: K=$K but GPUS=$GPUS lists ${#GPU_LIST[@]} GPUs" >&2
    exit 2
  fi
  K=${#GPU_LIST[@]}
else
  K=${K:-8}           # GPUs
  GPU_LIST=()
  for g in $(seq 0 $((K - 1))); do GPU_LIST+=("$g"); done
fi
P=${P:-1}             # driver processes per GPU (small fits leave an H100 mostly idle; P=2 roughly doubles throughput)
MODE=${MODE:-main}
N=$((K * P))          # total shards; shard i runs on GPU ${GPU_LIST[i % K]}
# CPU threads per process: PyTorch defaults to ~one per core, so N processes oversubscribe the CPU N-fold (measured
# 2026-09-27 on 8xH100 / 224 vCPU: CPU pinned at 99% with GPUs idle). Default: an even share of the cores, capped at 8.
THREADS=${THREADS:-$(( $(nproc) / N ))}
[ "$THREADS" -gt 8 ] && THREADS=8
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
D=experiments/062-h100-protocol-v1
PY=${PY:-./.venv/bin/python}
LOGDIR=${LOGDIR:-$D/logs}
SCRIPT=${SCRIPT:-$D/driver.py}
mkdir -p "$LOGDIR"

run_mode() {
  local mode=$1
  for i in $(seq 0 $((N - 1))); do
    (
      tries=0
      until CUDA_VISIBLE_DEVICES=${GPU_LIST[$((i % K))]} SLOT=$((i / K)) SHARD=$i/$N MODE=$mode \
          PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          "$PY" -X utf8 "$SCRIPT" >> "$LOGDIR/$mode.shard$i.log" 2>&1; do
        tries=$((tries + 1))
        echo "shard $i exited nonzero (attempt $tries)" >> "$LOGDIR/$mode.shard$i.log"
        [ $tries -ge 5 ] && { echo "shard $i GIVING UP" >> "$LOGDIR/$mode.shard$i.log"; break; }
        sleep 30
      done
      echo "shard $i finished" >> "$LOGDIR/$mode.shard$i.log"
    ) &
    sleep ${STAGGER:-20}   # stagger start-up (model + store loading) so the shards don't all hit the disk at once;
    #                        STAGGER=5 is plenty on a big pod (8xH100 / 2 TB RAM): 48 shards up in 4 min, not 16
  done
  wait
  echo "$mode: all $N shards returned ($K GPUs [${GPU_LIST[*]}] x $P per GPU, $THREADS CPU threads each)"
}

if [ "$MODE" = "both" ]; then
  run_mode main
  run_mode sweep
else
  run_mode "$MODE"
fi
[ "${MERGE:-1}" = "0" ] && exit 0
if [ -n "${GPUS:-}" ]; then
  # merge.py reads GPUS as a GPU COUNT (its projection); hand it K, not the list
  GPUS=$K PYTHONPATH=src $PY -X utf8 $D/merge.py > "$LOGDIR/merge.log" 2>&1 && echo "summary: $D/out/summary.md"
else
  PYTHONPATH=src $PY -X utf8 $D/merge.py > "$LOGDIR/merge.log" 2>&1 && echo "summary: $D/out/summary.md"
fi
