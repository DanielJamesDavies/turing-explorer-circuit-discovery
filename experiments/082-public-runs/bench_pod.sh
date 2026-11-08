#!/bin/bash
# QUICK H100 TIMING BENCH (2026-11-07): measure the costs the full-run estimate guesses at, on a small (2-GPU) pod,
# before renting the 6x / 8x pod. Uses the production scripts unchanged; nothing is written to /workspace.
#
#   after pod_bootstrap_public.sh (and arena A's P0 / P1 setup commands for part 2):
#   cd /root/turing && nohup bash experiments/082-public-runs/bench_pod.sh > /root/bench.out 2>&1 &
#
# Part 1  GPT-2 public track, S1-S8 only: a 524,288-window corpus (= arena A's corpus size, so part 2 reuses it),
#         the data pass on GPU 0 (-> windows/s, scaled to the 4M production corpus), records + frame, then the
#         lambda pilot at ONE lambda (W_0.001: 30 production v4 fits, P=6 under MPS, on both GPUs once S3 is done)
#         -> GPU-s per v4 fit on an H100 (the public tracks' "local -> H100 ratio r", the estimate's big unknown).
# Part 2  arena A run_pod.sh P0-P5 on 4 targets, one lambda (1e-3): its data pass, Ge + Dunefsky on the pod's CPUs /
#         GPUs, WCM a1 + a2 fits -> per-unit times (steps.tsv / cmds.tsv). Hard stop BENCH_AA_MAX_H.
# Part 3  pack every log to /root/bench-logs.tgz (send it back with runpodctl send).
#
# ENV: BENCH_PART [all | 1 | 2]  BENCH_AA_LIMIT [4]  BENCH_AA_MAX_H [2]  BENCH_GPUS [0,1]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
ROOT=$(pwd)
PART=${BENCH_PART:-all}
GPUS=${BENCH_GPUS:-0,1}
B=${BENCH_DIR:-/root/bench}
TGZ=$B-logs.tgz
mkdir -p "$B" || exit 1
say() { echo "[bench $(date +%H:%M:%S)] $*" | tee -a "$B/bench.events"; }
say "start | $(git log -1 --format='%h %s') | GPUs $GPUS | $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr -s ' ')"
nproc > "$B/nproc.txt"; free -g > "$B/free.txt"; nvidia-smi > "$B/nvidia-smi.txt"

if [ "$PART" = all ] || [ "$PART" = 1 ]; then
  say "part 1: GPT-2 S1-S8 (N_SEQ 524288, pilot arm W_0.001)"
  t0=$(date +%s)
  MODELS=gpt2-small N_SEQ=524288 DATA_GPU_gpt2_small="${GPUS%%,*}" FIT_GPUS="${GPUS#*,}" \
    PILOT_ARMS_gpt2_small=W_0.001 STOP_AFTER_gpt2_small=S8 CKPT=$B/ckpt RESULTS=$B/results \
    bash experiments/082-public-runs/chain.sh > "$B/chain.out" 2>&1
  say "part 1 done: rc $? in $(( ($(date +%s) - t0) / 60 )) min"
fi

if [ "$PART" = all ] || [ "$PART" = 2 ]; then
  say "part 2: arena A P0-P5, $((${BENCH_AA_LIMIT:-4})) targets, lambda 1e-3"
  t0=$(date +%s)
  AA_GPUS="$GPUS" AA_LIMIT="${BENCH_AA_LIMIT:-4}" AA_LAMBDAS=1e-3 STOP_AT=P5 AA_PACK=0 AA_MAX_H="${BENCH_AA_MAX_H:-2}" \
    AA_WORK=experiments/107-true-baselines/arena_a/out/bench CKPT=$B/ckpt RESULTS=$B/results \
    bash experiments/107-true-baselines/arena_a/run_pod.sh > "$B/arena_a.out" 2>&1
  say "part 2 done: rc $? in $(( ($(date +%s) - t0) / 60 )) min"
fi

say "pack logs -> $TGZ"
tar -czf "$TGZ" --ignore-failed-read --exclude='*.pt' --exclude='*.npy' "$B" \
  experiments/082-public-runs/logs experiments/082-public-runs/out_gpt2-small/pilot_lambda.json \
  experiments/082-public-runs/out_gpt2-small/sweep experiments/107-true-baselines/arena_a/out/bench 2> /dev/null
ls -la "$TGZ"
say "done"
