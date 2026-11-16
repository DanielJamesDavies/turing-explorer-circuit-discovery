#!/usr/bin/env bash
# 121 GPU chain (run inside WSL from the repo root). One GPU process at a time: the jobs run strictly in sequence.
# Order = value: TuringLLM arm (a) first (two jobs of <= 75 min), then Pythia-70m, then GPT-2 small.
# Every job is resumable per target and stops starting new targets at STOP_AT (22:45), so all GPU work ends by 23:00.
# The caller takes / releases .gpu-lock around this chain.
set -u
E=experiments/121-error-zeroed-refit-scale
LOG=$E/out/logs
mkdir -p "$LOG"
STOP_AT=${STOP_AT:-22:45}
job() {   # job NAME FAMILY MAX_MIN
  local name=$1 fam=$2 mx=$3
  echo "[$(date +%H:%M:%S)] start $name ($fam, MAX_MIN=$mx)" >> "$LOG/chain.log"
  FAMILY=$fam MAX_MIN=$mx STOP_AT=$STOP_AT PYTHONPATH=src .venv/bin/python -X utf8 $E/refit_scale.py \
    >> "$LOG/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] end $name rc=$?" >> "$LOG/chain.log"
}
job tl1 turingllm 75
job tl2 turingllm 75
job tl3 turingllm 40
job pythia pythia-70m 75
job gpt2 gpt2-small 75
echo "[$(date +%H:%M:%S)] CHAIN_DONE" >> "$LOG/chain.log"
