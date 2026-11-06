#!/usr/bin/env bash
# 104: the GPU queue after part B, one job at a time: wait for run_cross.sh to exit -> part D TuringLLM timing panel
# (run_timing_tl.sh) -> part C chain smoke (PHASE=tl, then PHASE=all). Each step resumes; rerun this after a stop.
#   wsl (repo root): nohup bash experiments/104-v3-cross-model/run_queue.sh > experiments/104-v3-cross-model/logs/queue.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."
E=experiments/104-v3-cross-model
say() { echo "[104Q $(date '+%F %T')] $*" | tee -a "$E/logs/queue.events"; }
while pgrep -f "104-v3-cross-model/run_cross.sh" > /dev/null; do sleep 30; done
say "run_cross.sh not running; timing panel start"
bash $E/run_timing_tl.sh || { say "timing panel FAILED rc=$?"; exit 1; }
say "timing panel done; smoke PHASE=tl start"
PHASE=tl bash $E/run_smoke_chain.sh > $E/smoke_phase_tl.out 2>&1
say "smoke PHASE=tl rc=$?; PHASE=all start"
PHASE=all bash $E/run_smoke_chain.sh > $E/smoke_phase_all.out 2>&1
say "smoke PHASE=all rc=$?"
say "queue done"
