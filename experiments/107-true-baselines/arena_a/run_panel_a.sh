#!/usr/bin/env bash
# Arena A, 2026-11-05 night: the 12-target v4 panel (lambda 1e-3, 400 steps), reusing the pilot's data pass.
# ONE GPU process at a time, in order: targets (make_panel_targets.py) -> fit (driver_tcz.py, MODE=main) -> summary (CPU).
# Same env as run_pilot_a.sh. No new GPU step once date +%s >= STOP_EPOCH. Resumable (the driver skips done targets).
#   WSL, repo root: STOP_EPOCH=$(date -d 'today 06:30' +%s) bash experiments/107-true-baselines/arena_a/run_panel_a.sh
set -u
cd "$(dirname "$0")/../../.."
A=experiments/107-true-baselines/arena_a
P=$A/out/pilot
Q=$A/out/panel
PY=.venv/bin/python
STOP_EPOCH=${STOP_EPOCH:-99999999999}
mkdir -p "$Q/logs"
say() { echo "[panel_a] $(date +%H:%M:%S) $*"; }
late() { [ "$(date +%s)" -ge "$STOP_EPOCH" ]; }
ENVV="PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 ARENA_A_BOS=1 ARENA_A_SCOPE=a1 \
FAMILY=gpt2-small-tcz DATA=data_public/gpt2-small SAE_PRELOAD=0 DEEP_BS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
(
  while true; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' ')
    echo "$(date +%s),$(cat "$Q/.step" 2> /dev/null),${m:-NA}" >> "$Q/gpu_mem.csv"
    sleep 10
  done
) &
SAMPLER=$!
trap 'kill $SAMPLER 2> /dev/null' EXIT
run() {   # run NAME command...
  local name=$1; shift
  if late; then say "skip $name (STOP_EPOCH reached)"; return 1; fi
  say "start $name"; echo "$name" > "$Q/.step"
  local t0=$(date +%s)
  env $ENVV "$@" > "$Q/logs/$name.log" 2>&1
  local rc=$?
  printf "%s\t%s\t%s\t%s\n" "$name" "$t0" "$(date +%s)" "$rc" >> "$Q/steps.tsv"
  echo "idle" > "$Q/.step"
  say "end $name (exit $rc, $(( $(date +%s) - t0 )) s)"
  return $rc
}
if [ "$(cat $Q/targets_panel.txt 2> /dev/null | grep -c .)" -lt 12 ]; then
  run targets env STORES=$P/datapass OUT=$Q/rec TARGETS_OUT=$Q/targets_panel.txt $PY -X utf8 -u $A/make_panel_targets.py \
    || { say "targets failed"; exit 1; }
fi
run fit_l1e-3 env STORES=$P/datapass OUT=$Q/wcm_l1e-3 CTX_DIR=$Q/rec/ctx TARGETS=$Q/targets_panel.txt MODE=main LAM=1e-3 \
  WCM_VERSION=v4 ORDER=list SKIP_SPEC=1 CURVE_STRIDE=20 $PY -X utf8 -u $A/driver_tcz.py
CUDA_VISIBLE_DEVICES= $PY -B $A/pilot_summary.py "$Q" > "$Q/logs/summary.log" 2>&1 || say "FAILED summary"
say "DONE"
