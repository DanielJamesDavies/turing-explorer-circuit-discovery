#!/usr/bin/env bash
# Arena A, 2026-11-05 night, after run_panel_a.sh: the necessity diagnosis. ONE GPU process at a time, in order:
#   diag      diag_necessity_tcz.py on the 18 circuits (pilot lambda 1e-3 + 4e-3, panel lambda 1e-3), forward only
#   arm fits  three DIAGNOSTIC arms (not settings changes) on the 3 pilot targets, v4 otherwise unchanged:
#               necw1    NEC_W=1 (the learned-mask necessity term, target 0.95): is necessity reachable at all?
#               l2.5e-4  LAM=2.5e-4: larger circuits (does necessity grow with coverage?)
#               s1200    STEPS=1200: three times the training budget (is the fit under-trained?)
#   diag_arms diag_necessity_tcz.py on the arm circuits
# No new GPU step once date +%s >= STOP_EPOCH. Then touches out/panel/GPU_DONE.
#   WSL, repo root: STOP_EPOCH=$(date -d 'today 06:30' +%s) bash experiments/107-true-baselines/arena_a/run_necdiag_a.sh
set -u
cd "$(dirname "$0")/../../.."
A=experiments/107-true-baselines/arena_a
P=$A/out/pilot
Q=$A/out/panel
PY=.venv/bin/python
STOP_EPOCH=${STOP_EPOCH:-99999999999}
mkdir -p "$Q/logs"
say() { echo "[necdiag_a] $(date +%H:%M:%S) $*"; }
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
# panel rerun: 3 of 12 panel fits died at 02:07 (card at 15.96 GB: "CUDA driver error: device not ready", then
# allocator asserts) after 9 fits in one process; each is refit in a fresh process of its own (same settings) in
# OUT wcm_l1e-3_rerun (the driver never retries a target with a status row, so the error rows stay in wcm_l1e-3)
for K in 9.z.19003 9.z.17708 11.tc.22949; do
  run rerun_$K env STORES=$P/datapass OUT=$Q/wcm_l1e-3_rerun CTX_DIR=$Q/rec/ctx TARGETS=$Q/rerun_$K.txt MODE=main \
    LAM=1e-3 WCM_VERSION=v4 ORDER=list SKIP_SPEC=1 CURVE_STRIDE=20 $PY -X utf8 -u $A/driver_tcz.py
done
DIRS0="pilot_l1e-3=$P/wcm_l1e-3:$P/rec/ctx,pilot_l4e-3=$P/wcm_l4e-3:$P/rec/ctx,panel_l1e-3=$Q/wcm_l1e-3:$Q/rec/ctx,\
panel_l1e-3_rerun=$Q/wcm_l1e-3_rerun:$Q/rec/ctx"
run diag env STORES=$P/datapass OUT=$Q/diag_nec DIRS="$DIRS0" WCM_VERSION=v4 $PY -X utf8 -u $A/diag_necessity_tcz.py
FIT="STORES=$P/datapass CTX_DIR=$P/rec/ctx TARGETS=$P/targets_pilot.txt MODE=main WCM_VERSION=v4 ORDER=list SKIP_SPEC=1 \
CURVE_STRIDE=20"
run arm_necw1 env $FIT OUT=$Q/arm_necw1 LAM=1e-3 NEC_W=1 $PY -X utf8 -u $A/driver_tcz.py
run arm_l2.5e-4 env $FIT OUT=$Q/arm_l2.5e-4 LAM=2.5e-4 $PY -X utf8 -u $A/driver_tcz.py
run arm_s1200 env $FIT OUT=$Q/arm_s1200 LAM=1e-3 STEPS=1200 $PY -X utf8 -u $A/driver_tcz.py
DIRS1="arm_necw1=$Q/arm_necw1:$P/rec/ctx,arm_l2.5e-4=$Q/arm_l2.5e-4:$P/rec/ctx,arm_s1200=$Q/arm_s1200:$P/rec/ctx"
run diag_arms env STORES=$P/datapass OUT=$Q/diag_nec DIRS="$DIRS1" WCM_VERSION=v4 $PY -X utf8 -u $A/diag_necessity_tcz.py
touch "$Q/GPU_DONE"
say "DONE (GPU_DONE written)"
