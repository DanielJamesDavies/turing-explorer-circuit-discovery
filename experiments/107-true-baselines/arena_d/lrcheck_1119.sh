#!/usr/bin/env bash
# Arena D lr check, night of 2026-11-19 (Daniel: revert arena D to v3's lr, confirm with a short local GPU check).
# The pod default is now DW_WCM=v3 (pod_stage_d.sh). This script fills lrtest's gaps through the PRODUCTION path
# (wcm_gemma.py run --wcm v3|v4), bf16, local unsloth weights, ~45-60 min on the 16 GB card:
#   1 res v3  gsres-gap6, rc, Caples data + mode, per_position, lam 1e-3 (= lrtest C0: reproduces 549 latents) and 3e-3
#   2 res v4  same, lam 3e-3            (lrtest has 1e-3 and 1e-2 only; v4 collapsed at 1e-2: Z 0.028)
#   3 plt v3  PLT, within_rc (not in lrtest), SFC data, pooled, lam 1e-4, 1e-3, 1e-2
#   4 plt v4  same
#   5 compare (CPU): lrcheck_compare.py -> out/lrcheck_1119/RESULT.md
# Pre-stated reading (lrtest's rule, README there): v4 "helps" only if it is smaller at equal-or-better Z / A / C / nec
# (within 0.02) on >= 2/3 of the new pairs AND loses no pass. Otherwise the revert to v3 is confirmed.
# GPU lock: waits until START (default 21:00) and until <repo>/.gpu-lock is absent, then writes `lrcheck-1119` into it.
# It never deletes anything: remove .gpu-lock with mcp__agent-gate__delete_path after it exits.
#   WSL, repo root:  bash experiments/107-true-baselines/arena_d/lrcheck_1119.sh > experiments/107-true-baselines/arena_d/out/lrcheck_1119.log 2>&1
set -u
cd "$(dirname "$0")/../../.." || exit 1
AD=experiments/107-true-baselines/arena_d
O=${LRC_OUT:-$AD/out/lrcheck_1119}
PY=${PY:-.venv/bin/python}
MEANS=$AD/out/caples_means/rc_gap6_bf16_SMOKE.pt
START=${START:-21:00}
HARD_END=${HARD_END:-07:30}     # no new step starts after this (next morning)
say() { echo "[lrcheck $(date +%H:%M:%S)] $*"; }
start_epoch=$(date -d "today $START" +%s)
end_epoch=$(date -d "tomorrow $HARD_END" +%s)
[ "$(date +%s)" -ge "$start_epoch" ] || say "waiting until $START"
while [ "$(date +%s)" -lt "$start_epoch" ]; do sleep 30; done
while [ -e .gpu-lock ]; do
  [ "$(date +%s)" -lt "$end_epoch" ] || { say "STOP: .gpu-lock never freed before $HARD_END"; exit 3; }
  sleep 30
done
echo lrcheck-1119 > .gpu-lock
say "took .gpu-lock"
m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' ')
[ -n "$m" ] && [ "$m" -le 3000 ] || { say "REFUSED: card holds '${m:-?}' MiB (> 3000 or no reading)"; exit 3; }
[ -s "$MEANS" ] || { say "REFUSED: no local means file $MEANS"; exit 3; }
[ -x "$PY" ] || { say "REFUSED: no $PY"; exit 3; }
mkdir -p "$O"
ENVV="env -u PYTHONPATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled TORCH_THREADS=4 \
OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GEMMA_HF_ID=unsloth/gemma-2-2b WCM_VERSION=v2"
FAILS=0
step() {   # step NAME ARGS...: one wcm_gemma.py run (one model load, several lambdas)
  local name=$1; shift
  [ "$(date +%s)" -lt "$end_epoch" ] || { say "SKIP $name: past $HARD_END"; FAILS=$((FAILS + 1)); return; }
  say "START $name"
  local t0; t0=$(date +%s)
  timeout 3600 $ENVV $PY -u $AD/wcm_gemma.py run "$@" --steps 400 --dtype bfloat16 --out "$O/$name" \
    > "$O/$name.log" 2>&1 < /dev/null || { say "$name FAILED (exit $?; $O/$name.log)"; FAILS=$((FAILS + 1)); }
  say "  $name: $(( $(date +%s) - t0 )) s; $(grep -E '\| task Z' "$O/$name.log" | tail -3 | tr '\n' ' ')"
}
RES="--family gemma-2-2b-gsres-gap6 --data caples --mode caples --means per_position --caples-means $MEANS --tasks rc"
PLT="--family gemma-2-2b-plt --data sfc --mode native --means pooled --tasks within_rc"
step res_v3 $RES --lams 1e-3,3e-3 --wcm v3
step res_v4 $RES --lams 3e-3 --wcm v4
step plt_v3 $PLT --lams 1e-4,1e-3,1e-2 --wcm v3
step plt_v4 $PLT --lams 1e-4,1e-3,1e-2 --wcm v4
say "5 CPU: compare"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= $PY -B $AD/lrcheck_compare.py "$O" || FAILS=$((FAILS + 1))
say "DONE ($FAILS failed steps). Remove .gpu-lock with mcp__agent-gate__delete_path."
[ "$FAILS" -eq 0 ]
