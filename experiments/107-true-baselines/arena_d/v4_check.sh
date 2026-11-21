#!/usr/bin/env bash
# Arena D v4 quick check (D7, 2026-11-16; V4_CHECK.md): two tiny Gemma-2-2B task-circuit fits through the PRODUCTION
# path (wcm_gemma.py run --wcm v4, as pod_stage_d.sh DW_WCM=v4 runs it), bf16 on the local unsloth weights, one GPU
# process at a time, ~15 min on the 16 GB card:
#   1 res  gemma-2-2b-gsres-gap6, rc, Caples data + Caples mode per_position (local bf16 means file), lam 1e-3
#   2 plt  gemma-2-2b-plt, rc, SFC data, pooled, lam 1e-3
# = lrtest's C1 units of the same family / task / lambda (C1 = v3 + v4's lr budget through a wrapper), so the check is:
# does the production v4 path give C1's circuits (provenance exact, numbers equal up to GPU nondeterminism)?
# GPU-lock protocol: whoever runs this first writes the single line `arena-d-v4` into <repo>/.gpu-lock (Write tool),
# and removes it with mcp__agent-gate__delete_path when this script has exited. This script refuses to start unless
# .gpu-lock holds exactly `arena-d-v4`, and refuses a card already holding > 3000 MiB. It never deletes anything.
#   WSL, repo root:  bash experiments/107-true-baselines/arena_d/v4_check.sh 2>&1 | tee experiments/107-true-baselines/arena_d/out/v4_check.log
set -u
cd "$(dirname "$0")/../../.." || exit 1
AD=experiments/107-true-baselines/arena_d
O=${V4C_OUT:-$AD/out/v4_check}
PY=${PY:-.venv/bin/python}
MEANS=$AD/out/caples_means/rc_gap6_bf16_SMOKE.pt
say() { echo "[v4_check $(date +%H:%M:%S)] $*"; }
lock=$(cat .gpu-lock 2> /dev/null)
[ "$lock" = arena-d-v4 ] || { say "REFUSED: .gpu-lock holds '${lock:-<no file>}', not 'arena-d-v4' (write it first)"; exit 3; }
m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' ')
[ -n "$m" ] || { say "REFUSED: no nvidia-smi reading"; exit 3; }
[ "$m" -le 3000 ] || { say "REFUSED: the card holds $m MiB (> 3000): another process"; exit 3; }
[ -s "$MEANS" ] || { say "REFUSED: no local means file $MEANS"; exit 3; }
[ -x "$PY" ] || { say "REFUSED: no $PY"; exit 3; }
mkdir -p "$O"
ENVV="env -u PYTHONPATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled TORCH_THREADS=4 \
OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GEMMA_HF_ID=unsloth/gemma-2-2b WCM_VERSION=v2"
FAILS=0
say "0 CPU: the v4 unit test (tiny model)"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 timeout 600 $PY -m pytest \
  $AD/test_wcm_gemma.py -q -p no:cacheprovider -k v4_lr_budget > "$O/unit.log" 2>&1 < /dev/null \
  || { say "unit test FAILED ($O/unit.log)"; FAILS=$((FAILS + 1)); }
say "  $(grep -E 'passed|failed' "$O/unit.log" | tail -1)"
say "1 GPU: res (gsres, Caples mode, rc, lam 1e-3, 400 steps, bf16, --wcm v4)"
t0=$(date +%s)
timeout 1200 $ENVV $PY -u $AD/wcm_gemma.py run --family gemma-2-2b-gsres-gap6 --data caples --mode caples \
  --means per_position --caples-means "$MEANS" --tasks rc --lams 1e-3 --steps 400 --dtype bfloat16 --wcm v4 \
  --out "$O/res" > "$O/res.log" 2>&1 < /dev/null || { say "res FAILED (exit $?; $O/res.log)"; FAILS=$((FAILS + 1)); }
say "  res: $(( $(date +%s) - t0 )) s; $(grep -E 'lam 0.001' "$O/res.log" | tail -1)"
say "2 GPU: plt (PLT, SFC data, pooled, rc, lam 1e-3, 400 steps, bf16, --wcm v4)"
t0=$(date +%s)
timeout 1200 $ENVV $PY -u $AD/wcm_gemma.py run --family gemma-2-2b-plt --data sfc --mode native --means pooled \
  --tasks rc --lams 1e-3 --steps 400 --dtype bfloat16 --wcm v4 \
  --out "$O/plt" > "$O/plt.log" 2>&1 < /dev/null || { say "plt FAILED (exit $?; $O/plt.log)"; FAILS=$((FAILS + 1)); }
say "  plt: $(( $(date +%s) - t0 )) s; $(grep -E 'lam 0.001' "$O/plt.log" | tail -1)"
say "3 CPU: compare with lrtest C1 / C0"
env -u PYTHONPATH CUDA_VISIBLE_DEVICES= $PY -B $AD/v4_check_compare.py "$O" || FAILS=$((FAILS + 1))
say "DONE ($FAILS failed steps). Now remove .gpu-lock with mcp__agent-gate__delete_path."
[ "$FAILS" -eq 0 ]
