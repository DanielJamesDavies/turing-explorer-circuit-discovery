#!/usr/bin/env bash
# DAN-159 / DAN-43 / DAN-77: the v1 method-comparison queue on ONE machine (README.md has the design and costs; the
# pod variant is run_pod.sh, which hands the same scripts to 062/launch.sh).
#
# Resumable and safe to interrupt: every stage skips the work already in its row / ranking / points files (cached per
# target and training-context hash; jsonl rows keyed per target, arm and context set), so running the script again
# continues where it stopped. Ctrl-C stops the whole queue (trap below), including the shard processes. Targets are
# sharded targets[i::K]; K processes share the GPU (K=1 is the safe default on 16 GB).
#
#   bash experiments/083-v1-comparison/run_local.sh                       # every stage, in order
#   STAGES=attr,fitted bash experiments/083-v1-comparison/run_local.sh     # some stages
#   CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx_v2 bash ...  # the rebuilt held-out sets (DAN-157):
#                                                                          # read by 062/driver.py, outputs in out_ctx_v2/
#   SMOKE=1 bash experiments/083-v1-comparison/run_local.sh                # 1 shallow target, tiny budgets, CPU,
#                                                                          # two sizes, out_smoke/
#
# Stages: ref    re-score the stored 062 WCM and unweighted circuits at every price (on these contexts / this GPU)
#         attr   C1: IG and AtP rankings, top-n at every WCM size and at the natural size, scored
#         fitted C3 (DAN-43) + activation-ranked members: ranked members at the production per-site counts, fitted
#         exact  exact activation patching on 10 stratified targets (ground truth for AtP / IG), then its summary
#         ge     C2: Ge hierarchical attribution, tau bisected to the W_0.001 size, every point scored
#         table  C4: the DAN-77 figure and Table 2 from the row files (CPU, seconds)
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-.venv/bin/python}
K=${K:-1}
STAGES=${STAGES:-ref,attr,fitted,exact,ge,table}
if [ "${SMOKE:-0}" = 1 ]; then
  export OUT083=${OUT083:-experiments/083-v1-comparison/out_smoke} TARGETS=experiments/083-v1-comparison/targets_smoke.txt
  export LIMIT=${LIMIT:-1} IG_STEPS=${IG_STEPS:-2} BISECT_STEPS=${BISECT_STEPS:-2} STEPS=${STEPS:-4}
  export SIZES=${SIZES:-W_0.001,W_0.004} REF_ARMS=${REF_ARMS:-W_0.001,U_0.001} N_EXACT=${N_EXACT:-1} CUDA_VISIBLE_DEVICES=
fi
if [ -z "${OUT083:-}" ]; then
  if [ -n "${CTX_DIR:-}" ]; then OUT083=experiments/083-v1-comparison/out_$(basename "$CTX_DIR")
  else OUT083=experiments/083-v1-comparison/out; fi
fi
export OUT083 PYTHONPATH=src
mkdir -p "$OUT083/logs"
trap 'echo "[$(date +%H:%M:%S)] interrupted" | tee -a "$OUT083/queue.log"; kill 0' INT TERM

stage() {   # stage <name> <script>: K shard processes in parallel, each logging to logs/<name>.shard<i>.log
  local name=$1 script=$2 i
  echo "[$(date +%H:%M:%S)] stage $name (K=$K, out $OUT083, ctx ${CTX_DIR:-as run})" | tee -a "$OUT083/queue.log"
  for ((i = 0; i < K; i++)); do
    SHARD=$i/$K "$PY" -X utf8 "experiments/083-v1-comparison/$script" >> "$OUT083/logs/$name.shard$i.log" 2>&1 &
  done
  wait
  echo "[$(date +%H:%M:%S)] stage $name done: $(cat "$OUT083/$name"/rows.shard*.jsonl 2>/dev/null | grep -c '"error"') \
error rows (retried on the next run)" | tee -a "$OUT083/queue.log"
}

for s in ${STAGES//,/ }; do
  case $s in
    ref)    stage ref rescore_ref.py ;;
    attr)   stage attr attr_baselines.py ;;
    fitted) stage fitted attr_fitted.py ;;
    exact)  stage exact exact_patching.py; REPORT=1 "$PY" -X utf8 experiments/083-v1-comparison/exact_patching.py ;;
    ge)     stage ge ge_hier.py ;;
    table)  "$PY" -X utf8 experiments/083-v1-comparison/table2.py 2>&1 | tee "$OUT083/logs/table2.log" ;;
    *)      echo "unknown stage: $s" >&2; exit 1 ;;
  esac
done
echo "[$(date +%H:%M:%S)] queue done" | tee -a "$OUT083/queue.log"
