#!/usr/bin/env bash
# DAN-159 on the pod (TuringLLM track, H100, the same GPU type as the 062 circuits): each 083 stage is handed to
# 062/launch.sh as SCRIPT=, which runs N = K x P shard processes (SHARD=i/N, CUDA_VISIBLE_DEVICES per shard, retries,
# staggered start-up, CPU thread caps) and waits for all of them. Every 083 script is SHARD-aware and resumable, so a
# killed shard is restarted by launch.sh and continues where it stopped. PROPOSAL: on a rented machine every command is
# agreed with Daniel first (remote-pod rule).
#
#   K=8 P=4 bash experiments/083-v1-comparison/run_pod.sh
#   CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx_v2 K=8 P=4 bash experiments/083-v1-comparison/run_pod.sh
#   STAGES=attr,fitted ... (same stage names as run_local.sh)
#   GPUS=2,3,4,5,6,7 ...   (only these GPUs; K = their count. The 082 chain's TuringLLM stage T5 calls this once per
#                           stage with GPUS = its fit pool, CTX_DIR = the rebuilt ctx_v2 and its own OUT083)
#
# Needs on the pod: the repo at the local HEAD, models / SAEs / outputs/ stores (062 bootstrap), the 062 run's
# out_full/out/{ctx or CTX_DIR, sweep/circuits, sweep/eval*} and out_{bands,depth,lowlam}/sweep (weak-price circuits).
set -u
cd "$(dirname "$0")/../.." || exit 1
D=experiments/083-v1-comparison
if [ -n "${GPUS:-}" ]; then K=$(echo "$GPUS" | tr ',' '\n' | grep -c .); fi   # 082 chain: GPUS = the fit pool
export K=${K:-8} P=${P:-4} STAGGER=${STAGGER:-5} MERGE=0
STAGES=${STAGES:-ref,attr,fitted,exact,ge,table}
if [ -z "${OUT083:-}" ]; then
  if [ -n "${CTX_DIR:-}" ]; then OUT083=$D/out_$(basename "$CTX_DIR"); else OUT083=$D/out; fi
fi
export OUT083
mkdir -p "$OUT083/logs"
for s in ${STAGES//,/ }; do
  case $s in
    ref)    script=rescore_ref.py ;;
    attr)   script=attr_baselines.py ;;
    fitted) script=attr_fitted.py ;;
    exact)  # 10 targets: one shard per GPU, then the summary
            echo "[$(date +%H:%M:%S)] stage exact: K=$K x 1 shards" | tee -a "$OUT083/queue.log"
            P=1 MODE=exact SCRIPT=$D/exact_patching.py LOGDIR=$OUT083/logs ./experiments/062-h100-protocol-v1/launch.sh
            REPORT=1 PYTHONPATH=src "${PY:-./.venv/bin/python}" -X utf8 $D/exact_patching.py; continue ;;
    ge)     script=ge_hier.py ;;
    table)  PYTHONPATH=src "${PY:-./.venv/bin/python}" -X utf8 $D/table2.py 2>&1 | tee "$OUT083/logs/table2.log"; continue ;;
    *)      echo "unknown stage: $s" >&2; exit 1 ;;
  esac
  echo "[$(date +%H:%M:%S)] stage $s: K=$K x P=$P shards" | tee -a "$OUT083/queue.log"
  # launch.sh names its logs <MODE>.shard<i>.log in LOGDIR; MODE is unused by the 083 scripts
  MODE=$s SCRIPT=$D/$script LOGDIR=$OUT083/logs ./experiments/062-h100-protocol-v1/launch.sh
  echo "[$(date +%H:%M:%S)] stage $s done" | tee -a "$OUT083/queue.log"
done
