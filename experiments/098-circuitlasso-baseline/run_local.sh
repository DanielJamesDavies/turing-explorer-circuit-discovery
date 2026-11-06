#!/usr/bin/env bash
# 098: CircuitLasso + the existing 083 ranked-member rows on one GPU, protocol v2, one process at a time.
#
# Default = the 8-target local check (093's v2 circuits as the WCM reference, as-run 062 records, like 093):
#   bash experiments/098-circuitlasso-baseline/run_local.sh
# The 190 sweep targets after the pod (references = the pod's v2 refit, records = the 094 v2 records):
#   REF_RUN=<pod TL_V2_OUT> CTX_DIR=<094 records dir> TARGETS=experiments/094-tl-v2-records/targets_tl8k_sweep190.txt \
#   OUT083=<the pod's dan159 dir, or a fresh one> STAGES=lasso,table bash experiments/098-circuitlasso-baseline/run_local.sh
#   (STAGES=ref,attr,fitted,lasso,table when the 083 rows were not run on the pod.)
# Resumable: every stage skips finished rows; re-run the same line after an interruption.
#
# Stages: ref    re-score the reference WCM circuits (083 rescore_ref.py; REF_ARMS)
#         attr   IG and AtP: top-n at the WCM sizes (RANKINGS) and the natural position union (083 attr_baselines.py)
#         fitted DAN-43 IG-ranked + activation-ranked members at the per-site counts, alpha = 1 and fitted (083)
#         lasso  CircuitLasso: rankings, top-n at the WCM sizes, natural (cross-validated) support, per-site + fitted
#         ge     Ge et al. hierarchical attribution, tau bisected to the W_0.001 size, every point scored (083 ge_hier.py;
#                not in the default STAGES: STAGES=ge,table)
#         table  083 table2.py (Table 2 + DAN-77 figure in OUT083) and 098 report.py (mini table, diagnostics)
set -u
cd "$(dirname "$0")/../.." || exit 1
E=experiments/098-circuitlasso-baseline
D83=experiments/083-v1-comparison
PY=${PY:-.venv/bin/python}
[ -n "${REF_RUN:-}" ] || { "$PY" -X utf8 "$E/make_ref093.py" || exit 1; }
export REF_RUN=${REF_RUN:-$E/out/ref093}
export CTX_DIR=${CTX_DIR:-experiments/062-h100-protocol-v1/out_full/out/ctx}
export TARGETS=${TARGETS:-$E/targets8.txt}
export OUT083=${OUT083:-$E/out/cmp}
export WCM_VERSION=${WCM_VERSION:-v2}
export REF_ARMS=${REF_ARMS:-W_all}
export ARMS=${ARMS:-abl_ig,atp} RANKINGS=${RANKINGS:-max,sfc}
export FIT_FROM=${FIT_FROM:-abl_ig:max,act:mean}
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
STAGES=${STAGES:-ref,attr,fitted,lasso,table}
mkdir -p "$OUT083/logs"
trap 'echo "[$(date +%H:%M:%S)] interrupted" | tee -a "$OUT083/queue.log"; kill 0' INT TERM
run() {   # run <name> <script>
  echo "[$(date +%H:%M:%S)] stage $1 (out $OUT083, ref $REF_RUN, ctx $CTX_DIR, WCM $WCM_VERSION)" | tee -a "$OUT083/queue.log"
  "$PY" -u -X utf8 "$2" >> "$OUT083/logs/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] stage $1 exit $?" | tee -a "$OUT083/queue.log"
}
for s in ${STAGES//,/ }; do
  case $s in
    ref)    run ref "$D83/rescore_ref.py" ;;
    attr)   run attr "$D83/attr_baselines.py" ;;
    fitted) run fitted "$D83/attr_fitted.py" ;;
    lasso)  run lasso "$E/lasso_rows.py" ;;
    ge)     run ge "$D83/ge_hier.py" ;;
    table)  REF=${REF:-local} "$PY" -X utf8 "$D83/table2.py" > "$OUT083/logs/table2.log" 2>&1
            "$PY" -X utf8 "$E/report.py" 2>&1 | tee "$OUT083/logs/report.log" ;;
    *)      echo "unknown stage: $s" >&2; exit 1 ;;
  esac
done
echo "[$(date +%H:%M:%S)] queue done" | tee -a "$OUT083/queue.log"
