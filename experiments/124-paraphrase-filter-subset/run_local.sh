#!/bin/bash
# 124 (DAN-217) on the LOCAL GPU (WSL, RTX 5070 Ti, one process): build (GPU) -> check -> prep -> score orig and para
# with the production driver (GPU) -> verify -> report. ~1.2 h for the 300-target subset (out/time_estimate.json).
# Every step resumes: rerun the same command after a kill. Nothing here refits a circuit (verify is the guard).
#
# GPU lock: <repo>/.gpu-lock must exist and hold exactly $JOB (default dan217-124) before this starts; the script
# refuses otherwise and never deletes the lock (remove it with agent-gate delete_path when the run is done).
#
#   wsl: cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && \
#        bash experiments/124-paraphrase-filter-subset/run_local.sh 2>&1 | tee experiments/124-paraphrase-filter-subset/out/run_local.log
# ENV  STEPS_ONLY [build,check,prep,score,verify,report]  VARIANTS [orig,para]  JOB [dan217-124]  PY [.venv/bin/python]
#      DRY_RUN=1 print the commands only
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO" || exit 1
JOB=${JOB:-dan217-124}
PY=${PY:-.venv/bin/python}
E=experiments/124-paraphrase-filter-subset
W=$E/out
STEPS_ONLY=${STEPS_ONLY:-build,check,prep,score,verify,report}
VARIANTS=${VARIANTS:-orig,para}
DRY=${DRY_RUN:-0}
CTX_ORIG=experiments/109-pod-run-analysis/out/archives/pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh
say() { echo "[run124 $(date +%H:%M:%S)] $*"; }
run() { say "\$ $*"; [ "$DRY" = 1 ] && return 0; "$@"; }
want() { [[ ",$STEPS_ONLY," == *",$1,"* ]]; }
lock_ok() {
  [ "$DRY" = 1 ] && return 0
  [ -f "$REPO/.gpu-lock" ] && [ "$(tr -d '[:space:]' < "$REPO/.gpu-lock")" = "$JOB" ] && return 0
  say "GPU LOCK: $REPO/.gpu-lock must hold '$JOB'; it holds '$(cat "$REPO/.gpu-lock" 2> /dev/null || echo '<no file>')'. Not starting."
  return 1
}
[ -f "$W/idf114.npz" ] || { say "missing $W/idf114.npz (run rebuild124.py STAGE=idf on the CPU first)"; exit 1; }
[ -s "$E/targets_subset.txt" ] || { say "missing $E/targets_subset.txt (select_subset.py)"; exit 1; }
lock_ok || exit 2
say "job $JOB | subset $(wc -l < "$E/targets_subset.txt") targets | variants $VARIANTS | steps $STEPS_ONLY"

want build  && { lock_ok && run env STAGE=build DEVICE=cuda PYTHONPATH=src "$PY" -X utf8 $E/rebuild124.py || exit 1; }
want check  && { run env STAGE=check DEVICE=cuda PYTHONPATH=src "$PY" -X utf8 $E/rebuild124.py || exit 1; }
want prep   && { run env STAGE=prep PYTHONPATH=src "$PY" -X utf8 $E/rescore124.py || exit 1; }
if want score; then
  for v in ${VARIANTS//,/ }; do
    if [ "$v" = orig ]; then ctx=$CTX_ORIG; else ctx=$W/ctx_$v; fi
    lock_ok || exit 2
    say "score $v (CTX_DIR $ctx)"
    run env MODE=main ORDER=list SKIP_SPEC=1 WCM_VERSION=v4 OUT=$W/rs_$v CTX_DIR="$ctx" TARGETS=$W/targets_$v.txt \
      PYTHONPATH=src "$PY" -X utf8 experiments/062-h100-protocol-v1/driver.py 2>&1 | tee -a "$W/rs_$v.log"
    [ "${PIPESTATUS[0]}" = 0 ] || [ "$DRY" = 1 ] || { say "driver failed on $v (rerun resumes)"; exit 1; }
  done
fi
want verify && { run env STAGE=verify "$PY" -X utf8 $E/rescore124.py || { say "VERIFY FAILED"; exit 1; }; }
want report && run env PYTHONPATH=src "$PY" -X utf8 $E/report124.py
say "DONE. Now remove the GPU lock: agent-gate delete_path <repo>/.gpu-lock"
