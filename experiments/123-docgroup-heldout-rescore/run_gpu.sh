#!/bin/bash
# 123 (DAN-217) end to end on a GPU machine: plan (CPU) -> TuringLLM build (GPU) -> check (CPU) -> prep (CPU) ->
# re-score every model x variant with the production driver (GPU) -> verify + report (CPU). Every step resumes: rerun
# the same command after a kill. Nothing here refits a circuit (rescore.py verify is the guard).
#
#   pod:   TL_ROOT=/root/turing-tl PUB_ROOT=/root/turing GPUS=0,1,2,3,4 P=6 \
#            bash /root/turing-tl/experiments/123-docgroup-heldout-rescore/run_gpu.sh 2>&1 | tee /root/run123.log
#   local (WSL, one GPU; TuringLLM only, archives as unpacked by 109 unpack.sh):
#          MODELS=turingllm GPUS=0 P=1 bash experiments/123-docgroup-heldout-rescore/run_gpu.sh
#
# ENV  TL_ROOT   TuringLLM checkout (its config.yaml + outputs/ + data/; 094 / 082 / 085 artefacts as the chain left
#                them). Unset = this repo, reading the 109 archives (common123 defaults)
#      PUB_ROOT  public checkout (082 out_<model>, data_public/, outputs_public/). Unset = this repo
#      MODELS    [turingllm,pythia-70m,gpt2-small]   VARIANTS [all: orig,doc,topic for TuringLLM; orig,doc public]
#      GPUS [0]  P [1] processes per GPU (production T3 ran 10 per H100 under MPS; MPS=1 starts the daemon)
#      BUILD_P [2] build processes per GPU   STEPS_ONLY [all] e.g. "plan,build,check" or "score,report"
#      WORK123   [<TL_ROOT>/experiments/123-docgroup-heldout-rescore/out] shared out/ (records, OUTs, logs)
#      DRY_RUN=1 print the commands only
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
TL_ROOT=${TL_ROOT:-$REPO}
PUB_ROOT=${PUB_ROOT:-$REPO}
export TL_ROOT PUB_ROOT
[ "$TL_ROOT" = "$REPO" ] && [ "$PUB_ROOT" = "$REPO" ] && unset TL_ROOT PUB_ROOT     # local: common123 archive defaults
TLR=${TL_ROOT:-$REPO}; PBR=${PUB_ROOT:-$REPO}
export WORK123=${WORK123:-$TLR/experiments/123-docgroup-heldout-rescore/out}
W123=$WORK123; export W123
MODELS=${MODELS:-turingllm,pythia-70m,gpt2-small}; export MODELS
GPUS=${GPUS:-0}; P=${P:-1}; BUILD_P=${BUILD_P:-2}
STEPS_ONLY=${STEPS_ONLY:-plan,build,check,prep,score,verify,report}
DRY=${DRY_RUN:-0}
TL_PY=${TL_PY:-$TLR/.venv/bin/python}; [ -x "$TL_PY" ] || TL_PY=python
E123=experiments/123-docgroup-heldout-rescore
say() { echo "[run123 $(date +%H:%M:%S)] $*"; }
run() { say "\$ $*"; [ "$DRY" = 1 ] && return 0; "$@"; }
want() { [[ ",$STEPS_ONLY," == *",$1,"* ]]; }
has() { [[ ",$MODELS," == *",$1,"* ]]; }

# the production records each 'orig' variant re-scores (the driver's CTX_DIR)
if [ -n "${TL_ROOT:-}" ]; then export CTX_ORIG_TURINGLLM=$TL_ROOT/experiments/082-public-runs/out_turingllm/ctx_fresh
else export CTX_ORIG_TURINGLLM=$REPO/experiments/109-pod-run-analysis/out/archives/pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh; fi
if [ -n "${PUB_ROOT:-}" ]; then
  export CTX_ORIG_PYTHIA_70M=$PUB_ROOT/experiments/082-public-runs/out_pythia-70m/ctx
  export CTX_ORIG_GPT2_SMALL=$PUB_ROOT/experiments/082-public-runs/out_gpt2-small/ctx
else
  export CTX_ORIG_PYTHIA_70M=$REPO/experiments/109-pod-run-analysis/out/archives/pub-pythia-70m-S10-results/experiments/082-public-runs/out_pythia-70m/ctx
  export CTX_ORIG_GPT2_SMALL=$REPO/experiments/109-pod-run-analysis/out/archives/pub-gpt2-small-S10-results/experiments/082-public-runs/out_gpt2-small/ctx
fi
[ "$DRY" = 1 ] || mkdir -p "$W123/logs" || exit 1
say "TL_ROOT=${TL_ROOT:-<repo, archives>} PUB_ROOT=${PUB_ROOT:-<repo, archives>} WORK123=$W123 MODELS=$MODELS GPUS=$GPUS P=$P"

if [ "${MPS:-0}" = 1 ] && command -v nvidia-cuda-mps-control > /dev/null && ! pgrep -f nvidia-cuda-mps-control > /dev/null; then
  run nvidia-cuda-mps-control -d
fi

cd "$TLR" || { [ "$DRY" = 1 ] || exit 1; }
want plan  && { run env STAGE=plan PYTHONPATH=src "$TL_PY" -X utf8 $E123/rebuild.py || exit 1; }
if want build && has turingllm; then
  run env GPUS="$GPUS" P="$BUILD_P" MERGE=0 STAGGER="${STAGGER:-10}" LOGDIR="$W123/logs/build" PY="$TL_PY" \
    SCRIPT=$E123/rebuild.py STAGE=build MODELS=turingllm bash experiments/062-h100-protocol-v1/launch.sh || exit 1
fi
want check && { run env STAGE=check PYTHONPATH=src "$TL_PY" -X utf8 $E123/rebuild.py || exit 1; }
want prep  && { run env STAGE=prep PYTHONPATH=src "$TL_PY" -X utf8 $E123/rescore.py || exit 1; }

if want score; then
  for m in ${MODELS//,/ }; do
    if [ "$m" = turingllm ]; then root=$TLR; vs=${VARIANTS:-orig,doc,topic}; else root=$PBR; vs=${VARIANTS:-orig,doc}; fi
    for v in ${vs//,/ }; do
      [ "$m" != turingllm ] && [ "$v" = topic ] && continue
      ctxv="CTX_ORIG_$(echo "$m" | tr 'a-z-' 'A-Z_')"
      if [ "$v" = orig ]; then ctx=${!ctxv}; else ctx=$W123/$m/ctx_$v; fi
      extra=()
      if [ "$m" != turingllm ]; then
        # the production data pass only (this repo's own data_public/<model> is the 069 smoke corpus): on a pod it is
        # restored from pub-<model>-S10-stores.tgz (scripts/pod_restore.sh)
        man="$root/data_public/$m/corpus_manifest.json"
        if [ "$DRY" != 1 ] && { [ ! -f "$man" ] || grep -qi smoke "$man"; }; then
          say "SKIP score $m: $man is missing or a smoke corpus (restore pub-$m-S10-stores.tgz into PUB_ROOT)"; continue
        fi
        extra=(FAMILY="$m" DATA="$root/data_public/$m" STORES="$root/outputs_public/$m" SAE_PRELOAD=0)
      fi
      say "score $m $v (CTX_DIR $ctx)"
      (cd "$root" 2> /dev/null || [ "$DRY" = 1 ] || exit 1; run env GPUS="$GPUS" P="$P" MODE=main ORDER=interleaved SKIP_SPEC=1 WCM_VERSION=v4 MERGE=0 \
          STAGGER="${STAGGER:-10}" OUT="$W123/$m/rs_$v" TARGETS="$W123/$m/targets_$v.txt" CTX_DIR="$ctx" \
          LOGDIR="$W123/logs/$m/rs_$v" "${extra[@]}" bash experiments/062-h100-protocol-v1/launch.sh) || exit 1
    done
  done
fi
want verify && run env STAGE=verify "$TL_PY" -X utf8 $E123/rescore.py
want report && run env PYTHONPATH=src "$TL_PY" -X utf8 $E123/report.py
say "DONE"
