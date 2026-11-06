#!/bin/bash
# Arena C (DAN-191 / DAN-196): run the official SFC code (third_party/feature-circuits@7fbd82b, untouched) on
# Pythia-70m-deduped: nodes-only discovery on each SVA train set, then SFC's own faithfulness evaluation
# (sfc_faithfulness.py, the notebook transcribed with our labelled edits). Run in WSL.
#
#   bash run_sfc.sh                         # all four tasks, all settings
#   DATASETS=rc SETTINGS=features THRESHOLDS=0.1 bash run_sfc.sh    # the F3 check only
#
# Overnight rules (Daniel, 2026-11-01): resumable (circuit.py loads an existing nodeall circuit; get_fcs caches by
# hash), and every step runs under a hard stop (STOP_AT, default 07:30; a step that would cross it is killed).
set -uo pipefail
ROOT="/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
FC="$ROOT/third_party/feature-circuits"
HERE="$ROOT/experiments/107-true-baselines/arena_c"
WORK=${WORK:-$HERE/out/sfc}
ENV=${SFC_ENV:-$HOME/venvs/sfc107}
PY="$ENV/bin/python"
DATASETS=${DATASETS:-rc,nounpp,simple,within_rc}
SETTINGS=${SETTINGS:-features,features_wo_errs,features_wo_some_errs,neurons}
THRESHOLDS=${THRESHOLDS:-}
BATCH=${BATCH:-10}
STOP_AT=${STOP_AT:-08:30}   # Daniel 2026-11-02: cap 08:30
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # OURS-5: torch 2.8 instead of < 2.6 (see setup_sfc_env.sh)
export PYTHONUNBUFFERED=1

# seconds left until the next STOP_AT (today if still ahead, else tomorrow)
left() {
  local now stop
  now=$(date +%s); stop=$(date -d "today $STOP_AT" +%s)
  [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $STOP_AT" +%s)
  echo $((stop - now))
}

mkdir -p "$WORK/circuits" "$WORK/faithfulness" "$WORK/logs"
cd "$WORK" || exit 1
ln -sfn "$FC/data" data
ln -sfn "$FC/annotations" annotations
"$PY" -c "import torch, nnsight; print('torch', torch.__version__, 'nnsight', nnsight.__version__)" | tee "$WORK/logs/env.txt"

need_neurons=0; [[ ",$SETTINGS," == *",neurons,"* ]] && need_neurons=1
for d in ${DATASETS//,/ }; do
  for neu in 0 $([ $need_neurons = 1 ] && echo 1); do
    tag="${d}_train$([ $neu = 1 ] && echo _neurons)"
    echo "[$(date +%T)] discovery $tag (nodes only, aggregation none, n 100, batch $BATCH; $(left) s to the stop)"
    /usr/bin/time -v timeout "$(left)" "$PY" "$FC/circuit.py" --model EleutherAI/pythia-70m-deduped \
      --dataset "${d}_train" --num_examples 100 --batch_size "$BATCH" --aggregation none --nodes_only \
      $([ $neu = 1 ] && echo --use_neurons) --circuit_dir circuits --device cuda:0 \
      > "$WORK/logs/discovery_$tag.log" 2>&1
    rc=$?
    grep -E 'Elapsed|Maximum resident' "$WORK/logs/discovery_$tag.log" | sed 's/^\s*/    /'
    if [ $rc = 124 ]; then echo "  hard stop reached during $tag"; exit 3; fi
    # circuit.py always plots after saving, and the per-position plotter cannot take edges=None (nodes only): it
    # raises in plot_circuit_posaligned. The circuit is saved before that (specs/sfc.md §8), so accept exactly this.
    pt="circuits/pythia-70m-deduped_${d}_train_n100_aggnone$([ $neu = 1 ] && echo _neurons)_nodeall.pt"
    if [ $rc != 0 ] && [ -s "$pt" ] && grep -q "in plot_circuit_posaligned" "$WORK/logs/discovery_$tag.log"; then
      echo "  $pt saved; the nodes-only plot failed as expected (harmless)"; rc=0
    fi
    [ $rc = 0 ] || { echo "  discovery $tag FAILED (rc $rc), see logs/discovery_$tag.log"; tail -20 "$WORK/logs/discovery_$tag.log"; exit 1; }
  done
done

echo "[$(date +%T)] evaluation: settings $SETTINGS, datasets $DATASETS${THRESHOLDS:+, thresholds $THRESHOLDS}"
timeout "$(left)" "$PY" "$HERE/sfc_faithfulness.py" --circuit_dir circuits --out faithfulness \
  --datasets "$DATASETS" --settings "$SETTINGS" ${THRESHOLDS:+--thresholds "$THRESHOLDS"} \
  2>&1 | tee "$WORK/logs/eval_$(date +%Y%m%d_%H%M%S).log"
rc=${PIPESTATUS[0]}
[ $rc = 124 ] && { echo "  hard stop reached during evaluation (finished datasets are cached)"; exit 3; }
exit $rc
