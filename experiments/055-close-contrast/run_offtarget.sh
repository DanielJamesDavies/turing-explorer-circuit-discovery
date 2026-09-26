#!/usr/bin/env bash
# Off-target term (056) on the 16 shared targets, close contrast contexts, gamma_C = gamma_A = 0.25, lambda 1e-3.
#   otall3   gamma_S 1e-3, mode all  (every lift above max(clean, empty))
#   otcut3   gamma_S 1e-3, mode cut  (only lifts above the clean Top-K cut)
#   otcut2   gamma_S 1e-2, mode cut
# Each arm: refit, the v3 eval on close contrast contexts (the headline battery), then the specificity diagnostic.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|cublas|n_gpus|artifact stores|bogus'
for spec in "otall3 1e-3 all" "otcut3 1e-3 cut" "otcut2 1e-2 cut"; do
  set -- $spec; arm=$1; g=$2; mode=$3
  if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
    ARM=$arm FLOOR_NEGCTX_MODE=close OFFTARGET=$g OT_MODE=$mode PYTHONPATH=src $PY -X utf8 $E/refit.py 2>&1 \
      | grep -v -i -E "$FILTER" > "$E/refit_$arm.log"
  fi
  echo "REFIT $arm DONE"
  out="$E/results/amp_eval_v3_${arm}_evclose.jsonl"
  if [ ! -s "$out" ]; then
    SEEDS=$E/seeds.txt TAG=${arm}_evclose CTR_SOURCE=close DATA=$E/data_$arm TABLES=$E/no_tables OUT=$E/results \
      PYTHONPATH=src $PY -X utf8 experiments/049-circuit-graph/amp_eval_pass_v2.py 2>&1 \
      | grep -v -i -E "$FILTER" > "$E/results/eval_${arm}_evclose.log"
  fi
  echo "EVAL $arm rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
done
ARMS=otall3,otcut3,otcut2 PYTHONPATH=src $PY -X utf8 experiments/056-specificity/specificity.py 2>&1 \
  | grep -v -i -E "$FILTER" > experiments/056-specificity/results/run_offtarget.log
echo "OFFTARGET DONE"
