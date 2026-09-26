#!/usr/bin/env bash
# Equal floor weights (gamma_C = gamma_A = 1, same weight as the zero term), close contrast contexts, same 16 targets.
#   eq1   lambda 1e-3: the plain test (total reconstruction weight doubles 1.5 -> 3, so nodes are relatively cheaper)
#   eq2   lambda 2e-3: node price matched to the default arm relative to the reconstruction terms
# Scored on the same four contrast sources as every other 055 arm.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|cublas|n_gpus|artifact stores'
for spec in "eq1 1e-3" "eq2 2e-3"; do
  set -- $spec; arm=$1; lam=$2
  if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
    ARM=$arm FLOOR_NEGCTX_MODE=close GAMMA_C=1 GAMMA_A=1 LAM=$lam PYTHONPATH=src $PY -X utf8 $E/refit.py 2>&1 \
      | grep -v -i -E "$FILTER" > "$E/refit_$arm.log"
  fi
  echo "REFIT $arm DONE"
  for ev in close random distant store; do
    out="$E/results/amp_eval_v3_${arm}_ev${ev}.jsonl"
    [ -s "$out" ] && continue
    SEEDS=$E/seeds.txt TAG=${arm}_ev${ev} CTR_SOURCE=$ev DATA=$E/data_$arm TABLES=$E/no_tables OUT=$E/results \
      PYTHONPATH=src $PY -X utf8 experiments/049-circuit-graph/amp_eval_pass_v2.py 2>&1 \
      | grep -v -i -E "$FILTER" > "$E/results/eval_${arm}_ev${ev}.log"
    echo "EVAL $arm ev=$ev rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
  done
done
echo "EQUAL DONE"
