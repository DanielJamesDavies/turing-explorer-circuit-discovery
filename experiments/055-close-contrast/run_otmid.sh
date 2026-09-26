#!/usr/bin/env bash
# Off-target term, cut mode, the middle weight between otcut3 (1e-3, too weak for the worst cases) and otcut2
# (1e-2, costs faithfulness): gamma_S 3e-3. Same 16 targets, close contrast contexts, gamma_C = gamma_A = 0.25,
# lambda 1e-3. Refit, v3 eval on close contrast contexts, specificity diagnostic.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|cublas|n_gpus|artifact stores|bogus'
arm=otcut3e3
if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
  ARM=$arm FLOOR_NEGCTX_MODE=close OFFTARGET=3e-3 OT_MODE=cut PYTHONPATH=src $PY -X utf8 $E/refit.py 2>&1 \
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
ARMS=$arm PYTHONPATH=src $PY -X utf8 experiments/056-specificity/specificity.py 2>&1 \
  | grep -v -i -E "$FILTER" > experiments/056-specificity/results/run_$arm.log
echo "OTMID DONE"
