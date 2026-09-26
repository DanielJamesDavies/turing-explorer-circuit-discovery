#!/usr/bin/env bash
# DAN-64 local pilot: refit the 16 targets with random and distant contrast contexts (close is fitted first by
# refit.py), then score every training arm (old store circuits + close / random / distant refits) under every
# evaluation contrast source. Run from the repo root inside WSL after data_close/ exists.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|cublas|n_gpus|artifact stores'

for mode in random distant; do
  if [ ! -f "$E/data_$mode/discovered_circuits.shard0.pt" ]; then
    FLOOR_NEGCTX_MODE=$mode PYTHONPATH=src $PY -X utf8 $E/refit.py 2>&1 | grep -v -i -E "$FILTER" > "$E/refit_$mode.log"
  fi
done
echo "REFITS DONE"

mkdir -p $E/results
for arm in old close random distant; do
  if [ "$arm" = old ]; then data=experiments/049-circuit-graph/data_full; tables=experiments/049-circuit-graph/tables_full
  else data=$E/data_$arm; tables=$E/no_tables; fi
  for ev in close random distant store; do
    out="$E/results/amp_eval_v3_${arm}_ev${ev}.jsonl"
    [ -s "$out" ] && continue
    SEEDS=$E/seeds.txt TAG=${arm}_ev${ev} CTR_SOURCE=$ev DATA=$data TABLES=$tables OUT=$E/results \
      PYTHONPATH=src $PY -X utf8 experiments/049-circuit-graph/amp_eval_pass_v2.py 2>&1 \
      | grep -v -i -E "$FILTER" > "$E/results/eval_${arm}_ev${ev}.log"
    echo "EVAL $arm ev=$ev rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
  done
done
echo "ALL DONE"
