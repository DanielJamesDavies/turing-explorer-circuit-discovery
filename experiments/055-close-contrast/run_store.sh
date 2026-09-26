#!/usr/bin/env bash
# Control arm: refit the same 16 targets locally with the OLD store contrast contexts, so overlap and score changes
# can be read against same-C refit noise (hardware/nondeterminism), not against the H100 fits alone.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|cublas|n_gpus|artifact stores'
if [ ! -f "$E/data_store/discovered_circuits.shard0.pt" ]; then
  FLOOR_NEGCTX_MODE=store PYTHONPATH=src $PY -X utf8 $E/refit.py 2>&1 | grep -v -i -E "$FILTER" > "$E/refit_store.log"
fi
for ev in close random distant store; do
  out="$E/results/amp_eval_v3_store_ev${ev}.jsonl"
  [ -s "$out" ] && continue
  SEEDS=$E/seeds.txt TAG=store_ev${ev} CTR_SOURCE=$ev DATA=$E/data_store TABLES=$E/no_tables OUT=$E/results \
    PYTHONPATH=src $PY -X utf8 experiments/049-circuit-graph/amp_eval_pass_v2.py 2>&1 \
    | grep -v -i -E "$FILTER" > "$E/results/eval_store_ev${ev}.log"
  echo "EVAL store ev=$ev rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
done
echo "STORE DONE"
