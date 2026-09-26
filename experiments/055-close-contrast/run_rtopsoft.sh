#!/usr/bin/env bash
# Gentler "distance from rank 1": gamma_R * ((smooth rank - 1) / 128)^p with p = 3 or 4 and gamma_R 1e-3
# (flat near rank 1, ~10x lower than rtop2 at the cut). One rank term, no off-target term, close contrast contexts,
# gamma_C = gamma_A = 0.25, lambda 1e-3. Main 16 first, then the near-threshold 5.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|n_gpus|artifact stores|bogus'
run_arm() {   # arm gamma_R power seedfile
  local arm=$1 gr=$2 pw=$3 sf=$4
  sleep 20      # let the previous process release GPU memory (rtop1 died of a cuBLAS alloc failure at start-up)
  if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
    ARM=$arm SEED_FILE=$sf FLOOR_NEGCTX_MODE=close OFFTARGET=0 RANK=$gr RANK_MODE=top RANK_POWER=$pw PYTHONPATH=src \
      $PY -X utf8 $E/refit.py 2>&1 | grep -v -i -E "$FILTER" > "$E/refit_$arm.log"
  fi
  echo "REFIT $arm DONE ($(grep -c ' nodes ' "$E/refit_$arm.log") targets)"
  sleep 20
  local out="$E/results/amp_eval_v3_${arm}_evclose.jsonl"
  if [ ! -s "$out" ]; then
    SEEDS=$sf TAG=${arm}_evclose CTR_SOURCE=close DATA=$E/data_$arm TABLES=$E/no_tables OUT=$E/results \
      PYTHONPATH=src $PY -X utf8 experiments/049-circuit-graph/amp_eval_pass_v2.py 2>&1 \
      | grep -v -i -E "$FILTER" > "$E/results/eval_${arm}_evclose.log"
  fi
  echo "EVAL $arm rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
  sleep 20
  ARMS=$arm SEEDS=$sf PYTHONPATH=src $PY -X utf8 experiments/056-specificity/specificity.py 2>&1 \
    | grep -v -i -E "$FILTER" > experiments/056-specificity/results/run_$arm.log
  echo "SPEC $arm DONE"
}
M=$E/seeds.txt; N=$E/seeds_nearthreshold.txt
run_arm rtopp3 1e-3 3 $M
run_arm rtopp4 1e-3 4 $M
run_arm ntrtopp3 1e-3 3 $N
run_arm ntrtopp4 1e-3 4 $N
echo "RTOPSOFT DONE"
