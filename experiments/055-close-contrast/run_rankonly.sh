#!/usr/bin/env bash
# Top-K hinge ALONE (gamma_R 1e-2, no off-target term), close contrast contexts, gamma_C = gamma_A = 0.25, lambda 1e-3.
#   rankonly     main 16      vs close (neither term), otcut3e3 (cut only), otrank2 (both)
#   ntrankonly   near-threshold 5   vs ntclose, ntcut3e3, ntrank2
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|cublas|n_gpus|artifact stores|bogus'
run_arm() {   # arm seedfile
  local arm=$1 sf=$2
  if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
    ARM=$arm SEED_FILE=$sf FLOOR_NEGCTX_MODE=close OFFTARGET=0 RANK=1e-2 PYTHONPATH=src \
      $PY -X utf8 $E/refit.py 2>&1 | grep -v -i -E "$FILTER" > "$E/refit_$arm.log"
  fi
  echo "REFIT $arm DONE ($(grep -c ' nodes ' "$E/refit_$arm.log") targets)"
  local out="$E/results/amp_eval_v3_${arm}_evclose.jsonl"
  if [ ! -s "$out" ]; then
    SEEDS=$sf TAG=${arm}_evclose CTR_SOURCE=close DATA=$E/data_$arm TABLES=$E/no_tables OUT=$E/results \
      PYTHONPATH=src $PY -X utf8 experiments/049-circuit-graph/amp_eval_pass_v2.py 2>&1 \
      | grep -v -i -E "$FILTER" > "$E/results/eval_${arm}_evclose.log"
  fi
  echo "EVAL $arm rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
  ARMS=$arm SEEDS=$sf PYTHONPATH=src $PY -X utf8 experiments/056-specificity/specificity.py 2>&1 \
    | grep -v -i -E "$FILTER" > experiments/056-specificity/results/run_$arm.log
  echo "SPEC $arm DONE"
}
run_arm rankonly $E/seeds.txt
run_arm ntrankonly $E/seeds_nearthreshold.txt
echo "RANKONLY DONE"
