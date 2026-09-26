#!/usr/bin/env bash
# Top-K hinge (rank_weight gamma_R, small) on top of the provisional protocol (close C, off-target cut 3e-3).
#   near-threshold 5:  ntrank3 (gamma_R 1e-3), ntrank2 (gamma_R 1e-2)   vs ntcut3e3
#   main 16:           otrank2 (gamma_R 1e-2), the regression check     vs otcut3e3
# Each arm: refit, v3 eval on close contrast contexts, specificity diagnostic.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|cublas|n_gpus|artifact stores|bogus'
run_arm() {   # arm gamma_R seedfile
  local arm=$1 gr=$2 sf=$3
  if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
    ARM=$arm SEED_FILE=$sf FLOOR_NEGCTX_MODE=close OFFTARGET=3e-3 OT_MODE=cut RANK=$gr PYTHONPATH=src \
      $PY -X utf8 $E/refit.py 2>&1 | grep -v -i -E "$FILTER" > "$E/refit_$arm.log"
  fi
  echo "REFIT $arm DONE"
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
run_arm ntrank3 1e-3 $E/seeds_nearthreshold.txt
run_arm ntrank2 1e-2 $E/seeds_nearthreshold.txt
run_arm otrank2 1e-2 $E/seeds.txt
echo "RANK DONE"
