#!/usr/bin/env bash
# Rank-keep weight sweep (the primary method: WCM + rank_mode keep, no off-target term) on the main 16.
# Close contrast contexts, gamma_C = gamma_A = 0.25, lambda 1e-3. gamma_R 1e-2 is rkeep2 (already run).
#   rkeep3e3  gamma_R 3e-3
#   rkeep3e2  gamma_R 3e-2
#   rkeep1    gamma_R 1e-1
# Each arm: refit, v3 eval on close contrast contexts, specificity diagnostic.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|n_gpus|artifact stores|bogus|epilogue compiled'
run_arm() {   # arm gamma_R
  local arm=$1 gr=$2 sf=$E/seeds.txt
  sleep 20      # let the previous process release GPU memory
  if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
    ARM=$arm SEED_FILE=$sf FLOOR_NEGCTX_MODE=close OFFTARGET=0 RANK=$gr RANK_MODE=keep PYTHONPATH=src \
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
run_arm rkeep3e3 3e-3
run_arm rkeep3e2 3e-2
run_arm rkeep1 1e-1
echo "RKEEP SWEEP DONE"
