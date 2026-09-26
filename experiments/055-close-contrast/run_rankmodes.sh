#!/usr/bin/env bash
# One rank term, no off-target term (Daniel, 2026-09-23: only the target's place in its site's Top-K matters).
# Close contrast contexts, gamma_C = gamma_A = 0.25, lambda 1e-3.
#   rkeep2 / ntrkeep2   rank_mode keep: beat the rival at the target's own clean rank, gamma_R 1e-2
#   rtop2  / ntrtop2    rank_mode top: gamma_R * ((smooth rank - 1) / 128)^2, gamma_R 1e-2
#   rtop1  / ntrtop1    rank_mode top, gamma_R 1e-1
# Compare with close (no rank term), rankonly / ntrankonly (rank_mode cut, 1e-2), otrank2 (primary: cut + off-target).
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|cublas|n_gpus|artifact stores|bogus'
run_arm() {   # arm gamma_R mode seedfile
  local arm=$1 gr=$2 mode=$3 sf=$4
  if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
    ARM=$arm SEED_FILE=$sf FLOOR_NEGCTX_MODE=close OFFTARGET=0 RANK=$gr RANK_MODE=$mode PYTHONPATH=src \
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
M=$E/seeds.txt; N=$E/seeds_nearthreshold.txt
run_arm rkeep2 1e-2 keep $M
run_arm rtop2  1e-2 top  $M
run_arm rtop1  1e-1 top  $M
run_arm ntrkeep2 1e-2 keep $N
run_arm ntrtop2  1e-2 top  $N
run_arm ntrtop1  1e-1 top  $N
echo "RANKMODES DONE"
