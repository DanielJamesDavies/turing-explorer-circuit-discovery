#!/usr/bin/env bash
# Near-threshold targets (clean rank close to the Top-K cut; activation read 0 in every earlier arm):
# 10.attn.16967 (clean rank 119 / K 128), 10.attn.36603 (63), and the pilot's 8.attn.29788, 9.attn.8863,
# 10.attn.38182 (tiny activations, 0.00 on every score). Close contrast contexts, gamma_C = gamma_A = 0.25, lambda 1e-3.
#   ntclose     baseline, no off-target term
#   ntcut3e3    off-target cut 3e-3 (the provisional default)
#   ntmargin    option 1: firing margin (k = 128) + off-target cut 3e-3
#   ntcut3e2    option 2: off-target cut 3e-2
#   ntcut1e1    option 2: off-target cut 1e-1
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
SF=$E/seeds_nearthreshold.txt
FILTER='warn|extension|cublas|n_gpus|artifact stores|bogus'
for spec in "ntclose 0 -" "ntcut3e3 3e-3 -" "ntmargin 3e-3 128" "ntcut3e2 3e-2 -" "ntcut1e1 1e-1 -"; do
  set -- $spec; arm=$1; g=$2; m=$3
  if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
    if [ "$m" = "-" ]; then MG=""; else MG=$m; fi
    ARM=$arm SEED_FILE=$SF FLOOR_NEGCTX_MODE=close OFFTARGET=$g OT_MODE=cut MARGIN=$MG PYTHONPATH=src \
      $PY -X utf8 $E/refit.py 2>&1 | grep -v -i -E "$FILTER" > "$E/refit_$arm.log"
  fi
  echo "REFIT $arm DONE"
  out="$E/results/amp_eval_v3_${arm}_evclose.jsonl"
  if [ ! -s "$out" ]; then
    SEEDS=$SF TAG=${arm}_evclose CTR_SOURCE=close DATA=$E/data_$arm TABLES=$E/no_tables OUT=$E/results \
      PYTHONPATH=src $PY -X utf8 experiments/049-circuit-graph/amp_eval_pass_v2.py 2>&1 \
      | grep -v -i -E "$FILTER" > "$E/results/eval_${arm}_evclose.log"
  fi
  echo "EVAL $arm rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
done
ARMS=ntclose,ntcut3e3,ntmargin,ntcut3e2,ntcut1e1 SEEDS=$SF PYTHONPATH=src $PY -X utf8 experiments/056-specificity/specificity.py 2>&1 \
  | grep -v -i -E "$FILTER" > experiments/056-specificity/results/run_nearthreshold.log
echo "NEARTHRESHOLD DONE"
