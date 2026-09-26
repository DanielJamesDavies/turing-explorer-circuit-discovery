#!/usr/bin/env bash
# Held-out leak fix (2026-09-24): the A and C ablation values are now built from the TRAINING split only.
# A/B on the primary config (close C, gamma_C = gamma_A = 0.25, lambda 1e-3, rank-keep 3e-3, off-target off):
#   rkeep3e3      the same config WITH the leak (fitted 2026-09-23, before the fix)
#   rkeep3e3fix   the fix (floors_train_only, the new default)
# Fits are deterministic, so any difference is the fix. Waits for the rank-keep sweep to finish first.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
FILTER='warn|extension|n_gpus|artifact stores|bogus|epilogue compiled'
until grep -q "RKEEP SWEEP DONE" "$E/run_rkeep_sweep.log" 2>/dev/null; do sleep 20; done
sleep 20
arm=rkeep3e3fix; sf=$E/seeds.txt
if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
  ARM=$arm SEED_FILE=$sf FLOOR_NEGCTX_MODE=close OFFTARGET=0 RANK=3e-3 RANK_MODE=keep FLOORS_TRAIN_ONLY=1 PYTHONPATH=src \
    $PY -X utf8 $E/refit.py 2>&1 | grep -v -i -E "$FILTER" > "$E/refit_$arm.log"
fi
echo "REFIT $arm DONE ($(grep -c ' nodes ' "$E/refit_$arm.log") targets)"
sleep 20
out="$E/results/amp_eval_v3_${arm}_evclose.jsonl"
if [ ! -s "$out" ]; then
  SEEDS=$sf TAG=${arm}_evclose CTR_SOURCE=close DATA=$E/data_$arm TABLES=$E/no_tables OUT=$E/results \
    PYTHONPATH=src $PY -X utf8 experiments/049-circuit-graph/amp_eval_pass_v2.py 2>&1 \
    | grep -v -i -E "$FILTER" > "$E/results/eval_${arm}_evclose.log"
fi
echo "EVAL $arm rows=$(wc -l < "$out" 2>/dev/null || echo 0)"
sleep 20
ARMS=$arm SEEDS=$sf PYTHONPATH=src $PY -X utf8 experiments/056-specificity/specificity.py 2>&1 \
  | grep -v -i -E "$FILTER" > experiments/056-specificity/results/run_$arm.log
echo "LEAKFIX DONE"
