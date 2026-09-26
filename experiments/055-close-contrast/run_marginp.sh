#!/usr/bin/env bash
# Option 1, corrected: firing margin (k = 128) + off-target cut 3e-3, with the loss normalised by the seed's PLAIN
# pre-activation scale (margin_plain_norm). The first margin arm (ntmargin) normalised by mean(margin^2), which is ~0
# for near-threshold seeds, so lambda became negligible and the fits kept 4k-200k latents.
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
PY=./.venv/bin/python
E=experiments/055-close-contrast
SF=$E/seeds_nearthreshold.txt
FILTER='warn|extension|cublas|n_gpus|artifact stores|bogus'
until grep -q "NEARTHRESHOLD DONE" "$E/run_nearthreshold.log" 2>/dev/null; do sleep 20; done
arm=ntmarginp
if [ ! -f "$E/data_$arm/discovered_circuits.shard0.pt" ]; then
  ARM=$arm SEED_FILE=$SF FLOOR_NEGCTX_MODE=close OFFTARGET=3e-3 OT_MODE=cut MARGIN=128 MARGIN_PLAIN_NORM=1 PYTHONPATH=src \
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
ARMS=$arm SEEDS=$SF PYTHONPATH=src $PY -X utf8 experiments/056-specificity/specificity.py 2>&1 \
  | grep -v -i -E "$FILTER" > experiments/056-specificity/results/run_$arm.log
echo "MARGINP DONE"
