#!/usr/bin/env bash
# Experiment 080, stage 0 (local GPU): the causal in-context next-token effect of every pool target
# (next_token_effect.py -> effect/effect.jsonl, effect_ctx.jsonl.gz, effects.parquet), which SAMPLING=causal samples
# from. Resumable: re-run after a restart (measured targets are skipped). Waits for any other 080 job on the card.
#
#   MODE=bench   time BENCH (default 50) targets after a warm-up one; prints sequences/s and the projected hours for
#                the full population and the subsample; writes effect/bench.json (its rows count as measured)
#   MODE=run     measure POOL (subsample: 300 per kind x band group + the 2 smoke targets, ~2.7k; or all: ~15k), then
#                write effects.parquet
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && MODE=bench nohup setsid bash experiments/080-latent-vs-output/queue_effect.sh > experiments/080-latent-vs-output/queue_effect_bench.log 2>&1 < /dev/null & disown'
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && MODE=run POOL=subsample nohup setsid bash experiments/080-latent-vs-output/queue_effect.sh > experiments/080-latent-vs-output/queue_effect.log 2>&1 < /dev/null & disown'
# Then (CPU): SAMPLING=causal PYTHONPATH=src python experiments/080-latent-vs-output/make_targets.py
cd "$(dirname "$0")/../.." || exit 1
E=experiments/080-latent-vs-output
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
MODE=${MODE:-run}
POOL=${POOL:-subsample}
BENCH=${BENCH:-50}
while pgrep -f "080-latent-vs-output/(run|calibrate|next_token_effect).py" > /dev/null; do sleep 30; done
mkdir -p $E/effect
echo "[$(date +%H:%M)] next-token effect: MODE $MODE POOL $POOL"
if [ "$MODE" = "bench" ]; then
  POOL=$POOL BENCH=$BENCH $PY -X utf8 -u $E/next_token_effect.py >> $E/effect/bench.log 2>&1
  echo "[$(date +%H:%M)] bench exit $?"; tail -3 $E/effect/bench.log
else
  POOL=$POOL $PY -X utf8 -u $E/next_token_effect.py >> $E/effect/run.log 2>&1
  echo "[$(date +%H:%M)] run exit $?"; tail -12 $E/effect/run.log
fi
echo "[$(date +%H:%M)] queue DONE"
