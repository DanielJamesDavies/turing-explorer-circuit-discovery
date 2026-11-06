#!/usr/bin/env bash
# Arena A item A9: the local GPU timing pilot (IMPLEMENTATION_PLAN section 2). ONE GPU process at a time, in order:
#   1 datapass  data_pass_tcz.py on the smoke corpus (data_public/gpt2-small, 2,048 x 64 wikitext tokens), pass 1 +
#               neg_ctx, batch 64, mid warm-up 4 (arena C's smoke settings)
#   2 targets   make_pilot_targets.py: 3 seeded smoke targets (tc L8-11, tc L2-6, z L8-11) + their records (rec/ctx)
#   3 diag_z    diag_zero_empty_tcz.py: the Z-empty panel, 1 latent per (layer, z / tc) = 24 targets, forward only
#   4 fit_l*    driver_tcz.py (062's driver, WCM_VERSION=v4, MODE=main, 400 steps) on the 3 targets at lambda 1e-3,
#               then 4e-3 (one OUT each), held-out strongest + mid scored, SKIP_SPEC=1
#   5 summary   pilot_summary.py (CPU): fit s / step (driver train / status rows), peak GPU memory per step
# A background sampler writes nvidia-smi memory.used every 10 s to out/pilot/gpu_mem.csv (epoch,step,MiB).
# Resumable: a finished step is skipped (its output / marker exists). No new step starts once date +%s >= STOP_EPOCH
# (set by the parent chain; default: no limit). Logs: out/pilot/logs/<step>.log; step times: out/pilot/steps.tsv.
# DRY_RUN=1 prints each step's command instead of running it.
#   WSL, repo root: STOP_EPOCH=$(date -d 'tomorrow 08:30' +%s) bash experiments/107-true-baselines/arena_a/run_pilot_a.sh
set -u
cd "$(dirname "$0")/../../.."
A=experiments/107-true-baselines/arena_a
P=$A/out/pilot
PY=.venv/bin/python
STOP_EPOCH=${STOP_EPOCH:-99999999999}
DRY_RUN=${DRY_RUN:-0}
mkdir -p "$P/logs"
say() { echo "[pilot_a] $(date +%H:%M:%S) $*"; }
late() { [ "$(date +%s)" -ge "$STOP_EPOCH" ]; }
# the settled environment (IMPLEMENTATION_PLAN section 0; latent_pilot/run_tonight.sh stage 2)
ENVV="PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 ARENA_A_BOS=1 ARENA_A_SCOPE=a1 \
FAMILY=gpt2-small-tcz DATA=data_public/gpt2-small SAE_PRELOAD=0 DEEP_BS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "idle" > "$P/.step"
if [ "$DRY_RUN" != 1 ]; then
  (
    while true; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' ')
      echo "$(date +%s),$(cat "$P/.step" 2> /dev/null),${m:-NA}" >> "$P/gpu_mem.csv"
      sleep 10
    done
  ) &
  SAMPLER=$!
  trap 'kill $SAMPLER 2> /dev/null' EXIT
fi
FAILS=0
step() {   # step NAME DONE_TEST command...   (DONE_TEST: a shell test string; true = already finished)
  local name=$1 done_test=$2; shift 2
  if eval "$done_test"; then say "skip $name (done)"; return 0; fi
  if late; then say "skip $name (STOP_EPOCH reached)"; return 1; fi
  if [ "$DRY_RUN" = 1 ]; then say "DRY $name: $*"; return 0; fi
  say "start $name"
  echo "$name" > "$P/.step"
  local t0 rc
  t0=$(date +%s)
  env $ENVV "$@" > "$P/logs/$name.log" 2>&1
  rc=$?
  printf "%s\t%s\t%s\t%s\n" "$name" "$t0" "$(date +%s)" "$rc" >> "$P/steps.tsv"
  echo "idle" > "$P/.step"
  if [ $rc -ne 0 ]; then say "FAILED $name (exit $rc; $P/logs/$name.log)"; FAILS=$((FAILS + 1)); return 1; fi
  say "done $name ($(( $(date +%s) - t0 )) s)"
}
fits_done() {   # fits_done OUT: an ok / skip status row for every target
  local n ok
  n=$(cat "$P/targets_pilot.txt" 2> /dev/null | grep -c .)
  ok=$(cat "$1"/main/status.shard*.jsonl 2> /dev/null | grep -c -E '"ok": true|"skip"')
  [ "$n" -gt 0 ] && [ "$ok" -ge "$n" ]
}
say "start; stop epoch $STOP_EPOCH ($(date -d @"$STOP_EPOCH" '+%a %H:%M' 2> /dev/null || echo none)); dry run $DRY_RUN"
# 1 data pass (CUDA, all 26 dictionaries preloaded by 069's configure: ~3.9 GB)
step datapass '[ -f $P/datapass/neg_ctx.pt ] && [ -f $P/datapass/data_pass_manifest.json ]' \
  $PY -X utf8 -u $A/data_pass_tcz.py --data data_public/gpt2-small --out $P/datapass --batch-size 64 --mid-warmup 4 \
  || { say "data pass failed: nothing else can run"; exit 1; }
# 2 targets + records
step targets '[ "$(cat $P/targets_pilot.txt 2> /dev/null | grep -c .)" -ge 3 ]' \
  STORES=$P/datapass OUT=$P/rec TARGETS_OUT=$P/targets_pilot.txt $PY -X utf8 -u $A/make_pilot_targets.py \
  || { say "no targets: fits cannot run"; [ "$DRY_RUN" = 1 ] || exit 1; }
# 3 Z-empty panel (its records go to diag_z/ctx); marker = the panel's closing line in its log
step diag_z 'grep -q "^panel:" $P/logs/diag_z.log 2> /dev/null' \
  STORES=$P/datapass OUT=$P/diag_z PER_SITE=1 WCM_VERSION=v4 $PY -X utf8 -u $A/diag_zero_empty_tcz.py
# 4 WCM fits, v4, 3 targets x 2 lambda
for LAMV in 1e-3 4e-3; do
  O=$P/wcm_l$LAMV
  step fit_l$LAMV "fits_done $O" \
    STORES=$P/datapass OUT=$O CTX_DIR=$P/rec/ctx TARGETS=$P/targets_pilot.txt MODE=main LAM=$LAMV WCM_VERSION=v4 \
    ORDER=list SKIP_SPEC=1 CURVE_STRIDE=20 $PY -X utf8 -u $A/driver_tcz.py
done
# 5 summary (CPU; runs even after STOP_EPOCH: it only reads files)
if [ "$DRY_RUN" = 1 ]; then
  say "DRY summary: CUDA_VISIBLE_DEVICES= $PY $A/pilot_summary.py $P"
else
  echo "summary" > "$P/.step"
  CUDA_VISIBLE_DEVICES= $PY -B $A/pilot_summary.py "$P" > "$P/logs/summary.log" 2>&1 || say "FAILED summary"
  cat "$P/logs/summary.log"
fi
say "DONE ($FAILS failed steps)"
[ "$FAILS" -eq 0 ]
