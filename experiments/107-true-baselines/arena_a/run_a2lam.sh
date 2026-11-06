#!/usr/bin/env bash
# a2lam (2026-11-06, agent `a2lam`): local check of WCM in arena A scope a2 (tc sites only, attention live and
# undecomposed = Dunefsky's universe) at other lambda / necessity settings. Information only: nothing here writes
# CHOSEN.env or any pod setting.
#   WSL, repo root:  DRY_RUN=1 bash experiments/107-true-baselines/arena_a/run_a2lam.sh    # print, run nothing
#                    bash experiments/107-true-baselines/arena_a/run_a2lam.sh
# Arms: lambda {2.5e-4, 5e-4, 1e-3} x NEC_W {0, 0.3}, minus (1e-3, 0.3) = out/dcompare/wcm_a2_nec03 (+ its score
# out/dcompare/score/a2_wcm_nec03, fire_count/a2_nec03), reused, not refitted.
# Per arm, into out/dcompare/a2lam/<arm>/:
#   fit    driver_tcz.py exactly as run_dcompare_score.sh's a2_wcm_fit (same ENVV, records, datapass, MODE=main,
#          ORDER=list, SKIP_SPEC=1, CURVE_STRIDE=20, 400 steps), keys = out/dcompare/wcm_a2_keys.txt (the 4 dev keys)
#   score  score_arena_a.py scope a2 exactly as a2_wcm_score (ho + home, --selftest, --donor with the same pool and
#          donor records), then --summary on the CPU
#   fire   fire_count_dc.py on the arm's circuits (CPU, scope a2)
# One GPU process at a time. Watchdog every 10 s (a2lam/watch.csv): kill + stop (a2lam/ABORTED) when GPU memory.used >
# 15300 MiB or host MemAvailable < 4 GB; a GPU stage is refused when the card already holds > 3000 MiB.
# Resumable: DONE markers per stage (fit/FIT_DONE, score/SCORE_DONE, fire/FIRE_DONE); the driver and scorer also resume.
# Timings: a2lam/steps.tsv. Exit 0 ok; 1 a stage failed; 8 refused; 9 aborted by the watchdog.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT" || exit 1
A=experiments/107-true-baselines/arena_a
P=$A/out/pilot
Q=$A/out/panel
W=$A/out/dcompare
SC=$W/score
L=$W/a2lam
PY=./.venv/bin/python
DRY_RUN=${DRY_RUN:-0}
KEYFILE=$W/wcm_a2_keys.txt
KEYS=$(grep . "$KEYFILE" | paste -sd, -)
ARMS=${A2L_ARMS:-"2.5e-4:0 2.5e-4:0.3 5e-4:0 5e-4:0.3 1e-3:0"}
GPU_MAX=15300
GPU_IDLE=3000
RAM_MIN_GB=4
EVERY=10
THREADS=4
DR=$Q/rec/ctx
POOL=$SC/random_pool.pt

ENVV="PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 \
ARENA_A_BOS=1 FAMILY=gpt2-small-tcz DATA=data_public/gpt2-small STORES=$P/datapass SAE_PRELOAD=0 DEEP_BS=1 \
WCM_VERSION=v4 NVIDIA_TF32_OVERRIDE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ARENA_A_CTX_DIRS=$Q/rec/ctx \
OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS"

say() { echo "[a2lam $(date +%H:%M:%S)] $*" | { [ "$DRY_RUN" = 1 ] && cat || tee -a "$L/run.log"; }; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }

ABORT=""
stage() {   # stage NAME command... : watchdog-guarded; rc 0 ok, 1 failed, 8 refused, 9 aborted
  local name=$1; shift
  local log=$L/logs/$name.log
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ env $ENVV $*   > $log"; return 0; fi
  local m=$(gpu_mib)
  if [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ]; then
    say "REFUSED $name: the GPU already holds $m MiB (> $GPU_IDLE): another process is on it"; return 8
  fi
  say "start $name (log $log)"
  local t0=$(date +%s) r why="" rc
  setsid env $ENVV "$@" >> "$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2> /dev/null; do
    sleep "$EVERY"
    m=$(gpu_mib); r=$(avail_mib)
    echo "$(date +%s),$name,${m:-NA},${r:-NA}" >> "$L/watch.csv"
    if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX MiB"
    elif [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then why="host MemAvailable $r MiB < $RAM_MIN_GB GB"; fi
    if [ -n "$why" ]; then
      say "ABORT $name: $why"
      echo "$(date '+%F %T') $name: $why" > "$L/ABORTED"
      kill -TERM -- -"$pid" 2> /dev/null; sleep 15; kill -KILL -- -"$pid" 2> /dev/null
      break
    fi
  done
  wait "$pid" 2> /dev/null; rc=$?
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$t0" "$(date +%s)" "$rc" "$why" >> "$L/steps.tsv"
  say "end $name (exit $rc, $(( $(date +%s) - t0 )) s)"
  [ -n "$why" ] && { ABORT=$why; return 9; }
  [ "$rc" -eq 0 ] && return 0 || return 1
}

FAILS=0
run_or_stop() { "$@"; local rc=$?; [ "$rc" = 9 ] && { say "aborted: $ABORT"; exit 9; }; [ "$rc" = 8 ] && exit 8
  [ "$rc" = 0 ] || FAILS=$((FAILS + 1)); return "$rc"; }

[ "$DRY_RUN" = 1 ] || mkdir -p "$L/logs"
say "a2lam | keys $KEYS | arms $ARMS | watchdog GPU > $GPU_MAX MiB / RAM < $RAM_MIN_GB GB every $EVERY s$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
for f in "$PY" "$A/driver_tcz.py" "$A/score_arena_a.py" "$W/fire_count_dc.py" "$P/datapass/top_ctx.pt" "$KEYFILE" "$POOL"; do
  [ -e "$f" ] || { say "missing $f; refused"; exit 8; }
done
[ -f "$L/ABORTED" ] && { say "refused: $L/ABORTED exists (read it, then move it away to resume)"; exit 8; }

for arm in $ARMS; do
  lam=${arm%%:*}; nec=${arm##*:}
  name=l${lam}_nec${nec}
  D=$L/$name
  [ "$DRY_RUN" = 1 ] || mkdir -p "$D/fit" "$D/score" "$D/fire"
  if [ -f "$D/fit/FIT_DONE" ]; then say "skip fit $name (FIT_DONE)"
  else
    run_or_stop stage "$name.fit" env ARENA_A_SCOPE=a2 OUT="$D/fit" CTX_DIR="$Q/rec/ctx" TARGETS="$KEYFILE" \
      MODE=main LAM="$lam" NEC_W="$nec" ORDER=list SKIP_SPEC=1 CURVE_STRIDE=20 "$PY" -X utf8 -u "$A/driver_tcz.py" \
      && { [ "$DRY_RUN" = 1 ] || touch "$D/fit/FIT_DONE"; } || continue
  fi
  if [ -f "$D/score/SCORE_DONE" ]; then say "skip score $name (SCORE_DONE)"
  else
    run_or_stop stage "$name.score" env ARENA_A_SCOPE=a2 OUT="$D/score" "$PY" -X utf8 -u "$A/score_arena_a.py" \
      --scope a2 --out "$D/score" --keys "$KEYS" --threads "$THREADS" --selftest --donor --wcm "l$lam=$D/fit" \
      --donor-pool "$POOL" --donor-records "$DR"
    if [ $? = 0 ]; then
      if [ "$DRY_RUN" = 1 ]; then echo "    \$ CUDA_VISIBLE_DEVICES= $PY score_arena_a.py --summary --out $D/score"
      elif env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PY" -X utf8 -u "$A/score_arena_a.py" \
          --summary --out "$D/score" > "$L/logs/$name.summary.log" 2>&1; then touch "$D/score/SCORE_DONE"
      else say "  $name: summary FAILED (see logs)"; FAILS=$((FAILS + 1)); fi
    fi
  fi
  if [ -f "$D/fire/FIRE_DONE" ]; then say "skip fire $name (FIRE_DONE)"
  else
    run_or_stop stage "$name.fire" env ARENA_A_SCOPE=a2 CUDA_VISIBLE_DEVICES= THREADS="$THREADS" \
      "$PY" -X utf8 -u "$W/fire_count_dc.py" "$D/fit/main/circuits" "$D/fire" \
      && { [ "$DRY_RUN" = 1 ] || touch "$D/fire/FIRE_DONE"; }
  fi
done
say "DONE ($FAILS failed stage(s))"
[ "$DRY_RUN" = 1 ] || { [ "$FAILS" = 0 ] && touch "$L/A2LAM_DONE"; }
[ "$FAILS" = 0 ]
