#!/bin/bash
# Exp 131 GPU queue (local RTX 5070 Ti 16 GB, WSL, ONE GPU process at a time): the off-term fits on the 16-target
# GPT-2 small panel (targets.txt), then the report.
#
#   WSL, repo root:  DRY_RUN=1 bash experiments/131-off-term/queue.sh        # print the plan, change nothing
#                    STOP_EPOCH=$(date -d 'tomorrow 07:45' +%s) nohup bash experiments/131-off-term/queue.sh \
#                        > experiments/131-off-term/out/queue.out 2>&1 &
#
# PHASES (N131_PHASES, space-separated; each a comma list of arms run.py fits per target in that order), by value:
#   "v4,o1"   the in-run v4 baseline and w_O = 1 on every target (pairs complete target by target)
#   "o0.25"   w_O = 0.25 on every target
#   "o4"      w_O = 4 (extra)
# run.py is resumable (a finished (target, arm) row is skipped), so a relaunch continues where the last one stopped.
#
# Safety: refuses to start when REPO/.gpu-lock exists, the GPU already holds > N131_GPU_IDLE_MIB [2500] MiB, host
# MemAvailable < N131_RAM_START_GB [12] GB, or OUT/ABORTED exists; writes its own .gpu-lock ("pid $$") and removes it at
# exit only while it still names this pid (exp 128 run_night.sh's pattern).
# Watchdog every N131_WATCH_S [10] s (OUT/watch.csv): GPU memory.used > N131_GPU_MAX_MIB [15300] or MemAvailable
# < N131_RAM_MIN_GB [3] GB kills run.py's process group; the target named in OUT/current.json is appended to
# OUT/tripped.txt (run.py then skips it in every arm) and the phase is relaunched: a per-fit trip never ends the queue.
# At most N131_MAX_TRIPS [6] trips in total (then the queue stops: something is wrong beyond one target), and at most 3
# launches of a phase that end in a non-trip error.
# STOP_EPOCH: passed to run.py (a (target, arm) is not started when its estimate would end after it); every job is
# killed at STOP_EPOCH + N131_HARD_GRACE [900] s. report.py (CPU) runs at the end whatever happened.
# Logs: OUT/queue.log (one line per event), job logs OUT/logs/<job>.log.
# Exit: 0 every phase done; 1 a phase failed; 7 STOP_EPOCH cut the plan; 8 refused; 9 too many watchdog trips.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
export PATH="$PATH:/usr/lib/wsl/lib"
E=experiments/131-off-term
O=${OUT131:-$E/out}
PY=./.venv/bin/python
DRY_RUN=${DRY_RUN:-0}
STOP=${STOP_EPOCH:-99999999999}
GPU_MAX=${N131_GPU_MAX_MIB:-15300}
GPU_IDLE=${N131_GPU_IDLE_MIB:-2500}
RAM_MIN_GB=${N131_RAM_MIN_GB:-3}
RAM_START_GB=${N131_RAM_START_GB:-12}
EVERY=${N131_WATCH_S:-10}
HARD_GRACE=${N131_HARD_GRACE:-900}
MAX_TRIPS=${N131_MAX_TRIPS:-6}
PHASES=${N131_PHASES:-"v4,o1 o0.25 o4"}
# Deep slice (2026-11-20) additions, all off by default:
#   N131_LOCK_OWNER   the lock is held by the caller: it must exist and contain this string; the queue neither writes
#                     nor removes it (the caller releases it)
#   N131_FIT_MAX_S    a (target, arm) running longer than this (from OUT/current.json) is a watchdog trip: the target
#                     goes to OUT/tripped.txt (skipped in every arm) and the phase is relaunched
#   N131_PANEL        the panel file preflight checks [experiments/131-off-term/out/panel.json]; TARGETS (run.py) is
#                     passed through from the environment
LOCK_OWNER=${N131_LOCK_OWNER:-}
FIT_MAX=${N131_FIT_MAX_S:-0}
PANEL=${N131_PANEL:-experiments/131-off-term/out/panel.json}
LOCK=$ROOT/.gpu-lock
LOGS=$O/logs
if [ -n "${STOP_EPOCH:-}" ]; then HARD_DEADLINE=$((STOP_EPOCH + HARD_GRACE)); else HARD_DEADLINE=99999999999; fi
RUNENV="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID N131_DEEP_BS=${N131_DEEP_BS:-1} OUT131=$O \
DONOR_DRAWS=${DONOR_DRAWS:-8}${STOP_EPOCH:+ STOP_EPOCH=$STOP_EPOCH}"

ts() { date '+%F %T'; }
say() {
  local line="[q131 $(date +%H:%M:%S)] $*"
  echo "$line"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$O"; echo "[$(ts)] $*" >> "$O/queue.log"; }
}
late() { [ "$(date +%s)" -ge "$STOP" ]; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
hm() { if [ "$1" -ge 99999999999 ]; then echo none; else date -d "@$1" '+%a %H:%M'; fi; }
cur_key() { sed -n 's/.*"seed": "\([^"]*\)".*"arm": "\([^"]*\)".*/\1 \2/p' "$O/current.json" 2> /dev/null; }

JOB=""
TRIPS=0
job() {   # job NAME CMD...: own process group, watchdog-guarded; rc = its rc, 9 = watchdog trip, 7 = hard deadline
  local name=$1; shift
  local log=$LOGS/$name.log m r why="" rc hard=0 kc jstart
  jstart=$(date +%s)
  mkdir -p "$LOGS"
  echo "[$(ts)] \$ $*" >> "$log"
  setsid "$@" >> "$log" 2>&1 < /dev/null &
  JOB=$!
  while kill -0 "$JOB" 2> /dev/null; do
    m=$(gpu_mib); r=$(avail_mib)
    echo "$(date +%s),$name,${m:-NA},${r:-NA}" >> "$O/watch.csv"
    if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX MiB"
    elif [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then why="host MemAvailable $r MiB < $RAM_MIN_GB GB"
    elif [ "$(date +%s)" -ge "$HARD_DEADLINE" ]; then why="hard deadline $(hm "$HARD_DEADLINE") reached"; hard=1
    elif [ "$FIT_MAX" -gt 0 ] && ! grep -q '"idle"' "$O/current.json" 2> /dev/null && [ -n "$(cur_key)" ] \
         && [ "$(stat -c %Y "$O/current.json" 2> /dev/null || echo 0)" -ge "$jstart" ]; then
      local t0c; t0c=$(sed -n 's/.*"t": "\([^"]*\)".*/\1/p' "$O/current.json" 2> /dev/null)
      t0c=$(date -d "$t0c" +%s 2> /dev/null || echo 0)
      if [ "$t0c" -gt 0 ] && [ $(( $(date +%s) - t0c )) -gt "$FIT_MAX" ]; then why="(target, arm) over $FIT_MAX s"; fi
    fi
    if [ -n "$why" ]; then
      kc=$(cur_key)
      say "KILL $name: $why (working on: ${kc:-?}): killing its process group"
      kill -TERM -- -"$JOB" 2> /dev/null; sleep 15; kill -KILL -- -"$JOB" 2> /dev/null
      if [ "$hard" = 0 ]; then
        echo "${kc:-unknown ?} $(date '+%F_%T') $why" >> "$O/tripped.txt"
      fi
      break
    fi
    sleep "$EVERY" & wait $!
  done
  wait "$JOB" 2> /dev/null; rc=$?
  JOB=""
  if [ -n "$why" ]; then echo "[$(ts)] killed: $why" >> "$log"; [ "$hard" = 1 ] && return 7; return 9; fi
  [ "$rc" = 0 ] || echo "[$(ts)] exit $rc" >> "$log"
  return "$rc"
}

FAILS=0
CUT=0
run_phase() {   # run_phase ARMS: run.py until it finishes the phase (rc 0), STOP_EPOCH (7), or too many trips / errors
  local arms=$1 n=0 bad=0 rc
  while :; do
    if late; then say "phase $arms: STOP_EPOCH reached: not (re)started"; return 7; fi
    n=$((n + 1))
    say "phase $arms: launch $n"
    job "run_${arms//,/_}_$n" env $RUNENV ARMS="$arms" $PY -X utf8 -u $E/run.py; rc=$?
    case $rc in
      0) say "phase $arms: done"; return 0 ;;
      7) say "phase $arms: stopped by STOP_EPOCH / the hard deadline (resumable)"; return 7 ;;
      9) TRIPS=$((TRIPS + 1))
         say "phase $arms: watchdog trip $TRIPS of $MAX_TRIPS (tripped.txt: $(tail -n 1 "$O/tripped.txt" 2> /dev/null)); relaunching"
         [ "$TRIPS" -ge "$MAX_TRIPS" ] && { say "too many watchdog trips: stopping the queue"; echo "$(ts) $TRIPS trips" > "$O/ABORTED"; return 9; } ;;
      *) bad=$((bad + 1)); say "phase $arms: run.py rc $rc ($bad of 3; log $LOGS/run_${arms//,/_}_$n.log)"
         [ "$bad" -ge 3 ] && return 1 ;;
    esac
  done
}

preflight() {
  local bad=0 f
  for f in "$PY" $E/run.py $E/off_engine.py $E/report.py ${TARGETS:-$E/targets.txt} "$PANEL" \
           experiments/062-h100-protocol-v1/driver.py src/eval/donor_test.py src/circuit/instrument/learned_mask.py \
           experiments/109-pod-run-analysis/out/archives/pub-gpt2-small-S10-results/experiments/082-public-runs/out_gpt2-small/donor/random_pool.pt \
           data_public/gpt2-small/corpus_manifest.json outputs_public/gpt2-small/latent_stats.pt; do
    [ -e "$f" ] || { say "  missing: $f"; bad=1; }
  done
  # the engine patch must still apply to today's src (every anchor exactly once)
  env PYTHONPATH=src CUDA_VISIBLE_DEVICES= $PY -c "import sys; sys.path.insert(0, '$E'); import off_engine; off_engine.patched_source()" \
    2> /dev/null || { say "  off_engine.patched_source() fails on today's src"; bad=1; }
  command -v nvidia-smi > /dev/null || { say "  nvidia-smi not found"; bad=1; }
  return "$bad"
}

say "exp 131 queue | $(git --no-pager log -1 --format='%h' 2> /dev/null) | phases: $PHASES | stop epoch $(hm "$STOP") (hard $(hm "$HARD_DEADLINE")) | watchdog GPU > $GPU_MAX MiB / RAM < $RAM_MIN_GB GB every $EVERY s$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
preflight || { say "preflight refused"; [ "$DRY_RUN" = 1 ] || exit 8; }
m=$(gpu_mib); r=$(avail_mib)
if [ "$DRY_RUN" = 1 ]; then
  say "dry run: GPU now ${m:-?} MiB (refuse above $GPU_IDLE); MemAvailable $r MiB (refuse below $((RAM_START_GB * 1024))); lock $LOCK $([ -e "$LOCK" ] && echo "EXISTS: $(head -c 200 "$LOCK")" || echo absent); ABORTED $([ -f "$O/ABORTED" ] && echo EXISTS || echo absent)"
  [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ] && say "dry run: NOTE the GPU holds $m MiB > $GPU_IDLE: a real start would be REFUSED"
  [ -e "$LOCK" ] && say "dry run: NOTE $LOCK exists: a real start would be REFUSED"
  for ph in $PHASES; do
    say "phase $ph: the plan (run.py DRY_RUN=1):"
    env $RUNENV ARMS="$ph" DRY_RUN=1 $PY -X utf8 -u $E/run.py 2>&1 | sed 's/^/    /'
    echo "    \$ env $RUNENV ARMS=$ph $PY -X utf8 -u $E/run.py   >> $LOGS/run_${ph//,/_}_<n>.log"
  done
  echo "    \$ $PY -I -B $E/report.py --out $O   >> $LOGS/report.log"
  exit 0
fi
if [ -n "$LOCK_OWNER" ]; then
  grep -q "$LOCK_OWNER" "$LOCK" 2> /dev/null || { say "refused: N131_LOCK_OWNER=$LOCK_OWNER but $LOCK does not hold it"; exit 8; }
  say "lock held by the caller ($LOCK_OWNER): not written or removed here"
else
  [ -e "$LOCK" ] && { say "refused: $LOCK exists: $(head -c 300 "$LOCK")"; exit 8; }
fi
[ -f "$O/ABORTED" ] && { say "refused: $O/ABORTED exists ($(cat "$O/ABORTED")): read it, move it away to resume"; exit 8; }
if [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ]; then say "refused: the GPU already holds $m MiB (> $GPU_IDLE, N131_GPU_IDLE_MIB)"; exit 8; fi
if [ -n "$r" ] && [ "$r" -lt $((RAM_START_GB * 1024)) ]; then say "refused: MemAvailable $r MiB < $RAM_START_GB GB"; exit 8; fi
[ -n "$m" ] || say "WARNING: nvidia-smi gave no reading; the GPU watchdog is blind"
if [ -z "$LOCK_OWNER" ]; then
  echo "exp 131 queue.sh (experiments/131-off-term) pid $$ started $(ts) on $(hostname); stop epoch ${STOP_EPOCH:-none}" > "$LOCK" \
    || { say "cannot write $LOCK"; exit 8; }
  trap 'grep -q "pid $$ " "$LOCK" 2> /dev/null && rm -f "$LOCK"' EXIT
fi
trap '[ -n "$JOB" ] && { kill -TERM -- -"$JOB" 2> /dev/null; sleep 10; kill -KILL -- -"$JOB" 2> /dev/null; }; say "signal: stopped"; exit 143' INT TERM HUP
mkdir -p "$O" "$LOGS"

QRC=0
for ph in $PHASES; do
  run_phase "$ph"; rc=$?
  case $rc in
    0) ;;
    7) CUT=1; break ;;
    9) QRC=9; break ;;
    *) FAILS=$((FAILS + 1)) ;;
  esac
done
say "report"
env CUDA_VISIBLE_DEVICES= $PY -I -B $E/report.py --out "$O" --panel "$PANEL" > "$LOGS/report.log" 2>&1 || say "report.py failed (log $LOGS/report.log)"
say "queue finished: $FAILS failed phase(s), $TRIPS watchdog trip(s)$([ "$CUT" = 1 ] && echo '; STOP_EPOCH cut the plan')"
[ "$QRC" = 0 ] || exit "$QRC"
[ "$FAILS" = 0 ] || exit 1
[ "$CUT" = 0 ] || exit 7
exit 0
