#!/bin/bash
# Exp 127 night runner (local RTX 5070 Ti 16 GB, WSL, ONE GPU process at a time): do WCM v4's fitted amplitudes mostly
# restore the site norm the circuit-only ablation removes? (README). TuringLLM production circuits, read only.
#
#   WSL, repo root:  DRY_RUN=1 bash experiments/127-norm-rescale/run_night.sh        # print the plan, change nothing
#                    STOP_EPOCH=$(date -d 'tomorrow 08:00' +%s) nohup bash experiments/127-norm-rescale/run_night.sh \
#                        > experiments/127-norm-rescale/out/run_night.out 2>&1 &
#
# STAGES, in this order (each skipped when OUT/state/<stage>.done exists; inside a stage run127.py skips targets that
# already have an ok row in its rows.jsonl, so a rerun resumes):
#   main54     run127.py, arms alpha1, fitted, oracle, oracle_lat + the downstream-read diagnostic, on 118 job 3's 54
#              targets (targets_fit54.txt: 6 per band x kind cell)                       -> out/main54
#   analyze1   (CPU) analyze.py -> out/report.md (the main answer exists after this)
#   site54     run127.py, arm rescale_site (norm-matched per-site scalar; reference row) -> out/site54
#   main162    as main54 on the other 162 of 118's 216 targets (targets_tl.txt minus the 54)  -> out/main162
#   site162    as site54 on those 162                                                         -> out/site162
#   analyze2   (CPU) analyze.py again (final report)
#
# Safety: refuses to start when REPO/.gpu-lock exists, the GPU already holds > N127_GPU_IDLE_MIB [2500] MiB, or
# OUT/ABORTED exists; writes its own .gpu-lock ("pid $$") and removes it at exit only while it still names this pid.
# Watchdog every N127_WATCH_S [30] s (OUT/watch.csv): GPU memory.used > N127_GPU_MAX_MIB [15300] or host MemAvailable
# < N127_RAM_MIN_GB [4] GB kills the running job's process group, writes OUT/ABORTED and stops (exit 9).
# STOP_EPOCH: no new stage once date +%s >= it (the analyze stages excepted: CPU, ~1 min); passed to run127.py
# (--stop-epoch: no new target after it; it exits 7, the stage is left resumable).
# N127_STAGES: run only these stages (comma list); N127_SKIP: skip these. Log: OUT/night.log (one line per stage
# start / end), job logs OUT/logs/<stage>.log.
# Exit: 0 every stage done; 1 a stage failed; 7 STOP_EPOCH cut the plan; 8 refused; 9 aborted by the watchdog.
set -uo pipefail
shopt -s nullglob
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
export PATH="$PATH:/usr/lib/wsl/lib"
E=experiments/127-norm-rescale
O=$E/out
E118=experiments/118-coefficient-diagnostics
PY=./.venv/bin/python
DRY_RUN=${DRY_RUN:-0}
STOP=${STOP_EPOCH:-99999999999}
GPU_MAX=${N127_GPU_MAX_MIB:-15300}
GPU_IDLE=${N127_GPU_IDLE_MIB:-2500}
RAM_MIN_GB=${N127_RAM_MIN_GB:-4}
EVERY=${N127_WATCH_S:-30}
ONLY=${N127_STAGES:-}
SKIP=${N127_SKIP:-}
LOCK=$ROOT/.gpu-lock
STATE=$O/state
LOGS=$O/logs

# inputs
T54=$E118/targets_fit54.txt
T216=$E118/targets_tl.txt
T162=$O/targets_rest162.txt
MAIN_ARMS=alpha1,fitted,oracle,oracle_lat
SITE_ARMS=rescale_site
# cost model (seconds per target on this card; GUESS from 118 job 1: 10 no-refit variants x 3 fills x 16 rows took
# 4.3 s / target median, Ctx build included, on the 54; a main unit here is ~13 read-equivalents (3 arms x 3 fills x
# 64 rows + natural pass + oracle_lat) plus the extra encode at member sites; site = 3 calibration passes x 3 fills x
# 48 rows + 1 arm) ; START = model + SAE + artefact load
MAIN_S=30; SITE_S=20; START_MIN=3

BASEENV="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID"

# ------------------------------------------------------------------------------------------------- helpers
ts() { date '+%F %T'; }
say() {
  local line="[n127 $(date +%H:%M:%S)] $*"
  echo "$line"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$O"; echo "[$(ts)] $*" >> "$O/night.log"; }
}
late() { [ "$(date +%s)" -ge "$STOP" ]; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
nlines() { if [ -f "$1" ]; then grep -c . "$1"; else echo 0; fi; }
fmin() { awk -v x="$1" 'BEGIN { printf "%.0f", x }'; }
sel() {   # sel STAGE: in N127_STAGES (if set) and not in N127_SKIP
  { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; } && [[ ",$SKIP," != *",$1,"* ]]
}

JOB=""
job() {   # job NAME CMD...: the command in its own process group, watchdog-guarded; rc = its rc, 9 = aborted
  local name=$1; shift
  local log=$LOGS/$name.log m r why="" rc
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ $*   >> $log"; return 0; fi
  mkdir -p "$LOGS"
  echo "[$(ts)] \$ $*" >> "$log"
  setsid "$@" >> "$log" 2>&1 < /dev/null &
  JOB=$!
  while kill -0 "$JOB" 2> /dev/null; do
    m=$(gpu_mib); r=$(avail_mib)
    echo "$(date +%s),$name,${m:-NA},${r:-NA}" >> "$O/watch.csv"
    if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX MiB"
    elif [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then why="host MemAvailable $r MiB < $RAM_MIN_GB GB"; fi
    if [ -n "$why" ]; then
      say "ABORT $name: $why: killing its process group"
      echo "$(ts) $name: $why" > "$O/ABORTED"
      kill -TERM -- -"$JOB" 2> /dev/null; sleep 15; kill -KILL -- -"$JOB" 2> /dev/null
      break
    fi
    sleep "$EVERY" & wait $!
  done
  wait "$JOB" 2> /dev/null; rc=$?
  JOB=""
  [ -n "$why" ] && return 9
  [ "$rc" = 0 ] || echo "[$(ts)] exit $rc" >> "$log"
  return "$rc"
}

declare -A DEPS=([site162]="main162")
declare -A EST=()
FAILS=0
CUT=0
ETA=0
stage() {   # stage NAME FUNC [always]: markers, deps, STOP_EPOCH, night.log; "always" = runs every time (CPU analysis)
  local name=$1 fn=$2 always=${3:-} d rc t0
  sel "$name" || return 0
  if [ -z "$always" ] && [ -f "$STATE/$name.done" ]; then
    say "$name: done earlier ($(cat "$STATE/$name.done"))"; return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    ETA=$(awk -v a="$ETA" -v b="${EST[$name]:-1}" 'BEGIN { print a + b }')
    say "$name: PLAN (~$(fmin "${EST[$name]:-1}") min; cumulative ~$(fmin "$ETA") min)$([ -n "${DEPS[$name]:-}" ] && echo "; needs ${DEPS[$name]}")"
    "$fn"
    return 0
  fi
  for d in ${DEPS[$name]:-}; do
    if [ ! -f "$STATE/$d.done" ]; then say "$name: skipped (dependency $d not done)"; return 0; fi
  done
  if [ -z "$always" ] && late; then say "$name: STOP_EPOCH reached: not started"; CUT=1; return 7; fi
  say "START $name (estimate ~$(fmin "${EST[$name]:-1}") min)"
  t0=$(date +%s)
  "$fn"; rc=$?
  say "END $name rc $rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if [ "$rc" = 0 ]; then mkdir -p "$STATE"; ts > "$STATE/$name.done"
  elif [ "$rc" = 9 ]; then say "watchdog abort: stopping the night (exit 9; read $O/ABORTED, move it away to resume)"; exit 9
  elif [ "$rc" = 7 ] || late; then CUT=1                 # cut by STOP_EPOCH (run127.py exits 7): resumable
  else FAILS=$((FAILS + 1)); fi
  return 0
}

# ------------------------------------------------------------------------------------------------- units
run() {   # run NAME TARGETS ARMS: run127.py on the GPU -> OUT/NAME; then every target must have an ok row
  local name=$1 tgt=$2 arms=$3 stop="" rc n ok
  [ -n "${STOP_EPOCH:-}" ] && stop=" --stop-epoch $STOP_EPOCH"
  job "$name" bash -c "env $BASEENV CUDA_VISIBLE_DEVICES=0 $PY -X utf8 -u $E/run127.py --targets $tgt --out $O/$name --arms $arms$stop"
  rc=$?
  [ "$DRY_RUN" = 1 ] && return 0
  [ "$rc" = 0 ] || return "$rc"
  n=$(nlines "$tgt")
  ok=$(grep -v '"error"' "$O/$name/rows.jsonl" 2> /dev/null | grep -o '"seed": "[^"]*"' | sort -u | wc -l)
  [ "$ok" -ge "$n" ] || { say "  $name: $ok of $n targets ok (error rows: rerun retries them)"; return 1; }
}
u_main54() { run main54 $T54 $MAIN_ARMS; }
u_site54() { run site54 $T54 $SITE_ARMS; }
u_main162() {
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ grep -vxFf $T54 $T216 > $T162"
  else grep -vxFf <(tr -d '\r' < "$T54") <(tr -d '\r' < "$T216") > "$T162"; say "  $(nlines "$T162") targets in $T162"; fi
  run main162 $T162 $MAIN_ARMS
}
u_site162() { run site162 $T162 $SITE_ARMS; }
u_analyze() {
  job analyze env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -I -B -X utf8 $E/analyze.py --out-dir $O
}

# ------------------------------------------------------------------------------------------------- estimates
N54=$(nlines "$T54")
N162=$(( $(nlines "$T216") - N54 ))
se() { awk -v n="$1" -v s="$2" -v z="$START_MIN" 'BEGIN { print z + n * s / 60 }'; }
EST=([main54]=$(se $N54 $MAIN_S) [analyze1]=1 [site54]=$(se $N54 $SITE_S) [main162]=$(se $N162 $MAIN_S)
     [site162]=$(se $N162 $SITE_S) [analyze2]=1)

# ------------------------------------------------------------------------------------------------- preflight
preflight() {   # read-only; prints every problem; rc 1 if any
  local bad=0 f k A=experiments/109-pod-run-analysis/out/archives
  local CIRC=$A/pub-turingllm-T3-main/root/turing-tl/experiments/094-tl-v2-records/out_v2/main/circuits
  local CTX=$A/pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh
  for f in "$PY" "$E/run127.py" "$E/analyze.py" "$E118/run118.py" "$E118/out/rows.jsonl" "$E118/out/fit_tk/rows.jsonl" \
           "$T54" "$T216" "$CIRC" "$CTX" outputs/candidates.pt outputs/latent_stats.pt; do
    [ -e "$f" ] || { say "  missing: $f"; bad=1; }
  done
  for k in $(tr -d '\r' < "$T216" 2> /dev/null); do        # 118's target lists have CRLF line ends
    [ -f "$CIRC/$k.pt" ] || { say "  no circuit $CIRC/$k.pt"; bad=1; }
    [ -f "$CTX/$k.pt" ] || { say "  no record $CTX/$k.pt"; bad=1; }
  done
  command -v nvidia-smi > /dev/null || { say "  nvidia-smi not found"; bad=1; }
  return "$bad"
}

# ------------------------------------------------------------------------------------------------- main
say "exp 127 night | $(git --no-pager log -1 --format='%h' 2> /dev/null) | targets 54 + $N162 | stop epoch $([ -n "${STOP_EPOCH:-}" ] && date -d "@$STOP_EPOCH" '+%a %F %H:%M' || echo none) | watchdog GPU > $GPU_MAX MiB / RAM < $RAM_MIN_GB GB every $EVERY s$([ -n "$ONLY" ] && echo " | only $ONLY")$([ -n "$SKIP" ] && echo " | skip $SKIP")$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
preflight || { say "preflight refused"; [ "$DRY_RUN" = 1 ] || exit 8; }
m=$(gpu_mib)
if [ "$DRY_RUN" = 1 ]; then
  say "dry run: GPU now holds ${m:-?} MiB (refuse above $GPU_IDLE); lock $LOCK $([ -e "$LOCK" ] && echo "EXISTS: $(head -c 200 "$LOCK")" || echo absent); ABORTED $([ -f "$O/ABORTED" ] && echo EXISTS || echo absent)"
  [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ] && say "dry run: NOTE the GPU holds $m MiB > $GPU_IDLE: a real start would be REFUSED (N127_GPU_IDLE_MIB)"
else
  [ -e "$LOCK" ] && { say "refused: $LOCK exists: $(head -c 300 "$LOCK")"; exit 8; }
  [ -f "$O/ABORTED" ] && { say "refused: $O/ABORTED exists ($(cat "$O/ABORTED")): read it, move it away to resume"; exit 8; }
  if [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ]; then say "refused: the GPU already holds $m MiB (> $GPU_IDLE, N127_GPU_IDLE_MIB)"; exit 8; fi
  [ -n "$m" ] || say "WARNING: nvidia-smi gave no reading; the GPU watchdog is blind"
  echo "exp 127 run_night.sh (experiments/127-norm-rescale) pid $$ started $(ts) on $(hostname); stop epoch ${STOP_EPOCH:-none}" > "$LOCK" \
    || { say "cannot write $LOCK"; exit 8; }
  # the lock is this run's own: removed at exit only while it still names this pid
  trap 'grep -q "pid $$ " "$LOCK" 2> /dev/null && rm -f "$LOCK"' EXIT
  trap '[ -n "$JOB" ] && { kill -TERM -- -"$JOB" 2> /dev/null; sleep 10; kill -KILL -- -"$JOB" 2> /dev/null; }; say "signal: stopped"; exit 143' INT TERM HUP
  mkdir -p "$O" "$STATE" "$LOGS"
fi

stage main54 u_main54
stage analyze1 u_analyze always
stage site54 u_site54
stage main162 u_main162
stage site162 u_site162
stage analyze2 u_analyze always
if [ "$DRY_RUN" = 1 ]; then say "dry run: total ~$(fmin "$ETA") min (~$(awk -v m="$ETA" 'BEGIN { printf "%.1f", m / 60 }') h) for the stages not done"; exit 0; fi
say "night finished: $FAILS failed stage(s)$([ "$CUT" = 1 ] && echo '; STOP_EPOCH cut the plan')"
[ "$FAILS" = 0 ] || exit 1
[ "$CUT" = 0 ] || exit 7
exit 0
