#!/usr/bin/env bash
# Experiment 108: does WCM with more contexts per optimiser step give better or more stable circuits? (README.md)
# Local GPU queue run (WSL, repo root). ONE GPU process at a time, every stage resumable (rerun = continue):
#   0 preflight     files present; refuse to start when the card already holds > GPU_IDLE_MIB [3000] MiB
#   1 timing        B4 then B16 on the first 2 targets (11.resid.10297, the deepest; 4.attn.35016, shallow), then the
#                   donor test of B4 on them: s / step, peak VRAM, start-up and donor cost (the fits are kept: they are
#                   the main run's B4 / B16 fits of those targets)
#   2 plan          plan.py sizes N (first N of targets.txt, 8..16) so everything fits by the deadline; DONOR auto
#   3 fits          arms in order B4, B48, B16m, B16, B4s2, B48s2 (one driver process per arm over the N targets)
#   4 donor         the donor-background test per arm (same order), if the plan kept it
#   5 analysis      analyze.py (CPU) -> out/analysis.md
# Deadline = min(now + BUDGET_H [11] h, STOP_EPOCH): no fit starts after it (fit_arm.py exits 75 between fits; a fit in
# progress finishes), no stage starts after it.
# Watchdog every WATCH_S [5] s while a GPU process runs (out/watch.csv): GPU memory.used > GPU_MAX_MIB [15300] or host
# MemAvailable < RAM_MIN_GB [4] GB -> the process group is killed (TERM, then KILL after 15 s), the fit in progress
# (out/<arm>/b108_current.txt) is recorded in out/<arm>/b108_aborted.txt and excluded, and the arm's driver restarts on
# the remaining targets (at most MAX_ABORTS [3] per arm).
# DRY_RUN=1: prints every command (plan.py runs on CPU in PRIOR mode to show a plan), runs nothing on the GPU.
#   nohup setsid bash experiments/108-batch-size/run.sh > experiments/108-batch-size/out/run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
E=experiments/108-batch-size
O=$E/out
PY=.venv/bin/python
CTX=${CTX_DIR:-experiments/062-h100-protocol-v1/out_full/out/ctx}
POOL=${DONOR_POOL:-experiments/073-strict-eval/out/random_pool.pt}
BUDGET_H=${BUDGET_H:-11}
DRY=${DRY_RUN:-0}
GPU_MAX=${GPU_MAX_MIB:-15300}
GPU_IDLE=${GPU_IDLE_MIB:-3000}
RAM_MIN_GB=${RAM_MIN_GB:-4}
WATCH_S=${WATCH_S:-5}
MAX_ABORTS=${MAX_ABORTS:-3}
ARMS="B4 B48 B16m B16 B4s2 B48s2"
T0=$(date +%s)
DEADLINE=$(awk -v t="$T0" -v h="$BUDGET_H" 'BEGIN { printf "%d", t + h * 3600 }')
[ -n "${STOP_EPOCH:-}" ] && [ "${STOP_EPOCH%.*}" -lt "$DEADLINE" ] && DEADLINE=${STOP_EPOCH%.*}
mkdir -p "$O/logs"
say() { echo "[108] $(date '+%a %H:%M:%S') $*"; }
late() { [ "$(date +%s)" -ge "$DEADLINE" ]; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
rseed() { case "$1" in *s2) echo "R_SEED=20261108";; *) echo "";; esac; }
TL=(PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 CTX_DIR="$CTX")

wait_idle() {   # the card must be (nearly) free before a GPU process starts; wait up to 120 s for the last one to exit
  [ "$DRY" = 1 ] && return 0
  local i m
  for i in $(seq 1 24); do
    m=$(gpu_mib); [ -n "$m" ] && [ "$m" -le "$GPU_IDLE" ] && return 0
    sleep 5
  done
  say "REFUSE: the card holds ${m:-?} MiB > $GPU_IDLE MiB (another GPU process?); stopping"
  return 1
}

# gpu_run LABEL ARM LOG cmd...: run one GPU process in its own process group under the watchdog.
# rc: the process's, 99 = watchdog abort, 98 = refused (card busy)
gpu_run() {
  local label=$1 arm=$2 log=$3; shift 3
  if [ "$DRY" = 1 ]; then echo "  \$ (watchdog $WATCH_S s: GPU > $GPU_MAX MiB / RAM < $RAM_MIN_GB GB) setsid env $* > $log 2>&1"; return 0; fi
  wait_idle || return 98
  local t1 rc=0 peak=0 m r why=""
  t1=$(date +%s)
  setsid env "$@" >> "$log" 2>&1 < /dev/null &
  local pid=$!                                              # setsid in a non-interactive shell: pid = process group
  trap 'kill -TERM -- -'"$pid"' 2> /dev/null; exit 130' INT TERM
  while kill -0 "$pid" 2> /dev/null; do
    m=$(gpu_mib); r=$(avail_mib)
    echo "$(date +%s),$label,${m:-},${r:-}" >> "$O/watch.csv"
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX"; fi
    if [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then why="host MemAvailable $r MiB < $RAM_MIN_GB GB"; fi
    if [ -n "$why" ]; then
      local cur=""
      [ -n "$arm" ] && [ -f "$O/$arm/b108_current.txt" ] && cur=$(head -n 1 "$O/$arm/b108_current.txt" | tr -d ' \r')
      say "ABORT $label: $why (fit in progress: ${cur:-none})"
      [ -n "$arm" ] && echo -e "${cur:-?}\t$why\t$(date +%s)" >> "$O/$arm/b108_aborted.txt"
      kill -TERM -- -"$pid" 2> /dev/null; sleep 15; kill -KILL -- -"$pid" 2> /dev/null
      rc=99; break
    fi
    sleep "$WATCH_S"
  done
  if [ "$rc" != 99 ]; then wait "$pid"; rc=$?; fi
  trap - INT TERM
  echo -e "$label\t$t1\t$(date +%s)\t$rc\t$peak" >> "$O/stages.tsv"
  say "$label: rc $rc, $(( $(date +%s) - t1 )) s, peak GPU memory.used $peak MiB"
  return $rc
}

# fit_arm ARM TARGETS LABEL: the arm's driver over TARGETS minus its aborted fits; restarts after a watchdog abort
fit_arm() {
  local arm=$1 tf=$2 label=$3 n_ab rc todo=$O/$1/targets_todo.txt
  mkdir -p "$O/$arm"
  while true; do
    late && { say "skip $label: past the deadline"; return 75; }
    if [ -f "$O/$arm/b108_aborted.txt" ]; then
      n_ab=$(wc -l < "$O/$arm/b108_aborted.txt")
      [ "$n_ab" -ge "$MAX_ABORTS" ] && { say "skip $label: $n_ab aborts"; return 0; }
      grep -v -x -F -f <(cut -f1 "$O/$arm/b108_aborted.txt") "$tf" > "$todo" || true
      grep -q '^?' "$O/$arm/b108_aborted.txt" && { say "skip $label: an abort between fits (host / card pressure)"; return 0; }
    else
      cp "$tf" "$todo"
    fi
    [ -s "$todo" ] || return 0
    say "fits $label ($(wc -l < "$todo") targets)"
    gpu_run "$label" "$arm" "$O/logs/$label.log" "${TL[@]}" $(rseed "$arm") ARM="$arm" OUT="$O/$arm" TARGETS="$todo" \
      STOP_EPOCH="$DEADLINE" "$PY" -u "$E/fit_arm.py"
    rc=$?
    [ "$DRY" = 1 ] && return 0
    [ "$rc" = 99 ] && continue                       # aborted: that fit is excluded, restart on the rest
    return $rc
  done
}

donor_arm() {   # ARM TARGETS LABEL
  local arm=$1 tf=$2 label=$3
  late && { say "skip $label: past the deadline"; return 75; }
  [ "$DRY" = 1 ] || [ -d "$O/$arm/main/circuits" ] || { say "skip $label: no circuits"; return 0; }
  say "donor $label"
  gpu_run "$label" "" "$O/logs/$label.log" "${TL[@]}" $(rseed "$arm") WCM_VERSION=v4 OUT="$O/$arm" DONOR_TARGETS="$tf" \
    DONOR_ARMS=main DONOR_VARIANTS=fit DONOR_POOL="$POOL" "$PY" -u src/eval/donor_test.py
}

# ---------------------------------------------------------------------------------------------------- 0 preflight
say "start; budget $BUDGET_H h, deadline $(date -d @$DEADLINE '+%a %H:%M'); DRY_RUN=$DRY"
for f in "$E/targets.txt" "$E/fit_arm.py" "$E/arms.py" "$E/plan.py" "$E/analyze.py" "$PY" "$POOL"; do
  [ -e "$f" ] || { say "missing $f"; exit 2; }
done
[ -d "$CTX" ] || { say "missing CTX_DIR $CTX"; exit 2; }
if [ "$DRY" != 1 ]; then
  m=$(gpu_mib)
  [ -z "$m" ] && { say "REFUSE: nvidia-smi gave no reading"; exit 3; }
  [ "$m" -gt "$GPU_IDLE" ] && { say "REFUSE: the card already holds $m MiB > $GPU_IDLE MiB"; exit 3; }
  say "card at $m MiB, host MemAvailable $(avail_mib) MiB"
fi

# ---------------------------------------------------------------------------------------------------- 1 timing
head -n 2 "$E/targets.txt" > "$O/timing_targets.txt"
fit_arm B4 "$O/timing_targets.txt" timing_B4; rc=$?; [ "$rc" = 98 ] && exit 3
fit_arm B16 "$O/timing_targets.txt" timing_B16; rc=$?; [ "$rc" = 98 ] && exit 3
if [ "${DONOR:-auto}" != 0 ]; then donor_arm B4 "$O/timing_targets.txt" timing_donor_B4; [ $? = 98 ] && exit 3; fi
late && { say "deadline reached in the timing stage"; exit 0; }

# ---------------------------------------------------------------------------------------------------- 2 plan
if [ "$DRY" = 1 ]; then
  echo "  \$ PRIOR=1 DEADLINE=$DEADLINE $PY $E/plan.py   (dry run: CPU, prior timings; real run uses the timing stage)"
  PRIOR=1 DEADLINE=$DEADLINE DONOR=${DONOR:-auto} "$PY" "$E/plan.py" || exit 4
  RUN_T=$O/targets_run.dry.txt
  PLAN=$O/plan.dry.json
else
  DEADLINE=$DEADLINE DONOR=${DONOR:-auto} "$PY" "$E/plan.py" \
    || { say "plan failed (a timing fit missing or aborted? see $O/stages.tsv and $O/*/b108_aborted.txt)"; exit 4; }
  RUN_T=$O/targets_run.txt
  PLAN=$O/plan.json
fi
grep -q '"donor": true' "$PLAN" && DO_DONOR=1 || DO_DONOR=0

# ---------------------------------------------------------------------------------------------------- 3 fits
for a in $ARMS; do
  fit_arm "$a" "$RUN_T" "fit_$a"; rc=$?
  [ "$rc" = 98 ] && exit 3
  [ "$rc" = 75 ] && break
done

# ---------------------------------------------------------------------------------------------------- 4 donor
if [ "$DO_DONOR" = 1 ]; then
  for a in $ARMS; do
    donor_arm "$a" "$RUN_T" "donor_$a"; rc=$?
    [ "$rc" = 98 ] && exit 3
    [ "$rc" = 75 ] && break
  done
else
  say "donor test not run (plan: no time, or DONOR=0)"
fi

# ---------------------------------------------------------------------------------------------------- 5 analysis
if [ "$DRY" = 1 ]; then
  echo "  \$ PYTHONPATH=src CUDA_VISIBLE_DEVICES= $PY $E/analyze.py > $O/analysis.md"
else
  PYTHONPATH=src CUDA_VISIBLE_DEVICES= "$PY" "$E/analyze.py" > "$O/analysis.md" 2> "$O/logs/analyze.log" \
    && say "analysis -> $O/analysis.md" || say "analysis failed (see $O/logs/analyze.log)"
fi
say "DONE"
