#!/bin/bash
# Exp 134 step 3 GPU job (local 16 GB card, WSL, ONE GPU process). Runs s3_run.py (phases reads, arms, sweep, donor in
# that order over the whole target list), relaunching after a watchdog trip, then s3_report.py (CPU).
# Patterned on exp 131 queue.sh (ours, adapted): the watchdog and the trip handling are 131's.
#
#   WSL, repo root:  DRY_RUN=1 bash experiments/134-sub1-coefficients/s3_job.sh
#                    STOP_EPOCH=$(date -d 'tomorrow 07:45' +%s) nohup bash experiments/134-sub1-coefficients/s3_job.sh \
#                        > experiments/134-sub1-coefficients/out/s3/job.out 2>&1 &
#
# Lock: the operator writes `sub1-134` into REPO/.gpu-lock (Write tool) BEFORE the start; this script REFUSES unless the
# lock exists and starts with `sub1-134`, and never removes it (the operator deletes it with delete_path afterwards).
# Also refuses when the GPU already holds > S134_GPU_IDLE_MIB [2500] MiB, an lrcheck / wcm_gemma process runs, or
# MemAvailable < 12 GB.
# Watchdog every 10 s (OUT/watch.csv): GPU memory.used > S134_GPU_MAX_MIB [15300] or MemAvailable < 3 GB kills the
# run's process group; the (target, phase) in OUT/current.json goes to OUT/tripped.txt (that phase skipped) and the run is
# relaunched; at most 6 trips. Every job is killed at STOP_EPOCH + 900 s.
# Exit: 0 done; 1 run failed 3 times; 7 STOP_EPOCH cut the plan; 8 refused; 9 too many trips.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
export PATH="$PATH:/usr/lib/wsl/lib"
E=experiments/134-sub1-coefficients
# MODEL134 [gpt2-small] | pythia-70m (step 3b): pythia uses out/s3py, s3py_targets.txt and the lock tag
# S134_LOCK_TAG (default sub1-134 for both; step 3b runs with S134_LOCK_TAG=sub1py-1120). gpt2-small is unchanged.
MODEL134=${MODEL134:-gpt2-small}
case "$MODEL134" in gpt2-small) PFX=s3 ;; pythia-70m) PFX=s3py ;; *) echo "MODEL134 $MODEL134?"; exit 8 ;; esac
LOCK_TAG=${S134_LOCK_TAG:-sub1-134}
O=${OUT134:-$E/out/$PFX}
PY=./.venv/bin/python
DRY_RUN=${DRY_RUN:-0}
GPU_MAX=${S134_GPU_MAX_MIB:-15300}
GPU_IDLE=${S134_GPU_IDLE_MIB:-2500}
RAM_MIN_GB=3
RAM_START_GB=12
EVERY=10
MAX_TRIPS=6
LOCK=$ROOT/.gpu-lock
LOGS=$O/logs
if [ -n "${STOP_EPOCH:-}" ]; then HARD_DEADLINE=$((STOP_EPOCH + 900)); else HARD_DEADLINE=99999999999; fi
RUNENV="MODEL134=$MODEL134 PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID OUT134=$O DONOR_DRAWS=${DONOR_DRAWS:-8} GRAD_BS=${GRAD_BS:-2} \
PHASES=${PHASES:-reads,arms,sweep,donor}${STOP_EPOCH:+ STOP_EPOCH=$STOP_EPOCH}"

ts() { date '+%F %T'; }
say() {
  local line="[s134 $(date +%H:%M:%S)] $*"
  echo "$line"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$O"; echo "[$(ts)] $*" >> "$O/job.log"; }
}
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
cur_key() { sed -n 's/.*"seed": "\([^"]*\)".*"phase": "\([^"]*\)".*/\1 \2/p' "$O/current.json" 2> /dev/null; }

JOB=""
job() {   # job NAME CMD...: own process group, watchdog-guarded; rc = its rc, 9 = watchdog trip, 7 = hard deadline
  local name=$1; shift
  local log=$LOGS/$name.log m r why="" rc hard=0 kc
  mkdir -p "$LOGS"
  echo "[$(ts)] \$ $*" >> "$log"
  setsid "$@" >> "$log" 2>&1 < /dev/null &
  JOB=$!
  while kill -0 "$JOB" 2> /dev/null; do
    m=$(gpu_mib); r=$(avail_mib)
    echo "$(date +%s),$name,${m:-NA},${r:-NA}" >> "$O/watch.csv"
    if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX MiB"
    elif [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then why="host MemAvailable $r MiB < $RAM_MIN_GB GB"
    elif [ "$(date +%s)" -ge "$HARD_DEADLINE" ]; then why="hard deadline reached"; hard=1; fi
    if [ -n "$why" ]; then
      kc=$(cur_key)
      say "KILL $name: $why (working on: ${kc:-?}): killing its process group"
      kill -TERM -- -"$JOB" 2> /dev/null; sleep 15; kill -KILL -- -"$JOB" 2> /dev/null
      [ "$hard" = 0 ] && echo "${kc:-unknown ?} $(date '+%F_%T') $why" >> "$O/tripped.txt"
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

say "exp 134 s3 job | stop epoch ${STOP_EPOCH:-none} | watchdog GPU > $GPU_MAX MiB / RAM < $RAM_MIN_GB GB$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
for f in "$PY" $E/s3_run.py $E/s3_report.py $E/${PFX}_targets.txt; do [ -e "$f" ] || { say "missing $f"; exit 8; }; done
m=$(gpu_mib); r=$(avail_mib)
others=$(pgrep -af 'lrcheck_1119|wcm_gemma' | grep -v pgrep || true)
lockc=$([ -e "$LOCK" ] && head -c 200 "$LOCK" || echo ABSENT)
say "GPU ${m:-?} MiB, MemAvailable $r MiB, lock: $lockc, other GPU jobs: ${others:-none}"
if [ "$DRY_RUN" = 1 ]; then
  echo "    \$ env $RUNENV $PY -X utf8 -u $E/s3_run.py   >> $LOGS/run_<n>.log"
  exit 0
fi
case "$lockc" in "$LOCK_TAG"*) ;; *) say "refused: $LOCK does not hold $LOCK_TAG ($lockc)"; exit 8 ;; esac
[ -n "$others" ] && { say "refused: another GPU job runs: $others"; exit 8; }
if [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ]; then say "refused: the GPU already holds $m MiB"; exit 8; fi
if [ -n "$r" ] && [ "$r" -lt $((RAM_START_GB * 1024)) ]; then say "refused: MemAvailable $r MiB < $RAM_START_GB GB"; exit 8; fi
[ -n "$m" ] || say "WARNING: nvidia-smi gave no reading; the GPU watchdog is blind"
trap '[ -n "$JOB" ] && { kill -TERM -- -"$JOB" 2> /dev/null; sleep 10; kill -KILL -- -"$JOB" 2> /dev/null; }; say "signal: stopped"; exit 143' INT TERM HUP
mkdir -p "$O" "$LOGS"

n=0; bad=0; trips=0; QRC=0
while :; do
  n=$((n + 1))
  say "launch $n"
  job "run_$n" env $RUNENV $PY -X utf8 -u $E/s3_run.py; rc=$?
  case $rc in
    0) say "run: done"; break ;;
    7) say "run: STOP_EPOCH cut the plan (resumable)"; QRC=7; break ;;
    9) trips=$((trips + 1)); say "watchdog trip $trips of $MAX_TRIPS ($(tail -n 1 "$O/tripped.txt" 2> /dev/null)); relaunching"
       [ "$trips" -ge "$MAX_TRIPS" ] && { say "too many trips: stopping"; QRC=9; break; } ;;
    *) bad=$((bad + 1)); say "run rc $rc ($bad of 3)"; [ "$bad" -ge 3 ] && { QRC=1; break; } ;;
  esac
done
say "report"
env CUDA_VISIBLE_DEVICES= MODEL134=$MODEL134 PYTHONPATH=src $PY -B $E/s3_report.py --out "$O" > "$LOGS/report.log" 2>&1 || say "s3_report.py failed ($LOGS/report.log)"
say "job finished rc $QRC ($trips trips). GPU work over: delete .gpu-lock with delete_path."
exit "$QRC"
