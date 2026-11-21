#!/bin/bash
# 133: one GPU job = one run133.py process on one config list, under the machine-wide .gpu-lock and a memory watchdog
# (the pattern of experiments/131-off-term/queue.sh and 081 queue_v4.sh).
#   WSL, repo root:  bash experiments/133-ioi-wcm-sweep/job133.sh <configs.json> <job name>
# Waits (up to WAIT_MAX_S, default 3 h) while REPO/.gpu-lock exists or the GPU holds > 2500 MiB, then takes the lock
# atomically (noclobber) as "pid <pid> ioi-wcm <job>", removes it at exit only while it names this pid.
# Watchdog every 10 s: GPU memory.used > 15300 MiB or MemAvailable < 3 GB kills the job (exit 9).
# STOP_EPOCH (default today 07:30): run133 starts no config that would end after it; HARD (default 08:10) kills.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
export PATH="$PATH:/usr/lib/wsl/lib"
CFG=$1
NAME=$2
LOCK=$ROOT/.gpu-lock
O=$HERE/out
mkdir -p "$O/logs"
LOG=$O/logs/$NAME.log
STOP=${STOP_EPOCH:-$(date -d 'today 07:30' +%s)}
HARD=${HARD_EPOCH:-$(date -d 'today 08:10' +%s)}
WAIT_MAX=${WAIT_MAX_S:-10800}
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

t0=$(date +%s)
while true; do
  if [ "$(date +%s)" -ge "$STOP" ]; then say "$NAME: STOP_EPOCH passed while waiting: not started"; exit 7; fi
  if [ $(( $(date +%s) - t0 )) -gt "$WAIT_MAX" ]; then say "$NAME: waited > $WAIT_MAX s for the GPU: giving up"; exit 8; fi
  if [ ! -e "$LOCK" ]; then
    m=$(gpu_mib)
    if [ -n "$m" ] && [ "$m" -le 2500 ]; then
      if ( set -o noclobber; echo "pid $$ ioi-wcm $NAME started $(date '+%F %T')" > "$LOCK" ) 2> /dev/null; then break; fi
    fi
  fi
  sleep 30
done
trap 'grep -q "pid $$ " "$LOCK" 2> /dev/null && rm -f "$LOCK"' EXIT
say "$NAME: lock taken ($(cat "$LOCK")); GPU $(gpu_mib) MiB; configs $CFG; stop $(date -d "@$STOP" +%H:%M) hard $(date -d "@$HARD" +%H:%M)"
setsid env MODEL=gpt2-small WCM_VERSION=v4 PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_BS=64 OUT="$O" CONFIGS="$CFG" STOP_EPOCH="$STOP" \
  ${EXTRA_ENV:-} ./.venv/bin/python -X utf8 -u "$HERE/run133.py" >> "$LOG" 2>&1 < /dev/null &
JOB=$!
why=""
while kill -0 "$JOB" 2> /dev/null; do
  m=$(gpu_mib); r=$(avail_mib)
  echo "$(date +%s),$NAME,${m:-NA},${r:-NA}" >> "$O/watch.csv"
  if [ -n "$m" ] && [ "$m" -gt 15300 ]; then why="GPU $m MiB > 15300"
  elif [ -n "$r" ] && [ "$r" -lt 3072 ]; then why="MemAvailable $r MiB < 3 GB"
  elif [ "$(date +%s)" -ge "$HARD" ]; then why="hard deadline"; fi
  if [ -n "$why" ]; then
    say "KILL $NAME: $why"
    kill -TERM -- -"$JOB" 2> /dev/null; sleep 15; kill -KILL -- -"$JOB" 2> /dev/null
    break
  fi
  sleep 10
done
wait "$JOB" 2> /dev/null; rc=$?
[ -n "$why" ] && rc=9
say "$NAME: end rc $rc ($(( ($(date +%s) - t0) / 60 )) min incl. wait)"
exit "$rc"
