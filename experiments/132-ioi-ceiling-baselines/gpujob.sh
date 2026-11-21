#!/bin/bash
# Exp 132 GPU job wrapper (local RTX 5070 Ti 16 GB, WSL): ONE GPU process machine-wide, shared with exp 133.
#
#   bash experiments/132-ioi-ceiling-baselines/gpujob.sh NAME CMD...      (from the repo root)
#
# - waits (polling every 20 s) while REPO/.gpu-lock exists, then writes its own lock "pid <pid> ioi-base <NAME>";
# - refuses to START after LAST_START (default today 07:30 local) and kills the job at HARD_STOP (default 08:10);
# - watchdog every 10 s: GPU memory.used > 15300 MiB or host MemAvailable < 3 GB kills the job's process group;
# - removes the lock at exit only while it still names this pid.
# Log: OUT/logs/NAME.log ; one line per event in OUT/gpujob.log.  Exit: the job's rc; 9 watchdog; 7 deadline; 8 refused.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
export PATH="$PATH:/usr/lib/wsl/lib"
O=$HERE/out
LOCK=$ROOT/.gpu-lock
NAME=$1; shift
LAST_START=${LAST_START:-$(date -d 'today 07:30' +%s)}
HARD_STOP=${HARD_STOP:-$(date -d 'today 08:10' +%s)}
GPU_MAX=15300
mkdir -p "$O/logs"
say() { echo "[$(date '+%F %T')] $NAME: $*" | tee -a "$O/gpujob.log"; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }

while :; do
  if [ "$(date +%s)" -ge "$LAST_START" ]; then say "refused: past the last start time"; exit 8; fi
  if [ ! -e "$LOCK" ]; then
    m=$(gpu_mib)
    if [ -n "$m" ] && [ "$m" -gt 2500 ]; then
      say "no lock but GPU holds $m MiB: waiting"
    else
      echo "pid $$ ioi-base $NAME started $(date '+%F %T')" > "$LOCK"
      sleep 1
      grep -q "pid $$ " "$LOCK" 2> /dev/null && break
    fi
  fi
  sleep 20 & wait $!
done
trap 'grep -q "pid $$ " "$LOCK" 2> /dev/null && rm -f "$LOCK"' EXIT
say "lock taken; start: $*"
log=$O/logs/$NAME.log
echo "[$(date '+%F %T')] \$ $*" >> "$log"
setsid "$@" >> "$log" 2>&1 < /dev/null &
JOB=$!
trap '[ -n "$JOB" ] && kill -TERM -- -"$JOB" 2> /dev/null; sleep 5; kill -KILL -- -"$JOB" 2> /dev/null; say "signal"; exit 143' INT TERM HUP
why=""; hard=0
while kill -0 "$JOB" 2> /dev/null; do
  m=$(gpu_mib); r=$(avail_mib)
  echo "$(date +%s),$NAME,${m:-NA},${r:-NA}" >> "$O/watch.csv"
  if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX"
  elif [ -n "$r" ] && [ "$r" -lt 3072 ]; then why="MemAvailable $r MiB < 3 GB"
  elif [ "$(date +%s)" -ge "$HARD_STOP" ]; then why="hard stop"; hard=1; fi
  if [ -n "$why" ]; then
    say "KILL: $why"
    kill -TERM -- -"$JOB" 2> /dev/null; sleep 10; kill -KILL -- -"$JOB" 2> /dev/null
    break
  fi
  sleep 10 & wait $!
done
wait "$JOB" 2> /dev/null; rc=$?
JOB=""
if [ -n "$why" ]; then [ "$hard" = 1 ] && { say "exit 7"; exit 7; }; say "exit 9"; exit 9; fi
say "done rc $rc"
exit "$rc"
