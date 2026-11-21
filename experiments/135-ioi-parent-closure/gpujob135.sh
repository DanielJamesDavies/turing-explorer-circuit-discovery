#!/bin/bash
# 135 GPU job: runs closure135.py under a watchdog. The .gpu-lock is taken and released by the agent (file tools),
# NOT by this script. Kills the job when GPU memory.used > 15300 MiB, MemAvailable < 3 GB, or at HARD_EPOCH.
#   WSL, repo root:  HARD_EPOCH=<unix s> bash experiments/135-ioi-parent-closure/gpujob135.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
export PATH="$PATH:/usr/lib/wsl/lib"
O=$HERE/out
mkdir -p "$O"
LOG=$O/gpujob.log
HARD=${HARD_EPOCH:?HARD_EPOCH needed}
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "start; GPU $(gpu_mib) MiB; hard $(date -d "@$HARD" +%H:%M)"
setsid env PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 MODEL=gpt2-small \
  WCM_VERSION=v4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${EXTRA_ENV:-} \
  ./.venv/bin/python -X utf8 -u "$HERE/closure135.py" >> "$O/run.log" 2>&1 < /dev/null &
JOB=$!
why=""
while kill -0 "$JOB" 2> /dev/null; do
  m=$(gpu_mib); r=$(avail_mib)
  echo "$(date +%s),${m:-NA},${r:-NA}" >> "$O/watch.csv"
  if [ -n "$m" ] && [ "$m" -gt 15300 ]; then why="GPU $m MiB > 15300"
  elif [ -n "$r" ] && [ "$r" -lt 3072 ]; then why="MemAvailable $r MiB < 3 GB"
  elif [ "$(date +%s)" -ge "$HARD" ]; then why="hard deadline"; fi
  if [ -n "$why" ]; then
    say "KILL: $why"
    kill -TERM -- -"$JOB" 2> /dev/null; sleep 15; kill -KILL -- -"$JOB" 2> /dev/null
    break
  fi
  sleep 10
done
wait "$JOB" 2> /dev/null; rc=$?
[ -n "$why" ] && rc=9
say "end rc $rc"
exit "$rc"
