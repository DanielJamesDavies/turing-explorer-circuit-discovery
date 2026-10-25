#!/bin/bash
# Pause (SIGSTOP) the golden_tree.py process while WSL MemAvailable < LOW GB, resume (SIGCONT) above HIGH GB, so the
# CPU golden check never pushes another agent's GPU job into an OOM kill. Logs one line per change + every 5 min.
#   bash memwatch.sh <logfile>      (exits when no golden_tree.py process is left)
#   KILL (GB): below it the golden process is killed outright (another agent's job must never be the OOM victim).
LOW=${LOW:-20}; HIGH=${HIGH:-22}; KILL=${KILL:-9}
log=$1; paused=0; n=0; waited=0
while :; do
  pid=""                                     # the python process itself, not a shell whose command line names it
  for p in $(pgrep -f "[g]olden_tree.py"); do
    case $(ps -o comm= -p "$p" 2> /dev/null) in python*) pid=$p; break ;; esac
  done
  if [ -z "$pid" ]; then
    [ "$waited" -lt 30 ] && { waited=$((waited + 1)); sleep 2; continue; }   # not started yet
    echo "$(date +%T) no golden process; exit" >> "$log"; exit 0
  fi
  waited=30
  av=$(awk '/MemAvailable/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
  if [ "$av" -lt "$KILL" ]; then kill -9 "$pid"; echo "$(date +%T) avail ${av}G < ${KILL}G: KILLED $pid" >> "$log"; exit 1; fi
  if [ "$paused" = 0 ] && [ "$av" -lt "$LOW" ]; then kill -STOP "$pid"; paused=1; echo "$(date +%T) avail ${av}G < ${LOW}G: PAUSED $pid" >> "$log"
  elif [ "$paused" = 1 ] && [ "$av" -ge "$HIGH" ]; then kill -CONT "$pid"; paused=0; echo "$(date +%T) avail ${av}G: resumed $pid" >> "$log"
  fi
  [ $((n % 60)) = 0 ] && echo "$(date +%T) avail ${av}G paused=$paused rss_golden=$(ps -o rss= -p "$pid" 2> /dev/null | awk '{printf "%.1fG", $1/1048576}')" >> "$log"
  n=$((n + 1)); sleep 2
done
