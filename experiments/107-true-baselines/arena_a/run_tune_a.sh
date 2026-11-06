#!/usr/bin/env bash
# Arena A, A13 (2026-11-05): the WCM tuning sweep on the dev set (TUNING.md). No arguments needed for the default grid.
#   WSL, repo root:  bash experiments/107-true-baselines/arena_a/run_tune_a.sh
# Grid = tune_grid.tsv (pre-registered), dev set = tune_dev_targets.txt (the 9 non-vacuous panel targets; excluded from
# the pod's production targets through make_targets.py EXCLUDE=tune_exclude.txt). Per config: fits (062's driver via
# driver_tcz.py, MODE=main, v4, lambda 1e-3) in 4 processes (L1-5 targets together, each L8+ target alone: in one
# long process the card crept to 15.96 GB in the panel run, fresh L9 fits peak at 13.6 GB), then the arena A scorer
# (score_arena_a.py, held-out + home, nec a / b) on the config's circuits, then tune_summary.py (CPU) after every config.
# ONE GPU process at a time. Resumable per (config, target): the driver skips targets with a status row; a target whose
# fit a guard killed is listed in OUT/<config>/KILLED and counts as a fail (never retried automatically: delete that
# line to retry it). The scorer skips scored units.
# Guards (every 10 s while a GPU process runs; GPU memory logged every 30 s to out/tune/gpu_mem.csv):
#   card memory.used > GPU_MAX_MIB (15300)      -> kill that process, mark its current target KILLED (gpu_mem)
#   host MemAvailable < RAM_MIN_KB (4 GB)        -> kill that process, mark its current target KILLED (host_ram)
#   STOP_AT (HH:MM, default 08:30; the next one) -> kill and stop the whole run (not marked: a rerun resumes it)
# Before each GPU process: wait while card memory.used > GPU_BUSY_MIB (3000; idle desktop ~1.6 GB) = another process.
# Env: DRY_RUN=1 (print every command, run nothing), STOP_AT, CONFIGS (comma list, default: the grid in order),
#      CPU_SMOKE=1 (the plumbing check: config r0, 1.z.21179 only, STEPS=2, CPU, threads 4, out/tune_cpu_smoke).
set -u
cd "$(dirname "$0")/../../.."
A=experiments/107-true-baselines/arena_a
P=$A/out/pilot
Q=$A/out/panel
PY=.venv/bin/python
GRID=$A/tune_grid.tsv
DEV=$A/tune_dev_targets.txt
DRY_RUN=${DRY_RUN:-0}
CPU_SMOKE=${CPU_SMOKE:-0}
GPU_MAX_MIB=${GPU_MAX_MIB:-15300}
GPU_BUSY_MIB=${GPU_BUSY_MIB:-3000}
RAM_MIN_KB=${RAM_MIN_KB:-4194304}
STOP_AT=${STOP_AT:-08:30}
STOP_EPOCH=$(date -d "today $STOP_AT" +%s)
[ "$STOP_EPOCH" -le "$(date +%s)" ] && STOP_EPOCH=$(date -d "tomorrow $STOP_AT" +%s)
if [ "$CPU_SMOKE" = 1 ]; then
  T=$A/out/tune_cpu_smoke; CONFIGS=${CONFIGS:-r0}; DEVKEYS="1.z.21179"; DEV_CUDA=""; THREADS=4; EXTRA="STEPS=2"
else
  T=$A/out/tune; CONFIGS=${CONFIGS:-$(grep -v '^#' "$GRID" | cut -f1 | paste -sd,)}; DEVKEYS=$(grep -v '^#' "$DEV" | xargs)
  DEV_CUDA=0; THREADS=4; EXTRA=""
fi
mkdir -p "$T/logs"
say() { echo "[tune_a] $(date +%H:%M:%S) $*" | tee -a "$T/run.log"; }
late() { [ "$(date +%s)" -ge "$STOP_EPOCH" ]; }
gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' '; }
ram_kb() { awk '/MemAvailable/ {print $2}' /proc/meminfo; }
ENVV="PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$DEV_CUDA PYTHONDONTWRITEBYTECODE=1 ARENA_A_BOS=1 ARENA_A_SCOPE=a1 \
FAMILY=gpt2-small-tcz DATA=data_public/gpt2-small SAE_PRELOAD=0 DEEP_BS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS"
FIT="STORES=$P/datapass CTX_DIR=$Q/rec/ctx MODE=main LAM=1e-3 WCM_VERSION=v4 ORDER=list SKIP_SPEC=1 CURVE_STRIDE=20"
say "start: configs $CONFIGS | dev $DEVKEYS | stop $(date -d @$STOP_EPOCH '+%a %H:%M') | DRY_RUN=$DRY_RUN CPU_SMOKE=$CPU_SMOKE"

grid_env() { awk -F'\t' -v n="$1" '!/^#/ && $1 == n {print $2}' "$GRID"; }
done_keys() { grep -ho '"seed": "[^"]*"' "$1"/main/status.shard*.jsonl 2> /dev/null | cut -d'"' -f4 | sort -u; }
killed_keys() { cut -f1 "$1/KILLED" 2> /dev/null | sort -u; }

wait_gpu_free() {   # another process on the card: wait (logged every 5 min), or give up at STOP_AT
  [ -z "$DEV_CUDA" ] && return 0
  local n=0 m
  while m=$(gpu_mem); [ -n "$m" ] && [ "$m" -gt "$GPU_BUSY_MIB" ]; do
    late && return 1
    [ $((n % 30)) -eq 0 ] && say "waiting: card memory.used $m MiB > $GPU_BUSY_MIB (another process)"
    n=$((n + 1)); sleep 10
  done
  return 0
}

# guarded NAME LOG command... -> 0 ok, 1 failed, 2 killed by a guard (GUARD_WHY), 3 STOP_AT
guarded() {
  local name=$1 log=$2; shift 2
  GUARD_WHY=""
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry] $*  > $log"; return 0; fi
  late && return 3
  wait_gpu_free || return 3
  echo "$name" > "$T/.step"
  local t0=$(date +%s)
  setsid env $ENVV "$@" > "$log" 2>&1 &
  local pid=$! tick=0 rc m r why=""
  while kill -0 "$pid" 2> /dev/null; do
    sleep 10; tick=$((tick + 1))
    m=$(gpu_mem); r=$(ram_kb)
    [ $((tick % 3)) -eq 0 ] && echo "$(date +%s),$name,${m:-NA},$r" >> "$T/gpu_mem.csv"
    if [ -n "$DEV_CUDA" ] && [ -n "$m" ] && [ "$m" -gt "$GPU_MAX_MIB" ]; then why="gpu_mem ${m}MiB"
    elif [ "$r" -lt "$RAM_MIN_KB" ]; then why="host_ram ${r}kB"
    elif late; then why="stop_at"; fi
    if [ -n "$why" ]; then
      say "GUARD $name: $why -> kill"
      kill -TERM -- -"$pid" 2> /dev/null; sleep 10; kill -KILL -- -"$pid" 2> /dev/null
      break
    fi
  done
  wait "$pid" 2> /dev/null; rc=$?
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$t0" "$(date +%s)" "$rc" "${why:-}" >> "$T/steps.tsv"
  echo idle > "$T/.step"
  [ "$why" = stop_at ] && return 3
  [ -n "$why" ] && { GUARD_WHY=$why; return 2; }
  [ "$rc" -eq 0 ] && return 0 || return 1
}

fit_group() {   # fit_group CONFIG GROUP KEYS... (one process; on a guard kill: mark the current target, start again)
  local cfg=$1 grp=$2; shift 2
  local O=$T/$cfg/fit genv rest k cur rc tries=0
  genv=$(grid_env "$cfg"); [ "$genv" = "-" ] && genv=""
  mkdir -p "$O"
  while :; do
    rest=""
    for k in "$@"; do
      done_keys "$O" | grep -qx "$k" && continue
      killed_keys "$T/$cfg" | grep -qx "$k" && continue
      rest="$rest $k"
    done
    [ -z "$rest" ] && return 0
    tries=$((tries + 1)); [ "$tries" -gt $(( $# + 1 )) ] && { say "$cfg/$grp: giving up"; return 1; }
    echo $rest | tr ' ' '\n' > "$T/$cfg/targets.$grp.txt"
    say "fit $cfg/$grp:$rest"
    guarded "fit_${cfg}_$grp" "$T/logs/fit_${cfg}_$grp.$tries.log" $FIT $genv $EXTRA OUT=$O \
      TARGETS=$T/$cfg/targets.$grp.txt $PY -X utf8 -u $A/driver_tcz.py
    rc=$?
    [ "$DRY_RUN" = 1 ] && return 0
    [ "$rc" -eq 3 ] && return 3
    cur=""
    for k in $rest; do done_keys "$O" | grep -qx "$k" || { cur=$k; break; }; done
    if [ -n "$cur" ] && [ "$rc" -ne 0 ]; then       # the process died on target $cur: count it as a fail
      printf "%s\t%s\t%s\n" "$cur" "${GUARD_WHY:-exit $rc}" "$(date -Is)" >> "$T/$cfg/KILLED"
      say "KILLED $cfg $cur (${GUARD_WHY:-exit $rc})"
    elif [ -n "$cur" ]; then
      printf "%s\t%s\t%s\n" "$cur" "no status row after a clean exit" "$(date -Is)" >> "$T/$cfg/KILLED"
    fi
  done
}

score_cfg() {   # the arena A scorer on the config's circuits (dev keys, ho + home)
  local cfg=$1 spec keys
  if [ "$(grid_env "$cfg")" = REUSE ]; then spec="l1e-3=$Q/wcm_l1e-3,l1e-3=$Q/wcm_l1e-3_rerun"
  else spec="l1e-3=$T/$cfg/fit"; fi
  keys=$(echo $DEVKEYS | tr ' ' ',')
  say "score $cfg"
  guarded "score_$cfg" "$T/logs/score_$cfg.log" ARENA_A_CTX_DIRS=$Q/rec/ctx $PY -X utf8 -u $A/score_arena_a.py \
    --scope a1 --out "$T/$cfg/score" --wcm "$spec" --keys "$keys" --threads $THREADS --stop-epoch "$STOP_EPOCH"
}

summary() {
  [ "$DRY_RUN" = 1 ] && { echo "  [dry] $PY -B $A/tune_summary.py --tune $T --grid $GRID --dev ${DEVKEYS// /,}"; return; }
  CUDA_VISIBLE_DEVICES= $PY -B $A/tune_summary.py --tune "$T" --grid "$GRID" --dev "${DEVKEYS// /,}" \
    > "$T/logs/summary.log" 2>&1 || say "FAILED summary (see logs/summary.log)"
}

for cfg in ${CONFIGS//,/ }; do
  [ -z "$(grid_env "$cfg")" ] && { say "unknown config $cfg (not in $GRID)"; exit 1; }
  if [ "$(grid_env "$cfg")" != REUSE ]; then
    shallow=""; deep=""
    for k in $DEVKEYS; do [ "${k%%.*}" -ge 8 ] && deep="$deep $k" || shallow="$shallow $k"; done
    [ -n "$shallow" ] && { fit_group "$cfg" shallow $shallow; [ $? -eq 3 ] && { say "STOP_AT"; summary; exit 0; }; }
    for k in $deep; do
      fit_group "$cfg" "$k" "$k"; [ $? -eq 3 ] && { say "STOP_AT"; summary; exit 0; }
    done
  fi
  score_cfg "$cfg"; rc=$?
  [ "$rc" -eq 3 ] && { say "STOP_AT"; summary; exit 0; }
  [ "$rc" -ne 0 ] && say "score $cfg exit $rc (${GUARD_WHY:-})"
  summary
done
say "DONE"
[ "$DRY_RUN" = 1 ] || touch "$T/TUNE_DONE"
