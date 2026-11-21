#!/bin/bash
# Exp 128 night runner (local RTX 5070 Ti 16 GB, WSL ~30 GB RAM, ONE GPU process at a time): pod preflight for the
# 2026-11-21 arena A run. Exercises the 2026-11-17 run_pod.sh changes for real at small scale, times Ge on the GPU,
# fits WCM a1 on the exp 126 OWT 131k corpus, and measures what the host RAM of a deep fit is used for (README).
#
#   WSL, repo root:  DRY_RUN=1 bash experiments/128-pod-preflight/run_night.sh        # print the plan, change nothing
#                    STOP_EPOCH=$(date -d 'tomorrow 07:45' +%s) nohup bash experiments/128-pod-preflight/run_night.sh \
#                        > experiments/128-pod-preflight/out/run_night.out 2>&1 &
#
# STAGES, in this order (each skipped when OUT/state/<stage>.done exists; inside a stage run_pod.sh / the scorers skip
# finished fits / rows, so a rerun resumes; a stage whose dependency is not done is skipped):
#   smoke         arena A run_smoke_local.sh -> run_pod.sh P4 P5 P6 P4x P6x P4h P6b P7 into a FRESH SMOKE_WORK, from
#                 P4, keys 3.tc.13057 3.z.23269 1.tc.23034 (2 tc, so AA_DF_PAR=3 runs 2 Dunefsky processes with work),
#                 lambdas 1e-3,1.6e-2, AA_WCM_P=1; Ge / Dunefsky / Ge recovery on the CPU (their envs are CPU builds)
#   smoke_check   (CPU, always) check_smoke.py -> OUT/smoke_check.md: every stage rc 0, P4x / P6x outputs,
#                 ge_recovery.x.*, df_par processes, summaries
#   ge_gpu        run_ge.py (unchanged) in ~/venvs/ge107gpu (= ge107 + tc107gpu's torch 2.8.0+cu128, built offline by
#                 build_ge107gpu.sh) on 3.tc.13057 7.z2.18144 9.tc.17800, K 2, 30 tau, --device cuda, one process per
#                 target; then ge_gpu_report.py against the CPU smoke out/ge_smoke_up -> OUT/ge_gpu.md. A no-op
#                 ("skipped: needs download approval") when the env cannot be built offline
#   a1_fit_shallow  WCM a1 (z + z2 + tc), lambda 1e-3, CHOSEN.env, on exp 126's 8 owt131k shallow tc targets (run_pod.sh
#                 P5 only, one launch)                                        -> OUT/owt_a1/fit_shallow
#   a1_score_shallow  score_arena_a.py --scope a1 --selftest on them          -> OUT/owt_a1/score_shallow
#   a1_fit_deep   the same on 126's 8 deep targets that fitted in a2 (L5-8), ONE run_pod.sh call per key, in ascending
#                 depth (AA_REDO=P5), so a RAM trip loses the deepest only   -> OUT/owt_a1/fit_deep
#   a1_score_deep score the deep keys whose fit finished                     -> OUT/owt_a1/score_deep
#   a1_compare    (CPU, always) compare_a1_a2.py: a1 here vs 126's a2 rows   -> OUT/a1_vs_a2.md
#   ram_l1e-3 ram_l2.5e-4 ram_s100   ONE deep a2 fit each (N128_RAM_KEY [9.tc.21268], the L9 target that tripped 126's
#                 watchdog), lambda 1e-3, 2.5e-4, then 1e-3 at STEPS=100; probe_driver.py (driver_tcz.py unchanged +
#                 phase / step / memory markers) under ram_monitor.py (2 s samples; kills ITS fit only at MemAvailable
#                 < 3 GB, GPU > N128_GPU_MAX_MIB, or N128_PROBE_MAX_MIN; the night goes on)   -> OUT/ram_probe/<name>
#   ram_report    (CPU, always) ram_report.py                                -> OUT/ram_report.md
#   t7_full       083 suite (chain T7: ref, attr, fitted, then table) at production settings on ONE TuringLLM target
#                 (N128_T7_KEY [3.attn.11307]) against its production v4 circuits copied from the 11-07 pod archives
#                 (t7_prep.py: OUT/t7_full/ref); 083/run_pod.sh unchanged, GPUS=0 P=1. Started only before
#                 N128_T7_LATEST_EPOCH [STOP_EPOCH - 2h15, i.e. 05:30], killed at N128_T7_KILL_EPOCH [STOP_EPOCH + 15 min]
#   t7_summary    (CPU, always) t7_summary.py                                -> OUT/t7_full/T7_SUMMARY.md
#
# Safety: refuses to start when REPO/.gpu-lock exists, the GPU already holds > N128_GPU_IDLE_MIB [2500] MiB, or
# OUT/ABORTED exists; writes its own .gpu-lock ("pid $$") and removes it at exit only while it still names this pid.
# Watchdog every N128_WATCH_S [10] s (OUT/watch.csv): GPU memory.used > N128_GPU_MAX_MIB [15300] or host MemAvailable
# < N128_RAM_MIN_GB [3] GB kills the running job's process group, writes OUT/ABORTED and stops the night (exit 9) --
# except for the stages in N128_SOFT [none] (comma list), where the trip kills only that job, the stage counts as
# failed and the night goes on; the ram_* stages always behave so (their own monitor).
# STOP_EPOCH: no new stage / key once date +%s >= it (the CPU report stages excepted); passed to run_pod.sh (soft stop;
# its hard stop AA_HARD_GRACE = N128_HARD_GRACE [1800] s later) and the scorers (--stop-epoch); every job is killed at
# STOP_EPOCH + N128_HARD_GRACE. N128_STAGES: run only these stages (comma list); N128_SKIP: skip these.
# Logs: OUT/night.log (one line per stage start / end), job logs OUT/logs/<job>.log.
# Exit: 0 every stage done; 1 a stage failed; 7 STOP_EPOCH cut the plan; 8 refused; 9 aborted by the watchdog.
set -uo pipefail
shopt -s nullglob
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
export PATH="$PATH:/usr/lib/wsl/lib"
E=experiments/128-pod-preflight
O=$E/out
A=experiments/107-true-baselines/arena_a
E126=experiments/126-wcm-overshoot
E83=experiments/083-v1-comparison
PY=./.venv/bin/python
DRY_RUN=${DRY_RUN:-0}
STOP=${STOP_EPOCH:-99999999999}
GPU_MAX=${N128_GPU_MAX_MIB:-15300}
GPU_IDLE=${N128_GPU_IDLE_MIB:-2500}
RAM_MIN_GB=${N128_RAM_MIN_GB:-3}
EVERY=${N128_WATCH_S:-10}
HARD_GRACE=${N128_HARD_GRACE:-1800}
SOFT=${N128_SOFT:-}
ONLY=${N128_STAGES:-}
SKIP=${N128_SKIP:-}
LOCK=$ROOT/.gpu-lock
STATE=$O/state
LOGS=$O/logs
if [ -n "${STOP_EPOCH:-}" ]; then
  HARD_DEADLINE=$((STOP_EPOCH + HARD_GRACE))
  T7_LATEST=${N128_T7_LATEST_EPOCH:-$((STOP_EPOCH - 8100))}
  T7_KILL=${N128_T7_KILL_EPOCH:-$((STOP_EPOCH + 900))}
else
  HARD_DEADLINE=99999999999; T7_LATEST=${N128_T7_LATEST_EPOCH:-99999999999}; T7_KILL=${N128_T7_KILL_EPOCH:-99999999999}
fi

# ---------------------------------------------------------------------------------------------- inputs
# smoke (arena A, run_smoke_local.sh): a fresh work dir; 2 tc keys so AA_DF_PAR=3 gives 2 Dunefsky processes with work
SMOKE_W=${N128_SMOKE_WORK:-$A/out/smoke_pod_1117}
SMOKE_KEYS=${N128_SMOKE_KEYS:-3.tc.13057,3.z.23269,1.tc.23034}
SMOKE_LAMS=${N128_SMOKE_LAMBDAS:-1e-3,1.6e-2}
SMOKE_EXT_BR=100                                          # run_smoke_local.sh's AA_DUNEFSKY_EXT_BRANCHES / _K (checked)
SMOKE_EXT_K=1
DF_PAR=${N128_DF_PAR:-3}
# ge_gpu
GE_GPU=${N128_GE_GPU_ENV:-$HOME/venvs/ge107gpu}
GE_KEYS=3.tc.13057,7.z2.18144,9.tc.17800                  # the 2026-11-05 CPU smoke's targets (out/ge_smoke_up, K 2, 30 tau)
GE_CPU_REF=$A/out/ge_smoke_up
# exp 126 OWT 131k (night 2, 2026-11-17): the shallow tc targets of its a2 fits (layer 0 dropped: no tc upstream in a2)
# and the 8 deep (L5-8) targets whose a2 fit finished (score_deep57), in ascending depth
OWT_CORPUS=$E126/out/corpus_owt131k
OWT_STO=$E126/out/owt131k/datapass
OWT_TGT=$E126/out/owt131k/targets
A1_SHALLOW=1.tc.17037,1.tc.14335,2.tc.15605,2.tc.3416,3.tc.11202,3.tc.9976,4.tc.6575,4.tc.16695
A1_DEEP=5.tc.1481,5.tc.8088,6.tc.9000,6.tc.24021,7.tc.3375,7.tc.15758,8.tc.22944,8.tc.14821
A2_ROWS=$E126/out/score/owt131k/rows.jsonl,$E126/out/score/owt131k_deep57/rows.jsonl
TUNE=$A/out/tune/CHOSEN.env
# ram_probe
RAM_KEY=${N128_RAM_KEY:-9.tc.21268}
RAM_SCOPE=${N128_RAM_SCOPE:-a2}
PROBE_DEEP_BS=${N128_PROBE_DEEP_BS:-1}                    # = 126's (every local fit); unset = the config's 4
PROBE_MAX_MIN=${N128_PROBE_MAX_MIN:-35}
PROBE_MIN_AVAIL_MIB=${N128_PROBE_MIN_AVAIL_MIB:-3072}
# t7_full
T7_KEY=${N128_T7_KEY:-3.attn.11307}
T7_DIR=$O/t7_full
T7_CTX=experiments/109-pod-run-analysis/out/archives/pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh
T7_STAGES="ref attr fitted"

BASEENV="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ARENA_A_BOS=1 FAMILY=gpt2-small-tcz SAE_PRELOAD=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WCM_VERSION=v4 DEEP_BS=1 NVIDIA_TF32_OVERRIDE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID"
FIT_COMMON="AA_GPUS=0 AA_PACK=0 AA_SCOPES=a1 AA_LAMBDAS=1e-3 AA_FROM=P5 STOP_AT=P5 AA_BASELINE_DEVICE=cpu \
AA_TUNE_ENV=$TUNE AA_N_SEQ=0 AA_WCM_P=1 DEEP_BS=1 NVIDIA_TF32_OVERRIDE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
AA_HARD_GRACE=$HARD_GRACE AA_TARGETS_DIR=$OWT_TGT AA_STORES=$OWT_STO AA_DATA_SRC=$OWT_CORPUS"
if [ -n "${STOP_EPOCH:-}" ]; then FIT_COMMON="$FIT_COMMON STOP_EPOCH=$STOP_EPOCH"; else FIT_COMMON="$FIT_COMMON AA_MAX_H=0"; fi

# ---------------------------------------------------------------------------------------------- helpers
ts() { date '+%F %T'; }
say() {
  local line="[n128 $(date +%H:%M:%S)] $*"
  echo "$line"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$O"; echo "[$(ts)] $*" >> "$O/night.log"; }
}
late() { [ "$(date +%s)" -ge "$STOP" ]; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
nlines() { if [ -f "$1" ]; then grep -c . "$1"; else echo 0; fi; }
ncsv() { echo "$1" | tr ',' '\n' | grep -c .; }
fmin() { awk -v x="$1" 'BEGIN { printf "%.0f", x }'; }
hm() { if [ "$1" -ge 99999999999 ]; then echo none; else date -d "@$1" '+%a %H:%M'; fi; }
inlist() { [[ ",$2," == *",$1,"* ]]; }
sel() { { [ -z "$ONLY" ] || inlist "$1" "$ONLY"; } && ! inlist "$1" "$SKIP"; }
tune_kv() {   # CHOSEN.env as run_pod.sh's tune_env reads it (whitelisted keys, plain values; '#' comments dropped)
  tr -d '\r' < "$TUNE" 2> /dev/null | sed 's/#.*//' | tr -d ' \t' | grep -E '^(AA_RANK_W|NEC_W|STEPS|AA_TUNE_CONFIG)=[A-Za-z0-9_.+-]*$' | paste -sd' ' -
}

JOB=""
STAGE_NOW=""
JOB_DEADLINE=""
job() {   # job NAME CMD...: the command in its own process group, watchdog-guarded; rc = its rc, 9 = watchdog trip,
  #         7 = killed at the hard deadline (STOP_EPOCH + N128_HARD_GRACE, or JOB_DEADLINE when set)
  local name=$1; shift
  local log=$LOGS/$name.log m r why="" rc dl=${JOB_DEADLINE:-$HARD_DEADLINE} hard=0
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ $*   >> $log"; return 0; fi
  mkdir -p "$LOGS"
  echo "[$(ts)] \$ $*" >> "$log"
  setsid "$@" >> "$log" 2>&1 < /dev/null &
  JOB=$!
  while kill -0 "$JOB" 2> /dev/null; do
    m=$(gpu_mib); r=$(avail_mib)
    echo "$(date +%s),$name,${m:-NA},${r:-NA}" >> "$O/watch.csv"
    if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX MiB"
    elif [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then why="host MemAvailable $r MiB < $RAM_MIN_GB GB"
    elif [ "$(date +%s)" -ge "$dl" ]; then why="hard deadline $(hm "$dl") reached"; hard=1; fi
    if [ -n "$why" ]; then
      say "KILL $name: $why: killing its process group"
      [ "$hard" = 1 ] || inlist "$STAGE_NOW" "$SOFT" || echo "$(ts) $name: $why" > "$O/ABORTED"
      kill -TERM -- -"$JOB" 2> /dev/null; sleep 15; kill -KILL -- -"$JOB" 2> /dev/null
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

declare -A DEPS=(
  [smoke_check]="" [a1_score_shallow]="a1_fit_shallow" [t7_summary]=""
)
declare -A EST=()
FAILS=0
CUT=0
ETA=0
stage() {   # stage NAME FUNC [always]: markers, deps, STOP_EPOCH, night.log; "always" = runs every time (CPU reports)
  local name=$1 fn=$2 always=${3:-} d rc t0
  sel "$name" || return 0
  if [ -z "$always" ] && [ -f "$STATE/$name.done" ]; then
    say "$name: done earlier ($(cat "$STATE/$name.done"))"; return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    ETA=$(awk -v a="$ETA" -v b="${EST[$name]:-1}" 'BEGIN { print a + b }')
    say "$name: PLAN (~$(fmin "${EST[$name]:-1}") min; cumulative ~$(fmin "$ETA") min)$([ -n "${DEPS[$name]:-}" ] && echo "; needs ${DEPS[$name]}")$(inlist "$name" "$SOFT" && echo '; SOFT watchdog')"
    STAGE_NOW=$name "$fn"
    return 0
  fi
  for d in ${DEPS[$name]:-}; do
    if [ ! -f "$STATE/$d.done" ]; then say "$name: skipped (dependency $d not done)"; return 0; fi
  done
  if [ -z "$always" ] && late; then say "$name: STOP_EPOCH reached: not started"; CUT=1; return 7; fi
  say "START $name (estimate ~$(fmin "${EST[$name]:-1}") min)"
  t0=$(date +%s)
  STAGE_NOW=$name
  "$fn"; rc=$?
  STAGE_NOW=""
  say "END $name rc $rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if [ "$rc" = 0 ]; then mkdir -p "$STATE"; ts > "$STATE/$name.done"
  elif [ "$rc" = 9 ] && ! inlist "$name" "$SOFT"; then
    say "watchdog abort: stopping the night (exit 9; read $O/ABORTED, move it away to resume)"; exit 9
  elif [ "$rc" = 9 ]; then say "  $name: watchdog trip in a N128_SOFT stage: only this job was killed; the night goes on"
    FAILS=$((FAILS + 1))
  elif [ "$rc" = 7 ] || late; then CUT=1                 # cut by STOP_EPOCH / the hard deadline: resumable
  else FAILS=$((FAILS + 1)); fi
  return 0
}

# ---------------------------------------------------------------------------------------------- smoke
u_smoke() {
  local envs="SMOKE_WORK=$SMOKE_W SMOKE_KEYS=$SMOKE_KEYS SMOKE_LAMBDAS=$SMOKE_LAMS AA_FROM=P4 AA_WCM_P=1 AA_DF_PAR=$DF_PAR \
SMOKE_GPU_IDLE_MIB=$GPU_IDLE SMOKE_RAM_MIN_GB=$RAM_MIN_GB SMOKE_GPU_MAX_MIB=$GPU_MAX AA_HARD_GRACE=$HARD_GRACE${STOP_EPOCH:+ STOP_EPOCH=$STOP_EPOCH}"
  if [ "$DRY_RUN" = 1 ]; then
    echo "    \$ env $envs bash $A/run_smoke_local.sh   >> $LOGS/smoke.log"
    echo "      its dry run (run_pod.sh plan; P4x / P6x / df_par lines):"
    local out
    out=$(env $envs DRY_RUN=1 bash "$A/run_smoke_local.sh" 2>&1)
    echo "$out" | grep -E 'P4x|P6x|--shard|ge_recovery\.x|interleaved|a2_skipped|lambdas|REFUSED|refused|missing|dry run rc' \
      | sed 's/^/        /' | cut -c1-600
    # the names run_smoke_local.sh passes must be the ones run_pod.sh reads (checked in the printed plan)
    echo "$out" | grep -q -- "--k 1 --branches $SMOKE_EXT_BR --iters 5 --shard 0/$DF_PAR" \
      && echo "      CHECK ok: P4x = Dunefsky K $SMOKE_EXT_K x $SMOKE_EXT_BR branches, $DF_PAR shards" \
      || echo "      CHECK FAILED: no P4x line with '--k 1 --branches $SMOKE_EXT_BR --iters 5 --shard 0/$DF_PAR'"
    echo "$out" | grep -q "ge_recovery.x.s0.jsonl" && echo "      CHECK ok: P6x writes score/all/ge_recovery.x.s0.jsonl" \
      || echo "      CHECK FAILED: no P6x ge_recovery.x line"
    echo "$out" | grep -q "ORDER=interleaved" && echo "      CHECK ok: P5 ORDER=interleaved, P=1" \
      || echo "      CHECK FAILED: P5 not interleaved"
    return 0
  fi
  if [ -e "$SMOKE_W" ] && [ ! -f "$STATE/smoke.started" ]; then
    say "  $SMOKE_W exists and this runner did not start it: not a fresh smoke dir (set N128_SMOKE_WORK)"; return 1
  fi
  mkdir -p "$STATE"; ts > "$STATE/smoke.started"
  job smoke env $envs bash "$A/run_smoke_local.sh"
  local rc=$?
  # rc 9 here = the smoke's OWN watchdog (or ours) killed run_pod.sh: the same rule as ours (ABORTED unless soft)
  if [ "$rc" = 9 ] && [ ! -f "$O/ABORTED" ] && ! inlist smoke "$SOFT"; then
    echo "$(ts) smoke: run_smoke_local.sh's watchdog aborted run_pod.sh ($(cat "$SMOKE_W/ABORTED" 2> /dev/null))" > "$O/ABORTED"
  fi
  return "$rc"
}
u_smoke_check() {
  job smoke_check env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -I -B $E/check_smoke.py --work $SMOKE_W \
    --lambdas $SMOKE_LAMS --ext-branches $SMOKE_EXT_BR --ext-k $SMOKE_EXT_K --out $O/smoke_check.md
}

# ---------------------------------------------------------------------------------------------- ge_gpu
u_ge_gpu() {
  if [ "$DRY_RUN" = 1 ]; then
    if [ -x "$GE_GPU/bin/python" ]; then echo "    (env $GE_GPU exists: $(cat "$GE_GPU/BUILT_BY_128" 2> /dev/null || echo 'not built by 128'))"
    else echo "    \$ bash $E/build_ge107gpu.sh   # offline copy (ge107 + tc107gpu's torch); rc 3 = needs a download -> no-op"; fi
  elif [ ! -x "$GE_GPU/bin/python" ]; then
    job ge_build bash "$E/build_ge107gpu.sh"; local brc=$?
    if [ "$brc" = 3 ]; then
      say "  ge_gpu skipped: needs download approval (torch 2.8.0+cu128 for cp310 is not on this machine; README)"
      echo "skipped: needs download approval" > "$O/ge_gpu.md"; return 0
    fi
    [ "$brc" = 0 ] || return "$brc"
  fi
  local k rc=0
  for k in ${GE_KEYS//,/ }; do                            # one process per target (their gate closure leaks VRAM)
    if [ "$DRY_RUN" != 1 ] && late; then return 7; fi
    job "ge_gpu_$k" env PATH=$GE_GPU/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib VIRTUAL_ENV=$GE_GPU PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
      "$GE_GPU/bin/python" -I -B $A/run_ge.py --targets $A/out/targets_smoke --out $O/ge_gpu --keys "$k" --k 2 --n-tau 30 \
      --device cuda --max-rss-gb 9 || { rc=$?; [ "$rc" = 9 ] && return 9; [ "$rc" = 7 ] && return 7; }
  done
  job ge_gpu_report env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -I -B $E/ge_gpu_report.py --gpu $O/ge_gpu \
    --cpu $GE_CPU_REF --out $O/ge_gpu.md || rc=1
  return "$rc"
}

# ---------------------------------------------------------------------------------------------- owt_a1
a1_fit() {   # a1_fit NAME KEYS [extra env]: run_pod.sh P5, scope a1, lambda 1e-3, own AA_WORK = OUT/owt_a1/NAME
  local name=$1 keys=$2 extra=${3:-}
  local W=$O/owt_a1/$name
  local envs="$FIT_COMMON AA_WORK=$W AA_KEYS=$keys${extra:+ $extra}"
  if [ "$DRY_RUN" = 1 ]; then
    echo "    \$ env $envs bash $A/run_pod.sh   >> $LOGS/a1_$name.log"
    env $envs DRY_RUN=1 bash "$A/run_pod.sh" 2>&1 | grep -E 'P5|REFUSED|WCM a1|launch.sh' | sed 's/^/        /' | cut -c1-900
    return 0
  fi
  job "a1_$name" env $envs bash "$A/run_pod.sh"
}
a1_ok() {   # a1_ok NAME: the seeds with an ok status row
  cat "$O/owt_a1/$1/wcm/a1_l1e-3"/main/status.shard*.jsonl 2> /dev/null | grep '"ok": true' \
    | sed 's/.*"seed": "\([^"]*\)".*/\1/' | sort -u
}
a1_score() {   # a1_score NAME FITNAME KEYS: score_arena_a.py --scope a1 -> OUT/owt_a1/score_NAME
  local name=$1 fit=$2 keys=$3 stop=""
  local S=$O/owt_a1/score_$name
  [ -n "${STOP_EPOCH:-}" ] && stop=" --stop-epoch $STOP_EPOCH"
  [ -n "$keys" ] || { say "  a1 score $name: no fitted keys"; return 1; }
  job "a1_score_$name" bash -c "env $BASEENV ARENA_A_SCOPE=a1 DATA=$OWT_CORPUS STORES=$OWT_STO CUDA_VISIBLE_DEVICES=0 ARENA_A_CTX_DIRS=$OWT_TGT/ctx OUT=$S $PY -X utf8 -u $A/score_arena_a.py --scope a1 --out $S --wcm l1e-3=$O/owt_a1/$fit/wcm/a1_l1e-3 --keys $keys --selftest$stop" \
    || return $?
  [ "$DRY_RUN" = 1 ] && return 0
  local got
  got=$(grep -c '"eval_set": "home"' "$S/rows.jsonl" 2> /dev/null)
  [ "${got:-0}" -ge "$(ncsv "$keys")" ] || { say "  a1 score $name: $got of $(ncsv "$keys") units"; late && return 7; return 1; }
}
u_a1_fit_shallow() {
  a1_fit fit_shallow "$A1_SHALLOW" || return $?
  [ "$DRY_RUN" = 1 ] && return 0
  local n; n=$(a1_ok fit_shallow | wc -l)
  [ "$n" -ge "$(ncsv "$A1_SHALLOW")" ] || { say "  a1 shallow: $n of $(ncsv "$A1_SHALLOW") fits ok"; return 1; }
}
u_a1_score_shallow() { a1_score shallow fit_shallow "$A1_SHALLOW"; }
u_a1_fit_deep() {   # one run_pod.sh call per key, ascending depth: a cut / trip loses only the deeper keys
  # 2026-11-18: each call passes the CUMULATIVE key list (keys so far + this one). run_pod.sh's fits_done counts ok rows,
  # not keys, so on 11-17 a one-key list looked "every target fitted" after the first key's row (7 of 8 skipped); with
  # the cumulative list the count is short by exactly this key, and the driver skips the rows already finished.
  local k rc=0 n=0 cum=""
  for k in ${A1_DEEP//,/ }; do
    cum=${cum:+$cum,}$k
    if [ "$DRY_RUN" != 1 ]; then
      a1_ok fit_deep | grep -qxF "$k" && { n=$((n + 1)); continue; }
      late && { say "  a1 deep: STOP_EPOCH reached before $k"; return 7; }
      say "  a1 deep fit $k"
    fi
    a1_fit fit_deep "$cum" "AA_REDO=P5"; rc=$?
    [ "$DRY_RUN" = 1 ] && { [ "$k" = "${A1_DEEP%%,*}" ] && echo "    (... and so on for each of: $A1_DEEP)"; break; }
    case $rc in
      0) n=$((n + 1)) ;;
      9) say "  a1 deep: watchdog trip on $k (MemAvailable / GPU): no deeper key started"; return 9 ;;
      7) return 7 ;;
      *) say "  a1 deep: $k failed (rc $rc); next key" ;;
    esac
  done
  [ "$DRY_RUN" = 1 ] || [ "$n" -ge "$(ncsv "$A1_DEEP")" ]
}
u_trim_identity() {   # 2026-11-18 donor-code trim: refit 5.tc.8088 (fitted 11-18 17:11 before the trim) and compare
  a1_fit trim_identity "5.tc.8088" || return $?
  [ "$DRY_RUN" = 1 ] && return 0
  job trim_compare env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -I -B $E/compare_trim.py \
    --old $O/owt_a1/fit_deep/wcm/a1_l1e-3/main --new $O/owt_a1/trim_identity/wcm/a1_l1e-3/main --out $O/trim_identity.md
}
u_a1_score_deep() {
  local keys
  if [ "$DRY_RUN" = 1 ]; then keys="<the deep keys with an ok a1 fit>"; else keys=$(a1_ok fit_deep | paste -sd, -); fi
  a1_score deep fit_deep "$keys"
}
u_a1_compare() {
  job a1_compare env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -I -B $E/compare_a1_a2.py \
    --a1 $O/owt_a1/score_shallow/rows.jsonl,$O/owt_a1/score_deep/rows.jsonl --a2 $A2_ROWS --out $O/a1_vs_a2.md
}

# ---------------------------------------------------------------------------------------------- ram_probe
ram_probe() {   # ram_probe NAME LAM [STEPS]: one deep fit under probe_driver.py + ram_monitor.py (kills only this fit)
  local name=$1 lam=$2 steps=${3:-} W i=1 mrc rc cap
  W=$O/ram_probe/$name
  while [ "$DRY_RUN" != 1 ] && [ -e "$W" ]; do i=$((i + 1)); W=$O/ram_probe/$name.r$i; done   # a fresh dir per attempt
  local envs="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ARENA_A_BOS=1 FAMILY=gpt2-small-tcz SAE_PRELOAD=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WCM_VERSION=v4 DATA=$OWT_CORPUS STORES=$OWT_STO CUDA_VISIBLE_DEVICES=0 \
SLOT=0 SHARD=0/1 MODE=main ARENA_A_SCOPE=$RAM_SCOPE OUT=$W/wcm CTX_DIR=$OWT_TGT/ctx TARGETS=$W/targets.txt LAM=$lam \
ORDER=list SKIP_SPEC=1 CURVE_STRIDE=20 $(tune_kv)${steps:+ STEPS=$steps}${PROBE_DEEP_BS:+ DEEP_BS=$PROBE_DEEP_BS} \
NVIDIA_TF32_OVERRIDE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
PROBE_STATUS=$W/status.jsonl PROBE_STEP_EVERY=5 PROBE_SCAN_STEPS=1,50,200"
  cap=$((PROBE_MAX_MIN * 60))
  if [ "$DRY_RUN" = 1 ]; then
    echo "    \$ echo $RAM_KEY > $W/targets.txt"
    echo "    \$ setsid env $envs $PY -X utf8 -u $E/probe_driver.py >> $W/driver.log 2>&1 &"
    echo "    \$ $PY -I -B $E/ram_monitor.py --pid <its pid> --csv $W/mon.csv --status $W/status.jsonl --log $W/driver.log --every 2 --min-avail-mib $PROBE_MIN_AVAIL_MIB --gpu-max-mib $GPU_MAX --max-s <min($cap, hard deadline - now)> --kill-file $W/KILLED"
    return 0
  fi
  local left=$((HARD_DEADLINE - $(date +%s)))
  [ "$left" -lt "$cap" ] && cap=$left
  [ "$cap" -gt 60 ] || { say "  $name: no time left before the hard deadline"; return 7; }
  mkdir -p "$W"
  echo "$RAM_KEY" > "$W/targets.txt"
  echo "[$(ts)] env $envs $PY -X utf8 -u $E/probe_driver.py" > "$W/cmd.txt"
  setsid env $envs $PY -X utf8 -u $E/probe_driver.py >> "$W/driver.log" 2>&1 < /dev/null &
  JOB=$!
  $PY -I -B $E/ram_monitor.py --pid "$JOB" --csv "$W/mon.csv" --status "$W/status.jsonl" --log "$W/driver.log" \
    --every 2 --min-avail-mib "$PROBE_MIN_AVAIL_MIB" --gpu-max-mib "$GPU_MAX" --max-s "$cap" --kill-file "$W/KILLED"
  mrc=$?
  wait "$JOB" 2> /dev/null; rc=$?
  JOB=""
  if [ "$mrc" = 9 ]; then say "  $name: the monitor killed the fit: $(cat "$W/KILLED" 2> /dev/null)"; return 0; fi
  [ "$mrc" = 0 ] || say "  $name: ram_monitor.py rc $mrc"
  grep -q '"ok": true' "$W"/wcm/main/status.shard0.jsonl 2> /dev/null \
    || { say "  $name: fit rc $rc, no ok status row ($(grep -o '"error": "[^"]*' "$W"/wcm/main/status.shard0.jsonl 2> /dev/null | head -c 200))"; return 1; }
  say "  $name: fit ok ($(grep -o '"t_fit": [0-9.]*\|"n": [0-9]*' "$W"/wcm/main/status.shard0.jsonl | paste -sd' ' -))"
}
u_ram_l1e3() { ram_probe ram_l1e-3 1e-3; }
u_ram_l25e4() { ram_probe ram_l2.5e-4 2.5e-4; }
u_ram_s100() { ram_probe ram_s100 1e-3 100; }
u_ram_report() {
  job ram_report env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -I -B $E/ram_report.py --dir $O/ram_probe --out $O/ram_report.md
}

# ---------------------------------------------------------------------------------------------- t7_full
u_t7_full() {
  local s R=$T7_DIR/ref rc
  if [ "$DRY_RUN" != 1 ] && [ "$(date +%s)" -ge "$T7_LATEST" ]; then
    say "  t7_full: not started: after $(hm "$T7_LATEST") (N128_T7_LATEST_EPOCH)"; return 7
  fi
  if [ "$DRY_RUN" = 1 ] || [ ! -d "$R" ]; then
    job t7_prep env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -I -B $E/t7_prep.py --key $T7_KEY --out $T7_DIR || return 1
  fi
  [ "$DRY_RUN" = 1 ] && [ -d "$R" ] && echo "    (ref dir $R exists: $(grep -o '"W_0.001": [0-9]*' "$T7_DIR/PREP.json"))"
  for s in $T7_STAGES table; do
    if [ "$DRY_RUN" != 1 ] && [ "$s" != table ] && [ -f "$STATE/t7_$s.done" ]; then continue; fi
    if [ "$DRY_RUN" != 1 ] && [ "$s" != table ] && late; then return 7; fi
    JOB_DEADLINE=$T7_KILL job "t7_$s" env GPUS=0 P=1 STAGGER=5 STAGES=$s PY=$PY WCM_VERSION=v4 CTX_DIR=$T7_CTX REF_RUN=$R \
      OUT=$R OUT083=$T7_DIR/dan159 TARGETS=$T7_DIR/targets.txt HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
      CUDA_DEVICE_ORDER=PCI_BUS_ID bash $E83/run_pod.sh
    rc=$?
    [ "$DRY_RUN" = 1 ] && continue
    [ "$rc" = 0 ] || return "$rc"
    [ "$s" = table ] && continue
    local nr ne
    nr=$(cat "$T7_DIR/dan159/$s"/rows*.jsonl 2> /dev/null | grep -c .); ne=$(cat "$T7_DIR/dan159/$s"/rows*.jsonl 2> /dev/null | grep -c '"error"')
    say "  t7 $s: $nr rows, $ne error rows"
    [ "$nr" -gt 0 ] && [ "$ne" = 0 ] || return 1
    ts > "$STATE/t7_$s.done"
  done
}
u_t7_summary() {
  job t7_summary env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $PY -I -B $E/t7_summary.py --out083 $T7_DIR/dan159 \
    --night-log $O/night.log --prep $T7_DIR/PREP.json --md $T7_DIR/T7_SUMMARY.md
}

# ---------------------------------------------------------------------------------------------- estimates (min)
# smoke: the 2026-11-06 2-target smoke took ~33 min at 1 lambda (P4 4, P5 11.4, P6 10.6, P4h 1.6, P6b 5.6); here 3
# targets x 2 lambdas: P5 4 launches x 1.5 + 10 fits x 2.8 = 34, P6 ~19, P4 6, P4x (CPU, 100 branches K 1, 2 tc) 4,
# P6x 6, P4h 3, P6b 10, P7 1. ge_gpu: CPU smoke 4.6 min for the 6 circuits; GPU guess 2-4 min + 3 x 0.5 min start-up.
# a1: 126's a2 fits 2.5 min (shallow) / 3-5.4 min (L5-8) on OWT; a1 has ~2x the candidate sites: x1.5 guess, + ~1.5
# min start-up per run_pod.sh call (8 calls for deep). Scoring: 126 ~0.6 min per unit + 2. ram: 126's L9 a2 fit ran
# > 10 min before the watchdog; cap N128_PROBE_MAX_MIN. t7: 083 README 750 s per target + 3 x 3 min start-ups + table.
EST=([smoke]=85 [smoke_check]=1 [ge_gpu]=8 [a1_fit_shallow]=33 [a1_score_shallow]=8 [a1_fit_deep]=70
     [a1_score_deep]=8 [a1_compare]=1 [ram_l1e-3]=25 [ram_l2.5e-4]=25 [ram_s100]=12 [ram_report]=1 [t7_full]=35
     [t7_summary]=1)

# ---------------------------------------------------------------------------------------------- preflight
preflight() {   # read-only; prints every problem; rc 1 if any
  local bad=0 f k
  for f in "$PY" "$A/run_pod.sh" "$A/run_smoke_local.sh" "$A/score_arena_a.py" "$A/driver_tcz.py" "$A/run_ge.py" \
           "$E/probe_driver.py" "$E/ram_monitor.py" "$E/ram_report.py" "$E/check_smoke.py" "$E/compare_a1_a2.py" \
           "$E/ge_gpu_report.py" "$E/t7_prep.py" "$E/t7_summary.py" "$E/build_ge107gpu.sh" "$E83/run_pod.sh" \
           "$TUNE" "$OWT_CORPUS/corpus_manifest.json" "$OWT_STO/data_pass_manifest.json" "$OWT_STO/neg_ctx.pt" \
           "$OWT_TGT/targets.txt" "$GE_CPU_REF/status.jsonl" "$A/out/targets_smoke/prompts.jsonl" \
           experiments/035-transcoder-compare/transcoders_ct/meta.json outputs/candidates.pt outputs/latent_stats.pt \
           "$T7_CTX/$T7_KEY.pt" "$HOME/.cache/huggingface/hub/models--gpt2" \
           "$HOME/.cache/huggingface/hub/models--pchlenski--gpt2-transcoders"; do
    [ -e "$f" ] || { say "  missing: $f"; bad=1; }
  done
  for k in ${A1_SHALLOW//,/ } ${A1_DEEP//,/ } $RAM_KEY; do
    grep -qxF "$k" "$OWT_TGT/targets.txt" 2> /dev/null || { say "  $k is not in $OWT_TGT/targets.txt"; bad=1; }
    [ -f "$OWT_TGT/ctx/$k.pt" ] || { say "  no record $OWT_TGT/ctx/$k.pt"; bad=1; }
  done
  for k in ${SMOKE_KEYS//,/ } ${GE_KEYS//,/ }; do
    grep -qxF "$k" "$A/out/targets_smoke/targets.txt" 2> /dev/null || { say "  $k is not in targets_smoke"; bad=1; }
  done
  [ "$(echo "$SMOKE_KEYS" | tr ',' '\n' | grep -F .tc. | grep -vc '^0\.')" -ge 2 ] \
    || say "  NOTE: fewer than 2 tc keys (L >= 1) in N128_SMOKE_KEYS: df_par will have only 1 process with work"
  if [ -e "$SMOKE_W" ] && [ ! -f "$STATE/smoke.started" ] && sel smoke && [ ! -f "$STATE/smoke.done" ]; then
    say "  $SMOKE_W exists but was not started by this runner: the smoke needs a FRESH dir (N128_SMOKE_WORK)"; bad=1
  fi
  [ -x "$GE_GPU/bin/python" ] || [ -x "$HOME/venvs/tc107gpu/bin/python" ] \
    || say "  NOTE: neither $GE_GPU nor ~/venvs/tc107gpu: ge_gpu will be a no-op (needs a download)"
  command -v nvidia-smi > /dev/null || { say "  nvidia-smi not found"; bad=1; }
  return "$bad"
}

# ---------------------------------------------------------------------------------------------- main
say "exp 128 night | $(git --no-pager log -1 --format='%h' 2> /dev/null) | stop epoch $(hm "$STOP") (hard $(hm "$HARD_DEADLINE")) | t7 start before $(hm "$T7_LATEST"), kill $(hm "$T7_KILL") | watchdog GPU > $GPU_MAX MiB / RAM < $RAM_MIN_GB GB every $EVERY s$([ -n "$SOFT" ] && echo " | soft: $SOFT")$([ -n "$ONLY" ] && echo " | only $ONLY")$([ -n "$SKIP" ] && echo " | skip $SKIP")$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
preflight || { say "preflight refused"; [ "$DRY_RUN" = 1 ] || exit 8; }
m=$(gpu_mib)
if [ "$DRY_RUN" = 1 ]; then
  say "dry run: GPU now holds ${m:-?} MiB (refuse above $GPU_IDLE); lock $LOCK $([ -e "$LOCK" ] && echo "EXISTS: $(head -c 200 "$LOCK")" || echo absent); ABORTED $([ -f "$O/ABORTED" ] && echo EXISTS || echo absent); MemAvailable $(avail_mib) MiB"
  [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ] && say "dry run: NOTE the GPU holds $m MiB > $GPU_IDLE: a real start would be REFUSED (N128_GPU_IDLE_MIB)"
else
  [ -e "$LOCK" ] && { say "refused: $LOCK exists: $(head -c 300 "$LOCK")"; exit 8; }
  [ -f "$O/ABORTED" ] && { say "refused: $O/ABORTED exists ($(cat "$O/ABORTED")): read it, move it away to resume"; exit 8; }
  if [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ]; then say "refused: the GPU already holds $m MiB (> $GPU_IDLE, N128_GPU_IDLE_MIB)"; exit 8; fi
  [ -n "$m" ] || say "WARNING: nvidia-smi gave no reading; the GPU watchdog is blind"
  echo "exp 128 run_night.sh (experiments/128-pod-preflight) pid $$ started $(ts) on $(hostname); stop epoch ${STOP_EPOCH:-none}" > "$LOCK" \
    || { say "cannot write $LOCK"; exit 8; }
  # the lock is this run's own: removed at exit only while it still names this pid
  trap 'grep -q "pid $$ " "$LOCK" 2> /dev/null && rm -f "$LOCK"' EXIT
  trap '[ -n "$JOB" ] && { kill -TERM -- -"$JOB" 2> /dev/null; sleep 10; kill -KILL -- -"$JOB" 2> /dev/null; }; say "signal: stopped"; exit 143' INT TERM HUP
  mkdir -p "$O" "$STATE" "$LOGS"
fi

stage smoke u_smoke
stage smoke_check u_smoke_check always
stage ge_gpu u_ge_gpu
stage a1_fit_shallow u_a1_fit_shallow
stage a1_score_shallow u_a1_score_shallow
stage trim_identity u_trim_identity
stage a1_fit_deep u_a1_fit_deep
stage a1_score_deep u_a1_score_deep
stage a1_compare u_a1_compare always
stage ram_l1e-3 u_ram_l1e3
stage ram_l2.5e-4 u_ram_l25e4
stage ram_s100 u_ram_s100
stage ram_report u_ram_report always
stage t7_full u_t7_full
stage t7_summary u_t7_summary always
if [ "$DRY_RUN" = 1 ]; then say "dry run: total ~$(fmin "$ETA") min (~$(awk -v m="$ETA" 'BEGIN { printf "%.1f", m / 60 }') h) for the stages not done"; exit 0; fi
say "night finished: $FAILS failed stage(s)$([ "$CUT" = 1 ] && echo '; STOP_EPOCH cut the plan')"
[ "$FAILS" = 0 ] || exit 1
[ "$CUT" = 0 ] || exit 7
exit 0
