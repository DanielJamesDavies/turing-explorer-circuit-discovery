#!/usr/bin/env bash
# Arena D lr-budget test (README.md here): C0 (arena D's WCM as the pod runs it) vs C1 (+ v4's lr budget) on the
# local 5070 Ti. Resumable (every finished fit is skipped: lrtest.py checks each arm's eval.jsonl). ONE GPU process at
# a time: the stages run one after another, each one python process:
#   1 res_main   gemma-2-2b-gsres-gap6, rc, Caples data + Caples mode per_position (the smoke's wcm_res), lam 1e-3 /
#                1e-4 / 1e-2 x C0, C1
#   2 plt_main   gemma-2-2b-plt, sfc data, pooled (the smoke's wcm_plt), rc / simple / nounpp x the 3 lambdas x C0, C1
#   3 res_seed2  gsres rc lam 1e-3, C0 and C1 with a second R-donor seed (replicate)
#   4 analyze    analyze.py (CPU) -> out/RESULTS.md + out/results.json
# 400 steps per fit (pod_stage_d.sh DW_STEPS), bf16 unsloth weights (the pod: fp32 google; labelled).
# Budget: LRTEST_BUDGET_S (10800 = 3 GPU-h, loads included); lrtest.py stops before a unit that would pass it, using
#   the largest measured unit time of that family (the first fit's s/step is logged with a projection).
# Guards: refuses to start if the card already holds > GPU_BUSY_MIB (3000) MiB, or MemAvailable < 4 GB, or another
#   run.sh holds out/run.lock. Watchdog (every 10 s, also the sampler -> out/gpu_mem.csv): GPU > GPU_ABORT_MIB (15300)
#   MiB or host MemAvailable < MEM_MIN_MB (4096) -> the running stage's process group is killed, ABORTED is written in
#   this invocation's state dir and no further stage starts. STOP_EPOCH (env): no new unit / stage after it (lrtest.py
#   checks before every fit); LRTEST_HARD_GRACE (600) s after it the running stage is killed.
# DRY_RUN=1: checks files and the venv, parses the scripts, prints every command, runs lrtest.py plan and analyze.py
#   on CPU (CUDA_VISIBLE_DEVICES=""); the GPU is not touched.
#   WSL, repo root:
#     bash experiments/107-true-baselines/arena_d/lrtest/run.sh > experiments/107-true-baselines/arena_d/lrtest/out/run.log 2>&1
set -u
cd "$(dirname "$0")/../../../.." || exit 1
LT=experiments/107-true-baselines/arena_d/lrtest
O=${LRTEST_OUT:-$LT/out}
PY=${PY:-.venv/bin/python}
DRY_RUN=${DRY_RUN:-0}
STOP_EPOCH=${STOP_EPOCH:-}
GPU_BUSY_MIB=${GPU_BUSY_MIB:-3000}
GPU_ABORT_MIB=${GPU_ABORT_MIB:-15300}
MEM_MIN_MB=${MEM_MIN_MB:-4096}
HARD_GRACE=${LRTEST_HARD_GRACE:-600}
STAGES="res_main plt_main res_seed2"
say() { echo "[lrtest.sh $(date +%H:%M:%S)] $*"; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' '; }
mem_mb() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
late() { [ -n "$STOP_EPOCH" ] && [ "$(date +%s)" -ge "$STOP_EPOCH" ]; }
# the environment of every GPU step (the smoke's, run_pilot_d.sh; plus 081 WCM_VERSION=v2 pinned as pod_stage_d.sh does)
ENVV="HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled TORCH_THREADS=4 \
OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GEMMA_HF_ID=unsloth/gemma-2-2b WCM_VERSION=v2 \
LRTEST_OUT=$O ${STOP_EPOCH:+STOP_EPOCH=$STOP_EPOCH} ${LRTEST_BUDGET_S:+LRTEST_BUDGET_S=$LRTEST_BUDGET_S} \
${LRTEST_DTYPE:+LRTEST_DTYPE=$LRTEST_DTYPE}"
CPUENV="env -u PYTHONPATH CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 WCM_VERSION=v2 LRTEST_OUT=$O"

say "start | out $O | stop epoch ${STOP_EPOCH:-none} | budget ${LRTEST_BUDGET_S:-10800} s | dry run $DRY_RUN"
if [ "$DRY_RUN" = 1 ]; then
  FAILS=0
  [ -x "$PY" ] && say "ok python $PY" || { say "MISSING python $PY"; FAILS=$((FAILS + 1)); }
  for f in $LT/lrtest.py $LT/analyze.py experiments/107-true-baselines/arena_d/wcm_gemma.py \
           experiments/107-true-baselines/arena_d/out/caples_means/rc_gap6_bf16_SMOKE.pt \
           third_party/ScalableSAECircuits/data/sva/rc_train.json; do
    [ -f "$f" ] && say "ok file $f" || { say "MISSING $f"; FAILS=$((FAILS + 1)); }
  done
  for f in $LT/lrtest.py $LT/analyze.py; do
    $PY -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" && say "ok parse $f" \
      || { say "PARSE ERROR $f"; FAILS=$((FAILS + 1)); }
  done
  command -v nvidia-smi > /dev/null && say "ok nvidia-smi (not queried in a dry run)" || say "WARNING no nvidia-smi"
  for st in $STAGES; do
    say "DRY stage $st: setsid env -u PYTHONPATH $ENVV $PY -u $LT/lrtest.py run --stage $st > $O/logs/$st.log"
  done
  say "DRY plan (CPU):"
  $CPUENV $PY -B $LT/lrtest.py plan || { say "FAILED plan"; FAILS=$((FAILS + 1)); }
  say "DRY analyze (CPU) -> $O/dryrun:"
  $CPUENV $PY -B $LT/analyze.py --out "$O" --write "$O/dryrun" || { say "FAILED analyze"; FAILS=$((FAILS + 1)); }
  say "DRY DONE ($FAILS problems)"
  [ "$FAILS" -eq 0 ]
  exit $?
fi

mkdir -p "$O/logs" "$O/state"
exec 9> "$O/run.lock"
flock -n 9 || { say "REFUSED: another run.sh holds $O/run.lock"; exit 3; }
S=$O/state/run_$(date +%s)          # this invocation's state (pid, stage, ABORTED): nothing is ever deleted
mkdir -p "$S"
command -v nvidia-smi > /dev/null || { say "REFUSED: no nvidia-smi"; exit 3; }
m=$(gpu_mib)
[ -n "$m" ] || { say "REFUSED: nvidia-smi gave no memory reading"; exit 3; }
if [ "$m" -gt "$GPU_BUSY_MIB" ]; then say "REFUSED: the card holds $m MiB (> $GPU_BUSY_MIB): another process"; exit 3; fi
a=$(mem_mb)
if [ "$a" -lt "$MEM_MIN_MB" ]; then say "REFUSED: host MemAvailable $a MB < $MEM_MIN_MB MB"; exit 3; fi
late && { say "STOP_EPOCH already passed: nothing started"; exit 0; }
echo idle > "$S/stage"
echo none > "$S/pid"
(   # sampler + watchdog
  while true; do
    m=$(gpu_mib); a=$(mem_mb); st=$(cat "$S/stage" 2> /dev/null); pid=$(cat "$S/pid" 2> /dev/null)
    echo "$(date +%s),$st,${m:-NA},${a:-NA}" >> "$O/gpu_mem.csv"
    why=""
    if [ -n "$m" ] && [ "$m" -gt "$GPU_ABORT_MIB" ] 2> /dev/null; then why="GPU $m MiB > $GPU_ABORT_MIB MiB"; fi
    if [ -n "$a" ] && [ "$a" -lt "$MEM_MIN_MB" ] 2> /dev/null; then why="${why:+$why; }host MemAvailable $a MB < $MEM_MIN_MB MB"; fi
    if [ -n "$STOP_EPOCH" ] && [ "$(date +%s)" -ge $((STOP_EPOCH + HARD_GRACE)) ]; then why="${why:+$why; }STOP_EPOCH + ${HARD_GRACE} s"; fi
    if [ -n "$why" ] && [ "$pid" != none ] && [ -n "$pid" ]; then
      echo "$(date -Is) stage $st: $why" >> "$S/ABORTED"
      echo "[lrtest.sh $(date +%H:%M:%S)] WATCHDOG: $why: killing stage $st (pgid $pid)"
      kill -TERM -- "-$pid" 2> /dev/null; sleep 15; kill -KILL -- "-$pid" 2> /dev/null
      echo none > "$S/pid"
    fi
    sleep 10
  done
) &
WD=$!
trap 'kill $WD 2> /dev/null' EXIT
FAILS=0
for st in $STAGES; do
  [ -f "$S/ABORTED" ] && { say "skip $st: aborted ($(tail -1 "$S/ABORTED"))"; continue; }
  late && { say "skip $st: STOP_EPOCH passed"; continue; }
  m=$(gpu_mib)
  if [ -n "$m" ] && [ "$m" -gt "$GPU_BUSY_MIB" ]; then
    say "skip $st (and the rest): the card holds $m MiB before the stage (> $GPU_BUSY_MIB)"; FAILS=$((FAILS + 1)); break
  fi
  say "start $st"
  echo "$st" > "$S/stage"
  t0=$(date +%s)
  setsid env -u PYTHONPATH $ENVV $PY -u $LT/lrtest.py run --stage "$st" >> "$O/logs/$st.log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$S/pid"
  wait "$pid"
  rc=$?
  echo none > "$S/pid"
  echo idle > "$S/stage"
  printf "%s\t%s\t%s\t%s\n" "$st" "$t0" "$(date +%s)" "$rc" >> "$O/steps.tsv"
  if [ "$rc" -ne 0 ]; then say "FAILED $st (exit $rc; $O/logs/$st.log)"; FAILS=$((FAILS + 1)); else say "done $st ($(( $(date +%s) - t0 )) s)"; fi
  grep -E "FIRST FIT|BUDGET|STOP_EPOCH" "$O/logs/$st.log" | tail -3
done
say "analyze (CPU)"
$CPUENV $PY -B $LT/analyze.py --out "$O" > "$O/logs/analyze.log" 2>&1 || { say "FAILED analyze"; FAILS=$((FAILS + 1)); }
tail -40 "$O/logs/analyze.log"
[ -f "$S/ABORTED" ] && say "ABORTED: $(cat "$S/ABORTED")"
say "DONE ($FAILS failed stages)"
[ "$FAILS" -eq 0 ] && [ ! -f "$S/ABORTED" ]
