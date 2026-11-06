#!/usr/bin/env bash
# Arena D item D7: the local GPU smoke (IMPLEMENTATION_PLAN §2), Thursday night 2026-11-05. ONE GPU process at a time,
# in order (every step resumes: a finished step is skipped):
#   1 d6           pytest -m gpu -k real_open_gate (test_wcm_gemma.py): open-gate PLT and gsres site models on the
#                  unsloth bf16 Gemma-2-2B, logits bitwise = raw HF on 8 SVA rc prompts (both families)
#   2 caples_means caples_means.py on the real model (caples107; unsloth bf16 via hf_model=, labelled SMOKE): the
#                  local means file out/caples_means/rc_gap6_bf16_SMOKE.pt (their running-mean bug kept + exact)
#   3 real_caples  ARENA_D_CAPLES_PT=<that file> pytest -m gpu -k real_caples (README §4.3 test 4 on the real model)
#   4 caples_smoke caples_smoke.py (bf16, one lambda 0.1, --n 176 / 4 mean batches). DEVIATION from the plan's "N 64":
#                  their do_training_run evaluates batch[-i] for i = 0..9, so it needs >= 10 batches (160 prompts);
#                  caples_smoke.py asserts this. 176 = 11 batches is the smallest run their code accepts with margin.
#   5 wcm_plt      wcm_gemma.py run, gemma-2-2b-plt, rc, SFC data, pooled means, lambda 1e-3, WCM_STEPS (50), bf16
#   6 wcm_res      wcm_gemma.py run, gemma-2-2b-gsres-gap6, rc, Caples data, --mode caples --means per_position with
#                  step 2's file (native per_position if step 2 failed), lambda 1e-3, WCM_STEPS, bf16
#                  -> per-family timings: t_fit / steps at the engine's micro-batch 4 x 7 tokens (rc)
#   7 f3           ag_fidelity.py F1 F3 F4 F5 F6 (ct107; bf16, unsloth, Dallas)
#   8 ag_tasks     ag_tasks.py run, rc held-out, first AG_N (4) prompts, bf16: AG timings for D10 (AG_TASKS=0 skips)
#   9 summary      pilot_summary_d.py (CPU) -> out/pilot/SUMMARY.txt + summary.json
# Background: nvidia-smi memory.used every 30 s -> out/pilot/gpu_mem.csv (epoch,step,MiB); above SPILL_MIB (15600 of
# the card's 16303 MiB) a WARNING line goes to out/pilot/warnings.txt (WDDM spills into shared host memory near full).
# Watchdog: at STOP_AT (default 08:30, today or tomorrow) it kills the running step's process group, writes
# out/pilot/STOPPED, and no new step starts; the summary still runs. Before each GPU step the script waits while the GPU
# shows more than GPU_BUSY_MIB (3000) used (another process), until the stop time.
# Progress for the +3 / +10 min checks: out/pilot/status.txt (current step, start time) and out/pilot/pilot_d.log.
# DRY_RUN=1: checks the venvs and files, parses every script, prints each command, runs the summary on an empty dir.
#   WSL, repo root:
#     nohup setsid bash experiments/107-true-baselines/arena_d/run_pilot_d.sh \
#         > experiments/107-true-baselines/arena_d/out/pilot/pilot_d.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
AD=experiments/107-true-baselines/arena_d
P=${PILOT_OUT:-$AD/out/pilot}
PY=.venv/bin/python
CT=${CT_PY:-$HOME/venvs/ct107/bin/python}
CAP=${CAPLES_PY:-$HOME/venvs/caples107/bin/python}
MEANS=$AD/out/caples_means/rc_gap6_bf16_SMOKE.pt
WCM_STEPS=${WCM_STEPS:-50}
AG_TASKS=${AG_TASKS:-1}
AG_N=${AG_N:-4}
SPILL_MIB=${SPILL_MIB:-15600}
GPU_BUSY_MIB=${GPU_BUSY_MIB:-3000}
DRY_RUN=${DRY_RUN:-0}
STOP_AT=${STOP_AT:-08:30}
now=$(date +%s)
STOP=$(date -d "today $STOP_AT" +%s); [ "$STOP" -le "$now" ] && STOP=$(date -d "tomorrow $STOP_AT" +%s)
mkdir -p "$P/logs"
say() { echo "[pilot_d] $(date +%H:%M:%S) $*"; }
late() { [ "$(date +%s)" -ge "$STOP" ]; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' '; }
# the environment of every step (README §2 local smokes; HF_TOKEN is never needed offline and never set here)
ENVV="HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled TORCH_THREADS=4 \
OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GEMMA_HF_ID=unsloth/gemma-2-2b STOP_AT=$STOP_AT"
echo "idle" > "$P/.step"
if [ "$DRY_RUN" != 1 ]; then
  (   # memory sampler
    while true; do
      m=$(gpu_mib); s=$(cat "$P/.step" 2> /dev/null)
      echo "$(date +%s),$s,${m:-NA}" >> "$P/gpu_mem.csv"
      if [ -n "$m" ] && [ "$m" -gt "$SPILL_MIB" ] 2> /dev/null; then
        echo "$(date '+%H:%M:%S') WARNING step $s: GPU memory $m MiB > $SPILL_MIB MiB (shared-memory spill risk)" \
          >> "$P/warnings.txt"
      fi
      sleep 30
    done
  ) &
  SAMPLER=$!
  (   # watchdog
    while [ "$(date +%s)" -lt "$STOP" ]; do sleep 30; done
    date -Is > "$P/STOPPED"
    pid=$(cat "$P/.pid" 2> /dev/null)
    echo "[pilot_d] $(date +%H:%M:%S) STOP_AT $STOP_AT reached: killing step $(cat "$P/.step") (pgid ${pid:-none})"
    if [ -n "$pid" ]; then kill -TERM -- "-$pid" 2> /dev/null; sleep 30; kill -KILL -- "-$pid" 2> /dev/null; fi
    pkill -TERM -f "107-true-baselines/arena_d/(wcm_gemma|caples_means|caples_smoke|ag_fidelity|ag_tasks|test_wcm_gemma)" \
      2> /dev/null || true
  ) &
  WATCHDOG=$!
  trap 'kill $SAMPLER $WATCHDOG 2> /dev/null' EXIT
fi
FAILS=0
wait_gpu() {   # another process on the GPU: wait (one GPU process at a time)
  local m
  while true; do
    m=$(gpu_mib)
    { [ -z "$m" ] || [ "$m" -le "$GPU_BUSY_MIB" ]; } && return 0
    late && return 1
    say "GPU shows $m MiB used (> $GPU_BUSY_MIB): waiting"
    sleep 60
  done
}
step() {   # step NAME DONE_TEST POST_TEST command...  (tests: shell strings; DONE true = skip; POST false = failed)
  local name=$1 done_test=$2 post_test=$3; shift 3
  if eval "$done_test"; then say "skip $name (done)"; return 0; fi
  if late; then say "skip $name (STOP_AT reached)"; return 1; fi
  if [ "$DRY_RUN" = 1 ]; then say "DRY $name: $*"; return 0; fi
  wait_gpu || { say "skip $name (GPU busy until STOP_AT)"; return 1; }
  say "start $name"
  echo "$name" > "$P/.step"
  echo "$name started $(date '+%Y-%m-%d %H:%M:%S')" > "$P/status.txt"
  local t0 rc pid
  t0=$(date +%s)
  setsid env $ENVV "$@" > "$P/logs/$name.log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$P/.pid"
  wait "$pid"
  rc=$?
  rm -f "$P/.pid"
  printf "%s\t%s\t%s\t%s\n" "$name" "$t0" "$(date +%s)" "$rc" >> "$P/steps.tsv"
  echo "idle" > "$P/.step"
  if [ $rc -eq 0 ] && ! eval "$post_test"; then rc=99; fi
  echo "$name finished rc $rc $(date '+%Y-%m-%d %H:%M:%S')" > "$P/status.txt"
  if [ $rc -ne 0 ]; then say "FAILED $name (exit $rc; $P/logs/$name.log)"; FAILS=$((FAILS + 1)); return 1; fi
  say "done $name ($(( $(date +%s) - t0 )) s)"
}
passed() {   # passed N LOG: pytest's closing line reads "N passed ..." with nothing failed / errored / skipped
  local last
  last=$(grep -E " in [0-9.]+s" "$P/logs/$2.log" 2> /dev/null | tail -1 | sed -E 's/^=+ //')
  echo "$last" | grep -qE "^$1 passed" && ! echo "$last" | grep -qE "failed|error|skipped"
}
say "start; hard stop $(date -d @"$STOP" '+%a %H:%M'); dry run $DRY_RUN; out $P"
if [ "$DRY_RUN" = 1 ]; then
  for x in "$PY" "$CT" "$CAP"; do [ -x "$x" ] && say "ok python $x" || { say "MISSING python $x"; FAILS=$((FAILS + 1)); }; done
  for f in wcm_gemma.py test_wcm_gemma.py caples_means.py caples_smoke.py ag_fidelity.py ag_tasks.py \
           pilot_summary_d.py caples_fidelity.py score_d.py; do
    $PY -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$AD/$f" && say "ok parse $f" \
      || { say "PARSE ERROR $f"; FAILS=$((FAILS + 1)); }
  done
  for f in third_party/ScalableSAECircuits/data/sva/rc_train.json $AD/out/fidelity/gemma-fact-dallas-austin.json; do
    [ -f "$f" ] && say "ok file $f" || { say "MISSING $f"; FAILS=$((FAILS + 1)); }
  done
  command -v nvidia-smi > /dev/null && say "ok nvidia-smi (GPU not queried in a dry run)" || say "WARNING no nvidia-smi"
fi
# 1 D6
step d6 'passed 2 d6' 'passed 2 d6' \
  $PY -m pytest $AD/test_wcm_gemma.py -m gpu -k real_open_gate -q -s -rs -p no:cacheprovider
# 2 Caples' means on the real model (the local file)
step caples_means '[ -f $MEANS ]' '[ -f $MEANS ]' \
  $CAP -I -u $AD/caples_means.py --hf-model unsloth/gemma-2-2b --dtype bfloat16 --device cuda --out $MEANS
# 3 test 4 on the real model
if [ -f "$MEANS" ] || [ "$DRY_RUN" = 1 ]; then
  step real_caples 'passed 1 real_caples' 'passed 1 real_caples' \
    ARENA_D_CAPLES_PT=$MEANS $PY -m pytest $AD/test_wcm_gemma.py -m gpu -k real_caples -q -s -rs -p no:cacheprovider
else
  say "skip real_caples (no means file)"
fi
# 4 Caples smoke (their utils, bf16, one lambda)
step caples_smoke '[ -f $P/caples_smoke/smoke_summary.json ]' '[ -f $P/caples_smoke/smoke_summary.json ]' \
  $CAP -I -u $AD/caples_smoke.py --device cuda --dtype bfloat16 --n 176 --mean-batches 4 --lams 0.1 \
  --out $P/caples_smoke
# 5-6 tiny WCM fits, both families (timings)
step wcm_plt '[ -s $P/wcm_plt/eval.jsonl ]' '[ -s $P/wcm_plt/eval.jsonl ]' \
  $PY -u $AD/wcm_gemma.py run --family gemma-2-2b-plt --tasks rc --lams 1e-3 --steps $WCM_STEPS --dtype bfloat16 \
  --out $P/wcm_plt
if [ -f "$MEANS" ] || [ "$DRY_RUN" = 1 ]; then
  RES_MODE=(--mode caples --means per_position --caples-means $MEANS)
else
  say "no means file: wcm_res runs native per_position"
  RES_MODE=(--mode native --means per_position)
fi
step wcm_res '[ -s $P/wcm_res/eval.jsonl ]' '[ -s $P/wcm_res/eval.jsonl ]' \
  $PY -u $AD/wcm_gemma.py run --family gemma-2-2b-gsres-gap6 --tasks rc --lams 1e-3 --steps $WCM_STEPS \
  --dtype bfloat16 "${RES_MODE[@]}" --out $P/wcm_res
# 7 F3 Dallas (+ F1, F4, F5, F6). AG_F6_MIN_FREE_GB (local only; pod default 0 = off): F6's eager rebuild is recorded
#   as skipped, not run, if < that many GB are free after the lazy model is freed (2026-11-05: both resident -> spill)
step f3 '[ -f $P/ag_fidelity_bf16.json ]' '[ -f $P/ag_fidelity_bf16.json ]' \
  AG_F6_MIN_FREE_GB=${AG_F6_MIN_FREE_GB:-8} $CT -B -I -u $AD/ag_fidelity.py --checks F1,F3,F4,F5,F6 --dtype bfloat16 --hf-model unsloth/gemma-2-2b \
  --device cuda --out $P/ag_fidelity_bf16.json
# 8 AG task graphs (timings): the prompts file first (CPU, repo .venv), then ct107 on the GPU
if [ "$AG_TASKS" = 1 ]; then
  if [ "$DRY_RUN" = 1 ]; then
    say "DRY ag_prompts: CUDA_VISIBLE_DEVICES= $PY -B $AD/ag_tasks.py prompts --tasks rc --out $P/ag_tasks"
  elif [ ! -f "$P/ag_tasks/prompts/rc.json" ]; then
    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 $PY -B $AD/ag_tasks.py prompts --tasks rc --out $P/ag_tasks \
      > "$P/logs/ag_prompts.log" 2>&1 || say "FAILED ag_prompts"
  fi
  step ag_tasks "[ \$(ls $P/ag_tasks/rc/held/p*.json 2> /dev/null | wc -l) -ge $AG_N ]" \
    "[ \$(ls $P/ag_tasks/rc/held/p*.json 2> /dev/null | wc -l) -ge $AG_N ]" \
    $CT -B -I -u $AD/ag_tasks.py run --tasks rc --split held --limit $AG_N --dtype bfloat16 \
    --hf-model unsloth/gemma-2-2b --device cuda --out $P/ag_tasks
fi
# 9 summary (CPU; runs even after STOP_AT: it only reads files)
if [ "$DRY_RUN" = 1 ]; then
  mkdir -p "$AD/out/dryrun/pilot_empty"
  say "DRY summary on an empty pilot dir:"
  CUDA_VISIBLE_DEVICES= $PY -B $AD/pilot_summary_d.py "$AD/out/dryrun/pilot_empty" --spill-mib "$SPILL_MIB" \
    > "$AD/out/dryrun/pilot_summary.log" 2>&1 && say "ok summary" || { say "FAILED summary"; FAILS=$((FAILS + 1)); }
else
  echo "summary" > "$P/.step"
  CUDA_VISIBLE_DEVICES= $PY -B $AD/pilot_summary_d.py "$P" --spill-mib "$SPILL_MIB" > "$P/logs/summary.log" 2>&1 \
    || say "FAILED summary"
  cat "$P/logs/summary.log"
fi
say "DONE ($FAILS failed steps)"
[ "$FAILS" -eq 0 ]
