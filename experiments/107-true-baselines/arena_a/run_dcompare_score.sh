#!/usr/bin/env bash
# dcompare (2026-11-06, agent `dcompare`): score Dunefsky et al.'s transcoder circuits and our WCM circuits on the 6
# panel tc targets with the ONE arena A scorer (score_arena_a.py, unchanged), on GPU 0, one GPU process at a time.
#   WSL, repo root:  DRY_RUN=1 bash experiments/107-true-baselines/arena_a/run_dcompare_score.sh    # print, run nothing
#                    bash experiments/107-true-baselines/arena_a/run_dcompare_score.sh
# Inputs (read only):
#   records  out/panel/rec/ctx (ARENA_A_CTX_DIRS): the records the panel fits and every earlier score used; held-out
#            strongest ("ho", 16 rows) and home ("home": a Dunefsky circuit's own prompt j; WCM's K = 8 prompts) come
#            from them; the scorer asserts each Dunefsky prompt = its record's prompt j
#   Dunefsky out/dcompare/dunefsky (run_dunefsky.py on out/dcompare/targets/prompts.jsonl = make_targets.prompts_for on
#            those records; K = 4 prompts x num_branches 5..60, num_iters 5) -> scope a2 (their node universe: tc only,
#            attention live and undecomposed)
#   WCM base out/panel/wcm_l1e-3 (v4, lambda 1e-3, 400 steps; 11.tc.22949 has no circuit: it did not fit on the card)
#   WCM tune out/tune/nec0.3/fit (the chosen config, NEC_W 0.3; dev targets 2.tc.17681 4.tc.13440 5.tc.22587 8.tc.15982)
#            Both WCM sets are a1 fits (tc + z + z2 sites) -> scored in scope a1 (the scorer cannot put an a1 fit in a2)
# Stages (each: score rows ho + home with Z / A / C / nec_a / nec_b / nec_b_tc, --selftest, --donor (R, fit variant,
# 8 draws), then --summary on the CPU):
#   a2_dunefsky   scope a2, Dunefsky units                                  -> out/dcompare/score/a2_dunefsky
#   a1_wcm_base   scope a1, WCM base                                        -> out/dcompare/score/a1_wcm_base
#   a1_wcm_nec03  scope a1, WCM nec0.3                                      -> out/dcompare/score/a1_wcm_nec03
#   OPTIONAL (DC_FIT_A2=1; off by default, GPU fit, not asked for): WCM fitted in scope a2 (tc only, same universe as
#   Dunefsky) at lambda 1e-3 + NEC_W 0.3, keys DC_FIT_A2_KEYS (default the 4 non-vacuous fitted ones), then scored in a2
#                 a2_wcm_fit -> out/dcompare/wcm_a2_nec03 ; a2_wcm_score -> out/dcompare/score/a2_wcm_nec03
# DC_STAGES picks / orders the three score stages (default all; the WCM stages do not need the Dunefsky run).
# a2_dunefsky is skipped (counted as a failed stage) while run_dunefsky.py is still running, unless DC_ALLOW_PARTIAL=1.
# Donor pool: out/dcompare/score/random_pool.pt (073's random pool, filtered against DC_DONOR_RECORDS = the panel
# records dir = the production rule; built once by the first stage, reused by the rest). If too few donors survive
# (the scorer raises "independent random donors after filtering"), the stage is retried ONCE with --donor-records run
# (the smoke's rule: filtered against the run's records only) into random_pool_run.pt, and that is logged.
# Watchdog every DC_WATCH_S [10] s per stage (score/watch.csv: epoch, stage, GPU MiB, host MemAvailable MiB): kill the
# stage's process group and stop the script (score/ABORTED) when GPU memory.used > DC_GPU_MAX_MIB [15300] or host
# MemAvailable < DC_RAM_MIN_GB [4] GB. A stage is refused when the card already holds > DC_GPU_IDLE_MIB [3000] MiB.
# Resumable: the scorer skips scored units / self-tests / donor rows. Timings: score/steps.tsv (stage, t0, t1, rc, why).
# Exit 0 ok; 1 a stage failed; 8 preflight refused; 9 aborted by the watchdog.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT" || exit 1
A=experiments/107-true-baselines/arena_a
P=$A/out/pilot
Q=$A/out/panel
W=$A/out/dcompare
SC=$W/score
PY=./.venv/bin/python
DRY_RUN=${DRY_RUN:-0}
KEYS=${DC_KEYS:-2.tc.17681,2.tc.2019,4.tc.13440,5.tc.22587,8.tc.15982,11.tc.22949}
GPU_MAX=${DC_GPU_MAX_MIB:-15300}
GPU_IDLE=${DC_GPU_IDLE_MIB:-3000}
RAM_MIN_GB=${DC_RAM_MIN_GB:-4}
EVERY=${DC_WATCH_S:-10}
THREADS=${DC_THREADS:-4}
DR=${DC_DONOR_RECORDS:-$Q/rec/ctx}
POOL=$SC/random_pool.pt
[ "$DR" = run ] && POOL=$SC/random_pool_run.pt
STAGES=${DC_STAGES:-a2_dunefsky a1_wcm_base a1_wcm_nec03}   # order = run order; e.g. DC_STAGES="a1_wcm_base a1_wcm_nec03"
FIT_A2=${DC_FIT_A2:-0}
FIT_A2_KEYS=${DC_FIT_A2_KEYS:-2.tc.17681,4.tc.13440,5.tc.22587,8.tc.15982}
WCM_BASE=$Q/wcm_l1e-3
WCM_TUNE=$A/out/tune/nec0.3/fit
DF=$W/dunefsky
WCM_A2=$W/wcm_a2_nec03

ENVV="PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 \
ARENA_A_BOS=1 FAMILY=gpt2-small-tcz DATA=data_public/gpt2-small STORES=$P/datapass SAE_PRELOAD=0 DEEP_BS=1 \
WCM_VERSION=v4 NVIDIA_TF32_OVERRIDE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ARENA_A_CTX_DIRS=$Q/rec/ctx \
OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS"

say() { echo "[dcompare $(date +%H:%M:%S)] $*" | { [ "$DRY_RUN" = 1 ] && cat || tee -a "$SC/run.log"; }; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
nlines() { grep -c . "$1" 2> /dev/null || echo 0; }

preflight() {   # read-only; prints every problem, rc 1 if any
  local bad=0 k f
  for f in "$PY" "$A/score_arena_a.py" "$P/datapass/top_ctx.pt" "$W/targets/prompts.jsonl" "$WCM_BASE/main/circuits" \
           "$WCM_TUNE/main/circuits"; do
    [ -e "$f" ] || { say "  missing: $f"; bad=1; }
  done
  for k in ${KEYS//,/ }; do
    [ -f "$Q/rec/ctx/$k.pt" ] || { say "  no record $Q/rec/ctx/$k.pt"; bad=1; }
    case $k in *.tc.*) ;; *) say "  $k is not a tc target (Dunefsky is tc only)"; bad=1 ;; esac
  done
  local n=$(ls "$DF"/circuits/*/p*.b*.i*.pt 2> /dev/null | wc -l)
  local want=$(( $(echo "$KEYS" | tr ',' '\n' | grep -c .) * 4 * 7 ))
  say "  Dunefsky circuits: $n / $want expected (K 4 x 7 branch values)"
  [ "$n" -gt 0 ] || { say "  no Dunefsky circuits under $DF"; bad=1; }
  [ "$n" -ge "$want" ] || say "  WARNING: Dunefsky run incomplete; scoring what is there"
  command -v nvidia-smi > /dev/null || { say "  nvidia-smi not found"; bad=1; }
  return "$bad"
}

ABORT=""
stage() {   # stage NAME command... : watchdog-guarded, one GPU process; rc 0 ok, 1 failed, 9 aborted
  local name=$1; shift
  local log=$SC/logs/$name.log
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ env $ENVV $*   > $log"; return 0; fi
  local m=$(gpu_mib)
  if [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ]; then
    say "REFUSED $name: the GPU already holds $m MiB (> $GPU_IDLE): another process is on it"; return 8
  fi
  say "start $name (log $log)"
  local t0=$(date +%s) r why="" rc
  setsid env $ENVV "$@" >> "$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2> /dev/null; do
    sleep "$EVERY"
    m=$(gpu_mib); r=$(avail_mib)
    echo "$(date +%s),$name,${m:-NA},${r:-NA}" >> "$SC/watch.csv"
    if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then why="GPU memory.used $m MiB > $GPU_MAX MiB"
    elif [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then why="host MemAvailable $r MiB < $RAM_MIN_GB GB"; fi
    if [ -n "$why" ]; then
      say "ABORT $name: $why"
      echo "$(date '+%F %T') $name: $why" > "$SC/ABORTED"
      kill -TERM -- -"$pid" 2> /dev/null; sleep 15; kill -KILL -- -"$pid" 2> /dev/null
      break
    fi
  done
  wait "$pid" 2> /dev/null; rc=$?
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$t0" "$(date +%s)" "$rc" "$why" >> "$SC/steps.tsv"
  say "end $name (exit $rc, $(( $(date +%s) - t0 )) s)"
  [ -n "$why" ] && { ABORT=$why; return 9; }
  [ "$rc" -eq 0 ] && return 0 || return 1
}

score() {   # score NAME SCOPE OUTDIR [scorer args]: the scorer + donor (with the one-time pool fallback) + summary
  local name=$1 scope=$2 out=$3; shift 3
  local args=(--scope "$scope" --out "$out" --keys "$KEYS" --threads "$THREADS" --selftest --donor "$@")
  local rc
  stage "$name" env ARENA_A_SCOPE="$scope" OUT="$out" "$PY" -X utf8 -u "$A/score_arena_a.py" "${args[@]}" \
    --donor-pool "$POOL" --donor-records "$DR"
  rc=$?
  if [ "$rc" = 1 ] && [ "$DR" != run ] && grep -q "independent random donors after filtering" "$SC/logs/$name.log" 2> /dev/null; then
    say "  $name: too few donors against $DR; retrying once with --donor-records run (random_pool_run.pt)"
    DR=run; POOL=$SC/random_pool_run.pt          # every later stage uses the fallback pool too
    stage "$name.poolrun" env ARENA_A_SCOPE="$scope" OUT="$out" "$PY" -X utf8 -u "$A/score_arena_a.py" "${args[@]}" \
      --donor-pool "$POOL" --donor-records run
    rc=$?
  fi
  [ "$rc" = 9 ] && return 9
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ CUDA_VISIBLE_DEVICES= $PY -X utf8 $A/score_arena_a.py --summary --out $out"
  else
    env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PY" -X utf8 -u "$A/score_arena_a.py" \
      --summary --out "$out" > "$SC/logs/$name.summary.log" 2>&1 || say "  $name: summary FAILED (see logs)"
  fi
  return "$rc"
}

[ "$DRY_RUN" = 1 ] || mkdir -p "$SC/logs"
say "dcompare scoring | keys $KEYS | out $SC | watchdog GPU > $GPU_MAX MiB / RAM < $RAM_MIN_GB GB every $EVERY s | donor records $DR | DC_FIT_A2=$FIT_A2$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
preflight || { say "preflight refused"; exit 8; }
FAILS=0
run_or_stop() { "$@"; local rc=$?; [ "$rc" = 9 ] && { say "aborted: $ABORT"; exit 9; }; [ "$rc" = 8 ] && exit 8
  [ "$rc" = 0 ] || FAILS=$((FAILS + 1)); }

want() { [[ " $STAGES " == *" $1 "* ]]; }
if want a2_dunefsky; then
  # only a python process whose argv runs run_dunefsky.py (2026-11-06: a bare `pgrep -f run_dunefsky.py` also matched
  # shells whose command string merely mentions it, e.g. a `while pgrep -f run_dunefsky.py` wait loop)
  if pgrep -f '^[^ ]*python[0-9.]* (-[^ ]+ )*[^ ]*run_dunefsky\.py' > /dev/null && [ "${DC_ALLOW_PARTIAL:-0}" != 1 ]; then
    say "SKIP a2_dunefsky: run_dunefsky.py is still running (DC_ALLOW_PARTIAL=1 scores what is there; a rerun later"
    say "  scores the rest, the scorer resumes per unit)"; FAILS=$((FAILS + 1))
  else
    run_or_stop score a2_dunefsky a2 "$SC/a2_dunefsky" --dunefsky "$DF"
  fi
fi
want a1_wcm_base && run_or_stop score a1_wcm_base a1 "$SC/a1_wcm_base" --wcm "l1e-3=$WCM_BASE"
want a1_wcm_nec03 && run_or_stop score a1_wcm_nec03 a1 "$SC/a1_wcm_nec03" --wcm "l1e-3=$WCM_TUNE"
if [ "$FIT_A2" = 1 ]; then
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$WCM_A2"; echo "$FIT_A2_KEYS" | tr ',' '\n' > "$W/wcm_a2_keys.txt"; }
  run_or_stop stage a2_wcm_fit env ARENA_A_SCOPE=a2 OUT="$WCM_A2" CTX_DIR="$Q/rec/ctx" TARGETS="$W/wcm_a2_keys.txt" \
    MODE=main LAM=1e-3 NEC_W=0.3 ORDER=list SKIP_SPEC=1 CURVE_STRIDE=20 "$PY" -X utf8 -u "$A/driver_tcz.py"
  run_or_stop score a2_wcm_score a2 "$SC/a2_wcm_nec03" --wcm "l1e-3=$WCM_A2"
fi
say "DONE ($FAILS failed stage(s))"
[ "$DRY_RUN" = 1 ] || { [ "$FAILS" = 0 ] && touch "$SC/SCORE_DONE"; }
[ "$FAILS" = 0 ]
