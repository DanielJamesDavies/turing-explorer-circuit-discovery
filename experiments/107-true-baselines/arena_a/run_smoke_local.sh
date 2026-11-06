#!/bin/bash
# Arena A: the 2-target LOCAL smoke of run_pod.sh's stage (plan A10 acceptance: "a 2-target smoke through P2-P7"), on
# THIS machine: one 16 GB RTX-class GPU under WSL. It runs run_pod.sh itself, with smoke settings, from P4:
#   reused    the pilot's data pass (out/pilot/datapass, wikitext smoke corpus data_public/gpt2-small) and the A6 targets
#             dir out/targets_smoke (records ctx/, prompts.jsonl), read only: P0-P3 are taken as done (AA_FROM=P4), so
#             nothing is fetched or built (the envs ge107 / tc107 and every asset are already local)
#   targets   SMOKE_KEYS [3.tc.13057,3.z.23269]: one tc, one z, both shallow (L3), so the 16 GB card is not the limit
#   P4        Ge (K = 2 prompts, 6 tau) + Dunefsky (K = 2, branches 5,10,20; the "beyond published range" knob at 100
#             branches on 1 prompt), plus the canary. ON THE CPU: the ge107 / tc107 wheels here are CPU builds, and
#             their pinned torch (2.3.1 / 2.2.0) cannot drive the local sm_120 card (README section 1.2). So the smoke
#             checks the commands, files, resume and scoring path, not the baselines' GPU path (the pod's P4 canary
#             does that). AA_BASELINE_DEVICE=cpu
#   P5        WCM v4, 1 lambda (SMOKE_LAMBDAS [1e-3]), 400 steps, scope a1 (2 targets) and a2 (the tc target), on cuda
#             (DEEP_BS=1, as every local fit on this card)
#   P4h       the held-out baselines: --emit-heldout-prompts (1 per target), Ge (6 tau) and Dunefsky (5,10,20) on them (CPU)
#   P6        score_arena_a.py --donor --selftest (cuda), --emit-ge-jobs, ge_recovery_tcz.py --check (CPU, as P4)
#   P6b       score_perprompt.py (A8b) on 1 train + 1 held prompt per target, --check (cuda), its Ge recovery (CPU), summary
#   P7        merge + --summary only (AA_PACK=0: no tgz, no RESULTS copy)
# ONE GPU process at a time (AA_GPUS=0: one worker, one WCM shard, the scorer runs scope a1 then a2). float32, TF32 off
# (driver_tcz.py puts matmul precision back to "highest"; NVIDIA_TF32_OVERRIDE=0 for every process besides).
# Watchdog every SMOKE_WATCH_S [30] s into WORK/watch.csv (epoch, stage, GPU MiB, host MemAvailable MiB): ABORT (the
# whole process group, written to WORK/ABORTED) when GPU memory.used > SMOKE_GPU_MAX_MIB [15300] or host
# MemAvailable < SMOKE_RAM_MIN_GB [4] GB. Start refused when the GPU already holds > SMOKE_GPU_IDLE_MIB [1500] MiB.
# Timings: run_pod.sh writes WORK/steps.tsv (per stage) and WORK/cmds.tsv (per command); this script writes
# WORK/smoke_summary.md (stage and command seconds, peak GPU MiB per stage, min free RAM, output counts).
# Resumable: rerun = run_pod.sh skips the finished stages / circuits / fits / rows. STOP_EPOCH is passed through.
#
#   WSL, repo root:  DRY_RUN=1 bash experiments/107-true-baselines/arena_a/run_smoke_local.sh     # print, run nothing
#                    STOP_EPOCH=$(date -d 'tomorrow 07:30' +%s) bash experiments/107-true-baselines/arena_a/run_smoke_local.sh
# Exit: run_pod.sh's rc; 9 = aborted by the watchdog; 8 = preflight refused.
set -uo pipefail
shopt -s nullglob
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT" || exit 1
A=experiments/107-true-baselines/arena_a
DRY_RUN=${DRY_RUN:-0}
W=${SMOKE_WORK:-$A/out/smoke_pod}
KEYS=${SMOKE_KEYS:-3.tc.13057,3.z.23269}
GPU_MAX=${SMOKE_GPU_MAX_MIB:-15300}
RAM_MIN_GB=${SMOKE_RAM_MIN_GB:-4}
GPU_IDLE=${SMOKE_GPU_IDLE_MIB:-1500}
EVERY=${SMOKE_WATCH_S:-30}
STO=$A/out/pilot/datapass
TGT=$A/out/targets_smoke

say() { echo "[smoke_a $(date +%H:%M:%S)] $*"; }

# the smoke settings of run_pod.sh (every name is documented in its header)
SMOKE_ENV=(
  AA_WORK="$W" AA_GPUS=0 AA_FROM="${AA_FROM:-P4}" STOP_AT=P7 AA_PACK=0   # AA_FROM / AA_REDO from the caller pass through
  AA_DONOR_RECORDS=run   # SMOKE ONLY: R pool filtered against the run's 2 target records (the 2048-window corpus
                         # keeps 28 of 512 donors against all 48 records, < donor_test's 32; 451 against these 2)
  AA_STORES="$STO" AA_TARGETS_DIR="$TGT" AA_N_SEQ=0 AA_DATA_SRC=data_public/gpt2-small
  AA_KEYS="$KEYS" AA_LAMBDAS="${SMOKE_LAMBDAS:-1e-3}" AA_SCOPES="a1 a2"
  AA_BASELINE_DEVICE=cpu AA_GE_K=2 AA_GE_NTAU=6
  AA_DUNEFSKY_K=2 AA_DUNEFSKY_BRANCHES=5,10,20 AA_DUNEFSKY_EXT=1 AA_DUNEFSKY_EXT_K=1 AA_DUNEFSKY_EXT_BRANCHES=100
  DEEP_BS=1 NVIDIA_TF32_OVERRIDE=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
  AA_ALLOW_UNTUNED=1     # P5 runs plain v4 when out/tune/CHOSEN.env does not exist yet (if it does, the smoke uses it)
  AA_HELD_K=1 AA_PP_PROMPTS=1 AA_PP_TAU_STRIDE=1 AA_PP_CHECK=1   # P4h + P6b (A8b) at smoke size: 1 held prompt each
)
[ -n "${STOP_EPOCH:-}" ] && SMOKE_ENV+=(STOP_EPOCH="$STOP_EPOCH")

preflight() {   # read-only checks; prints every problem, rc 1 if any
  local bad=0 k
  for f in "$STO/top_ctx.pt" "$STO/mid_ctx.pt" "$STO/neg_ctx.pt" "$STO/data_pass_manifest.json" \
           "$TGT/targets.txt" "$TGT/prompts.jsonl" "$HOME/venvs/ge107/bin/ge_env.sh" "$HOME/venvs/tc107/bin/tc_env.sh" \
           experiments/035-transcoder-compare/transcoders_ct/meta.json; do
    [ -e "$f" ] || { say "  missing: $f"; bad=1; }
  done
  for k in ${KEYS//,/ }; do
    grep -qxF "$k" "$TGT/targets.txt" 2> /dev/null || { say "  $k is not in $TGT/targets.txt"; bad=1; }
    [ -f "$TGT/ctx/$k.pt" ] || { say "  no record $TGT/ctx/$k.pt"; bad=1; }
  done
  echo "$KEYS" | tr ',' '\n' | grep -qF .tc. || { say "  SMOKE_KEYS has no tc target (Dunefsky / a2 need one)"; bad=1; }
  echo "$KEYS" | tr ',' '\n' | grep -qE '\.z2?\.' || { say "  SMOKE_KEYS has no z target"; bad=1; }
  command -v nvidia-smi > /dev/null || { say "  nvidia-smi not found"; bad=1; }
  return "$bad"
}

gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' '; }
avail_mib() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' /proc/meminfo; }
cur_stage() {   # the run_pod.sh stage that started last and is not done
  local f s
  f=$(ls -t "$W/state"/*.start 2> /dev/null | head -n 1)
  [ -n "$f" ] || { echo idle; return; }
  s=$(basename "$f" .start)
  [ -f "$W/state/$s.done" ] && echo "idle" || echo "$s"
}

write_summary() {   # WORK/smoke_summary.md from steps.tsv, cmds.tsv, watch.csv and the output dirs
  local S=$W/smoke_summary.md
  local st t0 t1 rc pk mn   # local: the steps.tsv read below must not overwrite the script's rc (2026-11-05: "rc ")
  {
    echo "# Arena A local smoke ($(date '+%F %T'))"
    echo
    echo "Targets $KEYS; settings: ${SMOKE_ENV[*]}"
    [ -f "$W/ABORTED" ] && { echo; echo "**ABORTED:** $(cat "$W/ABORTED")"; }
    echo
    echo "| stage | wall s | rc | peak GPU MiB | min host MemAvailable MiB |"
    echo "|---|---|---|---|---|"
    if [ -f "$W/steps.tsv" ]; then
      while IFS=$'\t' read -r st t0 t1 rc; do
        pk=$(awk -F, -v s="$st" '$2 == s && $3 + 0 > m { m = $3 + 0 } END { print m + 0 }' "$W/watch.csv" 2> /dev/null)
        mn=$(awk -F, -v s="$st" '$2 == s && (n == "" || $4 + 0 < n) { n = $4 + 0 } END { print n }' "$W/watch.csv" 2> /dev/null)
        echo "| $st | $((t1 - t0)) | $rc | $pk | $mn |"
      done < "$W/steps.tsv"
    fi
    echo
    echo "Peak GPU memory.used over the run: $(awk -F, '$3 + 0 > m { m = $3 + 0 } END { print m + 0 }' "$W/watch.csv" 2> /dev/null) MiB (abort above $GPU_MAX)."
    echo
    echo "| command (log) | script + args | s | rc |"
    echo "|---|---|---|---|"
    [ -f "$W/cmds.tsv" ] && awk -F'\t' '{ printf "| %s | `%s` | %d | %s |\n", $1, substr($5, 1, 110), $3 - $2, $4 }' "$W/cmds.tsv"
    echo
    echo "Outputs: Ge circuits $(ls "$W"/ge/circuits/*/p*.pt 2> /dev/null | wc -l), Dunefsky circuits" \
      "$(ls "$W"/dunefsky/circuits/*/p*.pt 2> /dev/null | wc -l), WCM ok rows" \
      "$(cat "$W"/wcm/*/main/status.shard*.jsonl 2> /dev/null | grep -c '"ok": true'), score rows" \
      "$(cat "$W"/score/all/rows.jsonl 2> /dev/null | wc -l), donor rows $(cat "$W"/score/all/donor.jsonl 2> /dev/null | wc -l)," \
      "Ge recovery rows $(cat "$W"/score/all/ge_recovery*.jsonl 2> /dev/null | wc -l)" \
      "(errors: $(cat "$W"/ge/status.jsonl "$W"/dunefsky/status.jsonl "$W"/score/all/ge_recovery*.jsonl 2> /dev/null | grep -c '"status": "error"'))."
    echo
    echo "Held-out baselines: Ge $(ls "$W"/ge_heldout/circuits/*/p*.pt 2> /dev/null | wc -l), Dunefsky" \
      "$(ls "$W"/dunefsky_heldout/circuits/*/p*.pt 2> /dev/null | wc -l); per-prompt rows" \
      "$(cat "$W"/perprompt/all/rows.jsonl 2> /dev/null | wc -l), its Ge recovery rows" \
      "$(cat "$W"/perprompt/all/ge_recovery*.jsonl 2> /dev/null | wc -l) (summary $W/perprompt/all/summary.md)."
    echo
    echo "Scorer summary: $W/score/all/summary.md. Logs: $W/logs/ (events.log, P4..P7)."
  } > "$S.tmp" && mv "$S.tmp" "$S"
  say "summary -> $S"
}

say "arena A local smoke | keys $KEYS | WORK $W | watchdog: GPU > $GPU_MAX MiB or RAM < $RAM_MIN_GB GB aborts, every $EVERY s$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
if ! preflight; then say "preflight refused"; exit 8; fi
if [ "$DRY_RUN" = 1 ]; then
  echo "    \$ (watchdog) every $EVERY s: nvidia-smi memory.used, /proc/meminfo MemAvailable -> $W/watch.csv; abort above $GPU_MAX MiB / below $RAM_MIN_GB GB"
  echo "    \$ env ${SMOKE_ENV[*]} setsid bash $A/run_pod.sh > $W/run_pod.out 2>&1   # its plan:"
  env "${SMOKE_ENV[@]}" DRY_RUN=1 bash "$A/run_pod.sh"
  rc=$?
  say "dry run rc $rc"
  exit "$rc"
fi

m=$(gpu_mib)
if [ -n "$m" ] && [ "$m" -gt "$GPU_IDLE" ]; then
  say "the GPU already holds $m MiB (> $GPU_IDLE): another process is on it; not starting"; exit 8
fi
mkdir -p "$W" || exit 1
rm -f "$W/ABORTED"
say "start run_pod.sh (log $W/run_pod.out)"
env "${SMOKE_ENV[@]}" setsid bash "$A/run_pod.sh" >> "$W/run_pod.out" 2>&1 &
PID=$!                                                   # setsid: PID is the process-group id of the whole run
trap 'kill -TERM -- -$PID 2> /dev/null' INT TERM
abort() {
  say "ABORT: $1"
  echo "$(date '+%F %T') $1" > "$W/ABORTED"
  kill -TERM -- -"$PID" 2> /dev/null
  sleep 15
  kill -KILL -- -"$PID" 2> /dev/null
}
while kill -0 "$PID" 2> /dev/null; do
  m=$(gpu_mib); r=$(avail_mib); st=$(cur_stage)
  echo "$(date +%s),$st,${m:-NA},${r:-NA}" >> "$W/watch.csv"
  if [ -n "$m" ] && [ "$m" -gt "$GPU_MAX" ]; then abort "GPU memory.used $m MiB > $GPU_MAX MiB (stage $st)"; break; fi
  if [ -n "$r" ] && [ "$r" -lt $((RAM_MIN_GB * 1024)) ]; then abort "host MemAvailable $r MiB < $RAM_MIN_GB GB (stage $st)"; break; fi
  sleep "$EVERY"
done
wait "$PID"
rc=$?
[ -f "$W/ABORTED" ] && rc=9
write_summary
tail -n 3 "$W/run_pod.out"
say "smoke finished (rc ${rc:-unset})"
exit "${rc:-1}"
