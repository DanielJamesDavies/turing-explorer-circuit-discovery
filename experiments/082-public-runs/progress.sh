#!/bin/bash
# One-screen view of the 082 public-model chain (DAN-39 B8). Read-only: it only reads markers, logs, row files and
# nvidia-smi, so it is safe to run at any time, as often as wanted (e.g. `watch -n 60 bash .../progress.sh`).
#
#   bash experiments/082-public-runs/progress.sh            # the pod run (WORK = experiments/082-public-runs)
#   LIMIT=2 bash experiments/082-public-runs/progress.sh    # the local smoke chain (WORK = 082/smoke)
# Same WORK / LOGS / STATE / RESULTS / DATA_ROOT / STORE_ROOT / MODELS defaults as chain.sh.
shopt -s nullglob
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
cd "$ROOT" || exit 1
E=experiments/082-public-runs
if [ -n "${LIMIT:-}" ]; then WORK=${WORK:-$E/smoke_v2}; RESULTS=${RESULTS:-$WORK/results}
else WORK=${WORK:-$E}; RESULTS=${RESULTS:-/workspace/results}; fi
LOGS=${LOGS:-$WORK/logs}
STATE=${STATE:-$LOGS/state}
DATA_ROOT=${DATA_ROOT:-data_public}
STORE_ROOT=${STORE_ROOT:-outputs_public}
MODELS=${MODELS:-"pythia-70m gpt2-small"}

lines() { local n=0 f; for f in "$@"; do n=$((n + $(wc -l < "$f"))); done; echo "$n"; }
errs() { local n=0 f; for f in "$@"; do n=$((n + $(grep -c '"error"' "$f"))); done; echo "$n"; }
count() { local a=("$@"); echo "${#a[@]}"; }
mins_since() { [ -f "$1" ] || { echo ""; return; }; echo $(( ($(date +%s) - $(date -r "$1" +%s)) / 60 )); }
rate() {   # rate DONE TOTAL STARTFILE -> "x/min, ETA h"
  local done=$1 total=$2 m
  m=$(mins_since "$3")
  if [ -z "$m" ] || [ "$m" -lt 1 ] || [ "$done" -lt 1 ]; then echo ""; return; fi
  awk -v d="$done" -v t="$total" -v m="$m" 'BEGIN { r = d / m; printf "%.1f/min", r; if (t > d) printf ", ETA %.1f h", (t - d) / r / 60 }'
}

echo "== 082 public runs | $(date '+%F %T') | $(git --no-pager log -1 --format='%h %s' 2> /dev/null)"
echo "-- chain (markers in $STATE)"
if [ -d "$STATE" ]; then
  for f in "$STATE"/*.start; do
    k=$(basename "$f" .start)
    if [ -f "$STATE/$k.done" ]; then st="done $(cat "$STATE/$k.done")"; else st="RUNNING since $(cat "$f") ($(mins_since "$f") min)"; fi
    printf '   %-28s %s\n' "$k" "$st"
  done
  for f in "$STATE"/lam_*; do printf '   %-28s %s\n' "$(basename "$f")" "$(tr -d '\n' < "$f")"; done
  [ -s "$STATE/failed.list" ] && echo "   FAILED arms: $(sort -u "$STATE/failed.list" | tr '\n' ' ')"
  [ -f "$LOGS/chain.events" ] && { echo "   last events:"; tail -n 5 "$LOGS/chain.events" | sed 's/^/     /'; }
else
  echo "   (no state yet)"
fi

for F in $MODELS; do
  OUT=$WORK/out_$F
  echo "-- $F"
  m=$DATA_ROOT/$F/corpus_manifest.json
  if [ -f "$m" ]; then
    echo "   corpus   $(grep -o '"sequences": [0-9]*' "$m" | head -1 | cut -d' ' -f2) sequences, duplicate_window_rate" \
         "$(grep -o '"duplicate_window_rate": [0-9.e-]*' "$m" | head -1 | cut -d' ' -f2), $(count "$DATA_ROOT/$F"/shard_*.npy) shards"
  else
    l=$LOGS/$F/corpus.log; [ -f "$l" ] && echo "   corpus   $(tail -n 1 "$l" | cut -c1-110)"
  fi
  for s in dp_pass1; do
    l=$LOGS/$F/$s.log
    [ -f "$l" ] && echo "   $s $(tail -n 1 "$l" | tr '\r' '\n' | tail -n 1 | cut -c1-110)"
  done
  [ -f "$STORE_ROOT/$F/records_check.json" ] && echo "   records  $(grep -o '"all_checks_pass": [a-z]*' "$STORE_ROOT/$F/records_check.json" | sort | uniq -c | tr '\n' ' ')"
  unset N_main N_pilot N_unweighted N_stable N_nulls
  for n in main pilot unweighted stable nulls; do
    t=$WORK/targets_${F}_$n.txt; [ -n "${LIMIT:-}" ] && t=$WORK/targets_${F}_$n.limit.txt
    [ -f "$t" ] && printf -v "N_$n" '%s' "$(wc -l < "$t")"
  done
  [ -f "$WORK/targets_${F}_main.txt" ] && echo "   targets  main ${N_main:-?}  pilot ${N_pilot:-?}  nulls ${N_nulls:-?}  stable ${N_stable:-?}  unweighted ${N_unweighted:--}"
  if [ -d "$OUT/sweep" ]; then
    printf '   pilot    '
    for d in "$OUT"/sweep/circuits/*/; do printf '%s %d  ' "$(basename "$d")" "$(count "$d"*.pt)"; done
    echo "| eval rows $(lines "$OUT"/sweep/eval.*.jsonl), errors $(errs "$OUT"/sweep/eval.*.jsonl)"
  fi
  if [ -d "$OUT/main" ]; then
    nc=$(count "$OUT"/main/circuits/*.pt)
    st=$(lines "$OUT"/main/status.*.jsonl)
    key=$(cd "$STATE" 2> /dev/null && ls ./"$F".S*.main.start 2> /dev/null | head -1)
    echo "   main     circuits $nc / ${N_main:-?}, status rows $st (errors $(errs "$OUT"/main/status.*.jsonl))," \
         "eval rows $(lines "$OUT"/main/eval.*.jsonl), spec rows $(lines "$OUT"/main/spec.*.jsonl)" \
         "$( [ -n "$key" ] && rate "$st" "${N_main:-0}" "$STATE/${key#./}")"
  fi
  for a in alpha1:alpha1 harder_nulls:nulls stable_core:stable_core strict:rows donor:rows donor_unweighted:rows; do
    d=$OUT/${a%%:*}; stem=${a#*:}
    [ -d "$d" ] || continue
    f=()
    for x in "$d/$stem.jsonl" "$d/$stem".shard*.jsonl; do [ -f "$x" ] && f+=("$x"); done
    echo "   $(printf '%-8s' "${a%%:*}") rows $(lines "${f[@]}") (errors $(errs "${f[@]}"))$( [ -d "$d/circuits" ] && echo ", circuits $(count "$d"/circuits/*.pt "$d"/circuits/*/*.pt)")"
  done
  gu=$(cat "$LOGS/$F"/*/*.shard*.log 2> /dev/null | grep -c "GIVING UP")
  [ "$gu" -gt 0 ] && echo "   WARNING: $gu shard(s) gave up (grep 'GIVING UP' $LOGS/$F/*/*.log)"
done

tlm=("$STATE"/turingllm.*)
if [ -d "$LOGS/turingllm" ] || [ ${#tlm[@]} -gt 0 ]; then   # the optional TuringLLM track
  if [ -n "${LIMIT:-}" ]; then TLW=${TL_WORK:-$WORK/out_turingllm}; else TLW=${TL_WORK:-${TL_ROOT:-/root/turing-tl}/$E/out_turingllm}; fi
  if [ -n "${LIMIT:-}" ]; then V2=${TL_V2_OUT:-$TLW/out_v2}
  else V2=${TL_V2_OUT:-${TL_ROOT:-/root/turing-tl}/experiments/094-tl-v2-records/out_v2}; fi
  echo "-- turingllm track, protocol v2 refit (markers turingllm.T*; logs $LOGS/turingllm; work $TLW; OUT $V2" \
       "marked $(cat "$V2/WCM_VERSION" 2> /dev/null || echo -))"
  echo "   scan     shard files $(count "$TLW"/scan/*shard*.pt), merged $( [ -f "$TLW/scan/scan_top512.pt" ] && echo yes || echo no)"
  echo "   records  build rows $(lines "$TLW"/records/build.shard*.jsonl) (errors $(errs "$TLW"/records/build.shard*.jsonl))," \
       "files $(count "$TLW"/ctx_fresh/*.pt), verify rows $(lines "$TLW"/records/verify.shard*.jsonl)" \
       "(not ok $(cat "$TLW"/records/verify.shard*.jsonl 2> /dev/null | grep -vc '"ok": true'))"
  if [ -d "$V2/main" ]; then
    st=$(lines "$V2"/main/status.*.jsonl)
    echo "   main     circuits $(count "$V2"/main/circuits/*.pt) / 8000, status rows $st (errors $(errs "$V2"/main/status.*.jsonl))," \
         "eval rows $(lines "$V2"/main/eval.*.jsonl), spec rows $(lines "$V2"/main/spec.*.jsonl)" \
         "$( [ -f "$STATE/turingllm.T3.start" ] && rate "$st" 8000 "$STATE/turingllm.T3.start")"
  fi
  f=("$V2"/alpha1/*.jsonl); [ ${#f[@]} -gt 0 ] && echo "   alpha1   rows $(lines "${f[@]}") (errors $(errs "${f[@]}"))"
  if [ -d "$V2/sweep" ]; then
    printf '   sweep    '
    for d in "$V2"/sweep/circuits/*/; do printf '%s %d  ' "$(basename "$d")" "$(count "$d"*.pt)"; done
    echo "| eval rows $(lines "$V2"/sweep/eval.*.jsonl), errors $(errs "$V2"/sweep/eval.*.jsonl)"
  fi
  f=("$V2"/stable_core/*.jsonl); [ ${#f[@]} -gt 0 ] && echo "   stable   rows $(lines "${f[@]}") (errors $(errs "${f[@]}"))"
  f=("$TLW"/harder_nulls/nulls*.jsonl); [ ${#f[@]} -gt 0 ] && echo "   nulls    rows $(lines "${f[@]}") (errors $(errs "${f[@]}"))"
  for d in rescore_asrun1000_asrun rescore_asrun1000_ctx_v2; do
    f=("$TLW/$d"/*/*.jsonl)
    [ ${#f[@]} -gt 0 ] && echo "   T9 $(printf '%-24s' "${d#rescore_asrun1000_}") rows $(lines "${f[@]}") (errors $(errs "${f[@]}"))"
  done
  for s in ref attr fitted exact ge; do
    f=("$TLW/dan159/$s"/*.jsonl); [ ${#f[@]} -gt 0 ] && echo "   083 $(printf '%-6s' "$s") rows $(lines "${f[@]}") (errors $(errs "${f[@]}"))"
  done
  if [ -n "${LIMIT:-}" ]; then NOR=${TL_NOR_OUT:-$TLW/out_v2_noR}
  else NOR=${TL_NOR_OUT:-${TL_ROOT:-/root/turing-tl}/experiments/094-tl-v2-records/out_v2_noR}; fi
  f=("$NOR"/sweep/eval.*.jsonl); [ ${#f[@]} -gt 0 ] && echo "   T10 noR  eval rows $(lines "${f[@]}") (errors $(errs "${f[@]}"))"
  for d in "$V2/donor/main" "$V2/donor/sweep" "$NOR/donor/sweep"; do   # T11 / T12: donor-background rows
    f=("$d"/rows*.jsonl)
    [ ${#f[@]} -gt 0 ] && echo "   donor $(printf '%-18s' "${d#"${d%/*/*/*}/"}") rows $(lines "${f[@]}") (errors $(errs "${f[@]}"))"
  done
  for d in "$LOGS"/turingllm/*/; do
    printf '   %-24s %s log lines, last: %s\n' "$(basename "$d")" "$(cat "$d"*.log 2> /dev/null | wc -l)" \
      "$(cat "$d"*.log 2> /dev/null | tail -n 1 | cut -c1-90)"
  done
fi

jf=("$STATE"/jobs/*.live.json)                  # per-job error accounting (job_errors.py, 2026-10-26)
if [ ${#jf[@]} -gt 0 ] || [ -s "$LOGS/job_errors.jsonl" ]; then
  python3 - "$LOGS/job_errors.jsonl" "${jf[@]}" << 'EOF'
import json, sys
jobs = {}
try:
    for ln in open(sys.argv[1]):
        try:
            r = json.loads(ln); jobs[r.get("job")] = dict(r, state="failed" if r.get("failed") else "done")
        except ValueError:
            pass
except OSError:
    pass
for f in sys.argv[2:]:
    try:
        r = json.load(open(f))
    except (OSError, ValueError):
        continue
    if str(r.get("time", "")) >= str(jobs.get(r.get("job"), {}).get("time", "")):
        jobs[r.get("job")] = r
bad = [j for j in jobs.values() if j.get("n_err")]
print("-- jobs: %d checked, %d with error rows%s" % (len(jobs), len(bad), "" if bad else " (none)"))
for j in sorted(bad, key=lambda j: (not j.get("failed"), str(j.get("job")))):
    print("   !! %s: %d of %d fits errored%s [%s] first: %s" % (j.get("job"), j.get("n_err", 0), j.get("n", 0),
          " -- FAILED (early gate)" if j.get("failed") else "", j.get("state", "done"), str(j.get("first_error"))[:110]))
EOF
fi

echo "-- GPUs"
if command -v nvidia-smi > /dev/null 2>&1; then
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
    | awk -F', ' '{ printf "   GPU %s %4s  %s / %s\n", $1, $2, $3, $4 }'
  echo "   processes: driver/companions $(pgrep -fc 'driver.py|harder_nulls.py|rescore_alpha1.py|stable_core.py|strict_eval.py|donor_test.py')," \
       "086 $(pgrep -fc '086-heldout-rebuild/'), 094 $(pgrep -fc '094-tl-v2-records/'), 083 $(pgrep -fc '083-v1-comparison/[a-z_]*\.py')," \
       "data passes $(pgrep -fc run_data_pass.py), corpus $(pgrep -fc prepare_corpus.py)," \
       "MPS $(pgrep -f nvidia-cuda-mps-control > /dev/null && echo on || echo off)"
else
  echo "   (no nvidia-smi)"
fi
echo "-- disk"
df -h "$ROOT" "$RESULTS" 2> /dev/null | awk 'NR == 1 || !seen[$0]++' | sed 's/^/   /'
echo "-- results on $RESULTS: $(count "$RESULTS"/*.tgz) archives, $(du -sh "$RESULTS" 2> /dev/null | cut -f1);" \
     "verified by the chain: $( [ -f "$STATE/verified.list" ] && sort -u "$STATE/verified.list" | wc -l || echo 0)"
[ -f "$STATE/verified.list" ] && tail -n 3 "$STATE/verified.list" | sed 's/^/   /'
exit 0
