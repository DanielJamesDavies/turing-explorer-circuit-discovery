#!/bin/bash
# RESTORE (2026-11-08): on a FRESH pod, after scripts/pod_quickstart.sh, unpack the 082 chain's archives from the
# volume back into place (outputs AND logs/state done-markers), so re-running the same launch command resumes the
# chain where it stopped (done stages skipped; fits resume per target).
#
#   bash /root/turing/scripts/pod_restore.sh 2>&1 | tee /root/restore.log      # then the dry run, then the launch
#
# Archives: every RESULTS/pub-*.tgz except corpus / S2 / S3 copies that an S10-stores archive supersedes, unpacked
# OLDEST FIRST (so later stage archives and the newest pub-save-*.tgz win), each md5-checked first.
# Two path kinds live in the same archives (as the chain writes them):
#   root/turing-tl/...   (absolute paths, leading / stripped)  -> unpacked at DEST_ROOT (/)
#   everything else      (relative to the public tree)          -> unpacked at PUB (/root/turing)
# ENV: RESULTS [/workspace/results]  PUB [/root/turing]  DEST_ROOT [/]  DRY_RUN [0] (list what it would do)
set -uo pipefail
RESULTS=${RESULTS:-/workspace/results}
PUB=${PUB:-/root/turing}
DEST_ROOT=${DEST_ROOT:-/}
DRY=${DRY_RUN:-0}
say() { echo "[restore $(date +%H:%M:%S)] $*"; }
[ -d "$PUB" ] || { say "no $PUB: run scripts/pod_quickstart.sh first"; exit 1; }

# explicit STAGE order (not file dates: a downloaded or re-copied archive has a new mtime): public S1..S10 per model,
# then TuringLLM T0..T12, then the save-now archives by their timestamp (newest last, so it wins)
mapfile -t all < <(cd "$RESULTS" && ls -1 pub-*.tgz 2> /dev/null | awk '{
  k = "0 00 " $0
  if (match($0, /^pub-(pythia-70m|gpt2-small)-S[0-9]+-/)) { s = substr($0, RSTART, RLENGTH); sub(/.*-S/, "", s); sub(/-$/, "", s); k = sprintf("1 %02d %s", s, $0) }
  else if (match($0, /^pub-turingllm-T[0-9]+-/)) { s = substr($0, RSTART, RLENGTH); sub(/.*-T/, "", s); sub(/-$/, "", s); k = sprintf("2 %02d %s", s, $0) }
  else if ($0 ~ /^pub-save-/) k = "3 00 " $0
  print k }' | sort | awk '{print $3}')
[ ${#all[@]} -gt 0 ] || { say "no pub-*.tgz in $RESULTS"; exit 1; }
use=()
for a in "${all[@]}"; do
  m=$(echo "$a" | sed -E 's/^pub-(pythia-70m|gpt2-small)-.*/\1/;t;s/.*//')
  if [ -n "$m" ] && [[ "$a" =~ -S(1-corpus|2-native|3-stores)\.tgz$ ]] && [ -f "$RESULTS/pub-$m-S10-stores.tgz" ]; then
    say "skip $a (inside pub-$m-S10-stores.tgz)"; continue
  fi
  use+=("$a")
done
say "${#use[@]} archives to unpack, oldest first"
n=0
for a in "${use[@]}"; do
  n=$((n + 1)); md5=${a%.tgz}.md5
  if [ -f "$RESULTS/$md5" ]; then
    (cd "$RESULTS" && md5sum -c --quiet "$md5") || { say "MD5 MISMATCH $a: stopping"; exit 1; }
  else
    say "  (no $md5: unpacking $a without an md5 check)"
  fi
  if [ "$DRY" = 1 ]; then say "[$n/${#use[@]}] would unpack $a"; continue; fi
  tar -xzf "$RESULTS/$a" --no-same-owner -C "$DEST_ROOT" --wildcards 'root/*' 2> /dev/null   # no such members: fine
  tar -xzf "$RESULTS/$a" --no-same-owner -C "$PUB" --exclude='root/*' || { say "unpack FAILED $a"; exit 1; }
  say "[$n/${#use[@]}] unpacked $a"
done
ST=$PUB/experiments/082-public-runs/logs/state
if [ -d "$ST" ]; then   # stale want.* priority flags are cleared by chain.sh itself at start
  say "done markers restored: $(ls "$ST"/*.done 2> /dev/null | wc -l) ($(ls "$ST" | grep -c 'turingllm.T.*done') TuringLLM)"
fi
say "RESTORED. Next: the dry run, then the SAME launch command as before (it resumes from the markers)."
