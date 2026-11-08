#!/bin/bash
# SAVE NOW (2026-11-08): pack the 082 chain's IN-FLIGHT work to the volume before a manual pod stop, so a stage that
# has not finished (and so has no stage archive yet) is not lost. Finished stages are already archived by the chain.
# Read-only for the run: it only reads the work dirs and writes one tgz + md5 to CKPT and RESULTS.
#
#   bash /root/turing/scripts/pod_save_now.sh            # then check the "verified" line, then stop the pod
#
# What it packs (whatever exists; paths kept exactly as the chain's own archives keep them, so
# scripts/pod_restore.sh can unpack it on top of the stage archives):
#   public tree  experiments/082-public-runs/logs   (chain.events, logs/state = the done markers, every launch log)
#                experiments/082-public-runs/out_*  (public OUTs: any arm still running)
#   TuringLLM    /root/turing-tl/experiments/094-tl-v2-records/out_v2/{sweep,stable_core,donor,alpha1,WCM_VERSION}
#                /root/turing-tl/experiments/094-tl-v2-records/out_v2_noR
#                /root/turing-tl/experiments/082-public-runs/out_turingllm/{harder_nulls,dan159,lists,chunks}
#   The big, already-archived parts (out_v2/main circuits, ctx_fresh records, stores, corpora) are left out: their
#   stage archives (T2 / T3 / S10) are on the volume. SAVE_MAIN=1 also packs out_v2/main (+ ~0.5 GB).
# ENV: PUB [/root/turing]  TLR [/root/turing-tl]  CKPT [/root/ckpt]  RESULTS [/workspace/results]  SAVE_MAIN [0]
set -uo pipefail
PUB=${PUB:-/root/turing}
TLR=${TLR:-/root/turing-tl}
CKPT=${CKPT:-/root/ckpt}
RESULTS=${RESULTS:-/workspace/results}
STAMP=$(date +%Y%m%d-%H%M)
NAME=pub-save-$STAMP
Z="-z"; command -v pigz > /dev/null 2>&1 && Z="-I pigz"
say() { echo "[save $(date +%H:%M:%S)] $*"; }

cd "$PUB" || { say "no $PUB"; exit 1; }
paths=()
for p in experiments/082-public-runs/logs experiments/082-public-runs/out_*; do [ -e "$p" ] && paths+=("$p"); done
V2=$TLR/experiments/094-tl-v2-records/out_v2
TW=$TLR/experiments/082-public-runs/out_turingllm
for p in "$V2"/sweep "$V2"/stable_core "$V2"/donor "$V2"/alpha1 "$V2"/WCM_VERSION \
         "$TLR"/experiments/094-tl-v2-records/out_v2_noR "$TW"/harder_nulls "$TW"/dan159 "$TW"/lists "$TW"/chunks; do
  [ -e "$p" ] && paths+=("$p")
done
[ "${SAVE_MAIN:-0}" = 1 ] && [ -e "$V2/main" ] && paths+=("$V2/main")
say "packing ${#paths[@]} paths (du: $(du -shc "${paths[@]}" 2> /dev/null | tail -1 | cut -f1)) -> $CKPT/$NAME.tgz"
mkdir -p "$CKPT" "$RESULTS" || exit 1
# a file still being written may change while read: tar rc 1 is expected, > 1 is a failure
tar $Z -cf "$CKPT/$NAME.tgz" "${paths[@]}" 2> "$CKPT/$NAME.tar.log"; rc=$?
[ "$rc" -le 1 ] || { say "tar FAILED (rc $rc, $CKPT/$NAME.tar.log)"; exit 1; }
(cd "$CKPT" && md5sum "$NAME.tgz" > "$NAME.md5") || exit 1
say "packed: $(tar -tzf "$CKPT/$NAME.tgz" | wc -l) entries, $(du -h "$CKPT/$NAME.tgz" | cut -f1)"
cp --no-preserve=mode,ownership "$CKPT/$NAME.tgz" "$CKPT/$NAME.md5" "$RESULTS/" 2> /dev/null
if (cd "$RESULTS" && md5sum -c --quiet "$NAME.md5"); then
  say "VERIFIED on $RESULTS: $NAME.tgz -- safe to stop the pod"
else
  say "copy / md5 check on $RESULTS FAILED: do NOT stop the pod yet"; exit 1
fi
