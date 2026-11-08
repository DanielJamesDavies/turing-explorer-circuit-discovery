#!/usr/bin/env bash
# Unpack every pod result archive (not the *-stores data-pass tarballs) into out/archives/<archive name>/, skipping
# archives already unpacked (a folder of that name exists). Checks each archive's md5 file when present.
#   bash experiments/109-pod-run-analysis/unpack.sh [RESULTS_DIR]
# Default RESULTS_DIR: ../Runs/pod-results-2026-11-07 next to the repo. Git Bash on Windows: pass /x/... paths
# (a "C:" path makes tar treat it as a remote host).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
RES=${1:-"$HERE/../../../Runs/pod-results-2026-11-07"}
DEST="$HERE/out/archives"
mkdir -p "$DEST"
for f in "$RES"/pub-*.tgz; do
  [ -e "$f" ] || continue
  name=$(basename "$f" .tgz)
  case "$name" in *-stores) echo "skip (data-pass store): $name"; continue ;; esac
  if [ -d "$DEST/$name" ]; then echo "already unpacked: $name"; continue; fi
  if [ -f "$RES/$name.md5" ]; then
    want=$(cut -d' ' -f1 "$RES/$name.md5"); have=$(md5sum "$f" | cut -d' ' -f1)
    if [ "$want" != "$have" ]; then echo "MD5 MISMATCH: $name ($have != $want); not unpacked"; continue; fi
  fi
  mkdir -p "$DEST/$name.partial"
  if tar -xzf "$f" -C "$DEST/$name.partial"; then
    mv "$DEST/$name.partial" "$DEST/$name"; echo "unpacked: $name"
  else
    echo "FAILED: $name (left in $name.partial)"
  fi
done
