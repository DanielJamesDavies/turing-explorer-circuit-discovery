#!/bin/bash
# Exp 128: build ~/venvs/ge107gpu OFFLINE (no pip, no network): a copy of ~/venvs/ge107 (Ge's env, torch 2.3.1+cpu,
# Python 3.10) whose torch is replaced by ~/venvs/tc107gpu's torch 2.8.0+cu128 (same Python 3.10.19; that copy was
# downloaded on 2026-11-16 for the Dunefsky GPU timing test, arena A README "Dunefsky GPU timing test"). The pod's
# ge107 stays torch 2.3.1+cu121 (H100 = sm_90); this env exists only to time Ge on the local sm_120 card.
#
#   bash experiments/128-pod-preflight/build_ge107gpu.sh          # refuses when the target exists
#
# What is copied: ge107 minus site-packages/{torch, torch-*.dist-info, torchgen}; then from tc107gpu:
# torch, torchgen, triton, nvidia (the cu12 runtime libs) and their dist-info dirs. Nothing else changes (Ge's
# pins: transformers 4.43.1, the vendored TransformerLens fork via PYTHONPATH, numpy 1.26.4, ...). The copy's
# bin/activate and bin/ge_env.sh still name ge107: do not source them; run_night.sh sets the env itself and calls
# ge107gpu/bin/python directly. Use with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (torch >= 2.6 defaults weights_only).
# Exit: 0 built (or already there), 3 a source env is missing (then a download would be needed: see README).
set -uo pipefail
SRC=${GE_SRC_ENV:-$HOME/venvs/ge107}
TOR=${TORCH_SRC_ENV:-$HOME/venvs/tc107gpu}
DST=${GE_GPU_ENV:-$HOME/venvs/ge107gpu}
SP=lib/python3.10/site-packages
say() { echo "[build_ge107gpu $(date +%H:%M:%S)] $*"; }

if [ -x "$DST/bin/python" ] && [ -f "$DST/BUILT_BY_128" ]; then say "$DST exists ($(cat "$DST/BUILT_BY_128"))"; exit 0; fi
[ -e "$DST" ] && { say "$DST exists but was not built by this script: not touched"; exit 1; }
for f in "$SRC/bin/python3.10" "$TOR/bin/python3.10" "$TOR/$SP/torch/__init__.py" "$TOR/$SP/torch-2.8.0+cu128.dist-info"; do
  [ -e "$f" ] || { say "missing $f: cannot build offline"; exit 3; }
done
vs=$("$SRC/bin/python" -c 'import sys; print(sys.version.split()[0])'); vt=$("$TOR/bin/python" -c 'import sys; print(sys.version.split()[0])')
[ "$vs" = "$vt" ] || { say "Python differs: ge107 $vs, tc107gpu $vt: the torch build cannot be reused"; exit 3; }

P=$DST.partial
[ -e "$P" ] && { say "$P exists (an interrupted build?): move it away first"; exit 1; }
say "copy $SRC -> $P (without its torch)"
rsync -a --exclude "/$SP/torch/" --exclude "/$SP/torch-*.dist-info/" --exclude "/$SP/torchgen/" "$SRC/" "$P/" || exit 1
say "add torch 2.8.0+cu128 (+ triton, nvidia libs) from $TOR"
items=(torch torchgen triton nvidia torch-2.8.0+cu128.dist-info)
for d in "$TOR/$SP"/triton-*.dist-info "$TOR/$SP"/nvidia_*.dist-info; do items+=("$(basename "$d")"); done
for it in "${items[@]}"; do
  [ -e "$TOR/$SP/$it" ] || { say "missing $TOR/$SP/$it"; exit 1; }
  rsync -a "$TOR/$SP/$it" "$P/$SP/" || exit 1
done
echo "built $(date '+%F %T') from $SRC + torch of $TOR (exp 128 build_ge107gpu.sh)" > "$P/BUILT_BY_128"
mv "$P" "$DST" || exit 1
CUDA_VISIBLE_DEVICES= "$DST/bin/python" -I -c 'import torch, transformers, numpy; print("ge107gpu: torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__, "| numpy", numpy.__version__)' \
  || { say "import check FAILED"; exit 1; }
say "done: $DST ($(du -sh "$DST" | cut -f1))"
