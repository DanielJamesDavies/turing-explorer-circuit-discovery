#!/bin/bash
# Fresh-pod bootstrap for a RunPod GLOBAL volume (2026-09-26). The global volume does not allow chmod, so git cannot
# live on it; it only holds the data:
#   /workspace/models          TuringLLM checkpoint + 36 SAEs      (~14 GB)
#   /workspace/data            interpretability corpus shards      (~12.6 GB)
#   /workspace/runs_store.tgz  discovery artifacts, packed from outputs/*.pt  (~2.5 GB)
# The code is cloned from GitHub onto the pod's own disk (ROOT, default /root/turing), branch multi-device.
#
#   curl -fsSL https://raw.githubusercontent.com/DanielJamesDavies/turing-explorer-circuit-discovery/multi-device/scripts/pod_bootstrap_global.sh | bash
#   (or: git clone ... && bash /root/turing/scripts/pod_bootstrap_global.sh)
#
# TuringLLM track of the 082 public-model session (2026-10-25): a SECOND checkout beside the public tree, so the
# public tree (/root/turing, its config.yaml and venv) is never reset or overwritten:
#   ROOT=/root/turing-tl TL_TRACK=1 bash /root/turing/scripts/pod_bootstrap_global.sh
# With TL_TRACK=1 it also installs what the TuringLLM track (082 chain.sh T0-T9; protocol v2 2026-10-25: the 8k refit
# reads the stores, data/ and the corpus index; T9 also the as-run records and the 15k main circuits) reads:
#   7  the 15k run's results  /workspace/results/062-full-20260927-1924.tgz -> 062/out_full/{out/main,out/sweep,...}
#   8  the local-only inputs  TL_EXTRAS (default /root/tl-extras.tgz, sent from Daniel's PC with runpodctl send):
#      062/out_full/out/ctx (the 15,046 as-run records, 2.4 GB), 062/out_random{,_scale} (65 stored DAN-17 circuits
#      for the 084 panel), 062/out_{bands,depth,lowlam} (083's weak-price WCM arms), 062/out_alpha1/alpha1.jsonl;
#      md5-checked against TL_EXTRAS_MD5 (the .md5 file next to it) before unpacking
#   9  085's corpus index (corpus_hash.npy, corpus_mult.npy; corpus_src.npy is no longer read): recomputed from data/
#      by 085/corpus_index.py (CPU, ~90 s locally), then corpus_hash.npy md5-checked against the local file
#
# Idempotent: safe to re-run; each step skips what already exists. The clone is reset to origin/BRANCH only in ROOT.
set -e
REPO=https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git
BRANCH=${BRANCH:-multi-device}
ROOT=${ROOT:-/root/turing}
TL_TRACK=${TL_TRACK:-0}
FULL_TGZ=${FULL_TGZ:-/workspace/results/062-full-20260927-1924.tgz}
TL_EXTRAS=${TL_EXTRAS:-/root/tl-extras.tgz}
TL_EXTRAS_MD5=${TL_EXTRAS_MD5:-${TL_EXTRAS%.tgz}.md5}
CORPUS_HASH_MD5=${CORPUS_HASH_MD5:-00d5a8e4a0ced9fc78abd942b282516a}   # experiments/085-heldout-audit/out, 2026-10-23
NSTEP=6; [ "$TL_TRACK" = 1 ] && NSTEP=9
E62=experiments/062-h100-protocol-v1

echo "== 1/$NSTEP code: clone $BRANCH onto the pod disk ($ROOT)"
if [ ! -d "$ROOT/.git" ]; then
  git clone --branch "$BRANCH" "$REPO" "$ROOT"
fi
cd "$ROOT"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
git --no-pager log --oneline -1

echo "== 2/$NSTEP production config"
cp config-h100-triamp.yaml config.yaml

echo "== 3/$NSTEP models + data from the global volume (skipped if already copied)"
for d in models data; do
  [ -d "$ROOT/$d" ] || cp -r --no-preserve=mode,ownership "/workspace/$d" "$ROOT/"
done
du -sh "$ROOT/models" "$ROOT/data"

echo "== 4/$NSTEP discovery artifacts"
if [ ! -f "$ROOT/outputs/candidates.pt" ]; then
  mkdir -p "$ROOT/outputs"
  tar xzf /workspace/runs_store.tgz -C "$ROOT/outputs"
fi
ls "$ROOT"/outputs/*.pt | wc -l

echo "== 5/$NSTEP venv over system torch + native extensions"
if [ ! -d "$ROOT/.venv" ]; then
  python3 -m venv --system-site-packages "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q pydantic==2.12.5 pydantic_core==2.41.5 pyyaml safetensors numpy==2.5.1 \
    matplotlib pandas pyarrow tqdm rich openpyxl psutil
fi
(cd "$ROOT/src/native" && "$ROOT/.venv/bin/python" setup.py build_ext --inplace > /dev/null 2>&1) \
  || echo "  (native build skipped/failed; PyTorch fallbacks work)"

echo "== 6/$NSTEP shard index cache (serial pre-build) + sanity"
cd "$ROOT" && PYTHONPATH=src ./.venv/bin/python -c "
import torch
from data.loader import DataLoader
DataLoader(device=torch.device('cpu'), pin_memory=False)
print('  shard indices ready')
from config import config
c = config.discovery
print('READY | methods', c.methods, '| floor', c.learned_mask.mask_floor_source,
      '| free_amplitude', c.learned_mask.free_amplitude, '| GPUs', torch.cuda.device_count())"

if [ "$TL_TRACK" = 1 ]; then
  echo "== 7/9 the 15k run's results ($FULL_TGZ -> $E62/out_full)"
  if [ ! -d "$ROOT/$E62/out_full/out/main/circuits" ]; then
    mkdir -p "$ROOT/$E62/out_full"
    cp --no-preserve=mode,ownership "$FULL_TGZ" /root/062-full.tgz           # off geesefs before reading it
    tar xzf /root/062-full.tgz -C "$ROOT/$E62/out_full" && rm -f /root/062-full.tgz
  fi
  echo "  main circuits: $(ls "$ROOT/$E62/out_full/out/main/circuits" | wc -l)," \
    "sweep arms: $(ls "$ROOT/$E62/out_full/out/sweep/circuits" | wc -l)"

  echo "== 8/9 local-only inputs ($TL_EXTRAS)"
  if [ ! -d "$ROOT/$E62/out_full/out/ctx" ] || [ ! -d "$ROOT/$E62/out_random/circuits" ]; then
    [ -f "$TL_EXTRAS" ] || { echo "  MISSING $TL_EXTRAS: send it from the local PC first (runbook)"; exit 1; }
    if [ -f "$TL_EXTRAS_MD5" ]; then
      (cd "$(dirname "$TL_EXTRAS")" && md5sum -c "$(basename "$TL_EXTRAS_MD5")") || { echo "  md5 mismatch"; exit 1; }
    else
      echo "  (no $TL_EXTRAS_MD5: unpacking without an md5 check)"
    fi
    tar xzf "$TL_EXTRAS" -C "$ROOT"
  fi
  echo "  as-run records: $(ls "$ROOT/$E62/out_full/out/ctx" | wc -l) (expect 15046);" \
    "DAN-17 circuits: $(ls "$ROOT/$E62/out_random/circuits" "$ROOT/$E62/out_random_scale/circuits" 2> /dev/null | grep -c '\.pt$')"

  echo "== 9/9 085 corpus index (recomputed from data/)"
  if [ ! -f "$ROOT/experiments/085-heldout-audit/out/corpus_hash.npy" ]; then
    (cd "$ROOT" && PYTHONPATH=src ./.venv/bin/python -X utf8 experiments/085-heldout-audit/corpus_index.py)
  fi
  got=$(md5sum "$ROOT/experiments/085-heldout-audit/out/corpus_hash.npy" | cut -c1-32)
  if [ "$got" = "$CORPUS_HASH_MD5" ]; then echo "  corpus_hash.npy md5 $got == local"
  else echo "  WARNING: corpus_hash.npy md5 $got != local $CORPUS_HASH_MD5 (send the local file instead)"; exit 1; fi
fi

echo ""
echo "Bootstrap complete ($ROOT). Next (see experiments/062-h100-protocol-v1/README.md):"
if [ "$TL_TRACK" = 1 ]; then
  echo "  TuringLLM track: run the 082 chain from the public tree with TL_TRACK=1 TL_ROOT=$ROOT TL_PY=$ROOT/.venv/bin/python"
else
  echo "  smoke:  cd $ROOT && OUT=$E62/out_smoke TARGETS=$E62/targets_smoke.txt MODE=main PYTHONPATH=src ./.venv/bin/python -X utf8 $E62/driver.py"
  echo "  launch: cd $ROOT && mkdir -p $E62/logs && MODE=both K=\$(nvidia-smi -L | wc -l) P=2 nohup bash $E62/launch.sh > $E62/logs/launch.log 2>&1 &"
fi
