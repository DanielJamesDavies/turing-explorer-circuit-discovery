#!/bin/bash
# Fresh-pod bootstrap for a RunPod GLOBAL volume (2026-09-26). The global volume does not allow chmod, so git cannot
# live on it; it only holds the data:
#   /workspace/models          TuringLLM checkpoint + 36 SAEs      (~14 GB)
#   /workspace/data            interpretability corpus shards      (~12.6 GB)
#   /workspace/runs_store.tgz  discovery artifacts, packed from outputs/*.pt  (~2.5 GB)
# The code is cloned from GitHub onto the pod's own disk (/root/turing), branch multi-device.
#
#   curl -fsSL https://raw.githubusercontent.com/DanielJamesDavies/turing-explorer-circuit-discovery/multi-device/scripts/pod_bootstrap_global.sh | bash
#   (or: git clone ... && bash /root/turing/scripts/pod_bootstrap_global.sh)
#
# Idempotent: safe to re-run; each step skips what already exists.
set -e
REPO=https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git
BRANCH=${BRANCH:-multi-device}

echo "== 1/6 code: clone $BRANCH onto the pod disk"
if [ ! -d /root/turing/.git ]; then
  git clone --branch "$BRANCH" "$REPO" /root/turing
fi
cd /root/turing
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
git log --oneline -1

echo "== 2/6 production config"
cp config-h100-triamp.yaml config.yaml

echo "== 3/6 models + data from the global volume (skipped if already copied)"
for d in models data; do
  [ -d "/root/turing/$d" ] || cp -r --no-preserve=mode,ownership "/workspace/$d" /root/turing/
done
du -sh /root/turing/models /root/turing/data

echo "== 4/6 discovery artifacts"
if [ ! -f /root/turing/outputs/candidates.pt ]; then
  mkdir -p /root/turing/outputs
  tar xzf /workspace/runs_store.tgz -C /root/turing/outputs
fi
ls /root/turing/outputs/*.pt | wc -l

echo "== 5/6 venv over system torch + native extensions"
if [ ! -d /root/turing/.venv ]; then
  python3 -m venv --system-site-packages /root/turing/.venv
  /root/turing/.venv/bin/pip install -q pydantic==2.12.5 pydantic_core==2.41.5 pyyaml safetensors numpy==2.5.1 \
    matplotlib pandas pyarrow tqdm rich openpyxl psutil
fi
(cd /root/turing/src/native && /root/turing/.venv/bin/python setup.py build_ext --inplace > /dev/null 2>&1) \
  || echo "  (native build skipped/failed; PyTorch fallbacks work)"

echo "== 6/6 shard index cache (serial pre-build) + sanity"
cd /root/turing && PYTHONPATH=src ./.venv/bin/python -c "
import torch
from data.loader import DataLoader
DataLoader(device=torch.device('cpu'), pin_memory=False)
print('  shard indices ready')
from config import config
c = config.discovery
print('READY | methods', c.methods, '| floor', c.learned_mask.mask_floor_source,
      '| free_amplitude', c.learned_mask.free_amplitude, '| GPUs', torch.cuda.device_count())"

echo ""
echo "Bootstrap complete. Next (see experiments/062-h100-protocol-v1/README.md):"
echo "  smoke:  cd /root/turing && OUT=experiments/062-h100-protocol-v1/out_smoke TARGETS=experiments/062-h100-protocol-v1/targets_smoke.txt MODE=main PYTHONPATH=src ./.venv/bin/python -X utf8 experiments/062-h100-protocol-v1/driver.py"
echo "  launch: cd /root/turing && mkdir -p experiments/062-h100-protocol-v1/logs && MODE=both K=\$(nvidia-smi -L | wc -l) P=2 nohup bash experiments/062-h100-protocol-v1/launch.sh > experiments/062-h100-protocol-v1/logs/launch.log 2>&1 &"
