#!/bin/bash
# Fresh-pod bootstrap for the tri-amp production run (RunPod H100,
# template runpod/pytorch 2.8.0-cu128, network volume mounted at
# /workspace holding the May repo + models + data, and
# /workspace/runs_store.tgz with the 20260531 discovery artifacts).
#
#   bash /workspace/turing/scripts/pod_bootstrap.sh
#
# Idempotent: safe to re-run; each step skips what already exists.
set -e

echo "== 1/7 refresh volume repo refs (fetch only, checkout untouched)"
cd /workspace/turing && git fetch origin

echo "== 2/7 clone to local disk at latest multi-device"
if [ ! -d /root/turing ]; then
  git clone /workspace/turing /root/turing
fi
cd /root/turing
git remote get-url github > /dev/null 2>&1 || git remote add github \
  https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git
git fetch github
git reset --hard github/multi-device
git log --oneline -1

echo "== 3/7 production config (survives every later git reset via cp)"
cp config-h100-triamp.yaml config.yaml

echo "== 4/7 models + data from volume (skipped if already copied)"
[ -d /root/turing/models ] || cp -r /workspace/turing/models /root/turing/
[ -d /root/turing/data ] || cp -r /workspace/turing/data /root/turing/

echo "== 5/7 discovery artifacts from volume tarball"
if [ ! -f /root/turing/outputs/candidates.pt ]; then
  mkdir -p /root/turing/outputs
  tar xzf /workspace/runs_store.tgz -C /root/turing/outputs
fi
ls /root/turing/outputs/*.pt | wc -l

echo "== 6/7 venv over system torch + native extensions"
if [ ! -d /root/turing/.venv ]; then
  python3 -m venv --system-site-packages /root/turing/.venv
  /root/turing/.venv/bin/pip install -q pydantic==2.12.5 \
    pydantic_core==2.41.5 pyyaml safetensors numpy==2.5.1 matplotlib \
    pandas pyarrow tqdm rich openpyxl psutil
fi
cd /root/turing/src/native && /root/turing/.venv/bin/python setup.py \
  build_ext --inplace > /dev/null 2>&1 || echo "  (native build skipped/failed — PyTorch fallbacks work)"

echo "== 7/7 sanity: import chain + config fields"
cd /root/turing && PYTHONPATH=src ./.venv/bin/python -c "
from config import config
from circuit.discovery_window import DiscoveryWindow, _parse_seed_shard
c = config.discovery
assert c.methods == ['ablation_gradient'], c.methods
assert c.learned_mask.free_amplitude, 'free_amplitude off!'
assert c.learned_mask.mask_floor_source == 'triple'
print('READY | methods', c.methods, '| floor', c.learned_mask.mask_floor_source,
      '| free_amplitude', c.learned_mask.free_amplitude,
      '| seed_shard default', c.seed_shard)"

echo ""
echo "Bootstrap complete. Launch shape (per GPU g, shards i of k):"
echo "  cd /root/turing && CUDA_VISIBLE_DEVICES=g SEED_SHARD=i/k \\"
echo "    PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
echo "    nohup ./.venv/bin/python src/discover_circuits.py > shard_i.log 2>&1 &"
echo "Resume is automatic per shard; rows in outputs/circuits/."
echo "Copy results to the volume before stopping the pod:"
echo "  cp outputs/circuits/discovered_circuits*.pt outputs/circuits/task_metrics*.jsonl /workspace/results/"
