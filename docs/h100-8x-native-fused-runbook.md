# 8x H100 Native Fused Pass-1 Runbook

This runbook describes how to set up and run distributed pass 1 on an 8xH100 pod using the native fused SAE top-k backend.

The native backend is the preferred H100 pass-1 path because profiling showed it materially improves the real workload:

```text
Standard PyTorch pass1 profile:
  ~9-12 s / batch
  peak CUDA allocated: ~55.32 GiB

Native fused exact pass1 profile:
  ~4.6-5.4 s / batch
  peak CUDA allocated: ~24.97-40.70 GiB depending on streaming/deferred mode
```

For the current H100 setup, use:

```yaml
data:
  batch_size: 4096

hardware:
  memory: "fast"
  parallel_kinds: false
  keep_model_loaded_for_neg_ctx: true

sae:
  encode_backend: "fused_exact_topk"
  topk_backend: "pytorch"
  fused_exact_topk_use_native: true

first_pass:
  sae_encode_mode: "streaming"
```

The native fused path uses the validated production feature block width of `10240` internally. Do not add this as a config setting; `TURINGLLM_FUSED_EXACT_TOPK_BLOCK_N` exists only as a developer override for future experiments.

## Prerequisites

Use a Linux CUDA pod with 8 visible H100 GPUs and enough disk for outputs.

Expected directories:

```text
/workspace/turing
/workspace/turing/data
/workspace/turing/models
/outputs
```

Before starting, confirm the repo, data, and model files are present:

```bash
cd /workspace/turing
ls
ls data | head
ls models/TuringLLM
nvidia-smi
```

## Pull Code And Build Native Extensions

If `config.yaml` has local pod edits, back it up before pulling:

```bash
cd /workspace/turing
cp config.yaml /tmp/config.before-pull.yaml
git checkout -- config.yaml
git pull
```

Build the native extensions on the pod:

```bash
cd /workspace/turing/src/native
python setup.py build_ext --inplace
```

Verify the native fused exact extension loads:

```bash
cd /workspace/turing
PYTHONPATH=src:src python - <<'PY'
from sae.fused_exact_topk import native_is_available
print("native available:", native_is_available())
PY
```

Expected:

```text
native available: True
```

If this is `False`, rebuild from `src/native` and inspect the import/build error.

## Prepare The 8xH100 Config

Start from the 8x distributed config:

```bash
cd /workspace/turing
cp config_examples/h100-8x-distributed-simple-exact.yaml config.yaml
```

Edit `config.yaml` so the important fields are:

```yaml
data:
  dataset_path: "data"
  n_shards: 3030
  batch_size: 4096

hardware:
  multi_gpu: false
  memory: "fast"
  compile: true
  parallel_kinds: false
  ann_device: "auto"
  keep_model_loaded_for_neg_ctx: true

sae:
  encode_backend: "fused_exact_topk"
  topk_backend: "pytorch"
  fused_exact_topk_use_native: true

first_pass:
  sae_encode_mode: "streaming"

distributed:
  mode: "distributed_simple_exact"
  output_base: "outputs"
  worker_count: 8
  devices: [0, 1, 2, 3, 4, 5, 6, 7]
  launch_strategy: "manual_commands"
  resume_policy: "fresh"
  cleanup_policy: "keep_all"
```

Sanity check config and runtime backend:

```bash
cd /workspace/turing
unset TURINGLLM_TOPK_IMPL
export TURINGLLM_SAE_ENCODE_IMPL=fused_exact_topk
export TURINGLLM_FUSED_EXACT_TOPK_USE_NATIVE=1

PYTHONPATH=src:src python - <<'PY'
from config import config
from sae.topk_sae import get_encode_backend
from sae.fused_exact_topk import native_is_available

print("n_shards:", config.data.n_shards)
print("batch_size:", config.data.batch_size)
print("memory:", config.hardware.memory)
print("parallel_kinds:", config.hardware.parallel_kinds)
print("first_pass:", config.first_pass.sae_encode_mode)
print("config encode:", config.sae.encode_backend)
print("runtime encode:", get_encode_backend())
print("native:", config.sae.fused_exact_topk_use_native)
print("native available:", native_is_available())
print("workers:", config.distributed.worker_count)
print("devices:", config.distributed.devices)
PY
```

Expected key values:

```text
n_shards: 3030
batch_size: 4096
parallel_kinds: False
first_pass: streaming
runtime encode: fused_exact_topk
native: True
native available: True
workers: 8
devices: [0, 1, 2, 3, 4, 5, 6, 7]
```

## Optional 1x Smoke Before 8x

Before using all GPUs, run a short 1x profile to confirm the native path is active:

```bash
cd /workspace/turing
python -m pipeline.distributed.controller \
  --config config.yaml \
  --output-base /outputs \
  --mode distributed_simple_exact \
  --worker-count 1 \
  --devices 0 \
  --dry-run | tee /tmp/pass1-native-1x-profile-dry-run.txt

export RUN_ROOT=$(awk '/^output_root:/ {print $2}' /tmp/pass1-native-1x-profile-dry-run.txt)
export MANIFEST="$RUN_ROOT/distributed/manifest.json"

PYTHONPATH=src:src python -m debug.profile_distributed_pass1 \
  --manifest "$MANIFEST" \
  --worker-id 0 \
  --warmup-batches 1 \
  --profile-batches 3
```

The profiler should show `local_topk_kernel` and `merge_topk_kernel`. If it shows `aten::topk` dominating, the native fused backend is not active.

Expected streaming native profile on 1xH100 with `batch_size=4096`:

```text
profiled_pass1_batch: ~4.6-4.8 s / batch
peak CUDA allocated: ~25 GiB
```

## Create The 8x Dry-Run Manifest

Run the 8x dry-run:

```bash
cd /workspace/turing
python -m pipeline.distributed.controller \
  --config config.yaml \
  --output-base /outputs \
  --mode distributed_simple_exact \
  --worker-count 8 \
  --devices 0,1,2,3,4,5,6,7 \
  --dry-run | tee /tmp/h100-8x-native-pass1-dry-run.txt

export RUN_ROOT=$(awk '/^output_root:/ {print $2}' /tmp/h100-8x-native-pass1-dry-run.txt)
export MANIFEST="$RUN_ROOT/distributed/manifest.json"
echo "$RUN_ROOT"
echo "$MANIFEST"
```

Review the dry-run output:

```text
worker_000 ... shards=[...]
worker_001 ... shards=[...]
...
worker_007 ... shards=[...]
```

Each worker should map to one physical GPU and receive a roughly even shard allocation.

## Launch 8 Pass-1 Workers

Launch one worker per GPU:

```bash
cd /workspace/turing

for W in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$W PYTHONPATH=src:src python -m pipeline.distributed.worker \
    --manifest "$MANIFEST" \
    --phase pass1 \
    --worker-id "$W" \
    > "$RUN_ROOT/distributed/pass1_worker_$(printf '%03d' "$W")_native_fused.log" 2>&1 &
done

wait
```

Monitor GPU utilization in another terminal:

```bash
watch -n 1 nvidia-smi
```

Monitor worker logs:

```bash
cd "$RUN_ROOT/distributed"
ls -lh pass1_worker_*_native_fused.log
```

For a quick view of progress:

```bash
for f in "$RUN_ROOT"/distributed/pass1_worker_*_native_fused.log; do
  echo "===== $f ====="
  tail -n 20 "$f"
done
```

## Expected Runtime

From 1xH100 streaming native pass1 profiling:

```text
~4.6-4.8 s / batch
batch_size = 4096
3030 shards ~= 6060 batches total
```

Estimated runtime:

```text
1xH100: ~7.7-8.1 hours
8xH100 ideal: ~1 hour
realistic: ~75-120 minutes
```

Add extra time for cold start, artifact saves, filesystem variation, and load imbalance.

## Expected Memory

The native fused streaming path was measured around:

```text
~25 GiB peak CUDA allocation on 1xH100 profile
```

This is much lower than the standard PyTorch path, which was measured around:

```text
~55 GiB peak CUDA allocation
```

If memory is unexpectedly high, confirm:

```text
first_pass.sae_encode_mode = streaming
hardware.parallel_kinds = false
sae.encode_backend = fused_exact_topk
sae.fused_exact_topk_use_native = true
```

## Troubleshooting

### Native backend silently not active

Check runtime backend:

```bash
PYTHONPATH=src:src python - <<'PY'
from sae.topk_sae import get_encode_backend
from sae.fused_exact_topk import native_is_available
print("runtime encode:", get_encode_backend())
print("native available:", native_is_available())
PY
```

Expected:

```text
runtime encode: fused_exact_topk
native available: True
```

If `runtime encode` is `standard`, check env vars and config.

### Native extension missing

Rebuild:

```bash
cd /workspace/turing/src/native
python setup.py build_ext --inplace
```

### Benchmark accidentally uses wrong block size

`debug.benchmark_sae_encode` accepts `--block-n`. The production native path defaults internally to `10240`, but the benchmark can override it. For microbenchmarks, use:

```bash
cd /workspace/turing/src
python -m debug.benchmark_sae_encode \
  --batch-size 2048 \
  --seq-len 64 \
  --block-n 10240 \
  --warmup 3 \
  --iters 10 \
  --use-native-fused-exact \
  --allow-slow-native
```

Expected native fused microbenchmark result on H100:

```text
~54 ms at B=2048, T=64, block_n=10240
```

### Shard table mismatch

Run controller and profiler/worker from repo root (`/workspace/turing`), not from `src`, unless `PYTHONPATH=src:src` is set. Relative data paths are resolved from the current working directory in some validation paths.

### `config.yaml` blocks `git pull`

The pod often edits `config.yaml` for a run. Back it up and reset before pulling:

```bash
cd /workspace/turing
cp config.yaml /tmp/config.before-pull.yaml
git checkout -- config.yaml
git pull
```

Then copy the desired example config back into place.

## After Pass 1

After all pass1 workers finish, the distributed partials should be under:

```text
$RUN_ROOT/distributed/
```

Before continuing to merge/reduce:

```bash
ls -lh "$RUN_ROOT/distributed"
for f in "$RUN_ROOT"/distributed/pass1_worker_*_native_fused.log; do
  echo "===== $f ====="
  tail -n 30 "$f"
done
```

Confirm no worker failed before starting pass1 merge or pass2.
