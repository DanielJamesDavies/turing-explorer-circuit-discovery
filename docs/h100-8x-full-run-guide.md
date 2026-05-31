# 8xH100 Full Distributed Run Guide

This is the command sequence I use for the full 8xH100 run. I either run it as
one supervised shell command or stage it deliberately so I can sanity-check each
part before starting the next one.

## 0. Assumptions

I need:

```text
8x H100 visible to nvidia-smi
/workspace/turing checked out
/workspace/turing/data present
/workspace/turing/models present
300-500 GiB fast output disk available
```

For the full run I should not use a small container disk or small network volume
for outputs. On RunPod, `/workspace` may be the network volume and can be much
smaller than the local pod disk. The expected large artifacts are:

```text
seq_latent_index: ~80 GiB
pass2 candidate dumps: ~200 GiB
top_coactivation.pt: ~1.5 GiB
```

Set the output base to fast local disk if available:

```bash
export OUTPUT_BASE=/root/outputs
mkdir -p "$OUTPUT_BASE"
```

If the pod only has a small local disk, use a large mounted volume instead. Do
not use a small `/workspace` network volume for outputs.

```bash
export OUTPUT_BASE=/workspace/outputs
mkdir -p "$OUTPUT_BASE"
```

## 1. Activate Environment And Pull

```bash
source /root/venvs/turing/bin/activate
cd /workspace/turing

cp config.yaml /tmp/config.before-h100-8x-full.yaml 2>/dev/null || true
git checkout -- config.yaml
git pull
```

## 2. Build Native Extensions

```bash
cd /workspace/turing/src/native
python setup.py build_ext --inplace
cd /workspace/turing
```

```bash
PYTHONPATH=src:src python - <<'PY'
from sae.fused_exact_topk import native_is_available
print("native available:", native_is_available())
PY
```

Expected:

```text
native available: True
```

## 3. Prepare Config

```bash
cp config_examples/h100-8x-distributed-simple-exact.yaml config.yaml
```

This example is the canonical full-run config. It pins the active discovery,
latent-store, persistence, and analysis settings, so I only edit it when I am
intentionally changing the experiment.

For the next counterfactual-gradient production run, use `32x32` contrast
settings. A completed 8xH100 run used `16x16` successfully, then pilots on 128
seeds showed `32x32` had similar speed, safe H100 memory use, and slightly
better acceptance. `64x32` was viable but slower without improving acceptance on
the same pilot sample.

```yaml
discovery:
  probe_batch_size: 4
  counterfactual_gradient:
    max_neg_sequences: 32
    neg_batch_size: 32
```

Check the important fields:

```bash
PYTHONPATH=src:src python - <<'PY'
from config import config
from sae.fused_exact_topk import native_is_available

tc = config.latents.top_coactivation
print("native:", native_is_available())
print("n_shards:", config.data.n_shards)
print("batch_size:", config.data.batch_size)
print("workers:", config.distributed.worker_count, config.distributed.devices)
print("neg_ctx:", config.latents.neg_ctx.backend, config.latents.neg_ctx.devices)
print("top_coact:", tc.mode, tc.n_latents_per_latent, tc.n_candidates_per_component, tc.candidate_oversample_factor)
print("candidate_width_M:", min(tc.n_latents_per_latent * tc.candidate_oversample_factor, 36 * tc.n_candidates_per_component))
print("discovery:", config.discovery.n_seeds, config.discovery.methods)
print("seed_criteria:", config.discovery.seed_criteria)
cfg = config.discovery.counterfactual_gradient
print("counterfactual_gradient:", cfg.neg_mode, cfg.top_k_scope, cfg.max_neg_sequences, cfg.neg_batch_size)
PY
```

Expected:

```text
native: True
n_shards: 3030
batch_size: 4096
workers: 8 [0, 1, 2, 3, 4, 5, 6, 7]
neg_ctx: multi_gpu_exact [0, 1, 2, 3, 4, 5, 6, 7]
top_coact: pmi 128 128 8
candidate_width_M: 1024
discovery: 16384 ['counterfactual_gradient']
seed_criteria: ['stratified_random']
counterfactual_gradient: random layer_kind 32 32
```

## 4. Create Manifest

```bash
PYTHONPATH=src:src python -m pipeline.distributed.controller \
  --config config.yaml \
  --output-base "$OUTPUT_BASE" \
  --mode distributed_simple_exact \
  --worker-count 8 \
  --devices 0,1,2,3,4,5,6,7 \
  --dry-run | tee /tmp/h100-8x-full-dry-run.txt

export RUN_ROOT=$(awk '/^output_root:/ {print $2}' /tmp/h100-8x-full-dry-run.txt)
export MANIFEST="$RUN_ROOT/distributed/manifest.json"

echo "$RUN_ROOT"
echo "$MANIFEST"
test -f "$MANIFEST" && echo "manifest ok"
```

Check the dry-run has:

```text
worker_count: 8
preflight_shards: 3030
worker_000 ... worker_007
```

## 5. Full Pipeline In One Command

This runs the same staged pipeline below, but through one supervised Python
wrapper. I use this only after the config sanity check and manifest dry-run look
correct.

```bash
PYTHONPATH=src:src python scripts/run_distributed_full_pipeline.py --manifest "$MANIFEST"
```

The wrapper caps worker CPU thread pools to `4` by default:

```text
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
```

Override this only for benchmarking:

```bash
PYTHONPATH=src:src python scripts/run_distributed_full_pipeline.py \
  --manifest "$MANIFEST" \
  --worker-threads 8
```

If I also want Pass 2 worker dumps removed automatically after reduce succeeds:

```bash
PYTHONPATH=src:src python scripts/run_distributed_full_pipeline.py \
  --manifest "$MANIFEST" \
  --cleanup-pass2-partials
```

## 6. Run The Pipeline Stage By Stage

Use this path when I want to benchmark, inspect, or recover each stage before
starting the next one.

For manual worker launches, I set the same CPU thread caps that the wrapper uses:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

### Optional Pass 1 Profile

I usually profile worker 0 before launching all 8 workers:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:src python -m debug.profile_distributed_pass1 \
  --manifest "$MANIFEST" \
  --worker-id 0 \
  --warmup-batches 1 \
  --profile-batches 2 \
  --output-dir "$RUN_ROOT/profile_trace"
```

Expected:

```text
profiled_pass1_batch: ~4.6s
local_topk_kernel present
merge_topk_kernel present
aten::topk small
Peak CUDA allocated: ~25 GiB
```

### Run Pass 1 Workers

```bash
for W in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$W PYTHONPATH=src:src python -m pipeline.distributed.worker \
    --manifest "$MANIFEST" \
    --phase pass1 \
    --worker-id "$W" \
    > "$RUN_ROOT/distributed/pass1_worker_$(printf '%03d' "$W")_native_fused.log" 2>&1 &
done

wait
```

Monitor:

```bash
watch -n 1 nvidia-smi
```

Inspect:

```bash
for f in "$RUN_ROOT"/distributed/pass1_worker_*_native_fused.log; do
  echo "===== $f ====="
  tail -n 30 "$f"
done
```

### Merge Pass 1

```bash
PYTHONPATH=src:src python -m pipeline.distributed.pass1_merge \
  --manifest "$MANIFEST" | tee "$RUN_ROOT/distributed/pass1_merge.log"
```

Check:

```bash
ls -lh "$RUN_ROOT"/*.pt
du -h --max-depth=1 "$RUN_ROOT/seq_latent_index" | sort -h | tail

python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
report = json.loads((root / "distributed/reports/pass1_sanity_report.json").read_text())
print("status:", report["status"])
print("seq_repr:", report.get("seq_repr_fill"))
PY
```

After Pass 1 merge succeeds, I can remove large Pass 1 worker partials:

```bash
for W in 0 1 2 3 4 5 6 7; do
  rm -rf "$RUN_ROOT/distributed/workers/worker_$(printf '%03d' "$W")/pass1"
done

df -h "$OUTPUT_BASE"
```

### Build Negative Context

```bash
rm -f "$RUN_ROOT/neg_ctx.pt" "$RUN_ROOT/neg_ctx_stats.json"
rm -rf "$RUN_ROOT/distributed/parts/neg_ctx"

PYTHONPATH=src:src python -m pipeline.negative_context \
  --output-root "$RUN_ROOT" \
  --manifest "$MANIFEST" \
  | tee "$RUN_ROOT/distributed/neg_ctx_multi_gpu.log"
```

Check:

```bash
ls -lh "$RUN_ROOT/neg_ctx.pt" "$RUN_ROOT/neg_ctx_stats.json"
```

Expected backend:

```text
multi_gpu_exact
```

### Run Pass 2 Workers

```bash
for W in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$W PYTHONPATH=src:src python -m pipeline.distributed.worker \
    --manifest "$MANIFEST" \
    --phase pass2 \
    --worker-id "$W" \
    > "$RUN_ROOT/distributed/pass2_worker_$(printf '%03d' "$W").log" 2>&1 &
done

wait
```

Inspect:

```bash
for f in "$RUN_ROOT"/distributed/pass2_worker_*.log; do
  echo "===== $f ====="
  tail -n 50 "$f"
done

ls -lh "$RUN_ROOT"/distributed/workers/worker_*/pass2/*
```

Expected candidate dump width:

```text
1024 candidates
```

### Reduce Pass 2

```bash
PYTHONPATH=src:src python -m pipeline.distributed.pass2_reduce \
  --output-root "$RUN_ROOT" \
  --top-ctx "$RUN_ROOT/top_ctx.pt" \
  --latent-stats "$RUN_ROOT/latent_stats.pt" \
  --candidate-dump "$RUN_ROOT/distributed/workers/worker_000/pass2/candidate_dump.partial.pt" \
  --candidate-dump "$RUN_ROOT/distributed/workers/worker_001/pass2/candidate_dump.partial.pt" \
  --candidate-dump "$RUN_ROOT/distributed/workers/worker_002/pass2/candidate_dump.partial.pt" \
  --candidate-dump "$RUN_ROOT/distributed/workers/worker_003/pass2/candidate_dump.partial.pt" \
  --candidate-dump "$RUN_ROOT/distributed/workers/worker_004/pass2/candidate_dump.partial.pt" \
  --candidate-dump "$RUN_ROOT/distributed/workers/worker_005/pass2/candidate_dump.partial.pt" \
  --candidate-dump "$RUN_ROOT/distributed/workers/worker_006/pass2/candidate_dump.partial.pt" \
  --candidate-dump "$RUN_ROOT/distributed/workers/worker_007/pass2/candidate_dump.partial.pt" \
  | tee "$RUN_ROOT/distributed/pass2_reduce.log"
```

Check:

```bash
ls -lh "$RUN_ROOT/top_coactivation.pt" "$RUN_ROOT/distributed/reports/pass2_reduce_report.json"

python - <<'PY'
import json, os
from pathlib import Path

report = json.loads((Path(os.environ["RUN_ROOT"]) / "distributed/reports/pass2_reduce_report.json").read_text())
print(json.dumps({
    "candidate_width": report.get("candidate_width"),
    "output_shape": report.get("output_shape"),
    "output_finite": report.get("output_finite"),
    "timing": report.get("timing"),
}, indent=2))
PY
```

After reduce succeeds, I can remove Pass 2 worker dumps if I need disk:

```bash
for W in 0 1 2 3 4 5 6 7; do
  rm -rf "$RUN_ROOT/distributed/workers/worker_$(printf '%03d' "$W")/pass2"
done
```

### Candidate Selection

```bash
TURING_TRUST_PASS2_REPLAY_ASSIGNMENTS=1 \
PYTHONUNBUFFERED=1 \
PYTHONPATH=src:src python -m pipeline.candidate_selection \
  --output-root "$RUN_ROOT" \
  --manifest "$MANIFEST" \
  | tee "$RUN_ROOT/distributed/candidate_selection.log"
```

Check assignments:

```bash
python - <<'PY'
import json, os
from pathlib import Path

m = json.loads(Path(os.environ["MANIFEST"]).read_text())
wa = m["work_assignments"]
print("scheduling_strategy:", wa.get("discovery_scheduling_strategy"))
print("candidate assignments:", sum(len(v) for v in wa["discovery_candidate_assignments"].values()))
for w in range(8):
    ids = wa["discovery_seed_ids"].get(str(w), [])
    print(f"worker {w} seeds:", len(ids), "first10:", ids[:10])
PY
```

Expected:

```text
scheduling_strategy: candidate_shuffled
worker seed counts: 2047-2048 each
first10 values: non-contiguous-looking candidate indices
```

### Run Discovery Workers

```bash
for W in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$W \
  TURING_TRUST_PASS2_REPLAY_ASSIGNMENTS=1 \
  PYTHONPATH=src:src python -m pipeline.distributed.worker \
    --manifest "$MANIFEST" \
    --phase discovery \
    --worker-id "$W" \
    > "$RUN_ROOT/distributed/discovery_worker_$(printf '%03d' "$W").log" 2>&1 &
done

wait
```

Inspect:

```bash
for f in "$RUN_ROOT"/distributed/discovery_worker_*.log; do
  echo "===== $f ====="
  tail -n 80 "$f"
done
```

Progress monitor:

```bash
watch -n 60 'date; echo; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits; echo; for f in "$RUN_ROOT"/distributed/discovery_worker_*.log; do echo "=== $(basename "$f") ==="; python - "$f" <<PY
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="replace").replace("\\r", "\\n")
lines = [l for l in text.splitlines() if "Discovering Circuits:" in l]
print(lines[-1] if lines else "no progress yet")
PY
done'
```

On the completed shuffled run, discovery processed `16,380` seeds with
`2047-2048` seeds per worker and balanced worker runtimes. That run found `7,525`
accepted circuits before merge.

### Merge Discovery

```bash
PYTHONPATH=src:src python - <<'PY'
import json, os
from pipeline.distributed.manifest import load_manifest
from pipeline.distributed.discovery_merge import run_circuit_store_merge

manifest = load_manifest(os.environ["MANIFEST"])
result = run_circuit_store_merge(manifest)
print(json.dumps({
    "merged_circuit_count": result.merged_circuit_count,
    "worker_circuit_counts": result.worker_circuit_counts,
    "report_path": str(result.report_path),
    "summary_path": str(result.summary_path),
}, indent=2))
PY
```

### Final Artifact Check

```bash
ls -lh "$RUN_ROOT"
ls -lh "$RUN_ROOT/circuits"
ls -lh "$RUN_ROOT/distributed/reports"
df -h "$OUTPUT_BASE"
```

```bash
python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
for rel in [
    "distributed/reports/pass1_sanity_report.json",
    "distributed/reports/pass2_reduce_report.json",
    "distributed/reports/discovery_merge_report.json",
    "circuits/summary.json",
]:
    path = root / rel
    print(rel, "exists=", path.exists(), "size=", path.stat().st_size if path.exists() else 0)
PY
```

## 7. Backup Essential Outputs Before Shutdown

Before terminating a rented GPU pod, copy the essential artifacts to durable
storage. The completed run's essential archive was about `2.9 GiB`; the full
output directory was much larger because it included worker-local intermediates.

Create an essential archive:

```bash
cd "$RUN_ROOT/.."

tar -czf "$(basename "$RUN_ROOT")-essential.tar.gz" \
  "$(basename "$RUN_ROOT")/circuits" \
  "$(basename "$RUN_ROOT")/candidates.pt" \
  "$(basename "$RUN_ROOT")/latent_stats.pt" \
  "$(basename "$RUN_ROOT")/top_ctx.pt" \
  "$(basename "$RUN_ROOT")/mid_ctx.pt" \
  "$(basename "$RUN_ROOT")/neg_ctx.pt" \
  "$(basename "$RUN_ROOT")/neg_ctx_stats.json" \
  "$(basename "$RUN_ROOT")/logit_ctx.pt" \
  "$(basename "$RUN_ROOT")/top_coactivation.pt" \
  "$(basename "$RUN_ROOT")/seq_repr.pt" \
  "$(basename "$RUN_ROOT")/distributed/manifest.json" \
  "$(basename "$RUN_ROOT")/distributed/reports" \
  "$(basename "$RUN_ROOT")/distributed/parts/candidate_selection" \
  "$(basename "$RUN_ROOT")/distributed"/*.log

ls -lh "$(basename "$RUN_ROOT")-essential.tar.gz"
sha256sum "$(basename "$RUN_ROOT")-essential.tar.gz"
```

Create a small provenance archive with the exact repo config and git state:

```bash
cd /workspace/turing

mkdir -p "$RUN_ROOT/provenance"
cp config.yaml "$RUN_ROOT/provenance/config.yaml"
git rev-parse HEAD > "$RUN_ROOT/provenance/git_head.txt"
git status --short > "$RUN_ROOT/provenance/git_status_short.txt"
git diff > "$RUN_ROOT/provenance/git_diff.patch"

cd "$RUN_ROOT/.."
tar -czf "$(basename "$RUN_ROOT")-provenance.tar.gz" \
  "$(basename "$RUN_ROOT")/provenance" \
  "$(basename "$RUN_ROOT")/distributed/manifest.json" \
  "$(basename "$RUN_ROOT")/distributed/reports"

ls -lh "$(basename "$RUN_ROOT")-provenance.tar.gz"
sha256sum "$(basename "$RUN_ROOT")-provenance.tar.gz"
```

Transfer from Windows using the pod's direct TCP SSH/SCP endpoint:

```powershell
scp -P <PORT> -i "$env:USERPROFILE\.ssh\id_ed25519" `
  root@<HOST>:/root/outputs/<RUN_ID>-essential.tar.gz `
  "X:\Projects\AIs\Turing\Publication\3 Implementation\Runs\<RUN_ID>-essential.tar.gz"

scp -P <PORT> -i "$env:USERPROFILE\.ssh\id_ed25519" `
  root@<HOST>:/root/outputs/<RUN_ID>-provenance.tar.gz `
  "X:\Projects\AIs\Turing\Publication\3 Implementation\Runs\<RUN_ID>-provenance.tar.gz"
```

Verify local hashes with:

```powershell
Get-FileHash "X:\Projects\AIs\Turing\Publication\3 Implementation\Runs\<RUN_ID>-essential.tar.gz" -Algorithm SHA256
Get-FileHash "X:\Projects\AIs\Turing\Publication\3 Implementation\Runs\<RUN_ID>-provenance.tar.gz" -Algorithm SHA256
```

## 8. Useful Recovery Commands

If I lose shell env vars:

```bash
export RUN_ROOT=/path/to/outputs/<run_id>
export MANIFEST="$RUN_ROOT/distributed/manifest.json"
export OUTPUT_BASE=$(dirname "$RUN_ROOT")
source /root/venvs/turing/bin/activate
cd /workspace/turing
```

If `git pull` is blocked by local pod config edits:

```bash
cp config.yaml /tmp/config.before-pull.yaml
git checkout -- config.yaml
git pull
cp config_examples/h100-8x-distributed-simple-exact.yaml config.yaml
```

If native fused is not available:

```bash
cd /workspace/turing/src/native
python setup.py build_ext --inplace
cd /workspace/turing
PYTHONPATH=src:src python - <<'PY'
from sae.fused_exact_topk import native_is_available
print(native_is_available())
PY
```
