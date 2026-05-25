# Native Fused Top-K Radix Fix Plan

Yes. Based on the data, the plan should be very focused: **do not touch cuBLASLt block GEMM yet**. The profiler says `local_topk_kernel` is 96.6% of time, while GEMM is 0.13%.

## Phase 0: Freeze Current Runtime Defaults
Keep production runs on:

```yaml
sae:
  encode_backend: "standard"
  topk_backend: "pytorch"
```

The native path should remain opt-in via `--use-native-fused-exact` / `TURINGLLM_FUSED_EXACT_TOPK_USE_NATIVE=1` only. The current native kernel is correctness-valid but not performance-valid.

## Phase 1: Replace Naive Local Top-K
Target: `src/native/linear_relu_topk_exact.cu`.

The broken part is here:

```31:101:src/native/linear_relu_topk_exact.cu
__global__ void local_topk_kernel(
    const c10::BFloat16* __restrict__ acts,
    c10::BFloat16* __restrict__ candidate_values,
    int32_t* __restrict__ candidate_indices,
    // ...
) {
  // ...
  for (int out = 0; out < local_k; ++out) {
    float best_value = -FLT_MAX;
    int best_index = -1;
    // scans block width again for every output
```

Replace this repeated max-selection with a radix-select local top-k kernel modeled on `src/sae/triton_topk.py`:

```1:19:src/sae/triton_topk.py
"""Triton radix-select top-k for non-negative BF16 tensors.

Works by streaming the [M, N] input in two phases:

  Phase 1 — eight 2-bit radix passes (MSB → LSB) to find the uint16 bit
             pattern of the K-th largest value (the pivot / threshold).
// ...
  Phase 2 — one collect pass that gathers values strictly above the
             threshold, plus exactly the right number of ties
```

Native algorithm:
- One CUDA block per row, same as now.
- Input is BF16 post-ReLU, so values are non-negative.
- Bitcast BF16 to `uint16`.
- Use 8 passes of 2-bit radix counting to find the local threshold.
- Collect all values above threshold plus needed ties.
- Write exactly `local_k` candidates for that feature block.

Success target:
- `local_topk_kernel` should drop from `~73 ms per block` at `B=128` to low single-digit ms or better.
- Full `B=128` native fused call should move from `~760 ms` toward the same order as the dense baseline, not hundreds of ms.

## Phase 2: Make Benchmarking Safer
Add a guard to `src/debug/benchmark_sae_encode.py` so native full-shape benchmarking does not accidentally hang again.

Recommended behavior:
- If `--use-native-fused-exact` and shape is large, print a warning unless `--allow-slow-native` is set.
- Or add `--profile-native-only` / `--native-max-batch-size` to make profiling intentional.

This is because the benchmark currently always times native if correctness passes:

```210:224:src/debug/benchmark_sae_encode.py
if valid_fused_exact:
    results.append(
        _time_backend(
            "fused_exact_topk",
            lambda: linear_relu_topk_exact(
                x,
                weight,
                bias,
```

## Phase 3: Re-profile Local Top-K
Run these after the radix kernel lands:

```bash
cd /workspace/turing/src

for B in 16 32 64 128 256; do
  echo "===== batch_size=$B ====="
  python -m debug.benchmark_sae_encode \
    --batch-size "$B" \
    --seq-len 64 \
    --warmup 1 \
    --iters 3 \
    --use-native-fused-exact
done
```

Then profile `B=128` again:

```bash
python - <<'PY'
import torch
from torch.profiler import profile, ProfilerActivity
from sae.fused_exact_topk import linear_relu_topk_exact

B, T, D, N, K = 128, 64, 1024, 40960, 128
x = torch.randn(B*T, D, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, device="cuda", dtype=torch.bfloat16)

linear_relu_topk_exact(x, w, b, K, block_n=4096, use_native=True)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    linear_relu_topk_exact(x, w, b, K, block_n=4096, use_native=True)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
PY
```

Pass criteria:
- `local_topk_kernel` no longer dominates at 96%.
- `merge_topk_kernel` or cuBLASLt may become visible, which is fine.
- Native path must still pass `native.tests.test_linear_relu_topk_exact`.

## Phase 4: Optimize Merge Only If Needed
Current merge is also repeated max-selection:

```104:187:src/native/linear_relu_topk_exact.cu
__global__ void merge_topk_kernel(
    const c10::BFloat16* __restrict__ candidate_values,
    const int32_t* __restrict__ candidate_indices,
    // ...
) {
  // ...
  for (int out = 0; out < k; ++out) {
```

But it only handles ~`num_blocks * k` candidates, usually `10 * 128 = 1280`, so it is much less urgent.

If merge rises above ~15-20% after Phase 1, replace it with the same radix-select approach over candidate values.

## Phase 5: Block Size Sweep
Only after local top-k is fixed, sweep `block_n`:

```bash
for BN in 1024 2048 4096 8192; do
  python -m debug.benchmark_sae_encode \
    --batch-size 256 \
    --seq-len 64 \
    --block-n "$BN" \
    --warmup 2 \
    --iters 5 \
    --use-native-fused-exact
done
```

Tradeoff:
- Smaller `block_n`: more GEMM launches, more candidate blocks, cheaper local selection per block.
- Larger `block_n`: fewer blocks, better GEMM shape, harder local selection.

## Phase 6: Decide Whether Deeper Fusion Is Worth It
Only if radix local top-k still cannot beat `cublaslt_relu + pytorch_topk`, consider deeper fusion:
- CUTLASS/CuTe GEMM epilogue that emits top-k candidates instead of block activations.
- More complex, higher risk, but avoids writing block activation buffers.

I would not start here. Your profiler says the immediate bug is the selector, not the GEMM.

Bottom line: **Phase 1 is the real fix**. Port the Triton radix-select idea into `local_topk_kernel`, validate, then re-profile.

## Production Promotion Results
H100 pass-1 profiling showed the native fused exact backend is worth promoting from env-var experiment to a guarded production H100 option.

Profile setup:
- GPU: 1x H100.
- `data.batch_size: 4096`.
- `data.n_shards: 32`.
- `first_pass.sae_encode_mode: "deferred"`.
- Native fused exact: `sae.encode_backend: "fused_exact_topk"`, `sae.fused_exact_topk_use_native: true`.
- The native fused exact path uses a production default feature block width of `10240` for `d_sae=40960`.
- Baseline: `sae.encode_backend: "standard"`, `sae.topk_backend: "pytorch"`.

Measured over 8 profiled pass-1 batches:

```text
Native fused exact:
  profiled_pass1_batch avg:       6.897 s
  deferred encode/update avg:     6.620 s
  peak CUDA allocated:           40.70 GiB

Standard PyTorch:
  profiled_pass1_batch avg:       9.670 s
  deferred encode/update avg:     9.398 s
  peak CUDA allocated:           55.32 GiB
```

Observed impact:
- `~1.40x` faster profiled pass-1 batches.
- `~14.62 GiB` lower peak CUDA allocation during the profiled window.
- Native fused exact avoids materializing the full dense SAE activation matrix and reduces allocator pressure.

Production guardrails added:
- `sae.fused_exact_topk_use_native` controls whether the native extension is required.
- The validated H100 block size (`10240`) is hardcoded as the production default; `TURINGLLM_FUSED_EXACT_TOPK_BLOCK_N` remains available only as an explicit developer override for future experiments.
- Native mode fails loudly if `linear_relu_topk_exact_ext` is missing.
- Native mode rejects unsafe block layouts where the trailing feature block is non-empty but narrower than `k`.
- Generic defaults stay on `standard + pytorch`; H100 native use is exposed through `config_examples/h100-1x-pass1-native-fused.yaml`.
