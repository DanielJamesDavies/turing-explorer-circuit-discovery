"""Triton radix-select top-k for non-negative BF16 tensors.

Works by streaming the [M, N] input in two phases:

  Phase 1 — eight 2-bit radix passes (MSB → LSB) to find the uint16 bit
             pattern of the K-th largest value (the pivot / threshold).
             Each pass examines a 2-bit digit (4 buckets: 0-3) and
             accumulates 4 explicit scalar counters — only 4 warp
             reductions per tile instead of 16, trading 2× more passes
             for 4× fewer reductions per pass (net 2× fewer total).

             Pass 0 is specialised: for non-negative BF16 bit 15 = 0
             always, so the digit ∈ {0, 1} and only c1 (count of
             elements with bit 14 = 1) is needed (1 reduction vs 4).

  Phase 2 — one collect pass that gathers values strictly above the
             threshold, plus exactly the right number of ties
             (values == threshold) via cumulative counting, to produce
             exactly K outputs per row.

Assumptions:
  • Input dtype is bfloat16.
  • All values are >= 0 (e.g. post-ReLU).
    Under this constraint the uint16 bit representation of a BF16 value is
    monotone in its numeric value, so ordinary integer comparisons on the
    raw bits give the correct ordering without any sign-bit flip.
  • N is divisible by the smallest BLOCK_N in the autotune configs (256).

Typical usage in the SAE encode path:
    vals, idxs = topk_nonneg_bf16(pre_acts, k=128)
"""

from __future__ import annotations

import torch

_AVAILABLE: bool | None = None


def is_available() -> bool:
    """Returns True if Triton is importable and CUDA is present."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import triton  # noqa: F401
            _AVAILABLE = torch.cuda.is_available()
        except ImportError:
            _AVAILABLE = False
    return _AVAILABLE


# ── Kernel (only compiled when Triton is present) ─────────────────────────────

def _prune_configs(configs, named_args, **kwargs):
    """Drop configs where BLOCK_N doesn't divide N — prevents no-op tile loops."""
    N = named_args["N"]
    return [c for c in configs if N % c.kwargs["BLOCK_N"] == 0]


def _build_kernel():
    """Deferred import so the module loads fine on CPU-only machines."""
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            # Small BLOCK_N — used when N is small (test cases: 256, 512, 1024).
            triton.Config({"BLOCK_N": 256},  num_warps=4),
            triton.Config({"BLOCK_N": 256},  num_warps=8),
            triton.Config({"BLOCK_N": 512},  num_warps=8),
            triton.Config({"BLOCK_N": 512},  num_warps=16),
            triton.Config({"BLOCK_N": 1024}, num_warps=16),
            # Large BLOCK_N — fewer tiles → fewer total reductions per pass.
            # For N=40960: BLOCK_N=8192 gives 5 tiles vs 80 with BLOCK_N=512.
            # Both are powers of 2 dividing 40960 = 5×2¹³.
            triton.Config({"BLOCK_N": 2048}, num_warps=4),
            triton.Config({"BLOCK_N": 2048}, num_warps=8),
            triton.Config({"BLOCK_N": 4096}, num_warps=8),
            triton.Config({"BLOCK_N": 4096}, num_warps=16),
            triton.Config({"BLOCK_N": 8192}, num_warps=16),
            triton.Config({"BLOCK_N": 8192}, num_warps=32),
        ],
        key=["N", "K"],
        prune_configs_by={"early_config_prune": _prune_configs},
    )
    @triton.jit
    def _radix_topk_kernel(
        X_ptr,                  # [M, N] bfloat16, non-negative (post-ReLU)
        Vals_ptr,               # [M, K] bfloat16 output
        Idxs_ptr,               # [M, K] int32 output
        N,                      # number of elements per row (runtime)
        K,                      # number of top elements (runtime)
        BLOCK_N: tl.constexpr,  # elements per tile — set by autotuner
    ):
        """
        One Triton program (CTA) per row of X.

        Phase 1 — eight 2-bit radix passes to find the pivot value.
          Each pass streams all N elements and accumulates 4 explicit scalar
          variables (c0..c3) — one per 2-bit bucket — as loop-carried
          accumulators.  Only 4 warp reductions per tile (vs 16 for 4-bit
          digits), giving 2× fewer total reductions per phase.  With large
          BLOCK_N the tile count drops proportionally, compounding the win.
          Pass 0 is further specialised: for non-negative BF16 bit 15 = 0
          always, so only 1 reduction per tile is needed (c1 only).
          Reversed prefix-sums of c0..c3 give the threshold digit, committed
          to `threshold` bit by bit.

        Phase 2 — one streaming pass to gather top-K values.
          Elements strictly above threshold are always included.  For ties at
          the threshold, an in-tile cumulative count (tie_rank) combined with
          a running cross-tile counter (ties_seen) includes exactly the right
          number.  Parallel scatter positions are computed via tl.cumsum on
          the boolean include mask.
        """
        row      = tl.program_id(0)
        base     = row * N
        out_base = row * K
        n_tiles  = N // BLOCK_N

        # ── Phase 1: 8×2-bit radix select ────────────────────────────────────
        #
        # threshold  — accumulates the 16-bit bit pattern of the pivot value.
        # k_rem      — elements still needed from the current 2-bit bucket.
        #
        # Using 2-bit digits (4 buckets: 0-3) requires 8 passes to cover all
        # 16 bits.  This is 2× more passes than the 4-bit approach but only
        # 4 reductions per tile instead of 16, giving 2× fewer total reductions
        # per phase.  Combined with large BLOCK_N (fewer tiles), the total
        # reduction count drops dramatically.
        #
        threshold = tl.zeros([], dtype=tl.int32)
        k_rem     = tl.cast(K, tl.int32)

        for pass_idx in tl.static_range(8):   # fully unrolled: MSB first, 2 bits/pass
            # Python-level constants resolved at JIT trace time:
            shift_val        = 14 - pass_idx * 2    # 14, 12, 10, 8, 6, 4, 2, 0
            prefix_shift_val = 16 - pass_idx * 2    # 16, 14, 12, 10, 8, 6, 4, 2
            multiplier       = 1 << shift_val        # 16384, 4096, ..., 1

            shift        = tl.constexpr(shift_val)
            prefix_shift = tl.constexpr(prefix_shift_val)

            if pass_idx == 0:
                # ── Specialised pass 0 ─────────────────────────────────────────
                # For non-negative BF16, bit 15 = 0 always.  The 2-bit digit at
                # shift=14 is therefore always 0 or 1 (never 2 or 3), so c2=c3=0
                # and we only need ONE reduction per tile: c1 = count(bit14 == 1).
                c1_0 = tl.zeros([], tl.int32)
                for tile in range(n_tiles):
                    offs  = base + tile * BLOCK_N + tl.arange(0, BLOCK_N)
                    x_bf  = tl.load(X_ptr + offs)
                    x_i32 = x_bf.to(tl.int16, bitcast=True).to(tl.int32)
                    c1_0 += tl.sum(((x_i32 >> tl.constexpr(14)) & 1).to(tl.int32))

                # threshold_digit = 1 iff count(bit14=1) >= k_rem, else 0.
                threshold_digit = tl.cast(c1_0 >= k_rem, tl.int32)
                threshold       = threshold | (threshold_digit * multiplier)
                # n_above = elements with digit > threshold_digit.
                # digit ∈ {0,1}: n_above = c1 if threshold_digit==0 else 0.
                n_above = tl.cast(threshold_digit < 1, tl.int32) * c1_0
                k_rem   = k_rem - n_above

            else:
                # ── Normal 2-bit pass ──────────────────────────────────────────
                # 4 explicit scalar loop-carried accumulators — one per 2-bit digit.
                # Scalar phi nodes are robustly supported across all Triton versions.
                c0 = tl.zeros([], tl.int32);  c1 = tl.zeros([], tl.int32)
                c2 = tl.zeros([], tl.int32);  c3 = tl.zeros([], tl.int32)

                for tile in range(n_tiles):
                    offs  = base + tile * BLOCK_N + tl.arange(0, BLOCK_N)
                    x_bf  = tl.load(X_ptr + offs)
                    # Bitcast BF16 → int16, widen to int32.
                    # Non-negative BF16: bit-15 == 0, so integer bit-order equals
                    # numerical order — no sign-bit correction needed.
                    x_i32 = x_bf.to(tl.int16, bitcast=True).to(tl.int32)

                    # Only count elements matching the prefix committed so far.
                    in_pfx = (x_i32 >> prefix_shift) == (threshold >> prefix_shift)
                    digit  = (x_i32 >> shift) & tl.constexpr(0x3)   # 2-bit bucket

                    # Count per bucket — 4 independent masked reductions per tile.
                    c0 += tl.sum((in_pfx & (digit == 0)).to(tl.int32))
                    c1 += tl.sum((in_pfx & (digit == 1)).to(tl.int32))
                    c2 += tl.sum((in_pfx & (digit == 2)).to(tl.int32))
                    c3 += tl.sum((in_pfx & (digit == 3)).to(tl.int32))

                # ── Find this pass's threshold digit (0-3) ────────────────────
                #
                # rev[d] = sum(c[d..3]) = # in-prefix elements with digit >= d.
                # rev is non-increasing; (rev[d] >= k_rem) forms a contiguous
                # True-block from d=0 to d=threshold_digit.
                # threshold_digit = (count of True entries across d=0..3) - 1.
                #
                rev3 = c3
                rev2 = rev3 + c2
                rev1 = rev2 + c1
                rev0 = rev1 + c0

                n_trues = (
                    tl.cast(rev0 >= k_rem, tl.int32) +
                    tl.cast(rev1 >= k_rem, tl.int32) +
                    tl.cast(rev2 >= k_rem, tl.int32) +
                    tl.cast(rev3 >= k_rem, tl.int32)
                )
                threshold_digit = n_trues - 1   # 0..3

                # Commit this digit into the running bit-pattern of the threshold.
                threshold = threshold | (threshold_digit * multiplier)

                # n_above = sum(c[d] for d > threshold_digit).
                n_above = (
                    tl.cast(threshold_digit < 1, tl.int32) * c1 +
                    tl.cast(threshold_digit < 2, tl.int32) * c2 +
                    tl.cast(threshold_digit < 3, tl.int32) * c3
                )
                k_rem = k_rem - n_above

        # ── Phase 2: collect values > threshold + k_rem ties == threshold ─────
        #
        # write_ptr  — next free slot in [M, K] output row (advances 0 → K).
        # ties_seen  — cumulative count of at-threshold elements across tiles,
        #              used to give each tie a unique global rank.
        #
        write_ptr = tl.zeros([], dtype=tl.int32)
        ties_seen = tl.zeros([], dtype=tl.int32)

        for tile in range(n_tiles):
            offs   = base + tile * BLOCK_N + tl.arange(0, BLOCK_N)
            x_bf   = tl.load(X_ptr + offs)
            x_i32  = x_bf.to(tl.int16, bitcast=True).to(tl.int32)
            glob_i = (tile * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)

            above  = x_i32 > threshold
            at_thr = x_i32 == threshold

            # 1-based cumulative count of at-threshold elements in this tile.
            tie_rank = tl.cumsum(at_thr.to(tl.int32), axis=0)
            # Include at-threshold elements until the global quota (k_rem) is met.
            incl_tie = at_thr & (ties_seen + tie_rank <= k_rem)
            include  = above | incl_tie

            # Parallel scatter: prefix-sum gives 1-based per-tile write offsets;
            # subtract 1 to make them 0-based, then add the running write_ptr.
            pos = write_ptr + tl.cumsum(include.to(tl.int32), axis=0) - 1
            tl.store(Vals_ptr + out_base + pos, x_bf,   mask=include)
            tl.store(Idxs_ptr + out_base + pos, glob_i, mask=include)

            write_ptr += tl.sum(include.to(tl.int32))
            ties_seen += tl.sum(at_thr.to(tl.int32))

    return _radix_topk_kernel


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = _build_kernel()
    return _kernel


# ── Public API ────────────────────────────────────────────────────────────────

def topk_nonneg_bf16(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the last dimension for a non-negative BF16 tensor.

    Semantically equivalent to ``x.topk(k, dim=-1, sorted=False)`` but uses
    a specialised Triton radix-select kernel that exploits:
      • Fixed k (SAE hyper-parameter, typically 128)
      • Non-negative inputs (post-ReLU) — BF16 bit patterns preserve order
      • Contiguous row layout for maximum memory-bandwidth utilisation

    Args:
        x: BF16 tensor of shape ``(..., N)``.  All values must be >= 0.
           N must be divisible by 256.
        k: Number of top elements to select from the last dimension.

    Returns:
        values:  ``(..., k)`` BF16 — the top-k values (unsorted).
        indices: ``(..., k)`` int64 — their positions in the last dimension.
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    assert N % 256 == 0, (
        f"topk_nonneg_bf16 requires N % 256 == 0 (smallest BLOCK_N). Got N={N}."
    )

    x_2d = x.reshape(-1, N)   # [M, N]
    M    = x_2d.shape[0]

    vals = torch.empty(M, k, dtype=x.dtype,     device=x.device)
    idxs = torch.empty(M, k, dtype=torch.int32, device=x.device)

    _get_kernel()[(M,)](x_2d, vals, idxs, N, k)

    out_shape = orig_shape[:-1] + (k,)
    # Return int64 indices: compatible with scatter_add_ and all PyTorch index ops.
    return vals.view(out_shape), idxs.view(out_shape).long()
