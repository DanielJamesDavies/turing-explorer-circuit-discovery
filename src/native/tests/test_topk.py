"""Correctness tests and benchmark for the Triton radix-select top-k kernel.

Run from the project src/ directory:
    python -m native.tests.test_topk
    python native/tests/test_topk.py

The benchmark reports the wall-clock time of both implementations for the
exact shape used in the SAE encode path (M=16640, N=40960, K=128, BF16)
and prints a speedup ratio.

Switching backends programmatically in your own scripts:
    from sae.topk_sae import set_topk_backend, get_topk_backend
    set_topk_backend("pytorch")   # force PyTorch mbtopk
    set_topk_backend("triton")    # use Triton radix-select (default)
"""

from __future__ import annotations

import sys
import os
import time

import torch

# Allow running as `python native/tests/test_topk.py` from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sae.triton_topk import topk_nonneg_bf16, is_available, _get_kernel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sorted_vals(vals: torch.Tensor, descending: bool = True) -> torch.Tensor:
    """Sort the K-dim of vals for order-independent comparison."""
    return vals.sort(dim=-1, descending=descending).values


def _ref_topk(x: torch.Tensor, k: int):
    """PyTorch reference: sorted=True so comparison is deterministic."""
    return x.topk(k, dim=-1, sorted=True)


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_random(M: int = 64, N: int = 512, K: int = 16) -> None:
    """Random non-negative BF16 values — typical case."""
    x = torch.rand(M, N, dtype=DTYPE, device=DEVICE).relu_()

    ref_v, ref_i = _ref_topk(x, K)
    new_v, new_i = topk_nonneg_bf16(x, K)

    new_v_sorted = _sorted_vals(new_v)
    assert torch.allclose(new_v_sorted, ref_v, atol=0, rtol=0), (
        f"[random] values mismatch\n  ref : {ref_v[0]}\n  got : {new_v_sorted[0]}"
    )
    # Verify indices actually map back to the right values.
    gathered = x.gather(-1, new_i)
    assert torch.allclose(gathered, new_v, atol=0, rtol=0), (
        "[random] index→value round-trip failed"
    )
    print(f"  [PASS] random        M={M}, N={N}, K={K}")


def test_ties(M: int = 16, N: int = 256, K: int = 8) -> None:
    """Input with many ties at the threshold — exercises tie-break logic."""
    # Fill with a constant so every element is a tie.
    x = torch.full((M, N), 1.0, dtype=DTYPE, device=DEVICE)

    ref_v, _ = _ref_topk(x, K)
    new_v, new_i = topk_nonneg_bf16(x, K)

    # All top-k values should equal 1.0.
    assert (new_v == 1.0).all(), f"[ties] expected all 1.0, got {new_v[0]}"
    # Indices must be unique per row (we can't pick the same position twice).
    for row in range(M):
        assert new_i[row].unique().numel() == K, (
            f"[ties] duplicate indices in row {row}: {new_i[row]}"
        )
    print(f"  [PASS] ties          M={M}, N={N}, K={K}")


def test_sparse(M: int = 32, N: int = 1024, K: int = 8) -> None:
    """Sparse input: fewer non-zeros than K — threshold should be 0."""
    x = torch.zeros(M, N, dtype=DTYPE, device=DEVICE)
    # Set exactly K//2 non-zero entries per row.
    nonzero_count = K // 2
    for row in range(M):
        cols = torch.randperm(N, device=DEVICE)[:nonzero_count]
        x[row, cols] = torch.rand(nonzero_count, dtype=DTYPE, device=DEVICE) + 0.1

    ref_v, _ = _ref_topk(x, K)
    new_v, new_i = topk_nonneg_bf16(x, K)

    new_v_sorted = _sorted_vals(new_v)
    assert torch.allclose(new_v_sorted, ref_v, atol=0, rtol=0), (
        f"[sparse] values mismatch\n  ref : {ref_v[0]}\n  got : {new_v_sorted[0]}"
    )
    print(f"  [PASS] sparse        M={M}, N={N}, K={K}")


def test_3d(B: int = 4, T: int = 8, N: int = 512, K: int = 16) -> None:
    """3-D input [..., N] — verifies shape handling in the Python wrapper."""
    x = torch.rand(B, T, N, dtype=DTYPE, device=DEVICE).relu_()

    ref_v, ref_i = _ref_topk(x, K)
    new_v, new_i = topk_nonneg_bf16(x, K)

    assert new_v.shape == (B, T, K), f"[3d] wrong shape {new_v.shape}"
    assert new_i.shape == (B, T, K), f"[3d] wrong shape {new_i.shape}"
    assert new_i.dtype == torch.int64, f"[3d] expected int64 indices, got {new_i.dtype}"
    new_v_sorted = _sorted_vals(new_v)
    assert torch.allclose(new_v_sorted, ref_v, atol=0, rtol=0), (
        f"[3d] values mismatch"
    )
    print(f"  [PASS] 3-D shape     B={B}, T={T}, N={N}, K={K}")


def test_full_size() -> None:
    """Full pipeline shape — M=16640, N=40960, K=128."""
    M, N, K = 16_640, 40_960, 128
    x = torch.rand(M, N, dtype=DTYPE, device=DEVICE).relu_()

    ref_v, _ = _ref_topk(x, K)
    new_v, new_i = topk_nonneg_bf16(x, K)

    new_v_sorted = _sorted_vals(new_v)
    assert torch.allclose(new_v_sorted, ref_v, atol=0, rtol=0), (
        f"[full_size] values mismatch (first row):\n"
        f"  ref : {ref_v[0, :8]}\n"
        f"  got : {new_v_sorted[0, :8]}"
    )
    print(f"  [PASS] full-size     M={M}, N={N}, K={K}")


# ── Agreement test (Triton ↔ PyTorch) ────────────────────────────────────────

def _check_agreement(x: torch.Tensor, k: int, label: str) -> None:
    """Assert Triton and PyTorch topk select the same K positions.

    Strategy
    --------
    Sort both sets of (value, index) pairs by value descending.  For any
    element whose value is *strictly above* the minimum chosen value (no tie at
    the boundary), the position *must* appear in both outputs.  Elements at the
    minimum value are "ties at the threshold" — different valid implementations
    may pick different positions among equals, so we only check that the *count*
    of ties included is correct, not which specific ones.
    """
    assert x.is_cuda and x.dtype == torch.bfloat16, "agreement test requires CUDA BF16"

    # PyTorch reference (unsorted — mimic production usage).
    pt_vals, pt_idxs = x.topk(k, dim=-1, sorted=False)
    tr_vals, tr_idxs = topk_nonneg_bf16(x, k)

    rows = x.shape[0] if x.dim() == 2 else x.reshape(-1, x.shape[-1]).shape[0]
    xf   = x.reshape(rows, -1)
    pt_v = pt_vals.reshape(rows, k)
    tr_v = tr_vals.reshape(rows, k)
    pt_i = pt_idxs.reshape(rows, k)
    tr_i = tr_idxs.reshape(rows, k)

    for row in range(rows):
        # Canonical sort: (value desc, index asc).
        # Two-step stable sort: secondary key first, then primary key.
        def _canonical(vals: torch.Tensor, idxs: torch.Tensor):
            # Step 1: sort by index ascending (secondary key).
            by_idx   = idxs.argsort(stable=True)
            v_tmp    = vals[by_idx]
            i_tmp    = idxs[by_idx]
            # Step 2: stable-sort by value descending (primary key).
            by_val   = v_tmp.float().argsort(descending=True, stable=True)
            return v_tmp[by_val], i_tmp[by_val]

        pt_sv, pt_si = _canonical(pt_v[row], pt_i[row].long())
        tr_sv, tr_si = _canonical(tr_v[row], tr_i[row].long())

        # 1. Values must match exactly (canonically sorted).
        assert torch.equal(pt_sv, tr_sv), (
            f"[{label}] row {row}: value sets differ\n"
            f"  PyTorch : {pt_sv}\n"
            f"  Triton  : {tr_sv}"
        )

        threshold = pt_sv[-1]  # the K-th largest value (may be a tie)

        # 2. Every position with value > threshold must appear in BOTH outputs.
        #    With canonical (value desc, idx asc) ordering, equal-valued elements
        #    are in the same index order on both sides, so a direct comparison works.
        unambiguous_mask = pt_sv > threshold
        assert torch.equal(
            pt_si[unambiguous_mask],
            tr_si[unambiguous_mask],
        ), (
            f"[{label}] row {row}: index mismatch for elements strictly above threshold\n"
            f"  PyTorch indices : {pt_si[unambiguous_mask].tolist()}\n"
            f"  Triton  indices : {tr_si[unambiguous_mask].tolist()}"
        )

        # 3. For tied elements at the threshold, only check the count is correct
        #    (both must include exactly the right number of ties, but either
        #    implementation may pick different positions among equals).
        n_ties_pt = int((pt_v[row] == threshold).sum())
        n_ties_tr = int((tr_v[row] == threshold).sum())
        assert n_ties_pt == n_ties_tr, (
            f"[{label}] row {row}: tie count differs at threshold={threshold}\n"
            f"  PyTorch: {n_ties_pt}  Triton: {n_ties_tr}"
        )

        # 4. Triton indices must be valid positions (point to correct values).
        gathered = xf[row][tr_si]
        assert torch.equal(gathered, tr_sv), (
            f"[{label}] row {row}: Triton index→value round-trip failed"
        )


def test_agreement() -> None:
    """Triton and PyTorch topk must select the exact same positions."""
    cases = [
        # (shape, K, description)
        ((64,   512),  16,  "2-D random"),
        ((16,   256),   8,  "ties (all ones)"),
        ((32,  1024),   8,  "sparse (many zeros)"),
        ((4, 8, 512),  16,  "3-D random"),
        ((16640, 40960), 128, "full pipeline shape"),
    ]

    for shape, k, desc in cases:
        if "ties" in desc:
            x = torch.ones(*shape, dtype=DTYPE, device=DEVICE)
        elif "sparse" in desc:
            x = torch.zeros(*shape, dtype=DTYPE, device=DEVICE)
            xf = x.reshape(-1, shape[-1])
            for row in range(xf.shape[0]):
                cols = torch.randperm(shape[-1], device=DEVICE)[: k // 2]
                xf[row, cols] = torch.rand(k // 2, dtype=DTYPE, device=DEVICE) + 0.1
        else:
            x = torch.rand(*shape, dtype=DTYPE, device=DEVICE).relu_()

        _check_agreement(x, k, desc)
        print(f"  [PASS] {desc:<26s}  shape={list(shape)}, K={k}")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(
    M: int = 16_640,
    N: int = 40_960,
    K: int = 128,
    n_warmup: int = 5,
    n_iter: int = 50,
) -> None:
    """Compare PyTorch topk vs Triton radix-select for the SAE encode shape."""
    x = torch.rand(M, N, dtype=DTYPE, device=DEVICE).relu_()

    def _time(fn, label: str) -> float:
        # Warmup
        for _ in range(n_warmup):
            fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        # Timed runs
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        elapsed_ms = (time.perf_counter() - t0) / n_iter * 1_000
        print(f"  {label:<30s}  {elapsed_ms:7.3f} ms / call")
        return elapsed_ms

    print(f"\n  Shape: M={M}, N={N}, K={K}, dtype={DTYPE}, device={DEVICE}")
    print(f"  ({n_iter} iterations after {n_warmup} warmup)\n")

    pt_ms = _time(lambda: x.topk(K, dim=-1, sorted=False), "PyTorch topk (mbtopk)")
    tr_ms = _time(lambda: topk_nonneg_bf16(x, K),          "Triton radix-select   ")

    speedup = pt_ms / tr_ms
    faster  = "Triton" if speedup > 1 else "PyTorch"
    ratio   = speedup if speedup > 1 else 1.0 / speedup
    print(f"\n  {faster} is {ratio:.2f}x faster.")


# ── Profiler ──────────────────────────────────────────────────────────────────

def profile_kernel(
    M: int = 16_640,
    N: int = 40_960,
    K: int = 128,
    n_warmup: int = 5,
    n_profile: int = 10,
    export_trace: bool = False,
) -> None:
    """Profile the Triton radix-select kernel using torch.profiler.

    Reports:
      • The BLOCK_N / num_warps config chosen by the autotuner.
      • Per-CUDA-kernel averages (time, # calls) sorted by GPU time.
      • Optionally exports a Chrome trace to topk_profile.json.
    """
    x = torch.rand(M, N, dtype=DTYPE, device=DEVICE).relu_()

    # Ensure the kernel is compiled and autotuned before profiling.
    for _ in range(n_warmup):
        topk_nonneg_bf16(x, K)
    torch.cuda.synchronize(DEVICE)

    # ── Report the autotuned configuration ───────────────────────────────────
    kernel = _get_kernel()
    if hasattr(kernel, "best_config"):
        cfg = kernel.best_config
        print(f"\n  Autotuned config : BLOCK_N={cfg.kwargs['BLOCK_N']}, "
              f"num_warps={cfg.num_warps}")
        n_tiles = N // cfg.kwargs["BLOCK_N"]
        n_reductions = n_tiles * (1 + 4 * 7)   # pass 0: 1/tile; passes 1-7: 4/tile
        print(f"  Tiles per row    : {n_tiles}  ({N} / {cfg.kwargs['BLOCK_N']})")
        print(f"  Phase-1 reductions/row : {n_reductions}  "
              f"(pass 0: 1×{n_tiles} tiles + passes 1-7: 4×{n_tiles}×7)")
    else:
        print("  (autotuned config not yet available — run a warmup first)")

    # ── torch.profiler ────────────────────────────────────────────────────────
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_flops=False,
    ) as prof:
        for _ in range(n_profile):
            topk_nonneg_bf16(x, K)
        torch.cuda.synchronize(DEVICE)

    # Print per-kernel averages sorted by total CUDA time.
    print()
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
    ))

    if export_trace:
        path = "topk_profile.json"
        prof.export_chrome_trace(path)
        print(f"  Chrome trace written to {path}")
        print(f"  Open at chrome://tracing or https://ui.perfetto.dev")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Test & benchmark the Triton top-k kernel.")
    parser.add_argument("--profile", action="store_true",
                        help="Run torch.profiler on the Triton kernel after benchmarking.")
    parser.add_argument("--trace", action="store_true",
                        help="Export a Chrome trace (topk_profile.json) — implies --profile.")
    args = parser.parse_args()

    if not is_available():
        print("Triton is not available on this machine — skipping tests.")
        return

    print(f"\n{'='*60}")
    print("  Correctness tests")
    print(f"{'='*60}")
    test_random()
    test_ties()
    test_sparse()
    test_3d()
    test_full_size()
    print(f"\n  All correctness tests passed.\n")

    print(f"{'='*60}")
    print("  Agreement tests  (Triton ↔ PyTorch, same positions)")
    print(f"{'='*60}")
    test_agreement()
    print(f"\n  All agreement tests passed.\n")

    print(f"{'='*60}")
    print("  Benchmark  (full pipeline shape)")
    print(f"{'='*60}")
    benchmark()
    print()

    if args.profile or args.trace:
        print(f"{'='*60}")
        print("  Kernel profile  (Triton radix-select)")
        print(f"{'='*60}")
        profile_kernel(export_trace=args.trace)
        print()


if __name__ == "__main__":
    main()
