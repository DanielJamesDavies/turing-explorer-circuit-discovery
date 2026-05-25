"""Benchmark production SAE encode top-k backends.

This script benchmarks the measured pass1 hot path in isolation:

  1. current cublasLt Linear+ReLU + PyTorch top-k
  2. cublasLt Linear+ReLU + Triton top-k

Run on the H100 pod from the repo root or ``src`` directory:

    python -m debug.benchmark_sae_encode --batch-size 2048 --seq-len 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sae.fused_linear_relu import linear_relu
from sae.triton_topk import is_available as triton_topk_available
from sae.triton_topk import topk_nonneg_bf16


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    avg_ms: float
    min_ms: float
    max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--d-sae", type=int, default=40960)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_backend(
    name: str,
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    for _ in range(warmup):
        values, indices = fn()
        _synchronize()
        del values, indices

    times: list[float] = []
    for _ in range(iters):
        _synchronize()
        t0 = time.perf_counter()
        values, indices = fn()
        _synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)
        del values, indices

    result = BenchmarkResult(
        name=name,
        avg_ms=sum(times) / len(times),
        min_ms=min(times),
        max_ms=max(times),
    )
    print(
        f"{result.name:<34s} avg={result.avg_ms:9.3f} ms "
        f"min={result.min_ms:9.3f} ms max={result.max_ms:9.3f} ms"
    )
    return result


def _sorted_values(values: torch.Tensor) -> torch.Tensor:
    return values.float().sort(dim=-1, descending=True).values


def _check_values(
    name: str,
    values: torch.Tensor,
    indices: torch.Tensor,
    ref_values: torch.Tensor,
    pre_acts: torch.Tensor | None = None,
) -> None:
    if not torch.allclose(_sorted_values(values), _sorted_values(ref_values), atol=0, rtol=0):
        actual_sorted = _sorted_values(values)
        ref_sorted = _sorted_values(ref_values)
        max_abs = (actual_sorted - ref_sorted).abs().max().item()
        raise AssertionError(
            f"{name}: top-k values do not match production reference "
            f"(max_abs={max_abs:.6g})"
        )
    if indices.dtype != torch.int64:
        raise AssertionError(f"{name}: expected int64 indices, got {indices.dtype}")
    if pre_acts is not None:
        gathered = pre_acts.gather(-1, indices)
        if not torch.equal(gathered, values):
            raise AssertionError(f"{name}: indices do not gather returned values")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_sae_encode requires CUDA")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows = args.batch_size * args.seq_len
    print(
        "Shape: "
        f"B={args.batch_size} T={args.seq_len} M={rows} "
        f"D={args.d_model} N={args.d_sae} K={args.k}"
    )

    torch.manual_seed(0)
    x = torch.randn(rows, args.d_model, device=device, dtype=dtype)
    weight = torch.randn(args.d_sae, args.d_model, device=device, dtype=dtype)
    bias = torch.randn(args.d_sae, device=device, dtype=dtype)

    pre_acts = linear_relu(x, weight, bias)
    ref_values, ref_indices = pre_acts.topk(args.k, dim=-1, sorted=False)
    _synchronize()

    if not args.skip_correctness:
        print("Correctness checks:")
        _check_values("pytorch", ref_values, ref_indices, ref_values, pre_acts)
        print("  [PASS] pytorch_topk")

        if triton_topk_available():
            triton_values, triton_indices = topk_nonneg_bf16(pre_acts, args.k)
            _synchronize()
            _check_values("triton_topk", triton_values, triton_indices, ref_values, pre_acts)
            print("  [PASS] triton_topk")

    print("\nBenchmarks:")
    results: list[BenchmarkResult] = []
    results.append(
        _time_backend(
            "cublaslt_relu + pytorch_topk",
            lambda: linear_relu(x, weight, bias).topk(args.k, dim=-1, sorted=False),
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    if triton_topk_available():
        results.append(
            _time_backend(
                "cublaslt_relu + triton_topk",
                lambda: topk_nonneg_bf16(linear_relu(x, weight, bias), args.k),
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    best = min(results, key=lambda item: item.avg_ms)
    baseline = results[0]
    speedup = baseline.avg_ms / best.avg_ms
    print(f"\nBest: {best.name} ({speedup:.3f}x vs baseline)")


if __name__ == "__main__":
    main()
