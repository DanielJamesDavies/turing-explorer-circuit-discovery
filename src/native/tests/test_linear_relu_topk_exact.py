"""Correctness tests for the exact fused Linear+ReLU+TopK prototype."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sae.fused_exact_topk import linear_relu_topk_exact, native_is_available


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _sorted_values(values: torch.Tensor) -> torch.Tensor:
    return values.float().sort(dim=-1, descending=True).values


def _tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 1e-2
    if dtype == torch.float16:
        return 1e-3
    return 1e-5


def _assert_matches_reference(
    label: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    *,
    block_n: int,
    use_native: bool = False,
) -> None:
    ref_acts = torch.relu(F.linear(x, weight, bias))
    ref_values, _ = ref_acts.topk(k, dim=-1, sorted=False)
    values, indices = linear_relu_topk_exact(
        x,
        weight,
        bias,
        k,
        block_n=block_n,
        use_native=use_native,
    )

    assert values.shape == ref_values.shape, f"{label}: value shape mismatch"
    assert indices.shape == ref_values.shape, f"{label}: index shape mismatch"
    assert indices.dtype == torch.int64, f"{label}: indices must be int64"
    tol = _tolerance(values.dtype)
    assert torch.allclose(_sorted_values(values), _sorted_values(ref_values), atol=tol, rtol=0), (
        f"{label}: top-k value multiset mismatch"
    )
    gathered = ref_acts.gather(-1, indices)
    assert torch.allclose(gathered, values, atol=tol, rtol=0), (
        f"{label}: indices do not gather returned values"
    )

    rows = indices.reshape(-1, indices.shape[-1])
    for row_idx, row in enumerate(rows):
        assert row.unique().numel() == row.numel(), f"{label}: duplicate indices in row {row_idx}"


def test_random_2d() -> None:
    torch.manual_seed(0)
    x = torch.randn(17, 32, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(128, 32, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(128, device=DEVICE, dtype=DTYPE)
    _assert_matches_reference("random_2d", x, weight, bias, 16, block_n=32)


def test_3d_pass1_shape() -> None:
    torch.manual_seed(1)
    x = torch.randn(3, 7, 32, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(160, 32, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(160, device=DEVICE, dtype=DTYPE)
    _assert_matches_reference("3d", x, weight, bias, 24, block_n=40)


def test_ties_and_sparse_values() -> None:
    x = torch.ones(5, 16, device=DEVICE, dtype=DTYPE)
    weight = torch.zeros(64, 16, device=DEVICE, dtype=DTYPE)
    bias = torch.zeros(64, device=DEVICE, dtype=DTYPE)
    bias[::2] = 1.0
    _assert_matches_reference("ties_sparse", x, weight, bias, 12, block_n=16)


def test_non_divisible_block() -> None:
    torch.manual_seed(2)
    x = torch.randn(11, 24, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(70, 24, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(70, device=DEVICE, dtype=DTYPE)
    _assert_matches_reference("non_divisible_block", x, weight, bias, 13, block_n=32)


def _native_ready() -> bool:
    return torch.cuda.is_available() and native_is_available()


def test_native_random_2d() -> None:
    if not _native_ready():
        return
    torch.manual_seed(3)
    x = torch.randn(17, 32, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(128, 32, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(128, device=DEVICE, dtype=torch.bfloat16)
    _assert_matches_reference("native_random_2d", x, weight, bias, 16, block_n=32, use_native=True)


def test_native_3d_pass1_shape() -> None:
    if not _native_ready():
        return
    torch.manual_seed(4)
    x = torch.randn(3, 7, 32, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(160, 32, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(160, device=DEVICE, dtype=torch.bfloat16)
    _assert_matches_reference("native_3d", x, weight, bias, 24, block_n=40, use_native=True)


def test_native_non_divisible_block() -> None:
    if not _native_ready():
        return
    torch.manual_seed(5)
    x = torch.randn(11, 24, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(70, 24, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(70, device=DEVICE, dtype=torch.bfloat16)
    _assert_matches_reference(
        "native_non_divisible_block",
        x,
        weight,
        bias,
        13,
        block_n=32,
        use_native=True,
    )


def test_native_matches_python_path() -> None:
    if not _native_ready():
        return
    torch.manual_seed(6)
    x = torch.randn(9, 19, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(75, 19, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(75, device=DEVICE, dtype=torch.bfloat16)
    py_values, py_indices = linear_relu_topk_exact(x, weight, bias, 15, block_n=20, use_native=False)
    native_values, native_indices = linear_relu_topk_exact(x, weight, bias, 15, block_n=20, use_native=True)

    tol = _tolerance(native_values.dtype)
    assert torch.allclose(_sorted_values(native_values), _sorted_values(py_values), atol=tol, rtol=0)
    ref_acts = torch.relu(F.linear(x, weight, bias))
    assert torch.allclose(ref_acts.gather(-1, native_indices), native_values, atol=tol, rtol=0)
    assert torch.allclose(ref_acts.gather(-1, py_indices), py_values, atol=tol, rtol=0)


def main() -> None:
    test_random_2d()
    print("  [PASS] random_2d")
    test_3d_pass1_shape()
    print("  [PASS] 3d_pass1_shape")
    test_ties_and_sparse_values()
    print("  [PASS] ties_and_sparse_values")
    test_non_divisible_block()
    print("  [PASS] non_divisible_block")
    if _native_ready():
        test_native_random_2d()
        print("  [PASS] native_random_2d")
        test_native_3d_pass1_shape()
        print("  [PASS] native_3d_pass1_shape")
        test_native_non_divisible_block()
        print("  [PASS] native_non_divisible_block")
        test_native_matches_python_path()
        print("  [PASS] native_matches_python_path")
    else:
        print("  [SKIP] native extension unavailable")


if __name__ == "__main__":
    main()
