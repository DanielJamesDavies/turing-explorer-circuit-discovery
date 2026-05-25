"""Correctness tests for experimental SAE encode/top-k backends."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sae.blockwise_topk import topk_blockwise
from sae.fused_linear_relu_topk import linear_relu_topk_blockwise


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _sorted_values(values: torch.Tensor) -> torch.Tensor:
    return values.float().sort(dim=-1, descending=True).values


def _assert_value_multiset_equal(
    label: str,
    actual_values: torch.Tensor,
    expected_values: torch.Tensor,
) -> None:
    assert torch.allclose(
        _sorted_values(actual_values),
        _sorted_values(expected_values),
        atol=0,
        rtol=0,
    ), f"{label}: top-k value multiset mismatch"


def _assert_indices_gather_values(
    label: str,
    source: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    assert indices.dtype == torch.int64, f"{label}: expected int64 indices, got {indices.dtype}"
    gathered = source.gather(-1, indices)
    assert torch.equal(gathered, values), f"{label}: indices do not gather values"
    rows = indices.reshape(-1, indices.shape[-1])
    for row_idx, row in enumerate(rows):
        assert row.unique().numel() == row.numel(), f"{label}: duplicate indices in row {row_idx}"


def test_blockwise_topk_random() -> None:
    x = torch.rand(17, 257, device=DEVICE, dtype=DTYPE)
    expected_values, _expected_indices = x.topk(16, dim=-1, sorted=False)
    values, indices = topk_blockwise(x, 16, block_n=64)
    _assert_value_multiset_equal("blockwise random", values, expected_values)
    _assert_indices_gather_values("blockwise random", x, values, indices)


def test_blockwise_topk_ties() -> None:
    x = torch.ones(5, 128, device=DEVICE, dtype=DTYPE)
    values, indices = topk_blockwise(x, 12, block_n=32)
    assert torch.equal(values, torch.ones_like(values))
    _assert_indices_gather_values("blockwise ties", x, values, indices)


def test_blockwise_topk_3d() -> None:
    x = torch.rand(3, 7, 256, device=DEVICE, dtype=DTYPE)
    expected_values, _expected_indices = x.topk(32, dim=-1, sorted=False)
    values, indices = topk_blockwise(x, 32, block_n=64)
    assert values.shape == (3, 7, 32)
    assert indices.shape == (3, 7, 32)
    _assert_value_multiset_equal("blockwise 3d", values, expected_values)
    _assert_indices_gather_values("blockwise 3d", x, values, indices)


def test_blockwise_fused_linear_relu_topk() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(128, 32, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(128, device=DEVICE, dtype=DTYPE)
    pre_acts = torch.relu(F.linear(x, weight, bias))
    expected_values, _expected_indices = pre_acts.topk(16, dim=-1, sorted=False)
    values, indices = linear_relu_topk_blockwise(x, weight, bias, 16, block_n=32)
    _assert_value_multiset_equal("fused blockwise", values, expected_values)
    _assert_indices_gather_values("fused blockwise", pre_acts, values, indices)


def main() -> None:
    test_blockwise_topk_random()
    print("  [PASS] blockwise random")
    test_blockwise_topk_ties()
    print("  [PASS] blockwise ties")
    test_blockwise_topk_3d()
    print("  [PASS] blockwise 3d")
    test_blockwise_fused_linear_relu_topk()
    print("  [PASS] blockwise fused linear+relu+topk")


if __name__ == "__main__":
    main()
