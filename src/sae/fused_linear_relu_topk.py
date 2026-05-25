"""Experimental blockwise Linear+ReLU+TopK SAE encoder.

The production fast path uses cublasLt to materialize the full dense SAE
activation matrix and then runs top-k. This prototype computes SAE feature
columns in blocks, keeps local top-k candidates per block, and merges candidates
to the final top-k. It is intended for benchmarking the fused design direction,
not as a guaranteed faster replacement for cublasLt.
"""

from __future__ import annotations

import os
import importlib.util

import torch
import torch.nn.functional as F

_ext = None
_available: bool | None = None


def _load() -> bool:
    global _ext, _available
    if _available is not None:
        return _available
    native_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "native"))
    try:
        candidates = [
            os.path.join(native_dir, f)
            for f in os.listdir(native_dir)
            if f.startswith("linear_relu_topk_ext") and f.endswith(".so")
        ]
        if not candidates:
            raise FileNotFoundError("linear_relu_topk_ext shared object not found")
        so_path = candidates[0]
        spec = importlib.util.spec_from_file_location("linear_relu_topk_ext", so_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _ext = mod
        _available = True
        print("[fused_linear_relu_topk] native blockwise prototype loaded.")
    except Exception as exc:
        _available = False
        print(f"[fused_linear_relu_topk] native prototype unavailable, using PyTorch fallback. Reason: {exc}")
    return _available


def is_available() -> bool:
    """Returns True if the native prototype extension is importable."""
    return _load()


def _block_size(default: int = 4096) -> int:
    raw = os.environ.get("TURINGLLM_FUSED_TOPK_BLOCK_N")
    if raw is None:
        return default
    value = int(raw)
    if value < 1:
        raise ValueError("TURINGLLM_FUSED_TOPK_BLOCK_N must be >= 1")
    return value


def linear_relu_topk_blockwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    *,
    block_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``relu(linear(x))`` top-k without storing full dense activations."""

    block_n = int(block_n or _block_size())
    if _load():
        return _ext.linear_relu_topk_blockwise(x, weight, bias, int(k), block_n)  # type: ignore[union-attr]

    orig_shape = x.shape
    d_model = int(orig_shape[-1])
    d_sae = int(weight.shape[0])
    rows = x.reshape(-1, d_model)

    value_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    for start in range(0, d_sae, block_n):
        end = min(start + block_n, d_sae)
        local_k = min(k, end - start)
        acts = F.linear(rows, weight[start:end], bias[start:end])
        acts = torch.relu(acts)
        local_values, local_indices = acts.topk(local_k, dim=-1, sorted=False)
        value_chunks.append(local_values)
        index_chunks.append(local_indices + start)

    candidates = torch.cat(value_chunks, dim=-1)
    candidate_indices = torch.cat(index_chunks, dim=-1)
    values, positions = candidates.topk(k, dim=-1, sorted=False)
    indices = candidate_indices.gather(-1, positions)

    out_shape = orig_shape[:-1] + (k,)
    return values.reshape(out_shape), indices.reshape(out_shape).long()
