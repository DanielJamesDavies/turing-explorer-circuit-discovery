"""Experimental exact blockwise Linear+ReLU+TopK SAE encoder.

This module preserves the standard SAE encode contract while avoiding full
dense activation materialization in the Python prototype. It computes exact
local top-K candidates for each SAE feature block, then merges those candidates
to the exact global top-K.
"""

from __future__ import annotations

import importlib.util
import os

import torch
import torch.nn.functional as F

_ext = None
_available: bool | None = None


def _block_size(default: int = 4096) -> int:
    raw = os.environ.get("TURINGLLM_FUSED_EXACT_TOPK_BLOCK_N")
    if raw is None:
        return default
    value = int(raw)
    if value < 1:
        raise ValueError("TURINGLLM_FUSED_EXACT_TOPK_BLOCK_N must be >= 1")
    return value


def _load_native() -> bool:
    global _ext, _available
    if _available is not None:
        return _available

    native_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "native"))
    try:
        candidates = [
            os.path.join(native_dir, name)
            for name in os.listdir(native_dir)
            if name.startswith("linear_relu_topk_exact_ext") and name.endswith(".so")
        ]
        if not candidates:
            raise FileNotFoundError("linear_relu_topk_exact_ext shared object not found")
        spec = importlib.util.spec_from_file_location("linear_relu_topk_exact_ext", candidates[0])
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _ext = mod
        _available = True
    except Exception:
        _available = False
    return _available


def native_is_available() -> bool:
    """Return whether the optional native fused exact extension can be loaded."""

    return _load_native()


def linear_relu_topk_exact(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    *,
    block_n: int | None = None,
    use_native: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact top-k of ``relu(linear(x))`` using block-local candidates."""

    block_n = int(block_n or _block_size())
    if use_native is None:
        use_native = os.environ.get("TURINGLLM_FUSED_EXACT_TOPK_USE_NATIVE", "0") == "1"
    if use_native and _load_native():
        return _ext.linear_relu_topk_exact(x, weight, bias, int(k), block_n)  # type: ignore[union-attr]

    orig_shape = x.shape
    d_model = int(orig_shape[-1])
    d_sae = int(weight.shape[0])
    if k > d_sae:
        raise ValueError(f"k must be <= d_sae, got k={k}, d_sae={d_sae}")

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
