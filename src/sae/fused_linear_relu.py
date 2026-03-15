"""Fused linear + ReLU via cublasLt RELU_BIAS epilogue.

On CUDA with BF16 tensors, computes relu(x @ W.T + b) inside a single GEMM
pass by applying bias-add and ReLU in the cublasLt epilogue, eliminating the
separate elementwise kernel that torch.compile would otherwise generate.

Falls back to two-pass PyTorch (F.relu + F.linear) on CPU or other dtypes.

The extension is compiled alongside the other native extensions via setup.py:
    cd src/native && python setup.py build_ext --inplace
"""

from __future__ import annotations

import importlib.util
import os
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
        so_path = next(
            os.path.join(native_dir, f)
            for f in os.listdir(native_dir)
            if f.startswith("linear_relu_ext") and f.endswith(".so")
        )
        spec = importlib.util.spec_from_file_location("linear_relu_ext", so_path)
        mod  = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                    # type: ignore[union-attr]
        _ext       = mod
        _available = True
        print("[fused_linear_relu] cublasLt RELU_BIAS epilogue compiled and ready.")
    except Exception as exc:
        _available = False
        print(f"[fused_linear_relu] cublasLt extension unavailable, using PyTorch fallback. Reason: {exc}")
    return _available  # type: ignore[return-value]


def is_available() -> bool:
    """Returns True if the cublasLt fused kernel is importable."""
    return _load()


def linear_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Computes relu(x @ weight.T + bias).

    Uses cublasLt RELU_BIAS epilogue when *x* is BF16 on CUDA so bias-add
    and ReLU happen inside the GEMM without a second kernel pass.
    Falls back to ``F.relu(F.linear(x, weight, bias))`` otherwise.
    """
    if (
        _load()
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and bias.dtype == torch.bfloat16
    ):
        return _ext.linear_relu_bf16(x, weight, bias)  # type: ignore[union-attr]
    return F.relu(F.linear(x, weight, bias))
