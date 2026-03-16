import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from .fused_linear_relu import is_available as _cublaslt_available, linear_relu as _linear_relu
from .triton_topk import (
    is_available as _triton_topk_available,
    topk_nonneg_bf16 as _topk_nonneg_bf16,
)

# ── Top-k backend selection ───────────────────────────────────────────────────
# Set env var TURINGLLM_TOPK_IMPL=pytorch to force the PyTorch fallback.
# Default: use the Triton radix-select kernel when available.

_USE_TRITON_TOPK: bool = (
    os.environ.get("TURINGLLM_TOPK_IMPL", "triton").strip().lower() != "pytorch"
)

_triton_topk_warmed_up: bool = False


def set_topk_backend(backend: str) -> None:
    """Switch the top-k implementation at runtime.

    Args:
        backend: ``'triton'`` (default) or ``'pytorch'``.

    Example::

        from sae.topk_sae import set_topk_backend
        set_topk_backend("pytorch")   # benchmark PyTorch
        set_topk_backend("triton")    # benchmark Triton
    """
    global _USE_TRITON_TOPK
    if backend not in ("triton", "pytorch"):
        raise ValueError(f"backend must be 'triton' or 'pytorch', got {backend!r}")
    _USE_TRITON_TOPK = backend == "triton"
    print(f"[topk_sae] top-k backend: {backend}")


def get_topk_backend() -> str:
    """Return the name of the currently active top-k backend."""
    if _USE_TRITON_TOPK and _triton_topk_available():
        return "triton"
    return "pytorch"


def _warmup_triton_topk(d_sae: int, k: int, device: torch.device) -> None:
    """Run a tiny forward pass to trigger Triton JIT compilation and autotuning.

    Called once from SAE.load() so that the first real encode() call has
    no cold-start latency.  Subsequent calls are no-ops (guarded by a flag).
    """
    global _triton_topk_warmed_up
    if _triton_topk_warmed_up:
        return
    warmup = torch.zeros(256, d_sae, dtype=torch.bfloat16, device=device)
    with torch.inference_mode():
        _topk_nonneg_bf16(warmup, k)
    del warmup
    _triton_topk_warmed_up = True
    print(f"[topk_sae] Triton radix-select top-k compiled and autotuned.")


@dataclass
class SAEConfig:
    d_sae: int = 40960
    k: int = 128


class SAE(nn.Module):
    """
    A simplified Top-K Sparse Autoencoder for inference.
    """
    def __init__(self, d_model: int, d_sae: int, k: int = 128, device: Optional[torch.device] = None, compile: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder_device = self.device
        self._compile = compile
        # Cached effective bias: b_enc - W_enc @ b_dec  (set in load())
        self._bias_eff: Optional[torch.Tensor] = None
        self.to(self.device)

    def _get_bias_eff(self) -> torch.Tensor:
        """Lazily computes and caches b_enc - W_enc @ b_dec.

        The effective bias absorbs the ``x - decoder_bias`` subtraction so that
        encode reduces to a single fused call: relu(x @ W_enc.T + b_eff).
        Invalidated automatically when device or dtype changes.
        """
        enc_bias = self.encoder.bias
        if (
            self._bias_eff is None
            or self._bias_eff.device != enc_bias.device
            or self._bias_eff.dtype != enc_bias.dtype
        ):
            with torch.no_grad():
                self._bias_eff = (
                    enc_bias - self.decoder_bias @ self.encoder.weight.T
                ).contiguous()
        return self._bias_eff

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor x into sparse top-K activations.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            top_acts:    (..., k) — values of the top-k activations after ReLU.
            top_indices: (..., k) — indices of the top-k activations (int64).
        """
        if x.dtype != self.encoder.weight.dtype:
            x = x.to(self.encoder.weight.dtype)

        # Custom fused kernels (_linear_relu, _topk_nonneg_bf16) do not register
        # autograd backward functions.  When gradients are needed (e.g. during
        # circuit attribution), fall back to standard differentiable PyTorch ops
        # so that gradients flow correctly from encoder outputs back to x.
        if torch.is_grad_enabled():
            pre_acts = torch.relu(nn.functional.linear(x, self.encoder.weight, self._get_bias_eff()))
            return pre_acts.topk(self.k, sorted=False, dim=-1)

        pre_acts = _linear_relu(x, self.encoder.weight, self._get_bias_eff())

        if (
            _USE_TRITON_TOPK
            and _triton_topk_available()
            and pre_acts.is_cuda
            and pre_acts.dtype == torch.bfloat16
        ):
            return _topk_nonneg_bf16(pre_acts, self.k)

        return pre_acts.topk(self.k, sorted=False, dim=-1)

    def decode(self, latents, ensure_device: bool = True):
        """
        Decodes the latents back into the original d_model space.
        
        Args:
            latents: Sparse activations of shape (..., d_sae)
        """
        if ensure_device is True and self.decoder_device != latents.device:
            self.move_decoder_to_vram()
            
        return self.decoder(latents) + self.decoder_bias

    def forward(self, x: torch.Tensor):
        """
        Full pass: encode and then decode.
        Reconstructs the sparse encoded_acts tensor for the decoder.
        """
        top_acts, top_indices = self.encode(x)
        encoded_acts = torch.zeros(*x.shape[:-1], self.d_sae, device=x.device, dtype=top_acts.dtype)
        encoded_acts.scatter_(dim=-1, index=top_indices, src=top_acts)
        reconstruction = self.decode(encoded_acts)
        return reconstruction, encoded_acts

    def load(self, path: str) -> None:
        """Loads the SAE weights from a file."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=False)

        target_dtype = None
        if self.device.type == "cuda":
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if target_dtype is not None:
            self.to(device=self.device, dtype=target_dtype)
        else:
            self.to(self.device)

        self.decoder_device = self.device

        # Pre-compute the effective bias so the first encode call is fast.
        # (Also triggers JIT compilation of the cublasLt extension if needed.)
        self._get_bias_eff()

        # Trigger Triton JIT compilation + autotuning for the top-k kernel.
        # This runs once globally (guarded by _triton_topk_warmed_up), so only
        # the first SAE loaded pays the compilation cost.
        if _USE_TRITON_TOPK and _triton_topk_available() and self.device.type == "cuda":
            _warmup_triton_topk(self.d_sae, self.k, self.device)

        # torch.compile is skipped when the cublasLt fused kernel is active:
        # the C extension causes a graph break so compilation provides no gain.
        # It is kept as a fallback for CPU / non-BF16 paths.
        if self._compile and self.device.type == "cuda" and not _cublaslt_available():
            self.encode = torch.compile(self.encode, mode="default")

    def move_decoder_to_vram(self):
        """Moves the decoder weights back to the same device as the encoder (VRAM)."""
        target_device = self.encoder.weight.device
        self.decoder.to(target_device)
        # If called inside torch.inference_mode() the .to() above creates inference
        # tensors, which would later raise when autograd tries to save them for
        # backward.  Clone to produce normal autograd-compatible parameters.
        for param in self.decoder.parameters():
            if param.is_inference():
                param.data = param.data.clone()
        self.decoder_device = target_device

    def remove_decoder_from_vram(self, empty_cache: bool = True):
        """Moves the decoder weights to CPU to save VRAM when only encoding is needed."""
        self.decoder.to('cpu')
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.decoder_device = 'cpu'


