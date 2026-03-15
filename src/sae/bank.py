from typing import Dict, List, cast, Optional
import os
import torch
from contextlib import nullcontext
from torch.profiler import ProfilerActivity, profile, record_function
from config import config
from .topk_sae import SAE, SAEConfig
from model.turingllm import TuringLLMConfig

class SAEBank:

    def __init__(
        self,
        device: Optional[torch.device] = None,
        devices: Optional[List[torch.device]] = None,
        load_decoders: bool = True,
        compile: bool = False,
    ):
        llm_config = TuringLLMConfig()
        sae_config = SAEConfig()

        self.n_layer = llm_config.n_layer
        self.d_model = llm_config.n_embd
        self.d_sae = sae_config.d_sae
        self.k = sae_config.k
        self.load_decoders = load_decoders
        self.compile = compile
        self.kinds = ["attn", "mlp", "resid"]

        if devices is not None and len(devices) >= 1:
            self.devices = devices
        elif device is not None:
            self.devices = [device]
        else:
            self.devices = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]

        self.device = self.devices[0]
        self.is_multi_gpu = len(self.devices) > 1

        self.layer_device_map: Dict[int, torch.device] = {}
        if self.is_multi_gpu:
            mid = self.n_layer // 2
            for layer in range(self.n_layer):
                self.layer_device_map[layer] = self.devices[0] if layer < mid else self.devices[1]
        else:
            for layer in range(self.n_layer):
                self.layer_device_map[layer] = self.devices[0]

        self.parallel_kinds: bool = bool(config.hardware.parallel_kinds)
        self._kind_streams: Optional[List[torch.cuda.Stream]] = None
        if self.parallel_kinds and self.device.type == "cuda":
            self._kind_streams = [torch.cuda.Stream(device=self.device) for _ in self.kinds]

        self.saes: Dict[str, List[Optional[SAE]]] = {kind: [None] * self.n_layer for kind in self.kinds}
        self.load_saes()

    def load_saes(self):
        for kind in self.kinds:
            for layer in range(self.n_layer):
                target_device = self.layer_device_map[layer]
                sae = SAE(self.d_model, self.d_sae, self.k, device=target_device, compile=self.compile)
                path = os.path.join(cast(str, config.weights.sae_path), f"sae-{kind}/sae_{kind}_layer_{layer}.pth")
                sae.load(path)
                if self.load_decoders:
                    sae.move_decoder_to_vram()
                else:
                    sae.remove_decoder_from_vram()
                self.saes[kind][layer] = sae

    def _autocast_ctx(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=autocast_dtype)

    def encode(self, x: torch.Tensor, kind: str, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        sae = self.saes[kind][layer]
        if sae is None:
            raise ValueError(f"SAE for {kind} layer {layer} is not loaded")

        target_device = self.layer_device_map[layer]
        if x.device != target_device:
            x = x.to(target_device, non_blocking=True)

        # Only use inference_mode if gradients are globally disabled.
        # This allows SAEGraphInstrument to perform differentiable passes.
        ctx = torch.no_grad() if not torch.is_grad_enabled() else nullcontext()
        with ctx, self._autocast_ctx(target_device):
            return sae.encode(x)

    def encode_layer_kinds_parallel(
        self,
        activations: tuple[torch.Tensor, ...],
        layer: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode all kinds for a layer using per-kind CUDA streams.

        Dispatches each kind's encode() onto its own pre-created CUDA stream so
        the GPU scheduler can overlap independent GEMM + top-k kernels.  After
        all dispatches the streams are synced back to the default stream before
        returning, so callers see fully completed results.

        Memory usage is identical to the sequential path — no extra intermediate
        tensors are created. The cublasLt fused kernel is preserved.

        Falls back silently to sequential if streams were not created (CPU device
        or parallel_kinds=False).

        Args:
            activations: Tuple of len(self.kinds) tensors, each [B, T, d_model].
            layer:       Layer index.

        Returns:
            List of (top_acts, top_indices) per kind, each [B, T, k].
        """
        if self._kind_streams is None:
            return [self.encode(activations[i], kind, layer) for i, kind in enumerate(self.kinds)]

        results: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.kinds)

        for kind_idx, kind in enumerate(self.kinds):
            with torch.cuda.stream(self._kind_streams[kind_idx]):
                results[kind_idx] = self.encode(activations[kind_idx], kind, layer)

        default_stream = torch.cuda.current_stream(self.device)
        for stream in self._kind_streams:
            default_stream.wait_stream(stream)

        return results  # type: ignore[return-value]

    def decode(self, latents: torch.Tensor, kind: str, layer: int) -> torch.Tensor:
        sae = self.saes[kind][layer]
        if sae is None:
            raise ValueError(f"SAE for {kind} layer {layer} is not loaded")

        target_device = self.layer_device_map[layer]
        if latents.device != target_device:
            latents = latents.to(target_device, non_blocking=True)

        ctx = torch.no_grad() if not torch.is_grad_enabled() else nullcontext()
        with ctx, self._autocast_ctx(target_device):
            return sae.decode(latents)

    def profile_encode(
        self,
        x: torch.Tensor,
        kind: str,
        layer: int,
        output_dir: str = "profile_trace",
        warmup: int = 5,
        active: int = 10,
    ) -> str:
        """
        Profiles sae.encode() for a single (kind, layer) using torch.profiler.

        Runs `warmup` calls to prime CUDA/autotune caches, then profiles
        `active` calls and prints a summary table sorted by CUDA time.
        Saves a Chrome trace to `output_dir/sae_encode_{kind}_layer{layer}.json`
        which can be opened in chrome://tracing or Perfetto.

        Args:
            x:          Sample input tensor of shape (..., d_model).
            kind:       SAE kind — "attn", "mlp", or "resid".
            layer:      Layer index.
            output_dir: Directory to write the trace file.
            warmup:     Number of warmup iterations (not profiled).
            active:     Number of profiled iterations.

        Returns:
            Path to the saved trace file.
        """
        sae = self.saes[kind][layer]
        if sae is None:
            raise ValueError(f"SAE for {kind} layer {layer} is not loaded")

        target_device = self.layer_device_map[layer]
        if x.device != target_device:
            x = x.to(target_device, non_blocking=True)

        for _ in range(warmup):
            with torch.inference_mode(), self._autocast_ctx(target_device):
                sae.encode(x)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            for _ in range(active):
                with record_function("sae_encode"), \
                     torch.inference_mode(), \
                     self._autocast_ctx(target_device):
                    sae.encode(x)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)

        os.makedirs(output_dir, exist_ok=True)
        trace_path = os.path.join(output_dir, f"sae_encode_{kind}_layer{layer}.json")
        prof.export_chrome_trace(trace_path)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        print(f"\nTrace saved → {trace_path}")
        return trace_path
