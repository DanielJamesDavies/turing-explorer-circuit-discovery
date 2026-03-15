from dataclasses import dataclass
from typing import List, Optional, cast

import torch

from config import config
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from sae.bank import SAEBank
from store.seq_repr import SeqRepr


@dataclass
class PipelineRuntime:
    fast: bool
    compile: bool
    devices: List[torch.device]
    device: torch.device
    cpu_device: torch.device
    multi_gpu: bool
    mid_ctx_warmup: int
    loader: Optional[DataLoader] = None
    model: Optional[Inference] = None
    bank: Optional[SAEBank] = None
    seq_repr: Optional[SeqRepr] = None


_runtime: Optional[PipelineRuntime] = None


def set_runtime(runtime: PipelineRuntime) -> None:
    global _runtime
    _runtime = runtime


def get_runtime() -> PipelineRuntime:
    if _runtime is None:
        raise RuntimeError("Pipeline runtime is not initialized. Call set_runtime() first.")
    return _runtime


def clear_runtime() -> None:
    global _runtime
    _runtime = None


def initialize_runtime() -> PipelineRuntime:
    runtime = build_runtime()
    set_runtime(runtime)
    return runtime


def build_runtime() -> PipelineRuntime:
    devices = detect_devices()
    return PipelineRuntime(
        fast=is_fast_memory(),
        compile=should_compile(),
        devices=devices,
        device=devices[0],
        cpu_device=torch.device("cpu"),
        multi_gpu=len(devices) > 1,
        mid_ctx_warmup=cast(int, config.latents.mid_ctx.warmup_batches or 100),
    )


def initialize_resources() -> None:
    runtime = get_runtime()
    print("Initializing DataLoader...")
    runtime.loader = DataLoader(device=runtime.device, pin_memory=runtime.fast)

    n_seqs = sum(runtime.loader._shard_sequence_counts)
    runtime.seq_repr = SeqRepr(n_seqs=n_seqs)
    print(
        f"Sequence repr store: {n_seqs:,} sequences × {runtime.seq_repr.repr_dim} dims "
        f"(mode={runtime.seq_repr.repr_mode})"
    )

    print("Initializing Model...")
    runtime.model = Inference(device=runtime.device, compile=runtime.compile)

    print(f"Initializing SAE Bank ({len(runtime.devices)} device{'s' if len(runtime.devices) > 1 else ''})...")
    runtime.bank = SAEBank(devices=runtime.devices, load_decoders=runtime.fast, compile=runtime.compile)
    print("")
