"""Device selection and partition policy for negative-context retrieval."""

from __future__ import annotations

from typing import Sequence, cast

import torch

from config import config


def _ann_device() -> torch.device:
    cfg = cast(str, config.hardware.ann_device or "auto")
    if cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("hardware.ann_device = 'gpu' but CUDA is not available.")
        return torch.device("cuda")
    if cfg == "cpu":
        return torch.device("cpu")
    if cfg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"hardware.ann_device = {cfg!r} but CUDA is not available.")
        device = torch.device(cfg)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"hardware.ann_device = {cfg!r} is outside visible CUDA range "
                f"0..{torch.cuda.device_count() - 1}."
            )
        return device
    raise ValueError(
        "hardware.ann_device must be one of 'auto', 'cpu', 'gpu', 'cuda', or 'cuda:N'."
    )


def parse_neg_ctx_devices(
    configured_devices: Sequence[int | str],
    cuda_count: int | None = None,
) -> list[torch.device]:
    if cuda_count is None:
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if not configured_devices:
        return [torch.device(f"cuda:{idx}") for idx in range(cuda_count)]

    devices: list[torch.device] = []
    for raw in configured_devices:
        if isinstance(raw, int):
            devices.append(torch.device(f"cuda:{raw}"))
            continue
        text = str(raw)
        if text.isdigit():
            devices.append(torch.device(f"cuda:{text}"))
        elif text == "cuda":
            devices.append(torch.device("cuda:0"))
        elif text.startswith("cuda:"):
            devices.append(torch.device(text))
        else:
            raise ValueError(f"Invalid neg_ctx device {raw!r}; use CUDA ids like 0 or 'cuda:0'.")

    seen: set[str] = set()
    deduped: list[torch.device] = []
    for device in devices:
        key = str(device)
        if key not in seen:
            seen.add(key)
            deduped.append(device)
    return deduped


def partition_components(n_components: int, devices: Sequence[torch.device]) -> dict[str, list[int]]:
    if not devices:
        raise ValueError("At least one device is required for component partitioning.")
    result = {str(device): [] for device in devices}
    for comp_idx in range(n_components):
        result[str(devices[comp_idx % len(devices)])].append(comp_idx)
    return result


def _validate_cuda_devices(devices: Sequence[torch.device]) -> None:
    if not devices:
        raise RuntimeError(
            "latents.neg_ctx.backend='multi_gpu_exact' requires at least one CUDA device."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "latents.neg_ctx.backend='multi_gpu_exact' requires CUDA, but CUDA is not available."
        )
    cuda_count = torch.cuda.device_count()
    for device in devices:
        if device.type != "cuda":
            raise RuntimeError(f"multi_gpu_exact only supports CUDA devices, got {device}.")
        idx = device.index if device.index is not None else 0
        if idx < 0 or idx >= cuda_count:
            raise RuntimeError(f"Configured CUDA device {device} is outside visible range 0..{cuda_count - 1}.")


__all__ = [
    "_ann_device",
    "_validate_cuda_devices",
    "parse_neg_ctx_devices",
    "partition_components",
]
