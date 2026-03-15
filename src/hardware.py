import torch
from typing import List
from config import config


def detect_devices() -> List[torch.device]:
    """
    Auto-detects available CUDA devices based on config.hardware.multi_gpu.
    - multi_gpu=true + 2+ GPUs  -> all available GPUs
    - multi_gpu=true + 1 GPU    -> single GPU (graceful fallback)
    - multi_gpu=false            -> primary GPU only
    - no CUDA                    -> CPU
    """
    n_gpus = torch.cuda.device_count()

    if n_gpus == 0:
        return [torch.device("cpu")]

    multi_gpu = bool(config.hardware.multi_gpu)
    if multi_gpu and n_gpus >= 2:
        return [torch.device(f"cuda:{i}") for i in range(n_gpus)]

    return [torch.device("cuda:0")]


def get_primary_device() -> torch.device:
    return detect_devices()[0]


def is_fast_memory() -> bool:
    return (config.hardware.memory or "efficient") == "fast"


def should_compile() -> bool:
    return bool(config.hardware.compile)


def is_multi_gpu() -> bool:
    return len(detect_devices()) > 1
