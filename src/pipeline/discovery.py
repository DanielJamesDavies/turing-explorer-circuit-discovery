from typing import Any, Dict, List, Optional

import torch

from circuit.discovery_window import run_discovery_window
from model.inference import Inference
from .runtime import get_runtime
from sae.bank import SAEBank


def prepare_discovery_resources() -> None:
    """Ensure resources needed by discovery are initialized."""
    runtime = get_runtime()
    if runtime.model is None or runtime.bank is None:
        print("Re-initializing model and SAE bank for discovery...")
        runtime.model = Inference(device=runtime.device, compile=runtime.compile)
        runtime.bank = SAEBank(devices=runtime.devices, load_decoders=runtime.fast, compile=runtime.compile)

    assert runtime.model is not None
    assert runtime.bank is not None
    assert runtime.loader is not None


def run_discovery(
    candidates: Optional[List[Dict[str, Any]]] = None,
    candidates_path: str = "outputs/candidates.pt",
) -> None:
    runtime = get_runtime()
    print("--- Discovery Window: Growing Faithful Circuits ---")
    prepare_discovery_resources()

    if candidates is not None:
        torch.save(candidates, candidates_path)

    run_discovery_window(runtime.model, runtime.bank, runtime.loader, candidates_path=candidates_path)
    print("  ✓ discovery window complete")
    print("")
