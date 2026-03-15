import torch
from typing import Dict, Optional, Tuple

from .bank import SAEBank
from pipeline.component_index import component_idx


class PendingEncode:
    """
    Holds SAE encode results launched on a CUDA stream.
    Must be synchronized before results are accessed.
    """

    def __init__(
        self,
        stream: Optional[torch.cuda.Stream],
        comp_results: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        primary_device: torch.device,
    ):
        self.stream = stream
        self.comp_results = comp_results
        self.primary_device = primary_device

    def synchronize(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        if self.stream is not None:
            self.stream.synchronize()

        results = {}
        for comp_idx, latents in self.comp_results.items():
            if latents[0].device != self.primary_device:
                results[comp_idx] = tuple(t.to(self.primary_device) for t in latents)
            else:
                results[comp_idx] = latents
        return results


def encode_layer_async(
    bank: SAEBank,
    layer_idx: int,
    activations: Tuple[torch.Tensor, ...],
    primary_device: torch.device,
) -> PendingEncode:
    """
    Encode all 3 SAE components for a layer on the target device's stream.
    The encode runs asynchronously; call .synchronize() on the returned
    PendingEncode before using the results.
    """
    target_device = bank.layer_device_map[layer_idx]
    stream = torch.cuda.Stream(device=target_device)

    comp_results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.cuda.stream(stream):
        n_kinds = len(bank.kinds)
        for kind_idx, kind in enumerate(bank.kinds):
            comp_idx = component_idx(layer_idx, kind_idx, n_kinds)
            latents = bank.encode(activations[kind_idx], kind, layer_idx)
            comp_results[comp_idx] = latents

    return PendingEncode(stream, comp_results, primary_device)
