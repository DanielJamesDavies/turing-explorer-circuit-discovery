from typing import Dict, Tuple, Union

import torch

from sae.async_encode import PendingEncode, encode_layer_async
from sae.bank import SAEBank
from .component_index import component_idx

Latents = Tuple[torch.Tensor, torch.Tensor]
EncodedLayer = Dict[int, Latents]


def encode_layer_components(
    bank: SAEBank,
    layer_idx: int,
    activations: Tuple[torch.Tensor, ...],
    *,
    primary_device: torch.device,
    multi_gpu: bool,
) -> Union[PendingEncode, EncodedLayer]:
    """
    Encode all component kinds for one layer.

    - multi_gpu=True: returns a PendingEncode (async stream work in flight).
    - multi_gpu=False: returns eager encoded results.
    """
    if multi_gpu:
        buffered = tuple(a.detach() for a in activations)
        return encode_layer_async(bank, layer_idx, buffered, primary_device)

    with torch.no_grad():
        results: EncodedLayer = {}
        n_kinds = len(bank.kinds)
        if bank.parallel_kinds:
            latents_list = bank.encode_layer_kinds_parallel(activations, layer_idx)
            for kind_idx, latents in enumerate(latents_list):
                results[component_idx(layer_idx, kind_idx, n_kinds)] = latents
            return results

        for kind_idx, kind in enumerate(bank.kinds):
            latents = bank.encode(activations[kind_idx], kind, layer_idx)
            results[component_idx(layer_idx, kind_idx, n_kinds)] = latents
        return results
