import torch
from contextlib import contextmanager

from typing import Optional


class Activations:
    def __init__(self):
        self.tensor: Optional[torch.Tensor] = None


@contextmanager
def capture_activations(model, callback=None, capture=True):
    """
    Captures activations in shape: [B, L, K, T, N]
    B=Batch, L=Layer, K=Kind (0=attn, 1=mlp, 2=resid), T=Token, N=Neuron
    """
    storage = []
    
    def make_hook(layer_idx):
        def hook_fn(_m, _i, o):
            # o is (attn, mlp, resid) from Block.forward
            if callback is not None:
                callback(layer_idx, o)
            if capture:
                storage.append([x.detach() for x in o])
        return hook_fn

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(model.transformer.h)]
    out = Activations()
    
    try:
        yield out
    finally:
        for h in handles: h.remove()
        if storage and capture:
            # stacked: [L, K, B, T, N]
            stacked = torch.stack([torch.stack(x) for x in storage])
            # permute to [B, L, K, T, N]
            out.tensor = stacked.permute(2, 0, 1, 3, 4)


@contextmanager
def patch(model, layer_idx: int, kind: str, value: torch.Tensor):
    """kind: attn, mlp, resid"""
    block = model.transformer.h[layer_idx]
    conf = {
        "attn": (block.attn.register_forward_hook, lambda m, i, o: value),
        "mlp": (block.mlp.register_forward_hook, lambda m, i, o: value),
        "resid": (block.norm_2.register_forward_pre_hook, lambda m, i: (value,))
    }
    register, hook = conf[kind]
    handle = register(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def multi_patch(model, transform):
    """
    transform: (layer_idx, kind, tensor) -> tensor (or None to skip)
    kind: attn, mlp, resid
    """
    def wrap(l, k, x):
        out = transform(l, k, x)
        return out if out is not None else x

    hs = []
    for l, b in enumerate(model.transformer.h):
        hs.append(b.attn.register_forward_hook(lambda m, i, o, l=l: wrap(l, "attn", o)))
        hs.append(b.mlp.register_forward_hook(lambda m, i, o, l=l: wrap(l, "mlp", o)))
        hs.append(b.register_forward_hook(lambda m, i, o, l=l: (o[0], o[1], wrap(l, "resid", o[2]))))
    
    try:
        yield
    finally:
        for h in hs: h.remove()


@contextmanager
def stop_grad_at(model, layer_idx: int, kind: str):
    """
    Zeros gradients flowing back through a specific submodule during backward.
    Used for intermediate stop-grads in edge attribution.
    """
    block = model.transformer.h[layer_idx]
    target = {
        "attn": block.attn,
        "mlp": block.mlp,
        "resid": block
    }[kind]

    def backward_hook(_module, grad_input, _grad_output):
        return tuple(torch.zeros_like(g) if g is not None else None for g in grad_input)

    handle = target.register_full_backward_hook(backward_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def multi_stop_grad(model, stop_grads: list[tuple[int, str]]):
    """
    Context manager to stop gradients at multiple locations simultaneously.
    stop_grads: list of (layer_idx, kind)
    """
    handles = []
    for layer_idx, kind in stop_grads:
        block = model.transformer.h[layer_idx]
        target = {
            "attn": block.attn,
            "mlp": block.mlp,
            "resid": block
        }[kind]

        def backward_hook(_module, grad_input, _grad_output):
            return tuple(torch.zeros_like(g) if g is not None else None for g in grad_input)

        handles.append(target.register_full_backward_hook(backward_hook))

    try:
        yield
    finally:
        for h in handles:
            h.remove()

