import torch
from contextlib import contextmanager

from typing import Optional


class Activations:
    def __init__(self):
        self.tensor: Optional[torch.Tensor] = None


def _is_site_model(model) -> bool:
    """True for adapter-managed models (model.hf_adapters.HFSiteModel), which
    own their site hooks: their SAE sites are not TuringLLM's three
    block-tuple slots (Pythia's parallel residual, GPT-2's resid_mid)."""
    return hasattr(model, "patch_sites") and hasattr(model, "capture_sites")


@contextmanager
def capture_activations(model, callback=None, capture=True):
    """
    Captures activations in shape: [B, L, K, T, N]
    B=Batch, L=Layer, K=Kind (0=attn, 1=mlp, 2=resid), T=Token, N=Neuron

    Adapter-managed models deliver one tensor per kind in their own kind
    order (the bank's ``kinds``), each in SAE input space.
    """
    if _is_site_model(model):
        with model.capture_sites(callback=callback, capture=capture) as out:
            yield out
        return
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
    """Replace one site's activation with ``value``. kind: attn, mlp, resid.

    "resid" is the block OUTPUT, the same site multi_patch and
    capture_activations call "resid". (Until 2026-10-09 it was a pre-hook on
    norm_2, which replaced only the MLP's input while the block kept its own
    residual — the DAN-68 bug class. It had no callers.)"""
    if _is_site_model(model):
        with model.patch_sites(
                lambda l, k, x: value if (l, k) == (layer_idx, kind) else None):
            yield
        return
    block = model.transformer.h[layer_idx]
    conf = {
        "attn": (block.attn.register_forward_hook, lambda m, i, o: value),
        "mlp": (block.mlp.register_forward_hook, lambda m, i, o: value),
        "resid": (block.register_forward_hook, lambda m, i, o: (o[0], o[1], value))
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

    Adapter-managed models call ``transform`` at each of their own sites in
    forward order, with ``x`` in SAE input space (model/hf_adapters.py).
    """
    if _is_site_model(model):
        with model.patch_sites(transform):
            yield
        return

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
    if _is_site_model(model):
        raise NotImplementedError("stop_grad_at: not implemented for adapter-managed models")
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
    if _is_site_model(model):
        raise NotImplementedError("multi_stop_grad: not implemented for adapter-managed models")
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

