"""HuggingFace model adapters: run the circuit engine on public models (DAN-34).

The engine reaches a model through three things only (052's coupling audit):

  * ``inference.model(tokens, return_all_logits=...) -> (logits, None)``;
  * ``hooks.multi_patch(model, transform)``: ``transform(layer, kind, x)`` is
    called once per SAE site in forward order and may return a replacement;
  * ``hooks.capture_activations(model, callback)``: ``callback(layer, acts)``
    once per layer, ``acts`` one tensor per kind in ``bank.kinds`` order.

``HFSiteModel`` provides all three for an HF causal LM. ``hooks`` dispatches to
its ``patch_sites`` / ``capture_sites`` when the model has them, so every
patcher (LearnedMaskPatcher, CircuitOnlyPatcher, the cf patcher, floors) runs
unchanged. The HF forward itself is untouched: sites are PyTorch hooks on the
HF modules at the SAEs' training hook points, so an unedited pass is the HF
model's own pass, bitwise.

Hook points (where each SAE reads, and where its edit enters the stream):

  Pythia (GPTNeoXLayer, use_parallel_residual=True:
          out = x + attn(ln1(x)) + mlp(ln2(x)), summed as mlp + attn + x)
    attn   GPTNeoXAttention output[0] (after `dense`)   sparsify "layers.L.attention"
    mlp    GPTNeoXMLP output                            sparsify "layers.L.mlp"
    resid  GPTNeoXLayer output[0]                       sparsify "layers.L"
    An attn edit enters the layer's sum, NOT the same-layer mlp's input.

  GPT-2 (GPT2Block: r = x + attn(ln_1(x)); out = r + mlp(ln_2(r)))
    attn       GPT2Attention output[0]                  TL hook_attn_out
    resid_mid  r = attn_out + block input               TL hook_resid_mid
    mlp        GPT2MLP output                           TL hook_mlp_out
    resid      GPT2Block output[0]                      TL hook_resid_post
    resid_mid has no module of its own. It is formed and edited INSIDE the
    attention hook: r = a' + x_in (bitwise the block's own `attn_output +
    residual`), r' = transform(r), and the attention output is returned as
    a' + (r' - r), so the block's residual add yields r' and BOTH the
    residual stream and ln_2 see the edit. (The 053 standalone edited r via a
    pre-hook on ln_2, which changes only the MLP input: the block keeps its
    saved `residual` and returns residual + mlp. That is DAN-68.)
    TransformerLens' centering/folding does not matter here: the SAEs
    layer-normalise their input, which removes the per-token mean anyway.

Layer-norm SAE input (GPT-2): the transform sees x_n = (x - mean)/(std + eps);
an edit x_n -> x_n' is applied as x + (x_n' - x_n) * (std + eps). A transform
that returns its input object unchanged leaves the stream bitwise untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from .families import GPT2_SMALL, PYTHIA_70M, ModelFamily, get_family
from .hooks import Activations
from .inference import Inference

# on_site(layer, kind, x_model) -> replacement for x_model, or None (no edit).
OnSite = Callable[[int, str, torch.Tensor], Optional[torch.Tensor]]

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def layer_norm_in(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """SAELens run_time_activation_ln_in: (x_n, scale) with x_n = (x - mean) /
    scale and scale = std + eps (torch's unbiased std, per token). The same
    arithmetic as 053's gpt2saes.norm_in, op for op."""

    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    scale = xc.std(dim=-1, keepdim=True) + eps
    return xc / scale, scale


def _site_hook(on_site: OnSite, layer: int, kind: str):
    """Forward hook for a module whose output (or output[0]) IS the site."""

    def hook(_m, _inp, out):
        is_tuple = isinstance(out, tuple)
        x = out[0] if is_tuple else out
        y = on_site(layer, kind, x)
        if y is None:
            return None
        return (y,) + tuple(out[1:]) if is_tuple else y
    return hook


class HFSiteModel(nn.Module):
    """An HF causal LM exposed through the engine's model protocol (see the
    module docstring). Subclasses register the architecture's site hooks."""

    def __init__(self, hf_model: nn.Module, family: ModelFamily) -> None:
        super().__init__()
        self.hf = hf_model
        self.family = family
        self.kinds: Tuple[str, ...] = family.kinds
        self.n_layer = family.n_layer
        sae = family.sae
        self.normalize = sae.normalize if sae is not None else None
        self.norm_eps = float(sae.norm_eps) if sae is not None else 0.0
        self.config = hf_model.config

    # ---------------------------------------------------------------- forward
    def forward(self, idx: torch.Tensor, targets=None, input_pos=None,
                return_all_logits: bool = False):
        """TuringLLM's call contract: (logits, None); logits at the last
        position only unless return_all_logits ([B, 1, V] vs [B, T, V])."""

        if targets is not None:
            raise NotImplementedError("HFSiteModel: training loss is not supported")
        out = self.hf(input_ids=idx, use_cache=False,
                      logits_to_keep=0 if return_all_logits else 1)
        return out.logits, None

    # ---------------------------------------------------------------- sites
    def _register(self, on_site: OnSite) -> List[Any]:
        raise NotImplementedError

    def to_site_space(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Model-space stream -> (SAE input, scale). scale is None when the
        family's SAEs read the stream as is."""

        if self.normalize == "layer_norm":
            return layer_norm_in(x, self.norm_eps)
        return x, None

    @contextmanager
    def patch_sites(self, transform: Callable[[int, str, torch.Tensor], Optional[torch.Tensor]]):
        """multi_patch for this model: ``transform`` sees the SAE input at each
        site and returns a replacement (or None / the same object = no edit)."""

        def on_site(layer: int, kind: str, x: torch.Tensor) -> Optional[torch.Tensor]:
            x_s, scale = self.to_site_space(x)
            y = transform(layer, kind, x_s)
            if y is None or y is x_s:
                return None
            if scale is None:
                return y
            return x + (y - x_s) * scale

        handles = self._register(on_site)
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def capture_sites(self, callback=None, capture: bool = True):
        """capture_activations for this model: per layer, one SAE-input tensor
        per kind (``self.kinds`` order) to ``callback(layer, acts)``; with
        ``capture`` the detached stack [B, L, K, T, N] is left on the yielded
        object's ``.tensor``."""

        storage: List[List[torch.Tensor]] = []
        pending: Dict[str, torch.Tensor] = {}
        last_kind = self.kinds[-1]

        def on_site(layer: int, kind: str, x: torch.Tensor) -> None:
            pending[kind] = self.to_site_space(x)[0]
            if kind == last_kind:
                acts = tuple(pending[k] for k in self.kinds)
                pending.clear()
                if callback is not None:
                    callback(layer, acts)
                if capture:
                    storage.append([a.detach() for a in acts])
            return None

        out = Activations()
        handles = self._register(on_site)
        try:
            yield out
        finally:
            for h in handles:
                h.remove()
            if storage and capture:
                stacked = torch.stack([torch.stack(x) for x in storage])   # [L, K, B, T, N]
                out.tensor = stacked.permute(2, 0, 1, 3, 4)


class PythiaSiteModel(HFSiteModel):
    """GPT-NeoX with the parallel residual (Pythia). Sites: attn, mlp, resid."""

    def __init__(self, hf_model: nn.Module, family: ModelFamily) -> None:
        super().__init__(hf_model, family)
        if not bool(getattr(hf_model.config, "use_parallel_residual", False)):
            raise ValueError(
                "PythiaSiteModel assumes use_parallel_residual=True (the family's "
                "same_layer_upstream table encodes it); this checkpoint is sequential")

    def layers(self):
        return self.hf.gpt_neox.layers

    def _register(self, on_site: OnSite) -> List[Any]:
        hs = []
        for l, layer in enumerate(self.layers()):
            hs.append(layer.attention.register_forward_hook(_site_hook(on_site, l, "attn")))
            hs.append(layer.mlp.register_forward_hook(_site_hook(on_site, l, "mlp")))
            hs.append(layer.register_forward_hook(_site_hook(on_site, l, "resid")))
        return hs


class GPT2SiteModel(HFSiteModel):
    """GPT-2. Sites: attn, resid_mid, mlp, resid (resid_mid edited inside the
    attention hook, see the module docstring)."""

    def layers(self):
        return self.hf.transformer.h

    def _register(self, on_site: OnSite) -> List[Any]:
        hs = []
        for l, block in enumerate(self.layers()):
            block_in: Dict[str, torch.Tensor] = {}

            def save_input(_m, args, kwargs, _b=block_in):
                _b["x"] = args[0] if args else kwargs["hidden_states"]

            def attn_hook(_m, _inp, out, l=l, _b=block_in):
                a = out[0]
                a_edit = on_site(l, "attn", a)
                a_new = a if a_edit is None else a_edit
                r = a_new + _b.pop("x")       # == the block's `attn_output + residual`
                r_edit = on_site(l, "resid_mid", r)
                if r_edit is not None:
                    a_new = a_new + (r_edit - r)
                if a_edit is None and r_edit is None:
                    return None
                return (a_new,) + tuple(out[1:])

            hs.append(block.register_forward_pre_hook(save_input, with_kwargs=True))
            hs.append(block.attn.register_forward_hook(attn_hook))
            hs.append(block.mlp.register_forward_hook(_site_hook(on_site, l, "mlp")))
            hs.append(block.register_forward_hook(_site_hook(on_site, l, "resid")))
        return hs


_SITE_MODELS = {PYTHIA_70M: PythiaSiteModel, GPT2_SMALL: GPT2SiteModel}


def resolve_dtype(dtype: Optional[str] = None) -> torch.dtype:
    """Public-family precision: ``dtype`` when given, else config.model.dtype,
    else float32."""

    if dtype is None:
        from config import config
        dtype = config.model.dtype
    return _DTYPES[dtype] if dtype is not None else torch.float32


def load_site_model(family: Optional[str] = None,
                    device: Optional[torch.device] = None,
                    dtype: Optional[str] = None) -> HFSiteModel:
    """Load the family's HF checkpoint (eval mode, eager attention, the
    family's precision) wrapped as an HFSiteModel. Honours HF_HUB_OFFLINE."""

    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    fam = get_family(family)
    if fam.name not in _SITE_MODELS:
        raise ValueError(f"no HF site adapter for family {fam.name!r}")
    hf = AutoModelForCausalLM.from_pretrained(
        fam.hf_model_id, dtype=resolve_dtype(dtype), attn_implementation="eager")
    hf.to(device if device is not None else torch.device("cpu")).eval()
    return _SITE_MODELS[fam.name](hf, fam)


class HFInference(Inference):
    """Inference for a public family. Same forward (inherited); compile and
    gradient checkpointing are no-ops (tiny models, eager HF modules).

    Built by ``Inference(device, family=...)`` (or config.model.family), so
    existing entry points need no changes."""

    def __init__(self, device: Optional[torch.device] = None, compile: bool = False,
                 family: Optional[str] = None, dtype: Optional[str] = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = load_site_model(family, self.device, dtype)
        self.family = self.model.family.name
        self._compiled = False
        self._eager_forwards = []

    def enable_compile(self):
        return

    def disable_compile(self):
        return

    def enable_grad_checkpointing(self):
        return

    def disable_grad_checkpointing(self):
        return
