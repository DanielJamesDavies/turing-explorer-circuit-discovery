"""SAE bank for the public model families (DAN-35).

Same contract as the TuringLLM SAEBank — ``encode(x, kind, layer) -> (top_acts,
top_idx)`` [.., k], ``decode(latents, kind, layer, add_bias=...)``, ``saes[kind]
[layer]`` with ``.encoder.weight`` / ``._get_bias_eff()``, ``kinds``, ``k``,
``d_sae`` — so every engine and eval consumer runs unchanged. The dictionaries
are loaded into the repo's own ``sae.topk_sae.SAE`` (relu -> top-k on
x W_enc^T + b_eff, b_eff = b_enc - W_enc b_dec), which is exactly:

  sparsify (Pythia)  pre = (x - b_dec) W_enc^T + b_enc; topk(relu(pre)); decode
                     c W_dec + b_dec. encoder.weight -> encoder.weight, W_dec ->
                     decoder.weight^T.
  SAELens (GPT-2)    the same on x_n (apply_b_dec_to_input); W_enc [d_in, d_sae]
                     -> encoder.weight^T. SAELens applies ReLU after top-k
                     instead of before: the same dense code (a slot whose
                     pre-activation is <= 0 is 0 either way).

The bank never normalises: ``x`` is the site activation the hook protocol
delivers, already in SAE input space (model/families.py). ``site_input`` converts
a raw model-space stream for callers that hold one.

Two differences from SAEBank, both deliberate:
  * lazy loading (``preload=False``): ``saes[kind][layer]`` loads on first index;
    iteration yields None for unloaded slots, as SAEBank's own lists do before
    load_saes(). CPU tests load only the sites they touch.
  * SAE parameters never require grad (gradients flow through activations
    only), so an SAE loaded mid-fit cannot accumulate weight grads.
"""

from __future__ import annotations

import contextlib
import json
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch

from model.families import ModelFamily, get_family
from .bank import SAEBank
from .topk_sae import SAE


class _LazySites:
    """``bank.saes[kind]``: a per-layer list that loads on first index."""

    def __init__(self, bank: "PublicSAEBank", kind: str) -> None:
        self._bank = bank
        self._kind = kind
        self._items: List[Optional[SAE]] = [None] * bank.n_layer

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, layer):
        if isinstance(layer, slice):
            return [self[i] for i in range(*layer.indices(len(self)))]
        if self._items[layer] is None:
            if (self._kind, int(layer)) in self._bank.missing_sites:
                return None
            self._items[layer] = self._bank._load_or_skip(self._kind, layer)
        return self._items[layer]

    def __iter__(self):
        return iter(list(self._items))

    def loaded(self) -> List[int]:
        return [i for i, s in enumerate(self._items) if s is not None]


class PublicSAEBank(SAEBank):
    """Top-K SAE bank for a public family; see the module docstring."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        devices: Optional[List[torch.device]] = None,
        load_decoders: bool = True,
        compile: bool = False,
        family: Optional[str] = None,
        preload: Optional[bool] = None,
        dtype: Optional[str] = None,
        skip_missing: Optional[bool] = None,
    ):
        from config import config
        from model.hf_adapters import resolve_dtype

        fam: ModelFamily = get_family(family)
        if fam.sae is None:
            raise ValueError(f"family {fam.name!r} has no public SAE source")
        self.family = fam.name
        self.source = fam.sae
        self.n_layer = fam.n_layer
        self.d_model = fam.d_model
        self.d_sae = fam.sae.d_sae
        self.k = fam.sae.k
        self.kinds = list(fam.kinds)
        # Read by eval.ablation_faithfulness.upstream_sites (the per-architecture
        # within-layer causal order; see model/families.py).
        self.same_layer_upstream: Dict[str, tuple] = dict(fam.same_layer_upstream)
        self.normalize = fam.sae.normalize
        self.norm_eps = float(fam.sae.norm_eps)
        self.dtype = resolve_dtype(dtype)
        self.load_decoders = True          # decoders always resident (tiny dictionaries)
        self.compile = False

        if devices is not None and len(devices) >= 1:
            self.devices = [torch.device(d) for d in devices]
        elif device is not None:
            self.devices = [torch.device(device)]
        else:
            self.devices = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
        self.device = self.devices[0]
        self.is_multi_gpu = len(self.devices) > 1
        mid = self.n_layer // 2
        self.layer_device_map = {
            layer: (self.devices[0] if (not self.is_multi_gpu or layer < mid) else self.devices[1])
            for layer in range(self.n_layer)
        }
        self.parallel_kinds = False
        self._kind_streams = None
        # DAN-37: sites whose dictionary could not be loaded (skip_missing only).
        self.skip_missing = bool(config.model.skip_missing_saes) if skip_missing is None else bool(skip_missing)
        self.missing_sites: set = set()

        self.saes = {kind: _LazySites(self, kind) for kind in self.kinds}
        if preload if preload is not None else bool(config.model.sae_preload):
            self.load_saes()

    # ------------------------------------------------------------------ loading
    def _load_or_skip(self, kind: str, layer: int) -> Optional[SAE]:
        """``_load``; under ``skip_missing`` a dictionary that is not available
        (not in the local HF cache when offline, or absent from the repo) marks
        the site missing and returns None instead of raising."""

        if not self.skip_missing:
            return self._load(kind, layer)
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
        try:
            return self._load(kind, layer)
        except (EntryNotFoundError, LocalEntryNotFoundError, FileNotFoundError) as error:
            self.missing_sites.add((kind, int(layer)))
            print(f"[public_bank] {self.family}: SAE {kind} L{layer} unavailable, site skipped "
                  f"({type(error).__name__})")
            return None

    def has_site(self, kind: str, layer: int) -> bool:
        """True when the (kind, layer) dictionary is loaded or loadable; a
        missing site (skip_missing) is False. Loads lazily."""

        return self.saes[kind][layer] is not None

    def _load(self, kind: str, layer: int) -> SAE:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if kind not in self.kinds:
            raise KeyError(f"{self.family}: unknown site kind {kind!r}")
        repo, weights_file, cfg_file = self.source.files(kind, layer)
        sd = load_file(hf_hub_download(repo, weights_file))
        if self.source.fmt == "sparsify":
            enc_w, enc_b = sd["encoder.weight"], sd["encoder.bias"]
        elif self.source.fmt == "saelens":
            cfg = json.load(open(hf_hub_download(repo, cfg_file)))
            self._check_saelens_cfg(cfg, kind, layer)
            enc_w, enc_b = sd["W_enc"].T, sd["b_enc"]
        else:
            raise ValueError(f"unknown SAE format {self.source.fmt!r}")
        w_dec, b_dec = sd["W_dec"], sd["b_dec"]
        if tuple(enc_w.shape) != (self.d_sae, self.d_model) or tuple(w_dec.shape) != (self.d_sae, self.d_model):
            raise ValueError(f"{repo}/{weights_file}: unexpected shapes "
                             f"{tuple(enc_w.shape)} / {tuple(w_dec.shape)}")

        target = self.layer_device_map[layer]
        sae = SAE(self.d_model, self.d_sae, self.k, device=torch.device("cpu"))
        with torch.no_grad():
            sae.encoder.weight.copy_(enc_w)
            sae.encoder.bias.copy_(enc_b)
            sae.decoder.weight.copy_(w_dec.T)
            sae.decoder_bias.copy_(b_dec)
        sae.to(device=target, dtype=self.dtype)
        sae.device = target
        sae.decoder_device = target
        for p in sae.parameters():
            p.requires_grad_(False)
        sae._get_bias_eff()
        return sae

    def _check_saelens_cfg(self, cfg: dict, kind: str, layer: int) -> None:
        """Refuse a dictionary whose stored config disagrees with the family
        spec the adapter implements (k, input normalisation, b_dec on input)."""
        want = {"k": self.k, "normalize_activations": self.normalize,
                "apply_b_dec_to_input": True, "d_sae": self.d_sae, "d_in": self.d_model}
        got = {"k": int((cfg.get("activation_fn_kwargs") or {}).get("k", -1)),
               "normalize_activations": cfg.get("normalize_activations"),
               "apply_b_dec_to_input": cfg.get("apply_b_dec_to_input"),
               "d_sae": cfg.get("d_sae"), "d_in": cfg.get("d_in")}
        if cfg.get("activation_fn_str") != "topk" or got != want:
            raise ValueError(f"{self.family} {kind} L{layer}: SAE config {got} "
                             f"(activation {cfg.get('activation_fn_str')!r}) != spec {want}")

    def load_saes(self):
        for kind in self.kinds:
            for layer in range(self.n_layer):
                self.saes[kind][layer]

    def release(self, kind: str, layer: int) -> None:
        """Drop one loaded SAE (it reloads on next use); for audits and CPU
        tests that walk every site without holding all dictionaries."""
        self.saes[kind]._items[layer] = None

    @contextlib.contextmanager
    def pin_decoders(self):
        yield

    def _autocast_ctx(self, device: torch.device):
        if self.dtype == torch.float32:
            return nullcontext()
        return super()._autocast_ctx(device)

    # ------------------------------------------------------------------ site space
    def site_input(self, x: torch.Tensor, kind: str, layer: int) -> torch.Tensor:
        if self.normalize == "layer_norm":
            from model.hf_adapters import layer_norm_in
            return layer_norm_in(x, self.norm_eps)[0]
        return x
