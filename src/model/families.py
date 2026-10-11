"""Model families: which transformer and which SAE dictionaries the engine runs on.

Selected by ``config.model.family`` (default ``"turingllm"``, the TuringLLM path,
which is untouched by everything in this module), or per call by the ``family``
argument of ``Inference`` / ``SAEBank``. Two public families (DAN-34/35):

  "pythia-70m"  EleutherAI/pythia-70m (HF GPT-NeoX) + EleutherAI/sae-pythia-70m-32k:
                Top-K k=16, 32,768 latents, 6 layers, sites attn / mlp / resid.
                Encode: (x - b_dec) W_enc^T + b_enc -> relu -> top-k (EleutherAI
                sparsify; 033's transcription). No input normalisation.
  "gpt2-small"  gpt2 (HF GPT2LMHeadModel) + jbloom/GPT2-Small-OAI-v5-32k-*-SAEs:
                Top-K k=32, 32,768 latents, 12 layers, sites attn / resid_mid /
                mlp / resid. normalize_activations="layer_norm": the SAE reads
                x_n = (x - mean) / (std + 1e-5) per token (torch's unbiased std),
                apply_b_dec_to_input, prepend_bos False, context 64 (053's
                transcription, verified against the repo cfg.json at load).

SITE ACTIVATION (the engine's x). For every family the tensor a patcher's
``transform(layer, kind, x)`` and an ``activations_callback`` receive is the SAE's
INPUT at that site. For layer-norm families that is x_n, not the model-space
stream; the adapter maps an edit x_n -> x_n' back to the model as
x <- x + (x_n' - x_n) * (std + eps), i.e. the usual SAE delta scaled by the
token's std (mean and b_dec cancel). So the engine's seed tap ``x @ w + b`` with
(w, b) = (encoder row, b_enc - W_enc b_dec) is the SAE's own pre-activation, and
``decode(delta, add_bias=False)`` stays a per-token-constant linear map.

CAUSAL ORDER WITHIN A LAYER (``same_layer_upstream``). Every site at a lower layer
feeds every site at a higher one (all edits enter the residual stream). Within a
layer the order is architecture-specific:

  turingllm   attn -> mlp -> resid            (sequential block: mlp reads x + attn)
  pythia-70m  attn, mlp -> resid              (use_parallel_residual=True: attn and
                                               mlp BOTH read the layer input, so the
                                               same-layer mlp does not see attn)
  gpt2-small  attn -> resid_mid -> mlp -> resid

Each tuple below lists the same-layer sites a site reads, in forward order. The
TuringLLM SAEBank does not expose this table, so ``upstream_sites`` keeps its
historical rule for it (identical to the turingllm row).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

TURINGLLM = "turingllm"
PYTHIA_70M = "pythia-70m"
GPT2_SMALL = "gpt2-small"


@dataclass(frozen=True)
class SAESource:
    """Where a family's dictionaries live and how they read their input."""

    fmt: str                        # "sparsify" (EleutherAI) | "saelens" (jbloom OAI v5)
    d_sae: int
    k: int
    normalize: Optional[str]        # None | "layer_norm"
    norm_eps: float
    prepend_bos: bool
    context_size: int
    # (kind, layer) -> (hf repo, weights file inside the repo, cfg.json or None)
    files: Callable[[str, int], Tuple[str, str, Optional[str]]]


@dataclass(frozen=True)
class ModelFamily:
    name: str
    hf_model_id: Optional[str]      # None for TuringLLM (local checkpoint)
    tokenizer_id: str
    n_layer: int
    d_model: int
    kinds: Tuple[str, ...]          # within-layer FORWARD order; comp_idx = layer * len(kinds) + kind_idx
    same_layer_upstream: Dict[str, Tuple[str, ...]]
    sae: Optional[SAESource]        # None for TuringLLM (SAEBank loads its own .pth files)


def _pythia_files(kind: str, layer: int) -> Tuple[str, str, Optional[str]]:
    hookpoint = {"resid": "layers.%d" % layer,
                 "attn": "layers.%d.attention" % layer,
                 "mlp": "layers.%d.mlp" % layer}[kind]
    return "EleutherAI/sae-pythia-70m-32k", hookpoint + "/sae.safetensors", None


_GPT2_REPO_KIND = {"attn": "attn-out", "resid_mid": "resid-mid",
                   "mlp": "mlp-out", "resid": "resid-post"}


def _gpt2_files(kind: str, layer: int) -> Tuple[str, str, Optional[str]]:
    repo = "jbloom/GPT2-Small-OAI-v5-32k-%s-SAEs" % _GPT2_REPO_KIND[kind]
    # resid-post is the only set whose folders keep the ".pt" suffix (053).
    folder = "v5_32k_layer_%d%s" % (layer, ".pt" if kind == "resid" else "")
    return repo, folder + "/sae_weights.safetensors", folder + "/cfg.json"


FAMILIES: Dict[str, ModelFamily] = {
    TURINGLLM: ModelFamily(
        name=TURINGLLM, hf_model_id=None,
        tokenizer_id="microsoft/Phi-3-mini-4k-instruct",
        n_layer=12, d_model=1024, kinds=("attn", "mlp", "resid"),
        same_layer_upstream={"attn": (), "mlp": ("attn",), "resid": ("attn", "mlp")},
        sae=None,
    ),
    PYTHIA_70M: ModelFamily(
        name=PYTHIA_70M, hf_model_id="EleutherAI/pythia-70m",
        tokenizer_id="EleutherAI/pythia-70m",
        n_layer=6, d_model=512, kinds=("attn", "mlp", "resid"),
        same_layer_upstream={"attn": (), "mlp": (), "resid": ("attn", "mlp")},
        sae=SAESource(fmt="sparsify", d_sae=32768, k=16, normalize=None, norm_eps=0.0,
                      prepend_bos=False, context_size=64, files=_pythia_files),
    ),
    GPT2_SMALL: ModelFamily(
        name=GPT2_SMALL, hf_model_id="gpt2", tokenizer_id="gpt2",
        n_layer=12, d_model=768, kinds=("attn", "resid_mid", "mlp", "resid"),
        same_layer_upstream={"attn": (), "resid_mid": ("attn",),
                             "mlp": ("attn", "resid_mid"),
                             "resid": ("attn", "resid_mid", "mlp")},
        sae=SAESource(fmt="saelens", d_sae=32768, k=32, normalize="layer_norm", norm_eps=1e-5,
                      prepend_bos=False, context_size=64, files=_gpt2_files),
    ),
}


def resolve_family(name: Optional[str] = None) -> str:
    """The family name to use: ``name`` when given, else ``config.model.family``."""

    if name is None:
        from config import config
        name = str(config.model.family)
    if name not in FAMILIES:
        raise ValueError(f"unknown model family {name!r}; known: {sorted(FAMILIES)}")
    return name


def get_family(name: Optional[str] = None) -> ModelFamily:
    return FAMILIES[resolve_family(name)]


@dataclass(frozen=True)
class StoreGrid:
    """The latent grid the data-pass stores are allocated on (DAN-37): flat
    component index comp = layer * len(kinds) + kind_idx, d_sae latents each.
    ``llm_config`` / ``sae_config`` mirror the TuringLLMConfig / SAEConfig
    fields the stores read (n_layer, n_embd; d_sae, k)."""

    family: str
    n_layer: int
    kinds: Tuple[str, ...]
    d_sae: int
    k: int
    d_model: int

    @property
    def n_components(self) -> int:
        return self.n_layer * len(self.kinds)

    @property
    def llm_config(self):
        from types import SimpleNamespace
        return SimpleNamespace(n_layer=self.n_layer, n_embd=self.d_model)

    @property
    def sae_config(self):
        from types import SimpleNamespace
        return SimpleNamespace(d_sae=self.d_sae, k=self.k)


def public_store_grid(name: Optional[str] = None) -> Optional[StoreGrid]:
    """The store grid of a public family, or None for TuringLLM.

    None means "keep the historical TuringLLMConfig() / SAEConfig() reads":
    the stores (store/context.py, latent_stats.py, logit_context.py,
    top_coactivation.py, seq_repr.py, neg_ctx/backends.py) only override their
    grid when this returns a value, so the TuringLLM path is unchanged."""

    fam = get_family(name)
    if fam.sae is None:
        return None
    return StoreGrid(family=fam.name, n_layer=fam.n_layer, kinds=tuple(fam.kinds),
                     d_sae=fam.sae.d_sae, k=fam.sae.k, d_model=fam.d_model)


def family_upstream_sites(family: ModelFamily, seed_layer: int, seed_kind: str) -> set:
    """Every (layer, kind) site that feeds (seed_layer, seed_kind) in ``family``."""

    sites = {(layer, kind) for layer in range(seed_layer) for kind in family.kinds}
    sites.update((seed_layer, kind) for kind in family.same_layer_upstream[seed_kind])
    return sites
