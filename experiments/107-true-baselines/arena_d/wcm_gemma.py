"""Arena D (DAN-203): WCM on Gemma-2-2B, two substrates. Experiment code; nothing under src/, experiments/062-*,
experiments/082-* or third_party/ is modified. register() adds, at runtime (as arena C's wcm_marks.py does):

  gemma-2-2b-plt          D1: circuit-tracer's per-layer transcoders (mwhanna/gemma-scope-transcoders @3412e19b),
                          one "tc" site per layer (26). READ x_n = TL ln2.hook_normalized (the RMS-normalised MLP input
                          before the (1 + w) weight, fp32, as Gemma2RMSNorm._norm computes it); WRITE at the
                          post_feedforward_layernorm output (TL hook_mlp_out): y <- y + (c_hat - c) W_dec. The real MLP
                          runs, so the transcoder error stays live.
  gemma-2-2b-gsres-gap6   D2: Caples et al.'s 2B substrate, Gemma Scope 16k residual JumpReLU SAEs at resid_post of
                          layers 0 / 6 / 12 / 18 / 24 (L0 105 / 70 / 82 / 74 / 73; the other 21 layers hold an inert
                          NullSAE). Two modes (GemmaResSiteModel.caples):
                            None    native WCM: x <- x + (c_hat - c) W_dec, the true error kept;
                            {...}   Caples' substrate: at positions >= 1 the site outputs decode(c_hat) + eps_bar[pos]
                                    (eps_bar = their per-position mean error over corrupted prompts), BOS untouched,
                                    exactly their build_sae_hook_fn(use_mean_error=True) splice.

The transcoder site, read before the MLP and written after it, uses ARENA A's mechanism (arena_a/README.md §5,
DAN-195): the hook calls on_site(L, "tc", x_in) and adds (y - x_in) to the module output, so the engine's one-tensor
site contract (transform(layer, kind, x) -> x + decode(c_hat) - decode(c)) is unchanged and no src change is needed.
The edit is exact as long as no engine path returns anything other than x + (a code-space difference decoded without
bias) at a site; arena A's risk 2 (that audit) applies here unchanged.

BOS (position 0) is never edited on either family (circuit-tracer zeroes features and errors at position 0; Caples
never edits BOS); the site models restore it after every transform.

Dictionaries: JumpReLU, d_sae 16,384, k = d_sae (whole code, an expanded arange as indices; arena C's convention). Our
SAE class's seed tap reads w . x + b_eff; here b_eff = b_enc (no pre-encoder b_dec subtraction: neither Gemma Scope nor
the PLTs centre the input), so the tap is the latent's own pre-activation, the quantity an AG row explains. The
`threshold` buffer is what src/sae/dense.seed_threshold reads (engine change 1, 2026-11-04).

BOS and means (engine change 3, 2026-11-04): both families' sites see position 0, so GemmaSAEBank sets
mean_skip_positions = 1 and eval.floors' pooled means (and 081's chunk weights) skip it.

Caples substrate, once per forward: Inference.forward always registers the capture hooks and, with a patcher, the patch
hooks too. GemmaResSiteModel applies the splice in exactly one hook per layer: the patch hook when one is active, else
the capture hook, else (a plain call) its own. So the open-gate reference of the objective and the scorer's full-model
read are the spliced model, as Caples' reference is.

Task runner (D2, 2026-11-05): `run` = arena C's wcm_marks.run on 081's Engine, both families. Data: "sfc" (SFC's 100
discovery / 40 evaluation prompts, arena C's load_examples with the Gemma tokenizer; D1's default) or "caples" (their
interp_bottlenecks.py filter and file-order batches: N 3000, batches of 16, training = the batches their 2B fit walks,
84 at portion 0.4; held-out [ours] = the first --n-held equal-length rc_test records; D2's default). Substrate --mode
native | caples (caples: the gsres family only; eps_bar and, under per_position means, the C fill from caples_means.py's
.pt, "theirs" = their running-mean functions, "exact" = the true means of the same 640 prompts).

CLI (WSL, repo root):
  CUDA_VISIBLE_DEVICES=... ./.venv/bin/python experiments/107-true-baselines/arena_d/wcm_gemma.py run \
      --family gemma-2-2b-gsres-gap6 --mode caples --caples-means out/caples_means/rc_gap6.pt --means per_position \
      --tasks rc --lams 1e-3,3e-3 [--data caples] [--steps 400] [--out DIR]
  env: STOP_AT=HH:MM / STOP_EPOCH (no new fit after it), HELDOUT_N (081's)
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
D081 = ROOT / "experiments" / "081-task-circuits"
D059 = ROOT / "experiments" / "059-context-pool"
ARENA_C = HERE.parent / "arena_c"
CAPLES_SRC = ROOT / "third_party" / "ScalableSAECircuits"
OUT_DEFAULT = HERE / "out" / "wcm"

import torch  # noqa: E402
from torch import nn  # noqa: E402

from model import families as _F  # noqa: E402
from model import hf_adapters as _HA  # noqa: E402
from sae.public_bank import PublicSAEBank as _PublicSAEBank  # noqa: E402
from sae.topk_sae import SAE as _SAE  # noqa: E402

# ------------------------------------------------------------------------------------------------------- constants
HF_ID = os.environ.get("GEMMA_HF_ID", "unsloth/gemma-2-2b")      # pod: google/gemma-2-2b (fp32 originals)
N_LAYER, D_MODEL, D_SAE = 26, 2304, 16384
FAM_PLT = "gemma-2-2b-plt"
FAM_RES = "gemma-2-2b-gsres-gap6"
PLT_REPO, PLT_REV = "mwhanna/gemma-scope-transcoders", "3412e19be8f72d47f8683cb92028a46513222b2b"
PLT_REV_CACHED = "bd5773156dea09893636c801df1237d0410307d2"     # the local snapshot; identical layer files
RES_REPO, RES_REV = "google/gemma-scope-2b-pt-res", "fd571b47c1c64851e9b1989792367b9babb4af63"
# Caples' picks: interp_bottlenecks.py's "L0 closest to 100" over sae-lens 5.3.3's directory, layers range(0, 26, 6)
# (verified in caples107, 2026-11-03).
RES_IDS = {0: "layer_0/width_16k/average_l0_105", 6: "layer_6/width_16k/average_l0_70",
           12: "layer_12/width_16k/average_l0_82", 18: "layer_18/width_16k/average_l0_74",
           24: "layer_24/width_16k/average_l0_73"}


# ------------------------------------------------------------------------------------------------------- dictionaries
class JumpReLUSAE(_SAE):
    """JumpReLU dictionary in our SAE class: c = pre * (pre > theta), pre = x W_enc^T + b_enc (no b_dec on the input).
    k = d_sae: encode returns the whole code and an expanded arange (nothing truncated)."""

    def __init__(self, d_model: int = D_MODEL, d_sae: int = D_SAE, device=None):
        super().__init__(d_model, d_sae, k=d_sae, device=device if device is not None else torch.device("cpu"))
        self.register_buffer("threshold", torch.zeros(d_sae))
        self._ar: Dict = {}

    def _get_bias_eff(self) -> torch.Tensor:
        return self.encoder.bias

    def _arange(self, like: torch.Tensor) -> torch.Tensor:
        a = self._ar.get(like.device)
        if a is None:
            a = torch.arange(self.d_sae, device=like.device)
            self._ar[like.device] = a
        return a.expand(*like.shape[:-1], self.d_sae)

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.encoder.weight.dtype:
            x = x.to(self.encoder.weight.dtype)
        return nn.functional.linear(x, self.encoder.weight, self.encoder.bias)

    def encode(self, x: torch.Tensor):
        p = self.pre(x)
        acts = p * (p > self.threshold).to(p.dtype)
        return acts, self._arange(acts)


class NullSAE(nn.Module):
    """Inert slot (arena C's NullSAE): code 0, decode 0, no parameters; no edit can enter."""

    null = True

    def __init__(self, device, dtype=torch.float32, d_model: int = D_MODEL, d_sae: int = D_SAE):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, d_sae
        self.device = self.decoder_device = device
        self.dtype = dtype

    def encode(self, x: torch.Tensor):
        z = torch.zeros(*x.shape[:-1], self.d_sae, device=x.device, dtype=self.dtype)
        return z, torch.arange(self.d_sae, device=x.device).expand_as(z)

    def decode(self, latents, ensure_device: bool = True, add_bias: bool = True):
        return latents.new_zeros(*latents.shape[:-1], self.d_model)

    def move_decoder_to_vram(self):
        return None

    def remove_decoder_from_vram(self, empty_cache: bool = True):
        return None


def jumprelu_from(W_enc: torch.Tensor, b_enc: torch.Tensor, W_dec: torch.Tensor, b_dec: torch.Tensor,
                  threshold: torch.Tensor) -> JumpReLUSAE:
    """Build from W_enc [d_sae, d], W_dec [d_sae, d] (circuit-tracer / sae-lens row layout), fp32 on the CPU."""
    d_sae, d = W_enc.shape
    sae = JumpReLUSAE(d, d_sae)
    with torch.no_grad():
        sae.encoder.weight.copy_(W_enc.float())
        sae.encoder.bias.copy_(b_enc.float())
        sae.decoder.weight.copy_(W_dec.float().T)
        sae.decoder_bias.copy_(b_dec.float())
        sae.threshold.copy_(threshold.float())
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae


def load_plt_state(layer: int):
    """(W_enc [16384, 2304], b_enc, W_dec [16384, 2304], b_dec, threshold): layer_<L>.safetensors as circuit-tracer's
    load_transcoder reads it (keys W_enc, W_dec, b_enc, b_dec, activation_function.threshold)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    from safetensors.torch import load_file
    try:
        path = hf_hub_download(PLT_REPO, "layer_%d.safetensors" % layer, revision=PLT_REV)
    except LocalEntryNotFoundError:
        # offline with only 038's snapshot cached: every layer_*.safetensors LFS sha256 there equals 3412e19b's
        # (checked 2026-11-03; the one later commit edits README metadata only)
        path = hf_hub_download(PLT_REPO, "layer_%d.safetensors" % layer, revision=PLT_REV_CACHED)
    sd = load_file(path)
    return sd["W_enc"], sd["b_enc"], sd["W_dec"], sd["b_dec"], sd["activation_function.threshold"]


def load_res_state(layer: int):
    """(W_enc [16384, 2304], b_enc, W_dec [16384, 2304], b_dec, threshold) from Gemma Scope's params.npz (its W_enc is
    stored [d_model, d_sae] and is transposed here, as sae-lens and circuit-tracer do)."""
    import numpy as np
    from huggingface_hub import hf_hub_download
    z = np.load(hf_hub_download(RES_REPO, RES_IDS[layer] + "/params.npz", revision=RES_REV))
    t = {k: torch.from_numpy(z[k]) for k in z.files}
    return t["W_enc"].T.contiguous(), t["b_enc"], t["W_dec"], t["b_dec"], t["threshold"]


def _files(kind: str, layer: int):
    if kind == "tc":
        return PLT_REPO, "layer_%d.safetensors" % layer, None
    return RES_REPO, RES_IDS.get(layer, "") + "/params.npz", None


# Arena-D families: name -> kind, dictionary layers, state loader (layer -> jumprelu_from args). register_tiny adds
# test families with the same site models and bank logic (tiny random Gemma-2 configurations, no download).
SPECS: Dict[str, Dict] = {
    FAM_PLT: dict(kind="tc", layers=tuple(range(N_LAYER)), state=load_plt_state),
    FAM_RES: dict(kind="resid", layers=tuple(sorted(RES_IDS)), state=load_res_state),
}


class GemmaSAEBank(_PublicSAEBank):
    """PublicSAEBank for the arena-D families (any other family: the parent, unchanged). mean_skip_positions = 1:
    the Gemma sites see BOS (position 0, never edited, restored by the site models), so eval.floors' pooled A / C means
    and 081's chunk weights exclude it (engine change 3, 2026-11-04)."""

    @property
    def mean_skip_positions(self) -> int:
        return 1 if self.family in SPECS else 0          # any other family (dispatch patched): unchanged

    @property
    def real_sites(self) -> List[Tuple[int, str]]:
        spec = SPECS.get(self.family)
        if spec is not None:
            return [(l, spec["kind"]) for l in spec["layers"]]
        return [(l, k) for l in range(self.n_layer) for k in self.kinds]

    def _load(self, kind: str, layer: int):
        spec = SPECS.get(self.family)
        if spec is None:
            return super()._load(kind, layer)
        if kind not in self.kinds:
            raise KeyError("%s: unknown site kind %r" % (self.family, kind))
        target = self.layer_device_map[layer]
        if layer not in spec["layers"]:
            return NullSAE(target, self.dtype, self.d_model, self.d_sae)
        sae = jumprelu_from(*spec["state"](layer))
        sae.to(device=target, dtype=self.dtype)
        sae.device = sae.decoder_device = target
        return sae


# ------------------------------------------------------------------------------------------------------- site models
def _layers(hf: nn.Module):
    return hf.model.layers


class GemmaPLTSiteModel(_HA.HFSiteModel):
    """Gemma-2 with one transcoder site per layer (kind "tc"): read before the MLP, write after it (arena A's
    mechanism). The transform sees x_n (fp32, [B, T, d]) and its edit (y - x_n) is added to the
    post_feedforward_layernorm output; position 0 is restored."""

    keep_bos = True

    def _register(self, on_site) -> List:
        hs = []
        for l, layer in enumerate(_layers(self.hf)):
            box: Dict[str, torch.Tensor] = {}

            def pre(m, args, _b=box):
                _b["xn"] = m._norm(args[0].float())          # == TL blocks.L.ln2.hook_normalized (fold_ln=False)

            def post(_m, _inp, out, _l=l, _b=box):
                xn = _b.pop("xn")
                y = on_site(_l, "tc", xn)
                if y is None or y is xn:
                    return None
                d = (y - xn)
                if self.keep_bos:
                    d = d.clone()
                    d[:, :1] = 0
                return out + d.to(out.dtype)

            hs.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(pre))
            hs.append(layer.post_feedforward_layernorm.register_forward_hook(post))
        return hs


class GemmaResSiteModel(_HA.HFSiteModel):
    """Gemma-2 with a resid_post site per layer (kind "resid"; only RES_IDS layers hold dictionaries). caples = None
    (native, error kept) or {layer: (sae, eps_bar [T, d])} for Caples' substrate."""

    keep_bos = True
    caples: Optional[Dict[int, Tuple[nn.Module, torch.Tensor]]] = None

    def __init__(self, hf_model: nn.Module, family) -> None:
        super().__init__(hf_model, family)
        self._n_patch = 0          # active patch_sites registrations
        self._n_hooks = 0          # active registrations of any kind
        self._role: Optional[str] = None

    # One splice per layer per forward (module docstring): the role of the registration being made is set here and
    # read by _register at context entry.
    @contextlib.contextmanager
    def patch_sites(self, transform):
        self._role = "patch"
        with super().patch_sites(transform):
            yield

    @contextlib.contextmanager
    def capture_sites(self, callback=None, capture: bool = True):
        self._role = "capture"
        with super().capture_sites(callback=callback, capture=capture) as out:
            yield out

    def forward(self, idx, targets=None, input_pos=None, return_all_logits: bool = False):
        if self.caples and self._n_hooks == 0:                 # a plain call: still the spliced model
            with self.patch_sites(lambda l, k, x: None):
                return super().forward(idx, targets, input_pos, return_all_logits)
        return super().forward(idx, targets, input_pos, return_all_logits)

    def _splice(self, l: int, x: torch.Tensor, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.caples or l not in self.caples:
            if y is None or not self.keep_bos:
                return y
            y = y.clone()
            y[:, :1] = x[:, :1]
            return y
        sae, eps_bar = self.caples[l]
        base = x if y is None else y                            # x + decode(c_hat) - decode(c)
        c = sae.encode(x)[0]
        err = x - sae.decode(c).to(x.dtype)                     # the true error at this site
        T = x.shape[1]
        z = base - err + eps_bar[:T].to(device=x.device, dtype=x.dtype)    # = decode(c_hat) + eps_bar[pos]
        z = z.clone()
        z[:, :1] = x[:, :1]                                    # BOS keeps the live stream (their torch.where)
        return z

    def _register(self, on_site) -> List:
        role, self._role = (self._role or "patch"), None
        if role == "patch" and self._n_patch and self.caples:
            raise RuntimeError("GemmaResSiteModel: nested patch_sites in Caples mode would splice twice")
        sm = self

        class _Count:                                          # a "handle" whose remove() undoes the counters
            def remove(self_):
                sm._n_hooks -= 1
                if role == "patch":
                    sm._n_patch -= 1

        self._n_hooks += 1
        if role == "patch":
            self._n_patch += 1
        hs: List = [_Count()]
        for l, layer in enumerate(_layers(self.hf)):
            def hook(_m, _inp, out, _l=l):
                is_tuple = isinstance(out, tuple)
                x = out[0] if is_tuple else out
                y = on_site(_l, "resid", x)
                if y is x:
                    y = None
                if role == "capture" and sm._n_patch:          # the patch hook (registered later) splices
                    return None
                z = self._splice(_l, x, y)
                if z is None:
                    return None
                return (z,) + tuple(out[1:]) if is_tuple else z
            hs.append(layer.register_forward_hook(hook))
        return hs


# ------------------------------------------------------------------------------------------------------- registration
def family_spec(name: str, n_layer: int = N_LAYER, d_model: int = D_MODEL, d_sae: int = D_SAE,
                kind: Optional[str] = None, hf_id: str = HF_ID) -> "_F.ModelFamily":
    kind = kind or ("tc" if name == FAM_PLT else "resid")
    return _F.ModelFamily(
        name=name, hf_model_id=hf_id, tokenizer_id=hf_id, n_layer=n_layer, d_model=d_model, kinds=(kind,),
        same_layer_upstream={kind: ()},
        sae=_F.SAESource(fmt="gemma-jumprelu", d_sae=d_sae, k=d_sae, normalize=None, norm_eps=0.0, prepend_bos=True,
                         context_size=1024, files=_files))


SITE_MODEL = {"tc": GemmaPLTSiteModel, "resid": GemmaResSiteModel}


def register(patch_bank_dispatch: bool = False) -> Tuple[str, str]:
    """Add both families and their site models (idempotent). With patch_bank_dispatch, SAEBank(...) builds
    GemmaSAEBank (the parent class for every other family)."""
    for name, site_model in ((FAM_PLT, GemmaPLTSiteModel), (FAM_RES, GemmaResSiteModel)):
        if name not in _F.FAMILIES:
            _F.FAMILIES[name] = family_spec(name)
        _HA._SITE_MODELS[name] = site_model
    if patch_bank_dispatch:
        import sae.public_bank as PB
        PB.PublicSAEBank = GemmaSAEBank
    return FAM_PLT, FAM_RES


def register_tiny(name: str, kind: str, n_layer: int, d_model: int, d_sae: int, layers: Sequence[int],
                  state: Callable[[int], tuple]) -> str:
    """[tests] A family with the arena-D site model and bank logic on a tiny configuration: `state(layer)` returns
    jumprelu_from's arguments for each dictionary layer (the others hold NullSAE)."""
    _F.FAMILIES[name] = family_spec(name, n_layer, d_model, d_sae, kind)
    _HA._SITE_MODELS[name] = SITE_MODEL[kind]
    SPECS[name] = dict(kind=kind, layers=tuple(sorted(layers)), state=state)
    return name


def make_inference(hf_model: nn.Module, family: str, device=None):
    """HFInference around an already-built HF model (tests: a tiny random Gemma-2), no checkpoint load."""
    inf = object.__new__(_HA.HFInference)                     # HFInference.__init__ would load fam.hf_model_id
    dev =torch.device(device) if device is not None else torch.device("cpu")
    inf.device = dev
    inf.model = _HA._SITE_MODELS[family](hf_model.to(dev).eval(), _F.FAMILIES[family])
    inf.family = family
    inf._compiled = False
    inf._eager_forwards = []
    return inf


def load_model(family: str, device=None, dtype: Optional[str] = None):
    """(inference, bank) for an arena-D family on `device`: the HF model through our site model, every dictionary
    preloaded (GemmaSAEBank)."""
    register()
    from config import config
    from model.inference import Inference
    config.model.family = family
    config.model.dtype = dtype
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf = Inference(dev, family=family)
    bank = GemmaSAEBank(device=dev, family=family, preload=True, dtype=dtype)
    return inf, bank


# ------------------------------------------------------------------------------------------------------- tasks
SVA_TASKS = ("simple", "within_rc", "rc", "nounpp")
TASK_NAME = {"simple": "sva_simple", "within_rc": "sva_within_rc", "rc": "sva_across_rc", "nounpp": "sva_across_pp"}
DATA_MODES = ("sfc", "caples")
# Gemma-tokenised prompt lengths incl. BOS (README §0: rc 7, simple 3, within_rc / nounpp 6); Caples' example_length
EXAMPLE_LEN = {"rc": 7, "simple": 3, "within_rc": 6, "nounpp": 6}
CAPLES_N, CAPLES_BS, CAPLES_PORTION, CAPLES_MEAN_BATCHES = 3000, 16, 0.4, 40
N_HELD_CAPLES = 400                    # [ours] held-out rc_test records scored per fit (decision: README §12)
_TOK: Dict = {}


def tokenizer(tok_id: str = HF_ID):
    if tok_id not in _TOK:
        from transformers import AutoTokenizer
        _TOK[tok_id] = AutoTokenizer.from_pretrained(tok_id)
    return _TOK[tok_id]


def _wcm_marks():
    """Arena C's module (load_examples, _row, curve_summary, _small, stop_passed): imported, never modified."""
    if str(ARENA_C) not in sys.path:
        sys.path.insert(0, str(ARENA_C))
    import wcm_marks  # noqa: E402
    return wcm_marks


def caples_train_batches(n_batches: int, portion: float = CAPLES_PORTION) -> int:
    """How many file-order batches their do_training_run walks: i = 0, 1, ... trains, then stops once i >= total * 1.1
    with total = n_batches * portion (utils.py:486-537): 84 for 187 batches at 0.4."""
    total = n_batches * portion
    i = 0
    while not i >= total * 1.1:
        i += 1
    return min(n_batches, i + 1)


def _caples_filter(path: Path, tok, length: int, n: Optional[int], exclude: Optional[set] = None,
                   stats: Optional[Dict] = None) -> List[Dict]:
    """interp_bottlenecks.py:197-210: both prefixes exactly `length` tokens (BOS included), file order, first n.
    exclude [ours, held-out only]: prefixes to drop (a record goes if its clean or patch prefix is in the set; a full
    prompt = prefix + answer, so a full-prompt repeat is a prefix repeat too); stats["dropped"] counts them."""
    out = []
    dropped = 0
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if len(tok(e["clean_prefix"]).input_ids) == len(tok(e["patch_prefix"]).input_ids) == length:
                if exclude is not None and (e["clean_prefix"] in exclude or e["patch_prefix"] in exclude):
                    dropped += 1
                    continue
                out.append(e)
                if n is not None and len(out) >= n:
                    break
    if stats is not None:
        stats["dropped"] = dropped
    return out


def _answer_id(tok, ans: str) -> int:
    ids = tok(ans, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError("answer %r is %d tokens (their to_tokens(...).squeeze(-1) needs one)" % (ans, len(ids)))
    return int(ids[0])


def build_task(task: str, data: str = "sfc", tok=None, n: int = CAPLES_N, batch_size: int = CAPLES_BS,
               portion: float = CAPLES_PORTION, mean_batches: int = CAPLES_MEAN_BATCHES, n_held: int = N_HELD_CAPLES,
               length: Optional[int] = None, family: str = FAM_RES) -> Dict:
    """One SVA task in 081's task schema, training rows first.
      sfc     arena C's build_task (SFC's load_examples, seed 12, min length; 100 discovery / 40 evaluation) with the
              Gemma tokenizer; the files are SFC's (= Caples' data/sva, same blobs).
      caples  their filter (both prefixes `length` tokens), the first n records of <task>_train.json in file order, cut
              to whole batches; training = the first caples_train_batches(...) batches (their fit's data), whose patch
              prompts are the C pool; their mean set = the patch prompts of batches 0..mean_batches-1 (recorded, and
              checked against the caples_means.py file); held-out [ours] = the first n_held equal-length records of
              <task>_test.json (SFC's test split, unused by them)."""
    tok = tok or tokenizer()
    if data == "sfc":
        WM = _wcm_marks()
        T = WM.build_task(task, "sfc", tok=tok)
        T.update(model=family, data_mode="sfc")
        return T
    if data != "caples":
        raise ValueError("data must be one of %s, got %r" % (DATA_MODES, data))
    WM = _wcm_marks()
    L = int(length or EXAMPLE_LEN[task])
    tr = _caples_filter(CAPLES_SRC / "data" / "sva" / ("%s_train.json" % task), tok, L, n)
    nb = len(tr) // batch_size
    if nb < 10:
        raise ValueError("%s: %d batches; their evaluation reads 10" % (task, nb))
    n_tb = caples_train_batches(nb, portion)
    train = tr[:n_tb * batch_size]
    mean_n = min(mean_batches, nb) * batch_size
    # held-out [ours], deduplicated (Daniel 2026-11-05): no rc_test record whose clean or patch prefix is a clean or
    # patch prefix of the training prompts or of the mean set (their patch prompts of batches 0..mean_batches-1; a
    # subset of training whenever training covers them, kept explicit). Their own sets are untouched.
    seen = {e[k] for e in train + tr[:mean_n] for k in ("clean_prefix", "patch_prefix")}
    dstats: Dict = {}
    held = _caples_filter(CAPLES_SRC / "data" / "sva" / ("%s_test.json" % task), tok, L, n_held, exclude=seen,
                          stats=dstats)
    rows = []
    for i, e in enumerate(train + held):
        r = WM._row(i, e, tok)
        r["good"], r["bad"] = [_answer_id(tok, e["clean_answer"])], [_answer_id(tok, e["patch_answer"])]
        rows.append(r)
    train_prefixes = {t["clean_prefix"] for t in train}
    return dict(name=TASK_NAME[task], sfc_task=task, family="sva", data_mode="caples", model=family,
                metric=dict(task="logit_diff", obj="logit_diff"), logit_metric="token", answer_mode="token",
                vac=dict(task=0.1, obj=0.1), contrast="original",
                split=dict(train=list(range(len(train))), held=list(range(len(train), len(rows)))),
                split_meta=dict(rule="Caples interp_bottlenecks.py filter (length %d), first %d records of %s_train in "
                                     "file order, %d batches of %d; training = batches 0..%d (portion %g); held-out = "
                                     "first %d length-%d records of %s_test whose clean / patch prefix is not a "
                                     "training or mean-set prefix (%d dropped before the cut)" % (
                                         L, n, task, nb, batch_size, n_tb - 1, portion, len(held), L, task,
                                         dstats["dropped"]),
                                n_train=len(train), n_held=len(held), held_dropped_dup=dstats["dropped"]),
                caples=dict(n_records=len(tr), n_batches=nb, batch_size=batch_size, train_batches=n_tb,
                            mean_batches=min(mean_batches, nb), mean_prompts=mean_n,
                            eval_batches=[0] + [nb - i for i in range(1, 10)]),
                prompt_len_train=[L], prompt_len_held=sorted({len(r["tokens"]) for r in rows[len(train):]}),
                overlap=dict(held_prefix_in_train=sum(e["clean_prefix"] in train_prefixes for e in held)),
                rows=rows)


# ------------------------------------------------------------------------------------------------------- Caples' means
CAPLES_FORMAT = "arena-d-caples-means-v1"
CAPLES_FILLS = ("theirs", "exact")


def load_caples_means(path, source: str = "theirs") -> Dict:
    """caples_means.py's file -> {"code": {layer: [T, d_sae]}, "err": {layer: [T, d]}, "mean_tokens": [n, T], "meta",
    "source", "sha"}. source "theirs" = their get_sae_means / get_sae_error_means (running-mean bug kept); "exact" =
    the true means of the same prompts, same substrate (their error-dropped splice)."""
    if source not in CAPLES_FILLS:
        raise ValueError("caples means source must be one of %s, got %r" % (CAPLES_FILLS, source))
    path = Path(path)
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("format") != CAPLES_FORMAT:
        raise ValueError("%s: not a %s file" % (path, CAPLES_FORMAT))
    s = d[source]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return dict(code={int(l): v["code"].float() for l, v in s.items()}, err={int(l): v["err"].float() for l, v in
                                                                              s.items()},
                mean_tokens=d["mean_tokens"].long(), meta=d.get("meta", {}), source=source, sha=sha, path=str(path))


def apply_caples(inf, bank, cm: Optional[Dict]) -> None:
    """Switch the residual site model to Caples' substrate with cm's eps_bar (None: back to native)."""
    sm = inf.model
    if not isinstance(sm, GemmaResSiteModel):
        raise TypeError("Caples' substrate exists for the residual family only")
    if cm is None:
        sm.caples = None
        return
    layers = [l for l, _ in bank.real_sites]
    if sorted(cm["err"]) != sorted(layers):
        raise ValueError("caples means layers %s != the bank's dictionary layers %s" % (sorted(cm["err"]), layers))
    for l in layers:
        e = cm["err"][l]
        if tuple(e.shape[1:]) != (bank.d_model,) or float(e[0].abs().max()) != 0.0:
            raise ValueError("eps_bar at layer %d: shape %s, BOS row max %g (theirs is exactly 0 at BOS)"
                             % (l, tuple(e.shape), float(e[0].abs().max())))
    sm.caples = {l: (bank.saes["resid"][l], cm["err"][l].to(inf.device)) for l in layers}


# ------------------------------------------------------------------------------------------------------- the engine
# Behavioural WCM on tasks = arena C's V3_LOGIT, unchanged (IMPLEMENTATION_PLAN §0): Z + A + C + R, no rank-keep,
# geometric lr. Not added: the x3.15 peak, warm-up, the censored term (validated on latent targets only).
MEANS_MODES = ("pooled", "per_position")
MODES = ("native", "caples")


def v3_logit() -> Dict:
    return dict(_wcm_marks().V3_LOGIT)


def _import_081():
    if str(D081) not in sys.path:
        sys.path.insert(0, str(D081))
    import task_common as C  # noqa: E402
    import engine as E  # noqa: E402
    return C, E


def make_engine(family: str, mode: str = "native", means: str = "pooled", caples_means: Optional[str] = None,
                caples_source: str = "theirs", device=None, dtype: Optional[str] = None, shared=None):
    """081's Engine on an arena-D family (arena C's MarksEngine pattern). Sites = the bank's dictionary sites. means:
    "pooled" = 081's [d_sae] means over every real token except BOS (bank.mean_skip_positions); "per_position" = [T,
    d_sae] means, A over the training clean prompts, C over the training patch prompts (arena C's), except in Caples
    mode, where C = Caples' per-position corrupted code mean from the caples_means file (`caples_source`). mode
    "caples": the gsres family's site model splices decode(c_hat) + eps_bar (the file's) at every dictionary site."""
    C, E = _import_081()
    if means not in MEANS_MODES:
        raise ValueError("means must be one of %s, got %r" % (MEANS_MODES, means))
    if mode not in MODES:
        raise ValueError("mode must be one of %s, got %r" % (MODES, mode))
    if mode == "caples" and means != "per_position":
        # Daniel 2026-11-05: Caples mode replicates their substrate with no substitution of ours; their mean ablation
        # is per position (get_sae_means, [T, d_sae]), so a pooled C would be our computation in its place
        raise ValueError("--mode caples is per-position only (their fill is a per-position mean): use --means "
                         "per_position; pooled means are for --mode native")
    V3 = v3_logit()

    class GemmaEngine(E.Engine):
        def __init__(self):
            if str(D059) not in sys.path:
                sys.path.insert(0, str(D059))
            import protocol_harness as H  # noqa: E402
            from config import config
            self.model = family
            self.inf, self.bank = shared if shared is not None else load_model(family, device, dtype)
            self.dev = self.inf.device
            config.discovery.probe_batch_size = 4
            self.H, self.R = H, None
            self.kinds = list(self.bank.kinds)
            self.sites = list(self.bank.real_sites)
            self.family, self.substrate, self.means_mode = family, mode, means
            self.cm = None
            if mode == "caples":
                if not caples_means:
                    raise ValueError("--mode caples needs --caples-means (caples_means.py's .pt)")
                self.cm = load_caples_means(caples_means, caples_source)
                apply_caples(self.inf, self.bank, self.cm)
            elif isinstance(self.inf.model, GemmaResSiteModel):
                apply_caples(self.inf, self.bank, None)
            self.c_fill = ("caples_" + caples_source) if mode == "caples" else "ours"

        def batch(self, task, contrast=None):
            B = super().batch(task, contrast)
            if self.cm is not None:                               # their mean set must be our first patch prompts
                mt = self.cm["mean_tokens"]
                ours = B["c_tokens"][:mt.shape[0]].cpu()
                if tuple(ours.shape) != tuple(mt.shape) or not torch.equal(ours, mt):
                    raise ValueError("caples means file %s was made on other prompts than this task's first %d "
                                     "patch prompts" % (self.cm["path"], mt.shape[0]))
            return B

        def engine_kwargs(self, B, *a, **kw):
            out = super().engine_kwargs(B, *a, **kw)
            out.update(V3)
            ct, cv = B["c_tokens"], B["c_valid"]
            n = B["n_train"]
            lens_c = set(int(x) for x in cv.sum(1).tolist())
            lens_p = set(int(x) for x in B["valid"][:n].sum(1).tolist())
            if len(lens_c) != 1 or lens_c != lens_p or int(ct.shape[1]) != int(B["tokens"].shape[1]):
                raise ValueError("R donors need one token length equal to the clean prompts' (patch %s, clean %s)"
                                 % (sorted(lens_c), sorted(lens_p)))
            out["resample_tokens"] = ct
            last = self.sites[-1]                                 # the seed slot must hold a real dictionary
            out["seed_layer"], out["seed_kind"] = int(last[0]), str(last[1])
            return out

        def means(self, B):
            if "means" in B:
                return B["means"]
            if means == "pooled":
                return super().means(B)
            n = B["n_train"]
            T = int(B["tokens"].shape[1])
            for nm, v in (("clean", B["valid"]), ("patch", B["c_valid"])):
                if not bool(v.all()) or int(v.shape[1]) != T:
                    raise ValueError("per-position means need every prompt (clean and patch) of one length T=%d: "
                                     "the %s prompts are padded" % (T, nm))
            A = self.site_means_pos(B["tokens"][:n])
            if self.cm is not None:
                Cm = {}
                for s in self.sites:
                    c = self.cm["code"][s[0]]
                    if tuple(c.shape) != (T, self.bank.d_sae):
                        raise ValueError("caples code mean at %s has shape %s, need (%d, %d)"
                                         % (s, tuple(c.shape), T, self.bank.d_sae))
                    Cm[s] = c.to(self.dev)
            else:
                Cm = self.site_means_pos(B["c_tokens"])
            B["means"] = dict(A=A, C=Cm)
            return B["means"]

        @torch.no_grad()
        def site_means_pos(self, tokens):
            """{site: [T, d_sae]} the mean dense code at each position over the rows of `tokens` (all real)."""
            from sae.dense import sparse_topk_to_dense
            k2i = {k: i for i, k in enumerate(self.kinds)}
            want = set(self.sites)
            tot, cnt = {}, 0
            for s0 in range(0, int(tokens.shape[0]), E.MEANS_BS):
                tk = tokens[s0:s0 + E.MEANS_BS]

                def hook(layer, acts):
                    for kd in self.kinds:
                        s = (layer, kd)
                        if s in want:
                            ta, ti = self.bank.encode(acts[k2i[kd]], kd, layer)
                            d = sparse_topk_to_dense(ta, ti, self.bank.d_sae, dtype=torch.float32).double().sum(0)
                            tot[s] = d if s not in tot else tot[s] + d

                self.inf.forward(tk, activations_callback=hook, return_activations=False, tokenize_final=False)
                cnt += int(tk.shape[0])
            return {s: (v / cnt).float() for s, v in tot.items()}

    return GemmaEngine()


def sizes(alphas: Dict, prompt_len: int) -> Dict:
    """Distinct (site, latent) members, and the per-position count a position-free member costs in Caples' units
    (members x (L - 1): BOS is never a node), README §3 / spec §9.3."""
    n = sum(len(d) for d in alphas.values())
    return dict(n_latents=n, native_nodes=n * (prompt_len - 1),
                latents_per_site={"%s_%d" % (s[1], s[0]): len(d) for s, d in sorted(alphas.items())})


def run(tasks: Sequence[str], lams: Sequence[float], family: str = FAM_RES, data: Optional[str] = None,
        mode: str = "native", means: str = "pooled", steps: int = 400, out: Path = OUT_DEFAULT, arm: str = "W",
        lr: Optional[float] = None, eng=None, log=print, caples_means: Optional[str] = None,
        caples_source: str = "theirs", task_kw: Optional[Dict] = None, tok=None) -> List[Dict]:
    """Fit + score each (task, lambda), resumable per (task, data, lambda, steps, means, mode[, caples source]): a
    finished circuit JSON is reloaded, never refitted, and a row already in eval.jsonl is not rescored. STOP_AT /
    STOP_EPOCH (env) stop before a new fit. arena C's wcm_marks.run, with the substrate in the key."""
    WM = _wcm_marks()
    C, _ = _import_081()
    data = data or ("sfc" if SPECS.get(family, {}).get("kind") == "tc" else "caples")
    started = _dt.datetime.now()
    stop_at = os.environ.get("STOP_AT")
    V3 = v3_logit()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vf = out / "WCM_VERSION"
    if vf.exists():                    # one lr schedule per --out (the resume keys do not carry it)
        lines = vf.read_text().splitlines()
        have = (json.loads(lines[1]).get("lr_schedule") or "constant") if len(lines) > 1 else "constant"
        if have != V3.get("lr_schedule", "constant"):
            raise RuntimeError("%s holds fits with lr_schedule %s; this run uses %s: use another --out"
                               % (out, have, V3.get("lr_schedule", "constant")))
    vf.write_text(WM.WCM_TAG + "\n" + json.dumps(V3) + "\n")
    eng = eng or make_engine(family, mode, means, caples_means, caples_source)
    mode, means = eng.substrate, eng.means_mode
    if mode == "caples" and data != "caples":
        raise ValueError("--mode caples needs --data caples (their filter, batches and mean set)")
    sub = mode if mode == "native" else "caples_" + eng.cm["source"]
    ev_path = out / "eval.jsonl"
    done = set()
    if ev_path.exists():
        for line in open(ev_path):
            try:
                r = json.loads(line)
                done.add((r["family"], r["task"], r["data_mode"], r["arm"], r["lam"], r["steps"], r["means"],
                          r["substrate"]))
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                pass
    lr = C.PROD_LR if lr is None else float(lr)
    rows = []
    for task in tasks:
        T = build_task(task, data, tok=tok or tokenizer(_F.FAMILIES[family].tokenizer_id), family=family,
                       **(task_kw or {}))
        if not T["split"]["held"]:
            log("%s %s: no held-out item; skipped" % (task, data))
            continue
        B = eng.batch(T)
        plen = T["prompt_len_train"][0]
        for lam in lams:
            key = (family, TASK_NAME[task], data, arm, float(lam), int(steps), means, sub)
            if key in done:
                log("skip (scored) %s" % (key,))
                continue
            cpath = out / "circuits" / family / task / data / sub / means / ("%s_%g_s%d.json" % (arm, lam, steps))
            t_fit, peak, prov = 0.0, None, {}
            if cpath.exists():
                rec = json.loads(cpath.read_text())
                alphas = C.alphas_from_json(rec["alphas"])
                t_fit, peak, prov = rec.get("t_fit", 0.0), rec.get("peak_mb"), rec.get("fit", {})
                log("loaded %s" % cpath)
            else:
                if WM.stop_passed(stop_at, started):
                    log("STOP_AT %s passed: no new fit" % stop_at)
                    return rows
                alphas, prov, t_fit, peak = eng.fit(B, arm, float(lam), lr, int(steps))
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cm = None if eng.cm is None else {k: eng.cm[k] for k in ("path", "sha", "source", "meta")}
                cpath.write_text(json.dumps(dict(task=task, data_mode=data, arm=arm, lam=float(lam), steps=int(steps),
                                                 means=means, substrate=sub, c_fill=eng.c_fill, family=family,
                                                 caples_means=cm, alphas=C.alphas_to_json(alphas),
                                                 t_fit=round(t_fit, 2), peak_mb=peak, fit=WM._small(prov),
                                                 curve=WM.curve_summary(prov), wcm=V3, wcm_tag=WM.WCM_TAG),
                                            default=str) + "\n")
                prov = WM._small(prov)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            ts = time.time()
            sc = eng.score(B, alphas)
            t_score = time.time() - ts
            peak_s = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
            row = dict(task=TASK_NAME[task], sfc_task=task, data_mode=data, arm=arm, lam=float(lam), steps=int(steps),
                       means=means, substrate=sub, c_fill=eng.c_fill, family=family, wcm="v3",
                       n_train=T["split_meta"]["n_train"], n_held=T["split_meta"]["n_held"], prompt_len=plen,
                       overlap=T["overlap"], **sizes(alphas, plen), t_fit=round(t_fit, 2), peak_fit_mb=peak,
                       t_score=round(t_score, 2), peak_score_mb=peak_s, circuit=str(cpath.relative_to(out)))
            row.update({k: v for k, v in sc.items() if not isinstance(v, (dict, list)) or k == "per_site"})
            row["fit_stats"] = {k: prov.get(k) for k in ("n_kept", "loss_final", "sparse_fill_calls", "resample_weight",
                                                         "resample_n_donors", "resample_draws", "resample_fill_calls",
                                                         "rank_weight", "rank_mode") if k in prov}
            with open(ev_path, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            rows.append(row)
            log("%s %s %s %s %s lam %g: %d latents | task Z %s A %s C %s nec %s | fit %.1fs score %.1fs"
                % (family, task, data, sub, means, lam, row["n_latents"], _f(row.get("task_free0")),
                   _f(row.get("task_freeM_topk")), _f(row.get("task_freeN_topk")), _f(row.get("task_nec")), t_fit,
                   t_score))
    return rows


def _f(x):
    return "None" if x is None else "%.3f" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--family", required=True, choices=(FAM_PLT, FAM_RES))
    ap.add_argument("--tasks", default="rc")
    ap.add_argument("--lams", default="1e-3")
    ap.add_argument("--data", default=None, choices=DATA_MODES, help="default: sfc (plt), caples (gsres)")
    ap.add_argument("--mode", default="native", choices=MODES)
    ap.add_argument("--means", default="pooled", choices=MEANS_MODES)
    ap.add_argument("--caples-means", default=None, help="caples_means.py's .pt (required for --mode caples)")
    ap.add_argument("--caples-source", default="theirs", choices=CAPLES_FILLS)
    ap.add_argument("--n-held", type=int, default=N_HELD_CAPLES)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dtype", default=None, help="bank / model dtype (default float32; bfloat16 = labelled)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.mode == "caples" and a.family != FAM_RES:
        ap.error("--mode caples exists for %s only" % FAM_RES)
    if a.mode == "caples" and a.means != "per_position":
        ap.error("--mode caples is per-position only (their fill is a per-position mean): use --means per_position; "
                 "pooled means are for --mode native")
    out = Path(a.out) if a.out else OUT_DEFAULT / a.family
    data = a.data or ("sfc" if a.family == FAM_PLT else "caples")
    eng = make_engine(a.family, a.mode, a.means, a.caples_means, a.caples_source, dtype=a.dtype)
    run([t for t in a.tasks.split(",") if t], [float(x) for x in a.lams.split(",")], a.family, data, a.mode,
        a.means, a.steps, out, eng=eng, caples_means=a.caples_means, caples_source=a.caples_source,
        task_kw=dict(n_held=a.n_held) if data == "caples" else None)


if __name__ == "__main__":
    main()
