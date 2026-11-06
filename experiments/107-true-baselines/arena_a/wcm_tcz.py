"""Arena A (DAN-195): WCM's site adapter for GPT-2 small + Dunefsky transcoders (MLPs) + Kissane z-SAEs (attention),
the substrate of Ge et al. (A1) and Dunefsky et al. (A2). Experiment code: nothing under src/ is changed; like arena C's
wcm_marks.py it registers a family, a site model and an SAE bank at runtime. Design and the src proposals:
WCM_ADAPTER.md (this folder). STATUS 2026-11-03: written and CPU-tested only (test_wcm_tcz.py); no fit has run.

Family "gpt2-small-tcz", kinds ("z", "z2", "tc"), all 24,576 wide (src takes one d_sae per family):
  (L, z)   Kissane L, latents 0..24,575, on the merged-heads input of h[L].attn.c_proj (= TL hook_z flattened,
           head-major). Kissane trained on TL's defaults fold_ln=True + fold_value_biases=True, i.e. z_TL = z_HF - b_V'
           with b_V' = b_V + ln_1.bias @ W_V, TL's LN-folded value bias (measured 2026-11-03, ge_checks.py format:
           centred FVU 0.20-0.49, L0 3-21 in TL; with b_V kept FVU up to 441). So the bank stores decoder_bias =
           b_dec + b_V' (the HF-space centre): b_eff = b_enc - W_enc (b_dec + b_V') is Kissane's encoder on z_TL
           exactly, and decode() is the HF-space reconstruction. (2026-11-05 fix: the raw b_V alone, used before,
           left ln_1.bias @ W_V in the input; tl_value_bias.)
  (L, z2)  layers 5 and 7 only: Kissane L latents 24,576..49,151 (those two dictionaries are 49,152 wide), encoded from
           the SAME clean c_proj input; at the other 10 layers an inert NullSAE. ReLU is per latent and decode linear,
           so the two halves' edits add exactly: c_proj_in <- x + (y_z - x) + (y_z2 - x). Member ids map back as
           latent + 24,576 (orig_latent()).
  (L, tc)  Dunefsky transcoder L (035's LN-folded transcoders_ct/, which read the FULL ln_2 output, i.e. HF's MLP
           input): encoded from the MLP input; the edit lands on the MLP OUTPUT: the hook calls on_site(L, "tc", x_in)
           and adds (y - x_in) = (c_hat - c) W_dec to mlp(x_in). That is 035's mlp_out <- mlp_out + (c_hat - c) W_dec;
           the transcoder error passes through untouched. decoder_bias = b_dec_out and the encoder bias is
           pre-compensated so b_eff = b_enc_t exactly (decode() = the MLP-output reconstruction).
  k = d_sae (ReLU dictionaries; acts = the whole code, idx = an expanded arange), as arena C.

BOS (README section 5, a decision for Daniel): with ARENA_A_BOS=1 (default) the site model prepends <|endoftext|> inside
forward and hides position 0 from every site (on_site sees x[:, 1:], the BOS row stays clean, logits come back for the
caller's positions only). Both dictionary sets were trained with BOS at 0; this makes BOS a frozen live prefix (never a
member, never ablated, never a root), the Ge (A14) and Dunefsky (A15) convention. ARENA_A_BOS=0 = no BOS.

Scopes: ARENA_A_SCOPE=a1 (default; z + z2 + tc) or a2 (tc only: every z / z2 slot a NullSAE, attention live and
undecomposed: the Dunefsky node universe, dunefsky.md section 7.1).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from model import families as _F  # noqa: E402
from model import hf_adapters as _HA  # noqa: E402
from sae.public_bank import PublicSAEBank as _PublicSAEBank  # noqa: E402
from sae.topk_sae import SAE as _SAE  # noqa: E402

FAMILY = "gpt2-small-tcz"
HF_ID = "gpt2"
N_LAYER, D_MODEL, D_SAE, N_HEAD, D_HEAD = 12, 768, 24576, 12, 64
KINDS = ("z", "z2", "tc")
SAME_LAYER = {"z": (), "z2": (), "tc": ("z", "z2")}
WIDE = (5, 7)                                         # Kissane L5 / L7: 49,152 latents = z + z2
BOS_ID = 50256
HF_CACHE = Path("/home/daniel/.cache/huggingface/hub")
Z_SNAP = HF_CACHE / "models--ckkissane--attn-saes-gpt2-small-all-layers/snapshots/97e0da0e9182695e9813d15ea32184da83f5c874"
TC_CT = ROOT / "experiments" / "035-transcoder-compare" / "transcoders_ct"
TC_FILE_PICKLE = "final_sparse_autoencoder_gpt2-small_blocks.%d.ln2.hook_normalized_24576.pt"   # the @4f3ca345 sources


def bos_prefix() -> bool:
    return os.environ.get("ARENA_A_BOS", "1") == "1"


def scope() -> str:
    s = os.environ.get("ARENA_A_SCOPE", "a1")
    assert s in ("a1", "a2"), s
    return s


def real_sites(sc: Optional[str] = None) -> List[Tuple[int, str]]:
    sc = sc or scope()
    out = []
    for l in range(N_LAYER):
        if sc == "a1":
            out.append((l, "z"))
            if l in WIDE:
                out.append((l, "z2"))
        out.append((l, "tc"))
    return out


def orig_latent(kind: str, latent: int) -> Tuple[str, int]:
    """our (kind, latent) -> (dictionary, its own latent id): z2 is the upper half of Kissane L5 / L7"""
    return ("z", latent + D_SAE) if kind == "z2" else (kind, latent)


# ------------------------------------------------------------------------------------------- dictionaries
class ReluSAE(_SAE):
    """A ReLU dictionary in our SAE class, k = d_sae (as arena C's MarksSAE): encode returns the whole code and an
    expanded arange as the indices, so nothing is truncated."""

    def __init__(self, d_model: int = D_MODEL, d_sae: int = D_SAE, device=None):
        super().__init__(d_model, d_sae, k=d_sae, device=device if device is not None else torch.device("cpu"))
        self._ar: Dict = {}

    def _arange(self, like):
        a = self._ar.get(like.device)
        if a is None:
            a = torch.arange(self.d_sae, device=like.device)
            self._ar[like.device] = a
        return a.expand(*like.shape[:-1], self.d_sae)

    def encode(self, x):
        if x.dtype != self.encoder.weight.dtype:
            x = x.to(self.encoder.weight.dtype)
        acts = torch.relu(nn.functional.linear(x, self.encoder.weight, self._get_bias_eff()))
        return acts, self._arange(acts)


class NullSAE(nn.Module):
    """inert slot: code 0, decode 0, nothing can be edited there (arena C's NullSAE)"""

    null = True

    def __init__(self, device, dtype=torch.float32, d_model: int = D_MODEL, d_sae: int = D_SAE):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, d_sae
        self.device = self.decoder_device = device
        self.dtype = dtype

    def encode(self, x):
        z = torch.zeros(*x.shape[:-1], self.d_sae, device=x.device, dtype=self.dtype)
        return z, torch.arange(self.d_sae, device=x.device).expand_as(z)

    def decode(self, latents, ensure_device: bool = True, add_bias: bool = True):
        return latents.new_zeros(*latents.shape[:-1], self.d_model)

    def move_decoder_to_vram(self):
        return None

    def remove_decoder_from_vram(self, empty_cache: bool = True):
        return None


def z_file(layer: int) -> Path:
    hits = sorted(Z_SNAP.glob("gpt2-small_L%d_Hcat_z_*.pt" % layer))
    assert len(hits) == 1, hits
    return hits[0]


@lru_cache(maxsize=1)
def _gpt2_biases() -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    snap = next((HF_CACHE / "models--gpt2" / "snapshots").iterdir())
    return {k: v.clone() for k, v in load_file(str(snap / "model.safetensors")).items()
            if "c_attn.bias" in k or "c_attn.weight" in k or "ln_1.bias" in k}


def _gpt2_param(name: str) -> torch.Tensor:
    sd = _gpt2_biases()
    return sd[name] if name in sd else sd["transformer." + name]


def gpt2_b_V(layer: int) -> torch.Tensor:
    """HF c_attn bias, V third, flattened head-major [768] (the raw b_V, before TL's fold_ln)"""
    return _gpt2_param("h.%d.attn.c_attn.bias" % layer)[2 * D_MODEL:].float()


def tl_value_bias(layer: int) -> torch.Tensor:
    """TL's b_V under its default fold_ln=True, i.e. what fold_value_biases removes from hook_z [768]:
    b_V' = b_V + ln_1.bias @ W_V (fold_ln moves LN1's bias into the value bias). So z_TL = z_HF - b_V'.
    (Fixed 2026-11-05, agent arenaA2: the adapter first removed the raw b_V only, which leaves ln_1.bias @ W_V in the
    input: centred FVU 13-22 at L0-L2 on the smoke corpus, L0 65-163; with b_V' FVU 0.28-0.81, L0 3-23, as measured
    in TL by ge_checks.py format.)"""
    W_V = _gpt2_param("h.%d.attn.c_attn.weight" % layer)[:, 2 * D_MODEL:].float()     # Conv1D: [in 768, out 2304]
    return gpt2_b_V(layer) + _gpt2_param("h.%d.ln_1.bias" % layer).float() @ W_V


def load_z_state(kind: str, layer: int):
    """(W_E [24576, 768], b_E, W_D [768, 24576], b_D): the z half (kind z: latents 0..24,575; z2: 24,576..) with the
    HF-space centre b_D = b_dec + b_V' (b_V' = tl_value_bias: TL's LN-folded value bias)."""
    sd = torch.load(z_file(layer), map_location="cpu", weights_only=True)
    lo = D_SAE if kind == "z2" else 0
    sl = slice(lo, lo + D_SAE)
    W_enc, b_enc, W_dec, b_dec = sd["W_enc"].float(), sd["b_enc"].float(), sd["W_dec"].float(), sd["b_dec"].float()
    assert W_enc.shape[1] in (D_SAE, 2 * D_SAE) and (kind == "z" or W_enc.shape[1] == 2 * D_SAE)
    return W_enc[:, sl].T.contiguous(), b_enc[sl].contiguous(), W_dec[sl].T.contiguous(), b_dec + tl_value_bias(layer)


def load_tc_state(layer: int):
    """(W_E, b_E, W_D, b_D) from 035's transcoders_ct (encode: x @ W_enc.T + b_enc on the FULL ln_2 output, no
    centring; decode: f @ W_dec + b_dec, b_dec = the MLP-output bias). Our SAE subtracts decoder_bias before encoding,
    so the encoder bias is pre-compensated: encoder.bias = b_enc + W_enc b_dec gives b_eff = b_enc exactly."""
    from safetensors.torch import load_file
    sd = load_file(str(TC_CT / ("layer_%d.safetensors" % layer)))
    W_enc, b_enc, W_dec, b_dec = sd["W_enc"].float(), sd["b_enc"].float(), sd["W_dec"].float(), sd["b_dec"].float()
    return W_enc.contiguous(), (b_enc + W_enc @ b_dec).contiguous(), W_dec.T.contiguous(), b_dec


class TCZSAEBank(_PublicSAEBank):
    """PublicSAEBank for gpt2-small-tcz (any other family: the parent, unchanged)."""

    @property
    def real_sites(self) -> List[Tuple[int, str]]:
        return real_sites() if self.family == FAMILY else [(l, k) for l in range(self.n_layer) for k in self.kinds]

    def _load(self, kind: str, layer: int):
        if self.family != FAMILY:
            return super()._load(kind, layer)
        if kind not in self.kinds:
            raise KeyError("%s: unknown site kind %r" % (self.family, kind))
        target = self.layer_device_map[layer]
        if (layer, kind) not in real_sites():
            return NullSAE(target, self.dtype)
        W_E, b_E, W_D, b_D = load_tc_state(layer) if kind == "tc" else load_z_state(kind, layer)
        sae = ReluSAE(self.d_model, self.d_sae, device=torch.device("cpu"))
        with torch.no_grad():
            sae.encoder.weight.copy_(W_E)
            sae.encoder.bias.copy_(b_E)
            sae.decoder.weight.copy_(W_D)
            sae.decoder_bias.copy_(b_D)
        sae.to(device=target, dtype=self.dtype)
        sae.device = sae.decoder_device = target
        for p in sae.parameters():
            p.requires_grad_(False)
        sae._get_bias_eff()
        return sae


# ------------------------------------------------------------------------------------------- the site model
class TCZSiteModel(_HA.GPT2SiteModel):
    """GPT-2 with sites (L, z), (L, z2) on c_proj's input and (L, tc) read on the MLP input, written on the MLP output.
    With bos_prefix(), BOS is prepended inside forward and position 0 is hidden from every site."""

    def forward(self, idx, targets=None, input_pos=None, return_all_logits: bool = False):
        if targets is not None:
            raise NotImplementedError("TCZSiteModel: training loss is not supported")
        if not bos_prefix():
            return super().forward(idx, targets, input_pos, return_all_logits)
        bos = torch.full((idx.shape[0], 1), BOS_ID, dtype=idx.dtype, device=idx.device)
        out = self.hf(input_ids=torch.cat([bos, idx], 1), use_cache=False,
                      logits_to_keep=0 if return_all_logits else 1)
        return (out.logits[:, 1:] if return_all_logits else out.logits), None

    @staticmethod
    def _split(x):
        return (x[:, :1], x[:, 1:]) if bos_prefix() else (None, x)

    @staticmethod
    def _join(head, body):
        return body if head is None else torch.cat([head, body], 1)

    def _register(self, on_site) -> List:
        hs = []
        for l, block in enumerate(self.layers()):
            def zpre(_m, args, l=l):
                head, x = self._split(args[0])
                ya = on_site(l, "z", x)
                yb = on_site(l, "z2", x)
                if ya is None and yb is None:
                    return None
                new = x
                if ya is not None:
                    new = new + (ya - x)
                if yb is not None:
                    new = new + (yb - x)
                return (self._join(head, new),) + tuple(args[1:])

            def mlp_hook(_m, args, out, l=l):
                head, x_in = self._split(args[0])
                y = on_site(l, "tc", x_in)
                if y is None:
                    return None
                h_out, body = self._split(out)
                return self._join(h_out, body + (y - x_in))

            hs.append(block.attn.c_proj.register_forward_pre_hook(zpre))
            hs.append(block.mlp.register_forward_hook(mlp_hook))
        return hs


def register(patch_bank_dispatch: bool = False) -> str:
    if FAMILY not in _F.FAMILIES:
        _F.FAMILIES[FAMILY] = _F.ModelFamily(
            name=FAMILY, hf_model_id=HF_ID, tokenizer_id=HF_ID, n_layer=N_LAYER, d_model=D_MODEL, kinds=KINDS,
            same_layer_upstream=dict(SAME_LAYER),
            sae=_F.SAESource(fmt="tcz", d_sae=D_SAE, k=D_SAE, normalize=None, norm_eps=0.0, prepend_bos=True,
                             context_size=128, files=lambda kind, layer: (None, None, None)))
    _HA._SITE_MODELS[FAMILY] = TCZSiteModel
    if patch_bank_dispatch:
        import sae.public_bank as PB
        PB.PublicSAEBank = TCZSAEBank
    return FAMILY


def load_model(device=None, dtype: Optional[str] = None, preload: bool = True):
    """(site model, bank) on `device` (CPU by default here)."""
    register()
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model = _HA.load_site_model(FAMILY, device=dev, dtype=dtype)
    bank = TCZSAEBank(device=dev, family=FAMILY, preload=preload, dtype=dtype)
    return model, bank
