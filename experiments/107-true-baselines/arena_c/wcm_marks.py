"""Arena C (DAN-194): WCM (production engine, frozen WCM_VERSION=v3) on Pythia-70m-deduped + the Marks et al. SAEs,
the dictionaries the official Sparse Feature Circuits code uses. Design: WCM_ADAPTER.md (this folder).

Nothing under src/, experiments/082-*, paper/ or third_party/ is modified. This module registers, at import of
register(), a model family "pythia-70m-deduped-marks" (model.families.FAMILIES), its HF site model
(model.hf_adapters._SITE_MODELS) and its SAE bank (MarksSAEBank, a PublicSAEBank subclass; with
register(patch_bank_dispatch=True) also the target of SAEBank(...)'s family dispatch, for 062's driver).

  sites      (0, embed) = gpt_neox.embed_in output; (i, attn) = layers[i].attention output[0]; (i, mlp) =
             layers[i].mlp output; (i, resid) = layers[i] output[0]: SFC's 19 submodules. Slots (1..5, embed) are
             inert NullSAE placeholders (no dictionary; code 0; nothing can be edited there).
  encode     f = relu(W_E (x - b_D) + b_E), x_hat = W_D f + b_D, as dictionary_learning.AutoEncoder (decoder normalised
             as from_pretrained(normalize_decoder=True)); bank.k = d_sae: encode returns the whole code (no truncation)
  splice     x <- x + (c_hat - c) W_D^T: the SAE error is preserved, so an untouched circuit is the model itself
  tasks      SFC's SVA files, selected by a line-for-line re-implementation of their load_examples (seed-12 shuffle,
             single-token answers, equal clean / patch length, prefix-stable answers, min length only):
             data mode sfc = 100 discovery / 40 evaluation examples (their protocol); sfc_test_all and dedup are ours
  fit        081's engine path (Engine.fit: objective "logit", Z + C + A, gamma 0.25, free coefficients, gap norm,
             400 x 4) under behavioural v3 = Z + A + C + R, no rank-keep (2026-11-02): the sparse fill (ste), R with
             gamma_R 1.0 whose donors are the training patch prompts. See WCM_ADAPTER.md section 4.

CLI (WSL, repo root; GPU):
  ./.venv/bin/python experiments/107-true-baselines/arena_c/wcm_marks.py tasks              # selection + leak stats
  ./.venv/bin/python experiments/107-true-baselines/arena_c/wcm_marks.py run --tasks simple --lams 1e-3 \
        [--mode sfc] [--steps 400] [--sites noembed|ge2|all] [--means pooled|per_position] [--out DIR]
  env: STOP_AT=07:30 (hard stop: no new fit starts after it; per-circuit resumable), HELDOUT_N (081's)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
D081 = ROOT / "experiments" / "081-task-circuits"
D059 = ROOT / "experiments" / "059-context-pool"
FC = ROOT / "third_party" / "feature-circuits"
SFC_DATA = FC / "data"
DICT_DIR = ROOT / "experiments" / "030-cross-sae" / "dictionaries" / "pythia-70m-deduped"
OUT_DEFAULT = HERE / "out" / "wcm"

import torch  # noqa: E402
from torch import nn  # noqa: E402

from model import families as _F  # noqa: E402
from model import hf_adapters as _HA  # noqa: E402
from sae.public_bank import PublicSAEBank as _PublicSAEBank  # noqa: E402
from sae.topk_sae import SAE as _SAE  # noqa: E402

FAMILY = "pythia-70m-deduped-marks"
HF_ID = "EleutherAI/pythia-70m-deduped"
N_LAYER, D_MODEL, D_SAE = 6, 512, 32768
KINDS = ("embed", "attn", "mlp", "resid")
SAME_LAYER = {"embed": (), "attn": ("embed",), "mlp": ("embed",), "resid": ("embed", "attn", "mlp")}
REAL_SITES: List[Tuple[int, str]] = [(0, "embed")] + [(l, k) for l in range(N_LAYER) for k in ("attn", "mlp", "resid")]
SFC_TASKS = ("simple", "within_rc", "rc", "nounpp")
TASK_NAME = {"simple": "sva_simple", "within_rc": "sva_within_rc", "rc": "sva_across_rc", "nounpp": "sva_across_pp"}
DATA_MODES = ("sfc", "sfc_test_all", "dedup")
N_DISCOVERY, N_EVAL, SFC_SEED = 100, 40, 12          # circuit.py --num_examples 100; notebook n_examples=40; seed 12
# Behavioural WCM (Daniel, 2026-11-02): Z + A + C + R, no rank-keep. 062 driver.WCM_VERSIONS["v3"] (sparse fill, ste,
# gamma_R 1.0, k 1) without WCM_RANK. R's donors = the task's training patch prompts. A behavioural rank-keep (the answer
# token's clean vocabulary rank) was tried on 2026-11-01 and removed: it collapsed fits (out/ablate_terms: simple
# lambda 1e-5, Z 0.00 with it, 0.89 without), and the logit difference already scores the answer against its rival.
# lr_schedule "geometric" (Daniel, 2026-11-02 "if geometric is better then use it"; ours, a fitting-procedure choice):
# lr * 0.05**progress, the gate anneal's own shape, so the gate step lr / T stays constant. The constant lr blew fits up
# late, cosine early (WCM_ADAPTER.md section 8); geometric curves are monotone and pass contiguously. The constant-lr
# runs (out/wcm_tasks_zacr, ...) are superseded by out/wcm_tasks_geo; a run never mixes schedules in one --out.
V3_LOGIT = dict(train_fill="sparse", sparse_grad="ste", resample_weight=1.0, resample_k=1, rank_weight=0.0,
                lr_schedule="geometric")
WCM_TAG = ("behavioural v3 = Z + A + C + R, no rank-keep, geometric lr (2026-11-02); R donors = training patch "
           "prompts")


def sfc_name(site: Tuple[int, str]) -> str:
    l, k = site
    return "embed" if k == "embed" else "%s_%d" % (k, l)


def dict_path(kind: str, layer: int) -> Path:
    folder = {"embed": "embed", "attn": "attn_out_layer%d", "mlp": "mlp_out_layer%d",
              "resid": "resid_out_layer%d"}[kind]
    folder = folder if kind == "embed" else folder % layer
    return DICT_DIR / folder / "10_32768" / "ae.pt"


def _marks_files(kind: str, layer: int):
    return None, str(dict_path(kind, layer)), str(dict_path(kind, layer).with_name("config.json"))


# ------------------------------------------------------------------------------------------- dictionaries
class MarksSAE(_SAE):
    """A Marks et al. ReLU SAE in our SAE class. k = d_sae: encode returns the whole ReLU code with an expanded arange
    as the indices (no Top-K, so nothing is truncated; sparse_topk_to_dense of it is the code itself)."""

    def __init__(self, d_model: int = D_MODEL, d_sae: int = D_SAE, device=None):
        super().__init__(d_model, d_sae, k=d_sae, device=device if device is not None else torch.device("cpu"))
        self._ar: Dict = {}

    def _arange(self, like: torch.Tensor) -> torch.Tensor:
        a = self._ar.get(like.device)
        if a is None:
            a = torch.arange(self.d_sae, device=like.device)
            self._ar[like.device] = a
        return a.expand(*like.shape[:-1], self.d_sae)

    def encode(self, x: torch.Tensor):
        if x.dtype != self.encoder.weight.dtype:
            x = x.to(self.encoder.weight.dtype)
        acts = torch.relu(nn.functional.linear(x, self.encoder.weight, self._get_bias_eff()))
        return acts, self._arange(acts)


class NullSAE(nn.Module):
    """The inert slot (layer >= 1, "embed"): no dictionary. Code always 0, decode always 0, no parameters, so the error
    is the whole stream and no edit can enter. Has no encoder: using it as a seed fails loudly."""

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


def load_marks_state(kind: str, layer: int):
    """(W_E [d_sae, d], b_E [d_sae], W_D [d, d_sae], b_D [d], normalised?) exactly as AutoEncoder.from_pretrained(path,
    normalize_decoder=True) leaves them: decoder columns rescaled to unit norm (and the encoder compensated) only when
    they are not already allclose to 1."""
    sd = torch.load(dict_path(kind, layer), map_location="cpu", weights_only=True)
    W_E, b_E = sd["encoder.weight"].float(), sd["encoder.bias"].float()
    W_D, b_D = sd["decoder.weight"].float(), sd["bias"].float()
    norms = torch.norm(W_D, dim=0)
    if torch.allclose(norms, torch.ones_like(norms)):
        return W_E, b_E, W_D, b_D, False
    return W_E * norms[:, None], b_E * norms, W_D / norms, b_D, True


class MarksSAEBank(_PublicSAEBank):
    """PublicSAEBank for the Marks family (any other family: the parent, unchanged). 19 real dictionaries + 5 inert
    NullSAE slots; float32 unless dtype says otherwise.

    embed (default: env MARKS_EMBED == "1", i.e. False): whether (0, embed) holds its real dictionary. False puts a
    NullSAE there too, which takes the embedding out of WCM's scope (no member, no edit, no ablation): the default for
    the production latent path (driver_marks.py, data_pass_marks.py), because SFC's evaluator never ablates the
    embedding and zeroing it explodes the Z empty circuit (WCM_ADAPTER.md sections 6-7). The task engine and the tests
    load it (embed=True) and simply leave it out of the site list."""

    def __init__(self, *args, embed: Optional[bool] = None, **kwargs):
        self.normalised: Dict[Tuple[str, int], bool] = {}
        self.embed_real = (os.environ.get("MARKS_EMBED", "0") == "1") if embed is None else bool(embed)
        super().__init__(*args, **kwargs)

    @property
    def real_sites(self) -> List[Tuple[int, str]]:
        return list(REAL_SITES) if self.family == FAMILY else [
            (l, k) for l in range(self.n_layer) for k in self.kinds]

    def _load(self, kind: str, layer: int):
        if self.family != FAMILY:
            return super()._load(kind, layer)
        if kind not in self.kinds:
            raise KeyError("%s: unknown site kind %r" % (self.family, kind))
        target = self.layer_device_map[layer]
        if kind == "embed" and (layer > 0 or not self.embed_real):
            return NullSAE(target, self.dtype)
        W_E, b_E, W_D, b_D, normed = load_marks_state(kind, layer)
        self.normalised[(kind, int(layer))] = normed
        sae = MarksSAE(self.d_model, self.d_sae, device=torch.device("cpu"))
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
class MarksPythiaSiteModel(_HA.HFSiteModel):
    """GPT-NeoX (parallel residual) with SFC's 19 sites: embed, and attn / mlp / resid per layer. The inert slot
    (l >= 1, embed) is called with the layer's input (a forward pre-hook) so per-layer activation tuples are complete;
    its return value is discarded (there is no module output to replace)."""

    def __init__(self, hf_model: nn.Module, family) -> None:
        super().__init__(hf_model, family)
        if not bool(getattr(hf_model.config, "use_parallel_residual", False)):
            raise ValueError("MarksPythiaSiteModel assumes use_parallel_residual=True")

    def _register(self, on_site) -> List:
        gx = self.hf.gpt_neox
        hs = [gx.embed_in.register_forward_hook(_HA._site_hook(on_site, 0, "embed"))]
        for l, layer in enumerate(gx.layers):
            if l > 0:
                def pre(_m, args, kwargs, _l=l):
                    on_site(_l, "embed", args[0] if args else kwargs["hidden_states"])   # inert: result discarded
                hs.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            hs.append(layer.attention.register_forward_hook(_HA._site_hook(on_site, l, "attn")))
            hs.append(layer.mlp.register_forward_hook(_HA._site_hook(on_site, l, "mlp")))
            hs.append(layer.register_forward_hook(_HA._site_hook(on_site, l, "resid")))
        return hs


def register(patch_bank_dispatch: bool = False) -> str:
    """Add the family to model.families.FAMILIES and its site model to hf_adapters._SITE_MODELS (idempotent). With
    patch_bank_dispatch, SAEBank(...) (sae.bank's family dispatch, which reads sae.public_bank.PublicSAEBank at call
    time) builds MarksSAEBank, which is the parent class for every other family."""
    if FAMILY not in _F.FAMILIES:
        _F.FAMILIES[FAMILY] = _F.ModelFamily(
            name=FAMILY, hf_model_id=HF_ID, tokenizer_id=HF_ID, n_layer=N_LAYER, d_model=D_MODEL, kinds=KINDS,
            same_layer_upstream=dict(SAME_LAYER),
            sae=_F.SAESource(fmt="marks", d_sae=D_SAE, k=D_SAE, normalize=None, norm_eps=0.0, prepend_bos=False,
                             context_size=128, files=_marks_files))
    _HA._SITE_MODELS[FAMILY] = MarksPythiaSiteModel
    if patch_bank_dispatch:
        import sae.public_bank as PB
        PB.PublicSAEBank = MarksSAEBank
    return FAMILY


def load_model(device=None, dtype: Optional[str] = None):
    """(inference, bank) on `device`: the HF model through our adapter, and every dictionary preloaded."""
    register()
    from config import config
    from model.inference import Inference
    config.model.family = FAMILY
    config.model.dtype = dtype
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf = Inference(dev, family=FAMILY)
    bank = MarksSAEBank(device=dev, family=FAMILY, preload=True, dtype=dtype, embed=True)
    return inf, bank


# ------------------------------------------------------------------------------------------- SFC's data, re-implemented
_TOK = {}


def tokenizer():
    if "tok" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["tok"] = AutoTokenizer.from_pretrained(HF_ID, local_files_only=True)
    return _TOK["tok"]


def load_examples(path, n_examples: Optional[int], tok=None, use_min_length_only: bool = False,
                  max_length: Optional[int] = None, seed: int = SFC_SEED):
    """SFC's data_loading_utils.load_examples @7fbd82b, line for line, with the HF tokenizer in place of nnsight's
    model.tokenizer (the same AutoTokenizer). n_examples=None (ours) returns every qualifying record instead of
    stopping at n; with an n the function, like theirs, returns None when fewer than n qualify."""
    tok = tok or tokenizer()
    with open(path, "r") as f:
        dataset_items = f.readlines()
    random.Random(seed).shuffle(dataset_items)
    examples = []
    if use_min_length_only:
        min_length = float("inf")
    for line in dataset_items:
        data = json.loads(line)
        clean_prefix = tok(data["clean_prefix"]).input_ids
        patch_prefix = tok(data["patch_prefix"]).input_ids
        clean_answer = tok(data["clean_answer"]).input_ids
        patch_answer = tok(data["patch_answer"]).input_ids
        clean_full = tok(data["clean_prefix"] + data["clean_answer"]).input_ids
        patch_full = tok(data["patch_prefix"] + data["patch_answer"]).input_ids
        if clean_answer[0] == tok.bos_token_id:
            clean_answer = clean_answer[1:]
        if patch_answer[0] == tok.bos_token_id:
            patch_answer = patch_answer[1:]
        if len(clean_answer) != 1 or len(patch_answer) != 1:
            continue
        if len(clean_prefix) != len(patch_prefix):
            continue
        if clean_prefix + clean_answer != clean_full:
            continue
        if patch_prefix + patch_answer != patch_full:
            continue
        if max_length is not None and len(clean_prefix) > max_length:
            continue
        if use_min_length_only:
            if (length := len(clean_prefix)) < min_length:
                examples = []
                min_length = length
            elif length > min_length:
                continue
        examples.append(data)
        if n_examples is not None and len(examples) >= n_examples:
            return examples
    return examples if n_examples is None else None


def _row(i: int, e: Dict, tok) -> Dict:
    t = tok(e["clean_prefix"]).input_ids
    c = tok(e["patch_prefix"]).input_ids
    return dict(id=i, group=e["clean_prefix"], prompt=e["clean_prefix"], tokens=t, roles=["other"] * len(t),
                cf_prompt=e["patch_prefix"], cf_tokens=c, cf_roles=["other"] * len(c),
                # SFC's answer index rule: tokenizer(answer).input_ids[-1] (circuit.py:546, ablation.py:165)
                good=[tok(e["clean_answer"]).input_ids[-1]], bad=[tok(e["patch_answer"]).input_ids[-1]],
                meta=dict(case=e.get("case"), clean_answer=e["clean_answer"], patch_answer=e["patch_answer"]))


def build_task(task: str, mode: str = "sfc", tok=None) -> Dict:
    """One SVA task in 081's task schema (tasks.py docstring), training rows first. Modes: sfc = SFC's 100 discovery
    examples (<task>_train.json) and 40 evaluation examples (<task>_test.json), both min length; sfc_test_all [ours] =
    every min-length test record as held-out; dedup [ours] = the min-length test records whose clean_prefix is not the
    prefix of any of the 100 discovery examples (what the fit saw; a train-FILE-disjoint set is empty for simple,
    within_rc and rc). Duplicates inside SFC's selections are kept (theirs are too)."""
    if mode not in DATA_MODES:
        raise ValueError("mode must be one of %s, got %r" % (DATA_MODES, mode))
    tok = tok or tokenizer()
    tr_path, te_path = SFC_DATA / ("%s_train.json" % task), SFC_DATA / ("%s_test.json" % task)
    disc = load_examples(tr_path, N_DISCOVERY, tok, use_min_length_only=True)
    if disc is None:
        raise ValueError("%s: fewer than %d min-length training records" % (task, N_DISCOVERY))
    ev40 = load_examples(te_path, N_EVAL, tok, use_min_length_only=True)
    ev_all = load_examples(te_path, None, tok, use_min_length_only=True)
    train_prefixes = {json.loads(l)["clean_prefix"] for l in open(tr_path)}
    disc_keys = {(e["clean_prefix"], e["clean_answer"]) for e in disc}
    if mode == "sfc":
        held = ev40
    elif mode == "sfc_test_all":
        held = ev_all
    else:
        disc_prefixes = {d["clean_prefix"] for d in disc}
        held = [e for e in ev_all if e["clean_prefix"] not in disc_prefixes]
    rows = [_row(i, e, tok) for i, e in enumerate(list(disc) + list(held))]
    L = {len(r["tokens"]) for r in rows[:len(disc)]}
    overlap = dict(
        eval40_in_discovery=sum((e["clean_prefix"], e["clean_answer"]) in disc_keys for e in (ev40 or [])),
        eval40_prefix_in_discovery=sum(e["clean_prefix"] in {d["clean_prefix"] for d in disc} for e in (ev40 or [])),
        eval40_prefix_in_train_file=sum(e["clean_prefix"] in train_prefixes for e in (ev40 or [])),
        test_min_len_records=len(ev_all), test_min_len_prefix_not_in_train=sum(
            e["clean_prefix"] not in train_prefixes for e in ev_all),
        test_min_len_prefix_not_in_discovery=sum(
            e["clean_prefix"] not in {d["clean_prefix"] for d in disc} for e in ev_all),
        discovery_unique_items=len(disc_keys))
    return dict(name=TASK_NAME[task], sfc_task=task, family="sva", data_mode=mode, model=FAMILY,
                metric=dict(task="logit_diff", obj="logit_diff"), logit_metric="token", answer_mode="token",
                vac=dict(task=0.1, obj=0.1), contrast="original",
                split=dict(train=list(range(len(disc))), held=list(range(len(disc), len(rows)))),
                split_meta=dict(rule="SFC load_examples(seed 12, min length): %d discovery from %s, held-out %s (%s)"
                                % (len(disc), tr_path.name, mode, te_path.name), n_train=len(disc), n_held=len(held)),
                prompt_len_train=sorted(L), prompt_len_held=sorted({len(r["tokens"]) for r in rows[len(disc):]}),
                overlap=overlap, rows=rows)


# ------------------------------------------------------------------------------------------- the engine
def _import_081():
    """081's task_common and engine modules (engine.py sets OUT / EVAL_SPLIT / CTR_SOURCE defaults in os.environ at
    import, so it is imported only here, never by register())."""
    if str(D081) not in sys.path:
        sys.path.insert(0, str(D081))
    import task_common as C  # noqa: E402
    import engine as E  # noqa: E402
    return C, E


SITE_SCOPES = ("noembed", "ge2", "all")
MEANS_MODES = ("pooled", "per_position")


def scope_sites(sites: str) -> List[Tuple[int, str]]:
    """noembed (default): the 18 attn / mlp / resid sites (the embedding is out of scope: SFC's evaluator never
    ablates it, and zeroing it explodes the Z empty circuit, WCM_ADAPTER.md section 7); ge2: layers >= 2, 12 sites
    (SFC's evaluated scope); all: the 19 SFC sites incl. embed (kept for diagnostics)."""
    if sites == "noembed":
        return [s for s in REAL_SITES if s[1] != "embed"]
    if sites == "ge2":
        return [s for s in REAL_SITES if s[1] != "embed" and s[0] >= 2]
    if sites == "all":
        return list(REAL_SITES)
    raise ValueError("sites must be one of %s, got %r" % (SITE_SCOPES, sites))


def make_engine(sites: str = "noembed", dtype: Optional[str] = None, device=None, means: str = "pooled",
                shared=None):
    """081's Engine on this substrate: the same batch / means / fit / score code, with our model, our bank, the site
    scope (scope_sites) and v3 on the logit objective. means: "pooled" = the protocol's [d_sae] A / C means over every
    real prompt token (081's Engine.means); "per_position" [ours: an adaptation for fixed-length templates, matching
    SFC's per-position mean ablation] = [T, d_sae] means, A over the training clean prompts and C over the training
    patch prompts, position by position (every prompt of a task has the same length T: asserted). Either way the SAME
    tensors go to the fit (floor_means / pos_floor_means) and to the scorer (means_A / means_C)."""
    C, E = _import_081()
    if means not in MEANS_MODES:
        raise ValueError("means must be one of %s, got %r" % (MEANS_MODES, means))

    class MarksEngine(E.Engine):
        def __init__(self):                                   # Engine.__init__'s public-family branch, our objects
            if str(D059) not in sys.path:
                sys.path.insert(0, str(D059))
            import protocol_harness as H  # noqa: E402
            from config import config
            self.model = FAMILY
            # shared=(inference, bank): reuse loaded objects (two means modes in one process); else load
            self.inf, self.bank = shared if shared is not None else load_model(device, dtype)
            self.dev = self.inf.device
            config.discovery.probe_batch_size = 4
            self.H, self.R = H, None
            self.kinds = list(KINDS)
            self.sites = scope_sites(sites)
            self.site_scope = sites
            self.means_mode = means

        def engine_kwargs(self, B, *a, **kw):
            out = super().engine_kwargs(B, *a, **kw)
            out.update(V3_LOGIT)
            # R's donors: the training patch prompts (the C mean's pool). Donor rows replace the clean rows position
            # by position, so they must have the clean prompts' token length (SFC's min-length sets: one length).
            ct, cv = B["c_tokens"], B["c_valid"]
            n = B["n_train"]
            lens_c = set(int(x) for x in cv.sum(1).tolist())
            lens_p = set(int(x) for x in B["valid"][:n].sum(1).tolist())
            if len(lens_c) != 1 or lens_c != lens_p or int(ct.shape[1]) != int(B["tokens"].shape[1]):
                raise ValueError("R donors need one token length equal to the clean prompts' (patch %s, clean %s)"
                                 % (sorted(lens_c), sorted(lens_p)))
            out["resample_tokens"] = ct
            # warm_path.py's λ path: {warm_state, return_state} for the next fit only ({} = the run's cold start)
            out.update(getattr(self, "extra_kw", None) or {})
            return out

        def means(self, B):
            if means == "pooled" or "means" in B:
                return super().means(B)
            n = B["n_train"]
            T = int(B["tokens"].shape[1])
            for nm, v in (("clean", B["valid"]), ("patch", B["c_valid"])):
                if not bool(v.all()) or int(v.shape[1]) != T:
                    raise ValueError("per-position means need every prompt (clean and patch) of one length T=%d: "
                                     "the %s prompts are padded" % (T, nm))
            B["means"] = dict(A=self.site_means_pos(B["tokens"][:n]), C=self.site_means_pos(B["c_tokens"]))
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

    return MarksEngine()


def sizes(alphas: Dict, prompt_len: int) -> Dict:
    """Size in both units (107 rule 6): distinct (site, latent) members; SFC's native (position, latent) count a
    position-free member costs under SFC's evaluator (members x prompt length), over all sites and at layers >= 2."""
    n = sum(len(d) for d in alphas.values())
    n_ge2 = sum(len(d) for s, d in alphas.items() if s[1] != "embed" and s[0] >= 2)
    return dict(n_latents=n, n_latents_ge2=n_ge2, sfc_nodes_all=n * prompt_len, sfc_nodes_ge2=n_ge2 * prompt_len,
                sfc_error_nodes=0, latents_per_site={sfc_name(s): len(d) for s, d in sorted(alphas.items())})


def _small(prov: Dict) -> Dict:
    out = {}
    for k, v in prov.items():
        try:
            s = json.dumps(v)
        except (TypeError, ValueError):
            continue
        if len(s) <= 2000 and k not in ("amp_kept", "curve"):
            out[k] = v
    return out


def curve_summary(prov: Dict) -> Dict:
    """First / middle / last value of every training-curve series (loss, penalty, rank, n_members, per-term)."""
    cv = prov.get("curve") or {}
    out = {}

    def pick(v):
        if not isinstance(v, list) or not v:
            return None
        return dict(first=v[0], mid=v[len(v) // 2], last=v[-1], n=len(v))

    for k, v in cv.items():
        if isinstance(v, dict):
            for t, s in v.items():
                out["%s.%s" % (k, t)] = pick(s)
        else:
            out[k] = pick(v)
    return out


def stop_passed(stop_at: Optional[str], started: _dt.datetime) -> bool:
    """True once the wall clock passes STOP_AT (HH:MM; the next occurrence after the process started), or, when the env
    STOP_EPOCH is set (run_wcm_tasks.sh fixes it once at launch, so later processes keep the same deadline), once
    time.time() passes it."""
    if os.environ.get("STOP_EPOCH"):
        return time.time() >= float(os.environ["STOP_EPOCH"])
    if not stop_at:
        return False
    h, m = (int(x) for x in stop_at.split(":"))
    stop = started.replace(hour=h, minute=m, second=0, microsecond=0)
    if stop <= started:
        stop += _dt.timedelta(days=1)
    return _dt.datetime.now() >= stop


def run(tasks: Sequence[str], lams: Sequence[float], mode: str = "sfc", steps: int = 400, sites: str = "noembed",
        out: Path = OUT_DEFAULT, arm: str = "W", lr: Optional[float] = None, eng=None, log=print,
        means: str = "pooled") -> List[Dict]:
    """Fit + score each (task, lambda), resumable per circuit: a finished circuit JSON is reloaded, never refitted, and
    an (identical) eval row already in eval.jsonl is not rescored. STOP_AT (env) stops before a new fit."""
    C, _ = _import_081()
    started = _dt.datetime.now()
    stop_at = os.environ.get("STOP_AT")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vf = out / "WCM_VERSION"
    if vf.exists():                    # one lr schedule per --out (the resume keys do not carry it)
        lines = vf.read_text().splitlines()
        have = (json.loads(lines[1]).get("lr_schedule") or "constant") if len(lines) > 1 else "constant"
        if have != V3_LOGIT.get("lr_schedule", "constant"):
            raise RuntimeError("%s holds fits with lr_schedule %s; this run uses %s: use another --out"
                               % (out, have, V3_LOGIT.get("lr_schedule", "constant")))
    vf.write_text(WCM_TAG + "\n" + json.dumps(V3_LOGIT) + "\n")
    eng = eng or make_engine(sites, means=means)
    sites, means = eng.site_scope, eng.means_mode
    ev_path = out / "eval.jsonl"
    done = set()
    if ev_path.exists():
        for line in open(ev_path):
            try:
                r = json.loads(line)
                done.add((r["task"], r["data_mode"], r["arm"], r["lam"], r["steps"], r["sites"], r.get("means")))
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                pass
    lr = C.PROD_LR if lr is None else float(lr)
    rows = []
    for task in tasks:
        T = build_task(task, mode)
        if not T["split"]["held"]:
            log("%s %s: no held-out item (e.g. dedup on simple: every test prefix is a discovery prefix); skipped"
                % (task, mode))
            continue
        B = eng.batch(T)
        plen = T["prompt_len_train"][0]
        for lam in lams:
            key = (TASK_NAME[task], mode, arm, float(lam), int(steps), sites, means)
            if key in done:
                log("skip (scored) %s" % (key,))
                continue
            cpath = out / "circuits" / task / mode / sites / means / ("%s_%g_s%d.json" % (arm, lam, steps))
            t_fit, peak, prov = 0.0, None, {}
            if cpath.exists():
                rec = json.loads(cpath.read_text())
                alphas = C.alphas_from_json(rec["alphas"])
                t_fit, peak, prov = rec.get("t_fit", 0.0), rec.get("peak_mb"), rec.get("fit", {})
                log("loaded %s" % cpath)
            else:
                if stop_passed(stop_at, started):
                    log("STOP_AT %s passed: no new fit" % stop_at)
                    return rows
                alphas, prov, t_fit, peak = eng.fit(B, arm, float(lam), lr, int(steps))
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps(dict(task=task, data_mode=mode, arm=arm, lam=float(lam), steps=int(steps),
                                                 sites=sites, means=means, family=FAMILY,
                                                 alphas=C.alphas_to_json(alphas),
                                                 t_fit=round(t_fit, 2), peak_mb=peak, fit=_small(prov),
                                                 curve=curve_summary(prov), wcm=V3_LOGIT, wcm_tag=WCM_TAG)) + "\n")
                prov = _small(prov)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            ts = time.time()
            sc = eng.score(B, alphas)
            t_score = time.time() - ts
            peak_s = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
            row = dict(task=TASK_NAME[task], sfc_task=task, data_mode=mode, arm=arm, lam=float(lam), steps=int(steps),
                       sites=sites, means=means, family=FAMILY, wcm="v3", n_train=T["split_meta"]["n_train"],
                       n_held=T["split_meta"]["n_held"], prompt_len=plen, overlap=T["overlap"],
                       **sizes(alphas, plen), t_fit=round(t_fit, 2), peak_fit_mb=peak, t_score=round(t_score, 2),
                       peak_score_mb=peak_s, circuit=str(cpath.relative_to(out)))
            row.update({k: v for k, v in sc.items() if not isinstance(v, (dict, list)) or k == "per_site"})
            row["fit_stats"] = {k: prov.get(k) for k in ("n_kept", "loss_final", "sparse_fill_calls", "resample_weight",
                                                         "resample_n_donors", "resample_draws", "resample_fill_calls",
                                                         "rank_weight", "rank_mode") if k in prov}
            if cpath.exists():
                cv = json.loads(cpath.read_text()).get("curve") or {}
                row["curve_last"] = {k: (v or {}).get("last") for k, v in cv.items()
                                     if k in ("rank", "terms.zero", "terms.floor", "terms.pos", "terms.resample")}
            with open(ev_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            rows.append(row)
            log("%s %s %s %s lam %g: %d latents (>=2: %d) | task Z %s A %s C %s nec %s | fit %.1fs score %.1fs | "
                "last: %s" % (task, mode, sites, means, lam, row["n_latents"], row["n_latents_ge2"],
                              _f(row.get("task_free0")), _f(row.get("task_freeM_topk")), _f(row.get("task_freeN_topk")),
                              _f(row.get("task_nec")), t_fit, t_score,
                              {k: (None if v is None else round(v, 4)) for k, v in row.get("curve_last", {}).items()}))
    return rows


def _f(x):
    return "None" if x is None else "%.3f" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tasks", "run"))
    ap.add_argument("--tasks", default=",".join(SFC_TASKS))
    ap.add_argument("--lams", default="1e-3")
    ap.add_argument("--mode", default="sfc", choices=DATA_MODES)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sites", default="noembed", choices=SITE_SCOPES)
    ap.add_argument("--means", default="pooled", choices=MEANS_MODES)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--lr_schedule", default=None, choices=(None, "constant", "cosine", "geometric"),
                    help="[ours, diagnostic] override the run's lr schedule (constant); use a separate --out")
    a = ap.parse_args()
    if a.lr_schedule:
        V3_LOGIT["lr_schedule"] = a.lr_schedule      # engine_kwargs applies V3_LOGIT over the config's schedule
    tasks = [t for t in a.tasks.split(",") if t]
    if a.cmd == "tasks":
        for t in tasks:
            for m in DATA_MODES:
                T = build_task(t, m)
                print(json.dumps(dict(task=t, mode=m, n_train=T["split_meta"]["n_train"],
                                      n_held=T["split_meta"]["n_held"], len_train=T["prompt_len_train"],
                                      len_held=T["prompt_len_held"], overlap=T["overlap"])))
        return
    run(tasks, [float(x) for x in a.lams.split(",")], a.mode, a.steps, a.sites, Path(a.out), means=a.means)


if __name__ == "__main__":
    main()
