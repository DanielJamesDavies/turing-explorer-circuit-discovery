"""CPU tests for wcm_tcz.py (arena A WCM adapter, DAN-195). Run in our .venv, CPU only:

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 ./.venv/bin/python -B -m pytest -q -p no:cacheprovider \
      experiments/107-true-baselines/arena_a/test_wcm_tcz.py

  1 splice: the site model with no edit, and with an open gate at every site (edit of exactly zero), equals the raw HF
    model with the same BOS prefix, bitwise.
  2 edit semantics: an edit d at (L, tc) == adding d to h[L].mlp's output; at (L, z) == adding d to c_proj's input;
    (L, z) and (L, z2) edits add.
  3 parity: our z code (z and z2 halves concatenated) == Kissane's formula on z_TL = z_HF - b_V (their training
    convention); our tc code == Dunefsky's formula relu((x_norm - b_dec) W_enc + b_enc) on the pure normalisation
    (sae_training pickles, @4f3ca345).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ARENA_A_BOS", "1")
os.environ.setdefault("ARENA_A_SCOPE", "a1")

import wcm_tcz as W  # noqa: E402

LAYERS = (0, 5, 7, 11)
TC_PICKLES = W.HF_CACHE / "models--pchlenski--gpt2-transcoders/snapshots/4f3ca345572ce2d75018a60d10b4dca8bd899e99"


@pytest.fixture(scope="module")
def mb():
    torch.manual_seed(0)
    model, bank = W.load_model(device="cpu", preload=False)
    toks = torch.randint(0, 50000, (2, 24))
    return model, bank, toks


def raw_logits(model, toks):
    bos = torch.full((toks.shape[0], 1), W.BOS_ID, dtype=toks.dtype)
    return model.hf(input_ids=torch.cat([bos, toks], 1), use_cache=False).logits[:, 1:]


@torch.no_grad()
def test_splice_exact(mb):
    model, bank, toks = mb
    ref = raw_logits(model, toks)
    got, _ = model(toks, return_all_logits=True)
    assert torch.equal(got, ref)

    def zero_edit(layer, kind, x):
        if (layer, kind) not in W.real_sites() or layer not in LAYERS:
            return None
        sae = bank.saes[kind][layer]
        acts, _ = sae.encode(x)
        d = sae.decode(acts, add_bias=False) - sae.decode(acts, add_bias=False)
        return x + d
    with model.patch_sites(zero_edit):
        got2, _ = model(toks, return_all_logits=True)
    assert torch.equal(got2, ref)


@torch.no_grad()
def test_edit_semantics(mb):
    model, bank, toks = mb
    d = torch.randn(W.D_MODEL) * 0.5
    for layer, kind in ((3, "tc"), (3, "z"), (5, "z2")):
        def edit(l, k, x, layer=layer, kind=kind):
            return x + d if (l, k) == (layer, kind) else None
        with model.patch_sites(edit):
            got, _ = model(toks, return_all_logits=True)
        blk = model.hf.transformer.h[layer]
        if kind == "tc":
            h = blk.mlp.register_forward_hook(lambda m, a, o: torch.cat([o[:, :1], o[:, 1:] + d], 1))
        else:
            h = blk.attn.c_proj.register_forward_pre_hook(
                lambda m, a: (torch.cat([a[0][:, :1], a[0][:, 1:] + d], 1),))
        try:
            ref = raw_logits(model, toks)
        finally:
            h.remove()
        assert (got - ref).abs().max() <= 1e-4, (layer, kind, float((got - ref).abs().max()))
    # z and z2 at the same layer add
    def both(l, k, x):
        return x + d if (l, k) in ((5, "z"), (5, "z2")) else None
    with model.patch_sites(both):
        got, _ = model(toks, return_all_logits=True)
    h = model.hf.transformer.h[5].attn.c_proj.register_forward_pre_hook(
        lambda m, a: (torch.cat([a[0][:, :1], a[0][:, 1:] + 2 * d], 1),))
    try:
        ref = raw_logits(model, toks)
    finally:
        h.remove()
    assert (got - ref).abs().max() <= 1e-4


def test_sites_and_ids():
    a1, a2 = W.real_sites("a1"), W.real_sites("a2")
    assert len(a1) == 26 and sum(k == "z2" for _, k in a1) == 2 and {l for l, k in a1 if k == "z2"} == set(W.WIDE)
    assert a2 == [(l, "tc") for l in range(W.N_LAYER)]
    assert W.orig_latent("z2", 7) == ("z", W.D_SAE + 7) and W.orig_latent("z", 7) == ("z", 7)
    assert W.orig_latent("tc", 7) == ("tc", 7)


@torch.no_grad()
def test_shared_input_sites(mb):
    """(L, z) and (L, z2) read the same clean c_proj input (README risk 1); a narrow layer's z2 slot is inert."""
    model, bank, toks = mb
    with model.capture_sites() as cap:
        model(toks, return_all_logits=True)
    acts = cap.tensor
    K = list(W.KINDS)
    assert acts.shape[3] == toks.shape[1]                     # BOS hidden: site positions = the caller's tokens
    for layer in W.WIDE:
        assert torch.equal(acts[:, layer, K.index("z")], acts[:, layer, K.index("z2")])
    sae = bank.saes["z2"][3]
    assert getattr(sae, "null", False)
    c, _ = sae.encode(acts[:, 3, K.index("z")])
    assert c.shape[-1] == W.D_SAE and not c.any()


@torch.no_grad()
def test_z_fvu_smoke_corpus(mb):
    """The bank's z dictionaries reconstruct real text (the parity test cannot see a wrong input convention: it
    applies the same offset on both sides). 8 smoke-corpus sequences, centred FVU of decode(encode(z_HF)) vs z_HF.
    Measured 2026-11-05 with the LN-folded b_V': 0.46 / 0.28 / 0.34 at L0-2 (raw b_V only: 21.6 / 21.3 / 12.9)."""
    import numpy as np
    model, bank, _ = mb
    a = np.load(str(W.ROOT / "data_public" / "gpt2-small" / "shard_0.npy"))
    sep = np.where(a == -1)[0]
    seqs = [a[s + 2:e][:64] for s, e in zip(np.r_[-1, sep[:-1]], sep)][:8]   # lead token dropped, as the loader
    toks = torch.tensor(np.stack(seqs))
    with model.capture_sites() as cap:
        model(toks, return_all_logits=True)
    acts = cap.tensor
    for layer in (0, 1, 2):
        x = acts[:, layer, list(W.KINDS).index("z")].reshape(-1, W.D_MODEL)
        sae = bank.saes["z"][layer]
        rec = sae.decode(sae.encode(x)[0], add_bias=True)
        fvu = float(((x - rec) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
        bank.release("z", layer)
        assert fvu < 0.75, (layer, fvu)


def _stub():
    class _Any:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, st):
            self.__dict__.update(st if isinstance(st, dict) else {})
    for n in ["sae_training", "sae_training.sparse_autoencoder", "sae_training.config", "sae_training.utils",
              "sae_training.geometric_median"]:
        if n not in sys.modules:
            m = types.ModuleType(n)
            m.__getattr__ = lambda name: _Any
            sys.modules[n] = m


@torch.no_grad()
def test_parity(mb):
    model, bank, toks = mb
    with model.capture_sites() as cap:
        model(toks, return_all_logits=True)
    acts = cap.tensor                                                   # [B, L, K, T, N]
    K = list(W.KINDS)
    for layer in LAYERS:
        # z: ours (z [+ z2]) vs Kissane on z_TL = z_HF - b_V
        z_hf = acts[:, layer, K.index("z")]
        ours = bank.saes["z"][layer].encode(z_hf)[0]
        if layer in W.WIDE:
            ours = torch.cat([ours, bank.saes["z2"][layer].encode(z_hf)[0]], -1)
        sd = torch.load(W.z_file(layer), map_location="cpu", weights_only=True)
        z_tl = z_hf - W.tl_value_bias(layer)
        ref = torch.relu((z_tl - sd["b_dec"]) @ sd["W_enc"] + sd["b_enc"])
        tol = 1e-5 * max(1.0, float(ref.abs().max()))
        assert ours.shape == ref.shape and (ours - ref).abs().max() <= tol, (layer, float((ours - ref).abs().max()))
        # tc: ours on the full ln_2 output vs Dunefsky on the pure normalisation
        x_full = acts[:, layer, K.index("tc")]
        ln = model.hf.transformer.h[layer].ln_2
        x_norm = (x_full - ln.bias) / ln.weight
        _stub()
        blob = torch.load(TC_PICKLES / (W.TC_FILE_PICKLE % layer), map_location="cpu", weights_only=False)
        s = blob["state_dict"]
        ref = torch.relu((x_norm - s["b_dec"]) @ s["W_enc"] + s["b_enc"])
        ours = bank.saes["tc"][layer].encode(x_full)[0]
        tol = 1e-4 * max(1.0, float(ref.abs().max()))
        assert (ours - ref).abs().max() <= tol, (layer, float((ours - ref).abs().max()))
        for kind in ("z", "z2", "tc"):
            if (layer, kind) in W.real_sites():
                bank.release(kind, layer)
