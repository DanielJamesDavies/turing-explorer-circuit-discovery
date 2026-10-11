"""DAN-37: the data-pass stores, candidate selection and pass-1 encoding follow the model family.

TuringLLM keeps its historical grid (12 layers x attn/mlp/resid, d_sae 40,960, k 128, d_model 1,024). A public family
allocates on its own grid: Pythia-70m 6 x 3 sites, GPT-2 small 12 x 4 sites, d_sae 32,768. A public bank built with
skip_missing marks a dictionary it cannot load as a missing site, and pass-1 encoding skips that component. Nothing
here allocates a store or loads a dictionary.
"""

import pytest
import torch
from huggingface_hub.utils import LocalEntryNotFoundError

from config import config
from model.families import public_store_grid


@pytest.fixture
def family(monkeypatch):
    def set_family(name):
        monkeypatch.setattr(config.model, "family", name)
    return set_family


def _stores():
    from store.context import Context
    from store.latent_stats import LatentStats
    from store.logit_context import LogitContext
    from store.seq_repr import SeqRepr
    from store.top_coactivation import TopCoactivation
    return dict(top=Context("top"), mid=Context("mid"), neg=Context("neg"),
                stats=LatentStats(device=torch.device("cpu")), logit=LogitContext(device=torch.device("cpu")),
                coact=TopCoactivation(device=torch.device("cpu")), seq_repr=SeqRepr(n_seqs=4))


@pytest.mark.parametrize("name, n_comp, d_sae, k, d_model", [
    ("turingllm", 36, 40960, 128, 1024),
    ("pythia-70m", 18, 32768, 16, 512),
    ("gpt2-small", 48, 32768, 32, 768),
])
def test_store_grid_follows_family(family, name, n_comp, d_sae, k, d_model):
    family(name)
    s = _stores()
    for key in ("top", "mid", "neg", "logit", "coact"):
        assert (s[key].num_components, s[key].d_sae) == (n_comp, d_sae), key
    assert s["stats"].num_components == n_comp
    assert s["stats"].sae_config.d_sae == d_sae
    assert s["coact"].sae_config.k == k                         # PMI's tokens = active_count[0].sum() / k
    assert s["seq_repr"].repr_dim == d_model
    assert s["seq_repr"].repr_buf.shape == (5, d_model)
    assert not s["top"]._allocated                               # nothing allocated at construction


def test_turingllm_has_no_public_grid(family):
    family("turingllm")
    assert public_store_grid() is None


def test_neg_ctx_backend_walks_family_components(family):
    from store.neg_ctx.backends import _num_components
    for name, n in (("turingllm", 36), ("pythia-70m", 18), ("gpt2-small", 48)):
        family(name)
        assert _num_components() == n


def test_candidate_kinds_follow_family(family):
    from circuit.feature_selection import _seed_passes_filter, _store_kinds
    family("turingllm")
    assert _store_kinds() == ["attn", "mlp", "resid"]
    family("gpt2-small")
    assert _store_kinds() == ["attn", "resid_mid", "mlp", "resid"]
    # comp 5 = layer 1, kind 1 (resid_mid) on GPT-2; layer 1, kind 2 (resid) on a 3-kind grid
    assert _seed_passes_filter({"comp_idx": 5}, {1}, {"resid_mid"})
    family("pythia-70m")
    assert _seed_passes_filter({"comp_idx": 5}, {1}, {"resid"})


def test_top_store_keeps_only_firing_contexts():
    """A row with fewer firing sequences than slots keeps empty slots (id 0), whatever topk's tie-break does."""
    from store.context import Context
    ctx = Context("top", device=torch.device("cpu"))
    ctx.num_components, ctx.d_sae, ctx.num_ctx_sequences = 1, 3, 4
    # 3 sequences x 2 tokens x k=1: latent 0 fires in sequences 11 and 13, latent 1 in 12, latent 2 never
    acts = torch.tensor([[[1.0], [0.5]], [[2.0], [0.0]], [[0.25], [0.0]]])
    idx = torch.tensor([[[0], [0]], [[1], [1]], [[0], [0]]])
    ctx.update_component(0, torch.tensor([11, 12, 13], dtype=torch.int32), (acts, idx))
    ctx.update_component(0, torch.tensor([14, 15, 16], dtype=torch.int32), (torch.zeros(3, 2, 1), idx))
    rows = {lat: sorted(int(s) for s in ctx.ctx_seq_idx[0, lat].tolist() if s > 0) for lat in range(3)}
    assert rows == {0: [11, 13], 1: [12], 2: []}
    assert bool(((ctx.ctx_seq_idx[0] > 0) == (ctx.ctx_seq_val[0].float() > 0)).all())


class _FakeBank:
    def __init__(self, kinds, missing, skip_missing=True):
        self.kinds = list(kinds)
        self.missing = set(missing)
        self.skip_missing = skip_missing
        self.parallel_kinds = False
        self.calls = []

    def has_site(self, kind, layer):
        return (kind, layer) not in self.missing

    def encode(self, x, kind, layer):
        self.calls.append((kind, layer))
        return x[..., :2], torch.zeros(x.shape[:-1] + (2,), dtype=torch.long)


def test_encoding_skips_missing_sites():
    from pipeline.encoding import encode_layer_components
    acts = tuple(torch.randn(1, 3, 4) for _ in range(4))
    bank = _FakeBank(["attn", "resid_mid", "mlp", "resid"], missing={("mlp", 10)})
    out = encode_layer_components(bank, 10, acts, primary_device=torch.device("cpu"), multi_gpu=False)
    assert sorted(out) == [40, 41, 43]                           # comp 42 = 10.mlp skipped
    assert ("mlp", 10) not in bank.calls
    # a bank without skip_missing (TuringLLM's) encodes every kind, as before
    plain = _FakeBank(["attn", "mlp", "resid"], missing={("mlp", 2)}, skip_missing=False)
    out = encode_layer_components(plain, 2, acts[:3], primary_device=torch.device("cpu"), multi_gpu=False)
    assert sorted(out) == [6, 7, 8]


def test_public_bank_marks_unloadable_site_missing(monkeypatch):
    from sae.public_bank import PublicSAEBank

    def fake_load(self, kind, layer):
        if layer == 10:
            raise LocalEntryNotFoundError("not in cache")
        return object()

    monkeypatch.setattr(PublicSAEBank, "_load", fake_load)
    bank = PublicSAEBank(device=torch.device("cpu"), family="gpt2-small", preload=False, skip_missing=True)
    assert bank.saes["mlp"][10] is None and not bank.has_site("mlp", 10)
    assert bank.has_site("mlp", 9)
    assert bank.missing_sites == {("mlp", 10)}
    bank.load_saes()
    assert bank.missing_sites == {(kind, 10) for kind in bank.kinds}

    strict = PublicSAEBank(device=torch.device("cpu"), family="gpt2-small", preload=False, skip_missing=False)
    with pytest.raises(LocalEntryNotFoundError):
        strict.saes["attn"][10]
