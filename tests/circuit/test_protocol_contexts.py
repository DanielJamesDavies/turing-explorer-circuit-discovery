"""Protocol-v1 contexts (circuit/protocol_contexts.py) on a fake engine: record shape, thin / skip rules, and the
training / eval ProbeDataset composition. The equivalence with the 062 run's cached records is a GPU check
(experiments/062-h100-protocol-v1/check_protocol_contexts.py)."""
from types import SimpleNamespace

import pytest
import torch

from circuit import protocol_contexts as PC
from circuit.context_split import stratified_order, stratified_split
from utils.neg_context_selector import NegContextSelection

KINDS = ["attn", "mlp", "resid"]
MISSING = {13}                                       # ids the fake loader cannot locate


def peak_of(sid):
    return float((sid * 37) % 101 + 1)


class FakeLoader:
    """Row for id s: zeros except value peak_of(s) at position s % 64, so the target's peak / anchor are known."""

    def get_batches_by_ids(self, ids, max_length=65):
        ids = [s for s in ids if s not in MISSING]
        for b0 in range(0, len(ids), 4):
            b = ids[b0:b0 + 4]
            t = torch.zeros(len(b), max_length, dtype=torch.long)
            for r, s in enumerate(b):
                t[r, s % 64] = int(peak_of(s))
            yield torch.tensor(b), t


class FakeBank:
    kinds = KINDS
    device = "cpu"

    def encode(self, act, kind, layer):
        return act, torch.full(act.shape, 5, dtype=torch.long)     # every token: latent 5 at value = the token id


class FakeInference:
    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, activations_callback=None, **kw):
        a = tokens.float().unsqueeze(-1)
        for layer in range(3):
            activations_callback(layer, (a, a, a))


class FakeSelector:
    def __init__(self, n):
        self.n = n

    def select(self, comp, latent, mode, max_sequences, **kw):
        assert mode == "close" and kw == PC._SELECT_KW
        n = min(self.n, max_sequences)
        return NegContextSelection(tokens=torch.arange(n * 64).reshape(n, 64), sequence_ids=list(range(1000, 1000 + n)),
                                   mode="close", metadata={})


def store(rows):
    k = max(len(r) for r in rows)
    t = torch.zeros(len(rows), 8, k, dtype=torch.long)
    for c, r in enumerate(rows):
        t[c, 5, :len(r)] = torch.tensor(r)
    return SimpleNamespace(ctx_seq_idx=t)


def cfg(**kw):
    from config import ContextProtocolV1Config
    return ContextProtocolV1Config(**kw)


def record(top, mid, n_neg=64, comp=4, **kw):
    rows_top, rows_mid = [[] for _ in range(6)], [[] for _ in range(6)]
    rows_top[comp], rows_mid[comp] = top, mid
    return PC.build_record(FakeInference(), FakeBank(), FakeLoader(), comp, 5, selector=FakeSelector(n_neg),
                           cfg=cfg(**kw), top_ctx=store(rows_top), mid_ctx=store(rows_mid))


TOP = list(range(100, 180))                          # 80 stored, capped at 64
MID = [0, 100, 101] + list(range(200, 260))          # sentinel and duplicates of the top store are dropped


def test_store_ids_dedup_and_sentinels():
    top, mid = PC.store_ids(store([[3, 0, 3, 4]]), store([[4, 5, 0, 6, 5]]), 0, 5)
    assert top == [3, 4] and mid == [5, 6]


def test_strong_holdout_fraction():
    assert PC.strong_holdout_frac(64, 0.25, 32) == 1 / 3
    assert PC.strong_holdout_frac(64, 0.25, 48) == 0.0
    assert PC.strong_holdout_frac(64, 0.25, 36) == 0.25


def test_full_record():
    rec = record(TOP, MID)
    s, m = rec["strong"], rec["mid"]
    assert rec["n_top"] == 80 and rec["n_mid"] == 60
    assert s["pos"].shape == (64, 64) and s["tgt"].shape == (64, 64) and s["ids"] == TOP[:64]
    assert torch.equal(s["peak"], torch.tensor([peak_of(i) for i in TOP[:64]]))
    assert s["arg"].tolist() == [i % 64 for i in TOP[:64]]
    assert (len(s["train"]), len(s["held"])) == (48, 16) and (s["train"], s["held"]) == stratified_split(s["peak"])
    assert len(m["train"]) == 45 and len(m["held"]) == 15
    # D: 32 of strongest-train, the two strongest included; B_mid: 16 of mid-train
    assert len(rec["D_train"]) == 32 and set(rec["D_train"]) <= set(s["train"])
    top2 = torch.argsort(s["peak"], descending=True, stable=True)[:2].tolist()
    assert set(top2) <= set(rec["D_train"])
    assert len(rec["B_mid"]) == 16 and set(rec["B_mid"]) <= set(m["train"])
    # contrast: stratified by similarity rank, train first; ids row-aligned
    order = stratified_order(-torch.arange(64, dtype=torch.float64))
    assert torch.equal(rec["neg"], torch.arange(64 * 64).reshape(64, 64)[order])
    assert rec["neg_ids"] == [1000 + j for j in order]
    assert PC.skip_reason(rec) is None and PC.training_arm(rec) == ("B", False)


def test_loader_skips_are_row_aligned():
    rec = record([13] + list(range(20, 40)), [])
    s = rec["strong"]
    assert 13 not in s["ids"] and s["pos"].shape[0] == len(s["ids"]) == 20
    assert s["arg"].tolist() == [i % 64 for i in s["ids"]]


def test_thin_target_without_mid_pool_trains_on_strongest_train():
    rec = record(TOP, MID[:9])                       # 6 mid-band contexts after dedup: < 8, no pool
    assert rec["mid"] is None and "B_mid" not in rec
    assert PC.training_arm(rec) == ("A", True)
    pd_ = PC.training_probe(rec, "cpu", holdout_frac=0.25)
    s = rec["strong"]
    assert pd_.metadata == dict(n_train=48, n=64)
    assert torch.equal(pd_.pos_tokens, torch.cat([s["pos"][s["train"]], s["pos"][s["held"]]]))


def test_too_few_top_contexts_is_a_skip_record():
    rec = record(TOP[:7], MID)
    assert rec["strong"] is None and rec["mid"] is not None and "neg" not in rec
    assert PC.skip_reason(rec) == "too_few_contexts"
    pd_ = PC.training_probe(rec, "cpu", holdout_frac=0.25)
    assert pd_.pos_tokens.shape[0] == 0 and pd_.metadata["skip"] == "too_few_contexts"


def test_no_contrast_is_a_skip_record():
    rec = record(TOP, MID, n_neg=3)
    assert rec["strong"] is None and rec["no_contrast"] and "D_train" in rec
    assert PC.skip_reason(rec) == "no_contrast"


def test_training_probe_arm_b_composition():
    rec = record(TOP, MID)
    s, m = rec["strong"], rec["mid"]
    pd_ = PC.training_probe(rec, "cpu", holdout_frac=0.25)
    want = torch.cat([s["pos"][rec["D_train"]], m["pos"][rec["B_mid"]], s["pos"][s["held"]]])
    assert torch.equal(pd_.pos_tokens, want)
    assert torch.equal(pd_.pos_argmax, torch.cat([s["arg"][rec["D_train"]], m["arg"][rec["B_mid"]], s["arg"][s["held"]]]))
    assert torch.equal(pd_.neg_tokens, rec["neg"])
    assert pd_.metadata == dict(n_train=48, n=64)
    n = pd_.pos_tokens.shape[0]
    assert n - int(round(n * 0.25)) == 48            # the engine's slice is exactly the training set


def test_small_pool_padding():
    rec = record(TOP[:17], [])                       # 17 strongest: 13 train / 4 held, no mid
    pd_ = PC.training_probe(rec, "cpu", holdout_frac=0.25)
    assert pd_.metadata == dict(n_train=13, n=17)


def test_padded_length_matches_engine_split():
    for n_tr in range(1, 130):
        n = PC.padded_length(n_tr)
        assert n - int(round(n * 0.25)) == n_tr


@pytest.mark.parametrize("held", ["strong", "mid"])
def test_eval_probe(held):
    rec = record(TOP, list(range(200, 264)))         # full mid pool: 48 / 16
    s, h = rec["strong"], rec[held]
    pd_, sel = PC.eval_contexts(rec, held, "cpu")
    assert torch.equal(pd_.pos_tokens[:48], s["pos"][s["train"]])
    assert torch.equal(pd_.pos_tokens[48:], h["pos"][h["held"]]) and pd_.metadata == dict(n_train=48, n=64)
    assert torch.equal(sel.select(0, 0, "close", 64).tokens, rec["neg"])


def test_eval_probe_short_mid_pool_is_padded_short():
    """The run's behaviour, reproduced: a mid pool under 64 holds out fewer than 16, so the eval list is shorter
    than the padded length and the engine's split moves strongest-train contexts into the held-out slice."""
    rec = record(TOP, MID)                           # 60 mid-band contexts: 15 held out
    pd_ = PC.eval_probe(rec, "mid", "cpu")
    assert pd_.metadata == dict(n_train=48, n=63)
    assert 63 - int(round(63 * 0.25)) == 47


def test_eval_probe_rejects_missing_pool():
    rec = record(TOP, MID[:9])
    with pytest.raises(ValueError):
        PC.eval_probe(rec, "mid", "cpu")


def test_counts_are_configurable():
    rec = record(TOP, MID, train_strong=40, train_mid=8, contrast_count=32)
    assert len(rec["D_train"]) == 40 and len(rec["B_mid"]) == 8 and rec["neg"].shape[0] == 32


# ------------------------------------------------------------------------------------------- config / engine wiring
def test_config_defaults_are_legacy():
    from config import DiscoveryConfig
    d = DiscoveryConfig()
    assert d.context_protocol == "legacy"
    c = d.context_v1
    assert (c.pool_size, c.min_pool, c.holdout_frac, c.keep_top, c.train_strong, c.train_mid, c.contrast_count,
            c.min_contrast, c.cache_dir) == (64, 8, 0.25, 2, 32, 16, 64, 4, None)
    with pytest.raises(ValueError):
        DiscoveryConfig(context_protocol="v2")


def test_v1_switch_routes_discovery_through_the_module(monkeypatch):
    from circuit.discovery.base import DiscoveryMethod
    from circuit.discovery.gradient_base import GradientDiscoveryBase
    from config import config
    rec = record(TOP, MID)

    class Builder:
        def protocol_contexts(self):
            return SimpleNamespace(record=lambda comp, latent: rec)

    class M(DiscoveryMethod):
        def discover(self, *a):
            return None

    monkeypatch.setattr(config.discovery, "context_protocol", "v1")
    monkeypatch.setattr(config.discovery.learned_mask, "holdout_frac", 0.25)
    m = M(None, FakeBank(), None, Builder())
    pd_ = m.build_probe_dataset(4, 5)
    want = PC.training_probe(rec, "cpu", holdout_frac=0.25)
    assert torch.equal(pd_.pos_tokens, want.pos_tokens) and pd_.metadata == want.metadata
    neg = GradientDiscoveryBase._floor_negatives(SimpleNamespace(), pd_, 4, 5, None)
    assert neg is pd_.neg_tokens
