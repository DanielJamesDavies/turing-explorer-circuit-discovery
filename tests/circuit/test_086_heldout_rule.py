"""DAN-157 held-out rule (circuit/heldout_rule.py) and the opt-in group-aware split in build_record, on synthetic rows:
twins, anchor-prefix copies, near duplicates, no source-file clause (dropped 2026-10-24), the contrast variant
(want="silent"), top-up order, the floor, the split invariants, and eval_probe's rebuilt contrast held-out set."""
import dataclasses
import inspect
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from circuit import heldout_rule as HR
from circuit import protocol_contexts as PC
from circuit.context_split import stratified_split
from utils.neg_context_selector import NegContextSelection


def row(seed, n=64):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randint(1, 30000, (n,), generator=g)


def fire_all(value=1.0):
    def f(cs):
        return [dict(act_at_anchor=value, anchor=10 if c.anchor is None else c.anchor, peak=value, mean=value)
                for c in cs]
    return f


def forbid(rows, ids=(), mid_rows=None):
    F = torch.stack(rows) if rows else torch.zeros(0, 64, dtype=torch.long)
    return HR.Forbidden(tokens=F, ids=set(ids), hashes=set(int(h) for h in HR.window_hash(F)) if rows else set(),
                        mid_ids=set(),
                        mid_hashes=set(int(h) for h in HR.window_hash(torch.stack(mid_rows))) if mid_rows else set())


def cand(tokens, i, anchor=None, origin="scan"):
    return HR.Candidate(tokens=tokens.clone(), id=i, origin=origin, anchor=anchor)


# ----------------------------------------------------------------------------------------------- pair statistics
def test_pair_stats_definitions():
    a = row(1)
    shifted = torch.cat([row(2)[:1], a[:63]])                      # a shifted right by one token
    pref = torch.cat([a[:20], row(3)[20:]])                         # same first 20 tokens
    ex, pfx, lcr, c8 = HR.pair_stats(torch.stack([a, shifted, pref]), a[None])
    assert ex[:, 0].tolist() == [True, False, False]
    assert pfx[:, 0].tolist()[0] == 64 and pfx[2, 0] == 20
    assert lcr[1, 0] == 63 and lcr[2, 0] == 20
    assert c8[0, 0] == 1.0 and c8[1, 0] > 0.95 and c8[2, 0] == pytest.approx(13 / 57)
    assert HR.near_of(lcr, c8)[:, 0].tolist() == [True, True, False]


def test_window_hash_matches_085_corpus_hash():
    w = np.random.default_rng(0).integers(1, 2 ** 62, size=64, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    r = row(5).numpy()
    with np.errstate(over="ignore"):
        ref = int((r.astype(np.uint64) * w).sum(dtype=np.uint64))
    assert int(HR.window_hash(torch.tensor(r))[0]) == ref


# ----------------------------------------------------------------------------------------------- select_heldout
def test_twins_prefix_near_and_silence():
    f0, f1 = row(10), row(11)
    fb = forbid([f0, f1], ids={77})
    cs = [
        cand(row(20), 77),                                          # forbidden id
        cand(f0, 101),                                              # twin: same content, other id
        cand(torch.cat([row(21)[:1], f1[:63]]), 102),               # 1-token shifted copy -> near
        cand(torch.cat([f0[:30], row(22)[30:]]), 103, anchor=12),   # anchor-prefix copy (prefix 30 > anchor 12)
        cand(torch.cat([f0[:30], row(23)[30:]]), 104, anchor=40),   # same prefix, anchor beyond it -> clean
        cand(row(24), 105),                                         # (formerly: a training row's source file) clean
        cand(row(25), 106),                                         # silent
        cand(row(26), 107),                                         # clean
    ]
    silent = {106}

    def fire(c_):
        return [dict(act_at_anchor=0.0 if c.id in silent else 2.0, anchor=12 if c.anchor is None else c.anchor,
                     peak=2.0, mean=1.0) for c in c_]
    acc, ex = HR.select_heldout(cs, fb, fire, n_held=16)
    reasons = {cs[i].id: v["reason"] for i, v in ex.items()}
    assert reasons == {77: "forbidden_id", 101: "forbidden_hash", 102: "near_F", 103: "apfx_F", 104: None,
                       105: None, 106: "silent", 107: None}
    assert [cs[i].id for i in acc] == [104, 105, 107]


def test_no_source_file_clause():
    """The source-file / document rule is gone: no use_source switch, no source field, no source_F reason."""
    assert "use_source" not in inspect.signature(HR.select_heldout).parameters
    assert "src" not in {f.name for f in dataclasses.fields(HR.Candidate)}
    assert "srcs" not in {f.name for f in dataclasses.fields(HR.Forbidden)}
    assert "source_F" not in HR.REASONS
    assert "use_source" not in {f.name for f in dataclasses.fields(PC.GroupSplit)}
    assert "use_source" not in inspect.signature(PC.group_split_for).parameters
    # 2026-10-24: no document rule in the public split either (duplicates only; docs = provenance)
    assert "doc_ids" not in {f.name for f in dataclasses.fields(PC.GroupSplit)}
    assert "docs" not in inspect.signature(HR.dup_groups).parameters
    assert not hasattr(PC, "_held_doc_exclusion")


# ----------------------------------------------------------------------------------------------- contrast variant
def silent_fn(firing=()):
    """want="silent" fire_fn: peak > 0 for the ids in `firing`, else silent everywhere."""
    def f(cs):
        return [dict(act_at_anchor=0.0, anchor=c.anchor, peak=1.5 if c.id in firing else 0.0, mean=0.0) for c in cs]
    return f


def test_contrast_variant_rejects_duplicates_and_firing_rows():
    tr, act = row(70), row(71)                     # a contrast-train row and an activating row
    fb = forbid([tr, act], ids={900})
    A = HR.CONTRAST_ANCHOR
    near = torch.cat([row(72)[:1], act[:63]])
    pfx = torch.cat([tr[:40], row(73)[40:]])       # a 40-token shared prefix: near (LCR >= 32), not exact
    short = torch.cat([tr[:20], row(74)[20:]])     # a 20-token shared prefix: clean (apfx at the last position = exact)
    cs = [cand(row(75), 900, A), cand(tr, 901, A), cand(near, 902, A), cand(pfx, 903, A), cand(short, 904, A),
          cand(row(76), 905, A), cand(row(77), 906, A), cand(row(77), 907, A),
          cand(torch.cat([row(78)[:1], row(77)[:63]]), 908, A), cand(row(79), 909, A)]
    acc, ex = HR.select_heldout(cs, fb, silent_fn({905}), n_held=16, want="silent")
    reasons = {cs[i].id: v["reason"] for i, v in ex.items()}
    assert reasons == {900: "forbidden_id", 901: "forbidden_hash", 902: "near_F", 903: "near_F", 904: None,
                       905: "fires", 906: None, 907: "exact_H", 908: "near_H", 909: None}
    assert [cs[i].id for i in acc] == [904, 906, 909]
    with pytest.raises(ValueError):
        HR.select_heldout(cs, fb, silent_fn(), want="loud")


def test_contrast_rows_have_no_anchor_prefix_rule():
    """Contrast candidates: whole-window exact + near only (Daniel, 2026-10-24). A short shared prefix that the
    anchor-prefix rule WOULD catch for an activating row at an early anchor is accepted for a contrast row, whatever
    anchor it carries; an exact copy and a >= 32-token shared run are still rejected."""
    tr = row(80)
    fb = forbid([tr])
    short = torch.cat([tr[:20], row(81)[20:]])                   # 20 shared leading tokens: not near
    for anchor in (HR.CONTRAST_ANCHOR, 5):                       # even with an early anchor: no apfx test
        acc, ex = HR.select_heldout([cand(short, 1, anchor)], fb, silent_fn(), want="silent")
        assert acc == [0] and ex[0]["reason"] is None
    acc, ex = HR.select_heldout([cand(short, 1, 5)], fb, fire_all(), want="fire")
    assert ex[0]["reason"] == "apfx_F"                           # the activating rule keeps apfx
    long = torch.cat([tr[:33], row(82)[33:]])
    _, ex = HR.select_heldout([cand(tr.clone(), 2, 5), cand(long, 3, 5)], fb, silent_fn(), want="silent")
    assert [ex[0]["reason"], ex[1]["reason"]] == ["forbidden_hash", "near_F"]     # the exact copy, by content hash
    # among accepted contrast rows: the same (exact + near only)
    a = row(83)
    cs = [cand(a, 10, 5), cand(torch.cat([a[:20], row(84)[20:]]), 11, 5), cand(a.clone(), 12, 5)]
    acc, ex = HR.select_heldout(cs, forbid([]), silent_fn(), want="silent")
    assert [cs[i].id for i in acc] == [10, 11] and ex[2]["reason"] == "exact_H"


def test_contrast_variant_fill_order_and_keep_fewer():
    """Originals first (record order), then refill rows in order until 16; short pools keep fewer (no padding)."""
    A = HR.CONTRAST_ANCHOR
    orig = [cand(row(300 + j), 300 + j, A, origin="original") for j in range(16)]
    orig[3] = cand(orig[1].tokens, 303, A, origin="original")            # an exact copy of another held row
    refill = [cand(row(400 + j), 400 + j, A, origin="refill") for j in range(5)]
    acc, ex = HR.select_heldout(orig + refill, forbid([]), silent_fn({305}), n_held=16, want="silent")
    ids = [(orig + refill)[i].id for i in acc]
    assert len(ids) == 16 and ids[:14] == [300, 301, 302] + list(range(304, 305)) + list(range(306, 316))
    assert ids[14:] == [400, 401]
    acc2, _ = HR.select_heldout(orig[:4], forbid([]), silent_fn(), n_held=16, want="silent")
    assert len(acc2) == 3                                                 # keep fewer: the duplicate is dropped


def test_duplicates_among_accepted_and_fill_order():
    base = row(30)
    cs = [cand(base, 1, anchor=20, origin="original"),
          cand(base, 2, anchor=20, origin="original"),                         # exact copy of an accepted row
          cand(torch.cat([base[:25], row(31)[25:]]), 3, anchor=20),            # prefix copy at its anchor
          cand(torch.cat([row(32)[:2], base[:62]]), 4, anchor=5),              # shifted near copy
          ] + [cand(row(100 + j), 10 + j) for j in range(30)]
    acc, ex = HR.select_heldout(cs, forbid([]), fire_all(), n_held=16, chunk=8)
    assert [ex[i]["reason"] for i in range(4)] == [None, "exact_H", "apfx_H", "near_H"]
    assert [cs[i].id for i in acc] == [1] + list(range(10, 25))                # originals first, then in order
    assert len(acc) == 16
    assert max(ex) < len(cs) - 1                                               # the visit stopped at 16


def test_lazy_load_only_reached_candidates():
    pool = {i: row(200 + i) for i in range(100)}
    loaded = []

    def load(cs):
        loaded.extend(c.id for c in cs)
        for c in cs:
            c.tokens = pool[c.id]
    cs = [HR.Candidate(tokens=None, id=i, origin="scan") for i in range(100)]
    acc, _ = HR.select_heldout(cs, forbid([]), fire_all(), n_held=16, chunk=8, load_fn=load)
    assert len(acc) == 16 and len(loaded) == 16                                # two chunks of 8
    with pytest.raises(ValueError):
        HR.select_heldout([HR.Candidate(tokens=None, id=1, origin="scan")], forbid([]), fire_all())


def test_mid_rows_never_enter_the_strongest_set():
    m = row(40)
    cs = [cand(m, 5), cand(row(41), 6)]
    fb = forbid([row(42)], mid_rows=[m])
    acc, ex = HR.select_heldout(cs, fb, fire_all(), exclude_mid=True)
    assert ex[0]["reason"] == "mid_pool" and [cs[i].id for i in acc] == [6]
    acc, _ = HR.select_heldout([cand(m, 5)], fb, fire_all(), exclude_mid=False)
    assert len(acc) == 1


def test_floor_constant_and_anchor_zero_rows():
    # a row read at position 0 sees only its first token: any training row with that first token is the same input
    f = row(50)
    c = cand(torch.cat([f[:1], row(51)[1:]]), 9, anchor=0)
    acc, ex = HR.select_heldout([c], forbid([f]), fire_all())
    assert ex[0]["reason"] == "apfx_F" and acc == []
    assert (HR.N_HELD, HR.FLOOR, HR.NEAR_LCR, HR.NEAR_C8) == (16, 8, 32, 0.5)


# ----------------------------------------------------------------------------------------------- group-aware split
def test_singleton_groups_reproduce_stratified_split():
    rng = random.Random(0)
    for n in list(range(1, 70)) + [128]:
        sc = [rng.random() for _ in range(n)]
        if n > 5:
            sc[3] = sc[4]                                            # a tie
        assert HR.group_stratified_split(sc, list(range(n))) == stratified_split(sc)


@pytest.mark.parametrize("seed", range(20))
def test_group_split_invariants(seed):
    rng = random.Random(seed)
    n = rng.randint(8, 64)
    sc = [rng.random() for _ in range(n)]
    groups = [rng.randrange(max(1, n // rng.randint(1, 6))) for _ in range(n)]
    tr, ho = HR.group_stratified_split(sc, groups)
    assert sorted(tr + ho) == list(range(n)) and not set(tr) & set(ho)
    assert not ({groups[i] for i in tr} & {groups[i] for i in ho})              # no group straddles
    top2 = sorted(range(n), key=lambda i: -sc[i])[:2]
    assert set(top2) <= set(tr)
    assert len(ho) <= round(n * 0.25)
    assert (tr, ho) == HR.group_stratified_split(sc, groups)                    # deterministic


def test_group_split_forced_groups():
    sc = [float(64 - i) for i in range(64)]
    groups = [i // 4 for i in range(64)]                                        # 16 groups of 4
    tr, ho, un = HR.group_split_ex(sc, groups, force_train={1, 2}, force_held={0, 3, 5, 7, 9, 11})
    gt, gh, gu = ({groups[i] for i in x} for x in (tr, ho, un))
    assert 0 in gh | gu and not {0, 3, 5, 7, 9, 11} & gt                        # force_held wins over keep_top
    assert {1, 2} <= gt and not (gt & gh) and not (gt & gu) and not (gh & gu)
    assert len(ho) == 16 and len(un) == 8                                       # 6 forced groups: 4 fit, 2 unused
    assert sorted(tr + ho + un) == list(range(64))


def test_doc_index_provenance():
    ix = PC.DocIndex(np.array([-1, 5, 5, 7, 9, 7, 5]))
    assert ix([1, 3, 6, 0, 99]).tolist() == [5, 7, 5, -1, -1]


def test_dup_groups_links_duplicates_only():
    a, b = row(60), row(61)
    rows = torch.stack([a, a, torch.cat([a[:10], row(62)[10:]]), b, torch.cat([row(63)[:1], b[:63]]), row(64), row(65)])
    anchors = torch.tensor([30, 30, 5, 30, 30, 30, 30])
    g = HR.dup_groups(rows, anchors)
    assert g[0] == g[1] == g[2] == 0                 # exact; prefix copy (row 2 anchored inside the shared prefix)
    assert g[3] == g[4] == 3                         # shifted near copy
    assert g[5] == 5 and g[6] == 6                   # unrelated rows stay apart (no document links any more)
    assert len(set(g)) == 4


# ----------------------------------------------------------------------------------------------- build_record path
DUP = {i: i - 1 for i in range(101, 180, 7)}          # every 7th id is an exact copy of its predecessor


def tokens_of(sid):
    return row(DUP.get(sid, sid))


class Loader:
    def get_batches_by_ids(self, ids, max_length=65):
        for b0 in range(0, len(ids), 4):
            b = ids[b0:b0 + 4]
            t = torch.zeros(len(b), max_length, dtype=torch.long)
            for r, s in enumerate(b):
                t[r, :64] = tokens_of(s)
            yield torch.tensor(b), t


class Bank:
    kinds = ["attn", "mlp", "resid"]
    device = "cpu"

    def encode(self, act, kind, layer):
        return act, torch.full(act.shape, 5, dtype=torch.long)


class Inference:
    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, activations_callback=None, **kw):
        a = tokens.float().unsqueeze(-1)
        for layer in range(3):
            activations_callback(layer, (a, a, a))


class Selector:
    """Contrast candidates 5000.. in order, skipping excluded_positive_ids (as NegContextSelector does)."""
    seen_excluded = None

    def excluded_positive_ids(self, comp, latent):
        return set()

    def select(self, comp, latent, mode, max_sequences, **kw):
        ex = self.excluded_positive_ids(comp, latent)
        Selector.seen_excluded = ex
        ids = [i for i in range(5000, 5200) if i not in ex][:min(64, max_sequences)]
        return NegContextSelection(tokens=torch.stack([row(i) for i in ids]), sequence_ids=ids, mode="close",
                                   metadata={})


def store(rows):
    t = torch.zeros(6, 8, max(len(r) for r in rows), dtype=torch.long)
    for c, r in enumerate(rows):
        t[c, 5, :len(r)] = torch.tensor(r)
    return SimpleNamespace(ctx_seq_idx=t)


def build(group_split=None):
    from config import ContextProtocolV1Config
    top, mid = [[] for _ in range(6)], [[] for _ in range(6)]
    top[4], mid[4] = list(range(100, 164)), list(range(300, 364))
    return PC.build_record(Inference(), Bank(), Loader(), 4, 5, selector=Selector(), cfg=ContextProtocolV1Config(),
                           top_ctx=store(top), mid_ctx=store(mid), group_split=group_split)


def test_build_record_default_unchanged_by_the_option():
    a, b = build(), build(None)
    assert "heldout" not in a and "group" not in a["strong"]
    assert a["strong"]["held"] == b["strong"]["held"] == stratified_split(a["strong"]["peak"])[1]


def test_build_record_group_split():
    docs = lambda ids: np.asarray([(i - 300) // 8 if 300 <= i < 364 else -1 for i in ids])   # mid: 8 per document
    rec = build(PC.GroupSplit(docs=docs))
    s, m, h = rec["strong"], rec["mid"], rec["heldout"]
    for d in (s, m):
        g = d["group"]
        assert not ({g[i] for i in d["train"]} & {g[i] for i in d["held"]})
        assert set(d["held"]) <= set(range(64)) and not set(d["held"]) & set(d["train"])
    # exact copies are grouped (and so never straddle); documents are provenance only: no document groups
    for i, j in DUP.items():
        if 100 <= j < 164 and i < 164:
            assert s["group"][i - 100] == s["group"][j - 100]
    assert len(set(m["group"])) > 8 and m["doc"][:8] == [0] * 8
    assert 12 <= len(s["held"]) <= 16                            # exact copies (DUP) are grouped / dropped only
    assert h["rule"] == HR.RULE_VERSION and h["strong"]["n_held"] == len(s["held"])
    assert h["mid"]["below_floor"] == (len(m["held"]) < HR.FLOOR)
    # no held row repeats a training-side row
    F = torch.cat([s["pos"][s["train"]], m["pos"][m["train"]], rec["neg"]])
    for d in (s, m):
        if d["held"]:
            ex, pfx, lcr, c8 = HR.pair_stats(d["pos"][d["held"]], F)
            assert not ex.any() and not HR.near_of(lcr, c8).any()
    assert build(PC.GroupSplit(docs=docs))["strong"]["held"] == s["held"]                     # deterministic


def test_build_record_documents_are_provenance_only():
    """Duplicates-only grouping (Daniel, 2026-10-24): shared documents neither group rows nor exclude contrast rows,
    so the split and the contrast equal the no-document split's; the joint duplicate groups still never straddle."""
    arr = np.full(6000, -1)
    for i in range(100, 164):
        arr[i] = (i - 100) // 4                                  # strongest: 16 documents of 4
    for i in range(300, 364):
        arr[i] = (i - 300) // 4                                  # mid-band: the SAME 16 documents
    for i in range(5000, 5200):
        arr[i] = (i - 5000) % 16                                 # contrast candidates: spread over them too
    ix = PC.DocIndex(arr)
    cfg = __import__("config").ContextProtocolV1Config()
    mk = lambda gs: PC.build_record(Inference(), Bank(), Loader(), 4, 5, selector=Selector(), cfg=cfg,
                                    top_ctx=store([[], [], [], [], list(range(100, 164)), []]),
                                    mid_ctx=store([[], [], [], [], list(range(300, 364)), []]), group_split=gs)
    rec, bare = mk(PC.GroupSplit(docs=ix)), mk(PC.GroupSplit())
    s, m = rec["strong"], rec["mid"]
    for n in ("strong", "mid"):
        assert rec[n]["train"] == bare[n]["train"] and rec[n]["held"] == bare[n]["held"]
        assert rec[n]["group"] == bare[n]["group"] and bare[n]["doc"] == [-1] * 64
    assert rec["neg_ids"] == bare["neg_ids"] and sorted(rec["neg_ids"]) == list(range(5000, 5064))   # no doc skip
    assert "contrast_excluded_ids" not in rec and Selector.seen_excluded == set()
    g = lambda d, ix_: {d["group"][j] for j in ix_}
    assert not g(s, s["train"]) & g(m, m["held"] + m["unused"])
    assert not g(m, m["train"]) & g(s, s["held"] + s["unused"])
    assert rec["heldout"]["strong"]["n_held"] == len(s["held"]) >= 12
    assert "use_source" not in rec["heldout"] and "n_forbidden_docs" not in rec["heldout"]


# ----------------------------------------------------------------------------------------------- rebuilt contrast
def test_eval_probe_default_contrast_unchanged():
    """A record without neg_held_v2 (every as-run record): the contrast is rec["neg"] and no n_neg_train is set."""
    rec = build()
    p = PC.eval_probe(rec, "strong", "cpu")
    assert torch.equal(p.neg_tokens, rec["neg"]) and "n_neg_train" not in p.metadata
    neg, n_ntr = PC.eval_contrast(rec)
    assert neg is rec["neg"] and n_ntr is None
    pd_, sel = PC.eval_contexts(rec, "strong", "cpu")
    assert torch.equal(sel.select().tokens, rec["neg"]) and torch.equal(pd_.neg_tokens, rec["neg"])


@pytest.mark.parametrize("n_v2", [16, 9, 1])
def test_eval_probe_rebuilt_contrast(n_v2):
    """With neg_held_v2: the run's 48 contrast-train rows, then the rebuilt held-out rows; n_neg_train = 48. The fit's
    probe (training_probe) still gets all 64 of rec["neg"]."""
    rec = build()
    v2 = torch.stack([row(7000 + j) for j in range(n_v2)])
    rec2 = dict(rec, neg_held_v2=v2, neg_held_v2_ids=list(range(7000, 7000 + n_v2)))
    p = PC.eval_probe(rec2, "strong", "cpu")
    assert p.metadata["n_neg_train"] == 48 and p.neg_tokens.shape[0] == 48 + n_v2
    assert torch.equal(p.neg_tokens[:48], rec["neg"][:48]) and torch.equal(p.neg_tokens[48:], v2)
    assert torch.equal(p.pos_tokens, PC.eval_probe(rec, "strong", "cpu").pos_tokens)
    pd_, sel = PC.eval_contexts(rec2, "mid", "cpu")
    assert torch.equal(sel.select().tokens, pd_.neg_tokens) and pd_.metadata["n_neg_train"] == 48
    for arm in ("A", "B"):
        t1, t2 = PC.training_probe(rec, "cpu", arm), PC.training_probe(rec2, "cpu", arm)
        assert torch.equal(t1.neg_tokens, rec["neg"]) and torch.equal(t2.neg_tokens, rec["neg"])
        assert "n_neg_train" not in t2.metadata and torch.equal(t1.pos_tokens, t2.pos_tokens)
    from eval.wcm_eval import contrast_split, split_n
    assert contrast_split(48 + n_v2, p.metadata) == 48
    assert contrast_split(64, {}) == split_n(64) == 48 and contrast_split(48 + n_v2, p.metadata, own=False) == \
        split_n(48 + n_v2)
    with pytest.raises(ValueError):
        contrast_split(48, p.metadata)                                    # no held-out contrast row left
