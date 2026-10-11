"""Self-checks for the DAN-64 sweep's logic (CPU, no model):

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/071-contrast-sweep/test_logic.py -q
"""
import importlib
import io
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logic as L  # noqa: E402


# ------------------------------------------------------------------------------------------- design
def test_sources_are_four_silent_arms_and_three_eval_sets():
    assert L.ARMS == L.BUILD_ORDER == ("close_silent", "random_silent", "distant_silent", "fused_silent")
    assert L.EVAL_SETS == ("close_silent", "random_silent", "distant_silent")       # fused trains only
    assert L.BUILD_ORDER[-1] == L.FUSED and L.SOURCES[L.FUSED] == ("fused", "silent")
    assert all(f == "silent" for _, f in L.SOURCES.values())
    assert L.REFERENCE == "close_silent" and L.LAMS == (3e-4, 1e-3, 3e-3) and L.REF_LAM == 1e-3
    assert not hasattr(L, "decide") and not hasattr(L, "COST")          # no decision rule


def test_set_versions_keep_the_eval_sets_tag():
    # the three eval sets keep the tag out_smoke2/sets was built with; fused has its own
    assert L.SET_VERSION == "071-disjoint-v1"
    assert all(L.set_version(n) == L.SET_VERSION for n in L.EVAL_SETS)
    assert L.set_version(L.FUSED) == "071-fused-v2"            # v1 (merged-then-split) files are refused


# ------------------------------------------------------------------------------------------- fused
def test_fused_quotas_mirror_src():
    assert L.fused_quotas(64) == {"close": 22, "distant": 21, "random": 21}
    assert L.fused_quotas(32) == {"close": 12, "distant": 10, "random": 10}
    for n in range(0, 100):
        q = L.fused_quotas(n)
        assert sum(q.values()) == n and q["distant"] == q["random"] == n // 3


def test_round_robin_merge():
    rr = L.round_robin({"close": ["c0", "c1", "c2"], "distant": ["d0"], "random": ["r0", "r1"]})
    assert rr == [("close", "c0"), ("distant", "d0"), ("random", "r0"), ("close", "c1"), ("random", "r1"),
                  ("close", "c2")]
    assert L.round_robin({"close": [], "distant": [], "random": []}) == []


def test_fused_held_counts_are_exact():
    assert L.fused_held_counts({"close": 22, "distant": 21, "random": 21}, 16) == {"close": 6, "distant": 5,
                                                                                   "random": 5}
    for n in range(4, 97):
        q = L.fused_quotas(n)
        h = L.fused_held_counts(q, n - L.n_train_of(n))
        assert sum(h.values()) == n - L.n_train_of(n)
        if n >= 12:      # parts of >= 4 (tiny parts keep their keep_top first contexts in training first: n = 8 is 2/0/0)
            assert all(abs(h[p] - q[p] * (n - L.n_train_of(n)) / n) < 1 for p in q)    # largest remainder: within 1
    assert L.fused_held_counts({"close": 2, "distant": 1, "random": 1}, 1) == {"close": 1, "distant": 0, "random": 0}


def test_part_split_is_stratified_with_the_given_count():
    from circuit.context_split import stratified_split
    for n in (22, 21, 9, 3):
        t, h = L.part_split(n, round(n * 0.25))
        _, ref = stratified_split(-torch.arange(n, dtype=torch.float64), holdout_frac=0.25, keep_top=2)
        assert h == sorted(ref) and sorted(t + h) == list(range(n))           # the v1 rule when the count agrees
    t, h = L.part_split(22, 6)
    assert len(h) == 6 and 0 in t and 1 in t and len({x // 4 for x in h}) == 6   # spans the order; first two train
    t, h = L.part_split(21, 4)
    assert len(h) == 4 and len(t) == 17
    t, h = L.part_split(2, 1)                                                  # only the keep_top ones: one goes
    assert h == [1] and t == [0]


def test_fused_split_is_exactly_thirds_in_training():
    q = L.fused_quotas(64)
    items = {p: ["%s%d" % (p, i) for i in range(q[p])] for p in L.FUSED_PARTS}
    merged = L.fused_split_merge(items)
    parts = [p for p, _ in merged]
    pc = L.part_counts(parts, 48, [p for p, _ in L.round_robin({p: [0] * v for p, v in L.fused_quotas(32).items()})])
    # pinned: training is exact thirds, held-out the 16 apportioned 6 / 5 / 5
    assert {p: (c["train"], c["held"]) for p, c in pc.items()} == {"close": (16, 6), "distant": (16, 5),
                                                                   "random": (16, 5)}
    assert parts[:6] == ["close", "distant", "random"] * 2 and parts[48:51] == ["close", "distant", "random"]
    for p in L.FUSED_PARTS:                                    # each part's split is its own stratified split
        t, h = L.part_split(q[p], pc[p]["held"])
        assert [x for s, x in merged[:48] if s == p] == [items[p][j] for j in t]
        assert [x for s, x in merged[48:] if s == p] == [items[p][j] for j in h]
    assert {p: c["reserve"] for p, c in pc.items()} == L.fused_quotas(32)
    # a short part: totals still exactly n_train_of(n) / the rest
    m = L.fused_split_merge({"close": list(range(22)), "distant": list(range(21)), "random": list(range(7))})
    assert len(m) == 50 and sum(1 for p, _ in m[:L.n_train_of(50)] if p == "random") >= 5


# ------------------------------------------------------------------------------------------- set order
def test_n_train_matches_engine_rule():
    assert [L.n_train_of(n) for n in (64, 63, 60, 22, 10, 4)] == [48, 47, 45, 16, 8, 3]


def test_stratified_positions_is_v1_close_order():
    from circuit.context_split import stratified_order
    for n in (64, 60, 21, 4):
        pos = L.stratified_positions(n)
        assert pos == stratified_order(-torch.arange(n, dtype=torch.float64), holdout_frac=0.25, keep_top=2)
        assert sorted(pos) == list(range(n))
        n_tr = L.n_train_of(n)
        assert 0 in pos[:n_tr] and 1 in pos[:n_tr]          # the two first-ranked contexts always train
    held = L.stratified_positions(64)[48:]
    assert len({h // 4 for h in held}) == 16                 # one held-out context per block of 4: spans the order


def test_content_key_ignores_dtype_and_separates_rows():
    a = torch.arange(64, dtype=torch.int64)
    assert L.content_key(a) == L.content_key(a.to(torch.int32)) == L.content_key(a.numpy())
    b = a.clone(); b[5] += 1
    assert L.content_key(a) != L.content_key(b)


# ------------------------------------------------------------------------------------------- disjointness
def test_disjoint_keep_drops_by_id_content_and_repeat_in_order():
    ids = [10, 11, 12, 13, 14, 15]
    keys = ["a", "b", "c", "b", "e", "f"]
    kept, drop = L.disjoint_keep(ids, keys, excl_ids=[11], excl_keys=["e"])
    # 11 excluded by id; "b" at 13 is then the first "b" kept; "e" excluded by content
    assert kept == [0, 2, 3, 5] and drop == {"id": 1, "content": 1, "dup": 0}
    kept, drop = L.disjoint_keep(ids, keys)
    assert kept == [0, 1, 2, 4, 5] and drop["dup"] == 1      # the repeated "b" is dropped


def test_split_main_reserve():
    m, r = L.split_main_reserve(list(range(100)), 64, 32)
    assert m == list(range(64)) and r == list(range(64, 96))
    m, r = L.split_main_reserve(list(range(70)), 64, 32)
    assert len(m) == 64 and r == list(range(64, 70))


def test_pairwise_overlap_and_ho_matrix():
    assert L.pairwise_overlap({"a": [1, 2], "b": [2, 3], "c": [4]}) == {("a", "b"): 1, ("a", "c"): 0, ("b", "c"): 0}
    S = {"close_silent": dict(keys=["c%d" % i for i in range(64)], n_train=48),
         "random_silent": dict(keys=["r%d" % i for i in range(64)], n_train=48)}
    m = L.ho_trained_matrix(S)
    assert set(m.values()) == {0} and len(m) == 4
    S["close_silent"]["keys"][50] = "c3"                     # a production duplicate across its own split
    m = L.ho_trained_matrix(S)
    assert m[("close_silent", "close_silent")] == 1 and m[("random_silent", "close_silent")] == 0
    assert L.ho_trained_allowed("close_silent", "close_silent")
    assert not L.ho_trained_allowed("random_silent", "random_silent")
    assert not L.ho_trained_allowed("random_silent", "close_silent")


# ------------------------------------------------------------------------------------------- the sweep's set build
@pytest.fixture
def sweep_module(tmp_path):
    """Import sweep.py without its environment side effects leaking into other tests; OUT -> tmp."""
    saved = dict(os.environ)
    try:
        sys.path.insert(0, str(HERE.parent / "062-h100-protocol-v1"))
        mod = importlib.import_module("sweep")
        old = mod.OUT
        mod.OUT = tmp_path
        yield mod
        mod.OUT = old
    finally:
        os.environ.clear(); os.environ.update(saved)


def _row(v):
    return torch.full((64,), int(v), dtype=torch.long)


class _FakeSelector:
    """close: ids 1000.. in order; random: a fixed order that starts with close ids and a duplicate window;
    distant: starts with random ids. Each id's tokens are the id itself, except id 7777 which repeats id 2001."""

    def __init__(self):
        self.calls = []
        self.orders = {
            "close": list(range(1000, 1300)),
            "random": [1000, 1063, 1070, 2000, 2001, 7777] + list(range(2002, 2400)),
            "distant": [2000, 2003, 1001] + list(range(3000, 3400)),
        }

    def select(self, comp, latent, mode, max_sequences, **kw):
        self.calls.append((mode, max_sequences))
        ids = self.orders[mode][:max_sequences]
        toks = torch.stack([_row(2001 if i == 7777 else i) for i in ids])
        return SimpleNamespace(sequence_ids=ids, tokens=toks, metadata={"mode": mode})


def _fake_sets(sweep, activating=(1002,)):
    S = object.__new__(sweep.Sets)
    prod_ids = list(range(1000, 1064))                         # the production close set = the first 64 close ids
    order = L.stratified_positions(64)
    rec = dict(neg=torch.stack([_row(prod_ids[j]) for j in order]), neg_ids=[prod_ids[j] for j in order])
    S.key, S.rec, S.cache, S.ho, S.ovl = "1.mlp.1", rec, {}, {}, {}
    S.comp, S.latent = 0, 1
    S._sel = _FakeSelector()
    S.activating = {L.content_key(_row(v)) for v in activating}
    S.fh, S.done = io.StringIO(), set()
    S.row = lambda st: dict(seed=S.key, set=st["name"])
    return S


def test_sets_are_disjoint_in_build_order(sweep_module):
    S = _fake_sets(sweep_module)
    S.build_all()                                              # asserts the disjointness itself
    C, R, D = (S.get(n) for n in L.EVAL_SETS)
    assert torch.equal(C["tokens"], S.rec["neg"])              # close = the production set as is
    assert C["v1_match"] == 1.0
    assert C["reserve_ids"] == list(range(1064, 1096))         # reserve = the re-selection past 64
    allk = {n: set(S.get(n)["keys"]) | set(S.get(n)["reserve_keys"]) for n in L.EVAL_SETS}
    assert not (allk["close_silent"] & allk["random_silent"]) and not (allk["close_silent"] & allk["distant_silent"])
    assert not (allk["random_silent"] & allk["distant_silent"])
    rid = set(R["ids"]) | set(R["reserve_ids"])
    assert not rid & set(range(1000, 1096)) and 7777 not in rid   # close ids and the duplicate window dropped
    assert 2000 not in set(D["ids"]) | set(D["reserve_ids"]) and 1001 not in set(D["ids"])
    assert R["n"] == D["n"] == 64 and R["n_train"] == 48 and len(R["reserve_ids"]) == 32
    assert R["meta"]["dropped"] == {"id": 3, "content": 0, "dup": 1}
    # random's set is its selector order minus the exclusions, stratified: the first kept context trains
    assert 2000 in R["ids"][:48] and set(R["ids"]) == {2000, 2001} | set(range(2002, 2064))
    assert set(S.ho.values()) == {0}
    assert all(v == 0 for d in S.ovl.values() for v in d.values())
    F = S.get(L.FUSED)                                         # built last, excluding all three sets
    assert not (set(F["keys"]) | set(F["reserve_keys"])) & (
        allk["close_silent"] | allk["random_silent"] | allk["distant_silent"])
    assert F["n"] == 64 and F["n_train"] == 48 and len(F["reserve_ids"]) == 32
    assert F["version"] == L.set_version(L.FUSED)
    assert {p: c["n"] for p, c in F["part_counts"].items()} == L.fused_quotas(64)
    assert {p: c["reserve"] for p, c in F["part_counts"].items()} == L.fused_quotas(32)
    # each part continues its sub-mode's own ranking past that source's set and reserve
    part_ids = {p: sorted(i for i, x in zip(F["ids"], F["parts"]) if x == p) for p in L.FUSED_PARTS}
    assert part_ids == {"close": list(range(1096, 1118)), "distant": list(range(3096, 3117)),
                        "random": list(range(2096, 2117))}
    assert F["reserve_ids"][:3] == [1118, 3117, 2117]          # the reserve is merged round robin too
    assert F["ids"][:3] == [1096, 3096, 2096]                  # training slices first, round robin
    assert {p: (c["train"], c["held"]) for p, c in F["part_counts"].items()} == {
        "close": (16, 6), "distant": (16, 5), "random": (16, 5)}
    assert F["parts"][48:51] == list(L.FUSED_PARTS)            # then the held-out slices
    assert [m for m, _ in S._sel.calls][-3:] == list(L.FUSED_PARTS)
    # the set files are cached with their version and reloaded
    S2 = _fake_sets(sweep_module)
    S2.build_all()
    assert S2._sel.calls == [] and S2.get("random_silent")["ids"] == R["ids"]
    assert S2.get(L.FUSED)["ids"] == F["ids"] and S2.get(L.FUSED)["parts"] == F["parts"]


def test_fused_excludes_its_earlier_sub_modes(sweep_module):
    S = _fake_sets(sweep_module)
    o = S._sel.orders
    # distant ranks two of fused-close's contexts past distant_silent's 96; random ranks one of fused-distant's
    # contexts past random_silent's 96. Neither is in an earlier set, so only fused's own exclusion drops them.
    o["distant"] = [2000, 2003, 1001] + list(range(3000, 3096)) + [1100, 1101] + list(range(3096, 3400))
    o["random"] = [1000, 1063, 1070, 2000, 2001, 7777] + list(range(2002, 2096)) + [3100] + list(range(2096, 2400))
    S.build_all()
    F = S.get(L.FUSED)
    sub = F["meta"]["submodes"]
    assert sub["distant"]["dropped"] == {"id": 3 + 96 + 2, "content": 0, "dup": 0}
    assert sub["random"]["dropped"] == {"id": 3 + 96 + 1, "content": 1, "dup": 0}    # 7777 = 2001 by content
    assert F["meta"]["dropped"]["id"] == 96 + 101 + 100
    assert len(set(F["ids"]) | set(F["reserve_ids"])) == 96            # no context taken twice
    assert 1100 in F["ids"] and F["parts"][F["ids"].index(1100)] == "close"
    assert 3100 in F["ids"] + F["reserve_ids"]
    p3100 = (F["parts"] + F["reserve_parts"])[(F["ids"] + F["reserve_ids"]).index(3100)]
    assert p3100 == "distant"
    # fused's training contexts are no held-out context of any eval set
    assert all(S.ho[(L.FUSED, e)] == 0 for e in L.EVAL_SETS)


def test_old_sets_stay_valid_and_only_fused_is_built(sweep_module):
    S = _fake_sets(sweep_module)
    for n in L.EVAL_SETS:                                      # an OUT built before fused existed (out_smoke2)
        S.get(n)
    path = lambda n: sweep_module.OUT / "sets" / n / "1.mlp.1.pt"  # noqa: E731
    before = {n: path(n).read_bytes() for n in L.EVAL_SETS}
    assert all(torch.load(path(n), weights_only=False)["version"] == "071-disjoint-v1" for n in L.EVAL_SETS)
    S2 = _fake_sets(sweep_module)
    S2.done = {(S2.key, n) for n in L.EVAL_SETS}               # their rows were logged then, without fused
    S2.build_all()
    assert [m for m, _ in S2._sel.calls] == list(L.FUSED_PARTS)  # only fused's three sub-mode selections
    assert all(path(n).read_bytes() == before[n] for n in L.EVAL_SETS)
    rows = [json.loads(x) for x in S2.fh.getvalue().splitlines()]
    assert [r["set"] for r in rows] == list(L.BUILD_ORDER)     # every row re-logged (now with fused's columns)
    S3 = _fake_sets(sweep_module)
    S3.done = {(S3.key, n) for n in L.BUILD_ORDER}
    S3.build_all()
    assert S3.fh.getvalue() == "" and S3._sel.calls == []      # nothing to do once all four are logged
    # and the three sets equal a fresh build of all four in one go
    S4 = _fake_sets(sweep_module)
    sweep_module.OUT = sweep_module.OUT / "fresh"
    S4.build_all()
    for n in L.EVAL_SETS:
        assert S4.get(n)["ids"] == S2.get(n)["ids"] and torch.equal(S4.get(n)["tokens"], S2.get(n)["tokens"])
        assert S4.get(n)["reserve_ids"] == S2.get(n)["reserve_ids"]
    assert S4.get(L.FUSED)["ids"] == S2.get(L.FUSED)["ids"]


def test_activating_contexts_are_excluded_from_reserves_and_later_sets(sweep_module):
    S = _fake_sets(sweep_module, activating=(1064, 2005, 3002, 1100, 2100, 3100))
    S.build_all()
    assert 1064 not in S.get("close_silent")["reserve_ids"]
    assert 2005 not in S.get("random_silent")["ids"] + S.get("random_silent")["reserve_ids"]
    assert 3002 not in S.get("distant_silent")["ids"] + S.get("distant_silent")["reserve_ids"]
    F = S.get(L.FUSED)
    assert not {1100, 2100, 3100} & set(F["ids"] + F["reserve_ids"]) and F["n"] == 64


def test_selection_grows_when_exclusions_eat_the_slack(sweep_module):
    S = _fake_sets(sweep_module)
    S._sel.orders["random"] = list(range(1000, 1096)) + list(range(2000, 2400))   # starts with all 96 close ids
    S.build_all()
    R = S.get("random_silent")
    assert R["n"] == 64 and len(R["reserve_ids"]) == 32 and min(R["ids"]) >= 2000
    # random_silent: 112, then 224; fused's random sub-mode (143 asked) grows too, past random_silent's 96 + the 96 ids
    assert [m for mode, m in S._sel.calls if mode == "random"] == [112, 224, 143, 286]


def test_stale_set_file_is_refused(sweep_module):
    p = sweep_module.OUT / "sets" / "close_silent" / "1.mlp.1.pt"
    p.parent.mkdir(parents=True)
    torch.save(dict(name="close_silent", tokens=torch.zeros(0, 64)), p)      # an old-design file (no version)
    with pytest.raises(RuntimeError, match="another design"):
        _fake_sets(sweep_module).get("close_silent")
    p = sweep_module.OUT / "sets" / L.FUSED / "1.mlp.1.pt"
    p.parent.mkdir(parents=True)
    for old in (L.SET_VERSION, "071-fused-v1"):               # the eval sets' tag; fused's merged-then-split v1
        torch.save(dict(name=L.FUSED, version=old, tokens=torch.zeros(0, 64)), p)
        with pytest.raises(RuntimeError, match="another design"):
            _fake_sets(sweep_module).get(L.FUSED)


def test_overlap_is_caught(sweep_module):
    S = _fake_sets(sweep_module)
    S.build_all()
    st = S.cache["random_silent"]
    st["keys"][50] = S.cache["close_silent"]["keys"][0]          # forge a leak: random's held-out = a close context
    S.done = set()
    with pytest.raises(AssertionError, match="not disjoint"):
        S.build_all()


def _set(n, start, n_train=None):
    toks = torch.arange(start, start + n).unsqueeze(1).repeat(1, 64)
    return dict(tokens=toks, n=n, n_train=n_train or L.n_train_of(n), keys=[L.content_key(r) for r in toks])


class _FakeSets:
    def __init__(self, d):
        self.d = d

    def get(self, name):
        return self.d[name]


def test_battery_scores_the_same_held_out_set_for_every_arm(sweep_module):
    d = {"close_silent": _set(64, 0), "random_silent": _set(64, 100), "distant_silent": _set(64, 200),
         L.FUSED: _set(64, 300)}
    for ev in L.EVAL_SETS:
        outs = []
        for arm in L.ARMS:
            toks, info = sweep_module.battery_neg(_FakeSets(d), ev, arm)
            assert info["n_ho_trained"] == 0
            assert info["n_ctr_train_overlap"] == (48 if arm == ev else 0)
            outs.append(toks)
        assert all(torch.equal(o, d[ev]["tokens"]) for o in outs)


def test_battery_raises_on_a_leak(sweep_module):
    d = {"close_silent": _set(64, 0), "random_silent": _set(64, 100)}
    d["random_silent"]["keys"][3] = d["close_silent"]["keys"][60]           # random trains on a close held-out
    with pytest.raises(AssertionError):
        sweep_module.battery_neg(_FakeSets(d), "close_silent", "random_silent")
    d[L.FUSED] = _set(64, 300)
    d[L.FUSED]["keys"][7] = d["random_silent"]["keys"][63]                 # fused trains on a random held-out
    with pytest.raises(AssertionError):
        sweep_module.battery_neg(_FakeSets(d), "random_silent", L.FUSED)
    c = d["close_silent"]
    c["keys"][55] = c["keys"][5]                                          # production duplicate: recorded, allowed
    _, info = sweep_module.battery_neg(_FakeSets(d), "close_silent", "close_silent")
    assert info["n_ho_trained"] == 1


# ------------------------------------------------------------------------------------------- pass rule
def test_pass_rule_bounds():
    base = {"free0_tk": 1.0, "freeM_topk_tk": 0.8, "freeN_topk_tk": 1.5, "phi_sup_blind_tk": 0.9}
    assert L.passes(base)
    assert not L.passes(dict(base, freeN_topk_tk=1.5001))
    assert not L.passes(dict(base, freeM_topk_tk=0.7999))
    assert not L.passes(dict(base, phi_sup_blind_tk=0.8999))
    assert not L.passes(dict(base, freeN_topk_tk=None))
    assert not L.passes(dict(base, free0_tk=float("nan")))


# ------------------------------------------------------------------------------------------- matched size
def test_interp_log_n():
    y, x = L.interp_log_n([1000, 100, 10], [0.2, 0.5, 0.8], 100)
    assert abs(y - 0.5) < 1e-12 and not x
    y, x = L.interp_log_n([1000, 100, 10], [0.2, 0.5, 0.8], math.sqrt(1000 * 100))   # log-midpoint
    assert abs(y - 0.35) < 1e-12 and not x
    assert L.interp_log_n([1000, 100], [0.2, 0.5], 5000) == (0.2, True)
    assert L.interp_log_n([1000, 100], [0.2, 0.5], 50) == (0.5, True)
    y, x = L.interp_log_n([100, 100, 1000], [0.4, 0.6, 0.0], 100)                   # equal sizes averaged
    assert abs(y - 0.5) < 1e-12 and not x
    assert L.interp_log_n([], [], 100) == (None, True)
    assert L.interp_log_n([100], [0.3], 100) == (0.3, False)


def test_matched_rates_and_cells_interpolate_to_the_reference_size():
    T, A, Lm, E = 4, 2, 3, 2
    N = np.zeros((T, A, Lm))
    N[:, 0, :] = [1000, 100, 10]                     # reference arm: lambda index 1 -> n* = 100
    N[:, 1, :] = [3000, 300, 30]
    P = np.zeros((T, A, Lm, E))
    P[:, 0, 1, :] = 1.0                              # reference passes everything at n*
    P[:, 1, 1, :] = 1.0                              # arm 1: 1.0 at size 300, 0 at 30 -> 0.523 at 100 (log scale)
    r, x, ns = L.matched_rates(P, N, np.arange(T), 0, 1)
    want = 1 - (math.log10(300) - 2) / (math.log10(300) - math.log10(30))
    assert ns == 100 and r[0] == 1.0 and not x.any() and abs(r[1] - want) < 1e-12
    cells, ex = L.matched_cells(P, N, np.arange(T), 100.0, "mean")
    assert cells.shape == (A, E) and np.allclose(cells.mean(axis=1), r) and not ex.any()
    X = np.full(P.shape, np.nan)
    X[:, 1, :, 0] = [[0.5], [0.7], [0.9], [1.1]] * np.ones((1, Lm))       # per target, constant over lambda
    med, _ = L.matched_cells(X, N, np.arange(T), 100.0, "median")
    assert abs(med[1, 0] - 0.8) < 1e-12 and np.isnan(med[0, 0])


# ------------------------------------------------------------------------------------------- report
def _synthetic(tmp_path, seed=1):
    rng = np.random.default_rng(seed)
    targets = ["%d.%s.%d" % (l, k, i) for l in (1, 6, 10) for k in ("mlp", "resid") for i in range(3)]
    ev, sets, rn, status = [], [], [], []
    sim = {"close_silent": 0.88, "random_silent": 0.68, "distant_silent": 0.25, L.FUSED: 0.6}
    psim = {"close": 0.85, "random": 0.65, "distant": 0.3}
    main = [p for p, _ in L.fused_split_merge({p: [0] * v for p, v in L.fused_quotas(64).items()})]
    q = L.part_counts(main, 48,
                      [p for p, _ in L.round_robin({p: [0] * v for p, v in L.fused_quotas(32).items()})])
    for t in targets:
        status.append(dict(seed=t, ok=True, device="NVIDIA H100 80GB HBM3"))
        for a in L.BUILD_ORDER:
            r = dict(seed=t, set=a, n=64, frac_fire=0.0, frac_fire_train=0.0, frac_fire_held=0.0,
                     pre_rel_med=0.1, pre_rel_p90=0.2, pre_rel_max=0.3, sim_med=sim[a], sim_min=sim[a] - 0.1,
                     n_activating_overlap=0, t_select=1.0, band=L.band(int(t.split(".")[0])), n_reserve=32,
                     meta={"dropped": {"id": 0, "content": 0, "dup": 0}})
            r.update({"ovl_content_%s" % o: 0 for o in L.BUILD_ORDER if o != a})
            r.update({"ovl_id_%s" % o: 0 for o in L.BUILD_ORDER if o != a})
            r.update({"ho_in_train_%s" % o: 0 for o in L.ARMS})
            if a == L.REFERENCE:
                r["v1_match"] = 1.0
            if a == L.FUSED:
                for p, c in q.items():
                    r.update({"part_%s_%s" % (w, p): v for w, v in c.items()})
                    r.update({"part_fire_%s" % p: 0.0, "part_pre_rel_med_%s" % p: 0.1, "part_sim_med_%s" % p: psim[p]})
            sets.append(r)
            for li, lam in enumerate(L.LAMS):
                n = int(1000 / (li + 1) * (1.2 if a == "random_silent" else 1.0))
                for e in L.EVAL_SETS:
                    v = 1.0 if rng.random() < (0.6 if a == "distant_silent" else 0.4) else 0.5
                    ev.append(dict(seed=t, arm=a, lam=L.lam_tag(lam), lam_value=lam, eval=e, n_nodes=n,
                                   free0_tk=v, freeM_topk_tk=v, freeN_topk_tk=v, phi_sup_blind_tk=0.95,
                                   phi_cf_alpha_blind_tk=0.8, vacuous_tk=False, n_ho_trained=0,
                                   n_ctr_train_overlap=48 if a == e else 0, t_fit=10.0, t_score=1.0,
                                   fresh_fit=True, n_train=48, n_ho=16, n_neg_train=48, n_neg_ho=16,
                                   all_sites_ablated=True))
            for e in L.EVAL_SETS:
                rn.append(dict(seed=t, arm=a, eval=e, draw=0, free0_tk=0.1, freeM_topk_tk=0.1, freeN_topk_tk=0.1,
                               phi_sup_blind_tk=0.2, phi_cf_alpha_blind_tk=0.1, n_ref=100, n_drawn=100,
                               short_sites=0, n_ho_trained=0))
    for name, rows in (("eval", ev), ("sets", sets), ("random", rn), ("status", status)):
        (tmp_path / ("%s.shard0.jsonl" % name)).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_report_runs_on_synthetic_rows(tmp_path):
    """report.py end to end on synthetic rows (catches table and bootstrap wiring errors)."""
    report = importlib.import_module("report")
    _synthetic(tmp_path)
    report.OUT, report.B, report.SMOKE, report.LINES[:] = tmp_path, 50, False, []
    report.main()
    text = (tmp_path / "report.md").read_text()
    assert "Train source x eval source" in text and "Contamination" in text and "L8-11" in text
    assert "paired-bootstrap" in text and "random - close" in text and "Set integrity" in text
    assert "fused - random" in text and "Fused vs random" in text and "fused_silent per sub-mode" in text
    assert "eval:fused" not in text                                       # fused is a training arm only
    assert "winner" not in text.lower() and "decision:" not in text.lower()


def test_report_smoke_mode_on_synthetic_rows(tmp_path):
    report = importlib.import_module("report")
    _synthetic(tmp_path)
    report.OUT, report.SMOKE, report.LINES[:] = tmp_path, True, []
    report.SRC_OUT = tmp_path / "none"
    report.main()
    text = (tmp_path / "smoke.md").read_text()
    assert "ALL CHECKS OK" in text, text
    assert "pairwise disjoint" in text and "close > random > distant" in text
    assert "fused_silent has its sub-mode thirds" in text and "fused_silent scored on all three eval sets" in text


def test_report_smoke_catches_a_missing_fused_third(tmp_path):
    report = importlib.import_module("report")
    _synthetic(tmp_path)
    p = tmp_path / "sets.shard0.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    for r in rows:
        if r["set"] == L.FUSED:
            r["part_n_random"], r["part_n_close"] = 0, 43
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report.OUT, report.SMOKE, report.LINES[:] = tmp_path, True, []
    report.SRC_OUT = tmp_path / "none"
    report.main()
    assert "[FAIL] fused_silent has its sub-mode thirds" in (tmp_path / "smoke.md").read_text()
