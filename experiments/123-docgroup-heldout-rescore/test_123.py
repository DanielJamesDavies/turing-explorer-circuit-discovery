"""CPU unit tests for 123 (DAN-217): grouping (no group crosses the split), determinism, resume, report logic.

  python -X utf8 -m pytest -q experiments/123-docgroup-heldout-rescore/test_123.py
The pilot-equivalence test reads the real records / scan / corpus index and is skipped when they are absent.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import common123 as C  # noqa: E402


# --------------------------------------------------------------------------------------------- fakes
def fake_rec(ids, train, held, docs=None, mid=None, neg_ids=None):
    n = len(ids)
    s = dict(pos=torch.arange(n * 64).reshape(n, 64) % 50000, tgt=torch.zeros(n, 64, dtype=torch.long),
             arg=torch.zeros(n, dtype=torch.long), peak=torch.ones(n), ids=list(ids), train=list(train),
             held=list(held), unused=[])
    if docs is not None:
        s["doc"] = list(docs)
    rec = dict(strong=s, D_train=list(train[:2]), B_mid=[], neg=torch.zeros(4, 64, dtype=torch.long),
               neg_ids=neg_ids, n_top=n, n_mid=0)
    if mid is not None:
        m_ids, m_train, m_held, m_docs = mid
        rec["mid"] = dict(pos=torch.zeros(len(m_ids), 64, dtype=torch.long), ids=list(m_ids), train=list(m_train),
                          held=list(m_held), doc=list(m_docs), arg=torch.zeros(len(m_ids), dtype=torch.long))
        rec["B_mid"] = list(m_train[:1])
    return rec


GROUP = {i: {i // 10} for i in range(1, 400)}           # doc = id // 10 (ten windows per document)
gof = GROUP.__getitem__


# --------------------------------------------------------------------------------------------- grouping
def test_tl_plan_no_group_crosses_split():
    # pool: 6 rows; train = ids 11, 21, 31 (docs 1, 2, 3); held = ids 12 (doc 1: leaky), 45, 55
    rec = fake_rec([11, 21, 31, 12, 45, 55], train=[0, 1, 2], held=[3, 4, 5])
    scan = [11, 13, 46, 77, 88, 22, 99]                  # 11 in pool (skipped); 13, 22 leaky; 46, 77, 88, 99 allowed
    pl = C.tl_plan(rec, scan, gof)
    assert pl["forbidden"] == {1, 2, 3}
    assert pl["orig_kept_idx"] == [4, 5]
    assert [i for _, i in pl["scan_allowed"]] == [46, 77, 88, 99]
    assert [r for r, _ in pl["scan_allowed"]] == [3, 4, 5, 7]          # 1-based scan ranks
    assert pl["counts"] == dict(orig_total=3, orig_allowed=2, scan_total=6, scan_allowed=4)
    allowed = [int(rec["strong"]["ids"][j]) for j in pl["orig_kept_idx"]] + [i for _, i in pl["scan_allowed"]]
    assert not any(gof(i) & pl["forbidden"] for i in allowed)


def test_mid_train_rows_forbid_groups_too():
    rec = fake_rec([11, 21, 55], train=[0, 1], held=[2], mid=([51, 61, 71], [0], [1, 2], [5, 6, 7]))
    assert C.training_ids(rec) == [11, 21, 51]
    pl = C.tl_plan(rec, [], gof)
    assert 5 in pl["forbidden"] and pl["orig_kept_idx"] == []          # held id 55 shares doc 5 with mid-train id 51


def test_twins_carry_their_groups():
    # ids 1..6; id 2 and id 5 are content twins (same hash) in files of different documents
    src = np.array([0, 0, 1, 2, 3, 4, 5])                              # index 0 = no id
    hashes = np.array([0, 100, 200, 300, 400, 200, 600])
    labels = {"doc": np.array([10, 11, 12, 13, 14, 15]), "topic": np.array([1, 1, 2, 2, 3, 3])}
    G = C.TLGroups(src=src, hashes=hashes, labels=labels)
    assert G.of(2, "doc") == {11, 14}                                   # its own file's doc and its twin's
    assert G.of(5, "doc") == {11, 14}
    assert G.of(1, "doc") == {10}
    assert G.of(2, "topic") == {1, 3}
    # a held row whose TWIN lies in a training document is forbidden
    rec = fake_rec([4, 3, 5], train=[0], held=[1, 2])
    pl = C.tl_plan(rec, [], lambda i: G.of(i, "doc"))
    assert G.of(4, "doc") == {13} and pl["orig_kept_idx"] == [1, 2]
    rec = fake_rec([2, 3, 6], train=[0], held=[1, 2])                   # train id 2 -> docs {11, 14}
    pl = C.tl_plan(rec, [5], lambda i: G.of(i, "doc"))
    assert pl["counts"]["scan_allowed"] == 0                            # scan id 5 = twin of the training row


def test_pub_rebuild_disjoint_and_training_untouched():
    rec = fake_rec([1, 2, 3, 4, 5, 6], train=[0, 1, 2], held=[3, 4, 5], docs=[7, 8, 9, 7, -1, 10],
                   mid=([21, 22, 23], [0], [1, 2], [11, 8, 12]))
    r2, info = C.pub_rebuild(rec, "drop")
    assert r2["strong"]["held"] == [4, 5]                               # doc 7 dropped; unknown doc (-1) kept
    assert info["n_old_leaky"] == 1 and info["unknown_doc_held"] == 1 and not info["unscorable"]
    assert r2["mid"]["held"] == [2]                                     # mid doc 8 = a strong-train doc: dropped
    assert rec["strong"]["held"] == [3, 4, 5] and rec["mid"]["held"] == [1, 2]   # input not mutated
    for k in ("train", "ids", "doc"):
        assert r2["strong"][k] == rec["strong"][k]
    assert torch.equal(r2["strong"]["pos"], rec["strong"]["pos"]) and r2["D_train"] == rec["D_train"]
    docs = {1: 7, 2: 8, 3: 9, 4: 7, 5: -1, 6: 10, 21: 11, 22: 8, 23: 12}
    leak, dup = C.check_disjoint(r2, lambda i: set() if docs[i] < 0 else {docs[i]})
    assert (leak, dup) == (0, 0)
    leak0, _ = C.check_disjoint(rec, lambda i: set() if docs[i] < 0 else {docs[i]})
    assert leak0 == 1                                                    # the check does catch a leaky set


def test_pub_mid_all_leaky_is_kept_and_flagged():
    rec = fake_rec([1, 2, 3], train=[0, 1], held=[2], docs=[7, 8, 9], mid=([21, 22], [0], [1], [11, 7]))
    r2, info = C.pub_rebuild(rec, "drop")
    assert r2["mid"]["held"] == [1] and info["mid_all_leaky_kept"]
    r3, info3 = C.pub_rebuild(rec, "keep")
    assert r3["mid"]["held"] == [1] and not info3["mid_all_leaky_kept"] and info3["mid_leaky"] == 1


def test_drop_mid_tl():
    rec = fake_rec([11, 21, 55], train=[0, 1], held=[2], mid=([51, 12, 71], [0], [1, 2], [0, 0, 0]))
    forb = C.forbidden_groups(C.training_ids(rec), gof)
    m2, info = C.drop_mid(rec, gof, forb)
    assert m2["held"] == [2] and info["mid_held_old"] == 2 and info["mid_held_new"] == 1


def test_leak_counts_contrast():
    rec = fake_rec([11, 21, 55], train=[0, 1], held=[2], neg_ids=[101, 102, 15, 105])   # 3 train / 1 held contrast
    lk = C.leak_counts(rec, gof)
    assert lk["contrast_held"] == 1 and lk["contrast_leaky_vs_contrast_train"] == 1     # 105: doc 10 = 101's doc
    assert lk["contrast_leaky_vs_activating_train"] == 0 and lk["strong_leaky"] == 0


# --------------------------------------------------------------------------------------------- determinism
def test_plan_and_pub_rebuild_deterministic():
    rng = np.random.default_rng(0)
    ids = list(rng.choice(np.arange(1, 400), size=64, replace=False))
    rec = fake_rec(ids, train=list(range(48)), held=list(range(48, 64)), docs=[i // 10 for i in ids])
    scan = list(rng.choice(np.arange(1, 400), size=200, replace=False))
    a, b = C.tl_plan(rec, scan, gof), C.tl_plan(rec, scan, gof)
    assert a == b
    r1, i1 = C.pub_rebuild(rec)
    r2, i2 = C.pub_rebuild(rec)
    assert i1 == i2 and r1["strong"]["held"] == r2["strong"]["held"] and r1["docdisjoint"] == r2["docdisjoint"]


# --------------------------------------------------------------------------------------------- resume
def test_done_keys_skips_truncated_and_error_rows(tmp_path):
    p = tmp_path / "build_doc.jsonl"
    p.write_text(json.dumps(dict(seed="a")) + "\n" + json.dumps(dict(seed="b", error="x")) + "\n" + '{"seed": "c"')
    (tmp_path / "build_doc.shard1.jsonl").write_text(json.dumps(dict(seed="d")) + "\n")
    assert C.done_keys(p) == {"a", "d"}
    assert C.shard_path(p, 1, 4).name == "build_doc.shard1.jsonl" and C.shard_path(p, 0, 1) == p


def _fake_public_tree(tmp_path, monkeypatch, keys):
    out = tmp_path / "out_gpt2-small"
    (out / "main" / "circuits").mkdir(parents=True)
    (out / "ctx").mkdir()
    (out / "WCM_VERSION").write_text("v4\n")
    (out / "WCM_R").write_text("r\n")
    for n, k in enumerate(keys):
        torch.save({"a": 1}, out / "main" / "circuits" / ("%s.pt" % k))
        held = [3, 4, 5] if n != 2 else [3]
        docs = [7, 8, 9, 7, 70 + n, 10] if n != 2 else [7, 8, 9, 7, 1, 1]  # target 2's only held row is leaky
        torch.save({k: fake_rec([1, 2, 3, 4, 5, 6], train=[0, 1, 2], held=held, docs=docs)}, out / "ctx" / ("%s.pt" % k))
    monkeypatch.setenv("PUB_OUT_GPT2_SMALL", str(out))
    monkeypatch.setattr(C, "WORK", tmp_path / "work")
    return out


def test_public_plan_resumes_per_target(tmp_path, monkeypatch):
    keys = ["1.attn.1", "2.mlp.2", "3.resid.3"]
    _fake_public_tree(tmp_path, monkeypatch, keys)
    import rebuild as RB
    monkeypatch.setattr(RB, "MID_RULE", "drop")
    RB.plan_pub("gpt2-small", RB.production_targets("gpt2-small"))
    path = C.WORK / "gpt2-small" / "build_doc.jsonl"
    assert len(C.read_jsonl(path)) == 3
    RB.plan_pub("gpt2-small", keys)                                     # nothing to redo
    assert len(C.read_jsonl(path)) == 3
    (C.WORK / "gpt2-small" / "ctx_doc" / "2.mlp.2.pt").unlink()         # lose one record + a torn line
    with open(path, "a") as fh:
        fh.write('{"seed": "x')
    RB.plan_pub("gpt2-small", keys)
    rows = C.read_jsonl(path)
    assert [r["seed"] for r in rows] == keys + ["2.mlp.2"]
    new = torch.load(C.WORK / "gpt2-small" / "ctx_doc" / "3.resid.3.pt", weights_only=False)["3.resid.3"]
    assert new["strong"]["held"] == [] and rows[2]["unscorable"]


def test_rescore_prep_lists_only_scorable_with_circuits(tmp_path, monkeypatch):
    keys = ["1.attn.1", "2.mlp.2", "3.resid.3"]
    out = _fake_public_tree(tmp_path, monkeypatch, keys)
    import rebuild as RB
    import rescore as RS
    RB.plan_pub("gpt2-small", keys)
    torch.save({"4.attn.4": fake_rec([1], [0], [])}, out / "ctx" / "4.attn.4.pt")   # a record without a circuit
    monkeypatch.setattr(RS, "LINK", "copy")
    monkeypatch.delenv("VARIANTS", raising=False)
    RS.prep("gpt2-small")
    assert C.read_targets(C.WORK / "gpt2-small" / "targets_orig.txt") == keys
    assert C.read_targets(C.WORK / "gpt2-small" / "targets_doc.txt") == keys[:2]
    info = json.loads((C.WORK / "gpt2-small" / "prep_doc.json").read_text())
    assert info["excluded"] == {"unscorable_empty_heldout": 1}
    o = C.variant_out("gpt2-small", "doc")
    assert (o / "WCM_VERSION").read_text() == "v4\n" and (o / "WCM_R").exists()
    assert sorted(p.stem for p in (o / "main" / "circuits").glob("*.pt")) == keys[:2]
    RS.prep("gpt2-small")                                               # idempotent
    assert C.read_targets(C.WORK / "gpt2-small" / "targets_doc.txt") == keys[:2]


# --------------------------------------------------------------------------------------------- report logic
def _frame(passes, vac=None, bands=None, kinds=None):
    n = len(passes)
    idx = ["%d.attn.%d" % (i % 12, i) for i in range(n)]
    return pd.DataFrame({"pass": passes, "vac": vac if vac is not None else [False] * n,
                         "band": bands or ["L0-4"] * n, "kind": kinds or ["attn"] * n}, index=idx)


def test_paired_counts_and_conservative():
    import report as R
    old = _frame([True, True, True, False, False, True], vac=[False] * 5 + [True])
    new = _frame([True, False, True, True, False, True])
    new = new.drop(new.index[2])                                         # target 2 not scored on the new set
    pr, s = R.paired(old, new, unscorable=[old.index[2]])
    assert s["n_old"] == 5 and s["n_pair"] == 4                          # target 5 vacuous on old; target 2 missing
    assert (s["pass_to_fail"], s["fail_to_pass"]) == (1, 1)
    assert s["old"]["k"] == 2 and s["new"]["k"] == 2 and s["diff_pp"] == 0
    assert s["unscorable"] == 1 and s["new_missing"] == 1
    assert s["cons_new"]["k"] == 2 and s["cons_old"]["k"] == 3 and s["cons_diff_pp"] == pytest.approx(-20.0)
    assert s["diff_lo"] <= s["diff_pp"] <= s["diff_hi"]


def test_paired_new_vacuous_excluded_from_pair_but_fails_conservative():
    import report as R
    old = _frame([True, True])
    new = _frame([True, True], vac=[False, True])
    _, s = R.paired(old, new)
    assert s["n_pair"] == 1 and s["new_vacuous"] == 1 and s["cons_new"]["k"] == 1


def test_cells_margins_add_up():
    import report as R
    bands = ["L0-4", "L5-7", "L8-11", "L0-4", "L5-7", "L8-11"] * 3
    kinds = ["attn"] * 6 + ["mlp"] * 6 + ["resid"] * 6
    old = _frame([True] * 18, bands=bands, kinds=kinds)
    new = _frame([True, False] * 9, bands=bands, kinds=kinds)
    pr, s = R.paired(old, new)
    ce = R.cells(pr, "turingllm")
    allrow = ce[(ce.band == "all") & (ce.kind == "all")].iloc[0]
    assert allrow.n_pair == 18 and allrow.pass_to_fail == 9 and allrow.diff_pp == pytest.approx(-50.0)
    assert ce[(ce.band != "all") & (ce.kind == "all")].n_pair.sum() == 18
    assert ce[(ce.band == "all") & (ce.kind != "all")].n_pair.sum() == 18
    assert ce[(ce.band != "all") & (ce.kind != "all")].n_pair.sum() == 18
    assert len(ce) == 1 + 3 + 3 + 9


def test_rate_uses_lib109_bootstrap():
    import report as R
    c = R.rate(pd.Series([True] * 80 + [False] * 20))
    assert c["k"] == 80 and c["n"] == 100 and c["pct"] == pytest.approx(80.0) and c["lo"] < 80 < c["hi"]
    assert R.rate(pd.Series([True] * 80 + [False] * 20)) == c                 # fixed seed: deterministic


# --------------------------------------------------------------------------------------------- pilot equivalence
P115 = HERE.parents[0] / "115-docdisjoint-rescore-pilot" / "out"


@pytest.mark.skipif(not (P115 / "build_doc.jsonl").exists() or not C.tl_paths()["scan"].exists()
                    or not (C.tl_paths()["corpus"] / "corpus_src.npy").exists(), reason="real data absent")
def test_plan_reproduces_pilot_group_clause_counts():
    """The CPU plan's group clause gives the 115 pilot's counts (built on the GPU) for all 36 pilot targets, both
    groups: allowed originals, allowed scan rows, old leaky rows, forbidden groups."""
    sys.path.insert(0, str(C.EXP / "086-heldout-rebuild"))
    import scan as SC
    G = C.TLGroups()
    sc = SC.load_scan(C.tl_paths()["scan"])
    n = 0
    for g in ("doc", "topic"):
        for line in open(P115 / ("build_%s.jsonl" % g)):
            want = json.loads(line)
            key = want["seed"]
            rec = torch.load(C.tl_paths()["ctx"] / ("%s.pt" % key), weights_only=False)[key]
            pl = C.tl_plan(rec, sc["rows"](key)["ids"].tolist(), lambda i, _g=g: G.of(i, _g))
            assert pl["counts"]["orig_allowed"] == want["orig_allowed"], key
            assert pl["counts"]["scan_allowed"] == want["scan_allowed"], key
            assert pl["counts"]["scan_total"] == want["scan_total"], key
            assert len(rec["strong"]["held"]) - pl["counts"]["orig_allowed"] == want["n_old_leaky"], key
            assert len(pl["forbidden"]) == want["n_forbidden_groups"], key
            if want["n_held"] == want["n_want"]:
                assert pl["counts"]["orig_allowed"] + pl["counts"]["scan_allowed"] >= want["n_want"]
            G.clear()
            n += 1
    assert n == 72
