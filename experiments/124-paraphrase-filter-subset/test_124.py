"""CPU unit tests for 124 (DAN-217): the paraphrase rule, determinism, resume, report pairing.

  python -X utf8 -m pytest -q -p no:cacheprovider experiments/124-paraphrase-filter-subset/test_124.py
The 114-reproduction tests read the real records / IDF cache and are skipped when they are absent.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import para124 as P  # noqa: E402

C = P.C
NF = P.K114.NF


# --------------------------------------------------------------------------------------------- fakes
def flat_tf():
    return P.tfidf_from_idf(np.ones(NF), 1)


RNG = np.random.default_rng(7)
BASE = {i: RNG.integers(100, 50000, size=64) for i in range(1, 2000)}       # independent random rows per id


def para_of(row, n_change=6, seed=0):
    """A light rewrite: n_change tokens replaced (cos stays far above 0.3 under unigram+bigram TF-IDF)."""
    r = row.copy()
    pos = np.random.default_rng(seed).choice(64, size=n_change, replace=False)
    r[pos] = 77777 + np.arange(n_change)
    return r


class FakeCorpus:
    def __init__(self, rows):
        self.rows = rows

    def windows(self, ids):
        return np.stack([self.rows[int(i)] for i in ids])


def make_rec(train_ids, held_ids, rows, mid=None, neg=None):
    ids = list(train_ids) + list(held_ids)
    s = dict(pos=torch.tensor(np.stack([rows[i] for i in ids])), tgt=torch.zeros(len(ids), 64, dtype=torch.long),
             arg=torch.zeros(len(ids), dtype=torch.long), peak=torch.ones(len(ids)), ids=ids,
             train=list(range(len(train_ids))), held=list(range(len(train_ids), len(ids))), unused=[])
    rec = dict(strong=s, D_train=s["train"][:2], B_mid=[], neg_ids=None,
               neg=torch.tensor(np.stack(neg)) if neg is not None else torch.zeros(0, 64, dtype=torch.long))
    if mid is not None:
        m_train, m_held = mid
        mids = list(m_train) + list(m_held)
        rec["mid"] = dict(pos=torch.tensor(np.stack([rows[i] for i in mids])), ids=mids,
                          train=list(range(len(m_train))), held=list(range(len(m_train), len(mids))),
                          arg=torch.zeros(len(mids), dtype=torch.long))
    return rec


def world():
    """train ids 1-4; held ids 11 (paraphrase of 1), 12 (paraphrase of mid-train 31), 13, 14 (unrelated);
    scan 21 (paraphrase of 2), 22, 23 (unrelated), 1 (in the pool: skipped); mid held 32 (paraphrase of 3), 33."""
    rows = dict(BASE)
    rows[11] = para_of(rows[1], seed=1)
    rows[12] = para_of(rows[31], seed=2)
    rows[21] = para_of(rows[2], seed=3)
    rows[32] = para_of(rows[3], seed=4)
    rec = make_rec([1, 2, 3, 4], [11, 12, 13, 14], rows, mid=([31], [32, 33]))
    return rec, rows, [1, 21, 22, 23]


# --------------------------------------------------------------------------------------------- the rule
def test_pararule_flags_paraphrases_not_unrelated():
    tf = flat_tf()
    a = BASE[1]
    rule = P.ParaRule(tf, np.stack([a, BASE[2]]))
    cm = rule.cos_max(np.stack([a, para_of(a), BASE[3]]))
    assert cm[0] == pytest.approx(1.0, abs=1e-5) and cm[1] > 0.6 and cm[2] < 0.05
    assert rule.flag(np.stack([para_of(a), BASE[3]])).tolist() == [True, False]


def test_threshold_is_inclusive():
    tf = flat_tf()
    rule = P.ParaRule(tf, BASE[1][None])
    c = float(rule.cos_max(para_of(BASE[1], n_change=30)[None])[0])
    assert P.ParaRule(tf, BASE[1][None], thresh=c).flag(para_of(BASE[1], n_change=30)[None])[0]
    assert not P.ParaRule(tf, BASE[1][None], thresh=c + 1e-4).flag(para_of(BASE[1], n_change=30)[None])[0]


def test_pararule_matches_114_formula():
    """cos_max is 114 check.main's cT = (vH @ vT.T).toarray(); cT.max(axis=1), on the same Tfidf object."""
    rows = np.stack([BASE[i] for i in range(1, 60)])
    tf = P.K114.Tfidf(rows)
    H, T = rows[:10], rows[10:]
    want = (tf(H) @ tf(T).T).toarray().max(axis=1)
    assert np.array_equal(P.ParaRule(tf, T).cos_max(H), want)


def test_paragroups_interface():
    rec, rows, scan = world()
    G = P.ParaGroups(rec, flat_tf(), FakeCorpus(rows))
    assert all(G.of(i) == {"P"} for i in (1, 2, 3, 4, 31))                   # training ids (strong + mid train)
    assert G.of(11) == {"P"} and G.of(12) == {"P"}                            # paraphrase of strong / mid training
    assert G.of(13) == set() and G.of(14) == set() and G.of(-1) == set() and G.of(0) == set()
    assert G.of(21) == {"P"} and G.of(22) == set()                            # scan ids read from the corpus
    assert G.cos[11] > 0.3 > G.cos[13]


def test_plan_drops_paraphrases_and_refill_candidates_are_clean():
    rec, rows, scan = world()
    tf = flat_tf()
    G = P.ParaGroups(rec, tf, FakeCorpus(rows))
    pl = C.tl_plan(rec, scan, G.of)
    s = rec["strong"]
    assert [s["ids"][j] for j in pl["orig_kept_idx"]] == [13, 14]
    assert [i for _, i in pl["scan_allowed"]] == [22, 23]
    assert pl["counts"] == dict(orig_total=4, orig_allowed=2, scan_total=3, scan_allowed=2)
    row = P.plan_one(rec, scan, P.ParaGroups(rec, tf, FakeCorpus(rows)), tf)
    assert row["n_old_leaky"] == 2 and row["mid_leaky"] == 1 and row["mid_held"] == 2 and row["cand_allowed"] == 4


def test_mid_drop_and_check_disjoint():
    rec, rows, _ = world()
    G = P.ParaGroups(rec, flat_tf(), FakeCorpus(rows))
    m2, info = C.drop_mid(rec, G.of, P.FORBIDDEN)
    assert [m2["ids"][j] for j in m2["held"]] == [33] and info["mid_held_new"] == 1
    leak, dup = C.check_disjoint(rec, G.of)
    assert (leak, dup) == (2, 0)                                              # the check catches the old set
    clean = dict(rec, strong=dict(rec["strong"], held=[6, 7]))                # ids 13, 14 only
    assert C.check_disjoint(clean, G.of) == (0, 0)
    assert P.training_unchanged(clean, rec)


def test_contrast_measure():
    rows = dict(BASE)
    neg = [BASE[100 + i] for i in range(12)] + [para_of(BASE[100]), para_of(BASE[1]), BASE[200], BASE[201]]
    rec = make_rec([1, 2], [11], rows, neg=neg)
    c = P.contrast_para(rec, flat_tf())
    assert c == dict(contrast_held=4, contrast_para_vs_contrast_train=1, contrast_para_vs_activating_train=1)


# --------------------------------------------------------------------------------------------- determinism
def test_idf_sample_is_114s_first_draw():
    a, b = P.idf_sample_ids(), P.idf_sample_ids()
    assert np.array_equal(a, b) and len(a) == 100000 and a.min() >= 1 and a.max() <= P.N_WIN_114
    rng = np.random.default_rng(20261114)                                     # 114 check.py: RNG module-level,
    assert np.array_equal(a, np.sort(rng.integers(1, P.N_WIN_114 + 1, size=100000)))   # first call = IDF sample


def test_plan_deterministic():
    rec, rows, scan = world()
    tf = flat_tf()
    a = P.plan_one(rec, scan, P.ParaGroups(rec, tf, FakeCorpus(rows)), tf)
    b = P.plan_one(rec, scan, P.ParaGroups(rec, tf, FakeCorpus(rows)), tf)
    assert a == b


def test_stratified_subset_seeded_and_proportional():
    keys = ["%d.%s.%d" % (l, k, i) for l in range(12) for k in ("attn", "mlp", "resid") for i in range(40 + 7 * l)]
    df = pd.DataFrame(index=keys)
    df["kind"] = [k.split(".")[1] for k in keys]
    df["band"] = [("L0-4" if int(k.split(".")[0]) <= 4 else "L5-7" if int(k.split(".")[0]) <= 7 else "L8-11")
                  for k in keys]
    a, alloc = P.stratified_subset(df, 300, 1)
    b, _ = P.stratified_subset(df.sample(frac=1, random_state=3), 300, 1)    # row order does not matter
    c, _ = P.stratified_subset(df, 300, 2)
    assert a == b and a != c and len(a) == 300 == sum(alloc.values()) and len(set(a)) == 300
    sizes = df.groupby(["band", "kind"]).size()
    for (bd, kd), n in sizes.items():
        q = 300 * n / len(df)
        got = sum(1 for k in a if df.loc[k, "band"] == bd and df.loc[k, "kind"] == kd)
        assert abs(got - q) < 1 and got == alloc["%s/%s" % (bd, kd)]


# --------------------------------------------------------------------------------------------- resume
def test_plan_stage_resumes(tmp_path, monkeypatch):
    rec, rows, scan = world()
    keys = ["0.attn.1", "5.mlp.2", "9.resid.3"]
    import rebuild124 as RB
    monkeypatch.setattr(P, "WORK", tmp_path)
    monkeypatch.setattr(RB, "keys", lambda: keys)
    monkeypatch.setattr(RB, "load_rec", lambda k: rec)
    monkeypatch.setattr(P, "load_tfidf", flat_tf)
    monkeypatch.setattr(P, "Corpus", lambda: FakeCorpus(rows))
    fake = types.ModuleType("scan")
    fake.load_scan = lambda p: dict(rows=lambda k: {"ids": torch.tensor(scan)}, where=set(keys))
    monkeypatch.setitem(sys.modules, "scan", fake)
    RB.stage_plan()
    path = tmp_path / "plan_para.jsonl"
    assert [r["seed"] for r in C.read_jsonl(path)] == keys
    RB.stage_plan()                                                           # nothing to redo
    assert len(C.read_jsonl(path)) == 3
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2]) + "\n" + lines[2][:20])              # a kill mid-write of the last row
    RB.stage_plan()
    got = C.read_jsonl(path)
    assert [r["seed"] for r in got] == keys and got[2] == json.loads(lines[2])


def test_prep_lists_scorable_only_and_is_idempotent(tmp_path, monkeypatch):
    import rescore124 as RS
    src = tmp_path / "out_v2"
    (src / "main" / "circuits").mkdir(parents=True)
    (src / "WCM_VERSION").write_text("v4\n")
    (src / "WCM_R").write_text("r\n")
    ctx = tmp_path / "ctx_fresh"
    ctx.mkdir()
    keys = ["0.attn.1", "5.mlp.2", "9.resid.3", "10.attn.4"]
    rec, rows, _ = world()
    for k in keys[:3]:
        torch.save({"a": 1}, src / "main" / "circuits" / ("%s.pt" % k))
    for k in keys:
        torch.save({k: rec}, ctx / ("%s.pt" % k))
    work = tmp_path / "work"
    (work / "ctx_para").mkdir(parents=True)
    torch.save({keys[0]: rec}, work / "ctx_para" / ("%s.pt" % keys[0]))
    torch.save({keys[1]: dict(rec, strong=dict(rec["strong"], held=[]))}, work / "ctx_para" / ("%s.pt" % keys[1]))
    (tmp_path / "subset.txt").write_text("\n".join(keys) + "\n")
    monkeypatch.setenv("TL_OUT_V2", str(src))
    monkeypatch.setenv("TL_CTX", str(ctx))
    monkeypatch.setenv("SUBSET", str(tmp_path / "subset.txt"))
    monkeypatch.setattr(P, "WORK", work)
    monkeypatch.setattr(RS, "LINK", "copy")
    RS.prep()
    assert C.read_targets(work / "targets_orig.txt") == keys[:3]              # 10.attn.4 has no circuit
    assert C.read_targets(work / "targets_para.txt") == keys[:1]              # 5.mlp.2 empty, 9.resid.3 not built
    info = json.loads((work / "prep_para.json").read_text())
    assert info["excluded"] == {"unscorable_empty_heldout": 1, "no_record": 1, "no_circuit": 1}
    RS.prep()
    assert C.read_targets(work / "targets_para.txt") == keys[:1]
    assert (work / "rs_para" / "WCM_VERSION").read_text() == "v4\n"


# --------------------------------------------------------------------------------------------- report pairing
def _eval_rows(scores):
    """scores: {seed: (Z, A, C, nec)} -> 062-style eval rows (held strong)."""
    return [dict(seed=k, held="strong", free0_tk=z, freeM_topk_tk=a, freeN_topk_tk=c, phi_sup_blind_tk=n,
                 vacuous_tk=False) for k, (z, a, c, n) in scores.items()]


def test_report_pairs_from_driver_rows(tmp_path, monkeypatch):
    import report124 as RP
    good, bad = (1.0, 1.0, 1.0, 1.0), (0.5, 1.0, 1.0, 1.0)
    orig = {"0.attn.1": good, "1.mlp.2": good, "6.resid.3": bad, "9.attn.4": good, "10.mlp.5": bad}
    para = {"0.attn.1": good, "1.mlp.2": bad, "6.resid.3": good, "9.attn.4": good}   # 10.mlp.5 unscorable
    for v, sc in (("orig", orig), ("para", para)):
        d = tmp_path / ("rs_%s" % v) / "main"
        d.mkdir(parents=True)
        with open(d / "eval.jsonl", "w") as fh:
            for r in _eval_rows(sc):
                fh.write(json.dumps(r) + "\n")
    monkeypatch.setattr(P, "WORK", tmp_path)
    o, n = RP.variant_rows("orig"), RP.variant_rows("para")
    assert o["pass"].sum() == 3 and len(n) == 4
    pr, s = RP.R.paired(o, n, {"10.mlp.5"})
    assert s["n_old"] == 5 and s["n_pair"] == 4 and (s["pass_to_fail"], s["fail_to_pass"]) == (1, 1)
    assert s["diff_pp"] == 0 and s["unscorable"] == 1 and s["cons_new"]["k"] == 3 and s["cons_old"]["k"] == 3
    ce = RP.R.cells(pr, "turingllm")
    allrow = ce[(ce.band == "all") & (ce.kind == "all")].iloc[0]
    assert allrow.n_pair == 4 and ce[(ce.band != "all") & (ce.kind != "all")].n_pair.sum() == 4


# --------------------------------------------------------------------------------------------- 114 reproduction
ROWS114 = C.EXP / "114-paraphrase-check" / "out" / "rows.jsonl"


@pytest.mark.skipif(not P.idf_path().exists() or not ROWS114.exists() or not C.tl_paths()["ctx"].exists(),
                    reason="real data absent")
def test_cached_idf_reproduces_114():
    z = np.load(P.idf_path())
    assert str(z["ids_md5"]) == P.ids_md5(P.idf_sample_ids()) and int(z["n"]) == 100000
    import rebuild124 as RB
    out = RB.verify_idf(P.load_tfidf(), n_rec=25, write=False)
    assert out["ok"] and out["max_abs_diff"] == 0.0


@pytest.mark.skipif(not (C.REPO / "data" / "shard_0.npy").exists() or not C.tl_paths()["ctx"].exists(),
                    reason="real data absent")
def test_corpus_reader_matches_record_rows():
    """Scan candidates are read from the corpus; record rows from the record. Both must be the same tokens."""
    f = sorted(C.tl_paths()["ctx"].glob("*.pt"))[0]
    k, rec = next(iter(torch.load(f, weights_only=False).items()))
    s = rec["strong"]
    ids = [int(x) for x in s["ids"][:20]]
    assert np.array_equal(P.Corpus().windows(ids), np.asarray(s["pos"][:20]).astype(np.int64))
