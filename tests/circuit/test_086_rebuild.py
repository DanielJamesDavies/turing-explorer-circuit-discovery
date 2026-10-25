"""DAN-157 rebuild (experiments/086-heldout-rebuild/rebuild.py) on a synthetic record: the rebuilt record keeps every
training-side field (rec["neg"] included: the fit's contrast rows), replaces the held-out strongest set under the rule
(twins, mid-band rows, silent rows; no source-file clause), tops up from the scan in order, flags the floor, dedupes
the contrast held-out set (vs contrast-train, vs activating rows, vs itself; refill from the selector; keep fewer),
and the independent verifier passes it and catches tampering."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from circuit import heldout_rule as HR
from circuit.context_split import stratified_split, stratified_subset

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("rebuild_086", REPO / "experiments" / "086-heldout-rebuild" / "rebuild.py")
RB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RB)

KEY = "3.mlp.7"
N_ID = 2000
SILENT = set()


def toks(i):
    g = torch.Generator().manual_seed(1000 + int(i))
    t = torch.randint(1, 30000, (64,), generator=g)
    if i in SILENT:
        t = t % 15000                                   # never above the firing threshold
    return t


TOK = {}


def tok(i):
    return TOK[i] if i in TOK else toks(i)


class FakeE:
    """Fires wherever a token is > 15000. contrast_pool: the run's contrast ids, then REFILL (the selector's next)."""
    REFILL = []
    calls = []

    def load(self, ids):
        pos = torch.stack([tok(i) for i in ids])
        tgt = torch.zeros_like(pos)
        tgt[:, :63] = pos[:, 1:]
        return pos, tgt

    def acts(self, key, tokens):
        t = tokens.cpu().float()
        return torch.where(t > 15000, t / 30000, torch.zeros_like(t))

    def contrast_pool(self, key, n_max):
        FakeE.calls.append(n_max)
        ids = (list(range(201, 265)) + list(FakeE.REFILL))[:n_max]
        return ids, torch.stack([tok(i) for i in ids])


def corpus():
    c = RB.Corpus.__new__(RB.Corpus)
    c.hash = np.zeros(N_ID, dtype=np.uint64)
    return c


def finish_corpus(c):
    for i in range(1, N_ID):
        c.hash[i] = HR.window_hash(tok(i))[0]


def pool(ids):
    pos = torch.stack([tok(i) for i in ids])
    tgt = torch.zeros_like(pos); tgt[:, :63] = pos[:, 1:]
    a = FakeE().acts(KEY, pos)
    peak, arg = a.max(-1)
    tr, ho = stratified_split(peak)
    return dict(pos=pos, tgt=tgt, arg=arg, peak=peak.float(), train=tr, held=ho, ids=list(ids))


def make(n_scan=200, refill=True, contrast_leaks=True):
    TOK.clear(); SILENT.clear(); FakeE.calls.clear()
    c = corpus()
    s_ids, m_ids, n_ids = list(range(1, 65)), list(range(101, 165)), list(range(201, 265))
    rec = dict(n_top=64, n_mid=64)
    rec["strong"] = pool(s_ids)
    s = rec["strong"]
    h0, h1, h2 = s["held"][0], s["held"][1], s["held"][2]
    t0 = s["train"][5]
    # a leak into the original held-out set: an exact twin of a training row (other id)
    TOK[s_ids[h0]] = tok(s_ids[t0]).clone()
    for k in ("pos", "tgt", "arg", "peak"):
        s[k][h0] = s[k][t0]
    # a strongest-train row that fires at one position only: a silent contrast row can be its near copy
    t1 = s["train"][10]
    one = toks(s_ids[t1]) % 15000
    one[40] = 20000
    TOK[s_ids[t1]] = one
    s["pos"][t1] = one
    s["tgt"][t1, :63] = one[1:]
    s["arg"][t1], s["peak"][t1] = 40, float(20000 / 30000)
    rec["mid"] = pool(m_ids)
    m = rec["mid"]
    hv = stratified_split(s["peak"][s["train"]], holdout_frac=1 / 3)[0]
    rec["D_train"] = [s["train"][j] for j in hv]
    rec["B_mid"] = [m["train"][j] for j in stratified_subset(m["peak"][m["train"]], 16)]
    neg = torch.stack([tok(i) % 15000 for i in n_ids])
    if contrast_leaks:
        neg[50] = neg[5]                                    # a held-out contrast row = a contrast-train row (exact)
        neg[53] = one.clone(); neg[53, 40] = 7              # near copy of an activating row, silent
        neg[55] = neg[54]                                   # two held-out contrast rows identical (exact_H)
    rec["neg"] = neg
    rec["neg_ids"] = n_ids
    for i, nid in enumerate(n_ids):
        TOK[nid] = rec["neg"][i]
    # contrast refill: a twin of the run's own contrast row (filtered), a near copy of an activating row, a firing row,
    # then clean silent rows
    FakeE.REFILL = [601, 602, 603] + list(range(604, 640)) if refill else []
    TOK[601] = neg[0].clone()
    TOK[602] = one.clone(); TOK[602][40] = 9
    TOK[603] = toks(603)
    for i in range(604, 640):
        TOK[i] = toks(i) % 15000
    # scan: pool rows (skipped), a twin of a train row, a mid-pool row, a (formerly same-source) row, a silent row
    sc = [s_ids[s["train"][0]], s_ids[h2], 401, 402, m_ids[m["held"][0]], 403, 404] + list(range(500, 500 + n_scan))
    TOK[401] = tok(s_ids[s["train"][3]])
    SILENT.add(403)
    finish_corpus(c)
    scan_rows = dict(ids=torch.tensor(sc), mean=torch.linspace(1, 0.1, len(sc)), peak=torch.ones(len(sc)),
                     arg=torch.zeros(len(sc), dtype=torch.int16))
    return rec, scan_rows, c, (h0, h1)


def test_rebuild_rule_and_schema():
    rec, sc, c, (h0, h1) = make()
    r2, row = RB.build_one(KEY, rec, sc, c, FakeE(), dict(n_top=512, partial=False))
    s0, s1 = rec["strong"], r2["strong"]
    # training side untouched; the first 64 rows are the original's; the fit's contrast rows too
    assert s1["train"] == s0["train"]
    for k in ("pos", "tgt", "arg", "peak"):
        assert torch.equal(s1[k][:64], s0[k])
    assert r2["D_train"] is rec["D_train"] and r2["B_mid"] is rec["B_mid"] and r2["neg"] is rec["neg"]
    assert r2["neg_ids"] is rec["neg_ids"] and r2["mid"]["train"] == rec["mid"]["train"]
    hv = r2["heldout_v2"]
    assert hv["rule"] == HR.RULE_VERSION and hv["n_pool_orig"] == 64 and "variant" not in hv
    held = s1["held"]
    assert len(held) == 16 and row["n_held"] == 16 and not row["below_floor"]
    assert h0 not in held                                            # the twin is gone
    assert h1 in held                                                # no source-file clause: a clean row stays
    orig = [j for j in held if j < 64]
    assert orig == [j for j in s0["held"] if j in set(orig)]          # originals first, original order
    assert held == orig + list(range(64, 64 + 16 - len(orig)))
    new_ids = s1["ids"][64:]
    assert not {401, 403} & set(new_ids)                             # twin / silent scan rows rejected
    assert not set(rec["mid"]["ids"]) & set(new_ids)                 # mid-band rows never top up the strongest set
    assert new_ids == [402, 404, 500, 501, 502][:len(new_ids)]       # strongest admissible scan rows, in order
    prov = {p["idx"]: p for p in hv["strong"]["rows"]}
    assert set(prov) == set(held)
    assert all(prov[j]["origin"] == ("original" if j < 64 else "scan") for j in held)
    assert all(prov[j]["act_at_anchor"] > 0 and "src" not in prov[j] for j in held)
    assert RB.verify_one(KEY, rec, r2, c, FakeE()) == []


def test_contrast_dedupe_and_refill():
    rec, sc, c, _ = make()
    r2, row = RB.build_one(KEY, rec, sc, c, FakeE(), dict(n_top=512))
    cv = r2["heldout_v2"]["contrast"]
    ids = r2["neg_held_v2_ids"]
    assert r2["neg_held_v2"].shape == (16, 64) and len(ids) == 16 and cv["n_train"] == 48
    kept = [201 + j for j in range(48, 64) if j not in (50, 53, 55)]
    assert ids[:13] == kept                                          # originals first, record order, leaks removed
    assert ids[13:] == [604, 605, 606]                               # refill: twin filtered, near / firing rejected
    assert torch.equal(r2["neg_held_v2"][:13], rec["neg"][[j for j in range(48, 64) if j not in (50, 53, 55)]])
    assert (cv["n_orig_kept"], cv["n_refill"], cv["below_floor"]) == (13, 3, False)
    assert cv["reasons"].get("forbidden_hash", 0) >= 1 and cv["reasons"]["near_F"] == 2 and cv["reasons"]["fires"] == 1
    assert cv["reasons"]["exact_H"] == 1 and cv["refill_fetched"] and cv["selector_prefix_ok"] is True
    assert FakeE.calls == [64 + RB.NEG_EXTRA]
    assert (row["neg_n_held"], row["neg_n_refill"]) == (16, 3)
    assert RB.verify_one(KEY, rec, r2, c, FakeE()) == []


def test_contrast_keep_fewer_and_no_refill_when_clean():
    rec, sc, c, _ = make(refill=False)
    r2, row = RB.build_one(KEY, rec, sc, c, FakeE(), dict(n_top=512))
    assert r2["neg_held_v2"].shape[0] == 13 and row["neg_n_held"] == 13 and not row["neg_below_floor"]
    assert RB.verify_one(KEY, rec, r2, c, FakeE()) == []
    rec, sc, c, _ = make(contrast_leaks=False)
    r2, row = RB.build_one(KEY, rec, sc, c, FakeE(), dict(n_top=512))
    assert torch.equal(r2["neg_held_v2"], rec["neg"][48:]) and r2["neg_held_v2_ids"] == list(range(249, 265))
    assert FakeE.calls == [] and not r2["heldout_v2"]["contrast"]["refill_fetched"]   # the selector never ran


def test_floor_and_scan_exhaustion():
    rec, sc, c, _ = make(n_scan=2)
    r2, row = RB.build_one(KEY, rec, sc, c, FakeE(), dict(n_top=512))
    n = len(r2["strong"]["held"])
    assert row["n_held"] == n and row["below_floor"] == (n < 8)
    assert row["scan_exhausted"] == (n < 16)
    assert RB.verify_one(KEY, rec, r2, c, FakeE()) == []


def test_verifier_catches_tampering():
    rec, sc, c, _ = make()
    r2, _ = RB.build_one(KEY, rec, sc, c, FakeE(), dict(n_top=512))
    bad = dict(r2, strong=dict(r2["strong"]))
    j = rec["strong"]["train"][0]
    bad["strong"]["held"] = [j] + r2["strong"]["held"][1:]
    rows = [dict(p) for p in r2["heldout_v2"]["strong"]["rows"]]
    rows[0]["idx"] = j
    bad["heldout_v2"] = dict(r2["heldout_v2"], strong=dict(r2["heldout_v2"]["strong"], rows=rows))
    f = RB.verify_one(KEY, rec, bad, c, FakeE())
    assert any("non-held" in x for x in f) and any("forbidden" in x for x in f)
    bad2 = dict(r2, D_train=list(r2["D_train"])[::-1])
    assert any("D_train" in x for x in RB.verify_one(KEY, rec, bad2, c, FakeE()))
    # the fit's contrast rows must stay the run's
    neg = rec["neg"].clone(); neg[0, 0] += 1
    assert any("field neg changed" in x for x in RB.verify_one(KEY, rec, dict(r2, neg=neg), c, FakeE()))
    # a contrast-train row smuggled into the contrast held-out set
    nh = r2["neg_held_v2"].clone(); nh[0] = rec["neg"][3]
    nid = list(r2["neg_held_v2_ids"]); nid[0] = 204
    cr = [dict(p) for p in r2["heldout_v2"]["contrast"]["rows"]]; cr[0]["id"] = 204
    bad3 = dict(r2, neg_held_v2=nh, neg_held_v2_ids=nid,
                heldout_v2=dict(r2["heldout_v2"], contrast=dict(r2["heldout_v2"]["contrast"], rows=cr)))
    f3 = RB.verify_one(KEY, rec, bad3, c, FakeE())
    assert any("contrast 0: a contrast-train id" in x for x in f3) and any("contrast 0: forbidden content" in x for x in f3)
    # a firing row in the contrast held-out set
    nh4 = r2["neg_held_v2"].clone(); nh4[2, 5] = 29000
    f4 = RB.verify_one(KEY, rec, dict(r2, neg_held_v2=nh4), c, FakeE())
    assert any("contrast 2: fires" in x for x in f4)


def test_row_silent_in_its_final_batch_is_replaced():
    """A row near the Top-K cut fires in its selection batch but not in the final set's batch: it must be replaced."""
    rec, sc, c, _ = make()
    fragile = 402                                            # the strongest admissible scan row
    seen = []

    class Fragile(FakeE):
        def acts(self, key, tokens):
            A = super().acts(key, tokens)
            for r in range(tokens.shape[0]):
                if torch.equal(tokens[r].cpu(), tok(fragile)):
                    seen.append(1)
                    if len(seen) > 1:                        # fires in the selection forward only
                        A[r] = 0.0
            return A
    r2, row = RB.build_one(KEY, rec, sc, c, Fragile(), dict(n_top=512))
    assert fragile not in r2["strong"]["ids"][64:]
    assert fragile in r2["heldout_v2"]["strong"]["unstable_fire"] and row["n_held"] == 16
    assert RB.verify_one(KEY, rec, r2, c, Fragile()) == []


def test_contrast_row_firing_in_its_final_batch_is_replaced():
    rec, sc, c, _ = make()
    flaky = 605
    seen = []

    class Flaky(FakeE):
        def acts(self, key, tokens):
            A = super().acts(key, tokens)
            for r in range(tokens.shape[0]):
                if torch.equal(tokens[r].cpu(), tok(flaky)):
                    seen.append(1)
                    if len(seen) == 2:                       # silent when selected, fires in the final batch once
                        A[r, 3] = 0.5
            return A
    r2, row = RB.build_one(KEY, rec, sc, c, Flaky(), dict(n_top=512))
    cv = r2["heldout_v2"]["contrast"]
    assert flaky in cv["unstable_silence"] and flaky not in r2["neg_held_v2_ids"]
    assert r2["neg_held_v2_ids"][13:] == [604, 606, 607] and row["neg_n_held"] == 16


def test_skip_record_copied():
    rec = dict(n_top=3, n_mid=0, strong=None, mid=None)
    r2, row = RB.build_one(KEY, rec, None, corpus(), FakeE(), {})
    assert r2["strong"] is None and r2["heldout_v2"]["skip"] and row["skip"] == "too_few_contexts"


def test_verify_predicates_match_the_rule():
    """The verifier's numpy predicates (independent code) agree with heldout_rule.pair_stats."""
    g = torch.Generator().manual_seed(3)
    base = torch.randint(1, 50, (64,), generator=g)                  # small vocabulary: many partial matches
    O = torch.stack([torch.roll(base, k) for k in range(0, 40, 3)] + [torch.randint(1, 50, (64,), generator=g)
                                                                      for _ in range(10)])
    ex, pfx, lcr, c8 = HR.pair_stats(base[None], O)
    assert (RB._lcr_vs(base.numpy(), O.numpy()) == lcr[0]).all()
    assert np.allclose(RB._c8_vs(base.numpy(), O.numpy()), c8[0])
    assert (RB._pfx_vs(base.numpy(), O.numpy()) == pfx[0]).all()
