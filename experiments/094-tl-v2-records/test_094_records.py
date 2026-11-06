"""CPU tests for 094: the 8k target list (frozen output, must-includes, quotas) and the record builder's pure parts
(the symmetric duplicate test, the greedy pool walk, the per-pool held-out fraction, the split over a deduped pool).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/094-tl-v2-records/test_094_records.py -q
"""
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))


def _load(name):
    spec = importlib.util.spec_from_file_location("e094_" + name, HERE / ("%s.py" % name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("build_records")
MT = _load("make_targets")
CFG = SimpleNamespace(holdout_frac=0.25, keep_top=2, pool_size=64, min_pool=8, train_strong=32, train_mid=16,
                      contrast_count=64, min_contrast=4)


# ----------------------------------------------------------------------------------------------- target list
def test_frozen_list_matches_manifest():
    man = json.loads((HERE / "targets_tl8k.manifest.json").read_text())
    for f, m in man["files"].items():
        assert hashlib.md5((HERE / f).read_bytes()).hexdigest() == m, f
    keys = (HERE / "targets_tl8k.txt").read_text().split()
    assert len(keys) == len(set(keys)) == 8000
    full = set((REPO / "experiments" / "062-h100-protocol-v1" / "targets_full.txt").read_text().split())
    assert set(keys) <= full
    for name in ("sweep190", "nulls126", "case6", "gallery9", "asrun1000", "stable128", "gate64"):
        sub = (HERE / ("targets_tl8k_%s.txt" % name)).read_text().split()
        assert set(sub) <= set(keys), name
    gate = (HERE / "targets_tl8k_gate64.txt").read_text().split()
    assert len(gate) == len(set(gate)) == 64 and len({MT.cell(k) for k in gate}) == 35   # spread over every cell
    assert man["files"]["targets_tl8k.txt"] == "84cb8370fd6e31e78bfbbfbc823507aa"         # the 8k list is unchanged
    assert "9.resid.20419" in keys
    per = Counter(MT.cell(k) for k in keys)
    assert per == Counter(man["quota"]) and len(per) == 35
    assert per["1.attn"] == 134 and set(per.values()) == {134, 231, 232}


def test_quotas_water_fill():
    import numpy as np
    q = MT.quotas({"0.mlp": 10, "1.attn": 2, "1.mlp": 10}, 15, np.random.default_rng(0))
    assert q["1.attn"] == 2 and sum(q.values()) == 15 and sorted(q.values()) == [2, 6, 7]


# ----------------------------------------------------------------------------------------------- duplicate test
def _row(seed, n=64):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(100, 50000, (n,), generator=g)


def test_dup_reason_symmetric():
    a, b = _row(1), _row(2)
    assert B.dup_reason(a, 10, b[None], [10]) == (None, None)
    assert B.dup_reason(a, 10, a[None].clone(), [10])[0] == "exact"
    # anchor-prefix under the OTHER row's anchor only: shared first 21 tokens, candidate anchor 40, other's anchor 20
    c = b.clone(); c[:21] = a[:21]
    assert B.dup_reason(c, 40, a[None], [20])[0] == "apfx"
    assert B.dup_reason(c, 40, a[None], [30])[0] is None
    # near in the OTHER direction: the other row's 8-grams are mostly inside the candidate, not vice versa
    d = _row(3); d[:40] = a[:40]                                      # 40-token shared run -> near either way (LCR)
    assert B.dup_reason(d, 63, a[None], [63])[0] in ("apfx", "near")
    r = _row(4)
    e = torch.cat([r[:1], a[1:32], r[1:3], a[34:64]])               # runs of 31 and 30: LCR < 32, c8 high, no prefix
    assert B.dup_reason(e, 0, a[None], [0])[0] == "near"


# ----------------------------------------------------------------------------------------------- walk
class FakeFire:
    """fire_fn: peak 1 at position 5 for every row, except ids in `silent` (peak 0)."""

    def __init__(self, silent=()):
        self.silent = set(silent)

    def __call__(self, cands):
        return [dict(act_at_anchor=0.0 if c.id in self.silent else 1.0, anchor=5,
                     peak=0.0 if c.id in self.silent else 1.0, mean=0.1) for c in cands]


def _cands(rows):
    from circuit.heldout_rule import Candidate
    return [Candidate(tokens=None, id=i + 1, origin="scan", info=dict(rank=i + 1)) for i in range(len(rows))]


def _loader(rows):
    def load(cs):
        for c in cs:
            c.tokens = rows[c.id - 1]
    return load


def test_walk_dedupes_and_skips_silent():
    base = [_row(10 + i) for i in range(6)]
    twin = base[1].clone()                                         # exact copy (a corpus twin)
    pfx = _row(99); pfx[:6] = base[2][:6]                          # anchor-prefix copy at anchor 5
    rows = [base[0], base[1], twin, base[2], pfx, base[3], base[4], base[5]]
    cs = _cands(rows)
    acc, ex = B.walk(cs, 4, FakeFire(silent={6}), _loader(rows), set())
    assert [cs[i].id for i in acc] == [1, 2, 4, 7]
    assert ex[2]["reason"] == "exact" and ex[4]["reason"] == "apfx" and ex[5]["reason"] == "silent"
    acc2, _ = B.walk(_cands(rows), 4, FakeFire(), _loader(rows), {1})       # excluded id (unstable)
    assert 0 not in acc2


def test_walk_against_fixed_rows():
    s = [_row(200 + i) for i in range(3)]
    rows = [s[0].clone(), _row(300), _row(301)]
    cs = _cands(rows)
    acc, ex = B.walk(cs, 8, FakeFire(), _loader(rows), set(), against=(torch.stack(s), [5, 5, 5]))
    assert [cs[i].id for i in acc] == [2, 3] and ex[0]["reason"] == "exact"


# ----------------------------------------------------------------------------------------------- split
def test_held_frac_mid_fixed16():
    assert B.held_frac(64, "strong", CFG) == 0.25
    assert B.held_frac(40, "strong", CFG) == 0.25
    assert B.held_frac(40, "mid", CFG) == pytest.approx(16 / 40)
    assert B.held_frac(31, "mid", CFG) == 0.25


def test_split_pools_dedup_pool_is_stratified():
    from circuit.context_split import stratified_split
    g = torch.Generator().manual_seed(7)
    rec = {}
    for name, n in (("strong", 64), ("mid", 40)):
        pos = torch.stack([_row(1000 * (name == "mid") + i) for i in range(n)])
        rec[name] = dict(pos=pos, arg=torch.full((n,), 5), peak=torch.rand(n, generator=g))
    assert B.split_pools(rec, CFG)                                  # all singletons
    tr, ho = stratified_split(rec["strong"]["peak"], holdout_frac=0.25, keep_top=2)
    assert rec["strong"]["train"] == tr and rec["strong"]["held"] == ho and len(ho) == 16
    m = rec["mid"]
    assert len(m["held"]) == 16 and len(m["train"]) == 24 and not m["unused"]
    assert not set(m["train"]) & set(m["held"])
