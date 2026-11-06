"""Arena C donor test for task circuits (donor_tasks.py), CPU. In WSL from the repo root:

  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= ./.venv/bin/python -m pytest experiments/107-true-baselines/arena_c/test_donor_tasks.py -q -s
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import donor_tasks as D  # noqa: E402
import wcm_marks as WM  # noqa: E402

SITES = WM.scope_sites("noembed")


@pytest.fixture(scope="module")
def ctx():
    inf, bank = WM.load_model(device="cpu")
    c = D.task_context("simple", "sfc", inf, bank, r_pool_n=64)
    return inf, bank, c


def test_self_donor_identity(ctx):
    inf, bank, c = ctx
    worst = 0.0
    for b in range(0, int(c["bases"].shape[0]), D.BASE_BATCH):
        bt = c["bases"][b:b + D.BASE_BATCH]
        g, bd = c["good"][b:b + D.BASE_BATCH], c["bad"][b:b + D.BASE_BATCH]
        donor = D.capture(inf, bank, bt, SITES)
        e = D.read(inf, bt, g, bd, D.DonorPatcher(bank, SITES, {}, {}, donor))
        worst = max(worst, float((e - c["natural"][b:b + D.BASE_BATCH]).abs().max()))
    print("RESULT self-donor empty circuit vs full model: max |d logit-diff| %.3g over %d bases" % (
        worst, int(c["bases"].shape[0])))
    assert worst < 1e-3


def test_full_circuit_recovers_one(ctx):
    inf, bank, c = ctx
    c2 = dict(c, sels={ds: v[:2] for ds, v in c["sels"].items()})         # 2 draws per pool keep the CPU test short
    full = {s: {i: 1.0 for i in range(WM.D_SAE)} for s in SITES}
    r = D.score_group(inf, bank, c2, SITES, [dict(alphas=full)])[0]
    rec = r["recovery"]
    print("RESULT full circuit recovery %s | empty leak %s" % (
        {ds: {v: (None if x is None else round(x, 6)) for v, x in d.items()} for ds, d in rec.items()},
        {ds: (None if x is None else round(x, 3)) for ds, x in r["empty_leak"].items()}))
    for ds in D.POOLS:
        for v in D.VARIANTS:
            assert rec[ds][v] is not None and abs(rec[ds][v] - 1.0) < 1e-4


def test_draw_determinism():
    sizes = dict(A=100, C=40, R=256)
    a, _ = D.draws_for("simple", sizes, 40)
    b, _ = D.draws_for("simple", sizes, 40)
    o, _ = D.draws_for("rc", sizes, 40)
    assert all(torch.equal(x, y) for ds in D.POOLS for x, y in zip(a[ds], b[ds]))
    assert any(not torch.equal(x, y) for ds in D.POOLS for x, y in zip(a[ds], o[ds]))
    assert all(len(a[ds]) == D.DRAWS and all(int(s.shape[0]) == 40 for s in a[ds]) for ds in D.POOLS)
    assert all(int(s.max()) < sizes[ds] for ds in D.POOLS for s in a[ds])
