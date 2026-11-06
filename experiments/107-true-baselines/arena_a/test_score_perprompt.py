"""CPU unit tests for score_perprompt.py's restriction (no model): firing mask, BOS drop, position handling, restrict,
top-k order and grid, Ge node positions, held-out order.
  .venv/bin/python -m pytest -q experiments/107-true-baselines/arena_a/test_score_perprompt.py"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_perprompt as P  # noqa: E402

ALPHAS = {(1, "z"): {5: 0.5, 9: 2.0, 11: 1.0}, (2, "tc"): {3: 1.5, 4: 1.0}}


def codes():
    # body positions 0..3 (anchor 2 -> position 3 must be ignored)
    z = torch.tensor([[0.0, 1.0, 0.0],      # pos 0: latent 9 fires
                      [0.2, 0.0, 0.0],      # pos 1: latent 5
                      [0.0, 0.0, -1.0],     # pos 2: nothing (negative is not firing)
                      [0.0, 0.0, 3.0]])     # pos 3 > anchor: latent 11 must NOT count
    tc = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.7, 0.0], [0.0, 9.0]])
    return {(1, "z"): ([5, 9, 11], z), (2, "tc"): ([3, 4], tc)}


def test_firing_mask_and_positions():
    lp = P.firing_latpos(codes(), anchor=2)
    assert lp == {(1, "z", 0, 9), (1, "z", 1, 5), (2, "tc", 2, 3)}
    assert P.firing_latpos(codes(), anchor=3) >= {(1, "z", 3, 11), (2, "tc", 3, 4)}
    assert P.firing_latpos(codes(), anchor=0) == {(1, "z", 0, 9)}


def test_bos_row_dropped():
    c = codes()
    bos = {s: (lats, torch.cat([torch.full((1, a.shape[1]), 5.0), a])) for s, (lats, a) in c.items()}
    assert P.firing_latpos(bos, anchor=2, bos_row=True) == P.firing_latpos(c, anchor=2)
    assert P.act_max(bos, 2, bos_row=True) == P.act_max(c, 2)
    # without dropping, the BOS row would make every latent fire at position 0
    assert len(P.firing_latpos(bos, anchor=2)) > 3


def test_act_max_ignores_positions_after_anchor():
    m = P.act_max(codes(), 2)
    assert m[(1, "z", 11)] == 0.0 and m[(2, "tc", 4)] == 0.0
    assert m[(1, "z", 9)] == 1.0 and m[(2, "tc", 3)] == pytest.approx(0.7)


def test_restrict_keeps_alphas_and_rejects_non_members():
    lp = P.firing_latpos(codes(), anchor=2)
    al, mem = P.restrict(ALPHAS, lp)
    assert mem == {(1, "z"): {5, 9}, (2, "tc"): {3}}
    assert al == {(1, "z"): {5: 0.5, 9: 2.0}, (2, "tc"): {3: 1.5}}
    with pytest.raises(ValueError):
        P.restrict(ALPHAS, {(1, "z", 0, 6)})
    assert P.restrict(ALPHAS, set()) == ({}, {})


def test_all_latpos_is_identity():
    lp = P.all_latpos(ALPHAS, 4)
    assert len(lp) == 5 * 4
    al, mem = P.restrict(ALPHAS, lp)
    assert al == ALPHAS and mem == {s: set(d) for s, d in ALPHAS.items()}
    assert P.sig(mem, al) == P.sig({s: set(d) for s, d in ALPHAS.items()}, ALPHAS)


def test_topk_order_and_grid():
    lp = P.firing_latpos(codes(), anchor=2)
    lats = sorted({(l, k, i) for l, k, _p, i in lp})
    order = P.topk_order(ALPHAS, P.act_max(codes(), 2), lats)
    # scores: (1,z,9) 2.0 x 1.0 = 2.0; (2,tc,3) 1.5 x 0.7 = 1.05; (1,z,5) 0.5 x 0.2 = 0.1
    assert order == [(1, "z", 9), (2, "tc", 3), (1, "z", 5)]
    assert P.latpos_of(lp, order[:1]) == {(1, "z", 0, 9)}
    assert P.k_grid(0) == [] and P.k_grid(1) == [] and P.k_grid(3) == [1, 2] and P.k_grid(8) == [1, 2, 4]
    assert P.k_grid(9) == [1, 2, 4, 8]


def test_topk_ties_by_key():
    al = {(0, "z"): {1: 1.0, 2: 1.0}}
    assert P.topk_order(al, {(0, "z", 1): 1.0, (0, "z", 2): 1.0}, [(0, "z", 2), (0, "z", 1)]) == [(0, "z", 1),
                                                                                                    (0, "z", 2)]


def test_ge_nodes_bos_prefixed():
    n = P.ge_nodes({(1, "z", 0, 9), (5, "z2", 3, 7), (2, "tc", 2, 3)})
    assert ["blocks.1.attn.hook_z.sae.hook_feature_acts", 1, 9] in [list(x) for x in n]
    assert ["blocks.5.attn.hook_z.sae.hook_feature_acts", 4, 7 + 24576] in [list(x) for x in n]
    assert ["blocks.2.hook_mlp_out.sae.hook_feature_acts", 3, 3] in [list(x) for x in n]


def test_held_order():
    assert P.held_order([1.0, 3.0, 3.0, 2.0, 9.0], 4, k=3) == [1, 2, 3]       # row 4 is outside the ho set
