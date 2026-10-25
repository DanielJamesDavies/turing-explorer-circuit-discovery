"""DAN-157 re-score bookkeeping: the frozen stratified subset maker (086 subset.py: deterministic, stratified counts)
and the report's pass rates (086 passrates.py: headline over targets with >= 8 clean held-out strongest rows, worst
case with below-floor targets as fails, the paired leakage effect)."""
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
D86 = REPO / "experiments" / "086-heldout-rebuild"


def _load(name):
    spec = importlib.util.spec_from_file_location("x086_" + name, D86 / ("%s.py" % name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SUB = _load("subset")
PR = _load("passrates")


def keys(n_per):
    """Synthetic targets: n_per[(band layer, kind)] keys each."""
    out = []
    for (l, kd), n in n_per.items():
        out += ["%d.%s.%d" % (l, kd, i) for i in range(n)]
    return out


# ----------------------------------------------------------------------------------------------- subset maker
def test_subset_is_deterministic_and_stratified():
    ks = keys({(l, kd): 300 for l in (1, 6, 10) for kd in ("attn", "mlp", "resid")})
    a, ca = SUB.stratified_subset(ks, 1000, 20261024)
    b, cb = SUB.stratified_subset(list(reversed(ks)), 1000, 20261024)       # input order does not matter
    assert a == b and ca == cb and len(a) == 1000 == len(set(a)) and a == sorted(a)
    cnt = Counter(SUB.cell_of(k) for k in a)
    assert sorted(cnt.values()) == [111] * 8 + [112]                           # 1000 = 9 x 111 + 1
    assert cnt[("L0-4", "attn")] == 112                                       # the remainder: first cell, sorted
    c, _ = SUB.stratified_subset(ks, 1000, 1)
    assert c != a                                                             # the seed matters


def test_subset_small_cells_give_everything_and_spread_the_rest():
    n_per = {(l, kd): 300 for l in (1, 6, 10) for kd in ("attn", "mlp", "resid")}
    n_per[(1, "attn")] = 40
    ks = keys(n_per)
    a, counts = SUB.stratified_subset(ks, 1000, 20261024)
    cnt = Counter(SUB.cell_of(k) for k in a)
    assert len(a) == 1000 and cnt[("L0-4", "attn")] == 40
    assert set(cnt[c] for c in cnt if c != ("L0-4", "attn")) == {120}         # (1000 - 40) / 8
    assert counts["L0-4/attn"] == 40
    assert SUB.allocate({"a": 3, "b": 10}, 100) == {"a": 3, "b": 10}          # fewer targets than asked: all


def test_frozen_list_matches_its_manifest():
    lst = D86 / "targets_asrun1000.txt"
    man = D86 / "targets_asrun1000.txt.manifest.json"
    if not lst.exists():
        pytest.skip("frozen subset not generated")
    import json
    m = json.loads(man.read_text())
    assert SUB.md5(lst) == m["md5"] and m["n"] == 1000 and m["seed"] == 20261024
    got = lst.read_text().split()
    assert len(got) == 1000 and Counter(SUB.cell_of(k) for k in got) == Counter(
        {tuple(c.split("/")): n for c, n in m["counts"].items()})


# ----------------------------------------------------------------------------------------------- pass rates
def r(seed, ok=True, n_held=16, vac=False, skip=None, **kw):
    if skip is not None:
        return dict(seed=seed, held="strong", skip=skip, n_held=n_held, **kw)
    v = 1.0 if ok else 0.3
    return dict(seed=seed, held="strong", free0_tk=v, freeM_topk_tk=v, freeN_topk_tk=v, phi_sup_blind_tk=0.95,
                vacuous_tk=vac, n_held=n_held, **kw)


def test_headline_counts_only_targets_at_or_above_the_floor():
    rows = [r("1.mlp.%d" % i) for i in range(6)] + [r("1.mlp.%d" % i, ok=False) for i in range(6, 8)]
    rows += [r("1.mlp.8", n_held=5), r("1.mlp.9", n_held=7, ok=False)]          # below the floor, scored + flagged
    rows += [r("1.mlp.10", skip="no_heldout", n_held=0), r("1.mlp.11", skip="rejected")]
    rows += [r("1.mlp.12", vac=True), r("1.mlp.13", n_held=8)]                  # vacuous (excluded); 8 = at the floor
    h = PR.headline(PR.classify(rows))
    assert (h["n"], h["passed"]) == (9, 7) and h["rate"] == pytest.approx(7 / 9)
    assert h["n_below"] == 3 and h["below_scored_rate"] == pytest.approx(0.5)
    assert h["n_worst"] == 12 and h["worst"] == pytest.approx(7 / 12)           # below floor counted as fails


def test_last_row_wins_and_errors_are_dropped():
    rows = [r("2.attn.1", ok=False), dict(seed="2.attn.1", held="strong", error="boom"), r("2.attn.1")]
    d = PR.classify(rows)
    assert len(d) == 1 and bool(d["pass"].iloc[0])


def test_paired_leakage_effect():
    asrun = [r("1.mlp.%d" % i) for i in range(10)] + [r("9.resid.%d" % i) for i in range(4)]
    v2 = [r("1.mlp.%d" % i, ok=i < 6) for i in range(10)] + [r("9.resid.0", n_held=3), r("9.resid.1"),
                                                            r("9.resid.2", vac=True), r("9.resid.3", ok=False)]
    ov, g = PR.paired(PR.classify(asrun), PR.classify(v2))
    assert ov["n_pairs"] == 13 and ov["n_below"] == 1 and ov["n_head"] == 12  # the vacuous pair is excluded
    assert ov["pass_asrun"] == 1.0 and ov["pass_v2"] == pytest.approx(7 / 12)
    assert ov["effect"] == pytest.approx(5 / 12)
    assert ov["pass_v2_worst"] == pytest.approx(7 / 13) and ov["effect_worst"] == pytest.approx(6 / 13)
    gm = g.set_index(["band", "kind"])
    assert gm.loc[("L0-4", "mlp"), "effect"] == pytest.approx(0.4)
    assert gm.loc[("L8-11", "resid"), "n_below"] == 1
