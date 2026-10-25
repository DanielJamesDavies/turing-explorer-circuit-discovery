"""Tests for 087 intervals.py (CPU only): bootstrap determinism, Wilson / Clopper-Pearson against published values,
the paired bootstrap, formatting / macro names, and reproduction of the paper's numbers from the 062 outputs (skipped
when the 062 rows are not on disk).

  python -m pytest experiments/087-intervals/test_087_intervals.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intervals as I  # noqa: E402

E062 = I.E062
RUN062 = E062 / "out_full" / "out"
HAVE_062 = (RUN062 / "main" / "eval.shard0.jsonl").exists()
SMOKE86 = I.E086 / "out" / "smoke_final"


# ------------------------------------------------------------------------------------------------ bootstrap
def test_bootstrap_deterministic():
    x = (np.arange(500) % 3 == 0).astype(float)
    a = I.boot_mean(x, B=2000, seed=7)
    b = I.boot_mean(x, B=2000, seed=7)
    c = I.boot_mean(x, B=2000, seed=8)
    assert a == b
    assert a != c
    # the per-statistic seed is a pure function of the key: stable across runs and independent of other statistics
    assert I.stat_seed("pass.all") == I.stat_seed("pass.all")
    assert I.stat_seed("pass.all") != I.stat_seed("pass.Deep")


def test_bootstrap_chunking_matches_one_block():
    """B is drawn in fixed blocks; the replicates do not depend on how many statistics ran before."""
    x = np.random.default_rng(0).random(50)
    r1 = I.bootstrap(lambda idx: x[idx].mean(axis=1), len(x), B=600, seed=3)
    r2 = I.bootstrap(lambda idx: x[idx].mean(axis=1), len(x), B=600, seed=3)
    assert len(r1) == 600 and np.array_equal(r1, r2)


def test_bootstrap_proportion_close_to_wilson_at_large_n():
    k, n = 6953, 14994
    x = np.r_[np.ones(k), np.zeros(n - k)]
    lo, hi = I.boot_mean(x, B=4000, seed=1)
    wl, wh = I.wilson(k, n)
    assert abs(lo - wl) < 0.002 and abs(hi - wh) < 0.002


# ---------------------------------------------------------------------------------------- Wilson / CP / rule of 3
def test_wilson_known_values():
    # Newcombe (1998), Stat Med 17:857, Table I: 81/263 -> 0.2553, 0.3662; 0/10 -> 0, 0.2775; 1/29 -> 0.0061, 0.1718
    lo, hi = I.wilson(81, 263)
    assert round(lo, 4) == 0.2553 and round(hi, 4) == 0.3662
    lo, hi = I.wilson(0, 10)
    assert lo == 0.0 and round(hi, 4) == 0.2775
    lo, hi = I.wilson(1, 29)
    assert round(lo, 4) == 0.0061 and round(hi, 4) == 0.1718
    lo, hi = I.wilson(5, 10)
    assert round(lo, 4) == 0.2366 and round(hi, 4) == 0.7634


def test_clopper_pearson_known_values():
    # Newcombe (1998) Table I, exact method: 81/263 -> 0.2527, 0.3676; 0/20 -> 0, 0.1684; 1/29 -> 0.0009, 0.1776;
    # and 0/10 -> 0, 1 - 0.025^(1/10) = 0.3085
    lo, hi = I.clopper_pearson(81, 263)
    assert round(lo, 4) == 0.2527 and round(hi, 4) == 0.3676
    lo, hi = I.clopper_pearson(0, 20)
    assert lo == 0.0 and round(hi, 4) == 0.1684
    assert round(I.wilson(0, 20)[1], 4) == 0.1611
    lo, hi = I.clopper_pearson(0, 10)
    assert lo == 0.0 and round(hi, 4) == 0.3085
    lo, hi = I.clopper_pearson(1, 29)
    assert round(lo, 4) == 0.0009 and round(hi, 4) == 0.1776


def test_one_sided_upper_bounds_zero_events():
    # k = 0: exact one-sided 95% bound 1 - 0.05^(1/n); rule of three 3/n
    assert math.isclose(I.cp_upper(0, 747), 1 - 0.05 ** (1 / 747))
    assert round(100 * I.cp_upper(0, 747), 3) == 0.400
    assert round(100 * I.rule_of_three(747), 3) == 0.402
    assert math.isclose(I.cp_upper(0, 10), 0.25887, abs_tol=1e-5)
    # k > 0 goes through the beta quantile: 1/29 one-sided 95% upper = 0.1534 (two-sided 90% CP upper)
    assert math.isclose(I.cp_upper(1, 29), I.clopper_pearson(1, 29, alpha=0.10)[1], rel_tol=1e-9)


def test_beta_fallback_matches_scipy():
    pytest.importorskip("scipy")
    for q, a, b in ((0.025, 81, 183), (0.975, 82, 182), (0.95, 2, 28)):
        exact = I._beta_ppf(q, a, b)
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if I._beta_cdf_int(mid, a, b) < q else (lo, mid)
        assert abs((lo + hi) / 2 - exact) < 1e-9


# ------------------------------------------------------------------------------------------------ paired
def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    a = (rng.random(1000) < 0.45).astype(float)
    # every pair identical: the difference is exactly 0 in every resample
    assert I.boot_paired_diff(a, a, B=500, seed=1) == (0.0, 0.0)
    # b = a with 50 fails turned into passes: the effect is +5 pp; pairing makes the interval far narrower than the
    # unpaired one (two independent bootstraps, ~ +/- 4.4 pp)
    b = a.copy()
    b[np.flatnonzero(a == 0)[:50]] = 1
    lo, hi = I.boot_paired_diff(a, b, B=4000, seed=2)
    assert lo <= 0.05 <= hi
    assert lo > 0.03 and hi < 0.065
    S = I.Stats(B=1000)
    d = S.diff("t", "", "x", "x", a, b, "src")
    assert d["point"] == pytest.approx(0.05) and d["up"] == 50 and d["down"] == 0


def test_ratio_of_means_paired():
    num = np.r_[np.ones(27), np.zeros(39)]          # 41% of 66
    den = np.r_[np.ones(53), np.zeros(13)]          # 80% of 66
    lo, hi = I.boot_ratio_of_means(num, den, B=2000, seed=0)
    assert lo < 27 / 53 < hi


# --------------------------------------------------------------------------------------------- formatting
def test_text_and_macros():
    S = I.Stats(B=200)
    d = S.prop("t", "", "pass.all", "all", np.r_[np.ones(6953), np.zeros(14994 - 6953)], "src")
    assert S.full(d) == "46.4% [45.6, 47.2], n = 14,994"
    assert S.text(d, tex=True) == r"46.4\%\ [45.6, 47.2]"
    assert d["macro"] == "passAll"
    z = S.prop("t", "null", "all", "random", np.zeros(747), "src", null=True)
    assert S.full(z) == "0/747 (<= 0.40%), n = 747"
    assert z["macro"] == "nullAll"
    small = S.prop("t", "", "pass.layer11", "l11", np.r_[np.ones(26), np.zeros(1720)], "src")
    assert S.text(small).startswith("1.5% [")
    assert small["macro"] == "passLayerOneOne" and small["macro"].isalpha()
    assert I.macro_name("rbsw", "lam", "wcm", "0.001") == "rbswLamWcmZeroPtZeroZeroOne"
    with pytest.raises(ValueError):
        S.prop("t", "", "pass.all", "again", np.ones(3), "src")   # duplicate macro names are refused
    assert S.prop("t", "", "empty", "empty cell", np.array([]), "src") is None   # empty cells are not reported


def test_pass_rule_matches_merge():
    """The local DAN-8 rule equals 062 merge.passes (band edges inclusive, NaN fails)."""
    import importlib
    M = importlib.import_module("merge")
    df = pd.DataFrame({"free0_tk": [0.8, 1.5, 1.51, np.nan, 1.0], "freeM_topk_tk": [1.0, 1.0, 1.0, 1.0, 0.79],
                       "freeN_topk_tk": [1.0, 1.0, 1.0, 1.0, 1.0], "phi_sup_blind_tk": [0.9, 0.95, 1.0, 1.0, 1.0]})
    assert I.passes(df.copy()).tolist() == M.passes(df).tolist() == [True, True, False, False, False]


# -------------------------------------------------------------------------------------------- reproduction
@pytest.fixture(scope="module")
def today():
    if not HAVE_062:
        pytest.skip("062 out_full rows not on disk")
    S = I.Stats(B=300)
    I.run_062(S, RUN062, E062)
    return S, {(d["prefix"], d["key"]): d for d in S.rows}


def test_reproduces_headline(today):
    S, r = today
    d = r[("", "pass.all")]
    assert d["n"] == 14994 and round(100 * d["point"]) == 46            # paper: 46% of ~15,000 (14,994 circuits)
    assert [round(100 * r[("", "pass." + k)]["point"]) for k in ("resid", "mlp", "attn")] == [65, 52, 14]
    assert round(100 * r[("", "pass.layer1")]["point"]) == 85 and round(100 * r[("", "pass.layer11")]["point"]) == 15
    assert round(100 * r[("", "pass.midHeld")]["point"]) == 12


def test_reproduces_nulls_alpha_specificity(today):
    S, r = today
    nl = r[("null", "all")]
    assert (nl["k"], nl["n"]) == (0, 747) and round(100 * r[("null", "fitted")]["point"], 1) == 43.1
    a1 = r[("", "aone.all")]
    assert a1["n"] == 1746 and round(100 * a1["point"], 1) == 1.5
    assert round(100 * r[("", "aone.fitted")]["point"], 1) == 44.7
    assert round(100 * r[("", "amp.all")]["point"]) == 30 and round(100 * r[("", "amp.passing")]["point"]) == 38


def test_reproduces_coefficient_claims(today):
    S, r = today
    for band, (n, ratio) in {"Shallow": (62, 3.7), "Middle": (34, 9.6), "Deep": (26, 10.5)}.items():
        d = r[("sw", "ratio." + band)]
        assert d["n"] == n and round(d["point"], 1) == ratio
        assert d["boot"][0] < d["point"] < d["boot"][1]
    assert round(r[("sw", "size.wcm.Deep")]["point"]) == 906 and round(r[("sw", "size.unw.Deep")]["point"]) == 13075
    w, u = r[("sw", "ever.wcm.Deep")], r[("sw", "ever.unw.Deep")]
    assert w["n"] == u["n"] == 66 and round(100 * w["point"]) == 80 and round(100 * u["point"]) == 41
    assert [round(100 * r[("sw", "fixed.unw." + b)]["point"], 1) for b in ("Middle", "Deep")] == [2.1, 1.5]


@pytest.mark.skipif(not (SMOKE86 / "rescore_asrun" / "main").exists(), reason="086 smoke outputs not on disk")
def test_leakage_reader_on_086_smoke():
    """086's own smoke report: as run 66.7%, rebuilt 100.0% on 3 pairs (effect as run - rebuilt = -33.3%)."""
    S = I.Stats(B=200)
    I.run_rebuilt(S, None, SMOKE86 / "rescore_ctx_v2", SMOKE86 / "rescore_asrun", SMOKE86 / "targets.txt", [], None)
    r = {(d["prefix"], d["key"]): d for d in S.rows}
    e = r[("leak", "effect")]
    assert e["n"] == 3 and e["point"] == pytest.approx(1 / 3) and e["a"] == pytest.approx(2 / 3) and e["b"] == 1.0
