"""Self-checks for experiment 080's logic (CPU, no model):

  PYTHONPATH=src python -m pytest experiments/080-latent-vs-output/test_common.py -q
"""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


# ------------------------------------------------------------------------------------------- design
def test_tiers_and_bands():
    assert C.tier_of(1.0) == "dp_high" and C.tier_of(5.0) == "dp_high"
    assert C.tier_of(0.999) == "dp_mid" and C.tier_of(0.1) == "dp_mid"
    assert C.tier_of(0.0999) == "dp_low" and C.tier_of(0.0) == "dp_low"
    assert [C.band(l) for l in (0, 4, 5, 7, 8, 11)] == ["L0-4", "L0-4", "L5-7", "L5-7", "L8-11", "L8-11"]
    assert C.parse("10.resid.32101") == (10, "resid", 32101)


def test_tier_vocabulary_never_says_writer_or_internal():
    """Every target is an internal latent: no tier name or label may imply otherwise (2026-10-18 rename)."""
    names = set(C.TIERS) | set(C.CAUSAL_TIERS) | set(C.TIER_LABEL.values())
    assert not any(w in n for n in names for w in ("writer", "internal", "weak"))
    assert C.TIER_LABEL["high"] == "high next-token effect" and C.TIER_LABEL["dp_low"] == "direct-path (064) low"
    assert [C.norm_tier(t) for t in ("writer", "weak", "internal", "high", None)] == \
        ["dp_high", "dp_mid", "dp_low", "high", None]
    assert C.SAMPLINGS[0] == "causal"                       # the default


def test_arms_scope_and_weights():
    assert C.DEFAULT_ARMS == ("LAT", "OUT_W", "OUT_U")
    assert C.ARMS["LAT"]["fit"] is False and C.ARMS["LAT"]["endpoint"] == "latent"
    assert all(C.ARMS[a]["endpoint"] == "output" for a in C.FIT_ARMS)
    assert C.ARMS["OUT_W"]["weighted"] and not C.ARMS["OUT_U"]["weighted"]
    assert C.ARMS["OUT_W"]["scope"] == C.ARMS["OUT_U"]["scope"] == C.ARMS["LAT"]["scope"] == "upstream"
    assert C.ARMS["OUT_W_FULL"]["scope"] == C.ARMS["OUT_U_FULL"]["scope"] == "all"
    assert C.ARMS["OUT_W_FULL"]["weighted"] and not C.ARMS["OUT_U_FULL"]["weighted"]


def test_weight_decay_keeps_the_decay_product():
    assert C.weight_decay(C.PROD_LR) == pytest.approx(C.PROD_WD)
    for lr in (0.02, 0.1, 0.5):
        assert C.PROD_STEPS * lr * C.weight_decay(lr) == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------- targets
def _pool(n=40):
    """n keys per kind per (tier, band) cell, unique across cells."""
    return {(t, b): ["%d.%s.%d" % ({"L0-4": 1, "L5-7": 6, "L8-11": 9}[b], k, 1000 * ti + i)
                     for k in ("attn", "mlp", "resid") for i in range(n)]
            for ti, t in enumerate(C.TIERS) for b in C.BANDS}


def test_cell_quotas():
    assert C.cell_quotas(50) == {"L0-4": 16, "L5-7": 17, "L8-11": 17}
    assert C.cell_quotas(51) == {"L0-4": 17, "L5-7": 17, "L8-11": 17}
    assert sum(C.cell_quotas(50).values()) == 50


def test_stratified_pick_is_stratified_deterministic_and_interleaved():
    pool = _pool()
    a, b = C.stratified_pick(pool), C.stratified_pick(pool)
    assert a == b and len(a) == 150 and len(set(a)) == 150
    first9 = a[:9]
    cells = {(t, bd) for t in C.TIERS for bd in C.BANDS for k in first9 if k in pool[(t, bd)]}
    assert len(cells) == 9                                   # the first round covers every cell once


def test_short_cell_hands_its_quota_to_the_tier():
    pool = _pool()
    pool[("dp_high", "L0-4")] = pool[("dp_high", "L0-4")][:5]
    picks = C.stratified_pick(pool)
    w = [k for k in picks if any(k in pool[("dp_high", b)] for b in C.BANDS)]
    assert len(w) == 50 and sum(k in pool[("dp_high", "L0-4")] for k in w) == 5


def test_calib_pick_is_disjoint_and_covers_the_cells():
    pool = _pool()
    main = C.stratified_pick(pool)
    cal = C.calib_pick(pool, exclude=main)
    assert len(cal) == 12 and not set(cal) & set(main)
    cov = {(t, b) for t in C.TIERS for b in C.BANDS if any(k in pool[(t, b)] for k in cal)}
    assert len(cov) == 9
    deep = [k for k in cal if any(k in pool[(t, "L8-11")] for t in C.TIERS)]
    assert len(deep) == 6


def _read_meta(name):
    import csv
    return {r["seed"]: r for r in csv.DictReader(open(HERE / name))}


def test_target_files_are_consistent_when_present():
    """The default-named lists (targets.txt / targets_calib.txt / targets_meta.csv), whichever design wrote them:
    causal (9 pairs, up to 9 per tier, calibration one per cell) or, until the effects are measured, withinkind
    (156 = 12 x 13, calibration 12). Main and calibration disjoint; smoke = the fixed two, never in main."""
    p = HERE / "targets.txt"
    if not p.exists():
        pytest.skip("target lists not generated")
    from collections import Counter
    main = p.read_text().split()
    cal = (HERE / "targets_calib.txt").read_text().split()
    smoke = (HERE / "targets_smoke.txt").read_text().split()
    meta = _read_meta("targets_meta.csv")
    assert smoke == list(C.SMOKE_TARGETS) and not set(main) & set(cal) and not set(smoke) & set(main)
    sampling = {meta[k].get("sampling") or "withinkind" for k in main}
    assert len(sampling) == 1
    if sampling == {"causal"}:
        pairs, n_cell = C.CAUSAL_PAIRS, C.N_PER_CAUSAL
        cnt = Counter(meta[k]["cell"] for k in main)
        assert set(cnt) <= {C.cell_name(k, b, t) for k, b, hi, lo in pairs for t in (hi, lo)}
        assert max(cnt.values()) <= n_cell and all(meta[k]["tier"] in C.CAUSAL_TIERS for k in main + cal)
    else:
        cells = {C.cell_name(k, b, t) for k, b, hi, lo in C.PAIRS for t in (hi, lo)}
        assert len(main) == 156 and len(cal) == 12 and set(smoke) <= set(cal)
        assert Counter(meta[k]["cell"] for k in main) == {c: 13 for c in cells}
        assert Counter(meta[k]["cell"] for k in cal) == {c: 1 for c in cells}
        for k in main + cal:
            assert meta[k]["tier"] == C.tier_of(float(meta[k]["boost_peak"]))
    for k in main + cal:                                     # every listed target sits in its own kind and band
        l, kind, _ = C.parse(k)
        assert meta[k]["kind"] == kind and meta[k]["band"] == C.band(l)


@pytest.mark.parametrize("suffix, n_main", [("_withinkind", 156), ("_tierband", 150)])
def test_legacy_lists_reproduce_under_their_flags(suffix, n_main):
    """The legacy designs' own files (SAMPLING=withinkind / tierband) carry the renamed direct-path tiers."""
    p = HERE / ("targets%s.txt" % suffix)
    if not p.exists():
        pytest.skip("not generated")
    main = p.read_text().split()
    meta = _read_meta("targets_meta%s.csv" % suffix)
    assert len(main) == n_main and {meta[k]["tier"] for k in main} <= set(C.TIERS)


# ------------------------------------------------------------------------------------------- within-kind sampling
def _wk_pool(n=6, short=None):
    """pool[(kind, tier, layer)]: n keys for every tier at every layer of every kind; `short` = {(kind, tier, layer):
    count} overrides."""
    pool = {}
    for ti, t in enumerate(C.TIERS):
        for kind in ("attn", "mlp", "resid"):
            for l in range(12):
                m = (short or {}).get((kind, t, l), n)
                pool[(kind, t, l)] = ["%d.%s.%d" % (l, kind, 1000 * ti + i) for i in range(m)]
    return pool


def _meta_of(pool):
    return {k: dict(kind=kind, tier=t, layer=l, band=C.band(l)) for (kind, t, l), ks in pool.items() for k in ks}


def test_pairs_are_the_specified_adjacent_tier_contrasts():
    assert C.PAIRS == (("resid", "L5-7", "dp_high", "dp_mid"), ("resid", "L8-11", "dp_high", "dp_mid"),
                       ("mlp", "L8-11", "dp_high", "dp_mid"), ("mlp", "L0-4", "dp_mid", "dp_low"),
                       ("attn", "L0-4", "dp_mid", "dp_low"), ("attn", "L8-11", "dp_mid", "dp_low"))
    adjacent = {("dp_high", "dp_mid"), ("dp_mid", "dp_low")}
    assert all((hi, lo) in adjacent for _, _, hi, lo in C.PAIRS)
    assert C.pair_of("mlp", "L0-4", "dp_low") == ("mlp", "L0-4", "dp_mid", "dp_low")
    assert C.pair_of("mlp", "L0-4", "dp_high") is None and C.pair_of("attn", "L5-7", "dp_mid") is None


def test_withinkind_cells_are_single_kind_single_band_and_reproducible():
    pool = _wk_pool()
    meta = _meta_of(pool)
    a, info = C.withinkind_pick(pool)
    b, _ = C.withinkind_pick(pool)
    assert a == b and len(a) == 12 * C.N_PER_CELL and len(set(a)) == len(a)
    assert C.withinkind_pick(pool, seed=1)[0] != a
    for kind, band, hi, lo in C.PAIRS:
        for t in (hi, lo):
            ks = info[C.pair_name(kind, band)]["picks"][t]
            assert len(ks) == C.N_PER_CELL
            assert {meta[k]["kind"] for k in ks} == {kind} and {meta[k]["band"] for k in ks} == {band}
            assert {meta[k]["tier"] for k in ks} == {t}
    first12 = a[:12]                                         # interleaved: the first round visits every cell
    assert len({(meta[k]["kind"], meta[k]["band"], meta[k]["tier"]) for k in first12}) == 12


def test_withinkind_tiers_are_layer_matched():
    pool = _wk_pool()
    meta = _meta_of(pool)
    _, info = C.withinkind_pick(pool)
    for kind, band, hi, lo in C.PAIRS:
        i = info[C.pair_name(kind, band)]
        assert i["n_matched"] == C.N_PER_CELL and not any(i["unmatched"].values())
        hist = {t: sorted(meta[k]["layer"] for k in i["picks"][t]) for t in (hi, lo)}
        assert hist[hi] == hist[lo]
        q = list(i["quota"].values())
        assert max(q) - min(q) <= 1                          # spread evenly over the band's layers


def test_withinkind_short_capacity_is_topped_up_and_flagged():
    """mlp dp_low exists only at L0 and L1 (2 each): matched capacity 4, the other 9 per tier are unmatched."""
    short = {("mlp", "dp_low", l): (2 if l in (0, 1) else 0) for l in range(5)}
    pool = _wk_pool(n=20, short=short)
    meta = _meta_of(pool)
    _, info = C.withinkind_pick(pool)
    i = info["mlp L0-4"]
    assert i["n_matched"] == 4 and len(i["picks"]["dp_low"]) == 4        # the short tier cannot be filled
    assert len(i["picks"]["dp_mid"]) == C.N_PER_CELL and len(i["unmatched"]["dp_mid"]) == C.N_PER_CELL - 4
    assert sorted(meta[k]["layer"] for k in i["picks"]["dp_low"]) == [0, 0, 1, 1]
    assert i["thin"] and i["n_available"]["dp_low"] == 4


def test_withinkind_calib_one_per_cell_disjoint_forced_and_matched():
    pool = _wk_pool()
    pool[("mlp", "dp_low", 1)].append("1.mlp.35042")
    pool[("resid", "dp_high", 11)].append("11.resid.15522")
    meta = _meta_of(pool)
    main, _ = C.withinkind_pick(pool, exclude=C.SMOKE_TARGETS)
    cal = C.withinkind_calib(pool, exclude=main)
    assert len(cal) == 12 and not set(cal) & set(main) and set(C.SMOKE_TARGETS) <= set(cal)
    cells = [(meta[k]["kind"], meta[k]["band"], meta[k]["tier"]) for k in cal]
    assert len(set(cells)) == 12
    for kind, band, hi, lo in C.PAIRS:
        ls = [meta[k]["layer"] for k in cal if meta[k]["kind"] == kind and meta[k]["band"] == band]
        assert len(ls) == 2 and ls[0] == ls[1]               # partner at the anchor's layer
    assert C.withinkind_calib(pool, exclude=main) == cal


def test_load_meta_merges_and_prefers_the_default(tmp_path):
    (tmp_path / "targets_meta_tierband.csv").write_text("seed,tier,cell\na,dp_mid,\nb,dp_high,\n")
    (tmp_path / "targets_meta.csv").write_text("seed,tier,cell\na,low,mlp L0-4 low\n")
    m = C.load_meta(tmp_path)
    assert m["a"]["cell"] == "mlp L0-4 low" and m["a"]["tier"] == "low" and m["b"]["tier"] == "dp_high"
    assert C.load_meta(tmp_path / "missing") == {}


# ------------------------------------------------------------------------------------------- P1 statistics
def test_pair_contrast_pools_the_simple_mean_of_pair_differences():
    cells = {"p1": ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),      # d = 1 exactly
             "p2": ([0.5, 0.5], [0.5, 0.5]),                # d = 0 exactly
             "p3": ([0.3, None], [])}                       # an empty tier: left out
    r = C.pair_contrast(cells, B=200)
    assert r["n_pairs"] == 2 and set(r["per_pair"]) == {"p1", "p2"}
    assert r["pooled"] == pytest.approx(0.5) and r["pooled_lo"] == pytest.approx(0.5)
    assert r["per_pair"]["p1"]["d"] == 1.0 and r["per_pair"]["p1"]["n_hi"] == 3
    assert C.pair_contrast(cells, B=200) == r                # deterministic
    assert math.isnan(C.pair_contrast({"x": ([], [1.0])}, B=10)["pooled"])


def test_pair_contrast_inverse_variance_favours_the_precise_pair():
    import numpy as np
    g = np.random.default_rng(0)
    cells = {"tight": (list(1.0 + 0.01 * g.standard_normal(20)), list(0.01 * g.standard_normal(20))),
             "noisy": (list(-1.0 + 2.0 * g.standard_normal(20)), list(2.0 * g.standard_normal(20)))}
    r = C.pair_contrast(cells, B=400)
    assert abs(r["iv"] - r["per_pair"]["tight"]["d"]) < abs(r["pooled"] - r["per_pair"]["tight"]["d"])


def test_fe_slope_uses_within_group_variation_only():
    """Between groups y falls with x, within groups it rises with slope 2: the fixed-effects slope is +2."""
    import numpy as np
    x, y, g = [], [], []
    for j, (base_x, base_y) in enumerate(((0.05, 5.0), (0.5, 0.0), (5.0, -5.0))):
        for k in range(10):
            xx = base_x * (1 + 0.1 * k)
            x.append(xx); y.append(base_y + 2.0 * math.log(xx / base_x)); g.append("g%d" % j)
    s = C.fe_slope(x, y, g, B=300)
    assert s["slope"] == pytest.approx(2.0, abs=1e-6) and s["lo"] == pytest.approx(2.0, abs=1e-6)
    assert s["n"] == 30 and s["n_groups"] == 3
    pooled = np.polyfit(np.log(x), y, 1)[0]
    assert pooled < 0                                        # the confounded pooled slope has the wrong sign
    assert C.fe_slope([0.0, -1.0], [1.0, 2.0], ["a", "a"], B=10)["n"] == 0


# ------------------------------------------------------------------------------------------- config
def test_run_config_is_never_silently_defaulted(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        C.load_run_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(arms={"OUT_W": {"lams": [], "lr": 0.05}}, logit_norm="gap", steps=400)))
    with pytest.raises(ValueError, match="lams"):
        C.load_run_config(bad)
    bad.write_text(json.dumps(dict(arms={"LAT": {"lams": [1e-3], "lr": 0.05}}, logit_norm="gap", steps=400)))
    with pytest.raises(ValueError, match="arm"):
        C.load_run_config(bad)
    bad.write_text(json.dumps(dict(arms={"OUT_W": {"lams": [1e-3], "lr": 0.05}}, steps=400)))
    with pytest.raises(ValueError, match="logit_norm"):
        C.load_run_config(bad)


@pytest.mark.parametrize("name", ["config_smoke.json", "config_smoke_timing.json"])
def test_smoke_configs_load(name):
    cfg = C.load_run_config(HERE / name)
    assert set(cfg["arms"]) == {"OUT_W", "OUT_U"} and cfg["logit_norm"] == "gap"


# ------------------------------------------------------------------------------------------- pass rules
def test_pass_rules():
    lat = {"free0_tk": 0.8, "freeM_topk_tk": 1.5, "freeN_topk_tk": 1.0, "phi_sup_blind_tk": 0.9}
    assert C.passes_latent(lat) and not C.passes_latent(dict(lat, freeN_topk_tk=None))
    assert not C.passes_latent(dict(lat, phi_sup_blind_tk=0.89))
    out = {"out_free0": 1.0, "out_freeM_topk": 1.0, "out_freeN_topk": 1.0, "out_phi_sup_blind": 0.95}
    assert C.passes_output(out) and not C.passes_output(dict(out, out_free0=float("nan")))
    assert not C.passes_output(lat) and not C.passes_latent(out)


# ------------------------------------------------------------------------------------------- matched size
def test_interp_is_071s():
    assert C.interp_log_n is C.L071.interp_log_n
    y, ex = C.interp_log_n([100, 1000], [0.2, 0.8], 10 ** 2.5)
    assert y == pytest.approx(0.5) and not ex
    y, ex = C.interp_log_n([100, 1000], [0.2, 0.8], 50)
    assert y == 0.2 and ex


def test_nearest_in_log():
    assert C.nearest_in_log([100, 1000, 10000], 400) == 1           # log10 400 = 2.60: nearer 3 than 2
    assert C.nearest_in_log([100, 1000], 10 ** 2.5) == 0             # a tie goes to the smaller circuit
    assert C.nearest_in_log([None, 0, 50], 400) == 2
    assert C.nearest_in_log([], 400) is None


def test_size_to_reach():
    n, st = C.size_to_reach([100, 1000, 10000], [0.2, 0.6, 1.0], 0.8)
    assert st == "ok" and n == pytest.approx(10 ** 3.5)
    assert C.size_to_reach([100, 1000], [0.9, 1.0], 0.8) == (100, "already")
    assert C.size_to_reach([100, 1000], [0.1, 0.5], 0.8) == (None, "never")
    n, st = C.size_to_reach([1000, 100], [1.0, 0.0], 0.5)            # order-free
    assert st == "ok" and n == pytest.approx(10 ** 2.5)


def test_overlap():
    o = C.overlap({1, 2, 3, 4}, {3, 4, 5})
    assert o["jaccard"] == pytest.approx(2 / 5) and o["cont_a_in_b"] == 0.5 and o["cont_b_in_a"] == pytest.approx(2 / 3)
    assert C.overlap(set(), set())["jaccard"] is None


# ------------------------------------------------------------------------------------------- bootstrap
def test_paired_bootstrap_is_paired_and_deterministic():
    a = [1.0, 2.0, 3.0, 4.0, None]
    b = [0.5, 1.5, 2.5, 3.5, 9.0]
    d1, d2 = C.paired_bootstrap(a, b, B=300), C.paired_bootstrap(a, b, B=300)
    assert d1 == d2
    assert d1["d"] == pytest.approx(0.5) and d1["d_lo"] == pytest.approx(0.5) and d1["d_hi"] == pytest.approx(0.5)
    assert d1["n_pair"] == 4 and d1["n"] == 4 and d1["n_b"] == 5
    assert d1["a_lo"] <= d1["a"] <= d1["a_hi"]
    m = C.paired_bootstrap([0.0, 1.0, 1.0, 0.0], stat="mean", B=200)
    assert m["a"] == 0.5 and 0.0 <= m["a_lo"] <= m["a_hi"] <= 1.0
    with pytest.raises(AssertionError):
        C.paired_bootstrap([1.0], [1.0, 2.0])


def test_spearman_bootstrap():
    s = C.spearman_bootstrap([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], B=200)
    assert s["rho"] == pytest.approx(1.0) and s["n"] == 6
    assert math.isnan(C.spearman_bootstrap([1, 2], [1, 2])["rho"])


# ------------------------------------------------------------------------------------------- calibration
def test_fit_failed_and_monotone():
    assert C.fit_failed({"out_free0": 0.04}) and C.fit_failed({"out_free0": -0.3}) and C.fit_failed({})
    assert C.fit_failed({"error": "x", "out_free0": 0.9}) and not C.fit_failed({"out_free0": 0.06})
    assert C.monotone([1e-4, 1e-3, 1e-2], [900, 300, 40])
    assert not C.monotone([1e-4, 1e-3, 1e-2], [0.9, 0.95, 0.4])
    assert C.monotone([1e-4, 1e-3, 1e-2], [0.9, 0.91, 0.4], tol=0.02)
    assert C.monotone([1e-3, 1e-4], [0.4, 0.9])                       # sorted by lambda first
    assert C.monotone([1e-3], [0.4]) is None


def _calib_rows():
    rows = []
    for t in ("a", "b", "c", "d"):
        for lr in (0.02, 0.05):
            for lam, n, z in ((1e-4, 3000, 0.95), (3e-4, 1200, 0.9), (1e-3, 500, 0.85), (3e-3, 200, 0.6),
                              (1e-2, 60, 0.3)):
                bad = lr == 0.02 and lam == 3e-3                       # lr 0.02 fails once per target
                rows.append(dict(seed=t, arm="OUT_W", lr=lr, lam=C.lam_tag(lam), n=n,
                                 out_free0=(-0.2 if bad else z)))
    return rows


def test_calib_summary_and_recommendation():
    S = C.calib_summary(_calib_rows(), {"a": 300, "b": 500, "c": 800, "d": 1000})
    s05, s02 = S["OUT_W|0.05"], S["OUT_W|0.02"]
    assert s05["fail_rate"] == 0.0 and s02["fail_rate"] == pytest.approx(0.2)
    assert s05["mono_size"] == 1.0 and s05["mono_faith"] == 1.0 and s02["mono_faith"] == 0.0
    assert s05["lams"]["0.001"]["n_median"] == 500 and s02["lams"]["0.003"]["oZ_neg"] == 4
    rec = C.recommend(S)
    w = rec["arms"]["OUT_W"]
    assert w["lr"] == 0.05 and len(w["lams"]) == 4 and w["brackets_lat"]
    assert min(w["lams"]) <= 1e-3 <= max(w["lams"])


def test_choose_grid_flags_a_grid_that_cannot_bracket():
    grid, ok = C.choose_grid({1e-4: 50, 3e-4: 40, 1e-3: 30, 3e-3: 20, 1e-2: 10}, 300, 1000)
    assert len(grid) == 4 and not ok
    rec = C.recommend({"OUT_W|0.05": dict(arm="OUT_W", lr=0.05, fail_rate=0.5, mono_faith=0.5, mono_size=1.0,
                                          lams={"0.001": dict(n_median=30)}),
                       "_lat": dict(p10=300, p90=1000)})
    assert len(rec["warnings"]) == 2


# ------------------------------------------------------------------------------------------- causal sampling
def _causal_meta(n_per_layer=6, seed=3):
    """A measured pool: every kind x layer with n_per_layer targets and a random effect_abs (nats)."""
    import random
    rng = random.Random(seed)
    meta = {}
    for kind in C.KINDS:
        for l in range(12):
            for i in range(n_per_layer):
                k = "%d.%s.%d" % (l, kind, i)
                meta[k] = dict(kind=kind, layer=l, band=C.band(l), effect_abs=rng.random())
    return meta


def test_causal_tiers_are_within_group_tertiles():
    eff = {"k%02d" % i: float(i) for i in range(10)}          # n = 10: floor(10 / 3) = 3 per tail, 4 dropped
    t = C.causal_tiers(eff)
    assert sorted(k for k, v in t.items() if v == "high") == ["k07", "k08", "k09"]
    assert sorted(k for k, v in t.items() if v == "low") == ["k00", "k01", "k02"]
    assert len(t) == 6
    tie = C.causal_tiers({"b": 1.0, "a": 1.0, "c": 1.0})       # ties broken by key: deterministic
    assert tie == {"a": "low", "c": "high"}
    assert C.causal_tiers({"a": 1.0, "b": 2.0}) == {}          # fewer than 3: no tier


def test_causal_pool_ranks_inside_each_group_not_globally():
    meta = _causal_meta()
    for k, m in meta.items():                                  # make resid uniformly 10x larger than attn
        m["effect_abs"] *= {"attn": 1.0, "mlp": 3.0, "resid": 10.0}[m["kind"]]
    pool, tiers, cuts = C.causal_pool(meta)
    assert set(cuts) == {C.pair_name(k, b) for k in C.KINDS for b in C.BANDS}
    for p, c in cuts.items():                                  # every group has both tiers, a third each
        assert c["n_low"] == c["n_high"] == c["n_pool"] // 3 and c["low_max"] < c["high_min"]
    assert {t for k, t in tiers.items() if meta[k]["kind"] == "attn"} == {"high", "low"}
    assert all(k in pool[(meta[k]["kind"], t, meta[k]["layer"])] for k, t in tiers.items())


def test_causal_pick_nine_layer_matched_pairs():
    meta = _causal_meta(n_per_layer=30)
    pool, tiers, _ = C.causal_pool(meta)
    main, info = C.withinkind_pick(pool, C.N_PER_CAUSAL, exclude=C.SMOKE_TARGETS, pairs=C.CAUSAL_PAIRS)
    again, _ = C.withinkind_pick(pool, C.N_PER_CAUSAL, exclude=C.SMOKE_TARGETS, pairs=C.CAUSAL_PAIRS)
    assert main == again and len(main) == 9 * 2 * C.N_PER_CAUSAL == 162 and len(set(main)) == 162
    assert len(C.CAUSAL_PAIRS) == 9 and all(hi == "high" and lo == "low" for _, _, hi, lo in C.CAUSAL_PAIRS)
    for kind, b, hi, lo in C.CAUSAL_PAIRS:
        i = info[C.pair_name(kind, b)]
        assert not i["thin"] and i["n_matched"] == C.N_PER_CAUSAL
        for t in (hi, lo):
            ks = i["picks"][t]
            assert {meta[k]["kind"] for k in ks} == {kind} and {meta[k]["band"] for k in ks} == {b}
            assert {tiers[k] for k in ks} == {t}
        assert sorted(meta[k]["layer"] for k in i["picks"][hi]) == sorted(meta[k]["layer"] for k in i["picks"][lo])
    cal = C.withinkind_calib(pool, exclude=main, pairs=C.CAUSAL_PAIRS)
    assert len(cal) == 18 and not set(cal) & set(main)
    assert len({(meta[k]["kind"], meta[k]["band"], tiers[k]) for k in cal}) == 18


def test_causal_pick_flags_a_thin_group():
    meta = _causal_meta(n_per_layer=30)
    for k in [k for k, m in meta.items() if m["kind"] == "mlp" and m["band"] == "L5-7"][12:]:
        del meta[k]                                            # 12 measured -> 4 high, 4 low
    pool, _, cuts = C.causal_pool(meta)
    assert cuts["mlp L5-7"]["n_high"] == 4
    _, info = C.withinkind_pick(pool, C.N_PER_CAUSAL, pairs=C.CAUSAL_PAIRS)
    assert info["mlp L5-7"]["thin"] and len(info["mlp L5-7"]["picks"]["high"]) == 4
    assert not info["mlp L0-4"]["thin"]


def test_design_pairs_follow_the_tiers_present():
    assert C.design_pairs(["high", "low"]) == C.CAUSAL_PAIRS
    assert C.design_pairs(["dp_high", "dp_mid"]) == C.PAIRS
    assert C.pair_of("attn", "L5-7", "high") == ("attn", "L5-7", "high", "low")
    assert C.pair_of("attn", "L5-7", "dp_mid") is None


def test_pooled_contrast_over_the_nine_groups():
    cells = {C.pair_name(k, b): ([1.0 + j] * 9, [0.5 + j] * 9) for j, (k, b, _, _) in enumerate(C.CAUSAL_PAIRS)}
    r = C.pair_contrast(cells, B=200)
    assert r["n_pairs"] == 9 and r["pooled"] == pytest.approx(0.5) and r["pooled_lo"] == pytest.approx(0.5)


def test_load_effects_refuses_without_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="queue_effect.sh"):
        C.load_effects(tmp_path / "effects.parquet")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pd.DataFrame([dict(seed="1.mlp.1", effect_abs=0.2, n_contexts=32, error=None),
                  dict(seed="1.mlp.2", effect_abs=0.3, n_contexts=3, error=None),       # too few contexts
                  dict(seed="1.mlp.3", effect_abs=None, n_contexts=0, error="load x")]
                 ).to_parquet(tmp_path / "effects.parquet")
    assert set(C.load_effects(tmp_path / "effects.parquet")) == {"1.mlp.1"}


def test_make_targets_refuses_causal_without_effects(tmp_path):
    """SAMPLING=causal without the effects file exits non-zero, names queue_effect.sh, and writes nothing."""
    import os
    import subprocess
    before = {p.name: p.stat().st_mtime for p in HERE.glob("targets*")}
    env = dict(os.environ, SAMPLING="causal", EFFECTS_FILE=str(tmp_path / "none.parquet"),
               PYTHONPATH=str(HERE.parent.parent / "src"))
    r = subprocess.run([sys.executable, str(HERE / "make_targets.py")], env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode != 0 and "queue_effect.sh" in (r.stdout + r.stderr) and "REFUSED" in (r.stdout + r.stderr)
    assert {p.name: p.stat().st_mtime for p in HERE.glob("targets*")} == before
