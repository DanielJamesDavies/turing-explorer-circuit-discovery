"""DAN-181 harder nulls: samplers, lags, deciles, coefficients, the alpha = 1 path and the report logic (CPU, seconds).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/084-harder-nulls/test_084_nulls.py -q
"""
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import nulls_core as NC  # noqa: E402
import report as RP  # noqa: E402


# ----------------------------------------------------------------------------------------------- targets
def test_bands():
    assert [NC.band(l) for l in (0, 4, 5, 7, 8, 11)] == ["L0-4", "L0-4", "L5-7", "L5-7", "L8-11", "L8-11"]
    assert [NC.band(l, 6) for l in range(6)] == ["L0-1", "L0-1", "L2-3", "L2-3", "L4-5", "L4-5"]


def test_stratified_order_balanced_interleaved_and_seeded():
    keys = ["%d.%s.%d" % (l, k, i) for l in range(12) for k in ("attn", "mlp", "resid") for i in range(20)]
    a = NC.stratified_order(keys, 90)
    assert a == NC.stratified_order(list(reversed(keys)), 90)          # input order does not matter
    cells = [(NC.band(int(k.split(".")[0])), k.split(".")[1]) for k in a]
    assert Counter(cells) == Counter({c: 10 for c in set(cells)}) and len(set(cells)) == 9
    assert len(set(cells[:9])) == 9                                    # every prefix of 9 covers every cell
    assert NC.stratified_order(keys, 90, seed=1) != a


def test_panel_prior_first_balanced_seeded():
    keys = ["%d.%s.%d" % (l, k, i) for l in range(12) for k in ("attn", "mlp", "resid") for i in range(20)]
    prior = NC.stratified_order(keys, 63, seed=5)                      # 7 per cell, as the DAN-17 sweep run
    order, sub_n = NC.panel(keys, 14, 7, prior=prior)
    assert len(order) == 126 and len(set(order)) == 126 and sub_n == 63
    assert set(order[:63]) == set(prior)                               # the subsample = the prior's targets
    cells = Counter((NC.band(int(k.split(".")[0])), k.split(".")[1]) for k in order)
    assert set(cells.values()) == {14} and len(cells) == 9
    assert len({(NC.band(int(k.split(".")[0])), k.split(".")[1]) for k in order[:9]}) == 9   # interleaved
    assert NC.panel(list(reversed(keys)), 14, 7, prior=prior) == (order, sub_n)
    assert NC.panel(keys, 14, 7, prior=prior, seed=1)[0][63:] != order[63:]
    o2, s2 = NC.panel(keys, 14, 7)                                     # no prior: seeded top-up only
    assert s2 == 63 and len(o2) == 126
    few = [k for k in keys if not k.startswith("5.attn")][:200] + ["5.attn.1", "6.attn.2"]
    o3, _ = NC.panel(few, 14, 7)                                       # a short cell takes all it has
    assert sum(1 for k in o3 if NC.band(int(k.split(".")[0])) == "L5-7" and ".attn." in k) == 2


def test_frozen_turingllm_panel():
    lst, man = HERE / "targets_turingllm_nulls.txt", HERE / "targets_turingllm_nulls.json"
    if not lst.exists():
        pytest.skip("panel not written")
    keys = lst.read_text().split()
    m = json.loads(man.read_text())
    assert len(keys) == m["n"] == 126 and m["sub_n"] == 63 and set(m["cells"].values()) == {14}
    sweep = set((REPO / "experiments" / "062-h100-protocol-v1" / "targets_sweep.txt").read_text().split())
    assert set(keys) <= sweep


def test_dan17_draw_is_random_nulls_procedure():
    live = {(1, "mlp"): set(range(50)), (2, "attn"): {3, 9}}
    ref_items = [((2, "attn"), [1, 2, 3]), ((1, "mlp"), [4, 5, 6, 7])]
    sup, diag = NC.dan17_draw("3.mlp.7", ref_items, live)
    rng = random.Random("3.mlp.7-null")                                # 062/random_null.main, draw 0
    want = {s: rng.sample(sorted(live.get(s, ())), min(len(v), len(live.get(s, ())))) for s, v in ref_items}
    assert sup == want and diag == dict(n_ref=7, n_drawn=6, short_sites=1)


def test_draw_counts():
    assert [NC.n_draws_for(p, 1, 63, 4) for p in (0, 62, 63, 251)] == [4, 4, 1, 1]
    assert NC.n_draws_for(0, 3, 10, 2) == 3


def test_plan_jobs():
    j = NC.plan_jobs(["wcm", "dan17"], True, ["anchor", "contrib"], 2, ["held", "train"], True)
    assert ("wcm_a1", "wcm", 0, "held", "one") in j and ("dan17", "dan17", 0, "train", "fit") in j
    assert sum(1 for x in j if x[1] == "anchor") == 2 * 3 and ("contrib_a1", "contrib", 1, "held", "one") in j
    assert not any(x[1] == "dan17" for x in NC.plan_jobs(["wcm", "dan17"], False, ["anchor"], 1, ["held"], False))
    assert not any(x[4] == "one" for x in NC.plan_jobs(["wcm"], False, ["anchor"], 1, ["held"], False))


# ----------------------------------------------------------------------------------------------- lags / deciles
def test_site_min_lag():
    # B=2, T=5, K=2; anchors 3 and 1
    acts = torch.zeros(2, 5, 2)
    idx = torch.zeros(2, 5, 2, dtype=torch.long)
    acts[0, 3, 0], idx[0, 3, 0] = 1.0, 7        # at the anchor: lag 0
    acts[0, 1, 1], idx[0, 1, 1] = 2.0, 8        # lag 2
    acts[0, 4, 0], idx[0, 4, 0] = 5.0, 9        # after the anchor: ignored
    acts[1, 0, 0], idx[1, 0, 0] = 1.0, 8        # lag 1 on context 1: min(2, 1) = 1
    idx[1, 1, 1] = 3                            # zero activation: ignored
    lag = NC.site_min_lag(acts, idx, torch.tensor([3, 1]), 12)
    assert lag[7] == 0 and lag[8] == 1 and lag[9] == NC.LAG_NONE and lag[3] == NC.LAG_NONE and lag[0] == NC.LAG_NONE


def test_freq_and_rank_deciles():
    counts = np.array([0] * 5 + list(range(1, 101)))
    d = NC.freq_deciles(counts)
    assert (d[:5] == -1).all() and Counter(d[5:].tolist()) == Counter({k: 10 for k in range(10)})
    assert all(np.diff(d[5:]) >= 0)
    r = NC.rank_deciles({i: float(i % 7) for i in range(50)})
    assert set(r.values()) == set(range(10)) and r[0] == 0
    p = NC.place_deciles({"a": -1.0, "b": 100.0, "c": 3.0}, list(range(10)))
    assert p == {"a": 0, "b": 9, "c": 3}


# ----------------------------------------------------------------------------------------------- samplers
def _profile(n_sites=3, d=400, seed=0):
    """A synthetic profile: lags 0..20, frequency deciles from counts, contributions with deciles over the pool."""
    rng = np.random.RandomState(seed)
    sites = [(l, "mlp") for l in range(n_sites)]
    prof = dict(lag={}, fdec={}, count={}, contrib={}, cdec={}, sites=sites)
    for s in sites:
        lat = sorted(rng.choice(d, 300, replace=False).tolist())
        cnt = rng.randint(1, 10000, size=d)
        fd = NC.freq_deciles(cnt)
        prof["lag"][s] = {l: int(rng.randint(0, 21)) for l in lat}
        prof["fdec"][s] = {l: int(fd[l]) for l in lat}
        prof["count"][s] = {l: int(cnt[l]) for l in lat}
        for l in lat:
            prof["contrib"][(s, l)] = float(rng.exponential())
    prof["cdec"] = NC.rank_deciles(prof["contrib"])
    prof["cbin"] = NC.member_bins(list(prof["contrib"].values())[::7], prof["contrib"])
    return prof


def _members(prof, per_site=(12, 5, 30), win=3, seed=1):
    rng = random.Random(seed)
    out = {}
    for s, n in zip(prof["sites"], per_site):
        out[s] = rng.sample(sorted(prof["lag"][s]), n)        # members anywhere in the prefix
    return out


@pytest.mark.parametrize("null", ["anchor", "contrib", "uniform"])
def test_samplers_counts_anchor_activity_exclusion_determinism(null):
    prof = _profile()
    ref = _members(prof)
    win = 3
    sup, diag = NC.draw_null(null, ref, prof, win, NC.rng_for("k", null, 0), exclude_members=True)
    pool = NC.anchor_pool(prof, win)
    assert diag["n_overlap"] == 0 and diag["exclude_members"] is True
    for s, v in ref.items():
        avail = [l for l in pool[s] if l not in set(v)]
        assert len(sup[s]) == min(len(v), len(avail))                  # per-site counts (all of a short pool)
        assert len(set(sup[s])) == len(sup[s])                         # without replacement
        assert all(prof["lag"][s][l] <= win for l in sup[s])           # anchor-active
        assert not set(sup[s]) & set(v)                                # production members excluded
    assert diag["n_ref"] == sum(map(len, ref.values())) and diag["n_drawn"] == sum(map(len, sup.values()))
    again, _ = NC.draw_null(null, ref, prof, win, NC.rng_for("k", null, 0), exclude_members=True)
    other, _ = NC.draw_null(null, ref, prof, win, NC.rng_for("k", null, 1), exclude_members=True)
    assert again == sup and other != sup
    bb = diag["by_bin"]                                                # shortfall table: every member counted once
    assert sum(v[0] for v in bb.values()) == diag["n_ref"] and sum(v[1] for v in bb.values()) == diag["n_drawn"]
    assert sum(v[2] for v in bb.values()) == diag["n_exact"]


@pytest.mark.parametrize("null", ["anchor", "contrib", "uniform"])
def test_default_includes_members_and_records_overlap(null):
    prof = _profile()
    ref = _members(prof)
    win = 20                                                           # every member anchor-active
    sup, diag = NC.draw_null(null, ref, prof, win, NC.rng_for("k", null, 0))
    assert diag["exclude_members"] is False
    ov = sum(len(set(sup[s]) & set(ref[s])) for s in ref)
    assert diag["n_overlap"] == ov
    pool = NC.anchor_pool(prof, win)
    for s, v in ref.items():
        assert len(sup[s]) == min(len(v), len(pool[s]))               # members are in the pool: no shortfall


def test_parse_exclude_default_is_contrib_only():
    assert NC.parse_exclude(None) == {"contrib"}                       # 2026-10-24 final default
    assert NC.parse_exclude("0") == set() and NC.parse_exclude("") == set()
    assert NC.parse_exclude("1") == {"anchor", "contrib", "uniform"}
    assert NC.parse_exclude("anchor,contrib") == {"anchor", "contrib"}
    with pytest.raises(ValueError, match="unknown"):
        NC.parse_exclude("contrb")


def test_default_scheme_anchor_keeps_members_contrib_substitutes():
    """The final scheme as harder_nulls applies it (exclude_members = null in parse_exclude(None)): the anchor-active
    null may re-draw production members, the contribution-matched null never does (a substitution test)."""
    prof = _profile()
    ref = _members(prof)
    ex = NC.parse_exclude(None)
    got = {}
    for null in ("anchor", "contrib"):
        sup, diag = NC.draw_null(null, ref, prof, 20, NC.rng_for("k", null, 0), exclude_members=null in ex)
        got[null] = (sup, diag)
    assert got["anchor"][1]["exclude_members"] is False and got["contrib"][1]["exclude_members"] is True
    assert got["contrib"][1]["n_overlap"] == 0
    assert all(not set(got["contrib"][0][s]) & set(v) for s, v in ref.items())
    assert got["anchor"][1]["n_overlap"] > 0                           # 300-latent pools, ~16% members: some re-drawn


def test_short_pool_takes_everything_and_flags_it():
    prof = _profile()
    s = prof["sites"][0]
    ref = {s: sorted(prof["lag"][s])[:200]}                          # more members than anchor-active non-members
    sup, diag = NC.draw_null("anchor", ref, prof, 0, random.Random(0), exclude_members=True)
    avail = [l for l, g in prof["lag"][s].items() if g == 0 and l not in set(ref[s])]
    assert sorted(sup[s]) == sorted(avail) and diag["short_sites"] == 1
    bb = diag["by_bin"]
    assert sum(v[0] for v in bb.values()) == 200 and sum(v[1] for v in bb.values()) == len(avail)


def test_frequency_decile_matching_exact_when_available():
    prof = _profile(d=4000)
    ref = _members(prof, per_site=(10, 10, 10))
    sup, diag = NC.draw_null("anchor", ref, prof, 20, random.Random(3), exclude_members=True)   # every bin available
    assert diag["n_exact"] == diag["n_ref"] == diag["n_drawn"] and diag["dist0"] == 0
    for s in ref:
        assert sorted(prof["fdec"][s][l] for l in ref[s]) == sorted(prof["fdec"][s][l] for l in sup[s])


def test_contribution_matching_first_then_frequency():
    prof = _profile(d=4000)
    ref = _members(prof, per_site=(10, 10, 10))
    prof["cbin"] = NC.member_bins([prof["contrib"][(s, l)] for s in ref for l in ref[s]], prof["contrib"])
    sup, diag = NC.draw_null("contrib", ref, prof, 20, random.Random(3), exclude_members=True)
    for s in ref:
        assert sorted(prof["cbin"][(s, l)] for l in ref[s]) == sorted(prof["cbin"][(s, l)] for l in sup[s])
    assert diag["dist0"] == 0
    sp_ref, sp_null = NC.set_profile(ref, prof, 20), NC.set_profile(sup, prof, 20)
    assert sp_ref["cbin_hist"][0] == sp_ref["cbin_hist"][11] == 0       # members lie inside their own range
    assert sp_null["cbin_hist"] == sp_ref["cbin_hist"]


def test_member_bins():
    mv = list(range(1, 101))                                            # members 1..100
    b = NC.member_bins(mv, {"lo": 0.5, "a": 1, "m": 50, "z": 100, "hi": 101})
    assert b == {"lo": -1, "a": 0, "m": 4, "z": 9, "hi": 10}
    full = NC.member_bins(mv, {i: float(i) for i in mv})
    assert Counter(full.values()) == Counter({k: 10 for k in range(10)})


def test_nearest_bin_fallback():
    s = (0, "attn")
    prof = dict(lag={s: {1: 0, 2: 0, 3: 0}}, fdec={s: {1: 0, 2: 5, 3: 9, 10: 6}}, count={}, contrib={}, cdec={})
    sup, diag = NC.draw_null("anchor", {s: [10]}, prof, 0, random.Random(0))
    assert sup[s] == [2] and diag["n_exact"] == 0 and diag["dist0"] == 1   # decile 6 -> nearest is 5
    assert diag["by_bin"] == {"6": [1, 1, 0, 1]}


def test_member_inclusion_toggle():
    prof = _profile()
    ref = _members(prof)
    sup, _ = NC.draw_null("uniform", ref, prof, 20, random.Random(0), exclude_members=False)
    assert any(set(sup[s]) & set(ref[s]) for s in ref)


# ----------------------------------------------------------------------------------------------- coefficients / sets
def test_alpha_inert_is_the_dan17_median():
    assert abs(NC.alpha_inert(400, 0.05, 0.05) - 0.7975) < 5e-5        # DAN-17: median alpha exactly 0.798
    assert NC.alpha_inert(400, 0.05, 0.05, "cosine") is None
    st = NC.alpha_stats([0.7975] * 8 + [1.3, 0.2], inert=NC.alpha_inert(400, 0.05, 0.05), tol=0.01)
    assert st["alpha_moved"] == 0.2 and st["alpha_up"] == 0.1 and st["alpha_down"] == 0.1 and st["alpha_n"] == 10
    assert NC.alpha_stats([0.5, 0.5, 0.5, 2.0])["alpha_inert"] == 0.5       # modal fallback


def test_set_profile_and_coverage():
    prof = _profile()
    ref = _members(prof)
    sp = NC.set_profile(ref, prof, 3)
    assert sp["n"] == sum(map(len, ref.values())) and sum(sp["fdec_hist"]) + sp["fdec_dead"] == sp["n"]
    assert sp["prefix_active"] == 1.0 and 0 <= sp["anchor_active"] <= 1
    cv = NC.coverage(ref, prof)
    assert cv["w63"] == 1.0 and cv["w0"] <= cv["w3"] <= cv["w16"]


def test_train_subset():
    assert NC.train_subset(list(range(32))) == list(range(0, 32, 2))
    assert NC.train_subset(list(range(48))) == list(range(0, 48, 3))
    assert NC.train_subset([4, 5]) == [4, 5]


# ----------------------------------------------------------------------------------------------- alpha = 1 path
def test_alpha1_scoring_path_subprocess():
    """harder_nulls.score_job: 'one' passes every member at alpha = 1 (rescore_alpha1.ones), 'fit' passes None;
    'train' scores the training-split record. Imported in a fresh interpreter (the driver's settings are read at
    import)."""
    code = r"""
import sys, json
sys.path.insert(0, %r)
import harder_nulls as H
from types import SimpleNamespace as NS
from circuit.types.feature_id import FeatureID
nodes = {0: NS(metadata={"feature_id": FeatureID(3, "mlp", 5), "role": "seed"}),
         1: NS(metadata={"feature_id": FeatureID(1, "attn", 7), "role": "ablation_support", "amplitude": 0.4}),
         2: NS(metadata={"feature_id": FeatureID(2, "resid", 9), "role": "ablation_support", "amplitude": 1.7})}
c = NS(nodes=nodes)
calls = []
class R:
    def score(self, c, rec, held, alphas=None):
        calls.append(dict(rec=rec["name"], held=held, alphas=None if alphas is None else
                          {"%%d.%%s" %% k: v for k, v in alphas.items()}))
        return {"free0_tk": 1.0}
for split, mode in (("held", "one"), ("held", "fit"), ("train", "fit")):
    H.score_job(R(), c, {"name": "held"}, {"name": "train"}, split, mode)
rec = {"strong": {"train": list(range(48)), "held": [99]}, "D_train": list(range(0, 64, 2))}
order = H.target_order(12)
print(json.dumps(dict(calls=calls, alphas=H.circuit_alphas(c), tr=H.train_rec(rec, "B")["strong"]["held"],
                      trA=H.train_rec(rec, "A")["strong"]["held"], orig=rec["strong"]["held"], win=H.WIN,
                      excl=sorted(H.EXCLUDE), refit=H.DAN17_REFIT, n_order=len(order), first=order[0],
                      rn=[d.name for d in H.RN_DIRS], sub_n=H.SUB_N)))
""" % str(HERE)
    env = {k: v for k, v in os.environ.items() if k not in ("FAMILY", "SHARD", "OUT", "REPORT", "STEPS", "LAM")}
    env.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["calls"][0] == dict(rec="held", held="strong", alphas={"1.attn": {"7": 1.0}, "2.resid": {"9": 1.0}})
    assert r["calls"][1]["alphas"] is None and r["calls"][2]["rec"] == "train"
    assert sorted(r["alphas"]) == [0.4, 1.7]
    assert r["tr"] == list(range(0, 64, 4)) and r["trA"] == list(range(0, 48, 3)) and r["orig"] == [99]
    # the 2026-10-24 defaults: WIN 8, members in the anchor pool and out of the contrib pool, the frozen 126 panel,
    # DAN-17 stored dirs then refit
    assert r["win"] == 8 and r["excl"] == ["contrib"] and r["refit"] is True and r["sub_n"] == 63
    assert r["n_order"] == 126 and r["first"] == (HERE / "targets_turingllm_nulls.txt").read_text().split()[0]
    assert r["rn"] == ["out_random", "out_random_scale"]


# ----------------------------------------------------------------------------------------------- report logic
def _rows():
    """Synthetic rows: 10 targets; WCM passes 6; anchor null 3 draws per target, never passes; anchor_a1 0 pass;
    one vacuous row; one error row."""
    rows = []
    good = dict(free0_tk=1.0, freeM_topk_tk=1.0, freeN_topk_tk=1.0, phi_sup_blind_tk=0.95, phi_cf_alpha_blind_tk=0.9)
    bad = dict(free0_tk=0.1, freeM_topk_tk=0.0, freeN_topk_tk=0.05, phi_sup_blind_tk=0.2, phi_cf_alpha_blind_tk=0.0)
    st = dict(n=3, fdec_hist=[1] * 3 + [0] * 7, fdec_dead=0, cdec_hist=[0] * 9 + [3], log10_count_median=3.0,
              contrib_median=0.5, contrib_mean=0.6, anchor_active=1.0, prefix_active=1.0)
    for t in range(10):
        key = "%d.mlp.%d" % (t, t)
        b = dict(seed=key, layer=t, kind="mlp", band=NC.band(t), n=100, n_ref=100)
        rows.append(dict(b, arm="wcm", null="wcm", draw=0, split="held", alpha="fit", **(good if t < 6 else bad)))
        rows.append(dict(b, arm="wcm", null="wcm", draw=0, split="train", alpha="fit", **good))
        for d in range(3):
            sc = dict(bad, free0_tk=0.1 * d)
            rows.append(dict(b, arm="anchor", null="anchor", draw=d, split="held", alpha="fit", alpha_moved=0.5,
                             alpha_median=1.1, set=st, n_exact=8, n_drawn=10, dist0=2, short_sites=0,
                             overlap_wcm=3, overlap_frac=0.3, by_bin={"0": [6, 6, 6, 0], "9": [4, 4, 2, 2]}, **sc))
            rows.append(dict(b, arm="anchor", null="anchor", draw=d, split="train", alpha="fit",
                             **(good if t < 4 else bad)))
            rows.append(dict(b, arm="anchor_a1", null="anchor", draw=d, split="held", alpha="one", **bad))
    rows.append(dict(seed="0.mlp.0", arm="contrib", draw=0, split="held", error="X"))
    rows.append(dict(seed="9.mlp.9", arm="wcm_a1", null="wcm", draw=0, split="held", alpha="one", vacuous_tk=True,
                     layer=9, kind="mlp", band="L8-11", **good))
    return rows


def test_report_tables():
    df = RP.frame(_rows())
    assert "contrib" not in set(df.arm) and df.vacuous.sum() == 1
    ov = RP.table_overview(df).set_index("arm")
    assert ov.loc["wcm", "targets"] == 10 and ov.loc["wcm", "pass_pct"] == "60.0"
    assert ov.loc["anchor", "targets"] == 10 and ov.loc["anchor", "draws"] == 30 and ov.loc["anchor", "pass_pct"] == "0.0"
    assert "wcm_a1" not in ov.index                                   # its only row is vacuous
    lo, hi = RP.wilson(6, 10)
    assert 0.31 < lo < 0.32 and 0.83 < hi < 0.84
    th = RP.table_train_held(df).set_index("arm")
    assert th.loc["anchor", "pass_train"] == "40.0" and th.loc["anchor", "pass_held"] == "0.0"
    assert th.loc["anchor", "fits_not_generalise"] == "40.0"
    q = RP.table_quantiles(df).set_index("arm")
    assert q.loc["anchor", "targets"] == 10 and q.loc["anchor", "wcm_pass_all_fail"] == "60.0"
    # WCM Z 1.0 on 6 targets (above every draw: q 1); 0.1 on 4 (draws 0.0 / 0.1 / 0.2: q 0.5)
    assert q.loc["anchor", "q_Z"] == "1.00" and q.loc["anchor", "top_Z"] == "60.0"
    al = RP.table_alpha(df).set_index("arm")
    assert al.loc["anchor", "moved"].startswith("0.50")
    sets = RP.table_sets(df, [dict(set_wcm=_rows()[2]["set"])]).set_index("set")
    assert set(sets.index) == {"wcm", "anchor"} and sets.loc["anchor", "members"] == 30
    cells = RP.table_cells(df)
    assert len(cells) == 4                                            # 3 bands + 'all'
    sh = RP.table_shortfall(df).set_index("member_bin")               # 30 anchor draws pooled
    assert sh.loc["0", "members"] == 180 and sh.loc["0", "exact_pct"] == "100.0"
    assert sh.loc["9", "members"] == 120 and sh.loc["9", "exact_pct"] == "50.0" and sh.loc["9", "mean_bin_dist"] == "0.50"
    mt = RP.table_match(df).set_index("arm")
    assert mt.loc["anchor", "overlap_frac"].startswith("0.30")


def test_quantile_ties_and_nans():
    assert RP.quantile(0.5, [0.1, 0.5, 0.9]) == 0.5
    assert RP.quantile(1.0, [0.1, 0.2]) == 1.0
    assert math.isnan(RP.quantile(float("nan"), [0.1])) and math.isnan(RP.quantile(0.3, [None]))


def test_report_build_writes_markdown(tmp_path):
    with open(tmp_path / "nulls.jsonl", "w") as fh:
        for r in _rows():
            fh.write(json.dumps(r) + "\n")
    (tmp_path / "profile.shard0.jsonl").write_text(json.dumps(dict(
        seed="0.mlp.0", win=3, pool_total=500, n_ref=100, cover_wcm={"w%d" % w: 0.5 for w in NC.COVER_WINDOWS} | {"prefix": 0.9},
        set_wcm=_rows()[2]["set"], dan17_source="refit")) + "\n")
    text, summary = RP.build(tmp_path)
    assert "## 1. Per arm" in text and "| wcm |" in text and summary["pool"][0]["targets"] == 1
    assert summary["pool"][0]["dan17_refit"] == 1 and summary["pool"][0]["members_in_pool"] == "included"
    assert "## 9. Match shortfall" in text and len(summary["shortfall"]) == 2
