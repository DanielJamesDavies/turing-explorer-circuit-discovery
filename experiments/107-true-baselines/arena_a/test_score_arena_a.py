"""CPU unit tests for score_arena_a.py's node mapping (no model): BOS drop, z2 map, site rule, collapse, Ge / Dunefsky
read-out, recovery and coverage; the ho_xdoc subset and the per-target summary.  .venv/bin/python -m pytest -q experiments/107-true-baselines/arena_a/test_score_arena_a.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_arena_a as S  # noqa: E402

SITES = ["blocks.%d.%s.sae.hook_feature_acts" % (l, s) for l in range(12) for s in ("attn.hook_z", "hook_mlp_out")]


def test_ge_site_and_z2_map():
    assert S.ge_site(SITES[0]) == ("z", 0)
    assert S.ge_site(SITES[11]) == ("tc", 5)
    assert S.to_ours("z", 5, 24575) == (5, "z", 24575)
    assert S.to_ours("z", 5, 24576) == (5, "z2", 0)
    assert S.to_ours("z", 7, 49151) == (7, "z2", 24575)
    assert S.to_ours("tc", 5, 24575) == (5, "tc", 24575)
    with pytest.raises(ValueError):
        S.to_ours("z", 4, 24576)          # only L5 / L7 are 49,152 wide
    with pytest.raises(ValueError):
        S.to_ours("tc", 7, 30000)


def test_ge_rows():
    # (site index, pos_bos, latent): site 14 = L7 z, site 15 = L7 tc
    rows = S.ge_rows([[14, 3, 30000], [14, 0, 5], [15, 2, 7], [1, 1, 9]], SITES)
    assert rows == [(7, "z2", 3, 30000 - 24576), (7, "z", 0, 5), (7, "tc", 2, 7), (0, "tc", 1, 9)]


def test_upstream_rule():
    assert S.is_upstream((3, "z"), (4, "tc"))
    assert S.is_upstream((4, "z"), (4, "tc")) and S.is_upstream((4, "z2"), (4, "tc"))
    assert not S.is_upstream((4, "tc"), (4, "tc"))                     # own site
    assert not S.is_upstream((4, "tc"), (4, "z")) and not S.is_upstream((4, "z"), (4, "z"))
    assert not S.is_upstream((5, "z"), (4, "tc"))
    assert not S.is_upstream((2, "z"), (4, "tc"), kinds=("tc",))       # a2 universe: tc only


def test_map_nodes_bos_offsite_collapse():
    raw = [(0, "tc", 0, 1),            # BOS: dropped
           (0, "tc", 1, 1), (0, "tc", 5, 1),   # same latent, two positions: 2 (latent,pos), 1 latent
           (2, "z", 3, 8),
           (4, "tc", 2, 9),            # target's own site: dropped (off-site)
           (6, "z", 2, 1)]             # downstream: dropped
    m = S.map_nodes(raw, (4, "tc"))
    assert m["latpos"] == {(0, "tc", 0, 1), (0, "tc", 4, 1), (2, "z", 2, 8)}   # pos shifted by -1
    assert m["members"] == {(0, "tc"): {1}, (2, "z"): {8}}
    assert (m["n_raw"], m["n_bos"], m["n_offsite"], m["n_latpos"], m["n_latent"]) == (6, 1, 2, 3, 2)


def test_collapse_and_sig():
    c = S.collapse({(1, "tc", 0, 5), (1, "tc", 3, 5), (1, "tc", 3, 6), (2, "z2", 0, 5)})
    assert c == {(1, "tc"): {5, 6}, (2, "z2"): {5}}
    assert S.members_sig(c) == S.members_sig({(2, "z2"): {5}, (1, "tc"): {6, 5}, (3, "z"): set()})


def _row(name, typ, layer, token, latent, contrib, root=False):
    return dict(name=name, type=typ, layer=layer, token=token, head=None, latent=latent, contrib=contrib, root=root)


def test_dunefsky_rows_and_coverage():
    rows = [_row("mlp3tc[7]@-1", "mlp", 3, -1, 7, 2.0, root=True),
            _row("mlp0tc[4]@5", "mlp", 0, 5, 4, 1.0),
            _row("mlp0tc[4]@0", "mlp", 0, 0, 4, 0.1),                 # BOS token
            _row("mlp1tc[2]@-1", "mlp", 1, -1, 2, 0.5),                # last token
            _row("mlp3tc[7]@4", "mlp", 3, 4, 7, 0.2),                  # root latent at an earlier token: dropped
            _row("attn2[1]@5", "attn", 2, 5, None, 0.3),
            _row("embed0@5", "embed", 0, 5, None, 0.3)]
    got = S.dunefsky_rows(rows, n_tok=7, root_layer=3, root_latent=7)
    assert got == [(0, "tc", 5, 4), (0, "tc", 0, 4), (1, "tc", 6, 2)]
    m = S.map_nodes(got, (3, "tc"), kinds=("tc",))
    assert m["latpos"] == {(0, "tc", 4, 4), (1, "tc", 5, 2)} and m["n_bos"] == 1 and m["n_latent"] == 2
    res = dict(rows=rows, root=2.0, error_nodes=[
        dict(name="mlp3tc[7]prune_error3@-1", type="prune_error", token=-1, contrib=0.5),
        dict(name="mlp0tc[4]prune_error0@5", type="prune_error", token=5, contrib=0.2),
        dict(name="mlp3tc[7]bias_error3@-1", type="bias_error", token=-1, contrib=-1.0)])
    assert S.dunefsky_coverage(res) == pytest.approx(0.75)


def test_coverage_ours():
    rows = [_row("mlp3tc[7]@-1", "mlp", 3, -1, 7, 2.0, root=True)]
    res = dict(rows=rows, root=2.0, error_nodes=[
        dict(name="mlp3tc[7]prune_error3@-1", type="prune_error", token=-1, contrib=0.5),
        dict(name="mlp3tc[7]bias_error3@-1", type="bias_error", token=-1, contrib=-2.0)])
    assert S.dunefsky_coverage(res) == pytest.approx(0.75)
    assert S.dunefsky_coverage_ours(res) == pytest.approx(1 - 0.5 / 4.0)
    res["error_nodes"] = res["error_nodes"][:1]                         # no bias_error node: undefined
    assert S.dunefsky_coverage_ours(res) is None


def test_from_ours_inverts_to_ours():
    for kind, layer, lat in (("z", 5, 30000), ("z", 7, 3), ("z", 2, 24575), ("tc", 9, 17800)):
        l, k, i = S.to_ours(kind, layer, lat)
        hp, li = S.from_ours(l, k, i)
        assert S.ge_site(hp) == (kind, layer) and li == lat


def test_ge_job_nodes():
    # Dunefsky: their positions (our pos + 1 for BOS)
    u = dict(latpos={(0, "tc", 4, 4), (1, "tc", 5, 2)}, members={(0, "tc"): {4}, (1, "tc"): {2}})
    assert S.ge_job_nodes(u) == [("blocks.0.hook_mlp_out.sae.hook_feature_acts", 5, 4),
                                 ("blocks.1.hook_mlp_out.sae.hook_feature_acts", 6, 2)]
    # WCM: position None = every non-BOS position (expanded by ge_recovery_tcz), z2 back to the dictionary's index
    u = dict(latpos=None, members={(5, "z2"): {1}, (2, "tc"): {9}})
    assert S.ge_job_nodes(u) == [("blocks.2.hook_mlp_out.sae.hook_feature_acts", None, 9),
                                 ("blocks.5.attn.hook_z.sae.hook_feature_acts", None, 24577)]


def test_fake_circuit_for_donor_test():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from eval.donor_test import keep_and_scales, n_members
    from eval.wcm_eval import circuit_alphas, circuit_key
    al = {(2, "tc"): {9: 1.0, 3: 1.0}, (5, "z2"): {1: 0.5}}
    c = S.fake_circuit("7.z.11", al)
    assert circuit_key(c) == "7.z.11" and circuit_alphas(c) == al and n_members(c) == 3
    kt, sc = keep_and_scales(c, torch.device("cpu"))
    assert kt[(2, "tc")].tolist() == [3, 9] and sc[(5, "z2")].tolist() == [0.5]
    # donor identity: members + alphas only (the prompt does not matter: the test runs on the held-out bases)
    ua = dict(alphas=None, members={(2, "tc"): {3, 9}}, j=0)
    ub = dict(alphas=None, members={(2, "tc"): {9, 3}}, j=1)
    uc = dict(alphas={(2, "tc"): {3: 1.0, 9: 0.7}}, members={(2, "tc"): {3, 9}}, j=None)
    assert S.alpha_sig(ua) == S.alpha_sig(ub) != S.alpha_sig(uc)


def test_compress_codes_exact():
    """compressed codes give the same dense code and the same ResamplePatcher fill (value > 0 entries, any order)"""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from sae.dense import sparse_topk_to_dense
    g = torch.Generator().manual_seed(0)
    D = 50
    ta = torch.relu(torch.randn(2, 3, D, generator=g) - 1.2)          # ReLU code at k = d_sae
    ti = torch.arange(D).expand(2, 3, D)
    v, i = S.compress_codes(ta, ti)
    assert v.shape[-1] == int((ta > 0).sum(-1).max()) < D
    assert torch.equal(sparse_topk_to_dense(v, i, D, dtype=torch.float32), ta)
    # the patcher's fill: scatter_add of every valid (> 0) donor entry
    def fill(a, j):
        out = torch.zeros(2, 3, D)
        return out.scatter_add_(-1, j, a * (a > 0))
    assert torch.allclose(fill(v, i), fill(ta, ti))
    v2, i2 = S.compress_codes(ta, ti, kmax=v.shape[-1] + 4)            # padding to a larger k: still exact
    assert torch.equal(sparse_topk_to_dense(v2, i2, D, dtype=torch.float32), ta)


def test_ge_recovery():
    r = S.ge_recovery(dict(embed=3.0, bias_model=1.0, bias_dict=-2.0, err_tc=0.5, err_z=0.5), 2.0)
    assert r["interp"] == 1.5 and r["nonerr"] == 1.0 and r["total"] == 1.5 and r["bias"] == -0.5 and r["err"] == 0.5


def test_xdoc_split():
    # strong pool rows 0..7: docs; train = 0..3, held = 4..7; mid rows: docs 20, 21 (train = both)
    sdoc, mdoc = [10, 11, 12, 13, 10, 14, 12, 15], [20, 14]
    idx, info = S.xdoc_split(sdoc, [0, 1, 2, 3], [4, 5, 6, 7], mdoc, [0, 1], 4, "wcm")
    assert idx == [3] and info["n"] == 1 and info["n_train_docs"] == 6 and info["note"] is None   # 10, 12 strong; 14 mid
    idx, info = S.xdoc_split(sdoc, [0, 1, 2, 3], [4, 5, 6, 7], mdoc, [0, 1], 4, "ge_hier", prompt_pool_j=2)
    assert idx == [0, 1, 3] and info["n_train_windows"] == 1                     # only the prompt's doc (12) is out
    idx, info = S.xdoc_split(sdoc, [0], [4, 5, 6, 7], mdoc, [0, 1], 2, "wcm")   # n_ho caps the held rows
    assert idx == [] and info["n_ho"] == 2                                        # held docs 10 (strong), 14 (mid)
    idx, info = S.xdoc_split([10, 10, 10], [0], [1, 2], None, [], 2, "dunefsky", prompt_pool_j=0)
    assert idx == [] and info["n"] == 0 and info["note"].startswith("empty")
    # no provenance: no doc list, a -1 document, or a mid pool without docs for WCM
    assert S.xdoc_split(None, [0], [1], None, [], 1, "wcm")[1]["note"] == "n/a: no provenance"
    assert S.xdoc_split([-1, -1], [0], [1], None, [], 1, "dunefsky", 0)[0] is None
    assert S.xdoc_split([1, 2], [0], [1], None, [0], 1, "wcm")[1]["note"] == "n/a: no provenance"
    assert S.xdoc_split([1, 2], [0], [1], None, [0], 1, "ge_hier", 0)[0] == [0]   # baselines do not need mid docs


def test_subset_metrics_matches_full_means():
    vac = lambda a, es: a <= 0 or any(a - e < 0.05 * a for e in es)  # noqa: E731  (= wcm_eval.vacuous_v2)
    V = dict(a_pos=[2.0, 4.0, 1.0], e0=[0.0, 1.0, 0.5], eM=[0.1, 0.2, 0.3], eN=[0.0, 0.0, 0.0],
             free0=[1.8, 3.0, 0.4], freeM=[2.0, 3.5, 1.2], freeN=[1.0, 4.0, 1.0], rm_a=[1.0, 2.0, 0.9],
             rm_b=[0.1, 0.9, 0.6], rm_btc=[0.2, 1.0, 0.5], clean=[2.0, 4.0, 1.0])
    full = S.subset_metrics(V, [0, 1, 2], vac)
    m = {k: sum(v) / 3 for k, v in V.items()}
    assert full["Z"] == pytest.approx(S.ratio(m["free0"], m["e0"], m["a_pos"]))
    assert full["nec_b"] == pytest.approx(S.nec(m["rm_b"], m["clean"], m["e0"]))
    sub = S.subset_metrics(V, [0], vac)
    assert sub["n"] == 1 and sub["Z"] == pytest.approx(0.9) and sub["A"] == pytest.approx(1.0)
    assert sub["C"] == pytest.approx(0.5) and sub["nec_a"] == pytest.approx(0.5) and sub["nec_b"] == pytest.approx(0.95)
    assert sub["pass_a"] is False and sub["pass_b"] is False                       # C 0.5 out of band
    sub = S.subset_metrics(V, [2], vac)                                            # e0 0.5, eM 0.3 vs a 1.0: fine
    assert sub["vacuous"] is False
    V2 = dict(V, e0=[1.99, 0, 0])
    assert S.subset_metrics(V2, [0], vac)["vacuous"] is True and S.subset_metrics(V2, [0], vac)["pass_b"] is None


def _r(method, key, j, t, n, pb, pa, uid=None, es="ho", **kw):
    kv = dict(ge_hier=("tau/a_t", 0.1 * (t + 1), t), ge_posthoc=("tau/a_t", 0.1 * (t + 1), t), dunefsky=("num_branches", t, t), wcm=("lambda", t, t))[method]
    return dict(dict(method=method, key=key, j=j, knob=kv[0], knob_value=kv[1], knob_index=kv[2], n_latent=n,
                     pass_b=pb, pass_a=pa, eval_set=es, num_iters=5 if method == "dunefsky" else None,
                     uid=uid or "%s|%s|p%s|t%s" % (method, key, j, t), Z=1.0, A=1.0, C=1.0, nec_a=0.5, nec_b=1.0),
                **kw)


def test_per_target_best_setting_and_rollup():
    rows = [
        # Ge target: setting 0 passes b on 1 of 2 prompts (n 50, 60); setting 1 on 2 of 2 (n 90, 100); setting 2 on
        # 2 of 2 too but larger -> best = setting 1 by size
        _r("ge_hier", "1.tc.5", 0, 0, 50, True, False), _r("ge_hier", "1.tc.5", 1, 0, 60, False, False),
        _r("ge_hier", "1.tc.5", 0, 1, 90, True, True), _r("ge_hier", "1.tc.5", 1, 1, 100, True, False),
        _r("ge_hier", "1.tc.5", 0, 2, 300, True, True), _r("ge_hier", "1.tc.5", 1, 2, 400, True, False),
        # its home rows: setting 1 passes on prompt 0 only
        _r("ge_hier", "1.tc.5", 0, 1, 90, True, True, es="home"), _r("ge_hier", "1.tc.5", 1, 1, 100, False, False,
                                                                      es="home"),
        # a vacuous Dunefsky target
        _r("dunefsky", "2.tc.7", 0, 10, 5, None, None), _r("dunefsky", "2.tc.7", 1, 10, 6, None, None),
        # Dunefsky: same pass_b rate, pass_a decides (b20 has pass_a on 1 of 2)
        _r("dunefsky", "3.tc.9", 0, 10, 5, True, False), _r("dunefsky", "3.tc.9", 1, 10, 6, False, False),
        _r("dunefsky", "3.tc.9", 0, 20, 9, True, True), _r("dunefsky", "3.tc.9", 1, 20, 9, False, False),
        # WCM: one circuit per lambda; lambda 2 fails, lambda 1 passes (xdoc subset of 3 rows, failing)
        _r("wcm", "1.tc.5", None, 2, 30, False, False),
        _r("wcm", "1.tc.5", None, 1, 70, True, True, xdoc=dict(n=3, pass_b=False, pass_a=False)),
    ]
    pt = {(e["method"], e["key"]): e for e in S.per_target(rows)}
    g = pt[("ge_hier", "1.tc.5")]
    assert g["best_setting"] == ["t", 1] and g["n_pass_b"] == 2 and g["best_n"] == 2 and g["n_latent"] == 95
    assert g["n_prompts"] == 2 and g["n_settings"] == 3 and g["home_pass_b_rate"] == 0.5 and g["min_n_pass_b"] == 50
    assert g["majority_pass_b"] and not g["majority_pass_a"] and g["xdoc_note"] == "not scored"
    assert pt[("dunefsky", "2.tc.7")]["vacuous"] and "best" not in pt[("dunefsky", "2.tc.7")]
    d = pt[("dunefsky", "3.tc.9")]
    assert d["best"] == "b20 i5" and d["pass_b_rate"] == 0.5 and not d["majority_pass_b"]   # 1 of 2 is not a majority
    w = pt[("wcm", "1.tc.5")]
    assert w["n_prompts"] is None and w["best"] == "lambda 1" and w["majority_pass_b"]
    assert w["xdoc_n"] == 3 and w["xdoc_pass_b_rate"] == 0.0
    roll = S.per_method_rollup(list(pt.values()))
    assert roll["ge_hier"] == dict(targets=1, vacuous=0, scored=1, pass_b=1, pass_a=0, any_pass_b=1)
    assert roll["dunefsky"] == dict(targets=2, vacuous=1, scored=1, pass_b=0, pass_a=0, any_pass_b=1)
    assert roll["wcm"]["pass_b"] == 1 and roll["wcm"]["scored"] == 1
    assert g["best_rule"] == d["best_rule"] == w["best_rule"] == "pass"
    # a re-scored unit (same uid, eval set): the last row wins
    rows2 = rows + [_r("wcm", "1.tc.5", None, 2, 30, True, True)]
    assert S.per_target(rows2)[-1]["best"] == "lambda 2"


def test_fold_band_and_q():
    assert S.fold_band(0.75) == pytest.approx(0.75) and S.fold_band(1.5) == pytest.approx(0.75)
    assert S.fold_band(1.0) == 1.0 and S.fold_band(3.0) == pytest.approx(0.375) and S.fold_band(-0.2) == -0.2
    assert S.fold_band(0.5) == 0.5 and S.fold_band(2.0) == pytest.approx(0.5625)
    assert S.q_score(dict(Z=1.0, A=3.0, C=0.9, nec_b=0.8)) == pytest.approx(0.375)       # A far above the band
    assert S.q_score(dict(Z=1.0, A=1.0, C=0.9, nec_b=0.2)) == 0.2
    assert S.q_score(dict(Z=None, A=1.0, C=0.9, nec_b=0.2)) is None


def test_per_target_q_fallback():
    """no setting passes (b) or (a): highest median q wins, not the smallest circuit; then smaller n on a q tie"""
    rows = [
        # t0: empty-ish circuit, q ~ 0; t1: Z / A / C near 1, nec_b 0.6 -> q 0.6; t2: A 4.0 (q 0.28), smaller
        _r("ge_posthoc", "4.z.1", 0, 0, 1, False, False, Z=0.0, A=0.0, C=0.0, nec_b=0.0),
        _r("ge_posthoc", "4.z.1", 1, 0, 1, False, False, Z=0.0, A=0.0, C=0.0, nec_b=0.0),
        _r("ge_posthoc", "4.z.1", 0, 1, 80, False, False, Z=1.0, A=0.9, C=1.1, nec_b=0.6),
        _r("ge_posthoc", "4.z.1", 1, 1, 90, False, False, Z=0.9, A=1.0, C=1.0, nec_b=0.6),
        _r("ge_posthoc", "4.z.1", 0, 2, 40, False, False, Z=1.0, A=4.0, C=1.0, nec_b=0.9),
        _r("ge_posthoc", "4.z.1", 1, 2, 40, False, False, Z=1.0, A=4.0, C=1.0, nec_b=0.9),
        # a q tie between t0 (n 50) and t1 (n 20): the smaller wins
        _r("dunefsky", "5.tc.2", 0, 10, 50, False, False, nec_b=0.5),
        _r("dunefsky", "5.tc.2", 0, 20, 20, False, False, nec_b=0.5),
        # one passing prompt anywhere: the pass rule, not the fallback (t1 passes a on 1 of 2)
        _r("ge_hier", "4.z.1", 0, 0, 5, False, False, nec_b=0.99), _r("ge_hier", "4.z.1", 0, 1, 70, False, True,
                                                                        nec_b=0.1),
    ]
    pt = {(e["method"], e["key"]): e for e in S.per_target(rows)}
    p = pt[("ge_posthoc", "4.z.1")]
    assert p["best_rule"] == "q-fallback" and p["best_setting"] == ["t", 1] and p["q"] == pytest.approx(0.6)
    d = pt[("dunefsky", "5.tc.2")]
    assert d["best_rule"] == "q-fallback" and d["best"] == "b20 i5"
    h = pt[("ge_hier", "4.z.1")]
    assert h["best_rule"] == "pass" and h["best_setting"] == ["t", 1]
    md = S.summary_per_target(rows, {}, lambda v, d=2: "-" if v is None else "%.*f" % (d, v))
    assert sum("(best: q-fallback)" in line for line in md) == 2


def test_ratio_nec_pass():
    assert S.ratio(1.0, 0.0, 2.0) == 0.5 and S.ratio(1.0, 2.0, 2.0) is None
    assert S.nec(0.5, 2.0, 0.0) == 0.75 and S.nec(0.5, 0.0, 0.0) is None
    assert S.passes(1.0, 0.8, 1.4, 0.95) and not S.passes(1.0, 0.8, 1.6, 0.95) and not S.passes(1.0, 1, 1, None)
