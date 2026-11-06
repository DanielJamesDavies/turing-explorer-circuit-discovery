"""Tests for the v1 method-comparison suite (experiments/083-v1-comparison, DAN-159 / DAN-43 / DAN-77).

Part 1  rankings: ordering, ties, keep_top, truncation to matched size (global and per site)
Part 2  ranking aggregation: AttrReducer (max / agg / sfc, causal-prefix mask, chunk invariance), and ig_reduce
        against circuit/instrument/ig_baseline.integrated_baseline_scores on an exactly linear stub pipeline
Part 3  matched-size selection (nearest unweighted arm per target)
Part 4  the DAN-8 pass rule and the Table 2 summary (vacuous exclusion, depth bands)
Part 5  the Ge bisection replay (034's discover_at_size, resumable)
"""
import importlib.util
import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.conftest import D_MODEL, D_SAE, KINDS, N_LAYERS, MockSAEBank

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "083-v1-comparison"


def _load(name, alias):
    path = EXP / name
    if not path.exists():
        pytest.skip("experiment code not present: %s" % path)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def C():
    return _load("cmp_common.py", "_cmp_common_083")


SITES = [(0, "attn"), (0, "mlp"), (1, "attn")]


# =============================================================================================== Part 1
class TestRankings:
    def test_order_by_abs_score_with_deterministic_ties(self, C):
        s = {SITES[0]: torch.tensor([0.0, -3.0, 1.0, 2.0]), SITES[1]: torch.tensor([2.0, 0.0, 0.0, -1.0]),
             SITES[2]: torch.tensor([0.0, 0.0, 0.0, 0.0])}
        r = C.ranking_from_site_scores(s, SITES, keep_top=0)
        got = list(zip(r["site"].tolist(), r["latent"].tolist()))
        # |-3| first; the two |2| tie: earlier site first; then the |1| tie: site 0 before site 1; zeros never ranked
        assert got == [(0, 1), (0, 3), (1, 0), (0, 2), (1, 3)]
        assert r["score"].tolist() == [-3.0, 2.0, 2.0, 1.0, -1.0]

    def test_keep_top_bounds_each_site_and_the_total(self, C):
        s = {SITES[0]: torch.arange(1, 11, dtype=torch.float32), SITES[1]: torch.arange(1, 11, dtype=torch.float32) + 0.5}
        r = C.ranking_from_site_scores(s, SITES[:2], keep_top=4)
        assert len(r["latent"]) == 4
        assert r["score"].tolist() == [10.5, 10.0, 9.5, 9.0]

    def test_custom_key_ranks_by_key(self, C):
        s = {SITES[0]: torch.tensor([5.0, 1.0, 3.0])}
        r = C.ranking_from_site_scores(s, SITES[:1], keep_top=0, key={SITES[0]: torch.tensor([0.1, 0.9, 0.5])})
        assert r["latent"].tolist() == [1, 2, 0]

    def test_truncate_global_matches_size_and_caps_at_available(self, C):
        s = {SITES[0]: torch.tensor([4.0, 3.0, 0.0]), SITES[1]: torch.tensor([0.0, 3.5, 1.0])}
        r = C.ranking_from_site_scores(s, SITES[:2], keep_top=0)
        m = C.truncate_global(r, SITES[:2], 2)
        assert m == {SITES[0]: [0], SITES[1]: [1]} and C.n_members(m) == 2
        assert C.n_members(C.truncate_global(r, SITES[:2], 100)) == 4
        assert C.truncate_global(r, SITES[:2], 0) == {}

    def test_truncate_per_site_counts_and_short_sites(self, C):
        s = {SITES[0]: torch.tensor([4.0, 3.0, 2.0, 1.0]), SITES[1]: torch.tensor([0.0, 9.0, 0.0, 0.0])}
        r = C.ranking_from_site_scores(s, SITES[:2], keep_top=0)
        m, short = C.truncate_per_site(r, SITES[:2], {SITES[0]: 2, SITES[1]: 3, SITES[2]: 0})
        assert m == {SITES[0]: [0, 1], SITES[1]: [1]}
        assert short == [SITES[1]]                         # asked 3, the site ranks only 1: all of it is taken

    def test_ranking_from_items_keeps_given_order(self, C):
        r = C.ranking_from_items([(SITES[1], 7, 0.1), (SITES[0], 2, -5.0)], SITES, keep_top=0)
        assert r["site"].tolist() == [1, 0] and r["latent"].tolist() == [7, 2]
        assert C.truncate_global(r, SITES, 1) == {SITES[1]: [7]}


# =============================================================================================== Part 2
class TestAttrReducer:
    def test_reductions_and_prefix_mask(self, C):
        a = torch.zeros(2, 3, 2)
        a[0, 0] = torch.tensor([1.0, -2.0]); a[0, 1] = torch.tensor([-3.0, 1.0]); a[0, 2] = torch.tensor([100.0, 100.0])
        a[1, 0] = torch.tensor([2.0, 0.5]); a[1, 2] = torch.tensor([-1.0, 4.0])
        red = C.AttrReducer(2)
        red.update(a, torch.tensor([1, 2]))                # context 0 reads positions 0..1: its position 2 is masked
        r = red.result()
        assert r["max"].tolist() == [-3.0, 4.0]
        assert torch.allclose(r["agg"], torch.tensor([(1 + 3 + 2 + 1) / 2, (2 + 1 + 0.5 + 4) / 2]))
        assert torch.allclose(r["sfc"], torch.tensor([(1 - 3 + 2 - 1) / 2, (-2 + 1 + 0.5 + 4) / 2]))

    def test_chunking_is_invariant(self, C):
        torch.manual_seed(0)
        a, anc = torch.randn(5, 4, 6), torch.tensor([3, 0, 2, 1, 3])
        one, two = C.AttrReducer(6), C.AttrReducer(6)
        one.update(a, anc)
        two.update(a[:2], anc[:2]); two.update(a[2:], anc[2:])
        for k in ("max", "agg", "sfc"):
            assert torch.allclose(one.result()[k], two.result()[k], atol=1e-6)


SEED_LAYER, SEED_KIND, SITE = 1, "resid", (0, "attn")
B, T = 4, 5


def _linear_stub(x0, w_map):
    """The test_ig_baseline stub: row 0 of each token sequence indexes x0, the site's output maps linearly and
    position-wise to the seed site, so IG is exact and position t only affects position t."""
    def forward_fn(tokens, activations_callback=None, patcher=None, **kwargs):
        x = x0[tokens[:, 0]]
        if activations_callback is not None:
            acts = tuple(x.clone() for _ in KINDS)
            for layer in range(N_LAYERS):
                activations_callback(layer, acts)
            return
        out0 = patcher.transform(*SITE, x.clone())
        patcher.transform(SEED_LAYER, SEED_KIND, out0 @ w_map)
    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    return inf


class TestIgReduce:
    def _setup(self):
        torch.manual_seed(3)
        bank = MockSAEBank()
        x0 = torch.randn(B, T, D_MODEL)
        tokens = torch.zeros(B, T, dtype=torch.long); tokens[:, 0] = torch.arange(B)
        kw = dict(site_floors={SITE: torch.rand(D_SAE) * 0.3}, w_seed=torch.randn(D_MODEL), b_seed=torch.tensor(0.1))
        return bank, _linear_stub(x0, torch.randn(D_MODEL, D_MODEL) * 0.2), tokens, torch.tensor([2, 4, 0, 3]), kw

    def _ours(self, A, bank, inf, tokens, anchors, kw, objective, bs):
        return A.ig_reduce(inf, bank, [SITE], tokens, anchors, seed=(SEED_LAYER, SEED_KIND), w_seed=kw["w_seed"],
                           b_seed=kw["b_seed"], objective=objective, target_act=0.7, floors=kw["site_floors"],
                           ig_steps=4, batch_size=bs)

    @pytest.mark.parametrize("objective", ["drive", "gap"])
    def test_sfc_equals_collapsed_ig_and_max_equals_single_chunk_union(self, objective):
        from circuit.instrument.ig_baseline import integrated_baseline_scores
        from circuit.instrument.position_aware import PositionAwareSpec
        A = _load("attr_baselines.py", "_attr_baselines_083")
        bank, inf, tokens, anchors, kw = self._setup()
        red, m0, m1, _ = self._ours(A, bank, inf, tokens, anchors, kw, objective, bs=2)
        ref, r0, r1 = integrated_baseline_scores(inf, bank, tokens=tokens, substitute_sites={SITE},
                                                 seed_layer=SEED_LAYER, seed_kind=SEED_KIND, pos_argmax=anchors,
                                                 objective=objective, target_act=0.7, ig_steps=4, batch_size=2, **kw)
        assert torch.allclose(red[SITE]["sfc"].double(), ref[SITE], atol=1e-5)
        assert (m0, m1) == pytest.approx((r0, r1), abs=1e-6)
        # one chunk: the position-aware union with an all-admitting cut keeps each latent's largest-|attr| entry, on the
        # chunk-mean scale (x 1/B of the per-sequence attribution)
        pa, _, _ = integrated_baseline_scores(inf, bank, tokens=tokens, substitute_sites={SITE}, seed_layer=SEED_LAYER,
                                              seed_kind=SEED_KIND, pos_argmax=anchors, objective=objective,
                                              target_act=0.7, ig_steps=4, batch_size=None,
                                              position_aware=PositionAwareSpec(peaks=anchors, select="abs", threshold=0.0),
                                              **kw)
        one, _, _, _ = self._ours(A, bank, inf, tokens, anchors, kw, objective, bs=None)
        assert torch.allclose(one[SITE]["max"].double() / B, pa[SITE], rtol=1e-5, atol=1e-6)
        assert torch.allclose(one[SITE]["max"], red[SITE]["max"], rtol=1e-4, atol=1e-4)   # per-sequence: chunk-invariant

    def test_atp_equals_ig_on_a_linear_metric(self):
        """Attribution patching = the gradient at the clean point x (natural - mean). On the linear stub with the
        linear "drive" metric the gradient is the same everywhere on the path, so AtP equals IG exactly."""
        A = _load("attr_baselines.py", "_attr_baselines_083")
        bank, inf, tokens, anchors, kw = self._setup()
        run = lambda alphas: A.ig_reduce(inf, bank, [SITE], tokens, anchors, seed=(SEED_LAYER, SEED_KIND),
                                         w_seed=kw["w_seed"], b_seed=kw["b_seed"], objective="drive", target_act=0.0,
                                         floors=kw["site_floors"], ig_steps=4, alphas=alphas, batch_size=2)
        ig, atp = run(None), run([1.0])
        for k in ("max", "agg", "sfc"):
            assert torch.allclose(ig[0][SITE][k], atp[0][SITE][k], rtol=1e-4, atol=1e-5)
        assert atp[1] is None and atp[2] == pytest.approx(ig[2], abs=1e-6)     # no alpha-0 read; same endpoint
        assert A.arm_alphas("atp") == [1.0] and len(A.arm_alphas("abl_ig")) == A.IG_STEPS

    def test_atp_is_the_clean_point_gradient(self):
        """On the nonlinear "gap" metric AtP is grad(metric) at the clean state (alpha = 1) times (natural - mean),
        per sequence; checked against a direct autograd computation."""
        from circuit.instrument.ig_baseline import InterpolatedCodeInstrument, collect_natural_codes
        A = _load("attr_baselines.py", "_attr_baselines_083")
        bank, inf, tokens, anchors, kw = self._setup()
        red, _, _, _ = A.ig_reduce(inf, bank, [SITE], tokens, anchors, seed=(SEED_LAYER, SEED_KIND),
                                   w_seed=kw["w_seed"], b_seed=kw["b_seed"], objective="gap", target_act=0.7,
                                   floors=kw["site_floors"], alphas=[1.0], batch_size=None)
        codes, res = collect_natural_codes(inf, bank, tokens, {SITE})
        inst = InterpolatedCodeInstrument(bank, {SITE}, codes, res, kw["site_floors"], 1.0, SEED_LAYER, SEED_KIND,
                                          kw["w_seed"], kw["b_seed"])
        inf.forward(tokens, patcher=inst, grad_enabled=True)
        peak = inst.seed_pre_act[torch.arange(B), anchors]
        g = torch.autograd.grad(-((peak - 0.7) ** 2).sum(), inst.anchors[SITE])[0]   # sum: per-sequence attribution
        per_seq = g * inst.deltas[SITE]
        assert torch.allclose(red[SITE]["sfc"], per_seq.sum(dim=(0, 1)) / B, rtol=1e-4, atol=1e-5)

    def test_union_uses_each_chunks_own_anchors(self):
        """The natural-size union reads each chunk's causal prefix at that chunk's anchors. With a zero floor the
        attribution sits on the latents active at each anchor; chunk 1 anchors at position 0 and chunk 2 at
        position 4, so selecting chunk 2 with chunk 1's anchors (031's ig_mean union) would admit nothing from it.
        At percentile 0 the union must be exactly the latents with any nonzero attribution."""
        A = _load("attr_baselines.py", "_attr_baselines_083")
        bank, inf, tokens, _, kw = self._setup()
        anchors = torch.tensor([0, 0, 4, 4])
        run = lambda tk, an, bs: A.ig_reduce(inf, bank, [SITE], tk, an, seed=(SEED_LAYER, SEED_KIND),
                                             w_seed=kw["w_seed"], b_seed=kw["b_seed"], objective="drive",
                                             target_act=0.0, floors={SITE: torch.zeros(D_SAE)}, ig_steps=3,
                                             batch_size=bs, union_pctl=0.0)
        red, _, _, union = run(tokens, anchors, 2)
        assert torch.equal(union[SITE], red[SITE]["max"] != 0)
        first, _, _, _ = run(tokens[:2], anchors[:2], 2)
        assert ((red[SITE]["max"] != 0) & (first[SITE]["max"] == 0)).any()   # latents only chunk 2 attributes to


# =============================================================================================== Part 3
class TestNearestSize:
    def test_log_distance_ties_and_ineligible(self, C):
        assert C.nearest_size(100, {"U_a": 50, "U_b": 190, "U_c": 1000}) == "U_b"        # log: 190 is nearer than 50
        assert C.nearest_size(100, {"U_a": 50, "U_b": 200}) == "U_a"                    # exact log tie -> smaller
        assert C.nearest_size(100, {"U_a": 0, "U_b": None}) is None
        assert C.nearest_size(None, {"U_a": 5}) is None
        assert C.nearest_size(10, {"U_x": 10, "U_y": 10}) == "U_x"                      # same size -> arm name

    def test_resolve_sizes_expands_all_prices_in_price_order(self, C):
        have = {"W_0.001": 40, "W_2.5e-05": 900, "W_0.004": 10, "U_0.001": 60, "U_1e-05": 2000}
        assert C.resolve_sizes(have, ["W_all"]) == ["W_2.5e-05", "W_0.001", "W_0.004"]
        assert C.resolve_sizes(have, ["W_0.001", "W_0.002", "U_all"]) == ["W_0.001", "U_1e-05", "U_0.001"]
        assert C.resolve_sizes(have, ["W_0.001", "W_all"]) == ["W_0.001", "W_2.5e-05", "W_0.004"]   # no duplicates

    def test_sweep_arm_names(self, C):
        assert all(C.SWEEP_ARM.match(a) for a in ("W_0.001", "W_2.5e-05", "U_1e-05"))
        assert not any(C.SWEEP_ARM.match(a) for a in ("W_0.001_s800", "W_0.001_g1_r0.03", "U_matched"))

    def test_merge_reference_modes(self, C):
        base = [dict(seed="a", arm="W_0.001", n=1, tag="062"), dict(seed="b", arm="W_0.001", n=2, tag="062")]
        local = [dict(seed="a", arm="W_0.001", n=1, tag="local"), dict(seed="c", arm="U_0.001", n=3, tag="local")]
        tags = lambda rows: sorted((r["seed"], r["tag"]) for r in rows)
        assert tags(C.merge_reference(base, local, "062")) == [("a", "062"), ("b", "062")]
        assert tags(C.merge_reference(base, local, "local")) == [("a", "local"), ("c", "local")]
        assert tags(C.merge_reference(base, local, "auto")) == [("a", "local"), ("b", "062"), ("c", "local")]
        with pytest.raises(ValueError):
            C.merge_reference(base, local, "nope")

    def test_train_hash_tracks_tokens_and_anchors(self, C):
        pos, arg = torch.arange(12).reshape(3, 4), torch.tensor([1, 2, 3])
        h = C.train_hash(pos, arg)
        assert h == C.train_hash(pos.clone(), arg.clone()) and len(h) == 10
        assert h != C.train_hash(pos, torch.tensor([1, 2, 0]))
        assert h != C.train_hash(pos.flip(0), arg)

    def test_sizes_by_arm(self, C):
        rows = [dict(seed="1.mlp.2", arm="W_0.001", n=10), dict(seed="1.mlp.2", arm="U_0.001", n=30),
                dict(seed="2.attn.5", arm="W_0.001", n=None)]
        assert C.sizes_by_arm(rows) == {"1.mlp.2": {"W_0.001": 10, "U_0.001": 30}}


# =============================================================================================== Part 4
def _row(seed, z=1.0, a=1.0, c=1.0, nec=0.95, vac=False, n=100):
    return dict(seed=seed, n=n, free0_tk=z, freeM_topk_tk=a, freeN_topk_tk=c, phi_sup_blind_tk=nec,
                phi_cf_alpha_blind_tk=1.0, vacuous_tk=vac, source="x.jsonl:1")


class TestPassRule:
    def test_band_edges_missing_and_necessity(self, C):
        assert C.passes(_row("0.mlp.1", z=0.75, a=1.5, c=1.0))
        assert not C.passes(_row("0.mlp.1", z=0.749))
        assert not C.passes(_row("0.mlp.1", a=1.501))
        assert not C.passes(_row("0.mlp.1", c=None))
        assert not C.passes(_row("0.mlp.1", nec=0.899))
        assert C.passes(_row("0.mlp.1", nec=0.9))

    def test_depth_bands(self, C):
        assert [C.depth_band(l) for l in (0, 4, 5, 7, 8, 11)] == ["L0-4", "L0-4", "L5-7", "L5-7", "L8-11", "L8-11"]

    def test_table_summary_counts_by_band_and_excludes_vacuous(self):
        T2 = _load("table2.py", "_table2_083")
        rows = [_row("0.mlp.1"), _row("3.resid.2", a=0.0), _row("6.attn.3"), _row("9.mlp.4", vac=True),
                _row("10.mlp.5", nec=0.5), _row("11.resid.6", n=300)]
        f = T2.frame(rows, "arm")
        assert f.passes.tolist() == [True, False, True, False, False, True]
        wcm_n = f.set_index("seed").n / 2
        s = T2.summarise(f, wcm_n)
        assert (s["pass"], s["N"], s["vacuous"], s["targets"]) == (3, 5, 1, 6)
        assert (s["pass_L0-4"], s["pass_L5-7"], s["pass_L8-11"]) == ("1/2", "1/1", "1/2")
        assert s["n_over_wcm"] == pytest.approx(2.0)
        assert s["A"] == pytest.approx(1.0)                # medians over non-vacuous rows


# =============================================================================================== Part 4b
class TestActivationRanking:
    def test_mean_and_coactivation_scores(self):
        F = _load("attr_fitted.py", "_attr_fitted_083")
        x = torch.tensor([[1.0, 0.0, 2.0, 0.0], [3.0, 0.0, 2.0, 0.0], [5.0, 0.0, 2.0, 1.0]])   # [contexts, latents]
        y = torch.tensor([1.0, 2.0, 3.0])
        sc = F.activation_scores({SITES[0]: x}, y)
        assert sc["mean"][SITES[0]].tolist() == pytest.approx([3.0, 0.0, 2.0, 1 / 3])
        co = sc["coact"][SITES[0]].tolist()
        assert co[0] == pytest.approx(1.0) and co[1] == 0.0 and co[2] == 0.0      # constant latents: 0, not NaN
        assert co[3] == pytest.approx(float(np.corrcoef([0, 0, 1], [1, 2, 3])[0, 1]))


class TestExactPatching:
    def test_pick_targets_is_stratified_and_deterministic(self):
        X = _load("exact_patching.py", "_exact_patching_083")
        keys = ["%d.%s.%d" % (l, k, i) for l in range(12) for k in ("attn", "mlp", "resid") for i in range(3)
                if not (l == 0 and k == "attn")]
        a, b = X.pick_targets(keys, 10), X.pick_targets(keys, 10)
        assert a == b and len(a) == len(set(a)) == 10
        from collections import Counter
        cells = Counter((X.C.depth_band(int(k.split(".")[0])), k.split(".")[1]) for k in a)
        assert len(cells) == 9                                                   # every (band, kind) cell
        assert sum(v for (bnd, _), v in cells.items() if bnd == "L8-11") == 4    # the extra draw is deep

    def test_patch_sets_one_latent_to_its_mean_and_stops_at_the_target(self):
        X = _load("exact_patching.py", "_exact_patching_083")
        bank = MockSAEBank()
        site, lat = (0, "attn"), torch.tensor([3, 7])
        x = torch.randn(2, 5, D_MODEL)
        means = torch.rand(D_SAE)
        sae = bank.saes["attn"][0]
        p = X.PatchOne(bank, site, lat, means, sae.W_dec, (1, "resid"), torch.randn(D_MODEL), torch.tensor(0.1),
                       torch.tensor([1, 4]))
        out = p.transform(0, "attn", x)
        from sae.dense import sparse_topk_to_dense
        ta, ti = bank.encode(x, "attn", 0)
        dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=torch.float32)
        for r in range(2):
            patched = dense[r].clone()
            patched[:, lat[r]] = means[lat[r]]
            expect = x[r] + (patched - dense[r]) @ sae.W_dec.T                      # decode difference, bias cancels
            assert torch.allclose(out[r], expect, atol=1e-4)
        assert torch.equal(p.transform(0, "mlp", x), x)                            # other sites live
        with pytest.raises(X._Stop):
            p.transform(1, "resid", x)
        assert p.pre.shape == (2,)

    def test_spearman(self):
        X = _load("exact_patching.py", "_exact_patching_083")
        assert X.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert X.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


# =============================================================================================== Part 5
def _discover_at_size_034(n_of, n_target, steps, tol):
    """034-ge-replication/runner.py's discover_at_size, with hier_run replaced by n_of(tau)."""
    n_all = n_of(0.0)
    if n_all <= n_target:
        return n_all, 0.0
    lo, hi = 1e-8, 1e2
    best = (n_all, 0.0)
    for _ in range(steps):
        mid = (lo * hi) ** 0.5
        n = n_of(mid)
        if abs(n - n_target) < abs(best[0] - n_target):
            best = (n, mid)
        if abs(n - n_target) <= tol * n_target:
            return n, mid
        if n > n_target:
            lo = mid
        else:
            hi = mid
    return best


class TestGePlan:
    @staticmethod
    def n_of(tau):
        return 5000 if tau == 0 else max(0, int(round(400 - 60 * math.log10(tau))))

    def _run(self, C, n_target, steps, tol, points=None):
        points = list(points or [])
        while True:
            what, x = C.ge_plan(points, n_target, steps, tol)
            if what == "done":
                return points, x
            points.append(dict(tau=x, n=self.n_of(x)))

    @pytest.mark.parametrize("n_target", [3, 120, 400, 777, 4999, 6000])
    def test_replay_equals_034(self, C, n_target):
        points, j = self._run(C, n_target, 7, 0.10)
        assert (points[j]["n"], points[j]["tau"]) == _discover_at_size_034(self.n_of, n_target, 7, 0.10)

    def test_resume_mid_bisection(self, C):
        full, j = self._run(C, 120, 7, 0.10)
        part, k = self._run(C, 120, 7, 0.10, points=full[:3])
        assert part == full and j == k

    def test_stale_points_are_refused(self, C):
        with pytest.raises(ValueError, match="replay expects"):
            C.ge_plan([dict(tau=0.0, n=5000), dict(tau=0.5, n=10)], 100, 7, 0.1)
