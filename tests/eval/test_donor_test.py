"""Donor-background test (eval/donor_test.py, protocol v2 §12.3).

Part 1  pure pieces: pooled recovery and its undefined cases, the resample pass rule at the band edges, the donor
        draw sequence (== 099 / 100's loop, deterministic, per target), the donor-fires split
Part 2  the R pool: the 8-gram-prefiltered exact / near filter == 099's brute-force pair_stats filter; record rows
Part 3  end to end on the tiny CPU engine of test_wcm_eval (MockModel + EncBank): deterministic rows, a circuit of
        every latent recovers 1 on every pool, the empty circuit 0, the self-donor identity
Part 4  CLI helpers: arm parsing, production-row lookup, summary

The GPU equivalence with experiment 100's rows is experiments/103-donor-stage (equiv.py).
"""
from pathlib import Path

import pytest
import torch

from eval import donor_test as DT
from eval import wcm_eval as W

ROOT = Path(__file__).resolve().parents[2]


# =============================================================================================== Part 1
class TestPooled:
    def test_pools_before_dividing(self):
        # (mean(1, 3) - mean(0, 1)) / (4 - 0.5) = 1.5 / 3.5, not the mean of per-draw ratios
        assert DT.pooled([1.0, 3.0], [0.0, 1.0], 4.0) == pytest.approx(1.5 / 3.5)
        per_draw = ((1 - 0) / 4 + (3 - 1) / 3) / 2
        assert DT.pooled([1.0, 3.0], [0.0, 1.0], 4.0) != pytest.approx(per_draw)

    def test_undefined(self):
        assert DT.pooled([1.0], [2.0], 2.0) is None                      # natural == empty
        assert DT.pooled([1.0], [3.0], 2.0) is None                      # natural below empty
        assert DT.pooled([1.0], [2.0], 2.0 + 1e-10) is None              # denominator <= 1e-9
        assert DT.pooled([], [0.0], 1.0) is None
        assert DT.pooled([1.0], [], 1.0) is None
        assert DT.pooled([1.0], [0.0], 1.0) == 1.0

    def test_split_recovery(self):
        out = torch.tensor([[2.0, 0.0], [4.0, 1.0]])
        emp = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        nat = torch.tensor([4.0, 2.0])
        m = torch.tensor([[True, False], [True, False]])
        assert DT.split_recovery(out, emp, nat, m) == pytest.approx(3.0 / 4.0)
        assert DT.split_recovery(out, emp, nat, ~m) == pytest.approx(0.5 / 2.0)
        assert DT.split_recovery(out, emp, nat, torch.zeros_like(m)) is None
        assert DT.split_recovery(out, emp + 4.0, nat, m) is None


def _prod(z=1.0, a=1.0, c=1.0, nec=0.95):
    return {"free0_tk": z, "freeM_topk_tk": a, "freeN_topk_tk": c, "phi_sup_blind_tk": nec}


class TestResamplePass:
    def test_band_is_the_production_band(self):
        assert W.BAND == (0.75, 1.5)

    @pytest.mark.parametrize("a, c, ok", [(0.75, 0.75, True), (1.5, 1.5, True), (0.7499, 1.0, False),
                                          (1.0, 0.7499, False), (1.5001, 1.0, False), (1.0, 1.50001, False),
                                          (None, 1.0, False), (1.0, None, False), (float("nan"), 1.0, False)])
    def test_edges(self, a, c, ok):
        assert DT.resample_pass(_prod(), {"A": a, "C": c, "R": 0.0}) is ok

    def test_r_is_not_in_the_rule(self):
        assert DT.resample_pass(_prod(), {"A": 1.0, "C": 1.0, "R": None})
        assert DT.resample_pass(_prod(), {"A": 1.0, "C": 1.0, "R": 0.0})

    @pytest.mark.parametrize("bad", [dict(z=0.7499), dict(a=1.51), dict(c=None), dict(nec=0.899)])
    def test_needs_the_production_pass(self, bad):
        assert not DT.resample_pass(_prod(**bad), {"A": 1.0, "C": 1.0})
        assert DT.resample_pass(_prod(), {"A": 1.0, "C": 1.0})

    def test_subset_of_production(self):
        g = torch.Generator().manual_seed(1)
        for _ in range(500):
            v = (torch.rand(6, generator=g) * 1.2 + 0.4).tolist()
            p = _prod(*v[:4])
            assert not DT.resample_pass(p, {"A": v[4], "C": v[5]}) or W.passes(p)


def _draws_100(layer, idx, sizes, nb, ns, draws=8):
    """experiments/100-scrub-frontier/run.py's donor loop (the generator part), verbatim in effect."""
    rng = torch.Generator().manual_seed(20261028 + layer * 1000 + idx)
    out = {}
    for ds in ("A", "C", "R"):
        out[ds] = [torch.cat([torch.randperm(sizes[ds], generator=rng)[:nb],
                              torch.randperm(sizes[ds], generator=rng)[:ns]]) for _ in range(draws)]
    return out


class TestDraws:
    @pytest.mark.parametrize("layer, idx", [(2, 34021), (6, 209), (10, 8194)])
    def test_is_100s_sequence(self, layer, idx):
        sizes = dict(A=48, C=16, R=512)
        got, short = DT.draw_selections(layer, idx, sizes, 16, 16)
        ref = _draws_100(layer, idx, sizes, 16, 16)
        for ds in DT.POOLS:
            assert len(got[ds]) == 8
            assert all(torch.equal(a, b) for a, b in zip(got[ds], ref[ds]))
        assert short == dict(A=False, C=False, R=False)

    def test_deterministic_and_per_target(self):
        sizes = dict(A=48, C=16, R=300)
        a, _ = DT.draw_selections(3, 7, sizes, 16, 16)
        b, _ = DT.draw_selections(3, 7, sizes, 16, 16)
        c, _ = DT.draw_selections(3, 8, sizes, 16, 16)
        assert all(torch.equal(x, y) for ds in DT.POOLS for x, y in zip(a[ds], b[ds]))
        assert not all(torch.equal(x, y) for ds in DT.POOLS for x, y in zip(a[ds], c[ds]))

    def test_one_donor_per_base_without_replacement(self):
        sel, _ = DT.draw_selections(1, 1, dict(A=48, C=16, R=100), 16, 16)
        for ds in DT.POOLS:
            for s in sel[ds]:
                assert s.shape == (32,)
                assert len(set(s[:16].tolist())) == 16 and len(set(s[16:].tolist())) == 16

    def test_short_pool(self):
        sel, short = DT.draw_selections(1, 1, dict(A=48, C=12, R=100), 16, 12)
        assert short == dict(A=False, C=True, R=False)
        assert all(s.shape == (28,) for s in sel["C"])
        assert set(sel["C"][0][:12].tolist()) == set(range(12))        # the first permutation is used whole

    def test_empty_pool_raises(self):
        with pytest.raises(RuntimeError):
            DT.draw_selections(1, 1, dict(A=48, C=0, R=100), 16, 16)


# =============================================================================================== Part 2
def _brute_keep(pool, rows):
    from circuit import heldout_rule as HR
    keep = []
    for j, row in enumerate(pool.numpy()):
        ex, _, lcr, c8 = HR.pair_stats(row[None], rows.numpy())
        _, _, _, c8r = HR.pair_stats(rows.numpy(), row[None])
        if not (ex.any() or HR.near_of(lcr, c8).any() or (c8r >= 0.5).any()):
            keep.append(j)
    return keep


class TestPoolFilter:
    def _planted(self):
        g = torch.Generator().manual_seed(0)
        pool = torch.randint(1, 5000, (40, 64), generator=g)
        rec = torch.randint(1, 5000, (60, 64), generator=g)
        rec[0] = pool[3]                                                   # exact copy
        rec[1, 10:44] = pool[5, 0:34]                                      # a 34-token run, shifted
        rec[2, :30] = pool[7, 30:60]                                       # a 30-token run: not near by lcr
        rec[3, 0:8] = pool[9, 50:58]                                       # one shared 8-gram: not near
        idx = torch.randperm(57, generator=g)[:30]                         # >= 50% of pool[11]'s 8-grams, scattered
        for n, i in enumerate(idx.tolist()):
            rec[4 + n % 20, :8] = pool[11, i:i + 8]
        rec[30, :] = 0                                                     # padding rows
        rec[31, :32] = pool[13, :32]                                       # exactly 32 shared tokens
        return pool, rec

    def test_equals_brute_force(self):
        pool, rec = self._planted()
        keep = DT.filter_pool(pool, [rec[:25], rec[25:]])
        assert keep == _brute_keep(pool, rec)
        assert 3 not in keep and 5 not in keep and 13 not in keep          # exact, run 34, run 32
        assert 9 in keep                                                   # one shared 8-gram is not a duplicate

    def test_random_rows_equal_brute_force(self):
        g = torch.Generator().manual_seed(4)
        pool = torch.randint(0, 6, (25, 64), generator=g)                  # tiny vocabulary: many shared grams
        rec = torch.randint(0, 6, (30, 64), generator=g)
        assert DT.filter_pool(pool, [rec]) == _brute_keep(pool, rec)

    def test_no_records_keeps_all(self):
        pool, _ = self._planted()
        assert DT.filter_pool(pool, []) == list(range(40))

    def test_record_rows(self):
        z = lambda n, v: torch.full((n, 64), v, dtype=torch.long)          # noqa: E731
        rec = dict(strong=dict(pos=z(4, 1)), mid=dict(pos=z(3, 2)), neg=z(5, 3))
        assert DT.record_rows(rec).shape == (12, 64)
        rec["neg_held_v2"] = z(2, 4)
        assert DT.record_rows(rec).shape == (14, 64)
        rec["mid"] = None
        assert DT.record_rows(rec).shape == (11, 64)


# =============================================================================================== Part 3
@pytest.fixture
def tiny(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_t_wcm_eval", ROOT / "tests" / "eval" / "test_wcm_eval.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    from circuit import protocol_contexts as PC
    from circuit.probe_dataset import ProbeDataset
    inf, bank = T._engine()
    latent, probe = T._target_and_contexts(inf, bank, n_pos=20, n_train=12, n_neg=16)
    # bases: 8 held-out activating + 8 held-out contrast (n_neg_train 8)
    pr = ProbeDataset(pos_tokens=probe.pos_tokens, target_tokens=probe.target_tokens, neg_tokens=probe.neg_tokens,
                      pos_argmax=probe.pos_argmax, metadata=dict(n_train=12, n=20, n_neg_train=8))
    monkeypatch.setattr(PC, "eval_probe", lambda rec, held, device, holdout_frac=None: pr)
    G = dict(inference=inf, bank=bank, device=torch.device("cpu"))
    key = "%d.%s.%d" % (T.SEED_SITE[0], T.SEED_SITE[1], latent)
    g = torch.Generator().manual_seed(9)
    pool = torch.randint(0, inf.emb.shape[0], (24, probe.pos_tokens.shape[1]), generator=g)
    return T, G, key, latent, probe, pool


def _full_circuit(T, bank, latent):
    from circuit.types.feature_id import FeatureID
    from store.circuits import Circuit, CircuitNode
    c = Circuit(name="full", metadata={})
    c.add_node(CircuitNode(metadata={"feature_id": FeatureID(T.SEED_SITE[0], T.SEED_SITE[1], latent), "role": "seed"}))
    for (l, kd) in sorted(W.upstream_sites(bank, *T.SEED_SITE)):
        for i in range(bank.d_sae):
            c.add_node(CircuitNode(metadata={"feature_id": FeatureID(l, kd, i), "role": "ablation_support"}))
    return c


def _empty_circuit(T, latent):
    from circuit.types.feature_id import FeatureID
    from store.circuits import Circuit, CircuitNode
    c = Circuit(name="empty", metadata={})
    c.add_node(CircuitNode(metadata={"feature_id": FeatureID(T.SEED_SITE[0], T.SEED_SITE[1], latent), "role": "seed"}))
    return c


class TestEndToEnd:
    def test_rows_complete_and_deterministic(self, tiny):
        T, G, key, latent, probe, pool = tiny
        c = T._circuit(G["inference"], G["bank"], latent, probe)
        r1, raw1 = DT.donor_target(G, key, {}, {"x": c}, pool, draws=3, sync=lambda: None)
        r2, _ = DT.donor_target(G, key, {}, {"x": c, "y": c}, pool, draws=3, sync=lambda: None)
        a, b = r1["x"], r2["x"]
        for f in ("recovery", "recovery_pre", "recovery_split", "empty_means", "means", "empty_leak",
                  "donor_fires_frac", "natural_mean"):
            assert a[f] == b[f], f                                         # same donors whatever else shares them
        assert r2["y"]["recovery"] == a["recovery"]
        assert set(a["recovery"]) == set(DT.VARIANTS)
        assert all(set(v) == set(DT.POOLS) for v in a["recovery"].values())
        assert a["n_base"] == 8 and a["n_silent"] == 8 and a["pool_sizes"] == dict(A=12, C=8, R=24)
        assert len(a["empty_means"]["A"]) == 3 and len(raw1["x"]["empty"]["R"][0]) == 8
        assert a["natural_mean"] > 0

    def test_every_latent_recovers_one_and_empty_zero(self, tiny):
        T, G, key, latent, probe, pool = tiny
        rows, _ = DT.donor_target(G, key, {}, {"full": _full_circuit(T, G["bank"], latent),
                                              "empty": _empty_circuit(T, latent)}, pool, draws=2, sync=lambda: None)
        for ds in DT.POOLS:
            e = rows["empty"]["recovery"]["fit"][ds]
            assert e is None or e == pytest.approx(0.0, abs=1e-6)
            f = rows["full"]["recovery"]["fit"][ds]
            if e is not None:
                assert f == pytest.approx(1.0, abs=1e-4)
                assert rows["full"]["recovery"]["a1"][ds] == pytest.approx(1.0, abs=1e-4)
                # pinned: every latent at its clean natural value on the base = the natural stream
                assert rows["full"]["recovery"]["pinned"][ds] == pytest.approx(1.0, abs=1e-4)
        assert rows["full"]["self_empty"]["max_abs"] < 1e-4
        assert any(rows["empty"]["recovery"]["fit"][ds] is not None for ds in DT.POOLS)   # the test is not vacuous


# =============================================================================================== Part 4
class TestCliHelpers:
    def test_parse_arms(self, tmp_path):
        a = DT.parse_arms("main,W_0.001, U_0.001,ext=/x/{key}.wcm.pt", tmp_path)
        assert a["main"]["path"] == str(tmp_path / "main" / "circuits" / "{key}.pt") and a["main"]["rows"] == ("main", None)
        assert a["W_0.001"]["path"] == str(tmp_path / "sweep" / "circuits" / "W_0.001" / "{key}.pt")
        assert a["U_0.001"]["rows"] == ("sweep", "U_0.001")
        assert a["ext"] == dict(path="/x/{key}.wcm.pt", rows=None)
        with pytest.raises(SystemExit):
            DT.parse_arms(" , ", tmp_path)

    def test_production_rows(self, tmp_path):
        import json
        (tmp_path / "main").mkdir()
        (tmp_path / "sweep").mkdir()
        rows_m = [dict(seed="1.a.1", held="strong", free0_tk=0.1), dict(seed="1.a.1", held="mid", free0_tk=0.2),
                  dict(seed="1.a.1", held="strong", free0_tk=0.3), dict(seed="1.a.2", held="strong", error="x")]
        rows_s = [dict(seed="1.a.1", held="strong", arm="W_0.001", free0_tk=0.4),
                  dict(seed="1.a.1", held="strong", arm="U_0.001", free0_tk=0.5),
                  dict(seed="1.a.1", arm="W_0.004", skip="rejected")]
        (tmp_path / "main" / "eval.shard0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_m) + "{trunc")
        (tmp_path / "sweep" / "eval.shard3.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_s))
        arms = DT.parse_arms("main,W_0.001,U_0.001,W_0.004", tmp_path)
        got = DT.production_rows(tmp_path, arms, {"1.a.1", "1.a.2"})
        assert got[("1.a.1", "main")]["free0_tk"] == 0.3                   # the last strongest row
        assert got[("1.a.1", "W_0.001")]["free0_tk"] == 0.4
        assert got[("1.a.1", "U_0.001")]["free0_tk"] == 0.5
        assert ("1.a.2", "main") not in got and ("1.a.1", "W_0.004") not in got

    def test_summary(self):
        rec = lambda a, c: dict(fit=dict(A=a, C=c, R=0.0), a1=dict(A=0, C=0, R=0), pinned=dict(A=0, C=0, R=0))  # noqa
        rows = [dict(arm="main", prod={}, pass_prod=True, resample_pass=True, recovery=rec(1.0, 1.0), t_target_total=4,
                     n_arms_sharing=1),
                dict(arm="main", prod={}, pass_prod=True, resample_pass=False, recovery=rec(None, 1.0), t_target_total=6,
                     n_arms_sharing=2),
                dict(arm="main", prod=dict(vacuous_tk=True), pass_prod=True, resample_pass=True, recovery=rec(1, 1),
                     t_target_total=2, n_arms_sharing=1),
                dict(arm="main", error="boom")]
        s = DT.summary(rows)["main"]
        assert (s["n"], s["n_scored"], s["prod_pass"], s["resample_pass"], s["undefined_A_C"]) == (3, 2, 2, 1, 1)
