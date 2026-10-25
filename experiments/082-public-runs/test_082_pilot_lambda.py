"""B5 tests (CPU): the lambda pilot report and its pre-stated rule (D2), on synthetic sweep rows and on the 062 TuringLLM
sweep (skipped when its outputs are absent). Also B4's per-family table (062/family_table.py).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest experiments/082-public-runs/test_082_pilot_lambda.py -q
"""
import importlib
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_lambda as P  # noqa: E402
import family_table as F  # noqa: E402  (062-h100-protocol-v1, on sys.path via pilot_lambda)

E062 = HERE.parent / "062-h100-protocol-v1"
GOOD = dict(free0_tk=1.0, freeM_topk_tk=1.0, freeN_topk_tk=1.0, phi_sup_blind_tk=1.0, vacuous_tk=False)
BAD = dict(GOOD, free0_tk=0.5)


# ------------------------------------------------------------------------------------------------ the rule
class TestRule:
    def test_keeps_1e3_inside_the_range_inclusive(self):
        for m in (100, 494, 1000, 100.0, 999.9):
            assert P.choose({3e-4: 2000, 1e-3: m, 3e-3: 494})[0] == 1e-3

    def test_outside_the_range_picks_the_median_nearest_494(self):
        assert P.choose({3e-4: 2600, 1e-3: 1001, 3e-3: 560})[0] == 3e-3
        assert P.choose({3e-4: 450, 1e-3: 99, 3e-3: 40})[0] == 3e-4
        # 1e-3 itself can win when it is still the nearest
        assert P.choose({3e-4: 2600, 1e-3: 1001, 3e-3: 5000})[0] == 1e-3

    def test_nearest_is_measured_on_a_log_scale(self):
        # linear |m - 494|: 300 (194) beats 700 (206); log: 700 (0.151) beats 300 (0.217)
        lam, why = P.choose({1e-3: 1200, 3e-3: 700, 1e-2: 300})
        assert lam == 3e-3 and "log scale" in why
        assert P.choose({1e-3: 50, 3e-3: 2000, 1e-2: 100})[0] == 3e-3          # 2000: 0.607 < 100: 0.694
        assert P.log_dist(988) == P.log_dist(247) and P.log_dist(0) == float("inf")

    def test_ties_go_to_the_lambda_nearer_1e3_then_the_larger(self):
        assert P.choose({3e-4: 988, 1e-3: 2000, 1e-2: 247})[0] == 3e-4            # equal size distance; log10 .52 < 1
        assert P.choose({1e-4: 988, 1e-3: 2000, 1e-2: 247})[0] == 1e-2            # equal both ways: the larger lambda

    def test_rule_cannot_be_applied_without_1e3(self):
        assert P.choose({3e-4: 500, 3e-3: 500})[0] is None
        lam, why = P.choose({3e-4: 500, 1e-3: None, 3e-3: 500})
        assert lam is None and "no fitted circuit" in why
        assert P.choose({3e-4: 500, 1e-3: float("nan")})[0] is None

    def test_missing_lambdas_are_not_candidates(self):
        assert P.choose({3e-4: None, 1e-3: 5000, 3e-3: 3000})[0] == 3e-3


# ------------------------------------------------------------------------------------------- synthetic run
def rows_for(lam, sizes, layers, quality, prefix="t"):
    out = []
    for j, (n, L) in enumerate(zip(sizes, layers)):
        out.append(dict(seed="%d.mlp.%d" % (L, j), arm="W_%g" % lam, held="strong", weighted=True, lam=lam, n=n,
                        **quality))
    return out


def write_run(tmp_path, rows):
    d = tmp_path / "sweep"
    d.mkdir(parents=True)
    half = len(rows) // 2
    for i, part in enumerate((rows[:half], rows[half:])):
        (d / ("eval.shard%d.jsonl" % i)).write_text("".join(json.dumps(r) + "\n" for r in part))
    return tmp_path


@pytest.fixture
def run(tmp_path):
    """Pythia-shaped pilot, 6 targets (layers 1..5 + 1): 3e-4 passes everything but is far too large; 1e-3 is
    too large too (median 1,500); 3e-3 has median 600 and fails everything. The size rule must pick 3e-3."""
    layers = [1, 2, 3, 4, 5, 1]
    rows = (rows_for(3e-4, [3000] * 6, layers, GOOD) + rows_for(1e-3, [1400, 1500, 1500, 1500, 1600, 1500], layers, GOOD)
            + rows_for(3e-3, [500, 600, 600, 600, 700, 600], layers, BAD))
    rows.append(dict(seed="9.mlp.99", arm="U_0.001", held="strong", weighted=False, lam=1e-3, n=10, **GOOD))
    rows.append(dict(seed="5.mlp.4", arm="W_0.003", skip="rejected"))         # later duplicate row wins: rejected
    rows.append(dict(seed="1.mlp.7", arm="W_0.003", error="RuntimeError: boom"))
    root = write_run(tmp_path, rows)
    tg = tmp_path / "targets.txt"
    tg.write_text("\n".join(["%d.mlp.%d" % (L, j) for j, L in enumerate(layers)] + ["1.mlp.7", "2.mlp.77"]) + "\n")
    return root, tg


class TestReport:
    def test_size_rule_ignores_pass_rate(self, run):
        root, tg = run
        r = P.report(root, "pythia-70m", tg)
        per = {x["lam"]: x for x in r["per_lambda"]}
        assert r["recommended_lam"] == 3e-3 and r["adopted"] is False
        assert per[3e-4]["pooled"]["pass_rate_reported"] == 1.0 and per[3e-3]["pooled"]["pass_rate_reported"] == 0.0
        assert per[1e-3]["pooled"]["median_size"] == 1500 and "outside [100, 1000]" in r["reason"]
        assert r["bands"] == ["L0-1", "L2-3", "L4-5"]

    def test_per_band_stats_and_size_share(self, run):
        root, tg = run
        per = {x["lam"]: x for x in P.report(root, "pythia-70m", tg)["per_lambda"]}
        b = per[3e-3]["by_band"]
        assert b["L0-1"]["circuits"] == 2 and b["L0-1"]["median_size"] == 550       # 1.mlp.0 (500) + 1.mlp.5 (600)
        assert b["L4-5"]["circuits"] == 1                                         # 5.mlp.4 was rejected later
        assert per[3e-3]["pooled"]["share_in_1e2_1e3"] == 1.0 and per[1e-3]["pooled"]["share_in_1e2_1e3"] == 0.0
        assert per[3e-3]["pooled"]["circuits"] == 5 and per[3e-3]["pooled"]["targets"] == 5

    def test_bookkeeping_and_coverage(self, run):
        root, tg = run
        r = P.report(root, "pythia-70m", tg)
        assert r["bookkeeping"]["ignored_arms"] == ["U_0.001"]
        assert r["bookkeeping"]["not_fitted"]["0.003"] == {"rejected": 1, "error": 1}
        per = {x["lam"]: x for x in r["per_lambda"]}
        assert per[3e-3]["targets_missing"] == 0 and per[1e-3]["targets_missing"] == 1      # 1.mlp.7 only at 3e-3
        assert r["coverage"]["no_row_at_any_lambda"] == 1 and r["complete"] is False
        assert r["paired_reference"]["targets"] == 5

    def test_vacuous_rows_are_out_of_the_pass_denominator(self, tmp_path):
        rows = rows_for(1e-3, [500, 500], [2, 3], GOOD) + rows_for(1e-3, [500], [4], dict(BAD, vacuous_tk=True))
        rows[-1]["seed"] = "4.mlp.9"
        s = P.report(write_run(tmp_path, rows), "pythia-70m")["per_lambda"][0]["pooled"]
        assert s["pass_rate_reported"] == 1.0 and s["vacuous"] == 1 and s["circuits"] == 3

    def test_complete_run_and_json(self, tmp_path):
        rows = rows_for(1e-3, [400, 600], [2, 9], GOOD) + rows_for(3e-3, [200, 300], [2, 9], GOOD)
        root = write_run(tmp_path / "run", rows)
        tg = tmp_path / "t.txt"
        tg.write_text("2.mlp.0\n9.mlp.1\n")
        out = tmp_path / "rec.json"
        r = P.main(["--out", str(root), "--family", "gpt2-small", "--targets", str(tg), "--json", str(out)])
        j = json.loads(out.read_text())
        assert r["complete"] is True and j["recommended_lam"] == 1e-3 and j["adopted"] is False
        assert j["bands"] == ["L0-4", "L5-7", "L8-11"] and j["grid"] == [1e-3, 3e-3]

    def test_empty_run(self, tmp_path):
        r = P.report(tmp_path, "pythia-70m")
        assert r["recommended_lam"] is None and r["per_lambda"] == [] and r["complete"] is False


@pytest.mark.skipif(not (E062 / "out_full" / "out" / "sweep").exists(), reason="062 sweep outputs not on disk")
def test_on_the_062_turingllm_sweep():
    r = P.report(E062 / "out_full" / "out", "turingllm", E062 / "targets_sweep.txt")
    per = {x["lam"]: x for x in r["per_lambda"]}
    assert r["grid"] == [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3] and r["recommended_lam"] == 1e-3
    assert per[1e-3]["pooled"]["circuits"] == 190 and 100 <= per[1e-3]["pooled"]["median_size"] <= 1000
    assert r["complete"] is True and r["coverage"]["no_row_at_any_lambda"] == 1


# ---------------------------------------------------------------------------------------- B4 family table
class TestFamilyTable:
    def test_turingllm_default_is_the_old_constants(self, monkeypatch):
        for v in ("FAMILY", "BANDS", "K_TOPK"):
            monkeypatch.delenv(v, raising=False)
        f = F.family()
        assert f["name"] == "turingllm" and f["k"] == 128 and f["n_layer"] == 12 and f["n_full"] == 15046
        assert f["kinds"] == ("attn", "mlp", "resid")
        assert [F.band_label(*b) for b in f["bands"]] == ["L0-4", "L5-7", "L8-11"]
        assert [F.band_label(*b) for b in f["curve_bands"]] == ["L0-3", "L4-7", "L8-11"]

    def test_public_families(self):
        p, g = F.family("pythia-70m"), F.family("gpt2-small")
        assert (p["k"], p["n_layer"], p["n_full"]) == (16, 6, 1024)
        assert [F.band_label(*b) for b in p["bands"]] == ["L0-1", "L2-3", "L4-5"] and p["curve_bands"] == p["bands"]
        assert (g["k"], g["n_layer"], g["n_full"]) == (32, 12, 1034) and "resid_mid" in g["kinds"]
        assert g["bands"] == F.family("turingllm")["bands"] and g["curve_bands"] == F.family("turingllm")["curve_bands"]

    def test_env_overrides_and_validation(self, monkeypatch):
        monkeypatch.setenv("BANDS", "0,1,2,3,4,5")
        monkeypatch.setenv("K_TOPK", "8")
        f = F.family("pythia-70m")
        assert f["k"] == 8 and len(f["bands"]) == 6 and F.band_label(*f["bands"][3]) == "L3"
        monkeypatch.setenv("BANDS", "0-2,4-5")
        with pytest.raises(ValueError, match="tile"):
            F.family("pythia-70m")
        with pytest.raises(ValueError, match="unknown FAMILY"):
            F.family("llama")

    def test_merge_depth_band_matches_the_old_rule_and_follows_family(self, monkeypatch):
        import merge as M
        old = lambda L: "L0-4" if L <= 4 else ("L5-7" if L <= 7 else "L8-11")   # merge.py before B4
        for v in ("FAMILY", "BANDS", "K_TOPK"):
            monkeypatch.delenv(v, raising=False)
        M = importlib.reload(M)
        assert all(M.depth_band(L) == old(L) for L in range(-1, 14)) and M.K_TOPK == 128 and M.N_FULL == 15046
        try:
            monkeypatch.setenv("FAMILY", "pythia-70m")
            M = importlib.reload(M)
            assert [M.depth_band(L) for L in range(6)] == ["L0-1", "L0-1", "L2-3", "L2-3", "L4-5", "L4-5"]
            assert M.K_TOPK == 16 and M.N_FULL == 1024
        finally:
            monkeypatch.delenv("FAMILY")
            importlib.reload(M)

    def test_pass_rule_depth_thirds(self, monkeypatch):
        import pandas as pd
        import pass_rule as R
        for v in ("FAMILY", "BANDS", "K_TOPK"):
            monkeypatch.delenv(v, raising=False)
        R = importlib.reload(R)
        assert R.K == 128 and R.DEPTH_EDGES == [-1, 3, 7, 11] and R.DEPTH_LABELS == ["L0-3", "L4-7", "L8-11"]
        cut = pd.cut(pd.Series(range(12)), R.DEPTH_EDGES, labels=R.DEPTH_LABELS)
        assert list(cut) == ["L0-3"] * 4 + ["L4-7"] * 4 + ["L8-11"] * 4
        try:
            monkeypatch.setenv("FAMILY", "gpt2-small")
            assert importlib.reload(R).K == 32
        finally:
            monkeypatch.delenv("FAMILY")
            importlib.reload(R)
