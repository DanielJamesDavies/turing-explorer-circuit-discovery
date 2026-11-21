"""Experiment 081, the v4 rerun (2026-11-18, DAN-221) on the tiny CPU model of test_081_engine_cpu.py:
  - WCM_VERSION=v4 reaches the mask engine: R on (gamma_R 1.0, donors = the C pool = the training items' own
    counterfactual prompts, draws made), geometric lr with a 0.1 warm-up at peak lr x3.15, absolute decay 0.05; no
    rank-keep / censor / necessity term (refused by src for a logit objective); v2 unchanged
  - W_CF: minimal pairs in the training rows (clean, cf of the same item, scored for the clean answers)
  - score_cf: minimal-pair faithfulness on the held-out clean vs counterfactual prompts
  - run.py STOP_EPOCH

  PYTHONPATH=src python -m pytest experiments/081-task-circuits/test_081_v4_cpu.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_081_engine_cpu import _engine, _task, restore_config  # noqa: E402,F401  (fixture re-exported)
import task_common as C  # noqa: E402

STEPS = 10


def _task_last(**kw):
    """_task with the counterfactual changing the LAST prompt token (the toy model's anchor read barely depends on
    token 0, so the stock counterfactual moves the full model's metric by ~0)."""
    t = _task(**kw)
    for r in t["rows"]:
        r["cf_tokens"] = list(r["tokens"][:-1]) + [(r["tokens"][-1] + 11) % 63 + 1]
    return t


@pytest.fixture
def v4(monkeypatch):
    monkeypatch.setattr(C, "WCM_VERSION", "v4")


class TestV4Engine:
    def test_v4_reaches_the_engine(self, v4):
        eng = _engine()
        B = eng.batch(_task())
        kw = eng.engine_kwargs(B, 1e-3, True, STEPS, 0.05, "gap")
        eng.reset()
        assert kw["resample_weight"] == 1.0 and kw["resample_k"] == 1 and kw["resample_tokens"] is B["c_tokens"]
        assert kw["lr_schedule"] == "geometric" and kw["warmup_frac"] == 0.1
        assert kw["lr"] == pytest.approx(0.05 * 3.15) and kw["weight_decay"] == 0.05
        assert kw["train_fill"] == "sparse" and kw["rank_weight"] == 0.0
        for arm in ("W", "U"):
            _, prov, _, _ = eng.fit(B, arm, 1e-3, 0.05, STEPS)
            print("v4 %s prov:" % arm, {k: prov.get(k) for k in (
                "wcm_version", "resample_weight", "resample_n_donors", "resample_draws", "resample_fill_calls",
                "lr_schedule", "lr_peak", "lr_sum", "warmup_steps", "weight_decay", "rank_weight", "censor_temp",
                "necessity_weight", "sparse_fill_calls")})
            assert prov["wcm_version"] == "v4"
            assert prov["resample_weight"] == 1.0 and prov["resample_n_donors"] == B["c_tokens"].shape[0] == 24
            assert prov["resample_draws"] > 0 and prov["resample_fill_calls"] > 0
            assert prov["lr_schedule"] == "geometric" and prov["lr_peak"] == pytest.approx(0.1575)
            assert prov["warmup_steps"] == 1 and prov["weight_decay"] == 0.05
            assert prov["rank_weight"] is None and prov["censor_temp"] is None and prov["necessity_weight"] == 0.0
            assert prov["sparse_fill_calls"] > 0
            assert "resample" in prov["curve"]["terms"] and max(prov["curve"]["terms"]["resample"]) > 0
        from config import config
        assert config.discovery.learned_mask.lr == C.PROD_LR                    # reset after the fit

    def test_v2_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(C, "WCM_VERSION", "v2")
        eng = _engine()
        B = eng.batch(_task())
        kw = eng.engine_kwargs(B, 1e-3, True, STEPS, 0.05, "gap")
        eng.reset()
        assert "resample_tokens" not in kw and "resample_weight" not in kw
        assert kw["lr"] == 0.05 and kw["weight_decay"] == pytest.approx(C.weight_decay(0.05))
        _, prov, _, _ = eng.fit(B, "W", 1e-3, 0.05, 4)
        assert prov["resample_weight"] is None and prov["lr_peak"] == 0.05

    def test_set_task_under_v4(self, v4):
        """greater-than's answer mode (token sets): the R donors are its XX01-style counterfactuals."""
        eng = _engine()
        B = eng.batch(_task(mode="set"))
        _, prov, _, _ = eng.fit(B, "W", 1e-3, 0.05, STEPS)
        assert prov["logit_metric"] == "set" and prov["resample_weight"] == 1.0 and prov["resample_draws"] > 0


class TestMinimalPairs:
    def test_cf_rows_in_training(self, v4):
        """W_CF: presentations alternate clean prompt / the same item's counterfactual, both with the clean item's
        answers as the targets; the budget (steps x batch) is unchanged."""
        eng = _engine()
        t = _task()
        B = eng.batch(t)
        kw = eng.engine_kwargs(B, 1e-3, True, STEPS, 0.05, "gap", cf_train=True)
        eng.reset()
        rows = eng.presentations(B, STEPS, 4)
        n = len(rows)
        assert kw["pos_tokens"].shape[0] == n == 40
        items = rows[:n // 2]
        ct, ca = B["cf_all"]
        for k in range(n):
            i = items[k // 2]
            if k % 2 == 0:
                assert torch.equal(kw["pos_tokens"][k], B["tokens"][i]) and int(kw["pos_argmax"][k]) == int(
                    B["anchors"][i])
                assert torch.equal(kw["pos_valid"][k], B["valid"][i])
            else:
                assert torch.equal(kw["pos_tokens"][k], ct[i]) and int(kw["pos_argmax"][k]) == int(ca[i])
                assert torch.equal(kw["pos_valid"][k], B["cf_valid_all"][i])
                assert ct[i][0] != B["tokens"][i][0]                              # the toy cf changes token 0
            assert int(kw["target_tokens"][k]) == int(B["good"][i, 0])
        assert max(items) < B["n_train"]                                         # never a held-out item
        _, prov, _, _ = eng.fit(B, "W_CF", 1e-3, 0.05, STEPS)
        assert prov["cf_train"] is True and prov["present"]["n_cf"] == 20 and prov["resample_weight"] == 1.0
        _, prov_w, _, _ = eng.fit(B, "W", 1e-3, 0.05, STEPS)
        assert prov_w["cf_train"] is False
        with pytest.raises(ValueError, match="cf_train"):
            eng.engine_kwargs(eng.batch(_task(mode="seq")), 1e-3, True, STEPS, 0.05, "gap", cf_train=True)

    def test_minimal_pair_score(self, v4):
        """score_cf: the full model's clean - cf difference equals the probe's, per item on the first n held-out
        items; frames are (D_f - D_E) / (D_full - D_E); n = every held-out item reuses the main row."""
        eng = _engine()
        t = _task_last()
        B = eng.batch(t)
        alphas, _, _, _ = eng.fit(B, "W", 1e-3, 0.05, STEPS)
        row = eng.score(B, alphas)
        mp = eng.score_cf(B, alphas, 5)
        pv = eng.probe(B, t)
        n_tr = B["n_train"]
        clean = pv["clean"][C.PRIMARY]["values"][n_tr:n_tr + 5]
        cf = pv["cf"][C.PRIMARY]["values"][n_tr:n_tr + 5]
        assert mp["mp_n"] == 5
        assert mp["mp_task_d_full"] == pytest.approx(sum(clean) / 5 - sum(cf) / 5, rel=1e-4, abs=1e-5)
        assert mp["mp_task_cf_full"] == pytest.approx(sum(cf) / 5, rel=1e-4, abs=1e-5)
        assert abs(mp["mp_task_d_full"]) > 1e-3                                  # the contrast moves the full model
        for f, e in eng.MP_FRAMES:
            want = (mp["mp_task_d_%s" % f] - mp["mp_task_d_%s" % e]) / (mp["mp_task_d_full"] - mp["mp_task_d_%s" % e])
            assert mp["mp_task_%s" % f] == pytest.approx(want)
        full = eng.score_cf(B, alphas, 10 ** 6, clean_row=row)                   # every held-out item: main row reused
        assert full["mp_n"] == B["n_held"] == 8
        assert full["mp_task_d_full"] == pytest.approx(row["task_full"] - full["mp_task_cf_full"])
        print("minimal-pair score (toy, 8 held-out):", {k: round(v, 4) for k, v in full.items()
                                                         if isinstance(v, float)})
        assert eng.score_cf(eng.batch(_task(contrast="generated"), {"pool": t["split"]["train"][:5]}),
                            alphas, 5) is None
        # a minimal-pair contrast the circuit does not carry: the empty circuit's frames move by ~nothing between
        # clean and cf, so its mp ratios are ~0 while the full model's difference is not
        empty = eng.score_cf(B, {}, 8)
        assert abs(empty["mp_task_free0"]) < 1e-6


class TestStopEpoch:
    def test_check_stop(self, monkeypatch):
        import run as R
        monkeypatch.setattr(R, "STOP_EPOCH", None)
        R.check_stop("x")
        monkeypatch.setattr(R, "STOP_EPOCH", 1.0)                                # long past
        with pytest.raises(R.StopEpoch, match="before fit"):
            R.check_stop("fit ioi W 1e-4")
        monkeypatch.setattr(R, "STOP_EPOCH", 4e9)                                # far future
        R.check_stop("x")
