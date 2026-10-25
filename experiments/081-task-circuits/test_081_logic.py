"""Experiment 081 logic tests (CPU, no model): split, list halves, the held-out cap, the training presentation order
(without replacement within an epoch), padding and teacher-forced rows, competence (with and without
counterfactuals), the contrast-set rule (ranking, silence, pooling), run config, calibration, curves and claim-2 ratios,
random-null draws, the head-level structure checks against the published GPT-2 circuits, the 062 link, verdicts.

  PYTHONPATH=src python -m pytest experiments/081-task-circuits/test_081_logic.py -q
"""
import json
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_common as C  # noqa: E402


# ------------------------------------------------------------------------------------------- split and arrays
class TestSplitAndArrays:
    def test_split_is_by_group_reproducible_and_75_25(self):
        groups = ["g%d" % (i // 3) for i in range(300)]
        tr, ho = C.split_groups(groups)
        assert (tr, ho) == C.split_groups(groups)
        assert sorted(tr + ho) == list(range(300)) and not set(tr) & set(ho)
        assert {groups[i] for i in tr}.isdisjoint({groups[i] for i in ho})
        assert len({groups[i] for i in ho}) == 25 and tr != sorted(tr)

    def test_list_split_is_disjoint_complete_and_reproducible(self):
        names = ["n%d" % i for i in range(99)]
        tr, ho = C.split_list(names, tag="t")
        assert (tr, ho) == C.split_list(names, tag="t") and not set(tr) & set(ho)
        assert sorted(tr + ho) == sorted(names) and len(ho) == round(99 * C.LIST_HOLD_FRAC)
        assert tr == [x for x in names if x in set(tr)]                      # source order kept
        assert C.split_list(names, tag="other") != (tr, ho)                   # the tag seeds the split
        with pytest.raises(ValueError, match="too short"):
            C.split_list(["a"], tag="t")

    def test_held_cap_samples_with_a_fixed_seed_and_reports_the_uncapped_size(self):
        held = list(range(100, 130))
        kept, n = C.cap_held(held, cap=10, seed="x")
        assert n == 30 and len(kept) == 10 and set(kept) <= set(held) and kept == C.cap_held(held, 10, "x")[0]
        assert C.cap_held(held, cap=0, seed="x") == (held, 30)               # 0 = no cap
        assert C.cap_held(held, cap=50, seed="x") == (held, 30)
        assert C.HELDOUT_CAP == int(__import__("os").environ.get("HELDOUT_CAP", 5000))

    def test_presentations_are_without_replacement_within_an_epoch(self):
        order = C.presentation_order(7, 30, seed="t")
        assert len(order) == 30 and order == C.presentation_order(7, 30, seed="t")
        for e in range(4):                                                    # 4 full epochs, then a partial one
            assert sorted(order[7 * e:7 * e + 7]) == list(range(7))
        assert len(set(order[28:])) == 2
        assert order[:7] != order[7:14]                                       # reshuffled per epoch
        big = C.presentation_order(1600, 1600, seed="t")
        assert sorted(big) == list(range(1600))                               # a pool of 1,600: never repeated
        assert len(set(C.presentation_order(5000, 1600, seed="t"))) == 1600
        assert C.PROD_STEPS * C.PROD_BATCH == 1600
        with pytest.raises(ValueError):
            C.presentation_order(0, 4)

    def test_pad_right_anchor_and_valid(self):
        toks, anc, val = C.pad_right([[5, 6, 7], [8], [1, 2]])
        assert toks == [[5, 6, 7], [8, 0, 0], [1, 2, 0]] and anc == [2, 0, 1]
        assert val == [[True] * 3, [True, False, False], [True, True, False]]
        with pytest.raises(ValueError):
            C.pad_right([[1, 2, 3]], width=2)

    def test_token_mode_arrays(self):
        task = dict(rows=[dict(tokens=[1, 2, 3], cf_tokens=[4, 5], good=[7], bad=[8]),
                          dict(tokens=[1], cf_tokens=[9, 9, 9, 9], good=[6], bad=[8])])
        A = C.task_arrays(task, [1, 0])
        assert A["mode"] == "token" and A["width"] == 4 and A["anchors"] == [0, 2] and A["cf_anchors"] == [3, 1]
        assert A["good"] == [[6], [7]] and A["rows_per_item"] == 1

    def test_seq_mode_rows_are_teacher_forced_pairs(self):
        task = dict(rows=[dict(tokens=[1, 2], cf_tokens=None, good=[7, 8, 9], bad=[5]),
                          dict(tokens=[3], cf_tokens=None, good=[6], bad=[4, 4])])
        assert C.answer_mode(task) == "seq"
        A = C.task_arrays(task, [0, 1])
        assert A["rows_per_item"] == 2 and A["n_items"] == 2
        assert A["tokens"] == [[1, 2, 7, 8], [1, 2, 0, 0], [3, 0, 0, 0], [3, 4, 0, 0]]
        assert A["anchors"] == [1, 1, 0, 0]
        assert A["answers"] == [[7, 8, 9], [5, -1, -1], [6, -1, -1], [4, 4, -1]]
        assert A["valid"][0] == [True, True, False, False]                 # prompt positions only
        assert "cf_tokens" not in A                                        # no patch prompts

    def test_answer_modes_and_metric_kinds(self):
        spec = C.task_spec("ioi")
        assert C.answer_mode(dict(rows=[dict(good=[1], bad=[2])])) == "token"
        assert C.answer_mode(dict(logit_metric="set", rows=[dict(good=[1, 2], bad=[3])])) == "set"
        assert C.metric_kinds("seq", spec) == dict(task="seq_diff", obj="seq_diff", logit_metric="seq")
        assert C.metric_kinds("token", spec)["task"] == "logit_diff"
        gt = C.task_spec("greater_than")
        assert (gt["task"], gt["obj"], gt["logit_metric"], gt["contrast"]) == ("set_probdiff", "set_logmass", "set",
                                                                               "original")
        assert C.task_spec("factual_known")["contrast"] == "generated"
        assert C.task_spec("blimp_wh_island")["contrast"] == "generated"


# ------------------------------------------------------------------------------------------- competence
class TestCompetence:
    def test_rule_with_counterfactuals(self):
        c = C.competence([1.0] * 8 + [-0.5] * 2, [0.0] * 10)
        assert c["competent"] and c["frac_correct"] == 0.8
        assert not C.competence([1.0] * 6 + [-0.1] * 4, [0.0] * 10)["competent"]
        assert not C.competence([0.1] * 10, [0.5] * 10)["competent"]
        assert C.competence([1.0] * 7 + [-0.1] * 3, [0.0] * 10)["competent"]           # 70% inclusive

    def test_rule_without_counterfactuals(self):
        c = C.competence([1.0] * 8 + [-0.5] * 2, None)
        assert c["competent"] and c["crit_cf"] is None and c["cf_gap"] is None
        assert not C.competence([-1.0] * 10, None)["competent"]


# ------------------------------------------------------------------------------------------- contrast sets
class TestContrastRule:
    def test_cosine_rank_with_exclusions(self):
        items = [[1, 0], [0, 1]]
        cands = [[1, 0.1], [0.1, 1], [-1, 0], [1, 0]]
        r = C.cosine_rank(items, cands, exclude=[[3], []])
        assert r[0] == [0, 1, 2] and r[1] == [1, 0, 2, 3]              # ties (cosine 0) by index
        assert C.cosine_rank(items, cands, exclude=[[3], []], top_k=2) == [[0, 1], [1, 0]]
        rng = random.Random(0)
        reps = [[rng.gauss(0, 1) for _ in range(5)] for _ in range(40)]
        ex = [[i] for i in range(40)]
        full = C.cosine_rank(reps, reps, exclude=ex)
        assert C.cosine_rank(reps, reps, exclude=ex, top_k=7, chunk=3) == [f[:7] for f in full]

    def test_silence_is_no_preference_for_the_good_answer(self):
        assert C.is_silent(0.0) and C.is_silent(-2.3) and not C.is_silent(0.01)

    def test_pool_round_robin_closest_silent_deduped(self):
        ranked = [[5, 6, 7], [5, 8, 9], [6, 9]]
        silent = {(0, 5): False, (0, 6): True, (0, 7): True, (1, 5): True, (1, 8): True, (2, 6): True, (2, 9): True}
        assert C.build_pool(ranked, silent, k=10) == [6, 5, 9, 7, 8]
        assert C.build_pool(ranked, silent, k=2) == [6, 5]
        assert C.build_pool(ranked, silent, k=10, item_order=[2, 0, 1]) == [6, 7, 5, 9, 8]
        assert C.build_pool(ranked, {}, k=4) == []                              # nothing measured silent

    def test_pool_size_constant(self):
        assert C.POOL_SIZE == 64 and C.SILENT_MAX == 0.0


# ------------------------------------------------------------------------------------------- run config
class TestRunConfig:
    def test_load_and_refusals(self, tmp_path):
        p = tmp_path / "c.json"
        with pytest.raises(FileNotFoundError, match="ADOPT"):
            C.load_run_config(p)
        good = dict(arms=dict(W=dict(lams=[1e-3], lr=0.05), U=dict(lams=[1e-4], lr=0.05)), steps=400,
                    logit_norm="gap")
        p.write_text(json.dumps(good))
        assert C.load_run_config(p)["n_draws"] == C.N_DRAWS
        p.write_text(json.dumps(dict(good, arms=dict(good["arms"], W_ZA=dict(lams=[1], lr=1)))))
        with pytest.raises(ValueError, match="W_ZA"):
            C.load_run_config(p)

    def test_smoke_config_is_valid(self):
        assert C.load_run_config(HERE / "config_smoke.json")["steps"] < 100

    def test_arms(self):
        assert C.ARMS["W_ZA"]["c_term"] is False and C.ARMS["W"]["c_term"] is True
        assert C.ARMS["U"]["weighted"] is False and C.ARMS["RND"]["random"] is True


# ------------------------------------------------------------------------------------------- calibration
class TestCalibration:
    def test_window(self):
        z = {1e-4: 1.0, 3e-4: 0.95, 1e-3: 0.85, 3e-3: 0.6, 1e-2: 0.2}
        assert C.choose_window(z, k=4) == ([3e-4, 1e-3, 3e-3, 1e-2], True)
        assert C.choose_window({1e-4: 1.0, 3e-4: 0.99, 1e-3: 0.97}, k=2)[1] is False

    def test_recommend(self):
        rows = []
        for task in ("a", "b"):
            for lam, n, z in ((1e-4, 900, 1.0), (1e-3, 300, 0.9), (1e-2, 50, 0.3)):
                rows.append(dict(task=task, arm="W", lr=0.05, lam_value=lam, n=n, task_free0=z, obj_free0=z,
                                 task_pass=z >= 0.8))
            rows.append(dict(task=task, arm="W", lr=0.1, lam_value=1e-3, error="boom"))
        S = C.calib_summary(rows)
        rec = C.recommend(S, k=3)
        assert rec["arms"]["W"]["lr"] == 0.05 and rec["arms"]["W"]["lams"] == [1e-4, 1e-3, 1e-2]
        assert rec["arms"]["W"]["brackets"] and not rec["warnings"]


# ------------------------------------------------------------------------------------------- curves
def _row(lam, n, z, a=None, c=None, nec=0.95, arm="W"):
    a = z if a is None else a
    c = z if c is None else c
    return dict(arm=arm, lam_value=lam, n=n, task_free0=z, task_freeM_topk=a, task_freeN_topk=c, task_nec=nec,
                task_acc_free0=0.9, task_free0_a1=0.1)


class TestCurves:
    def test_curve_rules(self):
        cv = C.curve([_row(1e-2, 50, 0.5), _row(1e-3, 200, 0.9), _row(1e-4, 800, 1.0, c=0.2),
                      dict(arm="W", lam_value=3e-3, error="x")])
        assert [p["n"] for p in cv] == [50, 200, 800]
        assert [p["passed"] for p in cv] == [False, True, False]
        assert [p["passed_ZA"] for p in cv] == [False, True, True]          # Z + A ignores the failing C

    def test_size_ratio(self):
        w = C.curve([_row(1e-2, 100, 0.5), _row(1e-3, 1000, 0.9)])
        u = C.curve([_row(1e-4, 1000, 0.5, arm="U"), _row(1e-5, 10000, 0.9, arm="U")])
        sr = C.size_ratio(w, u)
        assert sr["reach_ratio"] == pytest.approx(10.0) and sr["pass_W"] == 1000 and sr["pass_U"] == 10000
        u2 = C.curve([_row(1e-4, 1000, 0.5, arm="U"), _row(1e-5, 10000, 0.6, arm="U")])
        assert C.size_ratio(w, u2)["reach_ratio"] == float("inf")

    def test_u_at_w_and_primary(self):
        w = C.curve([_row(1e-3, 1000, 0.9)])
        u = C.curve([_row(1e-4, 100, 0.2, arm="U"), _row(1e-5, 10000, 1.0, arm="U")])
        assert C.u_at_w_sizes(w, u)[0]["U_Z"] == pytest.approx(0.6)
        cv = C.curve([_row(1e-2, 50, 0.5), _row(3e-3, 120, 0.9), _row(1e-3, 300, 1.0)])
        assert C.primary_lambda(cv) == (3e-3, True)


# ------------------------------------------------------------------------------------------- random null
class TestRandomNull:
    def test_draw(self):
        counts = {(0, "attn"): 3, (5, "mlp"): 2}
        active = {(0, "attn"): list(range(10)), (5, "mlp"): [7]}
        sup, short = C.draw_support(counts, active, random.Random("t"))
        assert len(sup[(0, "attn")]) == 3 and sup[(5, "mlp")] == [7] and short == 1
        assert C.draw_support(counts, active, random.Random("t"))[0] == sup

    def test_alphas_round_trip(self):
        al = {(3, "resid_mid"): {5: 1.5, 2: 0.25}, (0, "attn"): {1: 1.0}, (1, "resid"): {}}
        back = C.alphas_from_json(json.loads(json.dumps(C.alphas_to_json(al))))
        assert back == {(3, "resid_mid"): {5: 1.5, 2: 0.25}, (0, "attn"): {1: 1.0}}


# ------------------------------------------------------------------------------------------- structure
def _attn(layer, heads, bucket=C.ANCHOR, w=1.0):
    return dict(layer=layer, kind="attn", latent=layer, alpha=1.0, act={bucket: 1.0}, contrib={bucket: w},
                heads={bucket: {str(h): v for h, v in heads.items()}})


class TestStructure:
    def test_head_weights_and_enrichment(self):
        prof = [_attn(9, {6: 3.0, 1: 1.0}), _attn(10, {0: -2.0, 5: 2.0})]
        hw = C.head_weights(prof, C.ANCHOR)
        assert hw[(9, 6)] == pytest.approx(0.75) and hw[(10, 0)] == pytest.approx(0.5)
        e = C.head_enrichment(hw, ((9, 6), (9, 9), (10, 0)))
        assert e["share"] == pytest.approx(1.25 / 2) and e["base"] == pytest.approx(3 / 24)
        assert e["enrichment"] == pytest.approx(5.0)
        assert C.head_enrichment({}, ((9, 6),))["enrichment"] is None

    def test_ioi_checks(self):
        prof = [_attn(9, {6: 5.0, 9: 5.0, 2: 1.0}), _attn(10, {0: 4.0, 3: 1.0}),           # name movers at END
                _attn(7, {3: 2.0, 9: 2.0, 0: 0.1}), _attn(8, {6: 1.0, 10: 1.0, 2: 0.1}),  # S-inhibition at END
                _attn(5, {5: 1.0, 0: 0.1}, "s2"), _attn(6, {9: 1.0, 1: 1.0}, "s2"),          # induction at S2
                _attn(0, {1: 1.0}, "s2"), _attn(3, {0: 1.0, 4: 3.0}, "s2"),                   # duplicate-token
                _attn(2, {2: 1.0}, "s1_next"), _attn(4, {11: 0.1, 3: 1.0}, "s1_next")]       # previous-token
        ch = C.structure_checks("ioi", "ioi", prof)
        assert set(ch) == {"ioi_" + k for k in C.IOI_HEADS}
        assert ch["ioi_name_mover"]["held"] and ch["ioi_s_inhibition"]["held"] and ch["ioi_induction"]["held"]
        assert ch["ioi_duplicate_token"]["held"] is True and ch["ioi_previous_token"]["held"] is True
        prof[-1] = _attn(4, {11: 0.01, 3: 5.0}, "s1_next")
        prof[-2] = _attn(2, {2: 0.01, 0: 5.0}, "s1_next")
        assert C.structure_checks("ioi", "ioi", prof)["ioi_previous_token"]["held"] is False

    def test_greater_than_checks(self):
        prof = ([_attn(l, {h: 1.0}) for l, h in C.GT_HEADS["heads"]]
                + [dict(layer=l, kind="mlp", latent=l, alpha=1.0, act={C.ANCHOR: 1.0}, contrib={C.ANCHOR: 1.0})
                   for l in (8, 9, 10, 11, 2)]
                + [dict(layer=l, kind="resid", latent=l, alpha=1.0, act={"y1_yy": 1.0}, contrib={"y1_yy": 0.5})
                   for l in (0, 1, 1)])
        ch = C.structure_checks("greater_than", "greater_than", prof)
        assert ch["gt_heads"]["held"] and ch["gt_late_mlps"]["value"] == pytest.approx(0.8)
        assert ch["gt_late_mlps"]["held"] and ch["gt_layer_order"]["held"]

    def test_agreement_checks_are_secondary(self):
        prof = ([dict(layer=l, kind="mlp", latent=l, alpha=1.0, act={"subj": 1.0}, contrib={"subj": 1.0})
                 for l in (0, 1, 2)]
                + [dict(layer=l, kind="mlp", latent=l, alpha=1.0, act={C.ANCHOR: 1.0}, contrib={C.ANCHOR: 1.0})
                   for l in (7, 9, 11)])
        rc = C.structure_checks("sva", "sva_across_rc", prof)
        assert all(not c["primary"] for c in rc.values())
        checks = {"t": rc, "g": {"x": dict(held=False, primary=True)}}
        assert C.verdict_t4(checks)[0] == "NOT SUPPORTED"
        assert C.verdict_t4(checks, primary_only=False)[0] == "MIXED"


# ------------------------------------------------------------------------------------------- link and verdicts
class TestLinkAndVerdicts:
    def test_link(self):
        top = [(1, "mlp", 5), (2, "resid", 7)]
        task = {(1, "mlp", 5), (2, "resid", 7), (0, "attn", 1)}
        L = C.link_to_latent_circuits(top, task, {"1.mlp.5"}, {"1.mlp.5": {(0, "attn", 1), (0, "mlp", 3)}})
        assert L["n_top_targets"] == 1 and L["n_inter"] == 1 and L["task_in_union"] == pytest.approx(1 / 3)

    def test_verdicts(self):
        assert C.verdict_t1({"a": True, "b": False}) == "MIXED"
        r = {"a": dict(reach_ratio=5.0, u_fails_at_w_pass=True), "b": dict(reach_ratio=float("inf"),
                                                                          u_fails_at_w_pass=True)}
        assert C.verdict_t2(r)[0] == "SUPPORTED"
        assert C.verdict_t3([False] * 40, {"a": True})[0] == "SUPPORTED"
        assert C.verdict_t3([True] * 3 + [False] * 37, {"a": True})[0] == "NOT SUPPORTED"
        assert C.clopper_pearson_upper(0, 40) == pytest.approx(1 - 0.025 ** (1 / 40), abs=1e-6)
