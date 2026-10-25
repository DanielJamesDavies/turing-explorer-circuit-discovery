"""Experiment 081 task tests (CPU; GPT-2's tokenizer from data/external/gpt2-tokenizer, the original datasets under
data/external; each test is skipped when its source is absent): the original-data loaders (lists parsed, not executed),
right-padding and anchors, answers tokenised right after the prompt, matched counterfactuals, reproducible splits,
SFC's official train / test split, disjoint train / held-out lists for the generated tasks, the held-out cap,
BLiMP's critical-token rule on GPT-2 pieces, the committed task files, and the probe's summary logic.

  PYTHONPATH=src python -m pytest experiments/081-task-circuits/test_081_tasks.py -q
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blimp as BL  # noqa: E402
import originals as O  # noqa: E402
import task_common as C  # noqa: E402
import tasks as T  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    try:
        return T.Tok("gpt2-small")
    except Exception as e:  # noqa: BLE001
        pytest.skip("GPT-2 tokenizer unavailable: %s" % e)


def _need(key):
    try:
        O.source_dir(key)
    except FileNotFoundError as e:
        pytest.skip(str(e))


def _answer_follows(tok, row, ids):
    """The answer is exactly the tokens that follow the prompt text (with a leading space)."""
    return tok.encode(row["prompt"] + tok.decode(ids)) == row["tokens"] + list(ids)


# ------------------------------------------------------------------------------------------- parsing as data
class TestParsing:
    def test_literal_assignments_never_executes(self):
        src = "import os\nos.system('echo hacked')\nNAMES = ['a', 'b']\nX = f(1)\nY = 3\n"
        out = O.literal_assignments(src, ["NAMES", "Y"])
        assert out == {"NAMES": ["a", "b"], "Y": 3}
        with pytest.raises(ValueError, match="X"):
            O.literal_assignments(src, ["X"])

    def test_abba_swap_matches_the_file(self):
        assert O.abba_from_baba(["Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]"]) == \
            ["Then, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to [A]"]


# ------------------------------------------------------------------------------------------- originals
def _sides(t):
    return [t["rows"][i] for i in t["split"]["train"]], [t["rows"][i] for i in t["split"]["held"]]


class TestOriginals:
    def test_ioi(self, tok):
        _need("ioi_easy_transformer")
        t = O.ioi(tok, n_train=40, n_held=30)
        assert len(t["rows"]) == 70 and t["answer_mode"] == "token" and t["contrast"] == "original"
        assert (len(t["split"]["train"]), len(t["split"]["held"])) == (40, 30)
        for r in t["rows"]:
            assert r["roles"].count("s2") == 1 and "io" in r["roles"] and "s1_next" in r["roles"]
            assert tok.decode(r["good"]).strip() == r["meta"]["io"] and tok.decode(r["bad"]).strip() == r["meta"]["s"]
            assert len(r["cf_tokens"]) == len(r["tokens"])
            assert r["meta"]["io"] not in r["cf_prompt"] and r["meta"]["s"] not in r["cf_prompt"]   # ABC
            assert _answer_follows(tok, r, r["good"])

    def test_ioi_train_and_held_lists_are_disjoint(self, tok):
        """MIB-style: no name (IO, S or an ABC name), place or object of a held-out item occurs in training."""
        _need("ioi_easy_transformer")
        t = O.ioi(tok, n_train=200, n_held=200)
        tr, ho = _sides(t)
        names = lambda rs: {n for r in rs for n in (r["meta"]["io"], r["meta"]["s"], *r["meta"]["abc"])}  # noqa: E731
        assert names(tr) and names(ho) and not names(tr) & names(ho)
        d = O.source_dir("ioi_easy_transformer")
        L = O.literal_assignments((d / "ioi_dataset.py").read_text(encoding="utf8"), ["PLACES", "OBJECTS"])
        for key in ("PLACES", "OBJECTS"):
            used = lambda rs: {w for r in rs for w in L[key] if (" " + w) in r["prompt"]}             # noqa: E731
            assert used(tr) and used(ho) and not used(tr) & used(ho), key
        assert t["lists_split"]["names"] == dict(train=49, held=50) and "disjoint" in t["split_meta"]["rule"]

    def test_greater_than(self, tok):
        _need("greater_than_hanna")
        t = O.greater_than(tok, n_train=40, n_held=20)
        assert t["answer_mode"] == "set" and t["metric"]["task"] == "set_probdiff"
        two = [tok.encode("%02d" % v)[0] for v in range(100)]
        for r in t["rows"]:
            yy = r["meta"]["yy"]
            assert r["good"] == two[yy + 1:] and r["bad"] == two[:yy + 1] and 2 <= yy <= 98
            assert r["cf_prompt"] == r["prompt"].replace(str(r["meta"]["year"]), str(r["meta"]["year"])[:2] + "01")
            assert r["roles"][-1] == "y2_xx" and r["roles"].count("y1_yy") == 1
        assert [r["meta"]["yy"] for r in t["rows"][:3]] == [2, 3, 4]                # balanced YY cycling
        tr, ho = _sides(t)
        assert not {r["meta"]["noun"] for r in tr} & {r["meta"]["noun"] for r in ho}   # nouns disjoint
        assert sorted(r["meta"]["yy"] for r in ho) == list(range(2, 22))             # YY cycles on each side

    def test_greater_than_years_follow_hannas_rule(self, tok):
        ys = O.valid_years(tok)
        assert all(1100 <= y < 1800 for y in ys) and 1700 not in ys and 1732 in ys

    def test_sva_uses_the_official_split(self, tok):
        """SFC_SPLIT=official (default): training items are exactly the train file's distinct items, held-out items
        the test file's; the overlap between the files is counted, not removed."""
        _need("sfc_feature_circuits")
        d = O.source_dir("sfc_feature_circuits")
        for name in ("sva_simple", "sva_across_pp"):
            stem = O.SFC_FILES[name]
            t = O.sva(tok, name, split_mode="official")
            assert t["contrast"] == "original" and t["sfc_split"] == "official", name
            tr, ho = _sides(t)
            want = {}
            for side, f in (("train", "train"), ("held", "test")):
                recs = O._read_jsonl(d / "data" / ("%s_%s.json" % (stem, f)))
                want[side] = {(r["clean_prefix"], r["clean_answer"]) for r in recs}
            got = lambda rs: {(r["prompt"], r["meta"]["clean_answer"]) for r in rs}                # noqa: E731
            assert got(tr) == want["train"] and got(ho) == want["held"], name
            assert all(r["meta"]["sfc_file"] == "%s_train.json" % stem for r in tr)
            assert all(r["meta"]["sfc_file"] == "%s_test.json" % stem for r in ho)
            assert t["overlap"]["held_item_in_train"] == len(want["held"] & want["train"]), name
            for r in t["rows"][:10]:
                assert "subj" in r["roles"] and _answer_follows(tok, r, r["good"]), (name, r["prompt"])
        assert O.SFC_SPLIT == "official" or __import__("os").environ.get("SFC_SPLIT")

    def test_sva_prefix_split_is_prompt_disjoint(self, tok):
        _need("sfc_feature_circuits")
        t = O.sva(tok, "sva_simple", split_mode="prefix")
        tr, ho = _sides(t)
        assert ho and not {r["prompt"] for r in tr} & {r["prompt"] for r in ho} and t["overlap"] is None
        with pytest.raises(ValueError, match="SFC_SPLIT"):
            O.sva(tok, "sva_simple", split_mode="random")

    def test_gender(self, tok):
        _need("mathwin_gendered_pronoun")
        t = O.gender_pronoun(tok)
        assert len(t["rows"]) <= 100 and len(t["rows"]) >= 90                       # the original 5 x 20
        for r in t["rows"]:
            assert r["cf_prompt"].startswith("That person") and len(r["cf_tokens"]) == len(r["tokens"])
            assert tok.decode(r["good"]) == (" he" if r["meta"]["male"] else " she")

    def test_known_facts(self, tok):
        _need("rome_known_1000")
        t = O.factual_known(tok)
        assert t["contrast"] == "generated" and all(r["cf_tokens"] is None for r in t["rows"])
        assert len(t["rows"]) >= 1200                                             # all 1,209 facts but unusable ones
        tr, ho = _sides(t)
        assert len(ho) == pytest.approx(0.25 * len(t["rows"]), abs=15)
        assert not {r["group"] for r in tr} & {r["group"] for r in ho}            # by subject
        for r in t["rows"]:
            assert r["good"] != r["bad"] and _answer_follows(tok, r, r["good"])

    def test_induction(self, tok):
        t = T.induction(tok, n_train=20, n_held=10)
        assert (len(t["split"]["train"]), len(t["split"]["held"])) == (20, 10)
        for r in t["rows"]:
            j, L = r["meta"]["j"], r["meta"]["L"]
            assert len(set(r["tokens"][:L])) == L and r["tokens"][L:] == r["tokens"][:j + 1]
            assert r["good"] == [r["tokens"][j + 1]] and r["bad"] == [r["cf_tokens"][j + 1]]
            assert r["cf_tokens"][L:] == r["tokens"][L:]
        tr, ho = _sides(t)
        toks = lambda rs: {x for r in rs for x in r["tokens"] + r["cf_tokens"]}               # noqa: E731
        assert not toks(tr) & toks(ho)                                            # disjoint vocabulary halves

    def test_reproducible(self, tok):
        _need("ioi_easy_transformer")
        assert O.ioi(tok, n_train=20, n_held=10) == O.ioi(tok, n_train=20, n_held=10)
        a, b = O.ioi(tok, n_train=20, n_held=10), O.ioi(tok, n_train=30, n_held=10)
        assert {r["prompt"] for r in _sides(a)[1]} == {r["prompt"] for r in _sides(b)[1]}  # held-out items stay put

    def test_held_cap_drops_rows_and_records_the_uncapped_size(self, tok, monkeypatch):
        _need("ioi_easy_transformer")
        monkeypatch.setattr(C, "HELDOUT_CAP", 12)
        t = O.ioi(tok, n_train=20, n_held=30)
        assert len(t["split"]["held"]) == 12 and len(t["rows"]) == 32
        assert t["split_meta"]["n_held_uncapped"] == 30 and t["split_meta"]["held_cap"] == 12
        assert t["counts"]["held_dropped_by_cap"] == 18
        assert sorted(t["split"]["train"] + t["split"]["held"]) == list(range(32))
        assert [r["id"] for r in t["rows"]] == list(range(32))
        full = O.ioi(tok, n_train=20, n_held=30)                                  # the cap only removes held rows
        assert {r["prompt"] for r in _sides(t)[1]} <= {r["prompt"] for r in _sides(full)[1]}
        assert [r["prompt"] for r in _sides(t)[0]] == [r["prompt"] for r in _sides(full)[0]]


# ------------------------------------------------------------------------------------------- BLiMP under GPT-2
class TestBlimpGPT2:
    def test_critical_token_on_gpt2_pieces(self, tok):
        c, why = BL.critical(tok, "Katherine can't help herself.", "Katherine can't help himself.")
        assert why == "ok" and tok.decode(c["prefix"]) == "Katherine can't help"
        assert tok.decode([c["good"]]) == " herself" and tok.decode([c["bad"]]) == " himself"
        assert BL.critical(tok, "Dogs bark.", "Cats bark.")[1] == "differs_at_first_token"

    def test_qualify_respects_one_prefix_flag(self, tok):
        recs = [dict(sentence_good="Kate helps herself.", sentence_bad="Kate helps himself.", pair_id=0,
                     one_prefix_method=True, linguistics_term="t", field="f"),
                dict(sentence_good="Kate sold these cars.", sentence_bad="Kate sold this cars.", pair_id=1,
                     one_prefix_method=False, linguistics_term="t", field="f")]
        q = BL.qualify(tok, recs)
        assert q["n_qual"] == 1 and q["reasons"]["not_one_prefix_paradigm"] == 1

    def test_selection(self):
        s = {u: dict(qualifies=True, n_qual=1000 - i, n_prefixes=500, term=t) for i, (u, t) in enumerate(
            [("a1", "A"), ("a2", "A"), ("a3", "A"), ("b1", "B"), ("c1", "C")])}
        s["g1"] = dict(qualifies=True, n_qual=900, n_prefixes=2, term="G")
        assert BL.select(s, per_term=2, k=4) == ["a1", "a2", "b1", "c1"]

    def test_missing_data_refuses_with_the_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="snapshot_download"):
            BL.blimp_dir(tmp_path)


# ------------------------------------------------------------------------------------------- committed files
class TestCommittedTasks:
    def test_files_are_consistent(self):
        files = C.task_files(C.task_dir("gpt2-small"))
        if not files:
            pytest.skip("no task files (run make_tasks.py)")
        for p in files:
            t = json.loads(p.read_text())
            n = len(t["rows"])
            tr, ho = t["split"]["train"], t["split"]["held"]
            sm = t["split_meta"]
            assert t["name"] == p.stem and t["model"] == "gpt2-small"
            assert sorted(tr + ho) == list(range(n)) and ho, t["name"]
            assert t["answer_mode"] == C.answer_mode(t), t["name"]
            if t["contrast"] == "original":
                assert all(r["cf_tokens"] for r in t["rows"]), t["name"]
            assert (sm["n_train"], sm["n_held"]) == (len(tr), len(ho)) and sm["n_held_uncapped"] >= len(ho)
            assert sm["held_cap"] is None or len(ho) <= sm["held_cap"], t["name"]
            assert len(ho) == sm["n_held_uncapped"] or len(ho) == sm["held_cap"], t["name"]
            if t["name"] in ("ioi", "greater_than", "induction"):
                assert (len(tr), sm["n_held_uncapped"]) == (C.N_TRAIN_GEN, C.N_HELD_GEN), t["name"]
            if t["family"] == "sva":
                assert t["sfc_split"] == "official" and all(
                    t["rows"][i]["meta"]["sfc_file"].endswith("_test.json") for i in ho), t["name"]

    def test_ioi_file_regenerates_identically(self, tok):
        _need("ioi_easy_transformer")
        p = C.task_dir("gpt2-small") / "ioi.json"
        if not p.exists():
            pytest.skip("no task files")
        t = json.loads(p.read_text())
        assert json.loads(json.dumps(O.ioi(tok))) == t


# ------------------------------------------------------------------------------------------- probe summary
class TestProbeSummary:
    def _task(self, contrast):
        return dict(name="x", family="sva", expect=True, metric=dict(task="logit_diff", obj="logit_diff"),
                    contrast=contrast, answer_mode="token", split=dict(train=list(range(6)), held=list(range(6, 10))))

    def test_original_contrast_uses_the_cf_gap(self):
        import probe as P
        clean = [-1.0] * 6 + [2.0, 1.0, 1.5, -0.2]
        v = dict(values=clean, correct=[c > 0 for c in clean])
        vals = dict(clean={C.PRIMARY: v, C.OBJ: v, "top1": [False] * 6 + [True, True, False, False]},
                    cf={C.PRIMARY: dict(values=[0.0] * 10), C.OBJ: dict(values=[0.0] * 10)})
        s = P.summarise(self._task("original"), vals)
        assert s["competent"] and s["held"]["frac_correct"] == 0.75 and s["held"]["top1"] == 0.5
        vals["cf"][C.PRIMARY]["values"] = [0.0] * 6 + [3.0] * 4
        assert not P.summarise(self._task("original"), vals)["competent"]

    def test_generated_contrast_has_no_cf_criterion(self):
        import probe as P
        clean = [-1.0] * 6 + [2.0, 1.0, 1.5, -0.2]
        v = dict(values=clean, correct=[c > 0 for c in clean])
        s = P.summarise(self._task("generated"), dict(clean={C.PRIMARY: v, C.OBJ: v, "top1": None}, cf=None))
        assert s["competent"] and s["held"]["crit_cf"] is None and s["held"]["top1"] is None
