"""Experiment 081: tasks built from the papers' ORIGINAL data (2026-10-21), read from data/external/<source>/ (each with
a SOURCE.md: URL, commit, licence, date, sha256). Never downloads. Downloaded code is read as DATA: list literals are
extracted with ast.literal_eval from the parsed source, nothing is imported or executed.

  ioi              Wang et al. 2022, redwoodresearch/Easy-Transformer easy_transformer/ioi_dataset.py @ ea15315: NAMES,
                   BABA_TEMPLATES (ABBA derived by the file's own first-clause swap), PLACES, OBJECTS; "mixed" prompts
                   (half BABA, half ABBA); C = the ABC distribution of the IOI notebook (gen_flipped_prompts IO->RAND,
                   S->RAND, S1->RAND: three fresh names); generated, 1,600 training / 5,000 held-out items from
                   disjoint halves of NAMES, PLACES and OBJECTS
  greater_than     Hanna et al. 2023, hannamw/gpt2-greater-than dataset.py @ 0e6dc8c (sentences, the "01" corruption,
                   the validity rule of utils.get_valid_years); nouns from the ACDC repo's copy of their noun list
                   (ArthurConmy/Automatic-Circuit-Discovery acdc/greaterthan/utils.py @ bc99ace: NOUNS), because the
                   greater-than repository reads its nouns from a file it does not include; years 1100-1799 (ACDC's
                   centuries 11-17), YY balanced over 02..98 as YearDataset(balanced=True); generated, 1,600 / 5,000
                   items, nouns from disjoint halves, years shared
  sva_*            Marks et al. 2025, saprmarks/feature-circuits data/{simple,within_rc,rc,nounpp}_{train,test}.json @
                   7fbd82b: clean_prefix / patch_prefix / clean_answer / patch_answer; every record; the files' own
                   train / test split (SFC_SPLIT=official, default) or a prompt-disjoint 75 / 25 (SFC_SPLIT=prefix)
  gender_pronoun   Mathwin et al. 2023, the hackathon notebook Automatic_Circuits_Script.ipynb (itch.io, cmathw):
                   its 5 templates x 10 male + 10 female names (100 prompts: the original dataset is that small), C =
                   its baseline "That person is ..." applied to each template; 75 / 25 by prompt
  factual_known    Meng et al. 2022, ROME known_1000.json (rome.baulab.info): prompt -> the true object; bad = the
                   object of another fact with the same relation; multi-token answers (teacher-forced); no patch
                   prompts, so C is the generated contrast set; all facts, 75 / 25 by subject
Every held-out set is then capped at task_common.HELDOUT_CAP (5,000; env HELDOUT_CAP, 0 = none) by fixed-seed sampling.
"""
from __future__ import annotations

import ast
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import task_common as C
from tasks import Builder, Tok, answer_ids, encode_parts, gen_sides

SRC = {
    "ioi_easy_transformer": dict(dir="ioi_easy_transformer", files=("ioi_dataset.py",),
                                 repo="redwoodresearch/Easy-Transformer",
                                 commit="ea15315dd24481e9e2ac5c3ef335d82907a1dc34", licence="MIT"),
    "greater_than_hanna": dict(dir="greater_than_hanna", files=("dataset.py", "utils.py", "acdc_greaterthan_utils.py"),
                               repo="hannamw/gpt2-greater-than @ 0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a; nouns: "
                                    "ArthurConmy/Automatic-Circuit-Discovery @ bc99ace817974b5584b7ee203d596a8e2bbcd399",
                               commit="0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a",
                               licence="hannamw: no licence file; ACDC: MIT"),
    "sfc_feature_circuits": dict(dir="sfc_feature_circuits", files=("data/simple_train.json",),
                                 repo="saprmarks/feature-circuits", commit="7fbd82b895ae16294f4e6fc7bfc675d1d680d659",
                                 licence="MIT"),
    "mathwin_gendered_pronoun": dict(dir="mathwin_gendered_pronoun", files=("Automatic_Circuits_Script.ipynb",),
                                     repo="cmathw.itch.io (hackathon upload 7208110)", commit=None,
                                     licence="none stated"),
    "rome_known_1000": dict(dir="rome_known_1000", files=("known_1000.json",), repo="kmeng01/rome (data host "
                            "rome.baulab.info)", commit="0874014cd9837e4365f3e6f3c71400ef11509e04", licence="MIT"),
}


def source_dir(key: str):
    d = C.EXTERNAL / SRC[key]["dir"]
    missing = [f for f in SRC[key]["files"] if not (d / f).exists()]
    if missing:
        raise FileNotFoundError("%s: missing %s under %s; see that folder's SOURCE.md (or the README) for the exact "
                                "URL and commit to fetch" % (key, missing, d))
    return d


def provenance(key: str) -> Dict:
    d = C.EXTERNAL / SRC[key]["dir"]
    p = dict(SRC[key])
    # as_posix: the same string on Windows and in WSL, so committed task files regenerate identically on both
    p["source_md"] = (d / "SOURCE.md").relative_to(C.ROOT).as_posix() if (d / "SOURCE.md").exists() else None
    return p


def literal_assignments(source: str, names: Sequence[str]) -> Dict[str, object]:
    """Top-level `NAME = <literal>` assignments of a Python source, by parsing (never executing) it."""
    tree = ast.parse(source)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names and t.id not in out:
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    missing = [n for n in names if n not in out]
    if missing:
        raise ValueError("literal assignments not found: %s" % missing)
    return out


# ------------------------------------------------------------------------------------------- IOI
def abba_from_baba(templates: Sequence[str]) -> List[str]:
    """Easy-Transformer's own construction: in the first clause, the first [B] becomes [A] and the following [A]
    becomes [B] (character-level, as in ioi_dataset.py)."""
    out = []
    for t in templates:
        t = list(t)
        first = True
        for j in range(1, len(t) - 1):
            if "".join(t[j - 1:j + 2]) == "[B]" and first:
                t[j] = "A"
            elif "".join(t[j - 1:j + 2]) == "[A]" and first:
                first = False
                t[j] = "B"
        out.append("".join(t))
    return out


def ioi(tok: Tok, n_train: int = C.N_TRAIN_GEN, n_held: int = C.N_HELD_GEN, seed: int = C.TASK_SEED) -> Dict:
    """IOI prompts from Wang et al.'s templates and lists; good = " IO", bad = " S" (single tokens, as their dataset
    assumes); C = ABC (IO, S1 and S2 replaced by three fresh names, all distinct, same token length).
    Split (2026-10-21, as MIB's private test set): NAMES (99), PLACES (8) and OBJECTS (8) are each split into disjoint
    halves (task_common.split_list); training items (and their ABC names) use one half, held-out items the other. The
    15 templates (and their ABBA swaps) are the task itself and are shared."""
    d = source_dir("ioi_easy_transformer")
    L = literal_assignments((d / "ioi_dataset.py").read_text(encoding="utf8"),
                            ["NAMES", "BABA_TEMPLATES", "PLACES", "OBJECTS"])
    baba = list(L["BABA_TEMPLATES"])
    abba = abba_from_baba(baba)
    lists = {}
    for key in ("NAMES", "PLACES", "OBJECTS"):
        tr_l, ho_l = C.split_list(list(L[key]), tag="ioi-" + key.lower(), seed=seed)
        lists[key] = dict(train=tr_l, held=ho_l)
    b = Builder("ioi")

    def prompt_template(t):
        assert t.endswith(" [A]"), t
        return t[:-len(" [A]")]

    def parts(t, io, s1, s2, place, obj):
        # first occurrence of [B] (or [A]) in the first clause is S1 / IO; the second [B] is S2
        out, rest, seen_b = [], prompt_template(t), 0
        vals = []
        while "[" in rest:
            pre, after = rest.split("[", 1)
            field, rest = after.split("]", 1)
            if field == "A":
                v, r = io, "io"
            elif field == "B":
                v, r = (s1, "s1") if seen_b == 0 else (s2, "s2")
                seen_b += 1
            elif field == "PLACE":
                v, r = place, "other"
            else:
                v, r = obj, "other"
            vals.append((pre, v, r))
        for pre, v, r in vals:
            if pre.endswith(" "):
                pre, v = pre[:-1], " " + v
            if pre:
                out.append((pre, "other"))
            out.append((v, r))
        if rest:
            out.append((rest, "other"))
        return out

    def make(rng, side):
        names, places, objects = (lists[k][side] for k in ("NAMES", "PLACES", "OBJECTS"))
        abba_flag = rng.random() < 0.5
        k = rng.randrange(len(baba))
        t = abba[k] if abba_flag else baba[k]
        io, s = rng.sample(names, 2)
        place, obj = rng.choice(places), rng.choice(objects)
        x, y, z = rng.sample([m for m in names if m not in (io, s)], 3)
        clean = encode_parts(tok, parts(t, io, s, s, place, obj))
        cf = encode_parts(tok, parts(t, x, z, y, place, obj))          # IO->X, S1->Z, S2->Y (ABC)
        if clean is not None:
            nxt = [i + 1 for i, r in enumerate(clean[1]) if r == "s1"][-1:]
            for i in nxt:
                if i < len(clean[1]) and clean[1][i] == "other":
                    clean[1][i] = "s1_next"
        good = answer_ids(tok, clean[2], " " + io) if clean else None
        bad = answer_ids(tok, clean[2], " " + s) if clean else None
        b.add((io, s, place, obj, k, abba_flag), clean[2] if clean else str((io, s, k)), clean, cf, good, bad,
              dict(io=io, s=s, abc=(x, z, y), order="ABBA" if abba_flag else "BABA", template=k), single_token=True,
              side=side)

    gen_sides(b, make, n_train, n_held, seed)
    return b.finish(dict(source=provenance("ioi_easy_transformer"),
                         lists_split={k.lower(): dict(train=len(v["train"]), held=len(v["held"]))
                                      for k, v in lists.items()},
                         notes="mixed BABA / ABBA (Wang et al.'s 15 BABA templates and the file's ABBA swap); "
                               "C = ABC (the notebook's IO->RAND, S->RAND, S1->RAND flips), prepend_bos False"),
                    split_rule="generated: %d training / %d held-out items; names, places and objects from disjoint "
                               "halves of Wang et al.'s lists; templates shared" % (n_train, n_held))


# ------------------------------------------------------------------------------------------- greater-than
def valid_years(tok: Tok, start: int = 1100, end: int = 1800) -> List[int]:
    """Hanna et al.'s utils.get_valid_years rule: " YYYY" must tokenise to [" XX", "YY"] with a two-character second
    piece; the first and the last valid year of every century are dropped."""
    valid = []
    for year in range(start, end):
        ids = tok.encode(" %d" % year)
        pieces = [tok.piece(i) for i in ids]
        valid.append(len(ids) == 2 and len(pieces[1]) == 2)
    last, cur = None, None
    years = list(range(start, end))
    for i, y in enumerate(years):
        if valid[i]:
            if cur != y // 100:
                cur = y // 100
                valid[i] = False
                if last is not None:
                    valid[last] = False
            last = i
    if last is not None:
        valid[last] = False
    return [y for y, v in zip(years, valid) if v]


def greater_than(tok: Tok, n_train: int = C.N_TRAIN_GEN, n_held: int = C.N_HELD_GEN, seed: int = C.TASK_SEED) -> Dict:
    """Hanna et al.'s sentences, "The {noun} lasted from the year {XXYY} to the year {XX}" -> the next token, a
    two-digit YY' token. good = {YY' > YY}, bad = {YY' <= YY} (their probability difference P(> YY) - P(<= YY)); C =
    their corruption, start year XX01. Years balanced over YY = 02..98 (YearDataset balanced=True), cycling per side.
    Split (2026-10-21): the 120 nouns are split into disjoint halves (task_common.split_list), training items use one,
    held-out items the other. The years are SHARED: YY is the variable the task compares and must stay balanced over
    02..98 on both sides (each valid YY has only ~6 valid years across centuries 11-17)."""
    d = source_dir("greater_than_hanna")
    nouns_all = list(literal_assignments((d / "acdc_greaterthan_utils.py").read_text(encoding="utf8"),
                                         ["NOUNS"])["NOUNS"])
    nouns = dict(zip(("train", "held"), C.split_list(nouns_all, tag="greater_than-nouns", seed=seed)))
    years = valid_years(tok)
    by_yy: Dict[int, List[int]] = {}
    for y in years:
        by_yy.setdefault(y % 100, []).append(y)
    b = Builder("greater_than")
    # the two-digit answer tokens as Hanna et al. / ACDC define them: the stand-alone encoding of "%02d" (all 100 are
    # single GPT-2 tokens). Appending "00" to the text would re-tokenise " 17" + "00" as " 1700", so the prefix-diff
    # encoding is the wrong tool here.
    digits = [tok.encode("%02d" % v) for v in range(100)]
    if any(len(dd) != 1 for dd in digits):
        digits = None

    def parts(noun, year_xx, yy):
        return [("The", "other"), (" " + noun, "noun"), (" lasted from the year", "other"), (" " + year_xx, "y1_xx"),
                (yy, "y1_yy"), (" to the year", "mid"), (" " + year_xx, "y2_xx")]

    cur = {"train": 2, "held": 2}

    def make(rng, side):
        pool = by_yy.get(cur[side], [])
        cur[side] = cur[side] + 1 if cur[side] + 1 < 99 else 2
        if not pool:
            b.drop("no_valid_year_for_yy")
            return
        year = rng.choice(pool)
        noun = rng.choice(nouns[side])
        xx, yy = str(year)[:2], str(year)[2:]
        clean = encode_parts(tok, parts(noun, xx, yy))
        cf = encode_parts(tok, parts(noun, xx, "01"))
        y = int(yy)
        good = [dd[0] for dd in digits[y + 1:]] if digits else None
        bad = [dd[0] for dd in digits[:y + 1]] if digits else None
        b.add((noun, year), clean[2] if clean else str((noun, year)), clean, cf, good, bad,
              dict(noun=noun, year=year, yy=y), side=side)

    gen_sides(b, make, n_train, n_held, seed)
    return b.finish(dict(source=provenance("greater_than_hanna"),
                         lists_split=dict(nouns=dict(train=len(nouns["train"]), held=len(nouns["held"])),
                                          years="shared (%d valid years)" % len(years)),
                         notes="good = YY' > YY, bad = YY' <= YY (Hanna's probability difference); years 1100-1799 "
                               "under Hanna's validity rule; nouns from ACDC's copy of Hanna's list"),
                    logit_metric="set",
                    split_rule="generated: %d training / %d held-out items; nouns from disjoint halves of Hanna's "
                               "list; years shared (YY balanced on both sides)" % (n_train, n_held))


# ------------------------------------------------------------------------------------------- subject-verb agreement
SFC_FILES = {"sva_simple": "simple", "sva_within_rc": "within_rc", "sva_across_rc": "rc", "sva_across_pp": "nounpp"}
SFC_ROLES = {"simple": ["other", "subj"], "nounpp": ["other", "subj", "other", "other", "distractor"],
             "within_rc": ["other", "distractor", "other", "other", "subj"],
             "rc": ["other", "subj", "other", "other", "distractor", "rc_verb"]}


def _read_jsonl(path) -> List[Dict]:
    out = []
    for line in open(path, encoding="utf8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _word_parts(prefix: str, roles: Sequence[str]) -> List[Tuple[str, str]]:
    words = prefix.split(" ")
    if len(words) != len(roles):
        roles = ["other"] * len(words)
    return [((" " if i else "") + w, r) for i, (w, r) in enumerate(zip(words, roles))]


SFC_SPLITS = ("official", "prefix")
SFC_SPLIT = os.environ.get("SFC_SPLIT", "official")


def sva(tok: Tok, name: str, seed: int = C.TASK_SEED, split_mode: str = None) -> Dict:
    """SFC agreement data: prompt = clean_prefix, good = clean_answer, bad = patch_answer, C = patch_prefix (the
    opposite-number agreeing noun). Every record of both files is read; identical items (same prefix and answer) are
    kept once per file.

    split_mode (SFC_SPLIT, default "official", Daniel 2026-10-21):
      official  the files' own split: training items = *_train.json, held-out items = *_test.json (then the held-out
                cap). SFC's files repeat items heavily and the test file is NOT disjoint from the train file (every
                test prefix of simple / within_rc / rc occurs in its train file, and most test items are identical
                to a training item); both overlaps are counted in the task file (overlap) and reported, not removed.
      prefix    both files pooled, split 75 / 25 by prefix (task_common.split_groups): prompt-disjoint."""
    split_mode = split_mode or SFC_SPLIT
    if split_mode not in SFC_SPLITS:
        raise ValueError("SFC_SPLIT must be one of %s, got %r" % (SFC_SPLITS, split_mode))
    d = source_dir("sfc_feature_circuits")
    stem = SFC_FILES[name]
    roles = SFC_ROLES[stem]
    files = {"train": _read_jsonl(d / "data" / ("%s_train.json" % stem)),
             "held": _read_jsonl(d / "data" / ("%s_test.json" % stem))}
    b = Builder(name)
    for side in ("train", "held"):
        for r in files[side]:
            key = (r["clean_prefix"], r["clean_answer"])
            if split_mode == "official":
                key = (side,) + key                           # identical items are deduplicated within a file only
            clean = encode_parts(tok, _word_parts(r["clean_prefix"], roles))
            cf = encode_parts(tok, _word_parts(r["patch_prefix"], roles))
            good = answer_ids(tok, r["clean_prefix"], r["clean_answer"])
            bad = answer_ids(tok, r["clean_prefix"], r["patch_answer"])
            b.add(key, r["clean_prefix"], clean, cf, good, bad,
                  dict(case=r.get("case"), clean_answer=r["clean_answer"], patch_answer=r["patch_answer"],
                       sfc_file="%s_%s.json" % (stem, "train" if side == "train" else "test")),
                  match_len=False, side=side if split_mode == "official" else None)
    tr_ids, te_ids = b.sides["train"], b.sides["held"]
    tr_prefix = {b.rows[i]["prompt"] for i in tr_ids}
    tr_items = {(b.rows[i]["prompt"], b.rows[i]["meta"]["clean_answer"]) for i in tr_ids}
    overlap = dict(file_records=dict(train=len(files["train"]), test=len(files["held"])),
                   held_prefix_in_train=sum(1 for i in te_ids if b.rows[i]["prompt"] in tr_prefix),
                   held_item_in_train=sum(1 for i in te_ids
                                          if (b.rows[i]["prompt"], b.rows[i]["meta"]["clean_answer"]) in tr_items))
    if split_mode == "official":
        how = ("official: training items = %s_train.json, held-out items = %s_test.json (identical items kept once "
               "per file; the test file overlaps the train file, see overlap)" % (stem, stem))
    else:
        how = "prefix: %s_train.json + %s_test.json pooled, identical items kept once, 75 / 25 by prefix" % (stem, stem)
    return b.finish(dict(source=provenance("sfc_feature_circuits"), sfc_file=stem, sfc_split=split_mode,
                         overlap=overlap if split_mode == "official" else None, notes=how), split_rule=how)


# ------------------------------------------------------------------------------------------- gendered pronoun
def mathwin_lists(nb_path) -> Dict[str, object]:
    """templates, male_names, female_names and the baseline prompt from the notebook's code cells (parsed, not run)."""
    nb = json.loads(open(nb_path, encoding="utf8").read())
    found: Dict[str, object] = {}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in ("templates", "male_names", "female_names"):
                        try:
                            found[t.id] = ast.literal_eval(node.value)
                        except ValueError:
                            pass
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("That person"):
                found.setdefault("baseline", node.value)
    missing = [k for k in ("templates", "male_names", "female_names", "baseline") if k not in found]
    if missing:
        raise ValueError("Mathwin notebook: %s not found" % missing)
    return found


def gender_pronoun(tok: Tok, seed: int = C.TASK_SEED) -> Dict:
    """Mathwin et al.'s 5 templates x 20 names (their whole dataset: 100 prompts); good / bad = " he" / " she" by the
    name's gender (their answers / wrongs); C = their baseline, the name slot "So {name}" -> "That person", applied to
    every template (their notebook builds it for the first template). Rows whose name changes the token length
    relative to the baseline are dropped (the author's own note: Carol, Karen, Julie and Judy tokenise in two)."""
    d = source_dir("mathwin_gendered_pronoun")
    L = mathwin_lists(d / "Automatic_Circuits_Script.ipynb")
    b = Builder("gender_pronoun")
    for male, names in ((True, L["male_names"]), (False, L["female_names"])):
        for name in names:
            for k, t in enumerate(L["templates"]):
                pre, post = t.split("{name}")
                clean = encode_parts(tok, [(pre.rstrip(" "), "other"), (" " + name, "name"), (post, "other")])
                base = t.replace("So {name}", "That person")
                cf = encode_parts(tok, [(base, "other")])
                if cf is not None and clean is not None and len(cf[0]) == len(clean[0]):
                    cf = (cf[0], list(clean[1]), cf[2])
                he = answer_ids(tok, clean[2], " he") if clean else None
                she = answer_ids(tok, clean[2], " she") if clean else None
                good, bad = (he, she) if male else (she, he)
                b.add((name, k), clean[2] if clean else name, clean, cf, good, bad, dict(name=name, male=male,
                                                                                        template=k))
    return b.finish(dict(source=provenance("mathwin_gendered_pronoun"),
                         notes="the original 100-prompt dataset (5 templates x 20 names); baseline 'That person' "
                               "generalised to all 5 templates; prepend_bos False (theirs used a BOS)"),
                    split_rule="all items, 75 / 25 by prompt (no official split), seed %d" % C.TASK_SEED)


# ------------------------------------------------------------------------------------------- factual recall
def factual_known(tok: Tok, seed: int = C.TASK_SEED) -> Dict:
    """ROME known_1000 (all 1,209 facts): prompt -> " " + attribute (the true object; often several tokens: teacher-
    forced, summed); bad = the attribute of another fact of the same relation (seeded choice among different attribute
    strings). Grouped by subject, 75 / 25 by subject (no official split). No patch prompts: C is the generated
    contrast set (prepare.py)."""
    d = source_dir("rome_known_1000")
    recs = json.loads((d / "known_1000.json").read_text(encoding="utf8"))
    rng = random.Random("factual_known-%d" % seed)
    by_rel: Dict[str, List[str]] = {}
    for r in recs:
        by_rel.setdefault(r["relation_id"], [])
        if r["attribute"] not in by_rel[r["relation_id"]]:
            by_rel[r["relation_id"]].append(r["attribute"])
    order = rng.sample(recs, len(recs))
    b = Builder("factual_known")
    for r in order:
        others = sorted(a for a in by_rel[r["relation_id"]] if a != r["attribute"])
        if not others:
            b.drop("no_other_object_for_relation")
            continue
        wrong = rng.choice(others)
        clean = encode_parts(tok, [(r["prompt"], "other")])
        if clean is not None:
            subj_ids = tok.encode(r["subject"])
            roles = list(clean[1])
            for i in range(len(clean[0])):                          # mark the subject's tokens when it is a prefix
                if i < len(subj_ids) and r["prompt"].startswith(r["subject"]) and clean[0][i] == subj_ids[i]:
                    roles[i] = "subject"
            clean = (clean[0], roles, clean[2])
        good = answer_ids(tok, r["prompt"], " " + r["attribute"])
        bad = answer_ids(tok, r["prompt"], " " + wrong)
        b.add(r["known_id"], r["subject"], clean, False, good, bad,
              dict(known_id=r["known_id"], relation=r["relation_id"], attribute=r["attribute"], wrong=wrong))
    return b.finish(dict(source=provenance("rome_known_1000"),
                         notes="bad = another fact's object of the same relation; C = generated close-silent pool"),
                    split_rule="all facts, 75 / 25 by subject (no official split), seed %d" % C.TASK_SEED)


GENERATORS = {
    "ioi": ioi,
    "greater_than": greater_than,
    "sva_simple": lambda tok, **k: sva(tok, "sva_simple", **k),
    "sva_within_rc": lambda tok, **k: sva(tok, "sva_within_rc", **k),
    "sva_across_rc": lambda tok, **k: sva(tok, "sva_across_rc", **k),
    "sva_across_pp": lambda tok, **k: sva(tok, "sva_across_pp", **k),
    "gender_pronoun": gender_pronoun,
    "factual_known": factual_known,
}
