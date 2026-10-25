"""Experiment 081: tokenizer helpers, the task schema, and the one GENERATED task (induction). Every other task is built
from the papers' original data by originals.py and blimp.py (2026-10-21; the from-memory template recreations of
2026-10-19 were removed).

Task dict (written by make_tasks.py to tasks/<model>/<name>.json):
  name, family, metric {task, obj}, logit_metric, answer_mode ("token" | "set" | "seq"), contrast ("original" |
  "generated"), cite, source (provenance), desc, expect, counts (kept / every drop reason / multi-token answers /
  held_dropped_by_cap), split {train, held} (row ids), split_meta {rule, n_train, n_held, n_held_uncapped, held_cap},
  rows (only the training rows and the KEPT held-out rows):
    id, group, prompt, tokens       the prompt's token ids (no BOS), read at its LAST token (the anchor); right-padded
                                    at load time (task_common.pad_right), never left-padded (047's padding trap)
    roles                           one role label per prompt token (structure checks)
    cf_prompt, cf_tokens, cf_roles  the dataset's own patch / counterfactual prompt (None for generated-contrast tasks)
    good, bad                       answer token SEQUENCES (multi-token answers are teacher-forced, summed log-probs)
                                    or token SETS (greater-than)
    meta
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import task_common as C

Parts = Sequence[Tuple[str, str]]
GPT2_TOKENIZER_DIR = C.EXTERNAL / "gpt2-tokenizer"          # copied from the local HF cache (see its SOURCE.md)


# ------------------------------------------------------------------------------------------- tokenizer helpers
class Tok:
    """The model's tokenizer with an encode cache. gpt2-small: HF "gpt2" (from data/external/gpt2-tokenizer when
    present, else the HF cache, offline); turingllm: the repo Tokenizer (Phi-3's vocabulary)."""

    def __init__(self, model: str = C.MODEL, base=None):
        self.model = model
        if base is None:
            if model == "gpt2-small":
                from transformers import AutoTokenizer
                src = str(GPT2_TOKENIZER_DIR) if GPT2_TOKENIZER_DIR.exists() else "gpt2"
                base = AutoTokenizer.from_pretrained(src, local_files_only=True)
            else:
                from model.tokenizer import Tokenizer
                base = Tokenizer()
        self.base = base
        self._cache: Dict[str, List[int]] = {}

    def encode(self, text: str) -> List[int]:
        v = self._cache.get(text)
        if v is None:
            v = list(self.base.encode(text))
            self._cache[text] = v
        return list(v)

    def decode(self, ids) -> str:
        return self.base.decode(list(ids))

    def _hf(self):
        return getattr(self.base, "tokenizer", self.base)

    def piece(self, i: int) -> str:
        tk = self._hf()
        if hasattr(tk, "convert_ids_to_tokens"):
            return tk.convert_ids_to_tokens(int(i))
        return self.decode([i])

    def vocab_size(self) -> int:
        tk = self._hf()
        return len(tk) if hasattr(tk, "__len__") else 32000


WORD_MARKS = ("▁", "Ġ")                          # SentencePiece "▁", GPT-2 byte-level "Ġ"


def starts_word(piece: str) -> bool:
    return piece.startswith(WORD_MARKS)


def encode_parts(tok: Tok, parts: Parts) -> Optional[Tuple[List[int], List[str], str]]:
    """Encode a prompt slot by slot: (ids, one role per token, text), or None when a slot changes the encoding of
    the text before it (not prefix-stable) or adds no token of its own."""
    ids: List[int] = []
    roles: List[str] = []
    text = ""
    for piece, role in parts:
        text += piece
        new = tok.encode(text)
        if new[:len(ids)] != ids or len(new) == len(ids):
            return None
        roles += [role] * (len(new) - len(ids))
        ids = new
    return ids, roles, text


def answer_ids(tok: Tok, text: str, answer: str) -> Optional[List[int]]:
    """The answer's token ids right after `text` (prefix-diff encoding; any length >= 1), or None when appending it
    re-tokenises the prompt."""
    a, b = tok.encode(text), tok.encode(text + answer)
    if b[:len(a)] != a or len(b) == len(a):
        return None
    return b[len(a):]


def single_token_after(tok: Tok, text: str, word: str) -> Optional[int]:
    ids = answer_ids(tok, text, word)
    return ids[0] if ids is not None and len(ids) == 1 else None


class Builder:
    """Collects rows with drop counters; finish() adds the split (the given one, the train / held sides rows were
    added to, or 75 / 25 by group), the held-out cap (task_common.cap_held) and the counts."""

    def __init__(self, name: str, n: Optional[int] = None):
        self.name, self.n = name, (None if n is None else int(n))
        self.rows: List[Dict] = []
        self.seen = set()
        self.drops: Dict[str, int] = {}
        self.tried = 0
        self.sides: Dict[str, List[int]] = {"train": [], "held": []}

    def drop(self, why: str):
        self.drops[why] = self.drops.get(why, 0) + 1

    def full(self) -> bool:
        return self.n is not None and len(self.rows) >= self.n

    def n_side(self, side: str) -> int:
        return len(self.sides[side])

    def add(self, key, group: str, clean, cf, good: Optional[List[int]], bad: Optional[List[int]], meta: Dict,
            match_len: bool = True, single_token: bool = False, side: Optional[str] = None) -> bool:
        """clean / cf: (ids, roles, text) or None (cf None is allowed when the task has no patch prompts: pass
        cf=False). good / bad: answer token lists (None = could not be tokenised after the prompt). side: "train" /
        "held" puts the row on that side of the split (generated tasks, SFC's official files)."""
        self.tried += 1
        if key in self.seen:
            self.drop("duplicate")
            return False
        if clean is None or cf is None:
            self.drop("not_prefix_stable")
            return False
        if not good or not bad:
            self.drop("answer_not_tokenisable")
            return False
        if single_token and (len(good) != 1 or len(bad) != 1):
            self.drop("answer_not_single_token")
            return False
        if good == bad or (len(good) == 1 and len(bad) == 1 and good[0] == bad[0]):
            self.drop("good_equals_bad")
            return False
        if cf is not False and match_len and len(clean[0]) != len(cf[0]):
            self.drop("cf_length_mismatch")
            return False
        self.seen.add(key)
        if side is not None:
            self.sides[side].append(len(self.rows))
        self.rows.append(dict(id=len(self.rows), group=group, prompt=clean[2], tokens=clean[0], roles=clean[1],
                              cf_prompt=cf[2] if cf else None, cf_tokens=cf[0] if cf else None,
                              cf_roles=cf[1] if cf else None, good=list(good), bad=list(bad), meta=meta))
        return True

    def finish(self, extra: Optional[Dict] = None, split: Optional[Tuple[List[int], List[int]]] = None,
               logit_metric: Optional[str] = None, split_rule: Optional[str] = None) -> Dict:
        """The task dict. Split: `split` if given, else the sides rows were added to (each side in a seeded order),
        else 75 / 25 by group (task_common.split_groups). Then the held-out side is capped (task_common.cap_held,
        HELDOUT_CAP) and the rows not kept are dropped from the file (ids renumbered); split_meta records the rule
        and the uncapped size."""
        spec = C.task_spec(self.name)
        if split is not None:
            tr, ho = list(split[0]), list(split[1])
            rule = split_rule or "explicit"
        elif self.sides["train"] or self.sides["held"]:
            tr, ho = list(self.sides["train"]), list(self.sides["held"])
            rr = random.Random("%s-order-%d" % (self.name, C.TASK_SEED))
            rr.shuffle(tr)
            rr.shuffle(ho)
            rule = split_rule or "train / held sides"
        else:
            tr, ho = C.split_groups([r["group"] for r in self.rows])
            rule = split_rule or "75 / 25 by group (seed %d)" % C.TASK_SEED
        ho_kept, n_unc = C.cap_held(ho, seed="%s-%d" % (self.name, C.TASK_SEED))
        rows = self.rows
        if len(ho_kept) < len(ho):
            keep = sorted(tr + ho_kept)
            new = {o: k for k, o in enumerate(keep)}
            rows = [dict(self.rows[o], id=new[o]) for o in keep]
            tr, ho_kept = [new[i] for i in tr], [new[i] for i in ho_kept]
        task = dict(name=self.name, family=spec["family"], logit_metric=logit_metric or spec["logit_metric"],
                    vac=spec["vac"], cite=C.CITE[spec["cite"]], desc=spec["desc"], expect=spec["expect"],
                    contrast=spec["contrast"], model=C.MODEL, n_requested=self.n,
                    counts=dict(kept=len(self.rows), tried=self.tried, held_dropped_by_cap=n_unc - len(ho_kept),
                                **{"dropped_" + k: v for k, v in sorted(self.drops.items())}),
                    n_groups=len({r["group"] for r in rows}), split=dict(train=tr, held=ho_kept),
                    split_meta=dict(rule=rule, n_train=len(tr), n_held=len(ho_kept), n_held_uncapped=n_unc,
                                    held_cap=C.HELDOUT_CAP or None),
                    seed=C.TASK_SEED, rows=rows)
        task.update(extra or {})
        finalize(task, spec)
        return task


def finalize(task: Dict, spec: Dict) -> Dict:
    """Answer mode and metric kinds from the rows (multi-token answers switch a token task to "seq"), and the
    multi-token counts."""
    mode = C.answer_mode(task)
    kinds = C.metric_kinds(mode, spec)
    task.update(answer_mode=mode, metric=dict(task=kinds["task"], obj=kinds["obj"]), logit_metric=kinds["logit_metric"])
    if mode == "seq":
        task["vac"] = dict(task=0.1, obj=0.1)
    task["counts"]["multi_token_good"] = sum(1 for r in task["rows"] if len(r["good"]) > 1) if mode != "set" else 0
    task["counts"]["multi_token_bad"] = sum(1 for r in task["rows"] if len(r["bad"]) > 1) if mode != "set" else 0
    return task


def gen_sides(b: Builder, make: Callable[[random.Random, str], None], n_train: int, n_held: int, seed,
              tries_per_item: int = 200):
    """Generated tasks: make(rng, side) tries to add one row on `side` from that side's lists; each side is filled
    to its count (training first) with its own seeded rng, so the held-out items do not depend on n_train."""
    for side, n in (("train", int(n_train)), ("held", int(n_held))):
        rng = random.Random("%s-%s" % (seed, side))
        tries = 0
        while b.n_side(side) < n and tries < tries_per_item * max(1, n):
            tries += 1
            make(rng, side)
        if b.n_side(side) < n:
            b.drop("short_%s_side" % side)


# ------------------------------------------------------------------------------------------- induction (generated)
IND_LEN = 16


def word_pool(tok: Tok, lo: int = 1000, hi: int = 20000) -> List[int]:
    """Whole-word vocabulary tokens (a word marker, then 3+ ASCII letters), so random sequences look like nonsense
    text rather than byte fragments."""
    out = []
    for i in range(lo, min(hi, tok.vocab_size())):
        p = tok.piece(i)
        if starts_word(p) and len(p) >= 4 and p[1:].isascii() and p[1:].isalpha():
            out.append(i)
    return out


def induction(tok: Tok, n_train: int = C.N_TRAIN_GEN, n_held: int = C.N_HELD_GEN, seed: int = C.TASK_SEED,
              L: int = IND_LEN) -> Dict:
    """Repeated random tokens (Olsson et al. 2022; generated, the standard construction, no download): R = L distinct
    whole-word tokens, prompt = R + R[:j+1] for a random j in [2, L-2], anchor = the last token; good = R[j+1] (what
    followed R[j] the first time), bad = R'[j+1]. Counterfactual: R' + R[:j+1] with a fresh R' (no earlier occurrence
    to copy; same length). Metric: logit(good) - logit(bad).
    Split (2026-10-21): the whole-word vocabulary pool is split into disjoint halves (task_common.split_list); training
    items draw R and R' from one half, held-out items from the other, so no held-out token occurs in training."""
    b = Builder("induction")
    pool = word_pool(tok)
    pools = dict(zip(("train", "held"), C.split_list(pool, tag="induction-tokens", seed=seed)))

    def make(rng, side):
        RR = rng.sample(pools[side], 2 * L)                  # R and R': 2L distinct tokens of this side's half
        R, R2 = RR[:L], RR[L:]
        j = rng.randint(2, L - 2)
        ids, cf = R + R[:j + 1], R2 + R[:j + 1]
        roles = ["first"] * L + ["second"] * (j + 1)
        roles[j] = "prev_occurrence"
        roles[j + 1] = "answer_source"
        b.add(tuple(ids), "row%d" % len(b.rows), (ids, roles, tok.decode(ids)),
              (cf, ["first"] * L + ["second"] * (j + 1), tok.decode(cf)), [R[j + 1]], [R2[j + 1]], dict(j=j, L=L),
              side=side)

    gen_sides(b, make, n_train, n_held, seed)
    return b.finish(dict(pool_size=len(pool), source=dict(generated=True),
                         lists_split=dict(word_pool=dict(train=len(pools["train"]), held=len(pools["held"])))),
                    split_rule="generated: %d training / %d held-out items; R and R' from disjoint halves of the "
                               "whole-word token pool" % (n_train, n_held))
