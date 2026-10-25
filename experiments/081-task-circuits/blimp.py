"""Experiment 081: BLiMP minimal pairs as output-endpoint tasks (CPU; reads data/external/blimp, never downloads).

BLiMP (Warstadt et al. 2020; HF nyu-mll/blimp @ 877fba0, CC BY 4.0; data/external/blimp/SOURCE.md).

Single critical token (decided on the MODEL's tokenizer, here GPT-2's). A pair qualifies when
  - the two token sequences share an identical, non-empty prefix,
  - the first differing token on each side is a whole word on its own: it starts a word and the next token starts a
    new word, is punctuation, or is the end (anything after it may differ),
  - and BLiMP's own one_prefix_method flag is set (its authors' judgement that the prefix decides the critical word;
    without it, e.g. determiner_noun_agreement_2's "could distract these / this companies", the determiner is
    compared before the noun it agrees with).
prompt = the shared prefix, anchor = its last token, good / bad = the two critical tokens, metric logit(good) -
logit(bad). A subset qualifies when >= 80% of its pairs do (QUAL_MIN, env BLIMP_QUAL_MIN).

Contrast (2026-10-21, Daniel): BLiMP pairs share their prefix, so there is no patch prompt; the C ablation's mean is
the GENERATED close, silent contrast pool (prepare.py, task_common.build_pool), built from the subset's other
prefixes. (The 2026-10-19 "answer-flipped prefix" counterfactual is retired.)

Selection (cap): qualifying subsets with >= MIN_ROWS qualifying pairs and >= MIN_PREFIXES distinct prefixes, ranked by
qualifying pairs, at most PER_TERM per linguistics_term (phenomenon), the top MAX_SUBSETS; every qualifying pair of
each (2026-10-21), split 75 / 25 by prefix. The competence probe then decides which are fitted.
"""
from __future__ import annotations

import hashlib
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import task_common as C
from tasks import Builder, starts_word

BLIMP_DIR = C.EXTERNAL / "blimp"
BLIMP_SOURCE = dict(repo="nyu-mll/blimp", revision="877fba0801ffb7cbd8c39c1ff314a46f053f6036", licence="CC BY 4.0",
                    url="https://huggingface.co/datasets/nyu-mll/blimp", size="67 parquet files, 3,895,588 bytes",
                    downloaded="2026-10-19", source_md="data/external/blimp/SOURCE.md")
BLIMP_COMMAND = ('python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\'nyu-mll/blimp\', '
                 'repo_type=\'dataset\', revision=\'%s\', local_dir=\'data/external/blimp\')"' % BLIMP_SOURCE["revision"])
QUAL_MIN = float(os.environ.get("BLIMP_QUAL_MIN", 0.80))
PER_TERM = 2
MAX_SUBSETS = 8
N_ROWS = None                      # every qualifying pair (2026-10-21: full datasets); an int subsamples
MIN_ROWS = 160
MIN_PREFIXES = 40
PUNCT = set(".,;:!?'\")")


# ------------------------------------------------------------------------------------------- reading
def blimp_dir(path: Optional[Path] = None) -> Path:
    p = Path(path) if path is not None else BLIMP_DIR
    if not p.exists() or not any(p.glob("*/*.parquet")):
        raise FileNotFoundError(
            "BLiMP is not at %s. Source: %s (revision %s, %s, %s). Download command (run from the repo root):\n  %s"
            % (p, BLIMP_SOURCE["url"], BLIMP_SOURCE["revision"], BLIMP_SOURCE["licence"], BLIMP_SOURCE["size"],
               BLIMP_COMMAND))
    return p


def subsets(path: Optional[Path] = None) -> List[str]:
    return sorted(q.parent.name for q in blimp_dir(path).glob("*/*.parquet"))


def read_subset(uid: str, path: Optional[Path] = None) -> List[Dict]:
    import pandas as pd
    files = sorted((blimp_dir(path) / uid).glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no parquet for BLiMP subset %r under %s" % (uid, blimp_dir(path)))
    return pd.concat([pd.read_parquet(f) for f in files]).to_dict("records")


def checksum(path: Optional[Path] = None) -> str:
    """sha256 over "<relative path> <sha256>" lines of every parquet file, sorted by path (deterministic)."""
    root = blimp_dir(path)
    lines = []
    for f in sorted(root.glob("*/*.parquet")):
        lines.append("%s %s" % (f.relative_to(root).as_posix(), hashlib.sha256(f.read_bytes()).hexdigest()))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ------------------------------------------------------------------------------------------- qualification
def _boundary(tok, ids: Sequence[int], i: int) -> bool:
    """Position i+1 starts a new word (or is punctuation, or the end): ids[i] completes a word."""
    if i + 1 >= len(ids):
        return True
    p = tok.piece(ids[i + 1])
    return starts_word(p) or (len(p) > 0 and all(ch in PUNCT for ch in p))


def critical(tok, good: str, bad: str) -> Tuple[Optional[Dict], str]:
    """The single-critical-token reading of one minimal pair: (dict(prefix, good, bad), "ok") or (None, reason)."""
    g, b = tok.encode(good), tok.encode(bad)
    i = 0
    while i < min(len(g), len(b)) and g[i] == b[i]:
        i += 1
    if i == min(len(g), len(b)):
        return None, "no_differing_token"
    if i == 0:
        return None, "differs_at_first_token"
    if not (starts_word(tok.piece(g[i])) and starts_word(tok.piece(b[i]))):
        return None, "critical_not_word_initial"
    if not (_boundary(tok, g, i) and _boundary(tok, b, i)):
        return None, "critical_word_multi_token"
    return dict(prefix=g[:i], good=g[i], bad=b[i]), "ok"


def qualify(tok, records: Sequence[Mapping]) -> Dict:
    """Every pair's critical-token reading and the subset's qualifying share."""
    rows, reasons = [], defaultdict(int)
    for r in records:
        if r.get("one_prefix_method") is False or r.get("one_prefix_method") == 0:
            reasons["not_one_prefix_paradigm"] += 1
            continue
        c, why = critical(tok, r["sentence_good"], r["sentence_bad"])
        reasons[why] += 1
        if c is not None:
            c.update(pair_id=int(r.get("pair_id", len(rows))), sentence_good=r["sentence_good"],
                     sentence_bad=r["sentence_bad"])
            rows.append(c)
    n = len(records)
    return dict(n=n, n_qual=len(rows), share=len(rows) / n if n else 0.0, rows=rows, reasons=dict(reasons),
                term=(records[0].get("linguistics_term") if records else None),
                field=(records[0].get("field") if records else None),
                n_prefixes=len({tuple(r["prefix"]) for r in rows}))


# ------------------------------------------------------------------------------------------- tasks
def subset_task(tok, uid: str, q: Mapping, n_rows: Optional[int] = N_ROWS, seed: int = C.TASK_SEED) -> Dict:
    """One BLiMP subset as a task: every qualifying pair (n_rows=None; an int takes a seeded sample), split 75 / 25 by
    prefix (identical prefixes stay on one side; BLiMP has no official split); no patch prompts (contrast =
    "generated")."""
    rows = q["rows"]
    rng = random.Random("%s-%d" % (uid, seed))
    n_take = len(rows) if n_rows is None else min(int(n_rows), len(rows))
    pick = sorted(rng.sample(range(len(rows)), n_take))
    name = C.BLIMP_PREFIX + uid
    b = Builder(name, n_rows)
    for i in pick:
        r = rows[i]
        text = tok.decode(r["prefix"])
        b.add(r["pair_id"], text, (list(r["prefix"]), ["other"] * len(r["prefix"]), text), False,
              [int(r["good"])], [int(r["bad"])],
              dict(pair_id=r["pair_id"], sentence_good=r["sentence_good"], sentence_bad=r["sentence_bad"],
                   critical_good=tok.decode([r["good"]]), critical_bad=tok.decode([r["bad"]])))
    t = b.finish(dict(source=BLIMP_SOURCE, uid=uid, term=q["term"], field=q["field"]),
                 split_rule="all qualifying pairs, 75 / 25 by prefix (no official split), seed %d" % seed)
    t["counts"].update(pairs=q["n"], qualifying=q["n_qual"], qual_share=q["share"],
                       **{"not_qualifying_" + k: v for k, v in q["reasons"].items() if k != "ok"})
    return t


def survey(tok, path: Optional[Path] = None) -> Dict[str, Dict]:
    """Qualification of every BLiMP subset under this tokenizer: {uid: summary} (no rows)."""
    out = {}
    for uid in subsets(path):
        q = qualify(tok, read_subset(uid, path))
        out[uid] = dict(n=q["n"], n_qual=q["n_qual"], share=q["share"], term=q["term"], field=q["field"],
                        qualifies=q["share"] >= QUAL_MIN, n_prefixes=q["n_prefixes"], reasons=q["reasons"])
    return out


def select(summary: Mapping[str, Mapping], per_term: int = PER_TERM, k: int = MAX_SUBSETS,
           min_rows: int = MIN_ROWS, min_prefixes: int = MIN_PREFIXES) -> List[str]:
    """Qualifying subsets with >= min_rows qualifying pairs and >= min_prefixes distinct prefixes, ranked by
    qualifying pairs (ties: name), at most per_term per phenomenon, the top k."""
    cand = sorted((uid for uid, s in summary.items() if s["qualifies"] and s["n_qual"] >= min_rows
                   and s.get("n_prefixes", min_prefixes) >= min_prefixes),
                  key=lambda u: (-summary[u]["n_qual"], u))
    out, per = [], defaultdict(int)
    for uid in cand:
        t = summary[uid]["term"]
        if per[t] < per_term:
            out.append(uid)
            per[t] += 1
        if len(out) >= k:
            break
    return out


__all__ = ["blimp_dir", "subsets", "read_subset", "checksum", "critical", "qualify", "subset_task", "survey", "select"]
