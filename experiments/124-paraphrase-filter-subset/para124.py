"""Shared pieces of 124 (DAN-217, Daniel's 2026-11-16 design): the paraphrase rule, the corpus reader, the subset and
paths. Reuses 123 (common123: paths, jsonl helpers, plan / mid / check logic) and 114 (the TF-IDF vectoriser itself).

THE RULE (TuringLLM only; added on top of the DAN-157 rule, which 115 build_one applies unchanged):
  a held-out row is dropped when it is a paraphrase / near-duplicate of ANY of that target's activating training rows
  (strong.train 48 + mid.train), i.e. when  max_t cos(h, t) >= 0.3  under 114's TF-IDF:
    features   hashed token unigrams + bigrams, 2^20 buckets (114 check.feats, imported, not re-implemented)
    weighting  sublinear tf (1 + log tf), smooth idf log((1+n)/(1+df)) + 1, L2-normalised rows (114 check.Tfidf)
    idf        fitted on the SAME 100,000 corpus windows as 114: np.random.default_rng(20261114).integers(1, n_win+1,
               100000) as the FIRST draw of that generator, n_win = 24,820,215 (114 check.main). Cached in
               out/idf114.npz; `rebuild124.py STAGE=idf` rebuilds it and verifies it against 114's out/rows.jsonl.
    threshold  0.3 (114: recall 75% on known paraphrase pairs, 0.02% false positives against another target's rows)
  Same-file / same-topic rows that are not near-identical are ALLOWED (Daniel: document grouping is overkill).
Refill as 115 / 123: candidates = the original held rows (pool order) then the T1 top-512 scan rows strongest first;
the paraphrase test is applied to every candidate BEFORE the 086 rule (firing, exact / anchor-prefix / near), fill to
the original size, floor 8 reported. Mid-band: the same test, drop only (common123.drop_mid). Contrast: measured only.

How the rule plugs into 115 build_one / common123 (which take a "group_of(id) -> set" interface): ParaGroups.of(id)
returns {"P"} for every activating training id of the target and for every row whose max cos to those training rows is
>= 0.3, else the empty set. The forbidden set is then {"P"}, so "shares a group with a training row" means exactly
"is a paraphrase of a training row" (a training row's twin has cos 1.0 and is caught too).
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiments"
sys.path.insert(0, str(EXP / "123-docgroup-heldout-rescore"))
import common123 as C  # noqa: E402

WORK = Path(os.environ.get("WORK124", str(HERE / "out")))
THRESH = float(os.environ.get("PARA_THRESH", "0.3"))
IDF_SEED, IDF_N, N_WIN_114 = 20261114, 100000, 24820215
SUBSET_SEED, SUBSET_N = 20261116, 300
GROUP = "para"
RULE124 = ("124-paraphrase/2026-11-16 (086 DAN-157 rule + 114 TF-IDF cos >= %.2f vs activating training rows; "
           "idf = 114's 100k-window sample)" % THRESH)
MARK = "P"


# ------------------------------------------------------------------------------------------------ 114 vectoriser
def load_114():
    spec = importlib.util.spec_from_file_location("check114", EXP / "114-paraphrase-check" / "check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


K114 = load_114()


def idf_sample_ids(n_win=N_WIN_114, seed=IDF_SEED, n=IDF_N):
    """114's IDF sample: the first draw of default_rng(20261114), sorted (114 check.main, line 134)."""
    return np.sort(np.random.default_rng(seed).integers(1, n_win + 1, size=n))


def tfidf_from_idf(idf, n):
    tf = K114.Tfidf.__new__(K114.Tfidf)
    tf.idf, tf.n = np.asarray(idf, dtype=np.float64), int(n)
    return tf


def idf_path():
    return WORK / "idf114.npz"


def load_tfidf(path=None):
    p = Path(path or idf_path())
    if not p.exists():
        raise SystemExit("no IDF cache at %s: run `python -X utf8 experiments/124-paraphrase-filter-subset/"
                         "rebuild124.py` with STAGE=idf first" % p)
    z = np.load(p)
    return tfidf_from_idf(z["idf"], int(z["n"]))


def ids_md5(ids):
    return hashlib.md5(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ corpus reader
class Corpus:
    """Windows by corpus id, read exactly as 114 check.windows does (data/shard_<k>.npy + its .shard_indices offsets)
    but with seek + read, so it runs on Windows too (114 used os.pread, POSIX only)."""

    def __init__(self, data_dir=None):
        self.data = Path(data_dir or os.environ.get("TL_DATA", str(REPO / "data")))
        self._idx, self._hdr = {}, {}

    def _header(self, shard):
        h = self._hdr.get(shard)
        if h is None:
            with open(self.data / ("shard_%d.npy" % shard), "rb") as f:
                major, _ = np.lib.format.read_magic(f)
                _, _, dt = (np.lib.format.read_array_header_1_0(f) if major == 1
                            else np.lib.format.read_array_header_2_0(f))
                h = self._hdr[shard] = (f.tell(), dt)
        return h

    def windows(self, ids):
        by = {}
        for i in ids:
            by.setdefault((int(i) - 1) // 8192, []).append(int(i))
        out = {}
        for shard, lst in by.items():
            idx = self._idx.get(shard)
            if idx is None:
                idx = self._idx[shard] = np.load(self.data / ".shard_indices" / ("shard_%d.npy_sft1.idx.npy" % shard))
            base, dt = self._header(shard)
            with open(self.data / ("shard_%d.npy" % shard), "rb") as f:
                for i in sorted(set(lst)):
                    st, en = (int(x) for x in idx[(i - 1) % 8192])
                    f.seek(base + st * dt.itemsize)
                    out[i] = np.frombuffer(f.read((en - st) * dt.itemsize), dtype=dt).astype(np.int64)
        return np.stack([out[int(i)] for i in ids]) if len(ids) else np.zeros((0, 64), np.int64)


# ------------------------------------------------------------------------------------------------ the rule
def train_rows(rec):
    """The target's activating training rows, as 114 check.train_side builds them: strong.train then mid.train."""
    s, m = rec["strong"], rec.get("mid")
    rows = [np.asarray(s["pos"][s["train"]])]
    if m is not None and m.get("train"):
        rows.append(np.asarray(m["pos"][m["train"]]))
    return np.concatenate(rows).astype(np.int64)


class ParaRule:
    """max TF-IDF cosine of rows against one fixed reference set (114's cT = vH @ vT.T, max over the reference)."""

    def __init__(self, tf, ref_rows, thresh=THRESH):
        self.tf, self.thresh = tf, thresh
        self.vT = tf(np.asarray(ref_rows, dtype=np.int64)) if len(ref_rows) else None

    def cos_max(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            return np.zeros(0, np.float32)
        if self.vT is None:
            return np.zeros(len(rows), np.float32)
        return (self.tf(rows) @ self.vT.T).toarray().max(axis=1)

    def flag(self, rows):
        return self.cos_max(rows) >= self.thresh


def pool_tokens(rec):
    """id -> the token row the record stores for it (strong pool, then mid pool)."""
    out = {}
    for pool in ("mid", "strong"):                         # strong last: wins for an id in both pools
        p = rec.get(pool)
        if p is not None:
            for i, row in zip(p["ids"], np.asarray(p["pos"])):
                out[int(i)] = row
    return out


class ParaGroups:
    """The 115 / 123 group interface for the paraphrase rule, for ONE target: of(id, _) -> {"P"} for a training id or a
    paraphrase of a training row (cos >= THRESH), else set(). Token rows come from the record's pools when the id is
    there (what the scorer sees), else from the corpus (scan candidates). cos per id is cached (also `cos` for the
    provenance)."""

    def __init__(self, rec, tf, corpus=None, thresh=THRESH):
        self.rule = ParaRule(tf, train_rows(rec), thresh)
        self.train_ids = set(C.training_ids(rec))
        self.tok = pool_tokens(rec)
        self.corpus = corpus
        self.cos = {}

    def prefetch(self, ids):
        need = [int(i) for i in dict.fromkeys(int(x) for x in ids) if int(i) > 0 and int(i) not in self.cos
                and int(i) not in self.train_ids]
        if not need:
            return
        inpool = [i for i in need if i in self.tok]
        rest = [i for i in need if i not in self.tok]
        rows = [self.tok[i] for i in inpool]
        if rest:
            if self.corpus is None:
                raise ValueError("ids outside the record need a Corpus: %s" % rest[:5])
            rows += list(self.corpus.windows(rest))
        cm = self.rule.cos_max(np.stack(rows))
        for i, c in zip(inpool + rest, cm):
            self.cos[i] = float(c)

    def of(self, i, group=GROUP):
        i = int(i)
        if i in self.train_ids:
            return {MARK}
        if i <= 0:
            return set()
        if i not in self.cos:
            self.prefetch([i])
        return {MARK} if self.cos[i] >= self.rule.thresh else set()

    def clear(self):
        pass


FORBIDDEN = {MARK}


def contrast_para(rec, tf, thresh=THRESH):
    """Measured only: contrast held rows (rec["neg"][n_tr:], the scorers' 48 / 16 split) that are paraphrases of the
    contrast-TRAINING rows (rec["neg"][:n_tr]) or of the activating training rows."""
    neg = np.asarray(rec["neg"]).astype(np.int64)
    n = len(neg)
    if n == 0:
        return dict(contrast_held=0)
    n_tr = max(1, n - int(round(n * 0.25)))
    held = neg[n_tr:]
    c_ct = ParaRule(tf, neg[:n_tr], thresh).cos_max(held)
    c_at = ParaRule(tf, train_rows(rec), thresh).cos_max(held)
    return dict(contrast_held=int(len(held)), contrast_para_vs_contrast_train=int((c_ct >= thresh).sum()),
                contrast_para_vs_activating_train=int((c_at >= thresh).sum()))


def plan_one(rec, scan_ids, G, tf):
    """The CPU half of one target (the paraphrase test alone; no firing, no 086 duplicate rule): common123.tl_plan with
    the paraphrase interface, plus the mid-band and contrast measures and the cos distribution of the old held rows."""
    s, m = rec["strong"], rec.get("mid")
    G.prefetch([int(s["ids"][j]) for j in s["held"]] + [int(x) for x in scan_ids] +
               ([int(m["ids"][j]) for j in m["held"]] if m is not None else []))
    pl = C.tl_plan(rec, scan_ids, G.of)
    lk = C.leak_counts(dict(rec, neg_ids=None), G.of, FORBIDDEN)          # strong / mid only; contrast below
    n_want = len(s["held"])
    allowed = pl["counts"]["orig_allowed"] + pl["counts"]["scan_allowed"]
    old_cos = [G.cos.get(int(s["ids"][j]), 1.0) for j in s["held"]]
    row = dict(n_want=n_want, n_old_leaky=n_want - pl["counts"]["orig_allowed"], cand_allowed=allowed,
               cand_total=pl["counts"]["orig_total"] + pl["counts"]["scan_total"],
               short_certain=allowed < n_want, below_floor_certain=allowed < C.FLOOR, unscorable_certain=allowed == 0,
               old_cos_med=float(np.median(old_cos)) if old_cos else None,
               old_cos_max=float(np.max(old_cos)) if old_cos else None, **pl["counts"], **lk)
    row.update(contrast_para(rec, tf))
    return row


# ------------------------------------------------------------------------------------------------ checks
def training_unchanged(new, old):
    """123 rebuild.check's byte-identity test of every training-side field (strong.train, D_train, B_mid, neg,
    neg_ids, the old strong pool rows and ids, the mid pool and its split)."""
    import torch
    s, o = new["strong"], old["strong"]
    n0 = int(o["pos"].shape[0])
    return bool(s["train"] == o["train"] and new["D_train"] == old["D_train"]
                and new.get("B_mid") == old.get("B_mid") and torch.equal(new["neg"], old["neg"])
                and new.get("neg_ids") == old.get("neg_ids") and torch.equal(s["pos"][:n0], o["pos"])
                and list(s["ids"][:n0]) == list(o["ids"])
                and (old.get("mid") is None or (torch.equal(new["mid"]["pos"], old["mid"]["pos"])
                                                and new["mid"]["train"] == old["mid"]["train"])))


# ------------------------------------------------------------------------------------------------ subset
def stratified_subset(df, n=SUBSET_N, seed=SUBSET_SEED):
    """Proportional allocation (largest remainder) of n targets over band x kind cells, then a seeded draw without
    replacement inside each cell (cells sorted by seed key first, so the draw depends only on the key set and seed).
    df: index seed, columns band, kind. Returns (sorted keys, allocation dict)."""
    cells = sorted(df.groupby(["band", "kind"]).groups.items(), key=lambda kv: kv[0])
    tot = sum(len(v) for _, v in cells)
    quota = [(k, n * len(v) / tot) for k, v in cells]
    alloc = {k: int(np.floor(q)) for k, q in quota}
    rest = n - sum(alloc.values())
    for k, _ in sorted(quota, key=lambda kq: (-(kq[1] - np.floor(kq[1])), kq[0]))[:rest]:
        alloc[k] += 1
    rng = np.random.default_rng(seed)
    pick = []
    for k, v in cells:
        keys = sorted(str(x) for x in v)
        pick += [keys[i] for i in sorted(rng.choice(len(keys), size=min(alloc[k], len(keys)), replace=False))]
    return sorted(pick), {"%s/%s" % k: alloc[k] for k, _ in cells}


def subset_path():
    return Path(os.environ.get("SUBSET", str(HERE / "targets_subset.txt")))


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
