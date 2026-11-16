"""113 paraphrase check: do TuringLLM's held-out strongest contexts paraphrase a training context of the same target?

Read-only on data/, the 085 corpus index and the production (T2) records. Writes only experiments/113-paraphrase-check/out/.

Document structure (data/text_paths.txt; corpus_src.npy holds the 0-based line index of each window's source file):
  text/phi-3-mini/dataset/<rel>          Phi-3 Mini original
  text/data-augmentation/dataset/<rel>   the T5 paraphrases of that same original (same <rel>, ~11x the windows)
So   doc group   = <rel>                       (original + its paraphrases)
     topic group = dirname(<rel>)              (the 6 desc_k.txt files Phi-3 wrote for one leaf topic)

Measures, per held-out strongest row h of a target, against that target's 96 training-side rows (48 strongest-train +
48 mid-band train):
  cos   TF-IDF cosine on token unigrams + bigrams (hashed, IDF from 200k random corpus windows)
  ng8   share of h's 8-grams present in the training row (the 086 near-dup measure)
Baselines: the same max over 96 rows of (a) another random target's training rows and (b) 96 random corpus windows.
Calibration: a window of an original vs. every window of its paraphrase file (known paraphrase) and vs. as many windows
of an unrelated paraphrase file (known non-paraphrase).

  .venv/bin/python -I experiments/113-paraphrase-check/check.py [N_TARGETS=400]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CTX = (REPO / "experiments/109-pod-run-analysis/out/archives/pub-turingllm-T2-records/root/turing-tl/experiments/"
       "082-public-runs/out_turingllm/ctx_fresh")
SRC = REPO / "experiments/085-heldout-audit/out/corpus_src.npy"
OUT = HERE / "out"
NF = 1 << 20
RNG = np.random.default_rng(20261114)


# ---------------------------------------------------------------- corpus access
def window(i):
    return windows([int(i)])[0]


_IDX = {}


def windows(ids):
    """Windows by corpus id with exact preads (no mmap readahead: data/ sits on a slow mount)."""
    by = {}
    for i in ids:
        by.setdefault((int(i) - 1) // 8192, []).append(int(i))
    out = {}
    for shard, lst in by.items():
        idx = _IDX.get(shard)
        if idx is None:
            idx = _IDX[shard] = np.load(REPO / "data" / ".shard_indices" / ("shard_%d.npy_sft1.idx.npy" % shard))
        with open(REPO / "data" / ("shard_%d.npy" % shard), "rb") as f:
            major, _ = np.lib.format.read_magic(f)
            shape, fortran, dt = (np.lib.format.read_array_header_1_0(f) if major == 1
                                  else np.lib.format.read_array_header_2_0(f))
            base = f.tell()
            fd = f.fileno()
            for i in sorted(set(lst)):
                st, en = (int(x) for x in idx[(i - 1) % 8192])
                b = os.pread(fd, (en - st) * dt.itemsize, base + st * dt.itemsize)
                out[i] = np.frombuffer(b, dtype=dt).astype(np.int64)
    return np.stack([out[int(i)] for i in ids])


# ---------------------------------------------------------------- features
def feats(rows):
    """rows [n, 64] int -> sparse counts over hashed unigrams + bigrams."""
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    uni = rows % NF
    bi = (rows[:, :-1] * 1000003 + rows[:, 1:] + 7919) % NF
    cols = np.concatenate([uni, bi], axis=1)
    r = np.repeat(np.arange(n), cols.shape[1])
    m = sp.csr_matrix((np.ones(r.size, np.float32), (r, cols.ravel())), shape=(n, NF))
    m.sum_duplicates()
    return m


class Tfidf:
    def __init__(self, sample_rows):
        df = np.zeros(NF, np.float64)
        for b in range(0, len(sample_rows), 20000):
            m = feats(sample_rows[b:b + 20000])
            m.data[:] = 1
            df += np.asarray(m.sum(axis=0)).ravel()
        self.n = len(sample_rows)
        self.idf = np.log((1 + self.n) / (1 + df)) + 1

    def __call__(self, rows):
        m = feats(rows)
        m.data = 1 + np.log(m.data)
        m = m @ sp.diags(self.idf.astype(np.float32))
        nrm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()) + 1e-12
        return sp.diags(1 / nrm) @ m


def ng8(h, t):
    """share of h's 8-grams that occur in t (h, t 1-D)."""
    def grams(x):
        return {tuple(x[i:i + 8]) for i in range(len(x) - 7)}
    g = grams(h)
    return len(g & grams(t)) / max(1, len(g))


# ---------------------------------------------------------------- groups
def groups():
    paths = [p for p in (REPO / "data/text_paths.txt").read_text().split("\n") if p]
    rel = [p.split("/dataset/", 1)[1] for p in paths]
    aug = np.array([p.startswith("text/data-augmentation/") for p in paths])
    doc_key = {r: k for k, r in enumerate(sorted(set(rel)))}
    top_key = {}
    doc = np.array([doc_key[r] for r in rel])
    topic = np.array([top_key.setdefault(os.path.dirname(r), len(top_key)) for r in rel])
    return paths, rel, aug, doc, topic


def main():
    n_targets = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    OUT.mkdir(parents=True, exist_ok=True)
    src = np.load(SRC, mmap_mode="r")
    paths, rel, aug, doc, topic = groups()
    n_win = len(src) - 1
    print("corpus windows %d; files %d (%d originals, %d paraphrase files)" % (n_win, len(paths), (~aug).sum(), aug.sum()))
    src_all = np.asarray(src[1:])
    print("windows from paraphrase files: %.1f%%" % (100 * aug[src_all].mean()))

    # IDF sample
    samp_ids = np.sort(RNG.integers(1, n_win + 1, size=100000))
    samp = windows(samp_ids)
    tf = Tfidf(samp)
    pool = samp[RNG.permutation(len(samp))[:20000]]          # random-window baseline pool
    print("tfidf fitted", flush=True)

    # ------------------------------------------------ calibration on known paraphrase files
    cal = []
    orig_idx = np.flatnonzero(~aug)
    has_aug = {rel[i]: i for i in np.flatnonzero(aug)}
    by_src = None
    cand = [i for i in RNG.permutation(orig_idx) if rel[i] in has_aug][:200]
    order = np.argsort(src_all, kind="stable")
    ss = src_all[order]

    def ids_of(s):
        lo, hi = np.searchsorted(ss, s), np.searchsorted(ss, s, side="right")
        return order[lo:hi] + 1

    aug_files = np.flatnonzero(aug)
    for o in cand:
        oi, ai = ids_of(o), ids_of(has_aug[rel[o]])
        if len(oi) == 0 or len(ai) == 0:
            continue
        other = RNG.choice(aug_files)
        while doc[other] == doc[o]:
            other = RNG.choice(aug_files)
        xi = ids_of(other)
        q = windows([RNG.choice(oi)])
        ai = RNG.permutation(ai)[:300]
        xi = RNG.permutation(xi)[:len(ai)]                     # same number of candidates on both sides
        A, X = windows(ai), windows(xi)
        vq, vA, vX = tf(q), tf(A), tf(X)
        cA, cX = (vq @ vA.T).toarray().ravel(), (vq @ vX.T).toarray().ravel()
        jA = int(cA.argmax())
        cal.append(dict(orig=int(o), para=int(has_aug[rel[o]]), n_para=len(ai), cos_para=float(cA.max()),
                        ng8_para=float(ng8(q[0], A[jA])), cos_unrel=float(cX.max()), n_unrel=len(xi)))
    with open(OUT / "calibration.jsonl", "w") as f:
        for c in cal:
            f.write(json.dumps(c) + "\n")
    print("calibration pairs %d" % len(cal), flush=True)

    # ------------------------------------------------ production records
    files = sorted(CTX.glob("*.pt"))
    pick = RNG.choice(len(files), size=min(n_targets, len(files)), replace=False)
    recs = []
    for k in pick:
        d = torch.load(files[k], weights_only=False)
        key, rec = next(iter(d.items()))
        if not isinstance(rec, dict) or not rec.get("strong") or not rec["strong"].get("held"):
            continue
        recs.append((key, rec))
    print("records %d" % len(recs))

    def train_side(rec):
        s, m = rec["strong"], rec.get("mid")
        rows = [s["pos"][s["train"]].numpy()]
        ids = [np.asarray(s["ids"])[s["train"]]]
        if m is not None and m.get("train"):
            rows.append(m["pos"][m["train"]].numpy())
            ids.append(np.asarray(m["ids"])[m["train"]])
        return np.concatenate(rows), np.concatenate(ids)

    rows_out = []
    for t, (key, rec) in enumerate(recs):
        s = rec["strong"]
        H = s["pos"][s["held"]].numpy()
        hid = np.asarray(s["ids"])[s["held"]]
        Tr, tid = train_side(rec)
        _, other = recs[(t + 1 + RNG.integers(len(recs) - 1)) % len(recs)]
        Tx, _ = train_side(other)
        R = pool[RNG.integers(len(pool), size=len(Tr))]
        vH, vT, vX, vR = tf(H), tf(Tr), tf(Tx), tf(R)
        cT, cX, cR = (vH @ vT.T).toarray(), (vH @ vX.T).toarray(), (vH @ vR.T).toarray()
        tdoc, ttop, tsrc = set(doc[src_all[tid - 1]]), set(topic[src_all[tid - 1]]), set(src_all[tid - 1])
        for r in range(len(H)):
            j = int(cT[r].argmax())
            hs = int(src_all[hid[r] - 1])
            js = int(src_all[tid[j] - 1])
            rows_out.append(dict(
                key=key, row=r, held_id=int(hid[r]), match_id=int(tid[j]),
                cos=float(cT[r, j]), ng8=float(ng8(H[r], Tr[j])),
                ng8_max=float(max(ng8(H[r], x) for x in Tr)),
                cos_other_target=float(cX[r].max()), cos_random=float(cR[r].max()),
                held_src=hs, match_src=js, held_aug=bool(aug[hs]), match_aug=bool(aug[js]),
                match_same_file=hs == js, match_same_doc=bool(doc[hs] == doc[js]),
                match_same_topic=bool(topic[hs] == topic[js]),
                any_same_file=hs in tsrc, any_same_doc=bool(doc[hs] in tdoc), any_same_topic=bool(topic[hs] in ttop)))
        if t % 50 == 0:
            print("%d/%d" % (t, len(recs)), flush=True)
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print("rows %d -> %s" % (len(rows_out), OUT / "rows.jsonl"))


if __name__ == "__main__":
    main()
