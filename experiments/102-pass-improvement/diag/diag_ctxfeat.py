"""102 diag Q3: context-set features per target, from the as-run 062 context records (no model). CPU.

Record layout (062 out_full/out/ctx/<key>.pt = {key: rec}): rec["strong"] / rec["mid"] = {pos [64, 64] tokens,
arg [64] anchor position, peak [64] target activation at the anchor, train / held index lists, ids corpus ids};
rec["D_train"] (32 strongest-train rows used in training, a subset of strong.train), rec["B_mid"] (16 mid-band
training rows), rec["neg"] [64, 64] contrast contexts (first 48 = contrast-train, last 16 = contrast held-out).
Training set = arm B: strong[D_train] + mid[B_mid] (48); thin targets (mid None) = arm A: strong[train] (48).
Held-out = strong[held] (16). The anchor read is causal, so all context comparisons use the LEFT context only.

Features (prefix tr_ = training activating set, ho_ = held-out strongest, ct_ = contrast-train):
  diversity     tr_anchor_distinct (share of distinct anchor tokens), tr_anchor_entropy (bits), tr_anchor_top1
                (share of the most common anchor token), tr_bigram_top1, tr_trigram_top1, tr_dup16 (share of training
                rows whose 16-token left prefix repeats another training row's), tr_win_jacc (mean pairwise Jaccard of
                the 8-token left windows), tr_anchor_pos_mean, tr_early (share of anchors at position < 8)
  similarity    ho_anchor_seen / ho_bigram_seen / ho_trigram_seen (share of held-out anchors whose n-gram occurs in
                training), ho_dup16 (share of held-out rows whose 16-token left prefix equals a training row's: the
                DAN-157 leakage, measured against the rows actually trained on), ho_nn_jacc (mean over held-out of
                the max 8-token-window Jaccard with any training row)
  activation    tr_peak_mean, tr_peak_cv, tr_peak_maxmin, ho_peak_mean, ho_tr_ratio (held-out mean / training
                mean), st_peak_maxmin (all 64 strongest), mid_strong_ratio (mid mean / strongest mean)
  contrast      ct_has_top_anchor (share of contrast-train rows containing the top training anchor token),
                ct_has_any_anchor (... any training anchor token), ct_nn_jacc (mean over contrast-train of the max
                Jaccard of its last-8 window with any training window), ct_dup (contrast rows equal to an activating
                row)

  python diag_ctxfeat.py 72        -> out/ctxfeat_72.csv (095's targets)
  python diag_ctxfeat.py all       -> out/ctxfeat_15k.csv (every record in the 062 ctx dir; resumable, chunked)
"""
import csv
import math
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch

from diag_common import CTX62, DIAG, E93

OUT = DIAG / "out"
OUT.mkdir(exist_ok=True)
W = 8          # left window for Jaccard
P = 16         # left prefix for near-duplicate detection


def _rows(pool, idx):
    pos, arg, peak = pool["pos"], pool["arg"], pool["peak"]
    return [(pos[i].tolist(), int(arg[i]), float(peak[i])) for i in idx]


def ngram(toks, a, n):
    return tuple(toks[max(0, a - n + 1): a + 1])


def win(toks, a, w=W):
    return frozenset(toks[max(0, a - w + 1): a + 1])


def prefix(toks, a, p=P):
    return tuple(toks[max(0, a - p + 1): a + 1])


def jacc(x, y):
    u = len(x | y)
    return len(x & y) / u if u else 0.0


def entropy(c):
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values())


def features(key):
    try:
        rec = torch.load(CTX62 / ("%s.pt" % key), weights_only=False)[key]
    except Exception as e:  # noqa: BLE001
        return dict(seed=key, error=str(e)[:100])
    s, m = rec.get("strong"), rec.get("mid")
    if s is None:
        return dict(seed=key, error="no strong pool")
    thin = m is None
    tr = _rows(s, s["train"]) if thin else _rows(s, rec["D_train"]) + _rows(m, rec["B_mid"])
    ho = _rows(s, s["held"])
    st_all = _rows(s, range(len(s["peak"])))
    f = dict(seed=key, thin=thin, n_tr=len(tr), n_ho=len(ho), n_mid=rec.get("n_mid"))
    a1 = Counter(ngram(t, a, 1) for t, a, _ in tr)
    a2 = Counter(ngram(t, a, 2) for t, a, _ in tr)
    a3 = Counter(ngram(t, a, 3) for t, a, _ in tr)
    n = len(tr)
    f.update(tr_anchor_distinct=len(a1) / n, tr_anchor_entropy=entropy(a1), tr_anchor_top1=a1.most_common(1)[0][1] / n,
             tr_bigram_top1=a2.most_common(1)[0][1] / n, tr_trigram_top1=a3.most_common(1)[0][1] / n,
             tr_bigram_distinct=len(a2) / n, tr_trigram_distinct=len(a3) / n)
    pf = Counter(prefix(t, a) for t, a, _ in tr)
    f["tr_dup16"] = sum(v for v in pf.values() if v > 1) / n
    ws = [win(t, a) for t, a, _ in tr]
    jj = [jacc(ws[i], ws[j]) for i in range(n) for j in range(i + 1, n)]
    f["tr_win_jacc"] = float(np.mean(jj))
    f["tr_anchor_pos_mean"] = float(np.mean([a for _, a, _ in tr]))
    f["tr_early"] = float(np.mean([a < 8 for _, a, _ in tr]))
    # train <-> held-out
    f["ho_anchor_seen"] = float(np.mean([ngram(t, a, 1) in a1 for t, a, _ in ho]))
    f["ho_bigram_seen"] = float(np.mean([ngram(t, a, 2) in a2 for t, a, _ in ho]))
    f["ho_trigram_seen"] = float(np.mean([ngram(t, a, 3) in a3 for t, a, _ in ho]))
    f["ho_dup16"] = float(np.mean([prefix(t, a) in pf for t, a, _ in ho]))
    f["ho_nn_jacc"] = float(np.mean([max(jacc(win(t, a), w) for w in ws) for t, a, _ in ho]))
    hoa1 = Counter(ngram(t, a, 1) for t, a, _ in ho)
    f["ho_anchor_distinct"] = len(hoa1) / len(ho)
    # activation profile
    tp = np.array([p for _, _, p in tr])
    hp = np.array([p for _, _, p in ho])
    sp = np.array([p for _, _, p in st_all])
    f.update(tr_peak_mean=float(tp.mean()), tr_peak_cv=float(tp.std() / max(tp.mean(), 1e-9)),
             tr_peak_maxmin=float(tp.max() / max(tp.min(), 1e-9)), ho_peak_mean=float(hp.mean()),
             ho_tr_ratio=float(hp.mean() / max(tp.mean(), 1e-9)), st_peak_maxmin=float(sp.max() / max(sp.min(), 1e-9)),
             mid_strong_ratio=(None if thin else float(m["peak"].float().mean() / max(float(s["peak"].float().mean()), 1e-9))))
    # contrast
    neg = rec.get("neg")
    if neg is not None:
        nl = [x.tolist() for x in neg]
        ntr = nl[: len(nl) - int(round(len(nl) * 0.25))]
        top = a1.most_common(1)[0][0][0]
        anchors = {k[0] for k in a1}
        f["ct_has_top_anchor"] = float(np.mean([top in set(x) for x in ntr]))
        f["ct_has_any_anchor"] = float(np.mean([bool(anchors & set(x)) for x in ntr]))
        f["ct_nn_jacc"] = float(np.mean([max(jacc(frozenset(x[-W:]), w) for w in ws) for x in ntr]))
        act = {tuple(t) for t, _, _ in st_all}
        if m is not None:
            act |= {tuple(x.tolist()) for x in m["pos"]}
        f["ct_dup"] = float(np.mean([tuple(x) in act for x in nl]))
        f["n_neg"] = len(nl)
    f["top_anchor_tok"] = a1.most_common(1)[0][0][0]
    return f


def write(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "72"
    if mode == "72":
        keys = [k for k in (E93 / "targets.txt").read_text().split() if k]
        rows = [features(k) for k in keys]
        write(rows, OUT / "ctxfeat_72.csv")
        print("wrote %d rows" % len(rows))
        return
    keys = sorted(p.stem for p in CTX62.glob("*.pt"))
    done_path = OUT / "ctxfeat_15k_parts"
    done_path.mkdir(exist_ok=True)
    chunk = 1000
    with Pool(12) as pool:
        for ci in range(0, len(keys), chunk):
            part = done_path / ("part_%05d.csv" % ci)
            if part.exists():
                continue
            rows = pool.map(features, keys[ci: ci + chunk], chunksize=20)
            write(rows, part)
            print("chunk %d done (%d rows)" % (ci, len(rows)), flush=True)
    allrows = []
    for part in sorted(done_path.glob("part_*.csv")):
        with open(part) as fh:
            allrows += list(csv.DictReader(fh))
    write(allrows, OUT / "ctxfeat_15k.csv")
    print("wrote %d rows" % len(allrows))


if __name__ == "__main__":
    main()
