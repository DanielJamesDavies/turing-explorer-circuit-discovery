"""124 public-model check (CPU, measure only; DAN-217 item 4): how many held-out rows of Pythia-70m and GPT-2 small hit
the same TF-IDF test (114's vectoriser, cos >= 0.3 against any of the same target's activating training rows)?

No new rule for the public models. Their production corpora are not on this machine (data_public/<model> is the 069
smoke corpus), so the IDF cannot be fitted on 100k random corpus windows as 114 did for TuringLLM. Instead it is
fitted on the unique token rows stored in that model's own production records (every strongest, mid-band and contrast
row of every target, de-duplicated by content), with 114's code and settings otherwise unchanged. A control repeats
this records-only IDF on the TuringLLM subset and compares the hit rate with the corpus IDF, to show how much the
IDF source moves the count. Baseline: the same max-cos against ANOTHER (seeded random) target's training rows, as 114.

  python -X utf8 experiments/124-paraphrase-filter-subset/public_check.py
Writes out/public_check.json and out/public_rows_<model>.csv.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import para124 as P  # noqa: E402

C = P.C
SEED = 20261116


def records(ctx_dir, keys=None):
    files = sorted(Path(ctx_dir).glob("*.pt"))
    if keys is not None:
        want = set(keys)
        files = [f for f in files if f.stem in want]
    out = []
    for f in files:
        d = torch.load(f, weights_only=False)
        k, rec = next(iter(d.items()))
        if isinstance(rec, dict) and rec.get("strong") is not None:
            out.append((k, rec))
    return out


def all_rows(recs):
    rows = []
    for _, r in recs:
        rows.append(np.asarray(r["strong"]["pos"]))
        if r.get("mid") is not None:
            rows.append(np.asarray(r["mid"]["pos"]))
        if r.get("neg") is not None and len(r["neg"]):
            rows.append(np.asarray(r["neg"]))
    x = np.concatenate(rows).astype(np.int64)
    _, first = np.unique(x, axis=0, return_index=True)
    return x[np.sort(first)]


def held(rec, pool):
    p = rec.get(pool)
    if p is None or not len(p["held"]):
        return np.zeros((0, 64), np.int64)
    return np.asarray(p["pos"][p["held"]]).astype(np.int64)


def train_docs(rec):
    """Per-row source document of the training rows, in train_rows order (public records only; -1 = unknown)."""
    s, m = rec["strong"], rec.get("mid")
    if "doc" not in s:
        return None
    d = [int(s["doc"][j]) for j in s["train"]]
    if m is not None and m.get("train"):
        d += [int(m["doc"][j]) for j in m["train"]]
    return np.array(d)


def measure(recs, tf, thresh=P.THRESH):
    rng = np.random.default_rng(SEED)
    out = []
    for t, (k, rec) in enumerate(recs):
        rule = P.ParaRule(tf, P.train_rows(rec), thresh)
        _, other = recs[(t + 1 + int(rng.integers(len(recs) - 1))) % len(recs)]
        rule_o = P.ParaRule(tf, P.train_rows(other), thresh)
        hs, hm = held(rec, "strong"), held(rec, "mid")
        cs, co, cm = rule.cos_max(hs), rule_o.cos_max(hs), rule.cos_max(hm)
        row = dict(seed=k, strong_rows=len(hs), strong_hit=int((cs >= thresh).sum()),
                   strong_hit_other=int((co >= thresh).sum()), mid_rows=len(hm), mid_hit=int((cm >= thresh).sum()),
                   cos_med=float(np.median(cs)) if len(cs) else None)
        td = train_docs(rec)
        if td is not None and len(hs) and rule.vT is not None:
            j = (rule.tf(hs) @ rule.vT.T).toarray().argmax(axis=1)               # best training match per held row
            hd = np.array([int(rec["strong"]["doc"][x]) for x in rec["strong"]["held"]])
            hit = cs >= thresh
            row["strong_hit_same_doc"] = int((hit & (hd >= 0) & (hd == td[j])).sum())
        out.append(row)
    return pd.DataFrame(out).set_index("seed")


def summary(df):
    out = dict(targets=int(len(df)), strong_rows=int(df.strong_rows.sum()), strong_hit=int(df.strong_hit.sum()),
               strong_hit_other=int(df.strong_hit_other.sum()), targets_with_hit=int((df.strong_hit > 0).sum()),
               mid_rows=int(df.mid_rows.sum()), mid_hit=int(df.mid_hit.sum()),
               strong_hit_pct=100.0 * float(df.strong_hit.sum()) / max(1, int(df.strong_rows.sum())))
    if "strong_hit_same_doc" in df:
        out["strong_hit_same_doc"] = int(df.strong_hit_same_doc.fillna(0).sum())
    return out


def main():
    res = dict(thresh=P.THRESH, idf="records-only (unique stored rows of the model's production records)", models={})
    for m in ("pythia-70m", "gpt2-small"):
        circ = sorted(p.stem for p in (C.pub_paths(m)["out"] / "main" / "circuits").glob("*.pt"))
        recs = records(C.pub_paths(m)["ctx"], circ)                 # production targets (a main circuit file)
        rows = all_rows(recs)
        tf = P.K114.Tfidf(rows)
        df = measure(recs, tf)
        df.to_csv(P.WORK / ("public_rows_%s.csv" % m))
        res["models"][m] = dict(summary(df), idf_rows=int(len(rows)), ctx=str(C.pub_paths(m)["ctx"]))
        print(m, res["models"][m], flush=True)
    keys = C.read_targets(P.subset_path())
    recs = records(C.tl_paths()["ctx"], keys)
    tf_r = P.K114.Tfidf(all_rows(recs))
    a, b = measure(recs, tf_r), measure(recs, P.load_tfidf())
    res["tl_control"] = dict(targets=len(recs), rows=int(a.strong_rows.sum()), hit_records_idf=int(a.strong_hit.sum()),
                             hit_corpus_idf=int(b.strong_hit.sum()), other_records_idf=int(a.strong_hit_other.sum()),
                             other_corpus_idf=int(b.strong_hit_other.sum()))
    print("TL control", res["tl_control"])
    P.write_json(res, P.WORK / "public_check.json")


if __name__ == "__main__":
    main()
