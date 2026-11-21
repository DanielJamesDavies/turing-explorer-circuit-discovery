"""Exp 126 (CPU): context diversity of the target records (hypothesis C: a poor / small corpus).

Per target record (make_targets ctx/<key>.pt; tokens are the record's own strong / mid `pos` windows, i.e. the corpus
windows the record's `ids` point at; documents from the record's `doc` provenance lists):
  docs64       distinct source documents among the 64 strongest rows
  docs train / held / fit   ... among strong["train"] (48), strong["held"] (16), and the fit rows (D_train strong +
               B_mid mid-band; arm A targets: strong["train"])
  held_sd_fit  held rows whose document is also a fit row's document;  held_sd_tr  ... a strong-train or mid-train
               row's document (score_arena_a's ho_xdoc reference: the complement is ho_xdoc's n)
  near-dup     token 8-gram Jaccard between each held row and each fit row (window tokens, set of 8-grams):
               held_nd = held rows whose max Jaccard over the fit rows > JAC (0.5); max_jac_med = median over held rows
               of that max; also the same against all 48 strong-train rows (held_nd_tr)
Output: markdown (--out) with one row per target plus medians (all / shallow L0-4 tc), and a jsonl beside it.

  WSL, repo root:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES= ./.venv/bin/python -B experiments/126-wcm-overshoot/diversity.py \
      --ctx experiments/107-true-baselines/arena_a/out/targets_smoke/ctx --kind tc \
      --out experiments/126-wcm-overshoot/out/diversity.md [--title "..."] [--keys k1,k2]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
NG, JAC = 8, 0.5


def grams(row, n=NG):
    t = [int(x) for x in row]
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def jac(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def one(key, rec):
    s, m = rec["strong"], rec.get("mid")
    if s is None:
        return dict(key=key, skip=True)
    doc = s.get("doc")
    arm = "B" if m is not None and rec.get("B_mid") is not None else "A"
    tr, ho = list(s["train"]), list(s["held"])
    if arm == "B":
        fit_s, fit_m = list(rec["D_train"]), list(rec["B_mid"])
    else:
        fit_s, fit_m = tr, []
    mdoc = None if m is None else m.get("doc")
    out = dict(key=key, layer=int(key.split(".")[0]), kind=key.split(".")[1], arm=arm, n_strong=int(s["pos"].shape[0]),
               n_train=len(tr), n_held=len(ho), n_fit_strong=len(fit_s), n_fit_mid=len(fit_m))
    if doc is not None:
        dset = lambda idx, d=doc: {d[j] for j in idx}  # noqa: E731
        fit_docs = dset(fit_s) | ({mdoc[j] for j in fit_m} if mdoc is not None else set())
        tr_docs = dset(tr) | ({mdoc[j] for j in m["train"]} if mdoc is not None else set())
        out.update(docs64=len(set(doc)), docs_train=len(dset(tr)), docs_held=len(dset(ho)), docs_fit=len(fit_docs),
                   held_sd_fit=sum(doc[j] in fit_docs for j in ho), held_sd_tr=sum(doc[j] in tr_docs for j in ho),
                   top_doc_share=max(doc.count(d) for d in set(doc)) / len(doc))
    else:
        out.update(note="no provenance (doc) in the record")
    g_s = [grams(r) for r in s["pos"]]
    g_m = [grams(r) for r in m["pos"]] if m is not None else []
    fit_g = [g_s[j] for j in fit_s] + [g_m[j] for j in fit_m]
    tr_g = [g_s[j] for j in tr]
    mx_fit = [max((jac(g_s[j], g) for g in fit_g), default=0.0) for j in ho]
    mx_tr = [max((jac(g_s[j], g) for g in tr_g), default=0.0) for j in ho]
    out.update(held_nd=sum(x > JAC for x in mx_fit), held_nd_tr=sum(x > JAC for x in mx_tr),
               max_jac_med=st.median(mx_fit) if mx_fit else None, max_jac_max=max(mx_fit) if mx_fit else None)
    return out


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "%.2f" % v
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True, help="records dir (<key>.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="tc", help="only keys of this kind ('' = all)")
    ap.add_argument("--keys", default="")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    import torch
    paths = sorted(Path(a.ctx).glob("*.pt"), key=lambda p: tuple(int(x) if x.isdigit() else x for x in p.stem.split(".")))
    keys = set(k for k in a.keys.split(",") if k)
    rows = []
    for p in paths:
        key = p.stem
        if keys and key not in keys:
            continue
        if a.kind and key.split(".")[1] != a.kind:
            continue
        d = torch.load(p, weights_only=False, map_location="cpu")
        rows.append(one(key, d[key]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cols = ["key", "arm", "n_fit_strong", "n_fit_mid", "docs64", "docs_train", "docs_held", "docs_fit", "held_sd_fit",
            "held_sd_tr", "top_doc_share", "held_nd", "held_nd_tr", "max_jac_med", "max_jac_max"]
    L = ["# Context diversity%s" % (": " + a.title if a.title else ""), "",
         "Records: `%s` (%d %s targets). Held = the 16 held-out strongest rows (the `ho` eval set). Documents = the "
         "record's provenance (`doc`). held_sd_fit / held_sd_tr = held rows sharing a document with a fit row "
         "(D_train + B_mid) / with any strong-train or mid-train row (the ho_xdoc reference). held_nd = held rows "
         "whose max token %d-gram Jaccard over the fit rows > %.1f (held_nd_tr: over the 48 strong-train rows); "
         "max_jac_med / max = median / max over held rows of that max. top_doc_share = the largest share of the 64 "
         "strongest rows from one document." % (a.ctx, len(rows), a.kind or "all", NG, JAC), "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        if r.get("skip"):
            L.append("| %s | skip (no strong pool) |" % r["key"] + " |" * (len(cols) - 2))
            continue
        L.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    for name, sel in (("all", rows), ("shallow L0-4", [r for r in rows if r.get("layer", 99) <= 4])):
        sel = [r for r in sel if not r.get("skip")]
        if not sel:
            continue
        med = {c: st.median([r[c] for r in sel if r.get(c) is not None]) if any(r.get(c) is not None for r in sel)
               else None for c in cols[2:]}
        L.append("| **median %s (%d)** | | " % (name, len(sel)) + " | ".join(fmt(med[c]) for c in cols[2:]) + " |")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
