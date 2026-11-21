"""Exp 128 trim check (2026-11-18): is a fit after the donor-code trim (learned_mask.capture_donor_codes) the same as
the one fitted before it? Compares the circuit .pt files (every tensor / value, recursively) and the train rows'
loss numbers for each key present in both runs.

  python compare_trim.py --old <wcm main dir> --new <wcm main dir> --out <md>
"""
import argparse
import json
from pathlib import Path

import torch


def diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            out.append((path, "shape/dtype %s %s vs %s %s" % (tuple(a.shape), a.dtype, tuple(b.shape), b.dtype)))
        elif not torch.equal(a, b):
            d = (a.double() - b.double()).abs().max().item() if a.is_floating_point() else float((a != b).sum())
            out.append((path, "max |diff| %.3g" % d))
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append((path + "/" + str(k), "only in %s" % ("new" if k not in a else "old")))
            else:
                diff(a[k], b[k], path + "/" + str(k), out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((path, "len %d vs %d" % (len(a), len(b))))
        for j, (x, y) in enumerate(zip(a, b)):
            diff(x, y, "%s[%d]" % (path, j), out)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and a != a and b != b):
        out.append((path, "%r vs %r" % (a, b) if len(repr(a)) < 80 else "values differ"))
    return out


def train_rows(d):
    rows = {}
    for f in sorted(Path(d).glob("train.shard*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows[r["seed"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old, new = Path(a.old), Path(a.new)
    to, tn = train_rows(old), train_rows(new)
    lines = ["# Exp 128 trim check: fits before vs after the donor-code trim", "",
             "old %s" % old, "new %s" % new, "",
             "| key | circuit identical | differences (first 5) | n old / new | loss_final old / new |", "|---|---|---|---|---|"]
    for f in sorted((new / "circuits").glob("*.pt")):
        g = old / "circuits" / f.name
        if not g.exists():
            continue
        d = diff(torch.load(g, map_location="cpu", weights_only=False), torch.load(f, map_location="cpu", weights_only=False))
        k = f.stem
        ro, rn = to.get(k, {}), tn.get(k, {})
        lines.append("| %s | %s | %s | %s / %s | %s / %s |" % (
            k, "YES" if not d else "no", "; ".join("%s: %s" % x for x in d[:5]) or "-",
            ro.get("n_kept"), rn.get("n_kept"), ro.get("loss_final"), rn.get("loss_final")))
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
