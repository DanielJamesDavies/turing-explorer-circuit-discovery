"""Exp 128 owt_a1: WCM scope a1 (z + z2 + tc) against exp 126's scope a2 (tc only), same targets, same OWT 131k
records, lambda 1e-3, CHOSEN.env, held-out strongest (score_arena_a.py rows).

  python -I compare_a1_a2.py --a1 <rows.jsonl>[,...] --a2 <rows.jsonl>[,...] --out a1_vs_a2.md [--label l1e-3]

Per target: n latents, ho Z / A / C / nec_b, pass_b (ho and home), ho_xdoc max(Z, A, C); then per group (L0-4 /
L5-8) the pass counts, median n and median ho max(Z, A, C). A target missing on one side is shown with "-".
"""
import argparse
import json
import statistics
from pathlib import Path


def rows(spec, label):
    out = {}
    for p in [x for x in spec.split(",") if x]:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("method") != "wcm" or r.get("label") != label:
                continue
            out[(r["key"], r.get("eval_set"))] = r
    return out


def f(x, d=2):
    return "-" if x is None else ("%.*f" % (d, x) if isinstance(x, float) else str(x))


def mx(r):
    v = [r.get(k) for k in ("Z", "A", "C")]
    return max(v) if all(isinstance(x, (int, float)) for x in v) else None


def cell(R, k):
    ho, home = R.get((k, "ho")), R.get((k, "home"))
    if ho is None:
        return "- | - | - |"
    xd = ho.get("xdoc") or {}
    xm = mx(xd) if xd.get("n") else None
    return "%s | %s / %s / %s / %s | %s%s, home %s, xdoc %s |" % (
        ho.get("n_latent"), f(ho.get("Z")), f(ho.get("A")), f(ho.get("C")), f(ho.get("nec_b")),
        "pass" if ho.get("pass_b") else "fail", "" if ho.get("pass_b") else _why(ho),
        "pass" if (home or {}).get("pass_b") else ("fail" if home else "-"), f(xm))


def _why(r):
    out = []
    if r.get("nec_b") is not None and r["nec_b"] < 0.9:
        out.append("nec")
    if any(r.get(k) is not None and not 0.75 <= r[k] <= 1.5 for k in ("Z", "A", "C")):
        out.append("ZAC")
    return " (%s)" % ",".join(out) if out else ""


def group(R, keys):
    ho = [R[(k, "ho")] for k in keys if (k, "ho") in R]
    if not ho:
        return "no rows"
    ns = [r["n_latent"] for r in ho if isinstance(r.get("n_latent"), int)]
    ms = [mx(r) for r in ho if mx(r) is not None]
    return "ho pass_b %d of %d, median n %s, median ho max(Z,A,C) %s" % (
        sum(1 for r in ho if r.get("pass_b")), len(ho), f(statistics.median(ns)) if ns else "-",
        f(statistics.median(ms)) if ms else "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", required=True)
    ap.add_argument("--a2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="l1e-3")
    a = ap.parse_args()
    A1, A2 = rows(a.a1, a.label), rows(a.a2, a.label)
    keys = sorted({k for k, _ in A1} | {k for k, _ in A2}, key=lambda k: (int(k.split(".")[0]), k))
    L = ["# Exp 128 owt_a1: WCM a1 (z + z2 + tc) vs exp 126 a2 (tc only), OWT 131k, %s" % a.label, "",
         "| target | a1 n | a1 ho Z / A / C / nec_b | a1 verdict | a2 n | a2 ho Z / A / C / nec_b | a2 verdict |",
         "|---|---|---|---|---|---|---|"]
    for k in keys:
        L.append("| %s | %s %s" % (k, cell(A1, k), cell(A2, k)))
    sh = [k for k in keys if int(k.split(".")[0]) <= 4]
    dp = [k for k in keys if int(k.split(".")[0]) >= 5]
    L += ["", "| group | a1 | a2 |", "|---|---|---|",
          "| L0-4 (%d) | %s | %s |" % (len(sh), group(A1, sh), group(A2, sh)),
          "| L5-8 (%d) | %s | %s |" % (len(dp), group(A1, dp), group(A2, dp)), "",
          "Pass rule: Z, A, C in [0.75, 1.5] and nec_b >= 0.9 (held-out strongest, 16 rows). a2 rows: %s; a1 rows: %s."
          % (a.a2, a.a1)]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
