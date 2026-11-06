"""102 diag: shared loaders and pass rules (floor 0.75), CPU only, no model.

Pass rules (brief, 2026-10-31):
  production  free0_tk, freeM_topk_tk, freeN_topk_tk in [FLOOR, 1.5] and phi_sup_blind_tk >= 0.9; vacuous_tk excluded
  resample    production pass AND donor A, donor C recovery in [FLOOR, 1.5] (095: rs_A / rs_C; 100: recovery dict)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments"
DIAG = Path(__file__).resolve().parent
E62, E93, E95, E100 = (EXP / "062-h100-protocol-v1", EXP / "093-v2-validation", EXP / "095-v2-resample-appendix",
                       EXP / "100-scrub-frontier")
CTX62 = E62 / "out_full" / "out" / "ctx"
FLOOR, CEIL, NEC = 0.75, 1.5, 0.9
BANDS = (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11))
KINDS = ("attn", "mlp", "resid")
TERMS = ("Z", "A", "C", "nec")
PROD_KEYS = dict(Z="free0_tk", A="freeM_topk_tk", C="freeN_topk_tk", nec="phi_sup_blind_tk")


def band_of(layer):
    return next(b for b, lo, hi in BANDS if lo <= int(layer) <= hi)


def load_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def fnum(v):
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


def inband(v, lo=FLOOR):
    v = fnum(v)
    return v is not None and lo <= v <= CEIL


def band_deficit(v, lo=FLOOR):
    """How far a band term sits outside [lo, 1.5] (0 inside; None -> inf)."""
    v = fnum(v)
    if v is None:
        return float("inf")
    if v < lo:
        return lo - v
    if v > CEIL:
        return v - CEIL
    return 0.0


def nec_deficit(v):
    v = fnum(v)
    if v is None:
        return float("inf")
    return max(0.0, NEC - v)


def prod_pass(p, lo=FLOOR):
    if p.get("vacuous_tk"):
        return None
    nec = fnum(p.get("phi_sup_blind_tk"))
    return bool(inband(p.get("free0_tk"), lo) and inband(p.get("freeM_topk_tk"), lo)
                and inband(p.get("freeN_topk_tk"), lo) and nec is not None and nec >= NEC)


def term_values(p, rsA=None, rsC=None):
    d = {t: fnum(p.get(k)) for t, k in PROD_KEYS.items()}
    d["rsA"], d["rsC"] = fnum(rsA), fnum(rsC)
    return d


def deficits(vals, lo=FLOOR, with_rs=True):
    """Per-term deficit (necessity measured against its 0.9 bar; band terms against [lo, 1.5])."""
    out = {}
    for t in ("Z", "A", "C"):
        out[t] = band_deficit(vals[t], lo)
    out["nec"] = nec_deficit(vals["nec"])
    if with_rs:
        for t in ("rsA", "rsC"):
            out[t] = band_deficit(vals[t], lo)
    return out


def load_095():
    """{arm: {seed: row}} with derived pass flags at FLOOR."""
    rows = load_jsonl(E95 / "out" / "rows.jsonl")
    by = {}
    for r in rows:
        if "error" in r or "prod" not in r:
            continue
        p = r["prod"]
        r["P"] = prod_pass(p)
        r["RS"] = None if r["P"] is None else bool(r["P"] and inband(r.get("rs_A")) and inband(r.get("rs_C")))
        r["vals"] = term_values(p, r.get("rs_A"), r.get("rs_C"))
        by.setdefault(r["arm"], {})[r["seed"]] = r          # keep last
    return by


def load_093():
    by = {}
    for r in load_jsonl(E93 / "out" / "rows.jsonl"):
        if "error" in r:
            continue
        r["P"] = prod_pass(r)
        by.setdefault(r["arm"], {})[r["seed"]] = r
    return by


def load_062_eval():
    """v1 15k held-out strongest rows, dedupe by seed keep last."""
    out = {}
    for f in sorted((E62 / "out_full" / "out" / "main").glob("eval.*.jsonl")):
        for r in load_jsonl(f):
            if r.get("held") != "strong" or "error" in r:
                continue
            out[r["seed"]] = r
    return out


def md_table(header, rows):
    s = "| " + " | ".join(str(h) for h in header) + " |\n"
    s += "|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(fmt(x) for x in r) + " |\n"
    return s


def fmt(x):
    if isinstance(x, float):
        if x != x:
            return "-"
        return "%.3g" % x if abs(x) < 1e4 else "%.0f" % x
    return "-" if x is None else str(x)


def rate(xs):
    xs = [x for x in xs if x is not None]
    return "%d/%d" % (sum(bool(x) for x in xs), len(xs)) if xs else "-"


def auc(score, y):
    """AUC of `score` for y=1 (Mann-Whitney, ties half); None when a class is empty."""
    s = np.asarray(score, float)
    y = np.asarray(y, bool)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
