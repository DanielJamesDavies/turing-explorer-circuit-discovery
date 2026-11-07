"""Shared loading and statistics for the DAN-148 paper-table generators (paper/tables/*.py).

The pass rule, the vacuous exclusion, the depth bands and the row reader are NOT re-implemented here: they come from
experiments/062-h100-protocol-v1/merge.py (the scorer's own analysis module, which paper_main.py and pass_rule.py
also use), imported after FAMILY and OUT are set, because merge.py reads both at import.

Layouts (the driver writes the same files under every OUT):
  OUT/main/eval.shard*.jsonl     held-out scores per target (held = strong | mid), the main fits
  OUT/main/status.shard*.jsonl   one status row per target (skip / error / ok)
  OUT/sweep/eval.shard*.jsonl    the lambda-sweep scores, one row per (target, arm); arm W_<lam> (WCM) or U_<lam>
                                 (unweighted, every alpha = 1)
Default OUT per family (protocol v2, chain.sh): turingllm = experiments/094-tl-v2-records/out_v2, public models =
experiments/082-public-runs/out_<family>. The v1 15k run (validation only) is experiments/062-h100-protocol-v1/out_full/out.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E062 = ROOT / "experiments" / "062-h100-protocol-v1"
V1_OUT = E062 / "out_full" / "out"
DEFAULT_OUT = {
    "turingllm": ROOT / "experiments" / "094-tl-v2-records" / "out_v2",
    "pythia-70m": ROOT / "experiments" / "082-public-runs" / "out_pythia-70m",
    "gpt2-small": ROOT / "experiments" / "082-public-runs" / "out_gpt2-small",
}
HERE = Path(__file__).resolve().parent
KIND_ORDER = ("resid", "resid_mid", "mlp", "attn")
KIND_LABEL = {"resid": "residual", "resid_mid": "residual (mid)", "mlp": "MLP", "attn": "attention"}
Z95 = 1.959963984540054


def merge_module(family, out):
    """062/merge.py for one family and OUT (it reads FAMILY and OUT from the environment at import)."""
    os.environ["FAMILY"] = family
    os.environ["OUT"] = str(out)
    for p in (str(ROOT / "src"), str(E062)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "merge" in sys.modules:                                   # one family per process; refuse a silent mismatch
        M = sys.modules["merge"]
        if str(M.OUT) != str(out) or M.FAM["name"] != family:
            raise RuntimeError("merge.py already imported for %s / %s" % (M.FAM["name"], M.OUT))
        return M
    import merge as M  # noqa: E402
    return M


def clean(df):
    """Drop error and skip rows (a skipped target has no circuit)."""
    if df.empty:
        return df
    if "error" in df:
        df = df[df["error"].isna()]
    if "skip" in df:
        df = df[df["skip"].isna()]
    return df


def add_site(df, M):
    df = df.copy()
    df["layer"] = df.seed.str.split(".").str[0].astype(int)
    df["kind"] = df.seed.str.split(".").str[1]
    df["depth"] = df.layer.map(M.depth_band)
    return df


def wilson(k, n, z=Z95):
    """Wilson score interval for k successes of n, as fractions; (nan, nan) for n = 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def in_band(x, lo=0.75, hi=1.5):
    x = pd.to_numeric(x, errors="coerce")
    return (x >= lo) & (x <= hi)                                    # NaN compares False: a missing score fails


def pre_passes(df, nec_min=0.9):
    """The secondary pass rate (protocol.md section 12.1): the same rule on the pre-activation read, with 0 ("latent
    off") as the Z and necessity baseline. Z_pre = free0_raw_pre / a_pos_pre; necessity_pre = (a_pos_pre -
    phi_sup_blind_raw_pre) / a_pos_pre; A and C keep their own mean-ablation baselines (freeM_topk_pre, freeN_topk_pre).
    A non-positive or missing a_pos_pre fails."""
    a = pd.to_numeric(df.a_pos_pre, errors="coerce")
    ok = a > 0
    z = pd.to_numeric(df.free0_raw_pre, errors="coerce") / a.where(ok)
    nec = (a - pd.to_numeric(df.phi_sup_blind_raw_pre, errors="coerce")) / a.where(ok)
    return ok & in_band(z) & in_band(df.freeM_topk_pre) & in_band(df.freeN_topk_pre) & (nec >= nec_min)


DASH = "---"                                                        # a cell with no defined value (tab:matrix note)


def missing(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def tex_int(v):
    return DASH if missing(v) else "{:,.0f}".format(v).replace(",", "{,}")


def tex_num(v, p=2):
    return DASH if missing(v) else "%.*f" % (p, v)


def band_tex(lo, hi):
    return "%d" % lo if lo == hi else "%d--%d" % (lo, hi)
