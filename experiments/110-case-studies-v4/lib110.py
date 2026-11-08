"""Shared pieces of 110 (case studies on the 2026-11-07/08 production run, WCM v4, TuringLLM 8k).

Locations of the unpacked production archives (made by 109/unpack.sh), the score table (109's row reader and
062/merge.py's pass rule, imported, not re-implemented), and the environment that points 063's tooling
(validate.Validator -> 062 driver.Runner -> amp_eval_pass_v2) at the production OUT and the fresh records.

Call `use_production()` BEFORE importing 063's validate / 062's driver: it sets OUT, CTX_DIR, WCM_VERSION.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments"
ARCH = EXP / "109-pod-run-analysis" / "out" / "archives"
OUT_V4 = ARCH / "pub-turingllm-T3-main" / "root" / "turing-tl" / "experiments" / "094-tl-v2-records" / "out_v2"
CTX_FRESH = ARCH / "pub-turingllm-T2-records" / "root" / "turing-tl" / "experiments" / "082-public-runs" / \
    "out_turingllm" / "ctx_fresh"
OLD_OUT = EXP / "062-h100-protocol-v1" / "out_full" / "out"          # the 062 full run (the paper's old case study)
RESULTS = HERE / "out"
FIGS = ROOT / "paper" / "figures"

for _p in (str(ROOT / "src"), str(EXP / "109-pod-run-analysis"), str(EXP / "062-h100-protocol-v1"),
           str(EXP / "063-case-studies")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def use_production():
    os.environ["OUT"] = str(OUT_V4)
    os.environ["CTX_DIR"] = str(CTX_FRESH)
    os.environ["WCM_VERSION"] = "v4"
    os.environ.setdefault("TAG", "case110_unused")


def parse(key):
    l, k, i = key.split(".")
    return int(l), k, int(i)


def scores(held="strong"):
    """Production TuringLLM eval rows (keep-last per seed, held), with pass / vacuous flags (merge.py v2 rule)."""
    import lib109 as L
    df = L.frame(L.read_rows([OUT_V4 / "main"], "eval"))
    df = df[df.held == held].copy()
    df["vacuous"] = L.vacuous(df).values
    df["passes"] = (L.passes(df) & ~df["vacuous"]).values
    df["worst_dev"] = (df[L.HEAD].astype(float) - 1).abs().max(axis=1)
    return df.set_index("seed")


def old_scores():
    """The 062 full run's eval rows (held-out strongest), same rule."""
    import lib109 as L
    df = L.frame(L.read_rows([OLD_OUT / "main"], "eval"))
    if "held" in df:
        df = df[df.held == "strong"].copy()
    df["vacuous"] = L.vacuous(df).values
    df["passes"] = (L.passes(df) & ~df["vacuous"]).values
    return df.set_index("seed")


def members(c):
    """[(layer, kind, index, alpha)] of a circuit .pt (seed node skipped)."""
    out = []
    for nd in c.nodes.values():
        md = nd.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out.append((int(f.layer), str(f.kind), int(f.index), float(md.get("amplitude", 1.0))))
    return out


def load_circuit(key, old=False):
    import torch
    return torch.load((OLD_OUT if old else OUT_V4) / "main" / "circuits" / ("%s.pt" % key), weights_only=False)


def record(key):
    import torch
    return torch.load(CTX_FRESH / ("%s.pt" % key), weights_only=False)[key]
