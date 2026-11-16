"""Shared pieces of 113 (coefficient shape of the WCM v4 production circuits; CPU only).

Reuses experiments/109-pod-run-analysis/lib109.py for every archive location, row reader, the pass rule
(062 merge.py `passes`, `vacuous`), depth bands and the LaTeX writer. Adds:
  - members(model, arm): one row per circuit member (target seed, member layer / kind / index, fitted alpha = the
    node's `amplitude`, mask gate m = the node's `attribution_score`), read from the archived circuit .pt files and
    cached as out/members_<model>_<arm>.parquet;
  - freq_store(model): the corpus active counts (latent_stats.active_count) of the store the run used;
  - a target-cluster bootstrap (4,000 resamples, seed 20261114) for pooled member shares.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E109 = ROOT / "experiments" / "109-pod-run-analysis"
E084 = ROOT / "experiments" / "084-harder-nulls"
for _p in (str(ROOT / "src"), str(E109), str(E084)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lib109 as L  # noqa: E402

OUT = HERE / "out"
STORES = OUT / "stores"
TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"
PREFIX = "run2026_coef_"
BOOT_B = 4000
BOOT_SEED = 20261114
MODELS = L.MODELS
LABEL = L.LABEL

_TLX = "root/turing-tl/experiments/094-tl-v2-records/out_v2"
CIRCUIT_DIRS = {
    # main: the circuits the main eval rows score (T3 for TuringLLM; T8-cases holds 14 case-study copies, not used)
    ("turingllm", "main"): L.ARCHIVES / "pub-turingllm-T3-main" / _TLX / "main" / "circuits",
    ("pythia-70m", "main"): L.ARCHIVES / "pub-pythia-70m-S10-results" / L.LOCS["pythia-70m"]["out"] / "main" / "circuits",
    ("gpt2-small", "main"): L.ARCHIVES / "pub-gpt2-small-S10-results" / L.LOCS["gpt2-small"]["out"] / "main" / "circuits",
}
TL_SWEEP = L.ARCHIVES / "pub-turingllm-T4-sweep" / _TLX / "sweep" / "circuits"     # <arm>/<seed>.pt

# corpus active counts of the store each run used: TuringLLM's local store (the run's records were built from it);
# the public models' pod stores, extracted from Runs/pod-results-2026-11-07/pub-<model>-S10-stores.tgz
FREQ_STORE = {
    "turingllm": ROOT / "outputs" / "latent_stats.pt",
    "pythia-70m": STORES / "pythia" / "outputs_public" / "pythia-70m" / "latent_stats.pt",
    "gpt2-small": STORES / "gpt2" / "outputs_public" / "gpt2-small" / "latent_stats.pt",
}


def kinds_of(model):
    from model.families import FAMILIES
    return tuple(FAMILIES[model].kinds)


def comp_idx(model, layer, kind):
    ks = kinds_of(model)
    return int(layer) * len(ks) + ks.index(kind)


def _read_one(path):
    """(seed, [(layer, kind, index, alpha, m)]) for one circuit file; the seed node is skipped."""
    import torch
    c = torch.load(path, map_location="cpu", weights_only=False)
    out = []
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out.append((int(f.layer), str(f.kind), int(f.index), float(md.get("amplitude", np.nan)),
                    float(md.get("attribution_score", np.nan))))
    return Path(path).stem, out


def _read_many(paths):
    return [_read_one(p) for p in paths]


def circuit_dir(model, arm):
    if arm == "main":
        return CIRCUIT_DIRS[(model, "main")]
    assert model == "turingllm"
    return TL_SWEEP / arm


def members(model, arm="main", workers=None, refresh=False):
    """One row per member: seed, t_layer, t_kind, layer, kind, index, alpha, m. Cached as parquet."""
    cache = OUT / ("members_%s_%s.parquet" % (model, arm))
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    d = circuit_dir(model, arm)
    paths = sorted(str(p) for p in d.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(d)
    workers = workers or min(14, os.cpu_count() or 4)
    chunks = [paths[i::workers * 4] for i in range(workers * 4)]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_read_many, chunks):
            for seed, mem in res:
                tl, tk, _ = seed.split(".")
                for (l, k, i, a, m) in mem:
                    rows.append((seed, int(tl), tk, l, k, i, a, m))
    df = pd.DataFrame(rows, columns=["seed", "t_layer", "t_kind", "layer", "kind", "index", "alpha", "m"])
    df = df.sort_values(["seed", "layer", "kind", "index"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def strong_population(model):
    """Main eval rows, held-out strongest, non-vacuous (the paper's headline population), indexed by seed."""
    ev = L.main_eval(model)
    ev = ev[(ev.held == "strong") & ~ev.vac].copy()
    return ev.set_index("seed")


def freq_store(model):
    import torch
    p = FREQ_STORE[model]
    if not p.exists():
        return None
    o = torch.load(p, map_location="cpu", weights_only=False)
    return o["active_count"].numpy()


def boot_ci(x, stat=np.mean):
    return L.boot_ci(x, B=BOOT_B, seed=BOOT_SEED, stat=stat)


def boot_paired_diff(a, b):
    return L.boot_paired_diff(a, b, B=BOOT_B, seed=BOOT_SEED)


def cluster_share_ci(k, n):
    """95% bootstrap CI of the pooled share sum(k) / sum(n), resampling targets (clusters) with replacement."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    if len(k) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(k), size=(BOOT_B, len(k)))
    s = k[idx].sum(axis=1) / n[idx].sum(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def write_table(name, rows, columns, align, footer=None, note=None, csv_df=None):
    """paper/tables/run2026_coef_<name>.tex via lib109's writer, and out/run2026_coef_<name>.csv."""
    old = (L.TABLES, L.OUT, L.PREFIX)
    try:
        L.TABLES, L.OUT, L.PREFIX = TABLES, OUT, PREFIX
        p = L.write_table(name, rows, columns, align, note=note, csv_df=csv_df, footer=footer)
    finally:
        L.TABLES, L.OUT, L.PREFIX = old
    txt = p.read_text(encoding="utf-8").replace("generated by experiments/109-pod-run-analysis",
                                                "generated by experiments/113-coefficients-shape", 1)
    p.write_text(txt, encoding="utf-8")
    return p


def figure_style():
    plt, st = L.figure_style()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.titlepad": 4.0, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "lines.linewidth": 1.5,
                         "axes.linewidth": 0.8, "grid.linewidth": 0.5})
    return plt, st
