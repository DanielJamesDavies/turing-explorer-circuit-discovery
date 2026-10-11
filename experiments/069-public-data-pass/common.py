"""Shared setup for the 069 public-model data pass (DAN-37).

`configure(...)` must run BEFORE anything imports `store.*` or `pipeline.*`: the store singletons (top_ctx, mid_ctx,
neg_ctx, latent_stats, logit_ctx, top_coactivation) read their grid from `config.model.family` when they are created
(model.families.public_store_grid). It sets the family, points the loader at the family's shard directory, and
leaves every other setting at config.yaml's values unless overridden here. Nothing is written under data/ or outputs/:
the shard index cache goes to <data_root>/.shard_indices and every artefact to <out_root>.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

# The artefact set TuringLLM's pipeline writes (Runs/20260531-152059-37117a33 + global_negctx_ids.pt, written by the
# current neg_ctx stage) plus the seq_latent_index directory.
ARTEFACTS = ["latent_stats.pt", "top_ctx.pt", "mid_ctx.pt", "seq_repr.pt", "logit_ctx.pt", "neg_ctx.pt",
             "neg_ctx_stats.json", "global_negctx_ids.pt", "top_coactivation.pt", "candidates.pt"]

_PROTECTED = [os.path.join(REPO, "data"), os.path.join(REPO, "outputs"),
              os.path.abspath(os.path.join(REPO, "..", "Runs"))]


def _guard(path: str, what: str) -> str:
    path = os.path.abspath(path)
    for p in _PROTECTED:
        if path == p or path.startswith(p + os.sep):
            raise SystemExit("%s %s is inside TuringLLM's read-only %s" % (what, path, p))
    return path


def shard_count(data_root: str) -> int:
    return len([f for f in os.listdir(data_root) if f.startswith("shard_") and f.endswith(".npy")])


def configure(family: str, data_root: str, out_root: Optional[str] = None, *, batch_size: Optional[int] = None,
              mid_warmup: Optional[int] = None, skip_missing_saes: bool = False, dtype: Optional[str] = None,
              n_seeds: Optional[int] = None):
    """Point the global config at a public family's corpus. Returns the config object."""
    if "store.context" in sys.modules:
        raise RuntimeError("configure() must run before store modules are imported (their grid is set at import)")
    from config import config

    data_root = _guard(data_root, "data root")
    if out_root is not None:
        _guard(out_root, "output root")
    n_shards = shard_count(data_root)
    if n_shards == 0:
        raise SystemExit("no shard_*.npy under %s (run prepare_corpus.py first)" % data_root)

    config.model.family = family
    config.model.dtype = dtype
    config.model.skip_missing_saes = bool(skip_missing_saes)
    # Lazy loading when sites may be missing: a missing dictionary is found on first use, not at construction.
    config.model.sae_preload = not skip_missing_saes
    config.data.dataset_path = data_root
    config.data.n_shards = n_shards
    if batch_size is not None:
        config.data.batch_size = int(batch_size)
    if mid_warmup is not None:
        config.latents.mid_ctx.warmup_batches = int(mid_warmup)
    if n_seeds is not None:
        config.discovery.n_seeds = int(n_seeds)
    # The search cache is not part of the artefact set (run_pipeline builds it after discovery only).
    config.persist.search_cache_enabled = False
    return config


def check_store_grid(bank) -> dict:
    """The stores were allocated on the family's grid and it matches the bank's."""
    from store.context import top_ctx
    from store.latent_stats import latent_stats
    from store.top_coactivation import top_coactivation
    want = (int(bank.n_layer) * len(bank.kinds), int(bank.d_sae))
    for name, store in (("top_ctx", top_ctx), ("latent_stats", latent_stats), ("top_coactivation", top_coactivation)):
        got = (int(store.num_components), int(store.sae_config.d_sae))
        if got != want:
            raise RuntimeError("%s grid %s != bank grid %s (configure() ran too late?)" % (name, got, want))
    return dict(n_components=want[0], d_sae=want[1], kinds=list(bank.kinds), n_layer=int(bank.n_layer),
                k=int(bank.k), d_model=int(bank.d_model))


def load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def load_stores(out_root: str, n_seqs: int, which=("top", "mid", "neg", "seq_repr")) -> None:
    """Load a finished run's stores into the module singletons (store.context.*, store.seq_repr.seq_repr)."""
    import store.seq_repr as sr
    from store.context import mid_ctx, neg_ctx, top_ctx
    if "top" in which:
        top_ctx.load(os.path.join(out_root, "top_ctx.pt"))
    if "mid" in which:
        mid_ctx.load(os.path.join(out_root, "mid_ctx.pt"))
    if "neg" in which:
        neg_ctx.load(os.path.join(out_root, "neg_ctx.pt"))
    if "seq_repr" in which:
        rep = sr.SeqRepr(n_seqs=n_seqs)
        rep.load(os.path.join(out_root, "seq_repr.pt"))
        sr.seq_repr = rep
