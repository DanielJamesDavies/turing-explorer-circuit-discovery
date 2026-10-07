"""Protocol-v1 activating and contrast contexts per target (DAN-75 spec; moved into src by DAN-78).

The 062 production run built these with experiment code (059 pool_test.build / train_set / probe and 059
protocol_harness.patch_eval_contexts, injected into the engine per target). This module is that code, made the one
source for both sides: the fit (discovery under config.discovery.context_protocol = "v1") and the eval pass.

One record per target, the value the run cached in out/ctx/<L.kind.i>.pt:
  strong   the pool_size strongest stored contexts (top store, sentinel / duplicate ids dropped)
  mid      the mid-band reservoir (mid store, deduplicated against the top store), first pool_size
           each pool: pos / tgt tokens, the target's peak activation and its anchor (argmax) per context from one
           forward pass, a stratified train / held split by peak (context_split.py), and the loaded corpus ids;
           None when the store holds fewer than min_pool contexts
  D_train  the strongest training contexts: a stratified train_strong of strongest-train (32 of 48; ranks 1-2 kept)
  B_mid    the mid-band training contexts: train_mid spread over mid-train (stratified_subset)
  neg      contrast_count close contrast contexts (verified silent), similarity-stratified: train first, held last
  neg_ids  their corpus ids, row-aligned (None when the selector did not return one id per row)
Skip records have strong = None: fewer than min_pool stored top contexts, or (no_contrast = True) fewer than
min_contrast contrast contexts.

Training set (arm B): D_train + B_mid. A thin target (no mid-band pool) falls back to all of strongest-train (arm A).
Evaluation: strongest-train (the common A-ablation reference) followed by the chosen held-out set ("strong" | "mid").

Every consumer slices a ProbeDataset with the engine's rule n_train = n - round(n * holdout_frac) (learned_mask
split; amp_eval_pass_v2.split_n), so each ProbeDataset here is padded with held-out contexts to the length whose
slice is exactly the training set.
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch

from circuit.context_split import stratified_order, stratified_split, stratified_subset
from circuit.probe_dataset import ProbeDataset

# Forward / selector batch sizes and selector arguments exactly as the run used them (forward numerics depend on the
# batch size; the selector's own defaults differ from config.discovery.neg_context_selection).
FORWARD_BATCH = 16
_SELECT_KW = dict(batch_size=16, exact=False, non_activation_threshold=0.0, filter_batch_size=32, load_window_size=256)

Parts = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]       # (pos tokens, target tokens, anchors), row-aligned


def target_key(layer: int, kind: str, latent: int) -> str:
    return "%d.%s.%d" % (layer, kind, latent)


def _cfg(cfg):
    if cfg is not None:
        return cfg
    from config import config
    return config.discovery.context_v1


# ----------------------------------------------------------------------------------------------- record
def strong_holdout_frac(n_pool: int, holdout_frac: float, train_strong: int) -> float:
    """The fraction of strongest-train left out of the training set, so a full pool trains on train_strong
    (48 -> 32 holds out exactly 1/3); a smaller pool keeps the same fraction."""
    n_nominal = n_pool - int(round(n_pool * holdout_frac))
    if train_strong >= n_nominal:
        return 0.0
    return float(1 - Fraction(int(train_strong), int(n_nominal)))


def make_selector(inference, bank, loader):
    """The close-contrast selector, constructed as GradientDiscoveryBase._neg_context_selector does."""
    from store.context import mid_ctx, neg_ctx, top_ctx
    from store.seq_repr import seq_repr
    from utils.neg_context_selector import NegContextSelector
    if seq_repr is None:
        raise RuntimeError("seq_repr must be loaded before negative-context selection")
    return NegContextSelector(inference, bank, loader, neg_ctx, seq_repr, top_ctx, mid_ctx)


def _load(loader, ids: List[int]):
    """Tokens for `ids` (ProbeDatasetBuilder._load_all_ids's loader path) plus the corpus ids actually loaded, row-
    aligned (the loader skips ids it cannot locate)."""
    batches = list(loader.get_batches_by_ids(ids, max_length=65))
    t = torch.cat([tk for _, tk in batches], dim=0)
    got = [int(x) for b, _ in batches for x in b.tolist()]
    pos = t[:, :64]
    tgt = t[:, 1:65]
    if tgt.shape[1] < 64:
        tgt = torch.cat([tgt, torch.zeros(tgt.shape[0], 64 - tgt.shape[1], dtype=tgt.dtype, device=tgt.device)], 1)
    return pos, tgt, got


def _peak(inference, bank, layer: int, kind_idx: int, kind: str, latent: int, tokens: torch.Tensor):
    """The target's peak post-Top-K activation per context and its position."""
    from sae.dense import target_latent_activations
    vals, args = [], []

    def hook(layer_idx, activations):
        if layer_idx == layer:
            ta, ti = bank.encode(activations[kind_idx], kind, layer_idx)
            a = target_latent_activations(ta, ti, latent).float()
            vals.append(a.max(-1).values.cpu()); args.append(a.argmax(-1).cpu())
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), FORWARD_BATCH):
                inference.forward(tokens[s0:s0 + FORWARD_BATCH], activations_callback=hook, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    return torch.cat(vals), torch.cat(args)


def store_ids(top_ctx, mid_ctx, comp: int, latent: int) -> Tuple[List[int], List[int]]:
    """Stored context ids, strongest first: the top store, then the mid store minus anything already seen."""
    top_ids, seen = [], set()
    for sid in top_ctx.ctx_seq_idx[comp, latent].tolist():
        if int(sid) > 0 and int(sid) not in seen:
            top_ids.append(int(sid)); seen.add(int(sid))
    mid_ids = []
    for sid in mid_ctx.ctx_seq_idx[comp, latent].tolist():
        if int(sid) > 0 and int(sid) not in seen:
            mid_ids.append(int(sid)); seen.add(int(sid))
    return top_ids, mid_ids


def build_record(inference, bank, loader, comp: int, latent: int, selector=None, cfg=None,
                 top_ctx=None, mid_ctx=None) -> dict:
    """One target's context record (see the module docstring). `selector` defaults to a fresh close-contrast
    selector, as the run built one per target."""
    from pipeline.component_index import split_component_idx
    cfg = _cfg(cfg)
    if top_ctx is None or mid_ctx is None:
        from store.context import mid_ctx as _mid, top_ctx as _top
        top_ctx = _top if top_ctx is None else top_ctx
        mid_ctx = _mid if mid_ctx is None else mid_ctx
    kinds = list(bank.kinds)
    layer, kind_idx = split_component_idx(comp, len(kinds))
    kind = kinds[kind_idx]
    top_ids, mid_ids = store_ids(top_ctx, mid_ctx, comp, latent)
    n = int(cfg.pool_size)
    rec = dict(n_top=len(top_ids), n_mid=len(mid_ids))
    for name, ids in (("strong", top_ids[:n]), ("mid", mid_ids[:n])):
        if len(ids) < int(cfg.min_pool):
            rec[name] = None
            continue
        pos, tgt, got = _load(loader, ids)
        v, a = _peak(inference, bank, layer, kind_idx, kind, latent, pos)
        tr, ho = stratified_split(v, holdout_frac=cfg.holdout_frac, keep_top=cfg.keep_top)
        rec[name] = dict(pos=pos.cpu(), tgt=tgt.cpu(), arg=a, peak=v, train=tr, held=ho, ids=got)
    s = rec["strong"]
    if s is None:                                       # too few stored contexts (thin target): skip record
        return rec
    hf = strong_holdout_frac(n, cfg.holdout_frac, cfg.train_strong)
    d_keep, _ = stratified_split(s["peak"][s["train"]], holdout_frac=hf, keep_top=cfg.keep_top)
    rec["D_train"] = [s["train"][j] for j in d_keep]
    if rec["mid"] is not None:
        m = rec["mid"]
        rec["B_mid"] = [m["train"][j] for j in stratified_subset(m["peak"][m["train"]], int(cfg.train_mid))]
    # contrast: close selector, ranked by similarity -> stratified order (train first, held last)
    sel = make_selector(inference, bank, loader) if selector is None else selector
    cs = sel.select(comp, latent, "close", max_sequences=int(cfg.contrast_count), **_SELECT_KW)
    if cs is None or cs.tokens.shape[0] < int(cfg.min_contrast):     # no verified-silent contrast: unusable target
        rec["strong"] = None; rec["no_contrast"] = True
        return rec
    nt = cs.tokens[:int(cfg.contrast_count)].cpu()
    order = stratified_order(-torch.arange(nt.shape[0], dtype=torch.float64),
                             holdout_frac=cfg.holdout_frac, keep_top=cfg.keep_top)
    rec["neg"] = nt[order]
    nids = [int(x) for x in list(cs.sequence_ids)[:nt.shape[0]]]
    rec["neg_ids"] = [nids[j] for j in order] if len(nids) == nt.shape[0] else None
    return rec


def skip_reason(rec: dict) -> Optional[str]:
    """The run's status skip label for an unusable record, else None."""
    if rec.get("strong") is not None:
        return None
    return "no_contrast" if rec.get("no_contrast") else "too_few_contexts"


# ----------------------------------------------------------------------------------------------- sets
def pick(rec: dict, pool: str, idx: Sequence[int]) -> Parts:
    d = rec[pool]
    return d["pos"][idx], d["tgt"][idx], d["arg"][idx]


def cat(parts: Sequence[Parts]) -> Parts:
    return tuple(torch.cat([p[j] for p in parts], 0) for j in range(3))


def training_arm(rec: dict) -> Tuple[str, bool]:
    """(arm, thin): B = strongest + mid-band; a target without a mid-band pool falls back to A (strongest-train)."""
    return ("B", False) if rec.get("mid") is not None else ("A", True)


def training_set(rec: dict, arm: Optional[str] = None) -> Parts:
    arm = training_arm(rec)[0] if arm is None else arm
    if arm == "A":
        return cat([pick(rec, "strong", rec["strong"]["train"])])
    if arm == "B":
        return cat([pick(rec, "strong", rec["D_train"]), pick(rec, "mid", rec["B_mid"])])
    raise ValueError("protocol-v1 training arm must be 'A' or 'B', got %r" % (arm,))


def padded_length(n_train: int, holdout_frac: float = 0.25) -> int:
    """The smallest n with n - round(n * holdout_frac) == n_train (the engine's split rule)."""
    n = n_train
    while n - int(round(n * holdout_frac)) < n_train:
        n += 1
    while n - int(round(n * holdout_frac)) > n_train:          # not reachable for these sizes; guard only
        n -= 1
    return n


def probe_from(rec: dict, train: Parts, held_parts: Sequence[Parts], device, holdout_frac: float = 0.25) -> ProbeDataset:
    """A ProbeDataset whose first n_train contexts are `train`, padded with held-out contexts so the engine's
    [:n_train] slice is exactly the training set; contrast = rec["neg"]."""
    n_tr = int(train[0].shape[0])
    n = padded_length(n_tr, holdout_frac)
    filler = cat(held_parts) if held_parts else None
    need = n - n_tr
    parts = [train] + ([tuple(x[:need] for x in filler)] if need and filler is not None else [])
    pos, tgt, arg = cat(parts)
    return ProbeDataset(pos_tokens=pos.to(device), target_tokens=tgt.to(device), neg_tokens=rec["neg"].to(device),
                        pos_argmax=arg.to(device), metadata=dict(n_train=n_tr, n=int(pos.shape[0])))


def empty_probe(rec: dict, device) -> ProbeDataset:
    """What discovery gets for a skip record: no positives, so every method rejects the target."""
    z = torch.zeros((0, 64), dtype=torch.long, device=device)
    return ProbeDataset(pos_tokens=z, target_tokens=z.clone(), neg_tokens=z.clone(),
                        pos_argmax=torch.zeros(0, dtype=torch.long, device=device),
                        metadata=dict(n_train=0, n=0, skip=skip_reason(rec), n_top=rec.get("n_top")))


def _engine_holdout(holdout_frac):
    if holdout_frac is not None:
        return float(holdout_frac)
    from config import config
    return float(config.discovery.learned_mask.holdout_frac)


def training_probe(rec: dict, device, arm: Optional[str] = None, holdout_frac: Optional[float] = None) -> ProbeDataset:
    """The fit-side ProbeDataset: the training set (arm B, or A for a thin target), padded with held-out strongest
    then held-out mid-band contexts. `holdout_frac` defaults to the engine's (learned_mask.holdout_frac)."""
    if skip_reason(rec) is not None:
        return empty_probe(rec, device)
    held = [pick(rec, "strong", rec["strong"]["held"])] + ([pick(rec, "mid", rec["mid"]["held"])] if rec["mid"] else [])
    return probe_from(rec, training_set(rec, arm), held, device, _engine_holdout(holdout_frac))


def eval_probe(rec: dict, held: str, device, holdout_frac: Optional[float] = None) -> ProbeDataset:
    """The eval-side ProbeDataset: strongest-train (the common evaluation reference, whatever the training arm)
    followed by the chosen held-out set ("strong" | "mid"); contrast = the stratified close set rec["neg"]."""
    if held not in ("strong", "mid"):
        raise ValueError("held must be 'strong' or 'mid', got %r" % (held,))
    if rec.get(held) is None:
        raise ValueError("record has no %r pool" % held)
    return probe_from(rec, pick(rec, "strong", rec["strong"]["train"]), [pick(rec, held, rec[held]["held"])], device,
                      _engine_holdout(holdout_frac))


class FixedSelector:
    """A NegContextSelector stand-in that always returns the given contrast tokens (the eval pass re-selects through
    M0._neg_context_selector(); under protocol v1 it must get the record's stratified close set instead)."""

    def __init__(self, tokens: torch.Tensor):
        self.tokens = tokens

    def select(self, *a, **k):
        from utils.neg_context_selector import NegContextSelection
        return NegContextSelection(tokens=self.tokens, sequence_ids=[], mode="close", metadata={})


def eval_contexts(rec: dict, held: str, device) -> Tuple[ProbeDataset, FixedSelector]:
    """(ProbeDataset, contrast selector) for scoring a circuit on held-out `held` under protocol v1."""
    return eval_probe(rec, held, device), FixedSelector(rec["neg"].to(device))


# ----------------------------------------------------------------------------------------------- cache
class ProtocolContexts:
    """Per-engine record source: a small in-memory LRU over an optional on-disk cache (context_v1.cache_dir, one
    <L.kind.i>.pt per target holding {key: rec}, the 062 run's format, so its out/ctx can be read directly). The
    cache is keyed by target only: records must have been built with the same context_v1 settings."""

    def __init__(self, inference, bank, loader, max_cached: int = 16):
        self.inference, self.bank, self.loader = inference, bank, loader
        self.max_cached = int(max_cached)
        self._recs: "OrderedDict[str, dict]" = OrderedDict()

    def key(self, comp: int, latent: int) -> str:
        from pipeline.component_index import split_component_idx
        kinds = list(self.bank.kinds)
        layer, kind_idx = split_component_idx(comp, len(kinds))
        return target_key(layer, kinds[kind_idx], latent)

    def record(self, comp: int, latent: int, cfg=None) -> dict:
        cfg = _cfg(cfg)
        key = self.key(comp, latent)
        if key in self._recs:
            self._recs.move_to_end(key)
            return self._recs[key]
        path = Path(cfg.cache_dir) / ("%s.pt" % key) if cfg.cache_dir else None
        if path is not None and path.exists():
            rec = torch.load(path, weights_only=False)[key]
        else:
            rec = build_record(self.inference, self.bank, self.loader, comp, latent, cfg=cfg)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({key: rec}, path)
        self._recs[key] = rec
        while len(self._recs) > self.max_cached:
            self._recs.popitem(last=False)
        return rec


__all__ = ["build_record", "training_arm", "training_set", "training_probe", "eval_probe", "eval_contexts",
           "FixedSelector", "ProtocolContexts", "skip_reason", "strong_holdout_frac", "padded_length", "store_ids"]
