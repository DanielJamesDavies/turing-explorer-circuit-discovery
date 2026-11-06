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
  neg_held_v2 / neg_held_v2_ids   (086 rebuilt records only) the deduplicated contrast held-out rows; eval_probe then
           scores on rec["neg"]'s contrast-train slice + these (eval_contrast); rec["neg"] and the fit are unchanged
Skip records have strong = None: fewer than min_pool stored top contexts, or (no_contrast = True) fewer than
min_contrast contrast contexts.
build_record(..., group_split=GroupSplit(...)) (DAN-157, opt-in; the public-model path) replaces the per-row split with
the group-aware one and applies the held-out rule (circuit/heldout_rule.py); without it the record is the 062 run's.

Training set (arm B): D_train + B_mid. A thin target (no mid-band pool) falls back to all of strongest-train (arm A).
Evaluation: strongest-train (the common A-ablation reference) followed by the chosen held-out set ("strong" | "mid").

Every consumer slices a ProbeDataset with the engine's rule n_train = n - round(n * holdout_frac) (learned_mask
split; amp_eval_pass_v2.split_n), so each ProbeDataset here is padded with held-out contexts to the length whose
slice is exactly the training set.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

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


def target_activations(inference, bank, layer: int, kind_idx: int, kind: str, latent: int,
                       tokens: torch.Tensor) -> torch.Tensor:
    """The target's post-Top-K activation at every position, [n, 64] float on CPU: _peak's forward (same batch size
    and ops), keeping the whole row (the held-out rebuild needs the value at a given anchor and the window mean)."""
    from sae.dense import target_latent_activations
    out = []

    def hook(layer_idx, activations):
        if layer_idx == layer:
            ta, ti = bank.encode(activations[kind_idx], kind, layer_idx)
            out.append(target_latent_activations(ta, ti, latent).float().cpu())
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), FORWARD_BATCH):
                inference.forward(tokens[s0:s0 + FORWARD_BATCH], activations_callback=hook, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    return torch.cat(out) if out else torch.zeros(0, int(tokens.shape[1]))


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


@dataclass
class GroupSplit:
    """Opt-in group-aware split for build_record (DAN-157; the public-model path). Rows are grouped by DUPLICATES ONLY
    (exact, anchor-prefix, near: heldout_rule.dup_groups), the same rule as TuringLLM's: same-document grouping and
    the contrast document skip were dropped (Daniel, 2026-10-24). `docs(ids)` (optional) maps corpus sequence ids to
    their source document ids (-1 = unknown; doc_lookup reads 069's provenance/shard_<N>.docs.npy); it is recorded
    per row (`doc`, provenance) and never used to group, split or exclude."""
    docs: Optional[Callable[[Sequence[int]], Any]] = None


class DocIndex:
    """ids -> document ids (call), over a positional array (index 0 = no id). Provenance only."""

    def __init__(self, arr):
        import numpy as np
        self.arr = np.asarray(arr, dtype=np.int64)

    def __call__(self, ids):
        import numpy as np
        ix = np.asarray([int(i) for i in ids], dtype=np.int64)
        out = np.full(len(ix), -1, dtype=np.int64)
        ok = (ix > 0) & (ix < len(self.arr))
        out[ok] = self.arr[ix[ok]]
        return out


def group_split_for(data_dir, loader) -> GroupSplit:
    """The public path's GroupSplit: duplicates-only grouping; documents from <data_dir>/provenance are recorded per
    row when the provenance files exist (never used by the split or the contrast selection)."""
    try:
        return GroupSplit(docs=doc_lookup(data_dir, loader))
    except (FileNotFoundError, OSError, ValueError):
        return GroupSplit()


def doc_lookup(data_dir, loader) -> "DocIndex":
    """ids -> document ids from <data_dir>/provenance/shard_<N>.docs.npy (prepare_corpus), aligned with the loader's
    positional ids (1-based over its sorted, n_shards-capped shard list)."""
    import os
    import numpy as np
    parts = [np.full(1, -1, dtype=np.int64)]
    for f, n in zip(loader.shard_files, loader._shard_sequence_counts):
        p = os.path.join(str(data_dir), "provenance", f.replace(".npy", ".docs.npy"))
        d = np.load(p).astype(np.int64)
        if len(d) != int(n):
            raise ValueError("%s holds %d document ids for %d sequences" % (p, len(d), n))
        parts.append(d)
    return DocIndex(np.concatenate(parts))


def _group_split_pools(rec: dict, group_split: GroupSplit, cfg) -> None:
    """The group-aware split of both pools, in place. Groups are formed JOINTLY over the strongest and mid-band rows
    (heldout_rule.dup_groups: exact / anchor-prefix / near duplicates only), so a duplicate never trains in one pool
    while it is held out in the other: the strongest pool is split first (group_split_ex); in the mid-band pool its
    training groups must train and its held-out groups may not (held out while they fit, else unused).
    Each pool gains group (joint labels), doc (provenance, -1 when unknown), and unused (rows on neither side)."""
    from circuit.heldout_rule import dup_groups, group_split_ex
    names = [n for n in ("strong", "mid") if rec.get(n) is not None]
    if not names:
        return
    docs = {n: ([int(x) for x in group_split.docs(rec[n]["ids"])] if group_split.docs is not None
                else [-1] * len(rec[n]["ids"])) for n in names}
    lab = dup_groups(torch.cat([rec[n]["pos"] for n in names]), torch.cat([rec[n]["arg"] for n in names]))
    off, forced_tr, forced_ho = 0, set(), set()
    for n in names:
        d = rec[n]
        k = int(d["pos"].shape[0])
        g = lab[off:off + k]; off += k
        tr, ho, un = group_split_ex(d["peak"], g, holdout_frac=cfg.holdout_frac, keep_top=cfg.keep_top,
                                    force_train=forced_tr, force_held=forced_ho)
        d.update(train=tr, held=ho, unused=un, group=g, doc=docs[n])
        forced_tr |= {g[j] for j in tr}; forced_ho |= {g[j] for j in ho + un}


def _group_heldout_filter(rec: dict, group_split: GroupSplit, final_acts=None) -> None:
    """The held-out rule (circuit/heldout_rule.py) on a group-split record, in place: each held-out set keeps the rows
    that fire and are not exact / anchor-prefix / near duplicates of any training-side row (strongest-train, mid-band
    train, every contrast row), nor of an earlier kept held-out row (visited in pool order, the top store's ranking).
    Rows are only removed: this path has no top-up source. No document clause (dropped 2026-10-24), as in the
    TuringLLM rebuild."""
    from circuit import heldout_rule as HR
    s, m = rec["strong"], rec.get("mid")
    f_rows = [s["pos"][s["train"]]] + ([m["pos"][m["train"]]] if m is not None else []) + [rec["neg"]]
    f_ids = [s["ids"][j] for j in s["train"]] + ([m["ids"][j] for j in m["train"]] if m is not None else [])
    f_ids += list(rec.get("neg_ids") or [])
    F = torch.cat([t.long().cpu() for t in f_rows], 0)
    mid_ids = set(int(x) for x in m["ids"]) if m is not None else set()
    mid_h = set(int(x) for x in HR.window_hash(m["pos"])) if m is not None else set()
    forbid = HR.Forbidden(tokens=F, ids=set(int(x) for x in f_ids), hashes=set(int(x) for x in HR.window_hash(F)),
                          mid_ids=mid_ids, mid_hashes=mid_h)
    info = dict(rule=HR.RULE_VERSION, split="group", n_forbidden=int(F.shape[0]))
    for name in ("strong", "mid"):
        d = rec.get(name)
        if d is None:
            continue
        order = sorted(int(j) for j in d["held"])                   # pool order = the store's ranking
        cands = [HR.Candidate(tokens=d["pos"][j].long().cpu(), id=int(d["ids"][j]), origin="original",
                              anchor=int(d["arg"][j]), info=dict(pool_index=j)) for j in order]
        fire = lambda cs, _d=d: [dict(act_at_anchor=float(_d["peak"][c.info["pool_index"]]),
                                      anchor=int(_d["arg"][c.info["pool_index"]]),
                                      peak=float(_d["peak"][c.info["pool_index"]]), mean=None) for c in cs]
        acc, examined = HR.select_heldout(cands, forbid, fire, n_held=len(cands), exclude_mid=(name == "strong"))
        keep = {cands[i].info["pool_index"] for i in acc}
        n_split = len(d["held"])
        d["unused"] = sorted(list(d["unused"]) + [j for j in d["held"] if j not in keep])   # dropped rows: neither side
        d["held"] = [j for j in d["held"] if j in keep]
        # firing once more on the final set, forwarded as the scorer sees it (one batch, record order): a row near
        # the Top-K cut can fire in the pool's batch and not here; such rows are dropped (no top-up source here)
        unstable = []
        for _ in range(5):
            if final_acts is None or not d["held"]:
                break
            A = final_acts(d["pos"][d["held"]])
            bad = [j for j, a in zip(d["held"], A) if not float(a[int(d["arg"][j])]) > 0]
            if not bad:
                break
            unstable += bad
            d["held"] = [j for j in d["held"] if j not in set(bad)]
            d["unused"] = sorted(d["unused"] + bad)
        info[name] = dict(n_held_split=n_split, n_held=len(d["held"]), below_floor=len(d["held"]) < HR.FLOOR,
                          n_groups=len(set(d["group"])), max_group=max(Counter(d["group"]).values()),
                          dropped=dict({cands[i].info["pool_index"]: v["reason"] for i, v in examined.items()
                                        if v["reason"] is not None}, **{j: "unstable_fire" for j in unstable}),
                          unstable_fire=unstable,
                          reasons=HR.reason_counts(examined))
    rec["heldout"] = info


def build_record(inference, bank, loader, comp: int, latent: int, selector=None, cfg=None,
                 top_ctx=None, mid_ctx=None, group_split: Optional[GroupSplit] = None) -> dict:
    """One target's context record (see the module docstring). `selector` defaults to a fresh close-contrast
    selector, as the run built one per target.

    group_split (default None: the 062 run's record, unchanged) switches each pool to the group-aware split
    (circuit/heldout_rule.py: exact / anchor-prefix / near duplicates go wholly to train or to held-out; each pool
    gains `group` and `doc` (provenance) lists) and then applies the held-out rule to both held-out sets
    (rec["heldout"] holds the counts, the dropped rows and their reasons)."""
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
        pos_device = pos.device
        v, a = _peak(inference, bank, layer, kind_idx, kind, latent, pos)
        if group_split is None:
            tr, ho = stratified_split(v, holdout_frac=cfg.holdout_frac, keep_top=cfg.keep_top)
            rec[name] = dict(pos=pos.cpu(), tgt=tgt.cpu(), arg=a, peak=v, train=tr, held=ho, ids=got)
        else:
            rec[name] = dict(pos=pos.cpu(), tgt=tgt.cpu(), arg=a, peak=v, ids=got)
    if group_split is not None:
        _group_split_pools(rec, group_split, cfg)
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
    if group_split is not None:
        _group_heldout_filter(rec, group_split, final_acts=lambda t: target_activations(
            inference, bank, layer, kind_idx, kind, latent, t.to(pos_device)))
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


def probe_from(rec: dict, train: Parts, held_parts: Sequence[Parts], device, holdout_frac: float = 0.25,
               neg: Optional[torch.Tensor] = None, n_neg_train: Optional[int] = None) -> ProbeDataset:
    """A ProbeDataset whose first n_train contexts are `train`, padded with held-out contexts so the engine's
    [:n_train] slice is exactly the training set; contrast = rec["neg"] (or `neg`). n_neg_train (eval_contrast) goes
    into the metadata only when given: the scorers then take it as the contrast-train count instead of split_n."""
    n_tr = int(train[0].shape[0])
    n = padded_length(n_tr, holdout_frac)
    filler = cat(held_parts) if held_parts else None
    need = n - n_tr
    parts = [train] + ([tuple(x[:need] for x in filler)] if need and filler is not None else [])
    pos, tgt, arg = cat(parts)
    md = dict(n_train=n_tr, n=int(pos.shape[0]))
    if n_neg_train is not None:
        md["n_neg_train"] = int(n_neg_train)
    return ProbeDataset(pos_tokens=pos.to(device), target_tokens=tgt.to(device),
                        neg_tokens=(rec["neg"] if neg is None else neg).to(device),
                        pos_argmax=arg.to(device), metadata=md)


def eval_contrast(rec: dict, holdout_frac: float = 0.25) -> Tuple[torch.Tensor, Optional[int]]:
    """(contrast rows, n_neg_train) for scoring. Default (no rebuilt contrast): (rec["neg"], None), the run's 64 rows
    split by the scorer's 3:1 rule. A record rebuilt by 086 carries neg_held_v2 (the deduplicated contrast held-out
    rows, possibly fewer than 16): then rec["neg"]'s contrast-train slice (the first n - round(n * holdout_frac), the
    run's 48 rows, unchanged) followed by neg_held_v2, with n_neg_train = that train count. rec["neg"] itself is never
    changed, so the fit (training_probe) and every reader of rec["neg"] see the run's 64 rows."""
    v2 = rec.get("neg_held_v2")
    if v2 is None:
        return rec["neg"], None
    n = int(rec["neg"].shape[0])
    n_ntr = max(1, n - int(round(n * holdout_frac)))
    return torch.cat([rec["neg"][:n_ntr], v2.to(rec["neg"].dtype)], 0), n_ntr


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
    followed by the chosen held-out set ("strong" | "mid"); contrast = the stratified close set rec["neg"], or, on a
    record rebuilt by 086, its contrast-train rows followed by the deduplicated contrast held-out set (eval_contrast;
    metadata n_neg_train)."""
    if held not in ("strong", "mid"):
        raise ValueError("held must be 'strong' or 'mid', got %r" % (held,))
    if rec.get(held) is None:
        raise ValueError("record has no %r pool" % held)
    hf = _engine_holdout(holdout_frac)
    neg, n_ntr = eval_contrast(rec, hf)
    return probe_from(rec, pick(rec, "strong", rec["strong"]["train"]), [pick(rec, held, rec[held]["held"])], device,
                      hf, neg=neg, n_neg_train=n_ntr)


def eval_reference(rec: dict, device, split: str = "explicit", holdout_frac: Optional[float] = None,
                   n_seq: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """(A reference rows, C reference rows): the rows the scorers average their A and C means over, sliced exactly as
    they slice them from the held-out-strongest eval probe (amp_eval_pass_v2.score_circuit / eval.wcm_eval.
    score_circuit): the first n_train of the first n_seq activating rows (split "explicit": metadata n_train, the 48
    strongest-train rows; "as_run": the 3:1 rule) and the first n_neg_train of the first n_seq contrast rows (the 48
    contrast-train rows). Protocol v2's train_means_source="eval_ref" trains against these means."""
    from eval.wcm_eval import contrast_split, eval_split
    hf = _engine_holdout(holdout_frac)
    pd_ = eval_probe(rec, "strong", device, hf)
    pt, nt = pd_.pos_tokens[:n_seq], pd_.neg_tokens[:n_seq]
    n_tr = eval_split(int(pt.shape[0]), pd_.metadata, split, hf)
    n_ntr = contrast_split(int(nt.shape[0]), pd_.metadata, hf)
    return pt[:n_tr], nt[:n_ntr]


class FixedSelector:
    """A NegContextSelector stand-in that always returns the given contrast tokens (the eval pass re-selects through
    M0._neg_context_selector(); under protocol v1 it must get the record's stratified close set instead)."""

    def __init__(self, tokens: torch.Tensor):
        self.tokens = tokens

    def select(self, *a, **k):
        from utils.neg_context_selector import NegContextSelection
        return NegContextSelection(tokens=self.tokens, sequence_ids=[], mode="close", metadata={})


def eval_contexts(rec: dict, held: str, device) -> Tuple[ProbeDataset, FixedSelector]:
    """(ProbeDataset, contrast selector) for scoring a circuit on held-out `held` under protocol v1. The selector
    returns the probe's own contrast rows (rec["neg"], or eval_contrast's on a rebuilt record)."""
    pd_ = eval_probe(rec, held, device)
    return pd_, FixedSelector(pd_.neg_tokens)


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
           "FixedSelector", "ProtocolContexts", "skip_reason", "strong_holdout_frac", "padded_length", "store_ids",
           "GroupSplit", "DocIndex", "doc_lookup", "group_split_for", "target_activations", "eval_contrast"]
