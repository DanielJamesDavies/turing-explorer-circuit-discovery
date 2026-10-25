"""The held-out context rule (DAN-157, approved by Daniel 2026-10-24) and the group-aware split for new corpora.

Why: the 15k run's held-out strongest contexts mostly repeat training inputs (experiments/085-heldout-audit: 66.8% of
rows are an exact, anchor-prefix or near duplicate of a training-side row; ~6.6M corpus windows were written twice and
the stable peak sort puts the twins on both sides of the split). One implementation of the rule, shared by the
TuringLLM rebuild (experiments/086-heldout-rebuild/rebuild.py, re-scoring the fixed 15k circuits) and the group-aware
split that protocol_contexts.build_record applies on request (the public-model runs, not fitted yet).

Definitions (the 085 audit's, unchanged). For a candidate row h with anchor a_h and any other row o (64-token windows):
  exact   token-identical windows (or the same corpus id, or the same corpus content hash)
  apfx    anchor-prefix copy: tokens[0 .. a_h] of h equal tokens[0 .. a_h] of o (same positions). The model is causal
          and every read is at the anchor, so for the scored quantity h IS the input o holds there
  near    longest common contiguous token run (any alignment) >= NEAR_LCR (32), or >= NEAR_C8 (50%) of h's 8-gram
          positions occur in o
There is no source-file / document rule anywhere (Daniel, 2026-10-24 final: dropped completely, also from the
public-model group-aware split, which groups by duplicates only, the same rule as TuringLLM).

Held-out rule, per target and per held-out set (select_heldout):
  forbidden F = every training-side row of the target (strongest-train incl. the fit's rows, mid-band train, all
                contrast rows) plus, by id / content hash, their corpus twins
  a candidate is accepted iff it fires (post-Top-K activation > 0 at its anchor, recomputed by the caller's fire_fn),
  is not exact / apfx / near to any row of F, and is not exact / apfx / near to an already-accepted held-out row.
  Candidates are visited in the caller's order (survivors of the original held-out set first, then new rows strongest
  first) until n_held (16) are accepted; FLOOR (8) is the minimum for a usable set.

Contrast held-out rule (select_heldout(..., want="silent"); the 086 rebuild): the same predicates with the sides
swapped. F = the contrast-train rows plus every ACTIVATING row of the record (strongest and mid-band pools, rebuilt
rows included), by tokens, id and content hash; a candidate must be SILENT (the target's post-Top-K activation is 0 at
every position, recomputed by the caller) instead of firing. A contrast row has no anchor (it never fires), so there
is NO anchor-prefix rule for contrast candidates (Daniel, 2026-10-24): they are rejected for a whole-window exact copy
(tokens, id or content hash) or a near duplicate (a >= 32-token shared run, or >= 50% shared 8-grams) only, against F
and against the contrast rows accepted before them. (Candidates still carry anchor CONTRAST_ANCHOR = 63 for the
provenance; with it, a prefix copy would be the exact copy anyway.)

Group-aware split (dup_groups + group_stratified_split): rows of one pool that are exact / apfx (either row's anchor) /
near (either direction) duplicates of each other form one group (connected components; duplicates only, no document
links), and a group goes entirely to train or entirely to held-out. With all-singleton groups the split equals
context_split.stratified_split exactly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

RULE_VERSION = "heldout-v2.1/DAN-157/2026-10-24 (no source rule; contrast held-out deduped)"
NEAR_LCR = 32          # longest common run, tokens
NEAR_C8 = 0.5          # share of the candidate's 8-gram positions found in the other row
N_HELD = 16            # held-out rows per set (the 64-row probe cap: 48 reference + 16)
FLOOR = 8              # fewer accepted rows = no usable held-out set (reported, never dropped silently)
SEQ = 64
CONTRAST_ANCHOR = SEQ - 1   # contrast rows never fire: apfx is taken over the whole window (== exact)

# 069 dup_report / 085 corpus_index content hash: the same weights, so a window hashes identically everywhere
_HASH_W = np.random.default_rng(0).integers(1, 2 ** 62, size=SEQ, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
_GRAM_W = torch.tensor([1000003 ** k % (2 ** 61 - 1) for k in range(8)], dtype=torch.int64)

# rejection reasons, in the order they are tested
REASONS = ("forbidden_id", "forbidden_hash", "mid_pool", "exact_F", "near_F", "silent", "fires", "apfx_F",
           "exact_H", "apfx_H", "near_H", "full",          # full = passed, but n_held were already accepted
           "unstable_fire",                                 # 086 rebuild: silent in the final set's own batch
           "unstable_silence")                              # 086 contrast: fires in the final set's own batch


def window_hash(rows) -> np.ndarray:
    """uint64 content hash per 64-token row (085 corpus_index / 069 dup_report)."""
    a = np.asarray(rows.cpu().numpy() if isinstance(rows, torch.Tensor) else rows)
    if a.ndim == 1:
        a = a[None]
    with np.errstate(over="ignore"):
        return (a.astype(np.uint64) * _HASH_W[None, : a.shape[1]]).sum(axis=1, dtype=np.uint64)


# ----------------------------------------------------------------------------------------------- pair statistics
def _grams(T: torch.Tensor) -> torch.Tensor:
    return sum(T[:, k:k + T.shape[1] - 7] * _GRAM_W[k] for k in range(8))          # [n, L-7] (int64 wraparound)


def pair_stats(H: torch.Tensor, O: torch.Tensor, chunk: int = 32):
    """H [h, L], O [r, L] token rows. Returns numpy (exact [h,r] bool, pfx [h,r] int = length of the common prefix,
    lcr [h,r] int = longest common contiguous run at any alignment, c8 [h,r] float = share of H's 8-gram positions
    present in O). The 085 audit's pair_stats, chunked over H."""
    H = torch.as_tensor(H).long().cpu()
    O = torch.as_tensor(O).long().cpu()
    h, r = int(H.shape[0]), int(O.shape[0])
    if h == 0 or r == 0:
        z = np.zeros((h, r))
        return z.astype(bool), z.astype(np.int64), z.astype(np.int64), z.astype(np.float64)
    L = int(H.shape[1])
    og = _grams(O)
    ex, pf, lc, cc = [], [], [], []
    for s0 in range(0, h, chunk):
        Hc = H[s0:s0 + chunk]
        eq_al = Hc[:, None, :] == O[None, :, :]                                   # [c, r, L]
        pfx = eq_al.int().cumprod(-1).sum(-1)
        eq = Hc[:, None, :, None] == O[None, :, None, :]                           # [c, r, L, L]
        run = torch.zeros(Hc.shape[0], r, L, dtype=torch.int16)
        best = torch.zeros(Hc.shape[0], r, dtype=torch.int16)
        for i in range(L):
            prev = torch.zeros_like(run)
            prev[..., 1:] = run[..., :-1]
            run = (prev + 1) * eq[:, :, i, :]
            best = torch.maximum(best, run.max(-1).values)
        hg = _grams(Hc)
        c8 = (hg[:, None, :, None] == og[None, :, None, :]).any(-1).float().mean(-1)
        ex.append((pfx == L).numpy()); pf.append(pfx.numpy().astype(np.int64))
        lc.append(best.numpy().astype(np.int64)); cc.append(c8.numpy().astype(np.float64))
    return np.concatenate(ex), np.concatenate(pf), np.concatenate(lc), np.concatenate(cc)


def near_of(lcr, c8):
    return (lcr >= NEAR_LCR) | (c8 >= NEAR_C8)


# ----------------------------------------------------------------------------------------------- held-out selection
@dataclass(eq=False)                      # identity equality: list.index / `in` never compare tensors
class Candidate:
    """One candidate held-out row. anchor None = to be set by fire_fn (argmax of the recomputed activation)."""
    tokens: Optional[torch.Tensor]        # [64] long (None: select_heldout's load_fn fills it)
    id: int                               # corpus sequence id (<= 0 unknown)
    origin: str                           # "original" | "scan" | "refill" (contrast)
    hash: Optional[int] = None            # content hash (default: computed from tokens)
    anchor: Optional[int] = None
    info: dict = field(default_factory=dict)        # provenance carried through (pool index, scan rank, mean ...)


@dataclass(eq=False)
class Forbidden:
    tokens: torch.Tensor                  # [r, 64] every training-side row (contrast: contrast-train + activating rows)
    ids: set
    hashes: set
    mid_ids: set = field(default_factory=set)       # mid-band pool ids / hashes: never enter the STRONGEST set
    mid_hashes: set = field(default_factory=set)


FireFn = Callable[[List[Candidate]], List[dict]]
# fire_fn(cands) -> one dict per candidate: act_at_anchor (float, the post-Top-K activation at the candidate's anchor,
# or at the recomputed argmax when cand.anchor is None), anchor (int), peak (float), mean (float)


def select_heldout(cands: Sequence[Candidate], forbid: Forbidden, fire_fn: FireFn, n_held: int = N_HELD,
                   exclude_mid: bool = False, chunk: int = 32,
                   load_fn: Optional[Callable[[List[Candidate]], None]] = None, want: str = "fire"):
    """Greedy held-out selection under the rule (module docstring). Returns (accepted, examined):
    accepted = indices into cands in acceptance order; examined = {index: dict(reason=None | REASONS item, fire=...)}
    for every candidate looked at (the visit stops once n_held are accepted). load_fn(cands) fills in .tokens for
    candidates that arrive without them (loaded only when the visit reaches them).
    want="fire" (activating held-out sets): a candidate must fire at its anchor (act_at_anchor > 0, else "silent").
    want="silent" (contrast held-out set): it must be silent at every position (peak <= 0, else "fires"), and only
    whole-window exact copies and near duplicates are rejected (no anchor-prefix test); give the candidates anchor
    CONTRAST_ANCHOR (provenance)."""
    if want not in ("fire", "silent"):
        raise ValueError("want must be 'fire' or 'silent', got %r" % (want,))
    accepted: List[int] = []
    acc_rows: List[torch.Tensor] = []
    acc_anchor: List[int] = []
    examined: Dict[int, dict] = {}
    F = torch.as_tensor(forbid.tokens).long().cpu()
    i0 = 0
    n = len(cands)
    while i0 < n and len(accepted) < n_held:
        idx = list(range(i0, min(n, i0 + chunk)))
        i0 = idx[-1] + 1
        need = [cands[i] for i in idx if cands[i].tokens is None]
        if need:
            if load_fn is None:
                raise ValueError("candidates without tokens need a load_fn")
            load_fn(need)
        # 1. static checks against F (anchor-free)
        live = []
        for i in idx:
            c = cands[i]
            h = int(window_hash(c.tokens)[0]) if c.hash is None else int(c.hash)
            c.hash = h
            reason = None
            if c.id > 0 and c.id in forbid.ids:
                reason = "forbidden_id"
            elif h in forbid.hashes:
                reason = "forbidden_hash"
            elif exclude_mid and ((c.id > 0 and c.id in forbid.mid_ids) or h in forbid.mid_hashes):
                reason = "mid_pool"
            examined[i] = dict(reason=reason)
            if reason is None:
                live.append(i)
        if live and F.shape[0]:
            T = torch.stack([cands[i].tokens.long().cpu() for i in live])
            ex, pfx, lcr, c8 = pair_stats(T, F)
            nr = near_of(lcr, c8)
        else:
            ex = nr = pfx = None
        keep = []
        for j, i in enumerate(live):
            c = cands[i]
            if ex is not None and ex[j].any():
                examined[i]["reason"] = "exact_F"
            elif nr is not None and nr[j].any():
                examined[i]["reason"] = "near_F"
            else:
                keep.append(j)
        # 2. fire (recomputed by the caller), anchors, anchor-prefix vs F
        if keep:
            fires = fire_fn([cands[live[j]] for j in keep])
            assert len(fires) == len(keep)
        for j, fr in zip(keep, fires if keep else []):
            i = live[j]
            c = cands[i]
            examined[i]["fire"] = fr
            if c.anchor is None:
                c.anchor = int(fr["anchor"])
            if want == "fire" and not float(fr["act_at_anchor"]) > 0:
                examined[i]["reason"] = "silent"; continue
            if want == "silent" and float(fr["peak"]) > 0:
                examined[i]["reason"] = "fires"; continue
            if want == "fire" and pfx is not None and (pfx[j] >= c.anchor + 1).any():   # contrast: no apfx rule
                examined[i]["reason"] = "apfx_F"; continue
            # 3. against the rows accepted so far (sequential: acceptance order matters)
            if len(accepted) >= n_held:
                examined[i]["reason"] = "full"; continue
            if acc_rows:
                e2, p2, l2, c2 = pair_stats(c.tokens.long().cpu()[None], torch.stack(acc_rows))
                if e2[0].any():
                    examined[i]["reason"] = "exact_H"; continue
                if want == "fire" and (p2[0] >= c.anchor + 1).any():
                    examined[i]["reason"] = "apfx_H"; continue
                if near_of(l2, c2)[0].any():
                    examined[i]["reason"] = "near_H"; continue
            accepted.append(i)
            acc_rows.append(c.tokens.long().cpu()); acc_anchor.append(c.anchor)
    return accepted, examined


def reason_counts(examined: Dict[int, dict]) -> Dict[str, int]:
    return dict(Counter(v["reason"] or "accepted" for v in examined.values()))


# ----------------------------------------------------------------------------------------------- group-aware split
def dup_groups(tokens, anchors) -> List[int]:
    """Connected components of the within-pool duplicate graph: rows i, j are linked when they are exact copies,
    near duplicates (either direction) or anchor-prefix copies under either row's anchor. No document links
    (Daniel, 2026-10-24: duplicates only). Label = the smallest row index of the component (deterministic)."""
    T = torch.as_tensor(tokens).long().cpu()
    n = int(T.shape[0])
    if n == 0:
        return []
    a = [int(x) for x in torch.as_tensor(anchors).tolist()]
    ex, pfx, lcr, c8 = pair_stats(T, T)
    link = ex | near_of(lcr, c8) | near_of(lcr.T, c8.T)
    link |= pfx >= (np.asarray(a)[:, None] + 1)
    link |= pfx >= (np.asarray(a)[None, :] + 1)
    np.fill_diagonal(link, False)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(link)):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n)]


def group_split_ex(scores, groups: Sequence[int], holdout_frac: float = 0.25, keep_top: int = 2,
                   force_train=(), force_held=()) -> Tuple[List[int], List[int], List[int]]:
    """context_split.stratified_split with whole groups. Groups in force_held (e.g. groups held out in another pool)
    may not train: each goes to held-out while it fits in round(n * holdout_frac), else it is left unused. Groups in
    force_train, and those holding one of the keep_top strongest rows, train (force_held wins a conflict). Then walk the
    stratified held-out ranks (block b holds out rank b*m + (b + keep_top) mod m) and the remaining ranks from the weak
    end, taking each undecided rank's group for held-out when it fits; a group that does not fit trains.
    Returns (train, held, unused), each in descending-score order. No group straddles train and held-out."""
    s = torch.as_tensor(scores, dtype=torch.float64).flatten()
    n = int(s.numel())
    g = [int(x) for x in groups]
    assert len(g) == n
    n_hold = int(round(n * holdout_frac))
    ranked = torch.argsort(s, descending=True, stable=True).tolist() if n else []
    members: Dict[int, List[int]] = {}
    for i, gi in enumerate(g):
        members.setdefault(gi, []).append(i)
    no_train = {int(x) for x in force_held} & set(members)
    if n_hold <= 0 and not no_train:
        return list(range(n)), [], []                       # stratified_split's degenerate case, verbatim
    held_groups, unused, count = set(), set(), 0
    for r in range(n - 1, -1, -1):                          # forced-held groups, weakest first (deterministic)
        gi = g[ranked[r]]
        if gi in no_train and gi not in held_groups and gi not in unused:
            if count + len(members[gi]) <= n_hold:
                held_groups.add(gi); count += len(members[gi])
            else:
                unused.add(gi)
    forced = ({g[ranked[r]] for r in range(min(keep_top, n))} | {int(x) for x in force_train}) - no_train
    decided = forced | no_train
    m = max(2, int(round(1.0 / holdout_frac)))
    pattern = []
    for b in range((n + m - 1) // m):
        r = b * m + (b + keep_top) % m
        if keep_top <= r < n:
            pattern.append(r)
    pattern += [r for r in range(n - 1, keep_top - 1, -1) if r not in set(pattern)]
    for r in pattern:
        if count >= n_hold:
            break
        gi = g[ranked[r]]
        if gi in decided:
            continue
        decided.add(gi)
        if count + len(members[gi]) <= n_hold:
            held_groups.add(gi); count += len(members[gi])
    train = [ranked[r] for r in range(n) if g[ranked[r]] not in held_groups | unused]
    held = [ranked[r] for r in range(n) if g[ranked[r]] in held_groups]
    unused_rows = [ranked[r] for r in range(n) if g[ranked[r]] in unused]
    return train, held, unused_rows


def group_stratified_split(scores, groups: Sequence[int], holdout_frac: float = 0.25,
                           keep_top: int = 2) -> Tuple[List[int], List[int]]:
    """(train, held) of group_split_ex without forced groups. All-singleton groups give context_split.stratified_split's
    result exactly; the held-out set may be smaller when groups are large (the caller reports it)."""
    train, held, unused = group_split_ex(scores, groups, holdout_frac, keep_top)
    assert not unused
    return train, held


__all__ = ["RULE_VERSION", "NEAR_LCR", "NEAR_C8", "N_HELD", "FLOOR", "CONTRAST_ANCHOR", "REASONS", "window_hash",
           "pair_stats", "near_of",
           "Candidate", "Forbidden", "select_heldout", "reason_counts", "dup_groups", "group_split_ex",
           "group_stratified_split"]
