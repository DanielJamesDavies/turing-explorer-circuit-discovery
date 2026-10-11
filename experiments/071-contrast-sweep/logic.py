"""DAN-64 contrast sweep: the pure logic (no model, no GPU), shared by sweep.py and report.py and unit-tested by
test_logic.py.

  - SOURCES        the four contrast sources (close / random / distant / fused, all post-Top-K silent). close, random
                   and distant are training arms AND eval sets; fused (a mix of the three sub-modes, src's quotas) is a
                   training arm only. Every circuit is scored on the three held-out eval sets (the 4 x 3 table)
  - disjointness   the sets are built in BUILD_ORDER (close, random, distant, fused); each excludes every context
                   (train, held-out and reserve) of the sets built before it, and the target's activating contexts
                   (disjoint_keep). So all arms are scored on the same held-out sets and no arm ever trained on one
  - fused          fused_quotas() splits a count into sub-mode thirds (close n//3 + remainder, distant and random
                   n//3, as src's _select_fused). fused_split_merge() splits each part separately with v1's
                   stratified rule, held-out counts by largest remainder (fused_held_counts) so the totals are
                   exactly n_train / the rest, then merges round robin (close, distant, random, close, ...) the
                   training slices first and the held-out slices last. Of 64: train 16 / 16 / 16, held-out 6 / 5 / 5
  - set order      a source's contexts arrive in the selector's order; stratified_positions() puts the stratified
                   training contexts first and the held-out last, exactly as protocol v1 orders the close set
  - set versions   set_version(): the three eval sets keep the tag they were built with (071-disjoint-v1), so sets
                   already on disk stay valid; fused has its own tag
  - pass rule      protocol v1 (DAN-8): Z, A, C in [0.8, 1.5] and necessity >= 0.9, activation read
  - matched size   interp_log_n(): a value at a reference circuit size, linear in log10(size) over an arm's lambda grid

There is no decision rule: this is a test of contrast sources, reported descriptively (README section 4).

  PYTHONPATH=src .venv/bin/python -m pytest experiments/071-contrast-sweep/test_logic.py -q
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------------------------------- protocol constants
HOLDOUT_FRAC = 0.25          # protocol v1: 48 train / 16 held out of 64
KEEP_TOP = 2                 # the stratified split's always-train ranks
CONTRAST_COUNT = 64          # contexts per contrast set
MIN_CONTRAST = 4             # fewer usable contexts: the source is unusable for the target (v1's no_contrast rule)
RESERVE = 32                 # extra contexts per source, past its 64 (kept; also excluded from later sources)
SLACK = 16                   # extra contexts selected to absorb the exclusions (doubled while the set comes up short)
MAX_GROW = 4                 # at most this many doublings of the selection size
PASS_LO, PASS_HI, NEC_MIN = 0.8, 1.5, 0.9
HEAD = ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")      # Z, sparsity-preserving A, sparsity-preserving C
NEC = "phi_sup_blind_tk"
IND = "phi_cf_alpha_blind_tk"
# The medians reported per train x eval cell: Z, A, C, necessity, sufficiency to induce
METRICS: Dict[str, str] = {"Z": "free0_tk", "A": "freeM_topk_tk", "C": "freeN_topk_tk", "nec": NEC, "ind": IND}

# The selector arguments of the 15k run's close set (src/circuit/protocol_contexts._SELECT_KW), used for every source.
SELECT_KW = dict(batch_size=16, exact=False, non_activation_threshold=0.0, filter_batch_size=32, load_window_size=256)
SELECTION_SEED = 0           # the selector's default, as the run called it (fixes the random shuffle per target)

# name -> (selector mode, filter). "silent": kept only if the target's post-Top-K activation is 0 at every position.
SOURCES: Dict[str, Tuple[str, str]] = {
    "close_silent": ("close", "silent"),         # protocol v1 production: rec["neg"] as the 15k run trained on it
    "random_silent": ("random", "silent"),
    "distant_silent": ("distant", "silent"),
    "fused_silent": ("fused", "silent"),         # sub-mode thirds of close / distant / random; a training arm only
}
FUSED = "fused_silent"
FUSED_PARTS: Tuple[str, ...] = ("close", "distant", "random")      # src's sub-mode order: selection and round robin
BUILD_ORDER: Tuple[str, ...] = ("close_silent", "random_silent", "distant_silent", FUSED)  # each excludes the earlier
ARMS: Tuple[str, ...] = BUILD_ORDER
EVAL_SETS: Tuple[str, ...] = ("close_silent", "random_silent", "distant_silent")          # fused is not an eval set
REFERENCE = "close_silent"                       # the size reference for matched size (production), not a "winner"
LAMS: Tuple[float, ...] = (3e-4, 1e-3, 3e-3)
REF_LAM = 1e-3                                   # the production operating point; reference size = close_silent here
BANDS: Tuple[str, ...] = ("L0-4", "L5-7", "L8-11")
SET_VERSION = "071-disjoint-v1"                  # cached set files of another design are refused (use a fresh OUT)
# Per-set tags where a set differs from SET_VERSION. The three eval sets keep SET_VERSION, so the sets built before
# fused was added (e.g. out_smoke2/sets) stay valid; fused gets its own tag.
SET_VERSIONS: Dict[str, str] = {FUSED: "071-fused-v2"}  # v2: per-part split (v1, merged-then-split, is refused)


def set_version(name: str) -> str:
    """The version tag a cached set file of source `name` must carry."""
    return SET_VERSIONS.get(name, SET_VERSION)


def band(layer: int) -> str:
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def lam_tag(lam: float) -> str:
    return "%g" % lam


def n_train_of(n: int) -> int:
    """The engine's and the scorer's split rule: the first n - round(n / 4) contexts train."""
    return max(1, n - int(round(n * HOLDOUT_FRAC)))


def stratified_positions(n: int) -> List[int]:
    """A permutation of range(n), where position 0 is the selector's first context (most similar for close, least
    similar for distant, first drawn for random), putting the stratified training contexts first and the held-out
    ones last. This is what build_record applies to the v1 close set (stratified_order(-arange(n))), so held-out spans
    the whole selection order."""
    import torch
    from circuit.context_split import stratified_order
    return stratified_order(-torch.arange(n, dtype=torch.float64), holdout_frac=HOLDOUT_FRAC, keep_top=KEEP_TOP)


def content_key(row) -> str:
    """A context's identity for overlap checks: a digest of its 64 tokens (int64). Two corpus ids with identical
    tokens count as the same context, which is the right notion for train / held-out leakage."""
    import numpy as np
    if hasattr(row, "detach"):                                   # torch tensor (int32 from the token cache, or int64)
        row = row.detach().cpu().long().numpy()
    a = np.ascontiguousarray(np.asarray(row, dtype=np.int64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------- disjoint sources
def disjoint_keep(ids: Sequence[int], keys: Sequence[str], excl_ids: Iterable[int] = (),
                  excl_keys: Iterable[str] = ()) -> Tuple[List[int], Dict[str, int]]:
    """Positions of a selection (in the selector's order) that survive the exclusions: a context is dropped if its
    corpus id or its token content is excluded (an earlier source's train / held-out / reserve, or an activating
    context), or if its content already occurred earlier in this selection (so train and held-out of one source never
    share a context). Order is preserved. Returns (kept positions, drop counts)."""
    excl_ids, excl_keys = {int(i) for i in excl_ids}, set(excl_keys)
    seen, kept = set(), []
    drop = {"id": 0, "content": 0, "dup": 0}
    for j, (i, k) in enumerate(zip(ids, keys)):
        if int(i) in excl_ids:
            drop["id"] += 1
        elif k in excl_keys:
            drop["content"] += 1
        elif k in seen:
            drop["dup"] += 1
        else:
            seen.add(k)
            kept.append(j)
    return kept, drop


def split_main_reserve(kept: Sequence[int], n: int = CONTRAST_COUNT, reserve: int = RESERVE
                       ) -> Tuple[List[int], List[int]]:
    """The first n kept positions are the set (to be stratified), the next `reserve` its reserve."""
    kept = list(kept)
    return kept[:n], kept[n:n + reserve]


def fused_quotas(n: int) -> Dict[str, int]:
    """src's fused quotas (NegContextSelector._select_fused): close n//3 + the remainder, distant and random n//3."""
    q = n // 3
    return {"close": q + (n - 3 * q), "distant": q, "random": q}


def round_robin(parts: Dict[str, Sequence], order: Sequence[str] = FUSED_PARTS) -> List[Tuple[str, object]]:
    """Merge the sub-mode parts one item at a time in `order` (close, distant, random, close, ...), each part in its
    own order; a part that runs out is skipped. Returns [(sub-mode, item)]."""
    out, i = [], 0
    while True:
        added = False
        for p in order:
            if i < len(parts.get(p, ())):
                out.append((p, parts[p][i]))
                added = True
        if not added:
            return out
        i += 1


def fused_held_counts(sizes: Dict[str, int], n_held: int) -> Dict[str, int]:
    """Held-out contexts per sub-mode part, summing to exactly n_held: the largest-remainder apportionment of n_held
    in proportion to the part sizes (ties in FUSED_PARTS order). A part keeps its keep_top first contexts in training
    while the total allows it. Of 22 / 21 / 21 with 16 held out: 6 / 5 / 5 (5.5 / 5.25 / 5.25)."""
    n = sum(sizes.values())
    held = {p: 0 for p in sizes}
    if n == 0 or n_held <= 0:
        return held
    exact = {p: s * n_held / n for p, s in sizes.items()}
    names = [p for p in FUSED_PARTS if p in sizes] + [p for p in sizes if p not in FUSED_PARTS]
    for keep in (KEEP_TOP, 0):                       # second pass only if the keep_top caps make n_held unreachable
        cap = {p: max(0, s - keep) for p, s in sizes.items()}
        held = {p: min(cap[p], int(math.floor(exact[p]))) if keep else held[p] for p in sizes}
        while sum(held.values()) < n_held:
            room = [p for p in names if held[p] < cap[p]]
            if not room:
                break
            best = max(room, key=lambda p: (exact[p] - held[p], -names.index(p)))
            held[best] += 1
        if sum(held.values()) == n_held:
            return held
    raise ValueError("cannot hold out %d of %s" % (n_held, sizes))


def part_split(n: int, n_hold: int) -> Tuple[List[int], List[int]]:
    """(train, held-out) positions of one fused part of n contexts, in its selector order: v1's stratified rule
    (stratified_split on -arange(n), i.e. rank = selector position) with the held-out count set to n_hold, trimmed
    from the weak end or topped up with the weakest training positions (the keep_top first ones last), as src's rule
    fixes its own count. Both lists ascending."""
    import torch
    from circuit.context_split import stratified_split
    if n == 0:
        return [], []
    _, held = stratified_split(-torch.arange(n, dtype=torch.float64), holdout_frac=HOLDOUT_FRAC, keep_top=KEEP_TOP)
    held = sorted(held)[:n_hold]
    hs = set(held)
    spare = ([r for r in range(n - 1, KEEP_TOP - 1, -1) if r not in hs]
             + [r for r in range(min(KEEP_TOP, n) - 1, -1, -1) if r not in hs])
    held = sorted(held + spare[:n_hold - len(held)])
    hs = set(held)
    return [r for r in range(n) if r not in hs], held


def fused_split_merge(parts: Dict[str, Sequence]) -> List[Tuple[str, object]]:
    """fused's set order: each sub-mode part (in its selector order) split separately (part_split, held-out counts
    from fused_held_counts so the totals are exactly n_train_of(n) / the rest), then the training slices merged round
    robin, then the held-out slices merged round robin. So the first n_train_of(n) are training, the rest held out,
    as every consumer slices the set. Returns [(sub-mode, item)]."""
    sizes = {p: len(parts.get(p, ())) for p in FUSED_PARTS}
    n = sum(sizes.values())
    if n == 0:
        return []
    n_held = fused_held_counts(sizes, n - n_train_of(n))
    train, held = {}, {}
    for p in FUSED_PARTS:
        t, h = part_split(sizes[p], n_held[p])
        train[p], held[p] = [parts[p][j] for j in t], [parts[p][j] for j in h]
    out = round_robin(train) + round_robin(held)
    assert sum(len(v) for v in train.values()) == n_train_of(n) and len(out) == n
    return out


def part_counts(parts: Sequence[str], n_train: int, reserve_parts: Sequence[str] = ()) -> Dict[str, Dict[str, int]]:
    """Per sub-mode: contexts in the set, in its training split (the first n_train), held out, and in the reserve."""
    parts, reserve_parts = list(parts), list(reserve_parts)
    return {p: dict(n=parts.count(p), train=parts[:n_train].count(p), held=parts[n_train:].count(p),
                    reserve=reserve_parts.count(p)) for p in FUSED_PARTS}


def held_in_train(train_keys: Iterable[str], held_keys: Iterable[str]) -> int:
    """Held-out contexts (of an eval set) that are also training contexts (of an arm), by content."""
    return len(set(held_keys) & set(train_keys))


def pairwise_overlap(keys: Dict[str, Iterable]) -> Dict[Tuple[str, str], int]:
    """|keys[a] & keys[b]| for every unordered pair of sets (a before b in the dict's order)."""
    names = list(keys)
    ks = {n: set(keys[n]) for n in names}
    return {(a, b): len(ks[a] & ks[b]) for i, a in enumerate(names) for b in names[i + 1:]}


def ho_trained_matrix(sets: Dict[str, dict]) -> Dict[Tuple[str, str], int]:
    """(arm, eval) -> held-out contexts of the eval set that the arm trains on. sets[name] has "keys" (stratified:
    train first) and "n_train". Off-diagonal entries are 0 by construction; random's and distant's diagonal is 0 by
    the in-selection dedup; close_silent's diagonal counts the production set's own exact duplicates across its split
    (the production set is used as is)."""
    out = {}
    for a, A in sets.items():
        for e, E in sets.items():
            out[(a, e)] = held_in_train(A["keys"][:A["n_train"]], E["keys"][E["n_train"]:])
    return out


def ho_trained_allowed(arm: str, ev: str) -> bool:
    """Whether a nonzero held-out-in-training count is tolerated: only the production close set's own split (its
    exact duplicate windows are a property of the 15k run's set, which close_silent reuses unchanged)."""
    return arm == ev == REFERENCE


# ------------------------------------------------------------------------------------------- pass rule
def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def passes(row: dict) -> bool:
    """Protocol v1 (DAN-8) on one scorer row: Z, A, C all in [0.8, 1.5] and necessity >= 0.9; a missing score fails.
    C is the row's own eval set's C (freeN_topk_tk)."""
    vals = [_num(row.get(k)) for k in HEAD]
    nec = _num(row.get(NEC))
    if any(v is None for v in vals) or nec is None:
        return False
    return all(PASS_LO <= v <= PASS_HI for v in vals) and nec >= NEC_MIN


# ------------------------------------------------------------------------------------------- matched size
def interp_log_n(ns: Sequence[float], ys: Sequence[float], n_star: float) -> Tuple[Optional[float], bool]:
    """y at circuit size n_star, linear in log10(n) through one arm's (size, y) points over its lambda grid. Points
    are sorted by size; equal sizes are averaged. Outside the range the nearest endpoint is returned and flagged.
    Returns (y, extrapolated)."""
    pts: Dict[float, List[float]] = {}
    for n, y in zip(ns, ys):
        n, y = _num(n), _num(y)
        if n is None or y is None or n <= 0:
            continue
        pts.setdefault(math.log10(n), []).append(y)
    if not pts or _num(n_star) is None or n_star <= 0:
        return None, True
    xs = sorted(pts)
    yv = [sum(pts[x]) / len(pts[x]) for x in xs]
    x = math.log10(n_star)
    if len(xs) == 1:
        return yv[0], abs(x - xs[0]) > 1e-12
    if x < xs[0]:
        return yv[0], True
    if x > xs[-1]:
        return yv[-1], True
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return yv[i] + t * (yv[i + 1] - yv[i]), False
    return yv[-1], False                                         # not reachable


def matched_rates(P, N, idx, ref_arm: int, ref_lam: int, evals=None):
    """Pass rates at matched size over the targets `idx`.

    P [target, arm, lambda, eval] pass indicators; N [target, arm, lambda] circuit sizes. The reference size n* is the
    median size of arm ref_arm at lambda ref_lam; each arm's rate (mean over targets and the eval sets `evals`, default
    all) at each lambda is interpolated in log10(median size) to n*.
    Returns (rates [arm], extrapolated [arm], n*)."""
    import numpy as np
    P, N = np.asarray(P, dtype=float), np.asarray(N, dtype=float)
    idx = np.asarray(idx)
    ev = slice(None) if evals is None else list(evals)
    n_star = float(np.median(N[idx, ref_arm, ref_lam]))
    rates, extra = [], []
    for a in range(P.shape[1]):
        n_med = np.median(N[idx, a, :], axis=0)
        r = P[idx, a][:, :, ev].mean(axis=(0, 2))
        y, x = interp_log_n(n_med.tolist(), r.tolist(), n_star)
        rates.append(np.nan if y is None else y)
        extra.append(x)
    return np.array(rates), np.array(extra), n_star


def matched_cells(X, N, idx, n_star: float, stat: str = "mean"):
    """X [target, arm, lambda, eval] (pass indicators, or a score with NaN for missing); N [target, arm, lambda].
    Per arm x eval: the mean (pass rate) or the median (a score) over the targets `idx` at each lambda, interpolated
    in log10(median size) to n_star. Returns (values [arm, eval], extrapolated [arm])."""
    import numpy as np
    X, N = np.asarray(X, dtype=float), np.asarray(N, dtype=float)
    idx = np.asarray(idx)
    agg = np.nanmean if stat == "mean" else np.nanmedian
    out = np.full((X.shape[1], X.shape[3]), np.nan)
    extra = []
    for a in range(X.shape[1]):
        n_med = np.median(N[idx, a, :], axis=0)
        ex = True
        for e in range(X.shape[3]):
            col = X[idx, a, :, e]
            ys = [agg(col[:, j]) if np.isfinite(col[:, j]).any() else np.nan for j in range(col.shape[1])]
            y, ex = interp_log_n(n_med.tolist(), ys, n_star)
            out[a, e] = np.nan if y is None else y
        extra.append(ex)
    return out, np.array(extra)
