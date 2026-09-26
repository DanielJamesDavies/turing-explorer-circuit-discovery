"""Train / held-out split of a target's contexts (protocol v1, DAN-12 / DAN-15, 2026-09-24).

The pre-v1 rule sliced the context LIST (first n - round(n * holdout_frac) train, the rest held out). Both lists
arrive sorted (activating contexts strongest-first from the top-context store, contrast contexts most-similar-first
from the selector), so the held-out end was systematically weaker / less similar (experiments/058).

Stratified rule: rank the contexts by a score (the target's activation at its anchor for activating contexts,
similarity to the activating set for contrast contexts), cut the ranking into blocks of m = round(1 / holdout_frac),
and let block b hold out its position (b + keep_top) mod m. Held-out then spans the whole strength range, each
within-block position is held out equally often (so there is no systematic offset, unlike "last of each block"),
and the keep_top strongest contexts always train. The held-out count always equals round(n * holdout_frac), so
every consumer that slices the REORDERED list as [:n_train] / [n_train:] stays correct.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def stratified_split(scores: Sequence[float] | torch.Tensor, holdout_frac: float = 0.25,
                     keep_top: int = 2, offset: int | None = None) -> Tuple[List[int], List[int]]:
    """(train, held-out) indices into `scores`, each in descending-score order. `offset` shifts the rotation
    (default keep_top, the protocol rule); another value draws a different stratified sample of the same shape."""
    offset = keep_top if offset is None else int(offset)
    s = torch.as_tensor(scores, dtype=torch.float64).flatten()
    n = int(s.numel())
    n_hold = int(round(n * holdout_frac))
    if n_hold <= 0 or n == 0:
        return list(range(n)) if n else [], []
    ranked = torch.argsort(s, descending=True, stable=True).tolist()
    m = max(2, int(round(1.0 / holdout_frac)))
    held_ranks = []
    for b in range((n + m - 1) // m):
        r = b * m + (b + offset) % m
        if r < n and r >= keep_top:
            held_ranks.append(r)
    # exact count: trim from the weak end, or top up with the weakest remaining ranks (never the keep_top strongest)
    held_ranks = held_ranks[:n_hold]
    if len(held_ranks) < n_hold:
        spare = [r for r in range(n - 1, keep_top - 1, -1) if r not in set(held_ranks)]
        held_ranks += spare[:n_hold - len(held_ranks)]
    held_set = set(held_ranks)
    train = [ranked[r] for r in range(n) if r not in held_set]
    held = [ranked[r] for r in sorted(held_ranks)]
    return train, held


def stratified_order(scores: Sequence[float] | torch.Tensor, holdout_frac: float = 0.25,
                     keep_top: int = 2) -> List[int]:
    """A permutation putting the stratified training contexts first and the held-out ones last."""
    train, held = stratified_split(scores, holdout_frac, keep_top)
    return train + held


def stratified_subset(scores: Sequence[float] | torch.Tensor, k: int, phase: float = 0.5) -> List[int]:
    """k indices spread evenly over the score ranking (position `phase` within each of k equal rank bins; 0.5 = the
    middle), descending order."""
    s = torch.as_tensor(scores, dtype=torch.float64).flatten()
    n = int(s.numel())
    k = min(int(k), n)
    if k <= 0:
        return []
    ranked = torch.argsort(s, descending=True, stable=True).tolist()
    picks = sorted({min(n - 1, int((j + phase) * n / k)) for j in range(k)})
    return [ranked[r] for r in picks]


__all__ = ["stratified_split", "stratified_order", "stratified_subset"]
