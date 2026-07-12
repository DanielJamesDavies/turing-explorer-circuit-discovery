"""Iterative selection: the third discovery axis (one_shot | iterative).

Pure selection logic — no torch, no model. Takes an opaque score function
(one call = one grad pass at the current restored state) and greedily
builds the selected set round by round: score, take the top-k unselected
candidates by |score|, restore them, repeat. Positive scores become
positive-role candidates (activators/supports), negative scores
negative-role (inhibitors), matching the one-shot extractors' convention.

Stops early when the certificate closes: the restored-state metric is
within `certificate_tol` of the target metric (for gap objectives, the
loss is within tol of zero).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

Site = Tuple[int, str]
Candidate = Tuple[Site, int]  # ((layer, kind), latent_idx)

# score_fn(masks) -> (scores per site, restored_metric)
ScoreFn = Callable[[Dict[Site, "object"]], Tuple[Dict[Site, "object"], float]]


@dataclass
class IterativeSelectionResult:
    positives: Dict[Candidate, float] = field(default_factory=dict)
    negatives: Dict[Candidate, float] = field(default_factory=dict)
    round_of: Dict[Candidate, int] = field(default_factory=dict)
    metric_trajectory: List[float] = field(default_factory=list)
    rounds_used: int = 0
    stopped_early: bool = False


def run_iterative_selection(
    score_fn: ScoreFn,
    *,
    masks: Dict[Site, "object"],
    rounds: int,
    per_round_k: int,
    certificate_tol: float = 0.0,
    target_metric: float = 0.0,
    allow_negative: bool = True,
) -> IterativeSelectionResult:
    """Greedy restoration loop. `masks` are bool tensors mutated in place
    (they are the shared state the score_fn's instrument reads)."""

    import torch

    result = IterativeSelectionResult()
    for round_index in range(rounds):
        scores, metric = score_fn(masks)
        result.metric_trajectory.append(metric)
        if certificate_tol > 0 and abs(target_metric - metric) <= certificate_tol:
            result.stopped_early = True
            break

        flat: List[Tuple[float, Candidate]] = []
        for site, site_scores in scores.items():
            values = site_scores
            nonzero = torch.nonzero(values, as_tuple=False).squeeze(1)
            for latent in nonzero.tolist():
                candidate = (site, int(latent))
                if candidate in result.round_of:
                    continue
                value = float(values[latent])
                if value < 0 and not allow_negative:
                    continue  # never selected, never restored, never counted
                flat.append((value, candidate))
        if not flat:
            break
        flat.sort(key=lambda item: abs(item[0]), reverse=True)

        selected_this_round = 0
        for value, candidate in flat:
            if selected_this_round >= per_round_k:
                break
            site, latent = candidate
            result.round_of[candidate] = round_index
            if value >= 0:
                result.positives[candidate] = value
            else:
                result.negatives[candidate] = value
            masks[site][latent] = True
            selected_this_round += 1

        result.rounds_used = round_index + 1
        if selected_this_round == 0:
            break
    return result


__all__ = ["IterativeSelectionResult", "run_iterative_selection"]
