"""Shared attribution result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from circuit.types.feature_id import FeatureID


@dataclass
class UpstreamScores:
    """
    Scores produced by a single backward pass in compute_latent_upstream_scores.

    attribution:     {FeatureID: acts * grad} for latents that are active on the
                     current context.  Positive score = activator; negative = present
                     inhibitor.  Top-K selected by |score| across all predecessors.

    absent_gradient: {FeatureID: raw grad} for latents that are *inactive* (acts ≈ 0)
                     but have a strongly negative gradient — i.e. they would suppress the
                     target if they fired.  Top-K selected by most-negative gradient.
                     Empty when absent_inhibitor_top_k == 0.
    """

    attribution: Dict[FeatureID, float] = field(default_factory=dict)
    absent_gradient: Dict[FeatureID, float] = field(default_factory=dict)


__all__ = ["UpstreamScores"]
