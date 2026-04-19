from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from .base import AnalysisContext, CircuitAnalysis

if TYPE_CHECKING:
    from store.circuits import Circuit

logger = logging.getLogger(__name__)


class EdgeWeightGiniAnalysis(CircuitAnalysis):
    """
    Computes the Gini coefficient of the circuit's absolute edge weights.

    The Gini coefficient measures how concentrated influence is across edges:
        0.0 — all edges carry equal weight (perfectly uniform)
        1.0 — all weight is carried by a single edge (maximally concentrated)

    Output keys (stored under ``post_analysis``):
        edge_weight_gini (float): Gini coefficient in [0, 1], rounded to 4 d.p.

    Returns ``{"edge_weight_gini": 0.0}`` when there are fewer than 2 edges or
    when all edge weights are effectively zero.
    """

    def analyse(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        try:
            return self._run(circuit, context)
        except Exception as exc:
            logger.warning("[EdgeWeightGiniAnalysis] Unexpected error: %s", exc)
            return {}

    def _run(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        weights: List[float] = [
            abs(float(e.metadata.get("weight", 0.0)))
            for e in circuit.edges
        ]

        if len(weights) < 2:
            return {"edge_weight_gini": 0.0}

        total = sum(weights)
        if total < 1e-12:
            return {"edge_weight_gini": 0.0}

        # Sort ascending; standard area-under-Lorenz-curve formula
        w = sorted(weights)
        n = len(w)
        # gini = (2 * Σ (i+1)*w_i) / (n * Σ w_i)  -  (n+1)/n
        weighted_sum = sum((i + 1) * wi for i, wi in enumerate(w))
        gini = (2.0 * weighted_sum / (n * total)) - (n + 1) / n

        return {"edge_weight_gini": round(gini, 4)}
