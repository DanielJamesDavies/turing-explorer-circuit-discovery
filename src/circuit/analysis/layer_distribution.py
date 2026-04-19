from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List

from .base import AnalysisContext, CircuitAnalysis

if TYPE_CHECKING:
    from store.circuits import Circuit

logger = logging.getLogger(__name__)


class LayerDistributionAnalysis(CircuitAnalysis):
    """
    Computes the distribution of circuit feature nodes across transformer layers.

    Eligible nodes: valid FeatureID, kind not in {"logit", "token"}, kind does
    not end with "_err".

    Output keys (all stored under ``post_analysis``):
        layer_mean (float): mean layer index of eligible nodes.
        layer_std  (float): population standard deviation of layer indices.
        layer_min  (int):   shallowest layer present.
        layer_max  (int):   deepest layer present.

    Returns ``{}`` when there are no eligible nodes.
    """

    def analyse(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        try:
            return self._run(circuit, context)
        except Exception as exc:
            logger.warning("[LayerDistributionAnalysis] Unexpected error: %s", exc)
            return {}

    def _run(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        layers: List[int] = []
        for node in circuit.nodes.values():
            fid = node.feature_id
            if fid is None:
                continue
            if fid.kind in ("logit", "token"):
                continue
            if fid.kind.endswith("_err"):
                continue
            layers.append(fid.layer)

        if not layers:
            return {}

        n = len(layers)
        mean = sum(layers) / n
        variance = sum((l - mean) ** 2 for l in layers) / n
        std = math.sqrt(variance)

        return {
            "layer_mean": round(mean, 3),
            "layer_std":  round(std, 3),
            "layer_min":  int(min(layers)),
            "layer_max":  int(max(layers)),
        }
