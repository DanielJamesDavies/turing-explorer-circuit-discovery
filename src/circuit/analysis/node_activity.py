from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from .base import AnalysisContext, CircuitAnalysis

if TYPE_CHECKING:
    from store.circuits import Circuit

logger = logging.getLogger(__name__)


class NodeActivityAnalysis(CircuitAnalysis):
    """
    Computes the distribution of lifetime firing counts for circuit feature nodes,
    using ``latent_stats.active_count``.

    Eligible nodes: valid FeatureID, kind not in {"logit", "token"}, kind does
    not end with "_err".

    Output keys (stored under ``post_analysis``):
        activity_mean   (float): mean active_count across eligible nodes.
        activity_median (float): median active_count across eligible nodes.

    Returns ``{}`` when ``latent_stats`` is not allocated or there are no
    eligible nodes.
    """

    def analyse(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        try:
            return self._run(circuit, context)
        except Exception as exc:
            logger.warning("[NodeActivityAnalysis] Unexpected error: %s", exc)
            return {}

    def _run(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        ls = context.latent_stats
        if not ls._allocated:
            return {}

        counts: List[int] = []
        for node in circuit.nodes.values():
            fid = node.feature_id
            if fid is None:
                continue
            if fid.kind in ("logit", "token"):
                continue
            if fid.kind.endswith("_err"):
                continue
            comp, lat = fid.to_component_id(context.n_kinds, context.kinds)
            counts.append(int(ls.active_count[comp, lat]))

        if not counts:
            return {}

        mean = sum(counts) / len(counts)

        sorted_c = sorted(counts)
        n = len(sorted_c)
        mid = n // 2
        median = (
            float(sorted_c[mid])
            if n % 2 == 1
            else (sorted_c[mid - 1] + sorted_c[mid]) / 2.0
        )

        return {
            "activity_mean":   round(mean, 1),
            "activity_median": median,
        }
