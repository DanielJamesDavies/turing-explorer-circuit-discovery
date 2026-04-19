from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from .base import AnalysisContext, CircuitAnalysis

if TYPE_CHECKING:
    from store.circuits import Circuit

logger = logging.getLogger(__name__)


class NodeRarityAnalysis(CircuitAnalysis):
    """
    Computes the fraction of circuit feature nodes whose lifetime firing count
    falls at or below the global 10th percentile of ``latent_stats.active_count``.

    A high ``rarity_pct`` indicates the circuit relies heavily on rare, specialised
    latents; a low value suggests it is built from common, high-frequency features.

    Eligible nodes: valid FeatureID, kind not in {"logit", "token"}, kind does
    not end with "_err".

    Output keys (stored under ``post_analysis``):
        rarity_pct (float): percentage of eligible nodes at or below the global
            p10 active_count threshold, rounded to 2 d.p.

    Returns ``{}`` when ``latent_stats`` is not allocated or there are no
    eligible nodes.
    """

    def analyse(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        try:
            return self._run(circuit, context)
        except Exception as exc:
            logger.warning("[NodeRarityAnalysis] Unexpected error: %s", exc)
            return {}

    def _run(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        ls = context.latent_stats
        if not ls._allocated:
            return {}

        # Global p10 threshold across all (component, latent) pairs
        p10 = float(ls.active_count.float().quantile(0.1).item())

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

        rare = sum(1 for c in counts if c <= p10)
        rarity_pct = round(rare / len(counts) * 100, 2)

        return {"rarity_pct": rarity_pct}
