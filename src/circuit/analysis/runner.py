from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Type

from config import config
from .base import AnalysisContext, CircuitAnalysis
from .coactivation_overlap import CoactivationOverlapAnalysis
from .layer_distribution import LayerDistributionAnalysis
from .edge_weight_gini import EdgeWeightGiniAnalysis
from .node_activity import NodeActivityAnalysis
from .node_rarity import NodeRarityAnalysis
from .top_token_consistency import TopTokenConsistencyAnalysis
from .internode_coact_density import InternodeCoactDensityAnalysis

if TYPE_CHECKING:
    from store.circuits import Circuit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry — add new analyses here alongside their config name.
# ---------------------------------------------------------------------------

ANALYSIS_REGISTRY: Dict[str, Type[CircuitAnalysis]] = {
    "coactivation_overlap":    CoactivationOverlapAnalysis,
    "layer_distribution":      LayerDistributionAnalysis,
    "edge_weight_gini":        EdgeWeightGiniAnalysis,
    "node_activity":           NodeActivityAnalysis,
    "node_rarity":             NodeRarityAnalysis,
    "top_token_consistency":   TopTokenConsistencyAnalysis,
    "internode_coact_density": InternodeCoactDensityAnalysis,
}


def build_analyses(context: AnalysisContext) -> List[CircuitAnalysis]:
    """
    Instantiates all post-circuit analysis methods listed in
    ``config.analysis.methods``.

    Unknown names emit a warning and are skipped (matches the behaviour of
    ``_build_methods`` in discovery_window.py).
    """
    enabled: List[str] = list(config.analysis.methods)
    analyses: List[CircuitAnalysis] = []

    for name in enabled:
        cls = ANALYSIS_REGISTRY.get(name)
        if cls is None:
            logger.warning(
                "[PostCircuitAnalysis] Unknown analysis method %r — skipped. "
                "Available: %s",
                name,
                list(ANALYSIS_REGISTRY),
            )
            continue
        analyses.append(cls())

    return analyses


def run_post_circuit_analyses(
    circuit: "Circuit",
    context: AnalysisContext,
    analyses: List[CircuitAnalysis],
) -> None:
    """
    Runs all analyses in order and merges their results into
    ``circuit.metadata["post_analysis"]``.

    All analysis results are nested under the ``post_analysis`` key so they
    are clearly distinguished from discovery-time metadata and easy to
    enumerate as a group.

    Each analysis is responsible for catching its own exceptions (see
    CircuitAnalysis contract).  This function therefore never raises.
    """
    post: dict = circuit.metadata.setdefault("post_analysis", {})
    for analysis in analyses:
        result = analysis.analyse(circuit, context)
        post.update(result)
