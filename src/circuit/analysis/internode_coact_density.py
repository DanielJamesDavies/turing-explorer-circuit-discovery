from __future__ import annotations

import logging
from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

from .base import AnalysisContext, CircuitAnalysis

if TYPE_CHECKING:
    from store.circuits import Circuit

logger = logging.getLogger(__name__)

# Skip density computation for very large circuits to avoid O(n²) cost.
_MAX_ELIGIBLE_NODES = 500


class InternodeCoactDensityAnalysis(CircuitAnalysis):
    """
    Computes what percentage of unordered node pairs in the circuit are
    *mutually* co-activating: A's global ID appears in B's top-K coact list
    AND B's global ID appears in A's top-K coact list.

    A high density means the circuit nodes form a tight coactivation community
    — they regularly fire together in the corpus, independent of the seed.
    A low density means the circuit spans latents that rarely co-occur, which
    may indicate it captures long-range or cross-context causal paths.

    Eligible nodes: valid FeatureID, kind not in {"logit", "token"}, kind does
    not end with "_err".

    Output keys (stored under ``post_analysis``):
        internode_coact_density_pct (float): percentage of eligible node pairs
            that are mutually co-activating, rounded to 2 d.p.

    Returns ``{"internode_coact_density_pct": 0.0}`` when there are fewer than
    2 eligible nodes.  Returns ``{}`` when ``top_coactivation`` is not
    allocated or the circuit has more than ``_MAX_ELIGIBLE_NODES`` eligible
    nodes (skipped to avoid O(n²) cost).
    """

    def analyse(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        try:
            return self._run(circuit, context)
        except Exception as exc:
            logger.warning("[InternodeCoactDensityAnalysis] Unexpected error: %s", exc)
            return {}

    def _run(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        tc = context.top_coactivation
        if not tc._allocated:
            return {}

        # Collect eligible (global_id, comp, lat) triples
        eligible: List[Tuple[int, int, int]] = []
        for node in circuit.nodes.values():
            fid = node.feature_id
            if fid is None:
                continue
            if fid.kind in ("logit", "token"):
                continue
            if fid.kind.endswith("_err"):
                continue
            comp, lat = fid.to_component_id(context.n_kinds, context.kinds)
            gid = fid.to_global_id(context.n_kinds, context.d_sae, context.kinds)
            eligible.append((gid, comp, lat))

        n = len(eligible)
        if n < 2:
            return {"internode_coact_density_pct": 0.0}

        if n > _MAX_ELIGIBLE_NODES:
            logger.warning(
                "[InternodeCoactDensityAnalysis] %d eligible nodes exceeds limit "
                "of %d — skipping to avoid O(n²) cost.",
                n,
                _MAX_ELIGIBLE_NODES,
            )
            return {}

        # Build per-node coact sets (one tensor slice → Python set, done once per node)
        coact_sets: Dict[int, Set[int]] = {
            gid: set(tc.top_indices[comp, lat, :].tolist())
            for gid, comp, lat in eligible
        }

        node_gids = [gid for gid, _, _ in eligible]
        total_pairs = n * (n - 1) // 2
        mutual = sum(
            1
            for a, b in combinations(node_gids, 2)
            if b in coact_sets[a] and a in coact_sets[b]
        )

        pct = round(mutual / total_pairs * 100, 2)
        return {"internode_coact_density_pct": pct}
