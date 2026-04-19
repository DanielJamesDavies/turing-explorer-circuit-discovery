from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from .base import AnalysisContext, CircuitAnalysis

if TYPE_CHECKING:
    from store.circuits import Circuit

logger = logging.getLogger(__name__)


class TopTokenConsistencyAnalysis(CircuitAnalysis):
    """
    Computes what percentage of circuit feature nodes share the seed latent's
    top-1 predicted output token, as stored in ``logit_ctx``.

    A high value indicates the circuit nodes are collectively "pointed at" the
    same output token as the seed — a sign of mechanistic coherence.  A low
    value suggests the circuit spans multiple semantic roles or output targets.

    Eligible nodes: valid FeatureID, kind not in {"logit", "token"}, kind does
    not end with "_err".  The seed node itself is excluded from the count (it is
    the reference, not a comparator).  Nodes whose ``logit_ctx`` row is
    uninitialised (top-1 token == 0) are excluded from both numerator and
    denominator.

    Output keys (stored under ``post_analysis``):
        top_token_consistency_pct (float): percentage of eligible non-seed nodes
            whose top-1 predicted token matches the seed's, rounded to 2 d.p.

    Returns ``{}`` when:
        - ``logit_ctx`` is not allocated,
        - ``seed_comp`` / ``seed_latent`` are absent from metadata,
        - the seed's top-1 token is 0 (uninitialised), or
        - there are no eligible non-seed nodes with initialised logit data.
    """

    def analyse(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        try:
            return self._run(circuit, context)
        except Exception as exc:
            logger.warning("[TopTokenConsistencyAnalysis] Unexpected error: %s", exc)
            return {}

    def _run(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        lc = context.logit_ctx
        if not lc._allocated:
            return {}

        seed_comp = circuit.metadata.get("seed_comp")
        seed_latent = circuit.metadata.get("seed_latent")
        if seed_comp is None or seed_latent is None:
            return {}

        seed_token = int(lc.top_tokens[int(seed_comp), int(seed_latent), 0])
        if seed_token == 0:
            return {}

        seed_fid = None
        for node in circuit.nodes.values():
            if node.metadata.get("role") == "seed":
                seed_fid = node.feature_id
                break

        total = 0
        matches = 0
        for node in circuit.nodes.values():
            fid = node.feature_id
            if fid is None:
                continue
            if fid.kind in ("logit", "token"):
                continue
            if fid.kind.endswith("_err"):
                continue
            if seed_fid is not None and fid == seed_fid:
                continue  # exclude seed from comparison

            comp, lat = fid.to_component_id(context.n_kinds, context.kinds)
            node_token = int(lc.top_tokens[comp, lat, 0])
            if node_token == 0:
                continue  # uninitialised row — skip from both counts

            total += 1
            if node_token == seed_token:
                matches += 1

        if total == 0:
            return {}

        pct = round(matches / total * 100, 2)
        return {"top_token_consistency_pct": pct}
