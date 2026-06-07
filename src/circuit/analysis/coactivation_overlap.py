from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from .base import AnalysisContext, CircuitAnalysis

if TYPE_CHECKING:
    from store.circuits import Circuit

logger = logging.getLogger(__name__)

_INHIBITOR_ROLES = {"counterfactual_inhibitor"}
_ACTIVATOR_ROLES = {"counterfactual_activator", "ablation_support"}


class CoactivationOverlapAnalysis(CircuitAnalysis):
    """
    Computes what percentage of a circuit's SAE feature nodes also appear in
    the seed latent's stored top-K co-activating latents.

    For circuits that contain role-tagged nodes (``counterfactual_activator`` /
    ``counterfactual_inhibitor``), the metric is split by role in addition to
    the combined figure, because inhibitors are *negatively* related to the seed
    and should not be expected to appear in its top co-activations:

        coact_overlap_pct             — all eligible nodes combined
        coact_overlap_pct_activators  — activator nodes only
        coact_overlap_pct_inhibitors  — inhibitor nodes only

    For all other discovery methods only ``coact_overlap_pct`` is produced.

    Node eligibility rules (applied to all three metrics):
        - Must have a valid FeatureID.
        - kind must NOT be "logit" or "token".
        - kind must NOT end with "_err".
        - Seed node is included in the combined metric but excluded from the
          role-split metrics (it is neither an activator nor an inhibitor).
    """

    def analyse(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        try:
            return self._run(circuit, context)
        except Exception as exc:
            logger.warning("[CoactivationOverlapAnalysis] Unexpected error: %s", exc)
            return {}

    def _run(self, circuit: "Circuit", context: AnalysisContext) -> Dict[str, Any]:
        seed_comp = circuit.metadata.get("seed_comp")
        seed_latent = circuit.metadata.get("seed_latent")
        if seed_comp is None or seed_latent is None:
            return {}

        tc = context.top_coactivation
        if not tc._allocated:
            return {}

        seed_coact_ids: set[int] = set(
            tc.top_indices[int(seed_comp), int(seed_latent), :].tolist()
        )

        all_gids: List[int] = []
        activator_gids: List[int] = []
        inhibitor_gids: List[int] = []

        for node in circuit.nodes.values():
            fid = node.feature_id
            if fid is None:
                continue
            if fid.kind in ("logit", "token"):
                continue
            if fid.kind.endswith("_err"):
                continue

            gid = fid.to_global_id(context.n_kinds, context.d_sae, context.kinds)
            all_gids.append(gid)

            role = node.metadata.get("role", "")
            if role in _ACTIVATOR_ROLES:
                activator_gids.append(gid)
            elif role in _INHIBITOR_ROLES:
                inhibitor_gids.append(gid)

        if not all_gids:
            return {"coact_overlap_pct": 0.0}

        def _pct(gids: List[int]) -> float:
            if not gids:
                return 0.0
            return round(sum(1 for g in gids if g in seed_coact_ids) / len(gids) * 100, 2)

        result: Dict[str, Any] = {"coact_overlap_pct": _pct(all_gids)}

        # Emit role-split metrics only when the circuit has role-tagged nodes
        if activator_gids or inhibitor_gids:
            result["coact_overlap_pct_activators"] = _pct(activator_gids)
            result["coact_overlap_pct_inhibitors"] = _pct(inhibitor_gids)

        return result
