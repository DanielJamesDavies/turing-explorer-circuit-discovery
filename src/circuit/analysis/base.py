from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Sequence

if TYPE_CHECKING:
    from store.circuits import Circuit
    from store.top_coactivation import TopCoactivation
    from store.latent_stats import LatentStats
    from store.logit_context import LogitContext


@dataclass
class AnalysisContext:
    """
    Shared resources passed to every post-circuit analysis.

    Constructed once per DiscoveryWindow and reused across all circuits.
    Add new fields here when a new analysis needs additional shared resources.
    """

    n_kinds: int
    d_sae: int
    kinds: Sequence[str]
    top_coactivation: "TopCoactivation"
    latent_stats: "LatentStats"
    logit_ctx: "LogitContext"


class CircuitAnalysis(ABC):
    """
    Abstract base class for post-circuit analysis methods.

    Subclass this, implement ``analyse``, and register your class in
    ``runner.ANALYSIS_REGISTRY`` to make it available via config.

    Contract:
    - ``analyse`` must never raise — catch all exceptions internally, log a
      warning, and return a partial / empty dict.
    - Return only JSON-serialisable primitive values (int, float, str, bool,
      or flat lists of those) so results survive ``_save_summary`` without
      special handling.
    - Do NOT mutate the circuit or context directly; return a dict and let
      the runner merge it into ``circuit.metadata``.
    """

    @abstractmethod
    def analyse(
        self,
        circuit: "Circuit",
        context: AnalysisContext,
    ) -> Dict[str, Any]:
        """
        Analyse a discovered circuit and return a dict of metadata key/value
        pairs to merge into ``circuit.metadata``.

        Args:
            circuit: The fully-evaluated, accepted circuit.
            context: Shared resources (bank dimensions, loaded stores).

        Returns:
            Dict of new metadata entries.  Empty dict is valid (no-op).
        """
