import torch
from typing import Optional, Any, Set, Tuple, cast, Dict
from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from store.top_coactivation import top_coactivation
from store.latent_stats import latent_stats
from eval.faithfulness import evaluate_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from circuit.feature_id import FeatureID
from circuit.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx


class NeighborhoodExpansion(DiscoveryMethod):
    """
    Structural circuit discovery via two-hop co-activation neighborhood expansion.

    No gradient computation required. The circuit is built purely from pre-computed
    co-activation statistics in three steps:

      Hop 0  — The seed feature is added as the root node.
      Hop 1  — All stored top co-activation neighbors of the seed are added.
      Hop 2  — The top ``n_expand`` hop-1 nodes (by co-activation strength with the
                seed) are each expanded by their own top ``m_neighbors`` co-activating
                neighbors.

    Edge weights come directly from co-activation scores; direction is determined by
    causal ordering (earlier layer / component kind → later).

    Compared to ``CoactivationStatistical`` (which uses a weight threshold to control
    how many hop-1 nodes are included), this method guarantees a richer two-hop
    neighbourhood whose size is controlled by explicit branching factors rather than a
    threshold — making it easier to reason about the graph density.
    """

    KIND_ORDER = ["attn", "mlp", "resid"]

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        min_faithfulness: Optional[float] = None,
        n_expand: Optional[int] = None,
        m_neighbors: Optional[int] = None,
        min_active_count: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.neighborhood_expansion
        self.min_faithfulness = (
            min_faithfulness
            if min_faithfulness is not None
            else cast(float, config.discovery.min_faithfulness or 0.3)
        )
        self.n_expand = n_expand or cast(int, cfg.n_expand or 16)
        self.m_neighbors = m_neighbors or cast(int, cfg.m_neighbors or 16)
        self.min_active_count = (
            min_active_count
            if min_active_count is not None
            else cast(int, config.discovery.min_active_count or 50)
        )
        self.pruning_threshold = (
            pruning_threshold
            if pruning_threshold is not None
            else cast(float, cfg.pruning_threshold or 0.0)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _causal_before(
        self,
        a: FeatureID,
        b: FeatureID,
    ) -> bool:
        """Return True if component A is causally upstream of component B."""
        if a.layer != b.layer:
            return a.layer < b.layer
        return self.KIND_ORDER.index(a.kind) < self.KIND_ORDER.index(b.kind)

    def _add_node_if_new(
        self,
        circuit: Circuit,
        node_id_map: Dict[FeatureID, str],
        fid: FeatureID,
        role: str,
    ) -> Optional[str]:
        """
        Add a node for FeatureID if it passes the activity filter
        and hasn't been added yet.  Returns the node UUID on success.
        """
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        c_idx, l_idx = fid.to_component_id(n_kinds, kinds)
        
        if latent_stats.active_count[c_idx, l_idx] < self.min_active_count:
            return None

        if fid not in node_id_map:
            node = CircuitNode(metadata={
                "feature_id": fid,
                "role": role,
            })
            circuit.add_node(node)
            node_id_map[fid] = node.uuid

        return node_id_map[fid]

    def _add_edge(
        self,
        circuit: Circuit,
        node_id_map: Dict[FeatureID, str],
        fid_a: FeatureID,
        fid_b: FeatureID,
        weight: float,
    ) -> None:
        """
        Add a directed edge between two already-registered nodes, respecting
        causal order (earlier → later).
        """
        if self._causal_before(fid_a, fid_b):
            circuit.add_edge(node_id_map[fid_a], node_id_map[fid_b], weight=weight)
        else:
            circuit.add_edge(node_id_map[fid_b], node_id_map[fid_a], weight=weight)

    def _expand_neighbors(
        self,
        fid: FeatureID,
        limit: int,
        exclude: Set[FeatureID],
    ):
        """
        Yield (neighbor_fid, weight) for the top
        ``limit`` co-activation neighbors of fid that pass
        the activity filter and are not in ``exclude``.
        """
        d_sae = self.sae_bank.d_sae
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        
        comp_idx, latent_idx = fid.to_component_id(n_kinds, kinds)
        indices = top_coactivation.top_indices[comp_idx, latent_idx]
        values = top_coactivation.top_values[comp_idx, latent_idx]

        n_yielded = 0
        for g_idx, w in zip(indices.tolist(), values.tolist()):
            if n_yielded >= limit:
                break
            
            nfid = FeatureID.from_global_id(int(g_idx), n_kinds, d_sae, kinds)
            if nfid in exclude:
                continue
            
            c_idx, l_idx = nfid.to_component_id(n_kinds, kinds)
            if latent_stats.active_count[c_idx, l_idx] < self.min_active_count:
                continue
                
            n_yielded += 1
            yield nfid, float(w)

    # ------------------------------------------------------------------
    # Main discover method
    # ------------------------------------------------------------------

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """Build a two-hop co-activation neighbourhood circuit from the seed feature."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "neighborhood_expansion")
        try:
            return self._discover(seed_comp_idx, seed_latent_idx, logger)
        finally:
            logger.save()

    def _discover(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        logger: CircuitLogger,
    ) -> Optional[Circuit]:
        circuit = Circuit(name=f"NeighExp_S{seed_comp_idx}_{seed_latent_idx}")

        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts found)")
            return None

        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]
        seed_fid = FeatureID(seed_layer, seed_kind, seed_latent_idx)

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        # --- Hop 0: seed ---
        seed_node = CircuitNode(metadata={
            "feature_id": seed_fid,
            "role": "seed",
        })
        circuit.add_node(seed_node)
        node_id_map: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}

        # --- Hop 1: all top co-activation neighbors of the seed ---
        d_sae = self.sae_bank.d_sae
        seed_indices = top_coactivation.top_indices[seed_comp_idx, seed_latent_idx]
        seed_weights = top_coactivation.top_values[seed_comp_idx, seed_latent_idx]

        hop1_candidates: list = []  # (weight, fid)
        n_hop1_skipped = 0

        for g_idx, weight in zip(seed_indices.tolist(), seed_weights.tolist()):
            nfid = FeatureID.from_global_id(int(g_idx), n_kinds, d_sae, kinds)
            uuid = self._add_node_if_new(circuit, node_id_map, nfid, "hop1")
            if uuid is None:
                n_hop1_skipped += 1
                continue

            self._add_edge(circuit, node_id_map, seed_fid, nfid, float(weight))
            hop1_candidates.append((float(weight), nfid))

        logger.stage(
            "hop-1 expansion", len(circuit.nodes), len(circuit.edges),
            note=(
                f"{len(hop1_candidates)} neighbors added, "
                f"{n_hop1_skipped} skipped (below active_count)"
            ),
        )

        # --- Hop 2: expand the top n_expand hop-1 nodes ---
        hop1_candidates.sort(key=lambda x: x[0], reverse=True)
        already_in_circuit: Set[FeatureID] = set(node_id_map.keys())
        n_expanded = min(len(hop1_candidates), self.n_expand)
        n_hop2_added = 0

        for _, h1_fid in hop1_candidates[:n_expanded]:
            for n_fid, w in self._expand_neighbors(
                h1_fid, self.m_neighbors, exclude=already_in_circuit
            ):
                uuid = self._add_node_if_new(circuit, node_id_map, n_fid, "hop2")
                if uuid is None:
                    continue
                self._add_edge(circuit, node_id_map, h1_fid, n_fid, w)
                already_in_circuit.add(n_fid)
                n_hop2_added += 1

        logger.stage(
            "hop-2 expansion", len(circuit.nodes), len(circuit.edges),
            note=(
                f"{n_expanded} hop-1 nodes expanded (n_expand={self.n_expand}), "
                f"{n_hop2_added} hop-2 nodes added (m_neighbors={self.m_neighbors})"
            ),
        )

        if len(circuit.nodes) <= 1:
            logger.reject("no neighbors added (all below activity filter)")
            return None

        # --- Optional minimality pruning ---
        if self.pruning_threshold > 0:
            n_before = len(circuit.nodes)
            prune_non_minimal_nodes(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
                threshold=self.pruning_threshold,
            )
            logger.stage(
                "after pruning", len(circuit.nodes), len(circuit.edges),
                note=f"removed {n_before - len(circuit.nodes)} nodes",
            )

        # --- Evaluation (zero ablation) ---
        final_f = evaluate_faithfulness(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
        )
        final_s = evaluate_sufficiency(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            probe_data.pos_tokens, probe_data.target_tokens,
            pos_argmax=probe_data.pos_argmax,
        )
        final_c = evaluate_completeness(
            self.inference, self.sae_bank, self.avg_acts, circuit,
            probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
        )

        logger.eval(final_f, final_s, final_c)

        if final_f < self.min_faithfulness:
            logger.reject(
                f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}"
            )
            return None

        circuit.metadata.update({
            "faithfulness": final_f,
            "sufficiency": final_s,
            "completeness": final_c,
            "seed_comp": seed_comp_idx,
            "seed_latent": seed_latent_idx,
            "n_nodes": len(circuit.nodes),
            "n_edges": len(circuit.edges),
            "discovery_method": "neighborhood_expansion",
            "n_expand": self.n_expand,
            "m_neighbors": self.m_neighbors,
        })
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit
