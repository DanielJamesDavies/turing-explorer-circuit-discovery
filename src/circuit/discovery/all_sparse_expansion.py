import torch
from typing import Optional, Any, Set, Tuple, Dict, List, cast

from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from store.top_coactivation import top_coactivation
from store.latent_stats import latent_stats
from eval.faithfulness import evaluate_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from circuit.circuit_logger import CircuitLogger


class AllSparseExpansion(DiscoveryMethod):
    """
    All-kinds variable-depth co-activation expansion with no passthrough stage.

    Sparse all-kinds part (co-activation expansion)
    -----------------------------------------------
    Starting from any seed latent, the algorithm expands through the pre-computed
    co-activation graph for ``len(coact_depth)`` levels, retaining attn, mlp, and
    resid kind latents. No kind-based filtering is applied.

      Depth 0  - the seed latent (role="seed").
      Depth 1  - top ``coact_depth[0]`` neighbors across all kinds (role="hop1").
      Depth 2  - top ``coact_depth[1]`` neighbors across all kinds (role="hop2").
      ...and so on for each entry in coact_depth.

    Passthrough stage
    -----------------
    There is no passthrough stage for this method. Every node must come from
    co-activation expansion and therefore participates in the sparse graph.
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        coact_depth: Optional[List[int]] = None,
        min_faithfulness: Optional[float] = None,
        min_active_count: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        probe_batch_size: Optional[int] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.all_sparse_expansion
        if coact_depth is not None:
            self.coact_depth = list(coact_depth)
        elif cfg.coact_depth is not None:
            self.coact_depth = list(cfg.coact_depth)
        else:
            self.coact_depth = [32, 32]
        self.min_faithfulness = min_faithfulness or cast(
            float, config.discovery.min_faithfulness or 0.3
        )
        self.min_active_count = min_active_count or cast(
            int, config.discovery.min_active_count or 50
        )
        self.pruning_threshold = (
            pruning_threshold
            if pruning_threshold is not None
            else cast(float, cfg.pruning_threshold or 0.0)
        )
        self.probe_batch_size = probe_batch_size or cast(
            int, config.discovery.probe_batch_size or 16
        )

    def _expand_all_neighbors(
        self,
        comp_idx: int,
        latent_idx: int,
        limit: int,
        exclude: Set[Tuple[int, str, int]],
    ):
        """
        Yield (neighbor_comp_idx, neighbor_latent_idx, weight) for top ``limit``
        co-activation neighbors of (comp_idx, latent_idx) that are:
          1. above min_active_count
          2. not already in ``exclude``
        """
        d_sae = self.sae_bank.d_sae
        indices = top_coactivation.top_indices[comp_idx, latent_idx]
        values = top_coactivation.top_values[comp_idx, latent_idx]

        n_yielded = 0
        for g_idx, w in zip(indices.tolist(), values.tolist()):
            if n_yielded >= limit:
                break
            n_comp = int(g_idx) // d_sae
            n_lat = int(g_idx) % d_sae
            n_kind = self.sae_bank.kinds[n_comp % len(self.sae_bank.kinds)]
            n_layer = n_comp // len(self.sae_bank.kinds)
            key = (n_layer, n_kind, n_lat)
            if key in exclude:
                continue
            if latent_stats.active_count[n_comp, n_lat] < self.min_active_count:
                continue
            n_yielded += 1
            yield n_comp, n_lat, float(w)

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """Build an all-kinds sparse circuit from the seed feature."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "all_sparse_expansion")
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
        seed_layer = seed_comp_idx // len(self.sae_bank.kinds)
        seed_kind = self.sae_bank.kinds[seed_comp_idx % len(self.sae_bank.kinds)]

        circuit = Circuit(name=f"AllSparse_S{seed_comp_idx}_{seed_latent_idx}")

        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts found)")
            return None

        logger.header(
            seed_layer,
            seed_kind,
            seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        seed_key: Tuple[int, str, int] = (seed_layer, seed_kind, seed_latent_idx)
        seed_node = CircuitNode(
            metadata={
                "layer_idx": seed_layer,
                "latent_idx": seed_latent_idx,
                "kind": seed_kind,
                "role": "seed",
            }
        )
        circuit.add_node(seed_node)
        node_id_map: Dict[Tuple[int, str, int], str] = {seed_key: seed_node.uuid}
        in_circuit: Set[Tuple[int, str, int]] = {seed_key}

        frontier: List[Tuple[int, int, Tuple[int, str, int]]] = [
            (seed_comp_idx, seed_latent_idx, seed_key)
        ]

        for depth_idx, n_coacts in enumerate(self.coact_depth):
            role = f"hop{depth_idx + 1}"
            next_frontier: List[Tuple[int, int, Tuple[int, str, int]]] = []
            n_added = 0

            for parent_comp, parent_lat, parent_key in frontier:
                for n_comp, n_lat, w in self._expand_all_neighbors(
                    parent_comp, parent_lat, n_coacts, exclude=in_circuit
                ):
                    n_layer = n_comp // len(self.sae_bank.kinds)
                    n_kind = self.sae_bank.kinds[n_comp % len(self.sae_bank.kinds)]
                    key = (n_layer, n_kind, n_lat)

                    node = CircuitNode(
                        metadata={
                            "layer_idx": n_layer,
                            "latent_idx": n_lat,
                            "kind": n_kind,
                            "role": role,
                        }
                    )
                    circuit.add_node(node)
                    node_id_map[key] = node.uuid
                    in_circuit.add(key)
                    circuit.add_edge(node_id_map[parent_key], node.uuid, weight=w)
                    next_frontier.append((n_comp, n_lat, key))
                    n_added += 1

            logger.stage(
                f"depth-{depth_idx + 1} all-kinds expansion",
                len(circuit.nodes),
                len(circuit.edges),
                note=f"{len(frontier)} nodes expanded, {n_added} new nodes added",
            )
            frontier = next_frontier

        if len(circuit.nodes) <= 1:
            logger.reject("no neighbors found (all below activity filter or empty)")
            return None

        n_passthrough = 0
        logger.stage(
            "no passthrough stage",
            len(circuit.nodes),
            len(circuit.edges),
            note="all_sparse_expansion does not add passthrough nodes",
        )

        if self.pruning_threshold > 0:
            n_before = len(circuit.nodes)
            prune_non_minimal_nodes(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                probe_data.pos_tokens,
                pos_argmax=probe_data.pos_argmax,
                threshold=self.pruning_threshold,
            )
            logger.stage(
                "after pruning",
                len(circuit.nodes),
                len(circuit.edges),
                note=f"removed {n_before - len(circuit.nodes)} nodes",
            )

        _was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            final_f = evaluate_faithfulness(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                probe_data.pos_tokens,
                pos_argmax=probe_data.pos_argmax,
            )
            final_s = evaluate_sufficiency(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                probe_data.pos_tokens,
                probe_data.target_tokens,
                pos_argmax=probe_data.pos_argmax,
            )
            final_c = evaluate_completeness(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                probe_data.pos_tokens,
                pos_argmax=probe_data.pos_argmax,
            )
        finally:
            if _was_compiled:
                self.inference.enable_compile()

        logger.eval(final_f, final_s, final_c)

        if final_f < self.min_faithfulness:
            logger.reject(
                f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}"
            )
            return None

        circuit.metadata.update(
            {
                "faithfulness": final_f,
                "sufficiency": final_s,
                "completeness": final_c,
                "seed_comp": seed_comp_idx,
                "seed_latent": seed_latent_idx,
                "n_nodes": len(circuit.nodes),
                "n_edges": len(circuit.edges),
                "discovery_method": "all_sparse_expansion",
                "coact_depth": self.coact_depth,
                "n_passthrough": n_passthrough,
            }
        )
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit
