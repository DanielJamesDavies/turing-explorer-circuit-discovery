import torch
from typing import Optional, Any, cast
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
from pipeline.component_index import split_component_idx

class CoactivationStatistical(DiscoveryMethod):
    """
    Baseline circuit discovery using raw co-activation statistics.

    No gradient computation required. For each seed, the circuit is grown by
    including co-activation neighbors whose statistical weight exceeds a threshold.
    Edge weights are the co-activation scores directly.

    This is intentionally simple and fast — it serves as a baseline to validate
    the pipeline end-to-end and to compare against gradient-based methods.
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        min_faithfulness: Optional[float] = None,
        coactivation_threshold: Optional[float] = None,
        max_neighbors: Optional[int] = None,
        min_active_count: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.coactivation_statistical
        self.min_faithfulness = min_faithfulness or cast(float, config.discovery.min_faithfulness or 0.3)
        self.coactivation_threshold = coactivation_threshold or cast(float, cfg.coactivation_threshold or 0.1)
        self.max_neighbors = max_neighbors or cast(int, cfg.max_neighbors or config.discovery.max_neighbors or 32)
        self.min_active_count = min_active_count or cast(int, config.discovery.min_active_count or 50)
        self.pruning_threshold = pruning_threshold if pruning_threshold is not None else cast(float, cfg.pruning_threshold or 0.0)

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """Expands seed to co-activation neighbors above threshold, then evaluates faithfulness."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "coactivation_statistical")
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
        circuit = Circuit(name=f"CoactStat_S{seed_comp_idx}_{seed_latent_idx}")

        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts found)")
            return None

        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, len(self.sae_bank.kinds))
        seed_kind = self.sae_bank.kinds[seed_kind_idx]
        seed_key = (seed_layer, seed_kind, seed_latent_idx)

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        seed_node = CircuitNode(metadata={
            "layer_idx": seed_layer,
            "latent_idx": seed_latent_idx,
            "kind": seed_kind,
            "role": "seed",
        })
        circuit.add_node(seed_node)
        node_id_map = {seed_key: seed_node.uuid}

        kind_order = ["attn", "mlp", "resid"]
        d_sae = self.sae_bank.d_sae

        neighbor_globals = top_coactivation.top_indices[seed_comp_idx, seed_latent_idx]
        neighbor_weights = top_coactivation.top_values[seed_comp_idx, seed_latent_idx]

        n_added = 0
        n_skipped_active = 0
        n_skipped_threshold = 0

        for g_idx, weight in zip(neighbor_globals.tolist(), neighbor_weights.tolist()):
            if weight < self.coactivation_threshold:
                n_skipped_threshold += 1
                continue
            if n_added >= self.max_neighbors:
                break
            c_idx = int(g_idx) // d_sae
            l_idx = int(g_idx) % d_sae
            if latent_stats.active_count[c_idx, l_idx] < self.min_active_count:
                n_skipped_active += 1
                continue

            n_layer, n_kind_idx = split_component_idx(c_idx, len(self.sae_bank.kinds))
            n_kind = self.sae_bank.kinds[n_kind_idx]
            key = (n_layer, n_kind, l_idx)

            if key not in node_id_map:
                node = CircuitNode(metadata={
                    "layer_idx": n_layer,
                    "latent_idx": l_idx,
                    "kind": n_kind,
                    "role": "neighbor",
                })
                circuit.add_node(node)
                node_id_map[key] = node.uuid

            is_upstream = (
                n_layer < seed_layer or
                (n_layer == seed_layer and kind_order.index(n_kind) < kind_order.index(seed_kind))
            )
            if is_upstream:
                circuit.add_edge(node_id_map[key], node_id_map[seed_key], weight=float(weight))
            else:
                circuit.add_edge(node_id_map[seed_key], node_id_map[key], weight=float(weight))

            n_added += 1

        logger.stage(
            "hop-1 expansion", len(circuit.nodes), len(circuit.edges),
            note=(
                f"{n_added} added, {n_skipped_threshold} below threshold "
                f"({self.coactivation_threshold}), {n_skipped_active} below active_count"
            ),
        )

        if len(circuit.nodes) <= 1:
            logger.reject("no neighbors passed co-activation threshold / activity filter")
            return None

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

        if final_f >= self.min_faithfulness:
            circuit.metadata.update({
                "faithfulness": final_f,
                "sufficiency": final_s,
                "completeness": final_c,
                "seed_comp": seed_comp_idx,
                "seed_latent": seed_latent_idx,
                "n_nodes": len(circuit.nodes),
                "n_edges": len(circuit.edges),
                "discovery_method": "coactivation_statistical",
                "coactivation_threshold": self.coactivation_threshold,
            })
            logger.accept(len(circuit.nodes), len(circuit.edges))
            return circuit

        logger.reject(
            f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}"
        )
        return None
