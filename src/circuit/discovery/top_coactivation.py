import torch
from typing import Optional, Any, List, Tuple, Dict, cast
from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from store.top_coactivation import top_coactivation
from store.latent_stats import latent_stats
from eval.faithfulness import evaluate_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from circuit.sae_graph import SAEGraphInstrument
from circuit.attribution import compute_feature_attribution
from circuit.feature_id import FeatureID
from circuit.circuit_logger import CircuitLogger
from pipeline.component_index import component_idx, split_component_idx


class TopCoactivationDiscovery(DiscoveryMethod):
    """
    Discovers circuits by expanding the neighborhood of a seed latent
    using statistical co-activation data, followed by multi-hop causal attribution.
    """
    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        min_faithfulness: Optional[float] = None,
        max_neighbors: Optional[int] = None,
        max_hops: Optional[int] = None,
        min_active_count: Optional[int] = None,
        attribution_threshold: Optional[float] = None,
        pruning_threshold: Optional[float] = None,
        probe_batch_size: Optional[int] = None
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.top_coactivation
        self.min_faithfulness = (
            min_faithfulness
            if min_faithfulness is not None
            else cast(float, config.discovery.min_faithfulness or 0.3)
        )
        self.max_neighbors = max_neighbors or cast(int, cfg.max_neighbors or config.discovery.max_neighbors or 32)
        self.max_hops = max_hops or cast(int, cfg.max_hops or 2)
        self.min_active_count = min_active_count if min_active_count is not None else cast(int, config.discovery.min_active_count or 50)
        self.attribution_threshold = attribution_threshold if attribution_threshold is not None else cast(float, cfg.attribution_threshold or 0.01)
        self.pruning_threshold = pruning_threshold if pruning_threshold is not None else cast(float, cfg.pruning_threshold or 0.01)
        self.probe_batch_size = probe_batch_size or cast(int, config.discovery.probe_batch_size or 8)

        # Instance-level singletons for easier monkeypatching in tests
        self.top_coactivation = top_coactivation
        self.latent_stats = latent_stats

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """Executes the multi-hop causal attribution discovery episode."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "top_coactivation")
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
        circuit = Circuit(name=f"TopCoact_S{seed_comp_idx}_{seed_latent_idx}")

        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts found)")
            return None

        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, len(self.sae_bank.kinds))
        seed_kind = self.sae_bank.kinds[seed_kind_idx]
        seed_fid = FeatureID(seed_layer, seed_kind, seed_latent_idx)

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        n_probe = min(self.probe_batch_size, probe_data.pos_tokens.shape[0])
        probe_tokens = probe_data.pos_tokens[:n_probe]
        probe_argmax = probe_data.pos_argmax[:n_probe]
        logger.note(f"probe batch: {n_probe} sequences")

        self.inference.disable_compile()
        instrument = SAEGraphInstrument(self.sae_bank)
        _, _, _ = self.inference.forward(
            probe_tokens,
            patcher=instrument,
            grad_enabled=True,
            return_activations=False
        )
        self.inference.enable_compile()
        logger.note("forward pass complete")

        frontier: List[FeatureID] = [seed_fid]

        seed_node = CircuitNode(metadata={
            "feature_id": seed_fid,
            "role": "seed"
        })
        circuit.add_node(seed_node)
        node_id_map: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}

        visited: set[FeatureID] = set()
        kind_order = ["attn", "mlp", "resid"]
        d_sae = self.sae_bank.d_sae
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        hops_completed = 0

        # Upstream parent tracing (multi-hop)
        for hop in range(self.max_hops):
            hops_completed = hop + 1
            new_frontier = []

            for target_fid in frontier:
                if target_fid in visited:
                    continue
                visited.add(target_fid)

                comp_idx, latent_idx = target_fid.to_component_id(n_kinds, kinds)
                neighbor_globals = self.top_coactivation.top_indices[comp_idx, latent_idx]
                neighbor_weights = self.top_coactivation.top_values[comp_idx, latent_idx]

                candidate_nodes: List[FeatureID] = []

                for g_idx, weight in zip(neighbor_globals.tolist(), neighbor_weights.tolist()):
                    if weight <= 0 or len(candidate_nodes) >= self.max_neighbors:
                        continue
                    
                    nfid = FeatureID.from_global_id(int(g_idx), n_kinds, d_sae, kinds)
                    c_idx, l_idx = nfid.to_component_id(n_kinds, kinds)
                    
                    if self.latent_stats.active_count[c_idx, l_idx] < self.min_active_count:
                        continue

                    # Restrict to upstream only (earlier layer or earlier kind in same layer)
                    if nfid.layer > target_fid.layer:
                        continue
                    if nfid.layer == target_fid.layer and kind_order.index(nfid.kind) >= kind_order.index(target_fid.kind):
                        continue

                    candidate_nodes.append(nfid)

                if not candidate_nodes:
                    continue

                attributions = compute_feature_attribution(
                    instrument.graph,
                    target_layer=target_fid.layer,
                    target_kind=target_fid.kind,
                    target_latent_idx=target_fid.index,
                    pos_argmax=probe_argmax,
                    candidate_nodes=candidate_nodes
                )

                target_uuid = node_id_map[target_fid]

                for fid in candidate_nodes:
                    attr_score = attributions.get(fid, 0.0)

                    if abs(attr_score) >= self.attribution_threshold:
                        if fid not in node_id_map:
                            node = CircuitNode(metadata={
                                "feature_id": fid,
                                "role": "parent"
                            })
                            circuit.add_node(node)
                            node_id_map[fid] = node.uuid
                            new_frontier.append(fid)

                        circuit.add_edge(node_id_map[fid], target_uuid, weight=attr_score)

            logger.stage(
                f"upstream hop {hops_completed}", len(circuit.nodes), len(circuit.edges),
                note=f"{len(new_frontier)} new nodes",
            )
            frontier = new_frontier
            if not frontier:
                break

        # Downstream child tracing
        downstream_candidate_set: set[FeatureID] = set()
        for source_fid in list(node_id_map.keys()):
            comp_idx, latent_idx = source_fid.to_component_id(n_kinds, kinds)
            neighbor_globals = self.top_coactivation.top_indices[comp_idx, latent_idx]
            neighbor_weights = self.top_coactivation.top_values[comp_idx, latent_idx]

            n_downstream = 0
            for g_idx, weight in zip(neighbor_globals.tolist(), neighbor_weights.tolist()):
                if weight <= 0 or n_downstream >= self.max_neighbors:
                    continue
                
                nfid = FeatureID.from_global_id(int(g_idx), n_kinds, d_sae, kinds)
                c_idx, l_idx = nfid.to_component_id(n_kinds, kinds)
                
                if self.latent_stats.active_count[c_idx, l_idx] < self.min_active_count:
                    continue

                if nfid.layer < source_fid.layer:
                    continue
                if nfid.layer == source_fid.layer and kind_order.index(nfid.kind) <= kind_order.index(source_fid.kind):
                    continue

                downstream_candidate_set.add(nfid)
                n_downstream += 1

        circuit_node_fids = list(node_id_map.keys())
        n_downstream_added = 0
        for child_fid in downstream_candidate_set:
            upstream_sources = [
                fid for fid in circuit_node_fids
                if fid.layer < child_fid.layer or (
                    fid.layer == child_fid.layer and kind_order.index(fid.kind) < kind_order.index(child_fid.kind)
                )
            ]
            if not upstream_sources:
                continue

            child_attr = compute_feature_attribution(
                instrument.graph,
                target_layer=child_fid.layer,
                target_kind=child_fid.kind,
                target_latent_idx=child_fid.index,
                pos_argmax=probe_argmax,
                candidate_nodes=upstream_sources
            )

            for source_fid in upstream_sources:
                source_score = child_attr.get(source_fid, 0.0)
                if abs(source_score) >= self.attribution_threshold:
                    if child_fid not in node_id_map:
                        node = CircuitNode(metadata={
                            "feature_id": child_fid,
                            "role": "child"
                        })
                        circuit.add_node(node)
                        node_id_map[child_fid] = node.uuid
                        n_downstream_added += 1
                    circuit.add_edge(
                        node_id_map[source_fid],
                        node_id_map[child_fid],
                        weight=source_score
                    )

        logger.stage(
            "downstream expansion", len(circuit.nodes), len(circuit.edges),
            note=f"{n_downstream_added} child nodes added from {len(downstream_candidate_set)} candidates",
        )

        del instrument
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(circuit.nodes) <= 1:
            logger.reject("circuit has ≤1 node after attribution passes")
            return None

        if self.pruning_threshold > 0:
            n_before = len(circuit.nodes)
            prune_non_minimal_nodes(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                probe_data.pos_tokens,
                pos_argmax=probe_data.pos_argmax,
                threshold=self.pruning_threshold
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
                "hops": hops_completed,
                "n_nodes": len(circuit.nodes),
                "n_edges": len(circuit.edges),
                "discovery_method": "top_coactivation_attribution_bidirectional"
            })
            logger.accept(len(circuit.nodes), len(circuit.edges))
            return circuit

        logger.reject(
            f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}"
        )
        return None
