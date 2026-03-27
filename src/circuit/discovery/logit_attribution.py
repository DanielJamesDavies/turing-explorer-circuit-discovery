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
from circuit.attribution import compute_logit_attribution, compute_feature_attribution
from circuit.feature_id import FeatureID
from circuit.circuit_logger import CircuitLogger
from pipeline.component_index import component_idx, split_component_idx


class LogitAttribution(DiscoveryMethod):
    """
    Two-pass gradient-based circuit discovery.

    Pass 1 — Logit attribution:
        A single backward pass from the target token logit (at the seed's peak position)
        to all leaf anchors identifies every feature that causally influences the model's
        prediction on probe sequences.  Score = activation * gradient.

    Pass 2 — Feature-to-feature attribution:
        For each included feature (node B), a second backward pass from B's connected
        activation (not the detached leaf) to upstream leaf anchors gives pairwise causal
        edge weights between circuit nodes.  The co-activation store is used to restrict
        the candidate pool so we don't evaluate all O(N²) pairs.

    The seed is always included regardless of its logit attribution score, since it
    defines the probe context.
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        min_faithfulness: Optional[float] = None,
        logit_threshold: Optional[float] = None,
        edge_threshold: Optional[float] = None,
        max_neighbors: Optional[int] = None,
        min_active_count: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        probe_batch_size: Optional[int] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.logit_attribution
        self.min_faithfulness = (
            min_faithfulness
            if min_faithfulness is not None
            else cast(float, config.discovery.min_faithfulness or 0.3)
        )
        self.logit_threshold = logit_threshold if logit_threshold is not None else cast(float, cfg.logit_threshold or 0.001)
        self.edge_threshold = edge_threshold if edge_threshold is not None else cast(float, cfg.edge_threshold or 0.001)
        self.max_neighbors = max_neighbors or cast(int, cfg.max_neighbors or config.discovery.max_neighbors or 32)
        self.min_active_count = min_active_count if min_active_count is not None else cast(int, config.discovery.min_active_count or 50)
        self.pruning_threshold = pruning_threshold if pruning_threshold is not None else cast(float, cfg.pruning_threshold or 0.0)
        self.probe_batch_size = probe_batch_size or cast(int, config.discovery.probe_batch_size or 8)

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """Two-pass logit attribution discovery."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "logit_attribution")
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
        circuit = Circuit(name=f"LogitAttr_S{seed_comp_idx}_{seed_latent_idx}")

        # 1. Probe dataset
        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts found)")
            return None

        n_probe = min(self.probe_batch_size, probe_data.pos_tokens.shape[0])
        probe_tokens = probe_data.pos_tokens[:n_probe]
        probe_argmax = probe_data.pos_argmax[:n_probe]
        probe_targets = probe_data.target_tokens[:n_probe]

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
        logger.note(f"probe batch: {n_probe} sequences")

        # 2. Instrumented forward — need all_logits=True so we can index any position
        self.inference.disable_compile()
        instrument = SAEGraphInstrument(self.sae_bank)
        _, logits, _ = self.inference.forward(
            probe_tokens,
            patcher=instrument,
            grad_enabled=True,
            return_activations=False,
            all_logits=True,
            tokenize_final=False,
        )
        self.inference.enable_compile()

        if logits is None:
            del instrument
            logger.reject("forward pass returned no logits")
            return None

        logger.note("forward pass complete")

        # 3. Pass 1 — logit attribution: which features matter for the prediction?
        logit_attrs = compute_logit_attribution(
            instrument.graph,
            logits=logits,
            pos_argmax=probe_argmax,
            target_tokens=probe_targets,
        )

        # 4. Build node set: seed always included; others if above logit_threshold
        included: Dict[FeatureID, float] = {}
        included[seed_fid] = logit_attrs.get(seed_fid, 0.0)

        d_sae = self.sae_bank.d_sae
        neighbor_globals = top_coactivation.top_indices[seed_comp_idx, seed_latent_idx]
        neighbor_weights = top_coactivation.top_values[seed_comp_idx, seed_latent_idx]

        n_considered = 0
        n_passed = 0
        for g_idx, weight in zip(neighbor_globals.tolist(), neighbor_weights.tolist()):
            if weight <= 0 or n_considered >= self.max_neighbors:
                break
            
            fid = FeatureID.from_global_id(int(g_idx), n_kinds, d_sae, kinds)
            c_idx, l_idx = fid.to_component_id(n_kinds, kinds)
            
            if latent_stats.active_count[c_idx, l_idx] < self.min_active_count:
                continue
            
            score = logit_attrs.get(fid, 0.0)
            if abs(score) >= self.logit_threshold:
                included[fid] = score
                n_passed += 1
            n_considered += 1

        logger.stage(
            "logit attribution", len(included), 0,
            note=(
                f"{n_considered} candidates considered, {n_passed} passed "
                f"threshold={self.logit_threshold}"
            ),
        )

        if len(included) <= 1:
            del instrument
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.reject(
                f"only seed passed logit threshold ({self.logit_threshold}); "
                "no circuit neighbours found"
            )
            return None

        # 5. Add all included nodes to the circuit
        kind_order = ["attn", "mlp", "resid"]
        node_id_map: Dict[FeatureID, str] = {}

        for fid, logit_score in included.items():
            role = "seed" if fid == seed_fid else "attributed"
            node = CircuitNode(metadata={
                "feature_id": fid,
                "role": role,
                "logit_attribution": logit_score,
            })
            circuit.add_node(node)
            node_id_map[fid] = node.uuid

        # 6. Pass 2 — feature-to-feature edges
        included_fids = list(node_id_map.keys())

        for b_fid in included_fids:
            upstream_circuit = [
                fid for fid in included_fids
                if fid.layer < b_fid.layer or (fid.layer == b_fid.layer and kind_order.index(fid.kind) < kind_order.index(b_fid.kind))
            ]
            if not upstream_circuit:
                continue

            b_comp_idx, b_latent = b_fid.to_component_id(n_kinds, kinds)
            b_neighbors_globals = top_coactivation.top_indices[b_comp_idx, b_latent]
            b_neighbors_weights = top_coactivation.top_values[b_comp_idx, b_latent]

            coact_upstream: List[FeatureID] = []
            for g_idx, weight in zip(b_neighbors_globals.tolist(), b_neighbors_weights.tolist()):
                if weight <= 0:
                    break
                fid = FeatureID.from_global_id(int(g_idx), n_kinds, d_sae, kinds)
                if fid in upstream_circuit:
                    coact_upstream.append(fid)

            candidates = coact_upstream if coact_upstream else upstream_circuit

            edge_attrs = compute_feature_attribution(
                instrument.graph,
                target_layer=b_fid.layer,
                target_kind=b_fid.kind,
                target_latent_idx=b_fid.index,
                pos_argmax=probe_argmax,
                candidate_nodes=candidates,
            )

            for a_fid, edge_score in edge_attrs.items():
                if abs(edge_score) >= self.edge_threshold:
                    if a_fid in node_id_map:
                        circuit.add_edge(
                            node_id_map[a_fid],
                            node_id_map[b_fid],
                            weight=edge_score,
                        )

        logger.stage(
            "edge attribution", len(circuit.nodes), len(circuit.edges),
            note=f"threshold={self.edge_threshold}",
        )

        # 7. Free grad graph
        del instrument
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(circuit.nodes) <= 1:
            logger.reject("circuit collapsed to ≤1 node after edge pass")
            return None

        # 9. Optional minimality pruning (zero ablation)
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

        # 10. Final evaluation (zero ablation)
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
                "discovery_method": "logit_attribution",
                "logit_threshold": self.logit_threshold,
                "edge_threshold": self.edge_threshold,
            })
            logger.accept(len(circuit.nodes), len(circuit.edges))
            return circuit

        logger.reject(
            f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}"
        )
        return None
