import torch
from typing import Optional, Any, List, Set, Dict, cast

from .top_coact_sparse_expansion import TopCoactSparseExpansion
from config import config
from store.circuits import Circuit, CircuitNode
from circuit.feature_id import FeatureID
from circuit.circuit_logger import CircuitLogger
from circuit.sae_graph import SAEGraphInstrument
from circuit.attribution import compute_feature_gradient
from pipeline.component_index import split_component_idx
from eval.faithfulness import evaluate_faithfulness, evaluate_kind_local_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes


class HardNegativeCoactSparseExpansion(TopCoactSparseExpansion):
    """
    Expansion discovery method that identifies inhibitors by finding latents 
    unusually active in hard-negative contexts of the seed, validated by attribution.
    Evaluations are performed only after both activators and inhibitors are added.
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        coact_depth: Optional[List[int]] = None,
        neg_candidate_limit: Optional[int] = None,
        attribution_threshold: Optional[float] = None,
        min_faithfulness: Optional[float] = None,
        min_active_count: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        probe_batch_size: Optional[int] = None,
    ):
        cfg = config.discovery.hard_negative_coact_sparse_expansion
        super().__init__(
            inference=inference,
            sae_bank=sae_bank,
            avg_acts=avg_acts,
            probe_builder=probe_builder,
            target_kinds=("attn", "mlp", "resid"),
            passthrough_kinds=(),
            method_name="hard_negative_coact_sparse_expansion",
            config_node=cfg,
            coact_depth=coact_depth,
            min_faithfulness=min_faithfulness,
            min_active_count=min_active_count,
            pruning_threshold=pruning_threshold,
            probe_batch_size=probe_batch_size,
        )
        self.neg_candidate_limit = (
            neg_candidate_limit if neg_candidate_limit is not None else cfg.neg_candidate_limit
        )
        self.attribution_threshold = (
            attribution_threshold if attribution_threshold is not None else cfg.attribution_threshold
        )

    def _discover(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        logger: CircuitLogger,
    ) -> Optional[Circuit]:
        # --- 1. SETUP ---
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = divmod(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]
        seed_fid = FeatureID(seed_layer, seed_kind, seed_latent_idx)

        if seed_kind not in self.target_kinds:
            logger.cancel()
            logger.reject(f"seed kind '{seed_kind}' is not in {self.target_kinds}")
            return None

        circuit = Circuit(name=f"{self.__class__.__name__}_S{seed_comp_idx}_{seed_latent_idx}")

        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts found)")
            return None

        probe_tokens = probe_data.pos_tokens[: self.probe_batch_size]
        pos_argmax = probe_data.pos_argmax[: self.probe_batch_size]

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        # Depth 0: seed
        seed_node = CircuitNode(metadata={"feature_id": seed_fid, "role": "seed"})
        circuit.add_node(seed_node)
        node_id_map: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}
        in_circuit: Set[FeatureID] = {seed_fid}
        logger.stage("seed setup", 1, 0)

        # --- 2. ACTIVATOR EXPANSION (BFS) ---
        frontier: List[FeatureID] = [seed_fid]
        for depth_idx, n_coacts in enumerate(self.coact_depth):
            role = f"hop{depth_idx + 1}"
            next_frontier: List[FeatureID] = []
            n_added = 0

            for parent_fid in frontier:
                for nfid, w in self._expand_neighbors(parent_fid, n_coacts, exclude=in_circuit):
                    node = CircuitNode(metadata={"feature_id": nfid, "role": role})
                    circuit.add_node(node)
                    node_id_map[nfid] = node.uuid
                    in_circuit.add(nfid)
                    circuit.add_edge(node_id_map[parent_fid], node.uuid, weight=w)
                    next_frontier.append(nfid)
                    n_added += 1

            logger.stage(
                f"depth-{depth_idx + 1} targeted expansion",
                len(circuit.nodes), len(circuit.edges),
                note=f"{len(frontier)} nodes expanded, {n_added} new target nodes added",
            )
            frontier = next_frontier

        # --- 3. INHIBITOR SEARCH ---
        n_inhibitors_added = 0
        if probe_data.neg_tokens.shape[0] > 0:
            neg_tokens = probe_data.neg_tokens[: self.probe_batch_size]
            logger.note(f"Searching for inhibitors in {neg_tokens.shape[0]} negative sequences...")

            # Find candidates in NEGATIVE contexts
            neg_activations = self._collect_neg_activations(neg_tokens)
            
            # Select top candidates not already in activator backbone
            candidate_fids = sorted(
                [fid for fid, score in neg_activations.items() if fid not in in_circuit],
                key=lambda fid: neg_activations[fid],
                reverse=True
            )[: self.neg_candidate_limit]

            if candidate_fids:
                # Causal Validation (Gradient-based) on POSITIVE tokens
                instrument = SAEGraphInstrument(self.sae_bank)
                _was_compiled = self.inference._compiled
                self.inference.disable_compile()
                try:
                    self.inference.forward(probe_tokens, patcher=instrument, grad_enabled=True, return_activations=False)
                finally:
                    if _was_compiled:
                        self.inference.enable_compile()

                # Seed ID in circuit metadata for easier finding
                seed_node_uuid = node_id_map[seed_fid]

                # Compute gradients
                gradients = compute_feature_gradient(
                    instrument.graph,
                    target_layer=seed_layer,
                    target_kind=seed_kind,
                    target_latent_idx=seed_latent_idx,
                    pos_argmax=pos_argmax,
                    candidate_nodes=candidate_fids
                )

                for fid in candidate_fids:
                    grad = gradients.get(fid, 0.0)
                    if grad <= -self.attribution_threshold:
                        node = CircuitNode(metadata={
                            "feature_id": fid,
                            "role": "inhibitor",
                            "neg_act_score": neg_activations[fid],
                            "gradient": grad
                        })
                        circuit.add_node(node)
                        node_id_map[fid] = node.uuid
                        in_circuit.add(fid)
                        circuit.add_edge(node.uuid, seed_node_uuid, weight=grad)
                        n_inhibitors_added += 1

                logger.stage(
                    "inhibitor expansion", 
                    len(circuit.nodes), len(circuit.edges),
                    note=f"Added {n_inhibitors_added} inhibitors from {len(candidate_fids)} candidates"
                )
            else:
                logger.note("No inhibitor candidates found.")
        else:
            logger.note("No hard negative tokens found; skipping inhibitor search.")

        if len(circuit.nodes) <= 1:
            logger.reject("no target neighbors or inhibitors found")
            return None

        # --- 4. MINIMALITY PRUNING ---
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

        # --- 5. EVALUATION ---
        _was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            final_f_global = evaluate_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
            )
            final_f_local = evaluate_kind_local_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, target_kinds=self.target_kinds,
                pos_argmax=probe_data.pos_argmax,
            )
            final_f = final_f_local
            final_s = evaluate_sufficiency(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, probe_data.target_tokens,
                pos_argmax=probe_data.pos_argmax,
            )
            final_c = evaluate_completeness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
            )
        finally:
            if _was_compiled:
                self.inference.enable_compile()

        logger.eval(final_f, final_s, final_c)
        logger.note(f"EVAL_GLOBAL      faithfulness={final_f_global:.4f}")
        logger.note(f"EVAL_KIND_LOCAL  target_kinds={self.target_kinds}  faithfulness={final_f_local:.4f}")

        if final_f < self.min_faithfulness:
            logger.reject(f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}")
            return None

        circuit.metadata.update({
            "faithfulness":       final_f,
            "faithfulness_global": final_f_global,
            "faithfulness_kind_local": final_f_local,
            "kind_local_target_kinds": list(self.target_kinds),
            "sufficiency":        final_s,
            "completeness":       final_c,
            "seed_comp":          seed_comp_idx,
            "seed_latent":        seed_latent_idx,
            "n_nodes":            len(circuit.nodes),
            "n_edges":            len(circuit.edges),
            "n_inhibitors":       n_inhibitors_added,
            "discovery_method":   self.method_name,
            "coact_depth":        self.coact_depth,
        })
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit

    def _collect_neg_activations(self, neg_tokens: torch.Tensor) -> Dict[FeatureID, float]:
        """Runs a no-grad forward pass on neg_tokens to find active latents."""
        neg_activations: Dict[FeatureID, float] = {}
        
        def capture_hook(layer_idx: int, activations: tuple):
            kinds = self.sae_bank.kinds
            for k_idx, act in enumerate(activations):
                kind = kinds[k_idx]
                top_acts, top_indices = self.sae_bank.encode(act, kind, layer_idx)
                
                # Simple frequency/intensity accumulation
                active_mask = top_acts > 0
                if not active_mask.any():
                    continue
                    
                flat_indices = top_indices[active_mask]
                flat_acts = top_acts[active_mask]
                
                unique_idx, inverse_idx = flat_indices.unique(return_inverse=True)
                summed_acts = torch.zeros_like(unique_idx, dtype=torch.float32).scatter_add_(0, inverse_idx, flat_acts.float())
                
                for idx, total_act in zip(unique_idx.tolist(), summed_acts.tolist()):
                    fid = FeatureID(layer_idx, kind, int(idx))
                    neg_activations[fid] = neg_activations.get(fid, 0.0) + total_act

        _was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            with torch.no_grad():
                self.inference.forward(
                    neg_tokens,
                    activations_callback=capture_hook,
                    return_activations=False
                )
        finally:
            if _was_compiled:
                self.inference.enable_compile()
                
        return neg_activations
