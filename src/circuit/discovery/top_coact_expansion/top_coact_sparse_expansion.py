import torch
from typing import Optional, Any, Set, Tuple, Dict, List, cast

from ..base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from store.top_coactivation import top_coactivation
from store.latent_stats import latent_stats
from eval.faithfulness import evaluate_faithfulness, evaluate_kind_local_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from circuit.sae_graph import SAEGraphInstrument
from circuit.feature_id import FeatureID
from circuit.circuit_logger import CircuitLogger


class TopCoactSparseExpansion(DiscoveryMethod):
    """
    Base class for targeted variable-depth top-coactivation sparse expansion algorithms.
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        target_kinds: Tuple[str, ...],
        passthrough_kinds: Tuple[str, ...],
        method_name: str,
        config_node: Any,
        coact_depth: Optional[List[int]] = None,
        min_faithfulness: Optional[float] = None,
        min_active_count: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        probe_batch_size: Optional[int] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        self.target_kinds = target_kinds
        self.passthrough_kinds = passthrough_kinds
        self.method_name = method_name

        # Support for monkeypatching in tests
        self.top_coactivation = top_coactivation
        self.latent_stats = latent_stats

        if coact_depth is not None:
            self.coact_depth = list(coact_depth)
        elif config_node.coact_depth is not None:
            self.coact_depth = list(config_node.coact_depth)
        else:
            self.coact_depth = [32, 32]

        self.min_faithfulness = (
            min_faithfulness
            if min_faithfulness is not None
            else cast(float, config.discovery.min_faithfulness or 0.3)
        )
        self.min_active_count = (
            min_active_count
            if min_active_count is not None
            else cast(int, config.discovery.min_active_count or 50)
        )
        self.pruning_threshold = (
            pruning_threshold
            if pruning_threshold is not None
            else cast(float, config_node.pruning_threshold or 0.0)
        )
        self.probe_batch_size = probe_batch_size or cast(
            int, config.discovery.probe_batch_size or 16
        )

    def _expand_neighbors(
        self,
        fid: FeatureID,
        limit: int,
        exclude: Set[FeatureID],
    ):
        """
        Yield (neighbor_fid, weight) for the top
        ``limit`` co-activation neighbors of fid.
        """
        d_sae = self.sae_bank.d_sae
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        
        comp_idx, latent_idx = fid.to_component_id(n_kinds, kinds)
        indices = self.top_coactivation.top_indices[comp_idx, latent_idx]
        values  = self.top_coactivation.top_values[comp_idx, latent_idx]

        n_yielded = 0
        for g_idx, w in zip(indices.tolist(), values.tolist()):
            if n_yielded >= limit:
                break
            
            nfid = FeatureID.from_global_id(int(g_idx), n_kinds, d_sae, kinds)
            if nfid.kind not in self.target_kinds:
                continue

            if nfid in exclude:
                continue
            
            c_idx, l_idx = nfid.to_component_id(n_kinds, kinds)
            if self.latent_stats.active_count[c_idx, l_idx] < self.min_active_count:
                continue

            n_yielded += 1
            yield nfid, float(w)

    def _capture_passthrough_nodes(
        self, probe_tokens: torch.Tensor
    ) -> Set[FeatureID]:
        """
        Run a no-grad forward pass and collect FeatureIDs for components in self.passthrough_kinds.
        """
        if not self.passthrough_kinds:
            return set()

        instrument = SAEGraphInstrument(self.sae_bank)
        _was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            with torch.no_grad():
                self.inference.forward(
                    probe_tokens,
                    num_gen=1,
                    tokenize_final=False,
                    return_activations=False,
                    all_logits=False,
                    patcher=instrument,
                )
        finally:
            if _was_compiled:
                self.inference.enable_compile()

        passthrough: Set[FeatureID] = set()
        for (layer, kind), steps in instrument.graph.activations.items():
            if kind not in self.passthrough_kinds:
                continue
            for _, _, top_indices in steps:
                for latent_idx in top_indices.flatten().tolist():
                    passthrough.add(FeatureID(layer, kind, int(latent_idx)))

        return passthrough

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """Entry point for discovery."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, self.method_name)
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
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        seed_layer, seed_kind_idx = divmod(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]
        seed_fid = FeatureID(seed_layer, seed_kind, seed_latent_idx)

        if seed_kind not in self.target_kinds:
            # Cancel the log if it's a trivial mismatch so we don't spam the discovery_logs dir
            logger.cancel()
            logger.reject(
                f"seed kind '{seed_kind}' is not in {self.target_kinds}; "
                f"{self.__class__.__name__} requires a target kind seed"
            )
            return None

        circuit = Circuit(name=f"{self.__class__.__name__}_S{seed_comp_idx}_{seed_latent_idx}")

        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        logger.stage("probe dataset construction", 0, 0, note=f"n_pos={probe_data.pos_tokens.shape[0]}")
        
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts found)")
            return None

        probe_tokens = probe_data.pos_tokens[: self.probe_batch_size]

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        # Depth 0: seed
        seed_node = CircuitNode(metadata={
            "feature_id": seed_fid,
            "role":       "seed",
        })
        circuit.add_node(seed_node)
        node_id_map: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}
        in_circuit: Set[FeatureID] = {seed_fid}
        logger.stage("seed setup", 1, 0)

        # BFS expansion
        frontier: List[FeatureID] = [seed_fid]

        for depth_idx, n_coacts in enumerate(self.coact_depth):
            role = f"hop{depth_idx + 1}"
            next_frontier: List[FeatureID] = []
            n_added = 0

            for parent_fid in frontier:
                for nfid, w in self._expand_neighbors(
                    parent_fid, n_coacts, exclude=in_circuit
                ):
                    node = CircuitNode(metadata={
                        "feature_id":  nfid,
                        "role":       role,
                    })
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

        if len(circuit.nodes) <= 1:
            logger.reject("no target neighbors found (all below activity filter or empty)")
            return None

        # Passthrough capture
        passthrough_set = self._capture_passthrough_nodes(probe_tokens)
        n_passthrough = 0

        for pfid in sorted(list(passthrough_set), key=lambda x: (x.layer, x.kind, x.index)):
            if pfid in node_id_map:
                continue
            node = CircuitNode(metadata={
                "feature_id":  pfid,
                "role":       "passthrough",
            })
            circuit.add_node(node)
            node_id_map[pfid] = node.uuid
            n_passthrough += 1

        if self.passthrough_kinds:
            logger.stage(
                f"{'/'.join(self.passthrough_kinds)} passthrough", 
                len(circuit.nodes), len(circuit.edges),
                note=f"{n_passthrough} passthrough nodes added",
            )

        # Minimality pruning
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

        # Evaluation
        _was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            final_f_global = evaluate_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
            )
            logger.stage("eval: global faithfulness", len(circuit.nodes), len(circuit.edges))

            final_f_local = evaluate_kind_local_faithfulness(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                probe_data.pos_tokens,
                target_kinds=self.target_kinds,
                pos_argmax=probe_data.pos_argmax,
            )
            logger.stage("eval: local faithfulness", len(circuit.nodes), len(circuit.edges))

            final_f = final_f_local
            final_s = evaluate_sufficiency(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, probe_data.target_tokens,
                pos_argmax=probe_data.pos_argmax,
            )
            logger.stage("eval: sufficiency", len(circuit.nodes), len(circuit.edges))

            final_c = evaluate_completeness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
            )
            logger.stage("eval: completeness", len(circuit.nodes), len(circuit.edges))
        finally:
            if _was_compiled:
                self.inference.enable_compile()

        logger.eval(final_f, final_s, final_c)
        logger.note(f"EVAL_GLOBAL      faithfulness={final_f_global:.4f}")
        logger.note(f"EVAL_KIND_LOCAL  target_kinds={self.target_kinds}  faithfulness={final_f_local:.4f}")

        if final_f < self.min_faithfulness:
            logger.reject(
                f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}"
            )
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
            "discovery_method":   self.method_name,
            "coact_depth":        self.coact_depth,
            "n_passthrough":      n_passthrough,
        })
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit
