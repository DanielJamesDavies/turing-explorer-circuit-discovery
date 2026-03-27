import torch
from typing import Optional, Any, Dict, List, Set, cast

from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from circuit.feature_id import FeatureID
from circuit.circuit_logger import CircuitLogger
from circuit.sae_graph import SAEGraphInstrument
from circuit.attribution import compute_feature_attribution
from pipeline.component_index import split_component_idx
from eval.faithfulness import evaluate_faithfulness, evaluate_kind_local_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes


class DifferentialActivation(DiscoveryMethod):
    """
    Discovers circuits by finding latents that are differentially active between
    positive contexts (seed fires) and hard-negative contexts (seed expected but absent).

    Phase 1 — Differential scan:
        Run forward passes on pos and neg tokens, compute per-latent mean activation
        in each, then rank by delta = mean(pos) - mean(neg).
        Top positive deltas → activator candidates.
        Top negative deltas → inhibitor candidates.

    Phase 2 — Causal edge construction:
        Run an instrumented (grad-enabled) forward on pos tokens, then use
        compute_feature_attribution to build directed causal edges between
        the seed and each candidate.

    Phase 3 — Pruning and evaluation.
    """

    method_name = "differential_activation"

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        n_activator_candidates: Optional[int] = None,
        n_inhibitor_candidates: Optional[int] = None,
        attribution_threshold: Optional[float] = None,
        pruning_threshold: Optional[float] = None,
        min_faithfulness: Optional[float] = None,
        probe_batch_size: Optional[int] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.differential_activation
        self.n_activator_candidates = (
            n_activator_candidates if n_activator_candidates is not None
            else cfg.n_activator_candidates
        )
        self.n_inhibitor_candidates = (
            n_inhibitor_candidates if n_inhibitor_candidates is not None
            else cfg.n_inhibitor_candidates
        )
        self.attribution_threshold = (
            attribution_threshold if attribution_threshold is not None
            else cfg.attribution_threshold
        )
        self.pruning_threshold = (
            pruning_threshold if pruning_threshold is not None
            else cfg.pruning_threshold
        )
        self.min_faithfulness = (
            min_faithfulness if min_faithfulness is not None
            else cast(float, config.discovery.min_faithfulness or 0.2)
        )
        self.probe_batch_size = probe_batch_size or cast(int, config.discovery.probe_batch_size or 16)

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
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
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = kinds[seed_kind_idx]
        seed_fid = FeatureID(seed_layer, seed_kind, seed_latent_idx)

        circuit = Circuit(name=f"DiffAct_S{seed_comp_idx}_{seed_latent_idx}")

        # --- Build probe dataset ---
        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts)")
            return None

        pos_tokens = probe_data.pos_tokens[: self.probe_batch_size]
        neg_tokens = probe_data.neg_tokens[: self.probe_batch_size]
        pos_argmax = probe_data.pos_argmax[: self.probe_batch_size]

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        if neg_tokens.shape[0] == 0:
            logger.reject("no hard-negative sequences available for differential scan")
            return None

        # --- Seed node ---
        seed_node = CircuitNode(metadata={"feature_id": seed_fid, "role": "seed"})
        circuit.add_node(seed_node)
        node_id_map: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}
        logger.stage("seed setup", 1, 0)

        # ===================================================================
        # Phase 1: Differential Activation Scan
        # ===================================================================
        pos_acts = self._collect_activations(pos_tokens)
        neg_acts = self._collect_activations(neg_tokens)
        logger.note(f"Collected activations: {len(pos_acts)} pos latents, {len(neg_acts)} neg latents")

        # Compute delta = mean(pos) - mean(neg) for every latent seen in either set.
        # Normalise by the number of token positions so batch size doesn't dominate.
        n_pos_tokens = pos_tokens.shape[0] * pos_tokens.shape[1]
        n_neg_tokens = neg_tokens.shape[0] * neg_tokens.shape[1]

        all_fids = set(pos_acts.keys()) | set(neg_acts.keys())
        deltas: Dict[FeatureID, float] = {}
        for fid in all_fids:
            if fid == seed_fid:
                continue
            p = pos_acts.get(fid, 0.0) / n_pos_tokens
            n = neg_acts.get(fid, 0.0) / n_neg_tokens
            deltas[fid] = p - n

        # Activators: highest positive delta (strongly present in pos, weak/absent in neg)
        sorted_by_delta = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)
        activator_candidates = [fid for fid, d in sorted_by_delta if d > 0][: self.n_activator_candidates]

        # Inhibitors: most negative delta (strongly present in neg, weak/absent in pos)
        inhibitor_candidates = [fid for fid, d in sorted_by_delta if d < 0]
        inhibitor_candidates.reverse()  # most negative first
        inhibitor_candidates = inhibitor_candidates[: self.n_inhibitor_candidates]

        all_candidates = activator_candidates + inhibitor_candidates
        logger.stage(
            "differential scan",
            1, 0,
            note=f"{len(activator_candidates)} activators, {len(inhibitor_candidates)} inhibitors from {len(deltas)} latents",
        )

        if not all_candidates:
            logger.reject("no differential candidates found")
            return None

        # ===================================================================
        # Phase 2: Causal Edge Construction via Attribution
        # ===================================================================
        instrument = SAEGraphInstrument(self.sae_bank)
        _was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            self.inference.forward(
                pos_tokens, patcher=instrument, grad_enabled=True, return_activations=False,
            )
        finally:
            if _was_compiled:
                self.inference.enable_compile()

        logger.note("Instrumented forward pass complete")

        # Attribution: seed ← each candidate (who causally influences the seed?)
        upstream_candidates = [
            fid for fid in all_candidates
            if fid.layer < seed_fid.layer or (
                fid.layer == seed_fid.layer
                and ["attn", "mlp", "resid"].index(fid.kind) < ["attn", "mlp", "resid"].index(seed_fid.kind)
            )
        ]
        downstream_candidates = [fid for fid in all_candidates if fid not in upstream_candidates]

        n_edges = 0

        # Upstream edges: candidate → seed
        if upstream_candidates:
            attr_to_seed = compute_feature_attribution(
                instrument.graph,
                target_layer=seed_fid.layer,
                target_kind=seed_fid.kind,
                target_latent_idx=seed_fid.index,
                pos_argmax=pos_argmax,
                candidate_nodes=upstream_candidates,
            )
            for fid in upstream_candidates:
                score = attr_to_seed.get(fid, 0.0)
                if abs(score) >= self.attribution_threshold:
                    delta = deltas[fid]
                    role = "activator" if delta > 0 else "inhibitor"
                    node = CircuitNode(metadata={
                        "feature_id": fid,
                        "role": role,
                        "delta": delta,
                        "attribution": score,
                    })
                    circuit.add_node(node)
                    node_id_map[fid] = node.uuid
                    circuit.add_edge(node.uuid, seed_node.uuid, weight=score)
                    n_edges += 1

        # Downstream edges: seed → candidate
        for child_fid in downstream_candidates:
            try:
                attr_from_seed = compute_feature_attribution(
                    instrument.graph,
                    target_layer=child_fid.layer,
                    target_kind=child_fid.kind,
                    target_latent_idx=child_fid.index,
                    pos_argmax=pos_argmax,
                    candidate_nodes=[seed_fid],
                )
            except (KeyError, IndexError):
                continue

            score = attr_from_seed.get(seed_fid, 0.0)
            if abs(score) >= self.attribution_threshold:
                delta = deltas[child_fid]
                role = "activator" if delta > 0 else "inhibitor"
                node = CircuitNode(metadata={
                    "feature_id": child_fid,
                    "role": role,
                    "delta": delta,
                    "attribution": score,
                })
                circuit.add_node(node)
                node_id_map[child_fid] = node.uuid
                circuit.add_edge(seed_node.uuid, node.uuid, weight=score)
                n_edges += 1

        logger.stage(
            "causal edge construction",
            len(circuit.nodes), len(circuit.edges),
            note=f"{n_edges} edges from {len(all_candidates)} candidates",
        )

        del instrument
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(circuit.nodes) <= 1:
            logger.reject("no candidates passed attribution threshold")
            return None

        # ===================================================================
        # Phase 3: Minimality Pruning
        # ===================================================================
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

        # ===================================================================
        # Phase 4: Evaluation
        # ===================================================================
        target_kinds = ("attn", "mlp", "resid")
        _was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            final_f_global = evaluate_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
            )
            final_f_local = evaluate_kind_local_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, target_kinds=target_kinds,
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
        logger.note(f"EVAL_KIND_LOCAL  target_kinds={target_kinds}  faithfulness={final_f_local:.4f}")

        if final_f < self.min_faithfulness:
            logger.reject(f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}")
            return None

        n_activators = sum(1 for n in circuit.nodes.values() if n.metadata.get("role") == "activator")
        n_inhibitors = sum(1 for n in circuit.nodes.values() if n.metadata.get("role") == "inhibitor")

        circuit.metadata.update({
            "faithfulness":             final_f,
            "faithfulness_global":      final_f_global,
            "faithfulness_kind_local":  final_f_local,
            "kind_local_target_kinds":  list(target_kinds),
            "sufficiency":              final_s,
            "completeness":             final_c,
            "seed_comp":                seed_comp_idx,
            "seed_latent":              seed_latent_idx,
            "n_nodes":                  len(circuit.nodes),
            "n_edges":                  len(circuit.edges),
            "n_activators":             n_activators,
            "n_inhibitors":             n_inhibitors,
            "discovery_method":         self.method_name,
        })
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_activations(self, tokens: torch.Tensor) -> Dict[FeatureID, float]:
        """
        No-grad forward pass collecting total activation per latent.
        Returns {FeatureID: sum_of_activations} across all tokens in the batch.
        """
        result: Dict[FeatureID, float] = {}

        def capture_hook(layer_idx: int, activations: tuple):
            kinds = self.sae_bank.kinds
            for k_idx, act in enumerate(activations):
                kind = kinds[k_idx]
                top_acts, top_indices = self.sae_bank.encode(act, kind, layer_idx)

                active_mask = top_acts > 0
                if not active_mask.any():
                    continue

                flat_indices = top_indices[active_mask]
                flat_acts = top_acts[active_mask]

                unique_idx, inverse_idx = flat_indices.unique(return_inverse=True)
                summed = torch.zeros_like(unique_idx, dtype=torch.float32).scatter_add_(
                    0, inverse_idx, flat_acts.float()
                )

                for idx, total_act in zip(unique_idx.tolist(), summed.tolist()):
                    fid = FeatureID(layer_idx, kind, int(idx))
                    result[fid] = result.get(fid, 0.0) + total_act

        _was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            with torch.no_grad():
                self.inference.forward(
                    tokens, activations_callback=capture_hook, return_activations=False,
                )
        finally:
            if _was_compiled:
                self.inference.enable_compile()

        return result
