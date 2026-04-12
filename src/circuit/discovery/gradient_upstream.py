import torch
import gc
from typing import Optional, Any, List, Tuple, Dict, Set, cast
from collections import deque

from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from store.latent_stats import latent_stats
from eval.faithfulness import evaluate_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from eval.upstream_faithfulness import evaluate_upstream_faithfulness
from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.instrument.attribution import compute_latent_upstream_scores, UpstreamScores
from circuit.types.feature_id import FeatureID
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx, get_predecessor_components

class GradientUpstreamDiscovery(DiscoveryMethod):
    """
    Discovers circuits by propagating gradient attribution backwards through the model.
    
    Unlike standard gradient methods that use the seed's context for all layers,
    this method switches context at each hop to the upstream latent's own top_ctx.
    This grounds each attribution step in the input distribution where that specific
    latent is most reliably active.
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        depth: Optional[int] = None,
        top_k_per_hop: Optional[int] = None,
        attribution_threshold: Optional[float] = None,
        min_active_count: Optional[int] = None,
        max_ctx_sequences: Optional[int] = None,
        hop_batch_size: Optional[int] = None,
        absent_inhibitor_top_k: Optional[int] = None,
        absent_inhibitor_threshold: Optional[float] = None,
        pruning_threshold: Optional[float] = None,
        min_faithfulness: Optional[float] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.gradient_upstream
        
        self.depth = depth if depth is not None else cast(int, cfg.depth)
        self.top_k_per_hop = top_k_per_hop if top_k_per_hop is not None else cast(int, cfg.top_k_per_hop)
        self.attribution_threshold = (
            attribution_threshold if attribution_threshold is not None
            else cast(float, cfg.attribution_threshold)
        )
        self.min_active_count = (
            min_active_count if min_active_count is not None
            else cast(int, cfg.min_active_count)
        )
        self.max_ctx_sequences = (
            max_ctx_sequences if max_ctx_sequences is not None
            else cast(int, cfg.max_ctx_sequences)
        )
        self.hop_batch_size = (
            hop_batch_size if hop_batch_size is not None
            else cast(int, cfg.hop_batch_size)
        )
        self.absent_inhibitor_top_k = (
            absent_inhibitor_top_k if absent_inhibitor_top_k is not None
            else cast(int, cfg.absent_inhibitor_top_k)
        )
        self.absent_inhibitor_threshold = (
            absent_inhibitor_threshold if absent_inhibitor_threshold is not None
            else cast(float, cfg.absent_inhibitor_threshold)
        )
        self.pruning_threshold = (
            pruning_threshold if pruning_threshold is not None
            else cast(float, cfg.pruning_threshold)
        )
        self.min_faithfulness = (
            min_faithfulness if min_faithfulness is not None
            else cast(float, cfg.min_faithfulness)
        )

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """Backwards gradient BFS with context switching."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "gradient_upstream")
        try:
            with self.sae_bank.pin_decoders():
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

        circuit = Circuit(name=f"GradUpstream_S{seed_comp_idx}_{seed_latent_idx}")
        
        # 1. Probe dataset for seed
        # Use only max_ctx_sequences for discovery to save time on argmax calculation
        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx, n_pos=self.max_ctx_sequences)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset for seed")
            return None

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        # 2. Add seed node
        seed_node = CircuitNode(metadata={
            "feature_id": seed_fid,
            "role": "seed",
        })
        circuit.add_node(seed_node)
        fid_to_uuid = {seed_fid: seed_node.uuid}

        # 3. BFS Queue: (comp_idx, latent_idx, tokens, current_depth)
        seed_tokens = probe_data.pos_tokens[:self.max_ctx_sequences]
        queue = deque([(seed_comp_idx, seed_latent_idx, seed_tokens, 0)])
        visited: Set[Tuple[int, int]] = {(seed_comp_idx, seed_latent_idx)}

        # 4. Main BFS Loop
        while queue:
            node_comp, node_latent, node_tokens, current_depth = queue.popleft()
            
            if current_depth >= self.depth:
                continue

            print(f"[GradUpstream] Hop: {current_depth}/{self.depth} | Node: comp_{node_comp}_lat_{node_latent} | Queue: {len(queue)}")
            
            # Run one hop of backward attribution
            upstream_scores = self._run_hop(node_comp, node_latent, node_tokens, logger)

            node_fid = FeatureID.from_component_id(node_comp, node_latent, n_kinds, kinds)

            n_activators = 0
            n_active_inhibitors = 0
            n_absent_inhibitors = 0
            n_enqueued = 0

            # --- Attribution scores: split by sign into activator / active_inhibitor ---
            for upstream_fid, score in upstream_scores.attribution.items():
                if abs(score) < self.attribution_threshold:
                    continue

                upstream_comp, upstream_latent = upstream_fid.to_component_id(n_kinds, kinds)

                if latent_stats.active_count[upstream_comp, upstream_latent] < self.min_active_count:
                    continue

                role = "activator" if score > 0 else "active_inhibitor"

                if upstream_fid not in fid_to_uuid:
                    upstream_node = CircuitNode(metadata={
                        "feature_id": upstream_fid,
                        "role": role,
                        "attribution_score": score,
                        "depth": current_depth + 1,
                    })
                    circuit.add_node(upstream_node)
                    fid_to_uuid[upstream_fid] = upstream_node.uuid

                circuit.add_edge(
                    fid_to_uuid[upstream_fid],
                    fid_to_uuid[node_fid],
                    weight=score,
                )

                if role == "activator":
                    n_activators += 1
                    # Only activators are BFS-expanded — active_inhibitors are dead ends.
                    if (upstream_comp, upstream_latent) not in visited:
                        visited.add((upstream_comp, upstream_latent))
                        print(f"  [Context Switch] activator → comp_{upstream_comp}_lat_{upstream_latent}")
                        upstream_probe = self.build_probe_dataset(upstream_comp, upstream_latent, n_pos=self.max_ctx_sequences)
                        if upstream_probe.pos_tokens.shape[0] > 0:
                            upstream_tokens = upstream_probe.pos_tokens[:self.max_ctx_sequences]
                            queue.append((upstream_comp, upstream_latent, upstream_tokens, current_depth + 1))
                            n_enqueued += 1
                else:
                    n_active_inhibitors += 1
                    # Still mark visited so a different ancestor doesn't re-expand it.
                    visited.add((upstream_comp, upstream_latent))

            # --- Absent-gradient scores: absent_inhibitor nodes, never BFS-expanded ---
            if self.absent_inhibitor_top_k > 0:
                for upstream_fid, score in upstream_scores.absent_gradient.items():
                    upstream_comp, upstream_latent = upstream_fid.to_component_id(n_kinds, kinds)

                    if latent_stats.active_count[upstream_comp, upstream_latent] < self.min_active_count:
                        continue

                    if upstream_fid not in fid_to_uuid:
                        upstream_node = CircuitNode(metadata={
                            "feature_id": upstream_fid,
                            "role": "absent_inhibitor",
                            "attribution_score": score,
                            "depth": current_depth + 1,
                        })
                        circuit.add_node(upstream_node)
                        fid_to_uuid[upstream_fid] = upstream_node.uuid

                    circuit.add_edge(
                        fid_to_uuid[upstream_fid],
                        fid_to_uuid[node_fid],
                        weight=score,
                    )
                    visited.add((upstream_comp, upstream_latent))
                    n_absent_inhibitors += 1

            print(
                f"  [Hop Result] activators={n_activators} enqueued={n_enqueued} | "
                f"active_inhibitors={n_active_inhibitors} | absent_inhibitors={n_absent_inhibitors}"
            )

        logger.stage("discovery", len(circuit.nodes), len(circuit.edges))
        print(f"[GradUpstream] Discovery complete: {len(circuit.nodes)} nodes, {len(circuit.edges)} edges.")

        if len(circuit.nodes) <= 1:
            logger.reject("no upstream nodes found")
            return None

        # 5. Evaluation
        # Restrict ablation to only the layers where circuit nodes actually exist.
        # Layers outside this set run naturally — the circuit makes no claim about them.
        circuit_layers: Set[int] = {
            node.feature_id.layer
            for node in circuit.nodes.values()
            if node.feature_id is not None
        }
        print(f"[GradUpstream] Starting evaluation | circuit_layers={sorted(circuit_layers)}")

        eval_tokens = probe_data.pos_tokens[:self.max_ctx_sequences]
        eval_argmax = probe_data.pos_argmax[:self.max_ctx_sequences]
        eval_targets = probe_data.target_tokens[:self.max_ctx_sequences]

        # Disable compile for all evaluations: torch.compile inlines attn/mlp submodule
        # calls, bypassing their forward hooks that CircuitPatcher relies on. Eagerly-run
        # evaluations are all @torch.no_grad() so the overhead is minimal.
        self.inference.disable_compile()
        try:
            # Minimality pruning (optional)
            if self.pruning_threshold > 0:
                n_before = len(circuit.nodes)
                prune_non_minimal_nodes(
                    self.inference, self.sae_bank, self.avg_acts, circuit,
                    eval_tokens, pos_argmax=eval_argmax,
                    threshold=self.pruning_threshold,
                    circuit_layers=circuit_layers,
                )
                # Recompute circuit_layers after pruning — removed nodes may have vacated a layer.
                circuit_layers = {
                    node.feature_id.layer
                    for node in circuit.nodes.values()
                    if node.feature_id is not None
                }
                logger.stage(
                    "after pruning", len(circuit.nodes), len(circuit.edges),
                    note=f"removed {n_before - len(circuit.nodes)} nodes",
                )

            # Upstream faithfulness: how well do the discovered nodes explain the seed?
            up_faith = evaluate_upstream_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                seed_layer, seed_kind, seed_latent_idx,
                eval_tokens, pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )

            # Standard faithfulness: how well does the circuit explain the model's output?
            final_f = evaluate_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                eval_tokens, pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )

            # Sufficiency: does the circuit recover the top prediction?
            final_s = evaluate_sufficiency(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                eval_tokens, eval_targets,
                pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )

            # Completeness: how much was destroyed by removing only the circuit?
            final_c = evaluate_completeness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                eval_tokens, pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )
        finally:
            self.inference.enable_compile()

        logger.eval(final_f, final_s, final_c)
        logger.note(f"upstream_faithfulness: {up_faith:.4f}")

        if up_faith >= self.min_faithfulness:
            circuit.metadata.update({
                "faithfulness": final_f,
                "sufficiency": final_s,
                "completeness": final_c,
                "upstream_faithfulness": up_faith,
                "seed_comp": seed_comp_idx,
                "seed_latent": seed_latent_idx,
                "n_nodes": len(circuit.nodes),
                "n_edges": len(circuit.edges),
                "discovery_method": "gradient_upstream",
                "depth": self.depth,
                "top_k_per_hop": self.top_k_per_hop,
            })
            logger.nodes(list(circuit.nodes.values()))
            logger.accept(len(circuit.nodes), len(circuit.edges))
            return circuit

        logger.reject(
            f"upstream_faithfulness {up_faith:.4f} < min_faithfulness {self.min_faithfulness}"
        )
        return None

    def _run_hop(self, comp_idx: int, latent_idx: int, tokens: torch.Tensor, logger: CircuitLogger) -> UpstreamScores:
        """
        Runs grad-enabled forward passes in microbatches and accumulates upstream scores.

        Splits `tokens` into chunks of `hop_batch_size`, runs a separate
        SAEGraphInstrument forward+backward per chunk, accumulates attribution and
        absent-gradient scores by summation, then applies global top-K selection.
        Each chunk's computation graph is freed before the next chunk's forward pass.
        """
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        layer, kind_idx = split_component_idx(comp_idx, n_kinds)
        kind = kinds[kind_idx]
        predecessors = get_predecessor_components(comp_idx, n_kinds, kinds)

        # Accumulated scores across all chunks — summed then top-K'd at the end.
        # Using a large per-chunk top_k so every nonzero score is returned from
        # each chunk before the final global selection.
        accumulated_attribution: Dict[FeatureID, float] = {}
        accumulated_absent: Dict[FeatureID, float] = {}

        chunks = tokens.split(self.hop_batch_size, dim=0)
        n_chunks = len(chunks)

        self.inference.disable_compile()
        try:
            for chunk_idx, chunk in enumerate(chunks):
                print(f"    [GradPass] Chunk {chunk_idx + 1}/{n_chunks} | comp_{comp_idx}_lat_{latent_idx} | {len(chunk)} seqs")
                instrument = SAEGraphInstrument(self.sae_bank)
                try:
                    self.inference.forward(
                        chunk,
                        patcher=instrument,
                        grad_enabled=True,
                        return_activations=False,
                        tokenize_final=False,
                    )

                    _, acts_connected, _ = instrument.graph.get_latents(layer, kind)
                    # acts_connected.act is dense [B, T, d_sae] (scatter_ in SAEGraphInstrument)
                    target_acts = acts_connected.act[..., latent_idx]  # [B, T]
                    pos_argmax = target_acts.argmax(dim=-1)             # [B]

                    # Pass top_k=d_sae to get all nonzero scores from this chunk
                    # before global top-K is applied below.  Absent threshold is
                    # set to 0.0 so cross-chunk contributions below the threshold
                    # in a single chunk are not silently discarded before summing.
                    chunk_scores = compute_latent_upstream_scores(
                        instrument.graph,
                        target_layer=layer,
                        target_kind=kind,
                        target_latent_idx=latent_idx,
                        pos_argmax=pos_argmax,
                        predecessor_comp_indices=predecessors,
                        n_kinds=n_kinds,
                        kinds=kinds,
                        top_k=self.sae_bank.d_sae,
                        min_active_count=self.min_active_count,
                        active_count=latent_stats.active_count,
                        absent_inhibitor_top_k=self.sae_bank.d_sae if self.absent_inhibitor_top_k > 0 else 0,
                        absent_inhibitor_threshold=0.0,
                    )

                    for fid, s in chunk_scores.attribution.items():
                        accumulated_attribution[fid] = accumulated_attribution.get(fid, 0.0) + s
                    for fid, s in chunk_scores.absent_gradient.items():
                        accumulated_absent[fid] = accumulated_absent.get(fid, 0.0) + s

                finally:
                    # Free each chunk's computation graph before the next forward pass.
                    del instrument
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        finally:
            self.inference.enable_compile()

        # Global top-K selection across all accumulated chunks.
        top_attribution = sorted(accumulated_attribution.items(), key=lambda x: abs(x[1]), reverse=True)
        attribution_dict = {fid: s for fid, s in top_attribution[:self.top_k_per_hop]}

        # Apply real absent-inhibitor threshold on accumulated scores, then take top-K.
        top_absent = sorted(
            [(fid, s) for fid, s in accumulated_absent.items() if s < -self.absent_inhibitor_threshold],
            key=lambda x: x[1],  # most-negative first
        )
        absent_dict = {fid: s for fid, s in top_absent[:self.absent_inhibitor_top_k]}

        print(f"    [GradAccum] {n_chunks} chunks | attr: {len(attribution_dict)} top nodes | absent: {len(absent_dict)} absent inhibitors")
        return UpstreamScores(attribution=attribution_dict, absent_gradient=absent_dict)
