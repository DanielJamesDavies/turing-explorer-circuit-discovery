import logging
from typing import Optional, Any, Dict, Set, cast

import torch

logger = logging.getLogger(__name__)

from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from store.latent_stats import latent_stats
from eval.faithfulness import evaluate_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from eval.upstream_faithfulness import evaluate_upstream_faithfulness
from circuit.instrument.attribution import compute_direct_effects_matrix
from circuit.instrument.ct_influence import compute_ct_influence, prune_ct_graph
from circuit.types.feature_id import FeatureID
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx


class CircuitTracerBaseline(DiscoveryMethod):
    """
    SAE-adapted Circuit Tracing baseline, aligned with Anthropic's Attribution Graphs method.

    Builds a prompt-local direct-effects adjacency matrix over active SAE latents
    (attn/mlp/resid across all layers) via a linearised SAEGraphInstrument forward.
    Influence is propagated backward from logit root nodes via a truncated Neumann
    series, then the graph is globally pruned by scale-invariant fraction thresholds.

    Pipeline (per probe sequence):
      1. No-grad discovery → collect all active latents.
      2. Logit target selection via cumulative softmax probability (desired_logit_prob).
      3. Single retained-graph forward pass.
      4. Logit ranking passes (Pass A): backward from each logit target against the
         full feature set; rank features by one-hop logit influence; select top
         max_feature_nodes.
      5. Logit final passes (Pass B): re-run with the selected node list to fill adj.
      6. Feature backward passes for selected features only.
      7. Influence propagation: Neumann series seeded by softmax probabilities;
         converges to exact zero (finite-DAG property).
      8. Fraction-based pruning: keep the smallest node/edge sets that cover
         node_threshold / edge_threshold of total influence mass.

    Alignment with the original circuit tracer:
      - Logit targets: cumulative softmax probability (not fixed top-K by raw logit).
      - Logit demeaning: scalar = (logit_i − mean_logit).sum() per backward pass.
      - Logit root weights: actual softmax probabilities (not uniform 1/K).
      - Feature selection: logit-influence ranking (not peak activation magnitude).
      - Thresholds: fraction-based via _find_threshold (scale-invariant).

    VRAM budget (16 GB):
      probe_batch_size=1   — keeps each retained computation graph small.
      torch.cuda.empty_cache() + gc.collect() after freeing each retained graph.
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        probe_batch_size: Optional[int] = None,
        max_sequences: Optional[int] = None,
        target_chunk_size: Optional[int] = None,
        logit_top_k: Optional[int] = None,
        desired_logit_prob: Optional[float] = None,
        influence_max_iter: Optional[int] = None,
        node_threshold: Optional[float] = None,
        edge_threshold: Optional[float] = None,
        min_faithfulness: Optional[float] = None,
        pruning_threshold: Optional[float] = None,
        min_active_count: Optional[int] = None,
        max_feature_nodes: Optional[int] = None,
        stop_error_grad: Optional[bool] = None,
        include_error_nodes: Optional[bool] = None,
        online_ranking_interval: Optional[int] = None,
        feature_batch_size: Optional[int] = None,
        include_token_nodes: Optional[bool] = None,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.circuit_tracer_baseline

        self.probe_batch_size = (
            probe_batch_size if probe_batch_size is not None
            else cast(int, cfg.probe_batch_size)
        )
        self.max_sequences = (
            max_sequences if max_sequences is not None
            else cast(int, cfg.max_sequences)
        )
        self.target_chunk_size = (
            target_chunk_size if target_chunk_size is not None
            else cast(int, cfg.target_chunk_size)
        )
        self.logit_top_k = (
            logit_top_k if logit_top_k is not None
            else cast(int, cfg.logit_top_k)
        )
        self.desired_logit_prob = (
            desired_logit_prob if desired_logit_prob is not None
            else cast(float, cfg.desired_logit_prob)
        )
        self.influence_max_iter = (
            influence_max_iter if influence_max_iter is not None
            else cast(int, cfg.influence_max_iter)
        )
        self.node_threshold = (
            node_threshold if node_threshold is not None
            else cast(float, cfg.node_threshold)
        )
        self.edge_threshold = (
            edge_threshold if edge_threshold is not None
            else cast(float, cfg.edge_threshold)
        )
        self.min_faithfulness = (
            min_faithfulness if min_faithfulness is not None
            else cast(float, cfg.min_faithfulness)
        )
        self.pruning_threshold = (
            pruning_threshold if pruning_threshold is not None
            else cast(float, cfg.pruning_threshold)
        )
        self.min_active_count = (
            min_active_count if min_active_count is not None
            else cast(int, cfg.min_active_count)
        )
        self.max_feature_nodes = (
            max_feature_nodes if max_feature_nodes is not None
            else cast(int, cfg.max_feature_nodes)
        )
        self.stop_error_grad = (
            stop_error_grad if stop_error_grad is not None
            else cast(bool, cfg.stop_error_grad)
        )
        self.include_error_nodes = (
            include_error_nodes if include_error_nodes is not None
            else cast(bool, cfg.include_error_nodes)
        )
        self.online_ranking_interval = (
            online_ranking_interval if online_ranking_interval is not None
            else cast(int, cfg.online_ranking_interval)
        )
        self.feature_batch_size = (
            feature_batch_size if feature_batch_size is not None
            else cast(int, cfg.feature_batch_size)
        )
        self.include_token_nodes = (
            include_token_nodes if include_token_nodes is not None
            else cast(bool, cfg.include_token_nodes)
        )

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "circuit_tracer_baseline")
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

        logger.header(seed_layer, seed_kind, seed_latent_idx, self.max_sequences, 0)

        # ── 1. Probe dataset ──────────────────────────────────────────────────
        probe_data = self.build_probe_dataset(
            seed_comp_idx, seed_latent_idx, n_pos=self.max_sequences
        )
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset for seed")
            return None

        tokens = probe_data.pos_tokens[: self.max_sequences]
        target_tokens = probe_data.target_tokens[: self.max_sequences]
        eval_argmax = probe_data.pos_argmax[: self.max_sequences]

        print(
            f"[CTBaseline] Seed: L{seed_layer}.{seed_kind}.f{seed_latent_idx} | "
            f"{tokens.shape[0]} probe seqs | probe_batch={self.probe_batch_size} | "
            f"max_nodes={self.max_feature_nodes}",
            flush=True,
        )

        # ── 2. Build direct-effects adjacency matrix ──────────────────────────
        # compile must be off for grad-enabled forward passes; SAE graph hooks
        # require eager execution.
        self.inference.disable_compile()
        try:
            adj, all_nodes, logit_probs = compute_direct_effects_matrix(
                tokens=tokens,
                inference=self.inference,
                bank=self.sae_bank,
                logit_top_k=self.logit_top_k,
                probe_batch_size=self.probe_batch_size,
                kinds=kinds,
                n_kinds=n_kinds,
                min_active_count=self.min_active_count,
                active_count=latent_stats.active_count,
                max_feature_nodes=self.max_feature_nodes,
                stop_error_grad=self.stop_error_grad,
                desired_logit_prob=self.desired_logit_prob,
                include_error_nodes=self.include_error_nodes,
                online_ranking_interval=self.online_ranking_interval,
                feature_batch_size=self.feature_batch_size,
                include_token_nodes=self.include_token_nodes,
            )
        finally:
            self.inference.enable_compile()

        n_logit_nodes = sum(1 for n in all_nodes if n.kind == "logit")
        n_error_nodes = sum(1 for n in all_nodes if n.kind.endswith("_err"))
        n_token_nodes = sum(1 for n in all_nodes if n.kind == "token")
        n_feature_nodes = len(all_nodes) - n_logit_nodes - n_error_nodes - n_token_nodes
        print(
            f"[CTBaseline] Matrix built: {n_feature_nodes} feature nodes | "
            f"{n_error_nodes} error nodes"
            + (f" | {n_token_nodes} token nodes" if n_token_nodes else "")
            + f" | {n_logit_nodes} logit nodes | {len(adj)} edges",
            flush=True,
        )
        logger.stage("matrix_built", len(all_nodes), len(adj))

        if not all_nodes:
            logger.reject("no active latents found on probe sequences")
            return None

        # Guard: seed must have fired on at least one probe sequence
        if seed_fid not in all_nodes:
            logger.reject(
                f"seed latent {seed_fid} did not fire on any probe sequence"
            )
            return None

        # ── 3. Influence propagation (CPU, truncated Neumann series) ──────────
        print(
            f"[CTBaseline] Computing influence (Neumann series, max_iter={self.influence_max_iter})...",
            flush=True,
        )
        try:
            influence = compute_ct_influence(
                adj,
                all_nodes,
                n_logit_nodes,
                max_iter=self.influence_max_iter,
                logit_probabilities=logit_probs if logit_probs.shape[0] == n_logit_nodes else None,
            )
        except RuntimeError as exc:
            # Graceful degradation: graph has strong cycles (probe-sequence averaging
            # artifact).  Fall back to zeros so pruning keeps only logit nodes and
            # discovery continues rather than crashing the full pipeline.
            logger.warning(
                "[CTBaseline] Influence did not converge; falling back to zero "
                "influence.  Circuit will contain only logit nodes.  Details: %s",
                exc,
            )
            influence = torch.zeros(len(all_nodes), dtype=torch.float32)
        print("[CTBaseline] Influence done. Pruning graph...", flush=True)

        # ── 4. Prune by influence + edge threshold ────────────────────────────
        pruned_adj, kept_nodes = prune_ct_graph(
            adj,
            all_nodes,
            influence,
            self.node_threshold,
            self.edge_threshold,
            n_logit_nodes,
            logit_probabilities=logit_probs if logit_probs.shape[0] == n_logit_nodes else None,
            max_iter=self.influence_max_iter,
        )

        n_kept_features = sum(1 for n in kept_nodes if n.layer >= 0)
        n_kept_tokens = sum(1 for n in kept_nodes if n.kind == "token")
        print(
            f"[CTBaseline] After pruning: {n_kept_features} feature nodes"
            + (f" | {n_kept_tokens} token nodes" if n_kept_tokens else "")
            + f" | {len(pruned_adj)} edges",
            flush=True,
        )
        logger.stage("pruned", len(kept_nodes), len(pruned_adj))

        if seed_fid not in kept_nodes:
            logger.reject("seed node fell below influence threshold during pruning")
            return None

        # ── 5. Build Circuit from kept feature nodes ───────────────────────────
        # Logit sentinel nodes (layer=-1) are not added as CircuitNodes — they
        # served only to seed the influence propagation.
        circuit = Circuit(name=f"CTBaseline_S{seed_comp_idx}_{seed_latent_idx}")
        node_map: Dict[FeatureID, str] = {}

        for fid in kept_nodes:
            if fid.kind == "logit":
                continue  # skip logit sentinels
            role = (
                "seed" if fid == seed_fid
                else "error" if fid.kind.endswith("_err")
                else "token" if fid.kind == "token"
                else "feature"
            )
            cn = CircuitNode(metadata={"feature_id": fid, "role": role})
            circuit.add_node(cn)
            node_map[fid] = cn.uuid

        # Add edges between surviving feature nodes
        for (src_idx, tgt_idx), score in pruned_adj.items():
            src_fid = all_nodes[src_idx]
            tgt_fid = all_nodes[tgt_idx]
            if src_fid in node_map and tgt_fid in node_map:
                circuit.add_edge(node_map[src_fid], node_map[tgt_fid], weight=score)

        if len(circuit.nodes) <= 1:
            logger.reject("no upstream nodes survived pruning")
            return None

        # ── 6. Evaluate ───────────────────────────────────────────────────────
        # Restrict intervention to layers that actually have circuit nodes.
        # Layers outside this set run naturally — the circuit makes no claim
        # about their contribution (same convention as gradient_upstream).
        circuit_layers: Set[int] = {
            node.feature_id.layer
            for node in circuit.nodes.values()
            if node.feature_id is not None
        }
        print(
            f"[CTBaseline] Evaluating | "
            f"{len(circuit.nodes)} nodes | {len(circuit.edges)} edges | "
            f"layers={sorted(circuit_layers)}",
            flush=True,
        )

        # Compile must remain off during no-grad evaluation (CircuitPatcher uses
        # forward hooks that torch.compile inlines away).
        self.inference.disable_compile()
        try:
            # Optional minimality pruning
            if self.pruning_threshold > 0:
                n_before = len(circuit.nodes)
                prune_non_minimal_nodes(
                    self.inference,
                    self.sae_bank,
                    self.avg_acts,
                    circuit,
                    tokens,
                    pos_argmax=eval_argmax,
                    threshold=self.pruning_threshold,
                    circuit_layers=circuit_layers,
                )
                circuit_layers = {
                    node.feature_id.layer
                    for node in circuit.nodes.values()
                    if node.feature_id is not None
                }
                logger.stage(
                    "after_minimality_pruning",
                    len(circuit.nodes),
                    len(circuit.edges),
                    note=f"removed {n_before - len(circuit.nodes)} nodes",
                )

            # Upstream faithfulness: how well the circuit explains the seed's activation
            up_faith = evaluate_upstream_faithfulness(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                seed_layer,
                seed_kind,
                seed_latent_idx,
                tokens,
                pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )

            # Standard faithfulness: how well the circuit recovers the model's output
            final_f = evaluate_faithfulness(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                tokens,
                pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )

            # Sufficiency: does the circuit alone recover the top prediction?
            final_s = evaluate_sufficiency(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                tokens,
                target_tokens,
                pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )

            # Completeness: how much is destroyed by removing only the circuit?
            final_c = evaluate_completeness(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                tokens,
                pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )
        finally:
            self.inference.enable_compile()

        logger.eval(final_f, final_s, final_c)
        logger.note(
            f"upstream_faithfulness={up_faith:.4f} | "
            f"n_nodes={len(circuit.nodes)} | n_edges={len(circuit.edges)}"
        )

        if up_faith >= self.min_faithfulness:
            circuit.metadata.update(
                {
                    "faithfulness": final_f,
                    "sufficiency": final_s,
                    "completeness": final_c,
                    "upstream_faithfulness": up_faith,
                    "seed_comp": seed_comp_idx,
                    "seed_latent": seed_latent_idx,
                    "n_nodes": len(circuit.nodes),
                    "n_edges": len(circuit.edges),
                    "discovery_method": "circuit_tracer_baseline",
                    "probe_batch_size": self.probe_batch_size,
                    "target_chunk_size": self.target_chunk_size,
                    "logit_top_k": self.logit_top_k,
                    "node_threshold": self.node_threshold,
                    "edge_threshold": self.edge_threshold,
                }
            )
            logger.nodes(list(circuit.nodes.values()))
            logger.accept(len(circuit.nodes), len(circuit.edges))
            return circuit

        logger.reject(
            f"upstream_faithfulness {up_faith:.4f} < min_faithfulness {self.min_faithfulness}"
        )
        return None
