import torch
from typing import Optional, Any, Set, Tuple, Dict, List, cast

from .base import DiscoveryMethod
from config import config
from store.circuits import Circuit, CircuitNode
from store.top_coactivation import top_coactivation
from store.latent_stats import latent_stats
from eval.faithfulness import evaluate_faithfulness, evaluate_kind_local_faithfulness
from eval.sufficiency import evaluate_sufficiency
from eval.completeness import evaluate_completeness
from eval.minimality import prune_non_minimal_nodes
from circuit.sae_graph import SAEGraphInstrument
from circuit.circuit_logger import CircuitLogger


class AttnSparseExpansion(DiscoveryMethod):
    """
    Attn-targeted variable-depth co-activation expansion with full MLP/resid passthrough.

    Mirrors MlpSparseExpansion exactly, but targets the attention kind:

    Sparse attn part (co-activation expansion)
    -------------------------------------------
    Starting from an attn seed latent, the algorithm expands through the
    pre-computed co-activation graph for ``len(coact_depth)`` levels, retaining
    only attn-kind latents at every level.  mlp and resid neighbors are skipped.

      Depth 0  — the seed attn latent (role="seed").
      Depth 1  — top ``coact_depth[0]`` attn neighbors of the seed (role="hop1").
      Depth 2  — top ``coact_depth[1]`` attn neighbors of each depth-1 node (role="hop2").
      ...and so on for each entry in coact_depth.

    Live MLP/resid part (passthrough)
    -----------------------------------
    A single no-grad forward pass under SAEGraphInstrument captures every latent
    that fires for any mlp or resid component.  These are added with
    role="passthrough" and no edges.

    Args:
        coact_depth:       List of per-depth top-N limits, e.g. [32, 32].
                           Length determines expansion depth; each value is the
                           max attn neighbors added per node at that depth.
        min_faithfulness:  Minimum faithfulness to accept a circuit.
        min_active_count:  Minimum lifetime firing count for an attn candidate.
        pruning_threshold: Faithfulness-drop threshold for LOO pruning (0 = off).
        probe_batch_size:  Max probe sequences used for the passthrough capture.
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
        cfg = config.discovery.attn_sparse_expansion
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_attn_neighbors(
        self,
        comp_idx: int,
        latent_idx: int,
        limit: int,
        exclude: Set[Tuple[int, str, int]],
    ):
        """
        Yield (neighbor_comp_idx, neighbor_latent_idx, weight) for the top
        ``limit`` co-activation neighbors of (comp_idx, latent_idx) that are:
          1. kind == "attn"          ← the key filter for this algorithm
          2. above min_active_count
          3. not already in ``exclude``
        """
        d_sae = self.sae_bank.d_sae
        indices = top_coactivation.top_indices[comp_idx, latent_idx]
        values  = top_coactivation.top_values[comp_idx, latent_idx]

        n_yielded = 0
        for g_idx, w in zip(indices.tolist(), values.tolist()):
            if n_yielded >= limit:
                break
            n_comp = int(g_idx) // d_sae
            n_lat  = int(g_idx) % d_sae
            n_kind = self.sae_bank.kinds[n_comp % len(self.sae_bank.kinds)]

            if n_kind != "attn":
                continue

            n_layer = n_comp // len(self.sae_bank.kinds)
            key = (n_layer, n_kind, n_lat)
            if key in exclude:
                continue
            if latent_stats.active_count[n_comp, n_lat] < self.min_active_count:
                continue

            n_yielded += 1
            yield n_comp, n_lat, float(w)

    def _capture_passthrough_nodes(
        self, probe_tokens: torch.Tensor
    ) -> Dict[Tuple[int, str], Set[int]]:
        """
        Run a no-grad forward pass on probe_tokens under SAEGraphInstrument and
        collect every latent that fires for any mlp or resid component.

        Returns: { (layer, kind) -> set of active latent indices }
        """
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

        passthrough: Dict[Tuple[int, str], Set[int]] = {}
        for (layer, kind), steps in instrument.graph.activations.items():
            if kind not in ("mlp", "resid"):
                continue
            for _, _, top_indices in steps:
                key = (layer, kind)
                if key not in passthrough:
                    passthrough[key] = set()
                passthrough[key].update(int(v) for v in top_indices.flatten().tolist())

        return passthrough

    # ------------------------------------------------------------------
    # Main discover method
    # ------------------------------------------------------------------

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """Build an attn-sparse + mlp/resid-passthrough circuit from the seed feature."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "attn_sparse_expansion")
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
        seed_kind  = self.sae_bank.kinds[seed_comp_idx % len(self.sae_bank.kinds)]

        if seed_kind != "attn":
            logger.reject(
                f"seed kind '{seed_kind}' is not 'attn'; AttnSparseExpansion requires an attn seed"
            )
            return None

        circuit = Circuit(name=f"AttnSparse_S{seed_comp_idx}_{seed_latent_idx}")

        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts found)")
            return None

        probe_tokens = probe_data.pos_tokens[: self.probe_batch_size]

        logger.header(
            seed_layer, seed_kind, seed_latent_idx,
            probe_data.pos_tokens.shape[0],
            probe_data.neg_tokens.shape[0],
        )

        # ----------------------------------------------------------------
        # Depth 0: seed
        # ----------------------------------------------------------------
        seed_key: Tuple[int, str, int] = (seed_layer, seed_kind, seed_latent_idx)
        seed_node = CircuitNode(metadata={
            "layer_idx":  seed_layer,
            "latent_idx": seed_latent_idx,
            "kind":       seed_kind,
            "role":       "seed",
        })
        circuit.add_node(seed_node)
        node_id_map: Dict[Tuple[int, str, int], str] = {seed_key: seed_node.uuid}
        in_circuit: Set[Tuple[int, str, int]] = {seed_key}

        # ----------------------------------------------------------------
        # Variable-depth attn co-activation expansion (BFS)
        # ----------------------------------------------------------------
        frontier: List[Tuple[int, int, Tuple[int, str, int]]] = [
            (seed_comp_idx, seed_latent_idx, seed_key)
        ]

        for depth_idx, n_coacts in enumerate(self.coact_depth):
            role = f"hop{depth_idx + 1}"
            next_frontier: List[Tuple[int, int, Tuple[int, str, int]]] = []
            n_added = 0

            for parent_comp, parent_lat, parent_key in frontier:
                for n_comp, n_lat, w in self._expand_attn_neighbors(
                    parent_comp, parent_lat, n_coacts, exclude=in_circuit
                ):
                    n_layer = n_comp // len(self.sae_bank.kinds)
                    n_kind  = self.sae_bank.kinds[n_comp % len(self.sae_bank.kinds)]
                    key     = (n_layer, n_kind, n_lat)

                    node = CircuitNode(metadata={
                        "layer_idx":  n_layer,
                        "latent_idx": n_lat,
                        "kind":       n_kind,
                        "role":       role,
                    })
                    circuit.add_node(node)
                    node_id_map[key] = node.uuid
                    in_circuit.add(key)
                    circuit.add_edge(node_id_map[parent_key], node.uuid, weight=w)
                    next_frontier.append((n_comp, n_lat, key))
                    n_added += 1

            logger.stage(
                f"depth-{depth_idx + 1} attn expansion",
                len(circuit.nodes), len(circuit.edges),
                note=f"{len(frontier)} nodes expanded, {n_added} new attn nodes added",
            )
            frontier = next_frontier

        if len(circuit.nodes) <= 1:
            logger.reject("no attn neighbors found (all below activity filter or empty)")
            return None

        # ----------------------------------------------------------------
        # Passthrough: add every active mlp/resid latent from probe pass
        # ----------------------------------------------------------------
        passthrough_map = self._capture_passthrough_nodes(probe_tokens)
        n_passthrough = 0

        for (layer, kind), latent_set in passthrough_map.items():
            for latent_idx in sorted(latent_set):
                key = (layer, kind, latent_idx)
                if key in node_id_map:
                    continue
                node = CircuitNode(metadata={
                    "layer_idx":  layer,
                    "latent_idx": latent_idx,
                    "kind":       kind,
                    "role":       "passthrough",
                })
                circuit.add_node(node)
                node_id_map[key] = node.uuid
                n_passthrough += 1

        logger.stage(
            "mlp/resid passthrough", len(circuit.nodes), len(circuit.edges),
            note=f"{n_passthrough} passthrough nodes added",
        )

        # ----------------------------------------------------------------
        # Optional minimality pruning
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # Evaluation  (disable compile so CircuitPatcher hooks are not traced)
        # ----------------------------------------------------------------
        _was_compiled = self.inference._compiled
        self.inference.disable_compile()
        try:
            final_f_global = evaluate_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                probe_data.pos_tokens, pos_argmax=probe_data.pos_argmax,
            )
            final_f_local = evaluate_kind_local_faithfulness(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                probe_data.pos_tokens,
                target_kinds=("attn",),
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
        logger.note(f"EVAL_KIND_LOCAL  target_kinds=('attn',)  faithfulness={final_f_local:.4f}")

        if final_f < self.min_faithfulness:
            logger.reject(
                f"faithfulness {final_f:.4f} < min_faithfulness {self.min_faithfulness}"
            )
            return None

        circuit.metadata.update({
            "faithfulness":       final_f,
            "faithfulness_global": final_f_global,
            "faithfulness_kind_local": final_f_local,
            "kind_local_target_kinds": ["attn"],
            "sufficiency":        final_s,
            "completeness":       final_c,
            "seed_comp":          seed_comp_idx,
            "seed_latent":        seed_latent_idx,
            "n_nodes":            len(circuit.nodes),
            "n_edges":            len(circuit.edges),
            "discovery_method":   "attn_sparse_expansion",
            "coact_depth":        self.coact_depth,
            "n_passthrough":      n_passthrough,
        })
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit
