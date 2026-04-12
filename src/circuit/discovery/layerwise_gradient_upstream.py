import heapq
import os
import time
import torch
import torch.profiler
import gc
from typing import Optional, Any, List, Tuple, Dict, Set, cast

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
from pipeline.component_index import split_component_idx, get_all_upstream_components


class LayerwiseGradientUpstreamDiscovery(DiscoveryMethod):
    """
    Discovers circuits by sweeping layer-by-layer backwards through the model.

    Unlike GradientUpstreamDiscovery, which uses BFS with per-node depth counting
    and only scores against direct causal predecessors (e.g. resid@L-1), this method:

    1. Processes all nodes at the same (layer, kind) together as a group.
    2. Computes gradient attribution against ALL upstream (layer', kind') pairs
       where layer' < layer, not just the immediately adjacent layer.
    3. Drives iteration via a max-priority heap ordered by layer index, giving a
       natural topological ordering that mirrors the transformer's computation graph
       (highest layer first, resid → mlp → attn within a layer).
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        top_k_per_node: Optional[int] = None,
        attribution_threshold: Optional[float] = None,
        min_active_count: Optional[int] = None,
        max_ctx_sequences: Optional[int] = None,
        hop_batch_size: Optional[int] = None,
        absent_inhibitor_top_k: Optional[int] = None,
        absent_inhibitor_threshold: Optional[float] = None,
        max_layers_back: Optional[int] = None,
        include_same_layer: Optional[bool] = None,
        pruning_threshold: Optional[float] = None,
        min_faithfulness: Optional[float] = None,
        profile_first_node: bool = False,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.layerwise_gradient_upstream

        self.top_k_per_node = (
            top_k_per_node if top_k_per_node is not None else cast(int, cfg.top_k_per_node)
        )
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
        self.max_layers_back = (
            max_layers_back if max_layers_back is not None
            else cast(int, cfg.max_layers_back)
        )
        self.include_same_layer = (
            include_same_layer if include_same_layer is not None
            else cast(bool, cfg.include_same_layer)
        )
        self.pruning_threshold = (
            pruning_threshold if pruning_threshold is not None
            else cast(float, cfg.pruning_threshold)
        )
        self.min_faithfulness = (
            min_faithfulness if min_faithfulness is not None
            else cast(float, cfg.min_faithfulness)
        )
        self.profile_first_node = profile_first_node
        self._profiled = False   # set True after the first profile fires

    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """Layer-by-layer gradient sweep with full upstream attribution."""
        logger = CircuitLogger(seed_comp_idx, seed_latent_idx, "layerwise_gradient_upstream")
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

        circuit = Circuit(name=f"LayerwiseGradUpstream_S{seed_comp_idx}_{seed_latent_idx}")

        # 1. Probe dataset for seed
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
        fid_to_uuid: Dict[FeatureID, str] = {seed_fid: seed_node.uuid}

        # The earliest layer we will ever look back to.
        effective_min_layer = (
            max(0, seed_layer - self.max_layers_back) if self.max_layers_back > 0 else 0
        )

        # 3. Max-priority heap of (layer, kind) pairs to process — highest layer first.
        #    heapq is a min-heap, so we push (-layer, -kind_idx, kind_name).
        #    Within a layer the ordering is resid → mlp → attn (kind_idx descending),
        #    which matches the causal ordering of the transformer residual stream.
        kind_to_idx: Dict[str, int] = {k: i for i, k in enumerate(kinds)}

        # heap entries: (-layer, -kind_idx, kind_name)
        heap: List[Tuple[int, int, str]] = []
        # Tracks pairs ever pushed so we never push the same (layer, kind) twice.
        in_heap: Set[Tuple[int, str]] = set()
        # Tracks (comp_idx, latent_idx) pairs whose upstream gradient has been computed.
        expanded: Set[Tuple[int, int]] = set()

        def _push(layer: int, kind: str) -> None:
            key = (layer, kind)
            if key not in in_heap:
                in_heap.add(key)
                heapq.heappush(heap, (-layer, -kind_to_idx.get(kind, 0), kind))

        _push(seed_layer, seed_kind)

        def _nodes_at(layer: int, kind: str) -> List[FeatureID]:
            """Collect all FeatureIDs in the circuit at the given (layer, kind)."""
            return [fid for fid in fid_to_uuid if fid.layer == layer and fid.kind == kind]

        # 4. Main loop: process (layer, kind) groups from highest layer down
        while heap:
            neg_layer, _neg_kind_idx, kind = heapq.heappop(heap)
            layer = -neg_layer

            unexpanded = [
                fid for fid in _nodes_at(layer, kind)
                if fid.to_component_id(n_kinds, kinds) not in expanded
            ]

            if not unexpanded:
                continue

            print(
                f"[LayerwiseGradUpstream] Layer {layer} | kind={kind} | "
                f"{len(unexpanded)} node(s) | heap size: {len(heap)}"
            )

            # All nodes at (layer, kind) share the same upstream component scope.
            representative_comp = unexpanded[0].to_component_id(n_kinds, kinds)[0]
            all_upstream = get_all_upstream_components(
                representative_comp,
                n_kinds,
                kinds,
                min_layer=effective_min_layer,
                include_same_layer=self.include_same_layer,
            )

            for fid in unexpanded:
                comp_idx, latent_idx = fid.to_component_id(n_kinds, kinds)
                expanded.add((comp_idx, latent_idx))

                t0_probe = time.perf_counter()
                node_probe = self.build_probe_dataset(
                    comp_idx, latent_idx, n_pos=self.max_ctx_sequences
                )
                t_probe = (time.perf_counter() - t0_probe) * 1000
                if node_probe.pos_tokens.shape[0] == 0:
                    print(f"  [LayerwiseGradUpstream] Skip {fid}: empty probe dataset ({t_probe:.0f}ms)")
                    continue

                print(f"  [{fid}] probe={t_probe:.0f}ms | {node_probe.pos_tokens.shape[0]} pos seqs")
                node_tokens = node_probe.pos_tokens[:self.max_ctx_sequences]
                t0_node = time.perf_counter()
                if self.profile_first_node and not self._profiled:
                    upstream_scores = self._run_node_profiled(
                        comp_idx, latent_idx, node_tokens, all_upstream, logger, fid
                    )
                    self._profiled = True
                else:
                    upstream_scores = self._run_node(
                        comp_idx, latent_idx, node_tokens, all_upstream, logger
                    )
                t_node = (time.perf_counter() - t0_node) * 1000

                n_activators = 0
                n_active_inhibitors = 0
                n_absent_inhibitors = 0
                n_enqueued = 0

                # --- Attribution scores: activators and active inhibitors ---
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
                        })
                        circuit.add_node(upstream_node)
                        fid_to_uuid[upstream_fid] = upstream_node.uuid

                    circuit.add_edge(
                        fid_to_uuid[upstream_fid],
                        fid_to_uuid[fid],
                        weight=score,
                    )

                    if role == "activator":
                        n_activators += 1
                        # Only activators trigger further expansion.
                        if upstream_fid.layer >= effective_min_layer:
                            _push(upstream_fid.layer, upstream_fid.kind)
                            n_enqueued += 1
                    else:
                        n_active_inhibitors += 1
                        # Active inhibitors are terminal: added to circuit but not expanded.

                # --- Absent-inhibitor scores: never enqueued for expansion ---
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
                            })
                            circuit.add_node(upstream_node)
                            fid_to_uuid[upstream_fid] = upstream_node.uuid

                        circuit.add_edge(
                            fid_to_uuid[upstream_fid],
                            fid_to_uuid[fid],
                            weight=score,
                        )
                        n_absent_inhibitors += 1

                print(
                    f"  [{fid}] activators={n_activators} enqueued={n_enqueued} | "
                    f"active_inhibitors={n_active_inhibitors} | absent_inhibitors={n_absent_inhibitors} | "
                    f"node_total={t_node:.0f}ms"
                )

        logger.stage("discovery", len(circuit.nodes), len(circuit.edges))
        print(
            f"[LayerwiseGradUpstream] Discovery complete: "
            f"{len(circuit.nodes)} nodes, {len(circuit.edges)} edges."
        )

        if len(circuit.nodes) <= 1:
            logger.reject("no upstream nodes found")
            return None

        # 5. Evaluation — identical to GradientUpstreamDiscovery
        # Restrict ablation to only the layers where circuit nodes actually exist.
        circuit_layers: Set[int] = {
            node.feature_id.layer
            for node in circuit.nodes.values()
            if node.feature_id is not None
        }
        print(f"[LayerwiseGradUpstream] Starting evaluation | circuit_layers={sorted(circuit_layers)}")

        eval_tokens = probe_data.pos_tokens[:self.max_ctx_sequences]
        eval_argmax = probe_data.pos_argmax[:self.max_ctx_sequences]
        eval_targets = probe_data.target_tokens[:self.max_ctx_sequences]

        # Disable compile: torch.compile inlines attn/mlp submodule calls, bypassing
        # forward hooks that CircuitPatcher relies on.
        self.inference.disable_compile()
        try:
            if self.pruning_threshold > 0:
                n_before = len(circuit.nodes)
                prune_non_minimal_nodes(
                    self.inference, self.sae_bank, self.avg_acts, circuit,
                    eval_tokens, pos_argmax=eval_argmax,
                    threshold=self.pruning_threshold,
                    circuit_layers=circuit_layers,
                )
                circuit_layers = {
                    node.feature_id.layer
                    for node in circuit.nodes.values()
                    if node.feature_id is not None
                }
                logger.stage(
                    "after pruning", len(circuit.nodes), len(circuit.edges),
                    note=f"removed {n_before - len(circuit.nodes)} nodes",
                )

            up_faith = evaluate_upstream_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                seed_layer, seed_kind, seed_latent_idx,
                eval_tokens, pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )

            final_f = evaluate_faithfulness(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                eval_tokens, pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )

            final_s = evaluate_sufficiency(
                self.inference, self.sae_bank, self.avg_acts, circuit,
                eval_tokens, eval_targets,
                pos_argmax=eval_argmax,
                circuit_layers=circuit_layers,
            )

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
                "discovery_method": "layerwise_gradient_upstream",
                "max_layers_back": self.max_layers_back,
                "top_k_per_node": self.top_k_per_node,
            })
            logger.nodes(list(circuit.nodes.values()))
            logger.accept(len(circuit.nodes), len(circuit.edges))
            return circuit

        logger.reject(
            f"upstream_faithfulness {up_faith:.4f} < min_faithfulness {self.min_faithfulness}"
        )
        return None

    def _run_node_profiled(
        self,
        comp_idx: int,
        latent_idx: int,
        tokens: torch.Tensor,
        all_upstream_comps: List[int],
        logger: CircuitLogger,
        fid: FeatureID,
    ) -> UpstreamScores:
        """
        Identical to _run_node but wrapped in torch.profiler for one-shot diagnostics.

        Prints the top-25 CUDA-time operations to stdout and exports a Chrome trace
        to outputs/profile_layerwise_<fid>.json (open in chrome://tracing or
        https://ui.perfetto.dev).
        """
        trace_path = os.path.join(
            "outputs", f"profile_layerwise_{fid.layer}_{fid.kind}_{fid.index}.json"
        )
        os.makedirs("outputs", exist_ok=True)

        print(f"  [Profiler] Starting torch.profiler for {fid} → {trace_path}")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            result = self._run_node(comp_idx, latent_idx, tokens, all_upstream_comps, logger)

        print("\n" + "=" * 80)
        print(f"[Profiler] Top 25 ops by CUDA time — {fid}")
        print("=" * 80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=25,
        ))

        print(f"\n[Profiler] Top 10 ops by CPU time — {fid}")
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=10,
        ))

        print(f"\n[Profiler] Top 10 ops by self CUDA memory — {fid}")
        print(prof.key_averages().table(
            sort_by="self_cuda_memory_usage",
            row_limit=10,
        ))

        prof.export_chrome_trace(trace_path)
        print(f"[Profiler] Chrome trace saved → {trace_path}")
        print("=" * 80 + "\n")

        return result

    def _run_node(
        self,
        comp_idx: int,
        latent_idx: int,
        tokens: torch.Tensor,
        all_upstream_comps: List[int],
        logger: CircuitLogger,
    ) -> UpstreamScores:
        """
        Runs grad-enabled forward passes in microbatches and accumulates upstream scores.

        Identical in structure to GradientUpstreamDiscovery._run_hop, but receives
        `all_upstream_comps` (every component in every preceding layer) rather than
        deriving direct predecessors from the transformer's causal wiring internally.
        Scores from each microbatch are accumulated by summation; global top-K
        selection is applied once at the end.
        """
        n_kinds = len(self.sae_bank.kinds)
        kinds = self.sae_bank.kinds
        layer, kind_idx = split_component_idx(comp_idx, n_kinds)
        kind = kinds[kind_idx]

        accumulated_attribution: Dict[FeatureID, float] = {}
        accumulated_absent: Dict[FeatureID, float] = {}

        chunks = tokens.split(self.hop_batch_size, dim=0)
        n_chunks = len(chunks)

        self.inference.disable_compile()
        try:
            for chunk_idx, chunk in enumerate(chunks):
                print(
                    f"    [GradPass] Chunk {chunk_idx + 1}/{n_chunks} | "
                    f"comp_{comp_idx}_lat_{latent_idx} | {len(chunk)} seqs | "
                    f"n_upstream={len(all_upstream_comps)}"
                )
                instrument = SAEGraphInstrument(self.sae_bank)
                try:
                    t0_fwd = time.perf_counter()
                    self.inference.forward(
                        chunk,
                        patcher=instrument,
                        grad_enabled=True,
                        return_activations=False,
                        tokenize_final=False,
                    )
                    t_fwd = (time.perf_counter() - t0_fwd) * 1000

                    _, acts_connected, _ = instrument.graph.get_latents(layer, kind)
                    # acts_connected.act is dense [B, T, d_sae] (scatter_ in SAEGraphInstrument)
                    target_acts = acts_connected.act[..., latent_idx]  # [B, T]
                    pos_argmax = target_acts.argmax(dim=-1)             # [B]

                    # Per-chunk top_k = d_sae so every nonzero score survives to the
                    # global accumulation step.  Absent threshold is 0.0 so that
                    # cross-chunk contributions aren't silently dropped before summing.
                    t0_bwd = time.perf_counter()
                    chunk_scores = compute_latent_upstream_scores(
                        instrument.graph,
                        target_layer=layer,
                        target_kind=kind,
                        target_latent_idx=latent_idx,
                        pos_argmax=pos_argmax,
                        predecessor_comp_indices=all_upstream_comps,
                        n_kinds=n_kinds,
                        kinds=kinds,
                        top_k=self.sae_bank.d_sae,
                        min_active_count=self.min_active_count,
                        active_count=latent_stats.active_count,
                        absent_inhibitor_top_k=(
                            self.sae_bank.d_sae if self.absent_inhibitor_top_k > 0 else 0
                        ),
                        absent_inhibitor_threshold=0.0,
                    )
                    t_bwd = (time.perf_counter() - t0_bwd) * 1000

                    for fid, s in chunk_scores.attribution.items():
                        accumulated_attribution[fid] = accumulated_attribution.get(fid, 0.0) + s
                    for fid, s in chunk_scores.absent_gradient.items():
                        accumulated_absent[fid] = accumulated_absent.get(fid, 0.0) + s

                    print(
                        f"    [GradTiming] fwd={t_fwd:.0f}ms | bwd={t_bwd:.0f}ms"
                    )

                finally:
                    t0_gc = time.perf_counter()
                    del instrument
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    t_gc = (time.perf_counter() - t0_gc) * 1000
                    if t_gc > 50:
                        print(f"    [GradTiming] cleanup={t_gc:.0f}ms")

        finally:
            self.inference.enable_compile()

        # Global top-K across all accumulated microbatches.
        top_attribution = sorted(
            accumulated_attribution.items(), key=lambda x: abs(x[1]), reverse=True
        )
        attribution_dict = {fid: s for fid, s in top_attribution[:self.top_k_per_node]}

        # Apply the real absent-inhibitor threshold on accumulated scores, then top-K.
        top_absent = sorted(
            [(fid, s) for fid, s in accumulated_absent.items()
             if s < -self.absent_inhibitor_threshold],
            key=lambda x: x[1],  # most-negative first
        )
        absent_dict = {fid: s for fid, s in top_absent[:self.absent_inhibitor_top_k]}

        print(
            f"    [GradAccum] {n_chunks} chunks | {len(all_upstream_comps)} upstream comps | "
            f"attr: {len(attribution_dict)} top nodes | absent: {len(absent_dict)} absent inhibitors"
        )
        return UpstreamScores(attribution=attribution_dict, absent_gradient=absent_dict)
