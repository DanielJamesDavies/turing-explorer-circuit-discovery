from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

import torch

from .ablation_gradient import AblationGradientDiscovery
from .base import DiscoveryMethod
from .counterfactual_gradient import CounterfactualGradientDiscovery
from circuit.types.feature_id import FeatureID
from config import config
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.minimality import prune_non_minimal_nodes_cf, prune_non_minimal_nodes_suppression
from eval.magnitude_prune import prune_by_magnitude_bisection
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx
from store.circuits import Circuit, CircuitEdge, CircuitNode
from utils.neg_context_selector import NegContextSelector


class HybridGradientDiscovery(DiscoveryMethod):
    """
    Run the counterfactual and ablation gradient methods normally, then fuse any
    returned circuits into a single candidate and re-score the fused circuit.
    """

    method_name = "hybrid_gradient"

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
    ):
        super().__init__(inference, sae_bank, avg_acts, probe_builder)
        cfg = config.discovery.hybrid_gradient
        self.run_counterfactual = bool(cfg.run_counterfactual)
        self.run_ablation = bool(cfg.run_ablation)
        self.min_counterfactual_faithfulness = float(cfg.min_counterfactual_faithfulness)
        self.min_suppression_score = float(cfg.min_suppression_score)
        self.acceptance_mode = cast(str, cfg.acceptance_mode)
        self.pruning_enabled = bool(cfg.pruning_enabled)
        self.pruning_method = cast(str, cfg.pruning_method)
        self.pruning_threshold = float(cfg.pruning_threshold)
        self.pruning_objective = cast(str, cfg.pruning_objective)
        self.sfc_node_threshold = float(cfg.sfc_node_threshold)
        self.sfc_edge_threshold = float(cfg.sfc_edge_threshold)
        self.sfc_score_mode = cast(str, cfg.sfc_score_mode)
        self.probe_batch_size = cast(int, config.discovery.probe_batch_size)
        self.eval_sequence_count = cast(int, config.discovery.eval_sequence_count)
        self.eval_batch_size = cast(int, config.discovery.eval_batch_size)
        self.magnitude_prune = cast(bool, config.discovery.magnitude_prune)
        self.magnitude_prune_tolerance = cast(float, config.discovery.magnitude_prune_tolerance)
        self.magnitude_prune_target = cast(float, config.discovery.magnitude_prune_target)
        self.magnitude_prune_min_keep = cast(int, config.discovery.magnitude_prune_min_keep)
        self.magnitude_prune_objective = cast(str, config.discovery.magnitude_prune_objective)
        self._last_neg_selection_metadata: dict[str, Any] = {}

        self.counterfactual_method = CounterfactualGradientDiscovery(
            inference, sae_bank, avg_acts, probe_builder
        )
        self.ablation_method = AblationGradientDiscovery(
            inference, sae_bank, avg_acts, probe_builder
        )
        # Prune the fused union once (below), not each source pre-fusion — a joint
        # prune can drop redundancy that spans the two sources.
        self.counterfactual_method.magnitude_prune = False
        self.ablation_method.magnitude_prune = False

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

        source_circuits: List[Tuple[str, Circuit]] = []
        cf_returned = False
        ablation_returned = False

        if self.run_counterfactual:
            cf_circuit = self.counterfactual_method.discover(seed_comp_idx, seed_latent_idx)
            cf_returned = cf_circuit is not None
            if cf_circuit is not None:
                source_circuits.append((CounterfactualGradientDiscovery.method_name, cf_circuit))

        if self.run_ablation:
            ablation_circuit = self.ablation_method.discover(seed_comp_idx, seed_latent_idx)
            ablation_returned = ablation_circuit is not None
            if ablation_circuit is not None:
                source_circuits.append((AblationGradientDiscovery.method_name, ablation_circuit))

        logger.stage(
            "source discovery",
            sum(len(c.nodes) for _, c in source_circuits),
            sum(len(c.edges) for _, c in source_circuits),
            note=(
                f"counterfactual_returned={cf_returned} "
                f"ablation_returned={ablation_returned}"
            ),
        )

        if not source_circuits:
            logger.reject("both source methods returned no circuit")
            return None

        try:
            fused = fuse_circuits_by_feature_id(
                source_circuits,
                seed_comp_idx=seed_comp_idx,
                seed_latent_idx=seed_latent_idx,
                kinds=kinds,
            )
        except ValueError as error:
            logger.reject(str(error))
            return None

        logger.stage("fused circuit", len(fused.nodes), len(fused.edges))
        pre_prune_source_overlap = compute_source_overlap(
            fused,
            seed_comp_idx=seed_comp_idx,
            seed_latent_idx=seed_latent_idx,
            kinds=kinds,
        )

        probe_data = self.build_probe_dataset(seed_comp_idx, seed_latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            logger.reject("empty probe dataset (no positive contexts)")
            return None

        # Hybrid's own slices are all evaluation-side (fusion pruning + final
        # evals); the sub-methods slice their discovery inputs internally.
        pos_tokens_eval = probe_data.pos_tokens[: self.eval_sequence_count]
        pos_argmax_eval = probe_data.pos_argmax[: self.eval_sequence_count]
        neg_selection = self._select_neg_context(
            seed_comp_idx,
            seed_latent_idx,
            logger,
        )
        if neg_selection is None:
            return None
        neg_tokens_eval = neg_selection.tokens
        self._last_neg_selection_metadata = neg_selection.metadata

        circuit_layers = _circuit_layers(fused)
        pre_prune_nodes = len(fused.nodes)
        pre_prune_edges = len(fused.edges)

        should_prune = self.pruning_enabled and (
            (self.pruning_method == "leave_one_out" and self.pruning_threshold > 0)
            or self.pruning_method == "sfc_threshold"
        )
        if should_prune:
            removed = self._prune(
                fused,
                neg_tokens_eval=neg_tokens_eval,
                pos_tokens_eval=pos_tokens_eval,
                seed_layer=seed_layer,
                seed_kind=seed_kind,
                seed_latent_idx=seed_latent_idx,
                pos_argmax_eval=pos_argmax_eval,
                circuit_layers=circuit_layers,
            )
            circuit_layers = _circuit_layers(fused)
            logger.stage(
                "after pruning",
                len(fused.nodes),
                len(fused.edges),
                note=(
                    f"method={self.pruning_method} "
                    f"objective={self.pruning_objective} "
                    f"threshold={self.pruning_threshold} "
                    f"sfc_node_threshold={self.sfc_node_threshold} "
                    f"sfc_edge_threshold={self.sfc_edge_threshold} "
                    f"removed={len(removed)}"
                ),
            )

        # Global magnitude prune (optional) — scalable free-φ bisection over the
        # fused union, for the large position-aware allowed sets.
        if self.magnitude_prune:
            prune_by_magnitude_bisection(
                self.inference, self.sae_bank, fused,
                pos_tokens=pos_tokens_eval,
                seed_layer=seed_layer, seed_kind=seed_kind, seed_latent_idx=seed_latent_idx,
                pos_argmax=pos_argmax_eval,
                tolerance=self.magnitude_prune_tolerance,
                target=self.magnitude_prune_target,
                min_keep=self.magnitude_prune_min_keep,
                objective=self.magnitude_prune_objective,
                logger=logger,
            )
            circuit_layers = _circuit_layers(fused)

        post_prune_source_overlap = compute_source_overlap(
            fused,
            seed_comp_idx=seed_comp_idx,
            seed_latent_idx=seed_latent_idx,
            kinds=kinds,
        )
        cf_faith, sup_score = evaluate_counterfactual_faithfulness(
            self.inference,
            self.sae_bank,
            self.avg_acts,
            fused,
            neg_tokens=neg_tokens_eval,
            pos_tokens=pos_tokens_eval,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            pos_argmax=pos_argmax_eval,
            circuit_layers=circuit_layers,
        )
        logger.note(
            f"counterfactual_faithfulness: {cf_faith:.4f} | "
            f"posctx_suppression_score: {sup_score:.4f}"
        )

        fused.metadata.update(
            {
                "counterfactual_faithfulness": cf_faith,
                "posctx_suppression_score": sup_score,
                "ablation_suppression_score": sup_score,
                "seed_comp": seed_comp_idx,
                "seed_latent": seed_latent_idx,
                "n_nodes": len(fused.nodes),
                "n_edges": len(fused.edges),
                "discovery_method": self.method_name,
                "hybrid_source_methods": [name for name, _ in source_circuits],
                "source_counterfactual_returned": cf_returned,
                "source_ablation_returned": ablation_returned,
                "acceptance_mode": self.acceptance_mode,
                "pruning_enabled": self.pruning_enabled,
                "pruning_method": self.pruning_method,
                "pruning_objective": self.pruning_objective,
                "pruning_threshold": self.pruning_threshold,
                "sfc_node_threshold": self.sfc_node_threshold,
                "sfc_edge_threshold": self.sfc_edge_threshold,
                "sfc_score_mode": self.sfc_score_mode,
                "pre_prune_node_count": pre_prune_nodes,
                "pre_prune_edge_count": pre_prune_edges,
                "post_prune_node_count": len(fused.nodes),
                "post_prune_edge_count": len(fused.edges),
                "source_overlap": {
                    "pre_prune": pre_prune_source_overlap,
                    "post_prune": post_prune_source_overlap,
                },
                **_flat_source_overlap_metadata(
                    pre_prune_source_overlap,
                    prefix="source",
                ),
                **_flat_source_overlap_metadata(
                    post_prune_source_overlap,
                    prefix="post_prune",
                ),
                "neg_mode": self.counterfactual_method.neg_mode,
                "neg_selection": dict(self._last_neg_selection_metadata),
            }
        )

        if not self._passes_acceptance(cf_faith, sup_score):
            logger.reject(
                f"hybrid scores failed acceptance_mode={self.acceptance_mode} "
                f"(cf={cf_faith:.4f}, sup={sup_score:.4f})"
            )
            return None

        logger.nodes(list(fused.nodes.values()))
        logger.accept(len(fused.nodes), len(fused.edges))
        return fused

    def _passes_acceptance(self, cf_faith: float, sup_score: float) -> bool:
        cf_ok = cf_faith >= self.min_counterfactual_faithfulness
        sup_ok = sup_score >= self.min_suppression_score
        if self.acceptance_mode == "cf":
            return cf_ok
        if self.acceptance_mode == "suppression":
            return sup_ok
        if self.acceptance_mode == "both":
            return cf_ok and sup_ok
        return cf_ok or sup_ok

    def _select_neg_context(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        logger: CircuitLogger,
    ):
        mode = self.counterfactual_method.neg_mode
        cfg = config.discovery.neg_context_selection
        candidate_pool_size = (
            self.counterfactual_method.distant_pool_size
            if mode == "distant"
            else cfg.candidate_pool_size
        )
        selection = self._neg_context_selector().select(
            seed_comp_idx,
            seed_latent_idx,
            mode,
            max_sequences=self.counterfactual_method.max_neg_sequences,
            batch_size=self.counterfactual_method.neg_batch_size,
            candidate_pool_size=candidate_pool_size,
            exact=bool(cfg.exact_negctx_ranking),
            non_activation_threshold=float(cfg.non_activation_threshold),
            selection_seed=int(cfg.selection_seed),
            filter_batch_size=int(cfg.filter_batch_size),
            load_window_size=int(cfg.load_window_size),
            logger=logger,
        )
        if selection is None:
            logger.reject(f"neg_mode={mode}: no hybrid eval negative sequences available")
        return selection

    def _neg_context_selector(self) -> NegContextSelector:
        from store.context import mid_ctx, neg_ctx, top_ctx
        from store.seq_repr import seq_repr

        if seq_repr is None:
            raise RuntimeError("seq_repr must be loaded before negative-context selection")

        return NegContextSelector(
            self.inference,
            self.sae_bank,
            self.probe_builder.loader,
            neg_ctx,
            seq_repr,
            top_ctx,
            mid_ctx,
        )

    def _prune(
        self,
        circuit: Circuit,
        *,
        neg_tokens_eval: torch.Tensor,
        pos_tokens_eval: torch.Tensor,
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        pos_argmax_eval: torch.Tensor,
        circuit_layers: Set[int],
    ) -> List[str]:
        if self.pruning_method == "sfc_threshold":
            return prune_sfc_threshold(
                circuit,
                node_threshold=self.sfc_node_threshold,
                edge_threshold=self.sfc_edge_threshold,
            )
        if self.pruning_objective == "cf":
            return prune_non_minimal_nodes_cf(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                neg_tokens=neg_tokens_eval,
                pos_tokens=pos_tokens_eval,
                seed_layer=seed_layer,
                seed_kind=seed_kind,
                seed_latent_idx=seed_latent_idx,
                pos_argmax=pos_argmax_eval,
                threshold=self.pruning_threshold,
                circuit_layers=circuit_layers,
            )
        if self.pruning_objective == "suppression":
            return prune_non_minimal_nodes_suppression(
                self.inference,
                self.sae_bank,
                self.avg_acts,
                circuit,
                neg_tokens=neg_tokens_eval,
                pos_tokens=pos_tokens_eval,
                seed_layer=seed_layer,
                seed_kind=seed_kind,
                seed_latent_idx=seed_latent_idx,
                pos_argmax=pos_argmax_eval,
                threshold=self.pruning_threshold,
                circuit_layers=circuit_layers,
            )
        return prune_non_minimal_nodes_both(
            self.inference,
            self.sae_bank,
            self.avg_acts,
            circuit,
            neg_tokens=neg_tokens_eval,
            pos_tokens=pos_tokens_eval,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            pos_argmax=pos_argmax_eval,
            threshold=self.pruning_threshold,
            circuit_layers=circuit_layers,
        )


def fuse_circuits_by_feature_id(
    source_circuits: Iterable[Tuple[str, Circuit]],
    *,
    seed_comp_idx: int,
    seed_latent_idx: int,
    kinds: List[str],
) -> Circuit:
    sources = list(source_circuits)
    if not sources:
        raise ValueError("no source circuits to fuse")

    n_kinds = len(kinds)
    seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
    seed_fid = FeatureID(seed_layer, kinds[seed_kind_idx], seed_latent_idx)

    fused = Circuit(name=f"HybridGrad_S{seed_comp_idx}_{seed_latent_idx}")
    fused.metadata["source_circuit_uuids"] = {
        method_name: circuit.uuid for method_name, circuit in sources
    }

    fid_to_node: Dict[FeatureID, CircuitNode] = {}
    old_uuid_to_fid: Dict[str, FeatureID] = {}

    for method_name, circuit in sources:
        if not _circuit_has_seed(circuit, seed_fid):
            raise ValueError(f"{method_name} circuit does not contain expected seed {seed_fid}")
        for node in circuit.nodes.values():
            fid = node.feature_id
            if fid is None:
                continue
            old_uuid_to_fid[node.uuid] = fid
            if fid not in fid_to_node:
                new_node = CircuitNode(metadata=_metadata_with_source(node.metadata, method_name))
                fused.add_node(new_node)
                fid_to_node[fid] = new_node
            else:
                _merge_node_metadata(fid_to_node[fid].metadata, node.metadata, method_name)

    edge_by_fids: Dict[Tuple[FeatureID, FeatureID], CircuitEdge] = {}
    for method_name, circuit in sources:
        for edge in circuit.edges:
            source_fid = old_uuid_to_fid.get(edge.source_uuid)
            target_fid = old_uuid_to_fid.get(edge.target_uuid)
            if source_fid is None or target_fid is None:
                continue
            source_node = fid_to_node[source_fid]
            target_node = fid_to_node[target_fid]
            key = (source_fid, target_fid)
            if key not in edge_by_fids:
                new_edge = fused.add_edge(
                    source_node.uuid,
                    target_node.uuid,
                    **_edge_metadata_with_source(edge.metadata, method_name),
                )
                edge_by_fids[key] = new_edge
            else:
                _merge_edge_metadata(edge_by_fids[key].metadata, edge.metadata, method_name)

    if seed_fid not in fid_to_node:
        raise ValueError(f"fused circuit missing seed {seed_fid}")
    return fused


def compute_source_overlap(
    circuit: Circuit,
    *,
    seed_comp_idx: int,
    seed_latent_idx: int,
    kinds: List[str],
) -> Dict[str, Any]:
    n_kinds = len(kinds)
    seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
    seed_fid = FeatureID(seed_layer, kinds[seed_kind_idx], seed_latent_idx)
    buckets: dict[str, list[FeatureID]] = {
        "cf_only": [],
        "ablation_only": [],
        "intersection": [],
        "unknown": [],
    }
    by_layer: dict[str, dict[str, int]] = defaultdict(_empty_overlap_counts)
    by_kind: dict[str, dict[str, int]] = defaultdict(_empty_overlap_counts)
    by_role: dict[str, dict[str, int]] = defaultdict(_empty_overlap_counts)
    seed_count = 0

    for node in circuit.nodes.values():
        fid = node.feature_id
        if fid is None:
            continue
        if fid == seed_fid or node.metadata.get("role") == "seed":
            seed_count += 1
            continue
        bucket = _source_overlap_bucket(node.metadata.get("source_methods", []))
        buckets[bucket].append(fid)
        _increment_overlap_group(by_layer[str(fid.layer)], bucket)
        _increment_overlap_group(by_kind[str(fid.kind)], bucket)
        roles = node.metadata.get("roles")
        if isinstance(roles, list) and roles:
            role_key = "+".join(str(role) for role in sorted(roles))
        else:
            role_key = str(node.metadata.get("role", "unknown"))
        _increment_overlap_group(by_role[role_key], bucket)

    counts = {
        "cf_only_node_count": len(buckets["cf_only"]),
        "ablation_only_node_count": len(buckets["ablation_only"]),
        "intersection_node_count": len(buckets["intersection"]),
        "unknown_node_count": len(buckets["unknown"]),
        "seed_node_count": seed_count,
    }
    cf_count = counts["cf_only_node_count"] + counts["intersection_node_count"]
    ablation_count = counts["ablation_only_node_count"] + counts["intersection_node_count"]
    union_count = (
        counts["cf_only_node_count"]
        + counts["ablation_only_node_count"]
        + counts["intersection_node_count"]
    )
    counts.update(
        {
            "cf_node_count": cf_count,
            "ablation_node_count": ablation_count,
            "union_node_count": union_count,
            "jaccard": (counts["intersection_node_count"] / union_count) if union_count else 0.0,
        }
    )
    return {
        **counts,
        "by_layer": dict(sorted(by_layer.items(), key=lambda item: int(item[0]))),
        "by_kind": {kind: by_kind[kind] for kind in kinds if kind in by_kind},
        "by_role": dict(sorted(by_role.items())),
    }


def prune_sfc_threshold(
    circuit: Circuit,
    *,
    node_threshold: float,
    edge_threshold: float,
) -> List[str]:
    """
    SFC-style metadata pruning: threshold node and edge attribution scores once.

    This is intentionally not a causal minimality check. It mirrors the SFC
    discovery style of retaining nodes/edges whose approximate effects clear
    fixed thresholds, so it is cheap enough for calibration/grid runs.
    """
    removed_nodes: List[str] = []

    for node_uuid, node in list(circuit.nodes.items()):
        if node.metadata.get("role") == "seed":
            continue
        score = _abs_float(node.metadata.get("attribution_score"))
        if score < node_threshold:
            circuit.nodes.pop(node_uuid, None)
            removed_nodes.append(node_uuid)

    live_node_uuids = set(circuit.nodes)
    circuit.edges = [
        edge
        for edge in circuit.edges
        if edge.source_uuid in live_node_uuids
        and edge.target_uuid in live_node_uuids
        and _abs_float(edge.weight) >= edge_threshold
    ]

    connected_uuids = {
        uuid
        for edge in circuit.edges
        for uuid in (edge.source_uuid, edge.target_uuid)
    }
    for node_uuid, node in list(circuit.nodes.items()):
        if node.metadata.get("role") == "seed":
            continue
        if node_uuid not in connected_uuids:
            circuit.nodes.pop(node_uuid, None)
            removed_nodes.append(node_uuid)

    return removed_nodes


@torch.no_grad()
def prune_non_minimal_nodes_both(
    inference: Any,
    sae_bank: Any,
    avg_acts: torch.Tensor,
    circuit: Circuit,
    neg_tokens: torch.Tensor,
    pos_tokens: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    pos_argmax: Optional[torch.Tensor] = None,
    threshold: float = 0.05,
    circuit_layers: Optional[Set[int]] = None,
    max_candidates_per_iter: int = 32,
    max_iterations: int = 50,
) -> List[str]:
    removed_nodes: List[str] = []

    for _iter in range(max_iterations):
        base_cf, base_sup = evaluate_counterfactual_faithfulness(
            inference,
            sae_bank,
            avg_acts,
            circuit,
            neg_tokens=neg_tokens,
            pos_tokens=pos_tokens,
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent_idx,
            pos_argmax=pos_argmax,
            circuit_layers=circuit_layers,
        )

        candidates: List[tuple[float, str]] = []
        for node_uuid, node in circuit.nodes.items():
            if node.metadata.get("role") == "seed":
                continue
            score = float(node.metadata.get("attribution_score") or 0.0)
            candidates.append((score, node_uuid))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        eval_candidates = [uuid for _, uuid in candidates[:max_candidates_per_iter]]

        loo_drops: Dict[str, Tuple[float, float]] = {}
        original_nodes = circuit.nodes
        for node_uuid in eval_candidates:
            circuit.nodes = {k: v for k, v in original_nodes.items() if k != node_uuid}
            loo_cf, loo_sup = evaluate_counterfactual_faithfulness(
                inference,
                sae_bank,
                avg_acts,
                circuit,
                neg_tokens=neg_tokens,
                pos_tokens=pos_tokens,
                seed_layer=seed_layer,
                seed_kind=seed_kind,
                seed_latent_idx=seed_latent_idx,
                pos_argmax=pos_argmax,
                circuit_layers=circuit_layers,
            )
            loo_drops[node_uuid] = (base_cf - loo_cf, base_sup - loo_sup)
            circuit.nodes = original_nodes

        prunable = [
            (max(cf_drop, sup_drop), node_uuid)
            for node_uuid, (cf_drop, sup_drop) in loo_drops.items()
            if cf_drop < threshold and sup_drop < threshold
        ]
        if not prunable:
            break

        _, least_uuid = min(prunable, key=lambda item: item[0])
        circuit.nodes.pop(least_uuid)
        circuit.edges = [
            edge
            for edge in circuit.edges
            if edge.source_uuid != least_uuid and edge.target_uuid != least_uuid
        ]
        removed_nodes.append(least_uuid)

    return removed_nodes


def _abs_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def _circuit_has_seed(circuit: Circuit, seed_fid: FeatureID) -> bool:
    return any(
        node.feature_id == seed_fid and node.metadata.get("role") == "seed"
        for node in circuit.nodes.values()
    )


def _metadata_with_source(metadata: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    copied = copy.deepcopy(metadata)
    copied["source_methods"] = sorted({method_name, *copied.get("source_methods", [])})
    copied["roles"] = sorted({copied.get("role", "unknown"), *copied.get("roles", [])})
    score = copied.get("attribution_score")
    if score is not None:
        copied["attribution_scores"] = {method_name: score}
    return copied


def _merge_node_metadata(target: Dict[str, Any], incoming: Dict[str, Any], method_name: str) -> None:
    incoming_role = incoming.get("role", "unknown")
    target["source_methods"] = sorted({method_name, *target.get("source_methods", [])})
    target["roles"] = sorted({incoming_role, *target.get("roles", [])})

    if target.get("role") != "seed" and incoming_role == "counterfactual_inhibitor":
        target["role"] = incoming_role
    elif target.get("role") not in ("seed", "counterfactual_inhibitor"):
        target["role"] = incoming_role

    score = incoming.get("attribution_score")
    if score is not None:
        scores = dict(target.get("attribution_scores", {}))
        scores[method_name] = score
        target["attribution_scores"] = scores
        existing_score = target.get("attribution_score")
        if existing_score is None or abs(float(score)) > abs(float(existing_score)):
            target["attribution_score"] = score


def _edge_metadata_with_source(metadata: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    copied = copy.deepcopy(metadata)
    copied["source_methods"] = sorted({method_name, *copied.get("source_methods", [])})
    weight = copied.get("weight")
    if weight is not None:
        copied["weights_by_method"] = {method_name: weight}
    return copied


def _merge_edge_metadata(target: Dict[str, Any], incoming: Dict[str, Any], method_name: str) -> None:
    target["source_methods"] = sorted({method_name, *target.get("source_methods", [])})
    weight = incoming.get("weight")
    if weight is not None:
        weights = dict(target.get("weights_by_method", {}))
        weights[method_name] = weight
        target["weights_by_method"] = weights
        existing_weight = target.get("weight")
        if existing_weight is None or abs(float(weight)) > abs(float(existing_weight)):
            target["weight"] = weight


def _source_overlap_bucket(source_methods_value: Any) -> str:
    source_methods = (
        {str(method) for method in source_methods_value if method is not None}
        if isinstance(source_methods_value, list)
        else set()
    )
    has_cf = CounterfactualGradientDiscovery.method_name in source_methods
    has_ablation = AblationGradientDiscovery.method_name in source_methods
    if has_cf and has_ablation:
        return "intersection"
    if has_cf:
        return "cf_only"
    if has_ablation:
        return "ablation_only"
    return "unknown"


def _empty_overlap_counts() -> dict[str, int]:
    return {
        "cf_only_node_count": 0,
        "ablation_only_node_count": 0,
        "intersection_node_count": 0,
        "unknown_node_count": 0,
        "union_node_count": 0,
    }


def _increment_overlap_group(group: dict[str, int], bucket: str) -> None:
    key = f"{bucket}_node_count"
    group[key] = int(group.get(key, 0)) + 1
    if bucket != "unknown":
        group["union_node_count"] = int(group.get("union_node_count", 0)) + 1


def _flat_source_overlap_metadata(overlap: Dict[str, Any], *, prefix: str) -> Dict[str, Any]:
    return {
        f"{prefix}_cf_node_count": overlap.get("cf_node_count", 0),
        f"{prefix}_ablation_node_count": overlap.get("ablation_node_count", 0),
        f"{prefix}_intersection_node_count": overlap.get("intersection_node_count", 0),
        f"{prefix}_union_node_count": overlap.get("union_node_count", 0),
        f"{prefix}_cf_only_node_count": overlap.get("cf_only_node_count", 0),
        f"{prefix}_ablation_only_node_count": overlap.get("ablation_only_node_count", 0),
        f"{prefix}_jaccard": overlap.get("jaccard", 0.0),
    }


def _circuit_layers(circuit: Circuit) -> Set[int]:
    return {
        node.feature_id.layer
        for node in circuit.nodes.values()
        if node.feature_id is not None
    }


__all__ = [
    "HybridGradientDiscovery",
    "compute_source_overlap",
    "fuse_circuits_by_feature_id",
    "prune_non_minimal_nodes_both",
]
