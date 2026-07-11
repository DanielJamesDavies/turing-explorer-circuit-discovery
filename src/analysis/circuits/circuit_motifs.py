"""Mine recurring circuit motifs and first-pass circuit families."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import (
    BAR_WIDTH,
    BLUE,
    configure_matplotlib,
    integer_ticks,
    panel_figsize,
    round_bars,
    save_figure,
    style_suptitle,
)
from circuit.types.feature_id import FeatureID
from store.circuits import Circuit

from .coact_overlap import load_circuit_summary_rows
from .node_hop_overlap import DEFAULT_KINDS, load_circuit_store, resolve_circuit_store_path


SUITE_NAME = "circuit-motifs"
_ACTIVATOR_ROLES = {"counterfactual_activator", "ablation_support", "cluster_activator"}
_INHIBITOR_ROLES = {"counterfactual_inhibitor", "cluster_inhibitor"}


@dataclass(frozen=True)
class NormalizedNode:
    uuid: str
    layer: int
    kind: str
    latent_idx: int
    role: str
    is_seed: bool

    @property
    def exact_key(self) -> tuple[int, str, int, str]:
        return (self.layer, self.kind, self.latent_idx, self.role)

    @property
    def abstract_key(self) -> tuple[int, str, str]:
        return (self.layer, self.kind, self.role)


@dataclass(frozen=True)
class NormalizedEdge:
    source_uuid: str
    target_uuid: str
    weight: float
    sign: str


@dataclass(frozen=True)
class NormalizedCircuit:
    uuid: str
    name: str
    seed_comp: int
    seed_latent: int
    nodes: dict[str, NormalizedNode]
    edges: list[NormalizedEdge]
    faithfulness: float
    posctx_suppression_score: float
    post_analysis: Mapping[str, Any]


@dataclass(frozen=True)
class MotifInstance:
    circuit_uuid: str
    circuit_name: str
    motif_level: str
    motif_kind: str
    signature: str
    exact_signature: str
    abstract_signature: str
    node_uuids: tuple[str, ...]
    edge_indices: tuple[int, ...]
    exact_node_keys: tuple[tuple[int, str, int, str], ...]
    edge_weights: tuple[float, ...]


@dataclass(frozen=True)
class CircuitMotifAnalysisResult:
    figure_path: Path
    summary_path: Path
    motifs_table_path: Path
    membership_table_path: Path
    cohesion_table_path: Path
    family_table_path: Path
    summary: dict[str, object]

    @property
    def table_path(self) -> Path:
        return self.motifs_table_path


def plot_circuit_motifs(
    run_root: str | Path,
    *,
    circuit_store_path: str | Path | None = None,
    output_root: str | Path | None = None,
    min_support: int = 2,
    high_faithfulness_quantile: float = 0.75,
    similarity_threshold: float = 0.05,
    max_pair_motif_support: int = 512,
    max_edges_per_circuit: int = 32,
    max_fan_in_edges_per_target: int = 12,
    max_signed_pair_edges_per_role: int = 8,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> CircuitMotifAnalysisResult:
    """Generate motif and circuit-family analysis outputs from discovered circuits."""

    root = resolve_run_root(run_root)
    store_path = resolve_circuit_store_path(root, circuit_store_path)
    circuits = load_circuit_store(store_path)
    summary_rows = _try_load_summary_rows(root / "circuits" / "summary.json")
    stats = compute_circuit_motifs(
        circuits,
        summary_rows=summary_rows,
        min_support=min_support,
        high_faithfulness_quantile=high_faithfulness_quantile,
        similarity_threshold=similarity_threshold,
        max_pair_motif_support=max_pair_motif_support,
        max_edges_per_circuit=max_edges_per_circuit,
        max_fan_in_edges_per_target=max_fan_in_edges_per_target,
        max_signed_pair_edges_per_role=max_signed_pair_edges_per_role,
        kinds=kinds,
    )

    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "circuit-motifs.png"
    motifs_table_path = output_dirs["tables"] / "motifs.csv"
    membership_table_path = output_dirs["tables"] / "circuit-motif-membership.csv"
    cohesion_table_path = output_dirs["tables"] / "circuit-cohesion.csv"
    family_table_path = output_dirs["tables"] / "circuit-family-membership.csv"
    summary_path = output_dirs["summaries"] / "circuit-motif-analysis.json"

    _write_plot(figure_path, stats)
    _write_motif_table(motifs_table_path, stats)
    _write_membership_table(membership_table_path, stats)
    _write_cohesion_table(cohesion_table_path, stats)
    _write_family_table(family_table_path, stats)
    summary = _build_summary(store_path, stats)
    write_json(summary_path, summary)

    return CircuitMotifAnalysisResult(
        figure_path=figure_path,
        summary_path=summary_path,
        motifs_table_path=motifs_table_path,
        membership_table_path=membership_table_path,
        cohesion_table_path=cohesion_table_path,
        family_table_path=family_table_path,
        summary=summary,
    )


def compute_circuit_motifs(
    circuits: Mapping[str, Circuit],
    *,
    summary_rows: Sequence[Mapping[str, Any]] | None = None,
    min_support: int = 2,
    high_faithfulness_quantile: float = 0.75,
    similarity_threshold: float = 0.05,
    max_pair_motif_support: int = 512,
    max_edges_per_circuit: int = 32,
    max_fan_in_edges_per_target: int = 12,
    max_signed_pair_edges_per_role: int = 8,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> dict[str, object]:
    """Mine small recurring motifs and derive circuit-family memberships."""

    if min_support <= 0:
        raise ValueError("min_support must be positive")
    if not 0.0 <= high_faithfulness_quantile <= 1.0:
        raise ValueError("high_faithfulness_quantile must be between 0 and 1")
    if similarity_threshold < 0.0:
        raise ValueError("similarity_threshold must be non-negative")
    if max_edges_per_circuit <= 0:
        raise ValueError("max_edges_per_circuit must be positive")
    if max_fan_in_edges_per_target <= 1:
        raise ValueError("max_fan_in_edges_per_target must be greater than 1")
    if max_signed_pair_edges_per_role <= 0:
        raise ValueError("max_signed_pair_edges_per_role must be positive")

    summary_lookup = _summary_lookup(summary_rows or [])
    normalized: dict[str, NormalizedCircuit] = {}
    for circuit in circuits.values():
        graph = normalize_circuit(circuit, summary_lookup=summary_lookup, kinds=kinds)
        if graph is not None and graph.nodes:
            normalized[graph.uuid] = graph

    latent_support = _exact_latent_support(normalized.values())
    instances = []
    for graph in normalized.values():
        instances.extend(
            mine_circuit_motifs(
                graph,
                max_edges_per_circuit=max_edges_per_circuit,
                max_fan_in_edges_per_target=max_fan_in_edges_per_target,
                max_signed_pair_edges_per_role=max_signed_pair_edges_per_role,
            )
        )

    motif_groups: dict[tuple[str, str, str], list[MotifInstance]] = defaultdict(list)
    for instance in instances:
        motif_groups[(instance.motif_level, instance.motif_kind, instance.signature)].append(instance)

    faithfulness_values = [graph.faithfulness for graph in normalized.values()]
    high_threshold = _quantile(sorted(faithfulness_values), high_faithfulness_quantile) if faithfulness_values else 0.0
    high_circuit_uuids = {uuid for uuid, graph in normalized.items() if graph.faithfulness >= high_threshold}

    motif_rows: list[dict[str, Any]] = []
    retained_groups: dict[str, list[MotifInstance]] = {}
    retained_scores: dict[str, float] = {}
    retained_supports: dict[str, int] = {}
    for motif_id, (_key, group) in enumerate(
        _sorted_supported_groups(motif_groups, min_support=min_support),
        start=1,
    ):
        row, score = _motif_row(
            f"M{motif_id:06d}",
            group,
            circuit_count=len(normalized),
            graphs=normalized,
            high_circuit_uuids=high_circuit_uuids,
            latent_support=latent_support,
        )
        motif_rows.append(row)
        retained_groups[str(row["motif_id"])] = group
        retained_scores[str(row["motif_id"])] = score
        retained_supports[str(row["motif_id"])] = int(row["support_count"])

    membership_rows = _membership_rows(normalized, retained_groups, retained_scores)
    family_rows, projected_edge_count, skipped_pair_motifs = _family_rows(
        normalized,
        membership_rows,
        retained_supports,
        similarity_threshold=similarity_threshold,
        max_pair_motif_support=max_pair_motif_support,
    )
    cohesion_rows = [_cohesion_row(graph, membership_rows) for graph in normalized.values()]
    cohesion_rows.sort(key=lambda row: str(row["uuid"]))

    return {
        "circuit_count": len(normalized),
        "raw_motif_instance_count": len(instances),
        "motif_count": len(motif_rows),
        "min_support": int(min_support),
        "high_faithfulness_quantile": float(high_faithfulness_quantile),
        "high_faithfulness_threshold": float(high_threshold),
        "similarity_threshold": float(similarity_threshold),
        "max_pair_motif_support": int(max_pair_motif_support),
        "max_edges_per_circuit": int(max_edges_per_circuit),
        "max_fan_in_edges_per_target": int(max_fan_in_edges_per_target),
        "max_signed_pair_edges_per_role": int(max_signed_pair_edges_per_role),
        "projected_edge_count": int(projected_edge_count),
        "skipped_pair_motif_count": int(skipped_pair_motifs),
        "motif_rows": motif_rows,
        "membership_rows": membership_rows,
        "cohesion_rows": cohesion_rows,
        "family_rows": family_rows,
        "motif_support_summary": _summary([int(row["support_count"]) for row in motif_rows]),
        "membership_weight_summary": _summary([float(row["membership_weight"]) for row in membership_rows]),
        "family_size_summary": _summary([int(row["hard_family_size"]) for row in family_rows]),
    }


def normalize_circuit(
    circuit: Circuit,
    *,
    summary_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> NormalizedCircuit | None:
    """Convert a free-form Circuit object into an eligible directed graph."""

    seed_comp = _metadata_int(circuit.metadata, "seed_comp")
    seed_latent = _metadata_int(circuit.metadata, "seed_latent")
    n_kinds = len(kinds)
    nodes: dict[str, NormalizedNode] = {}
    for node_uuid, node in circuit.nodes.items():
        fid = node.feature_id
        if not _eligible_feature(fid, kinds):
            continue
        assert fid is not None
        role = str(node.metadata.get("role") or "unknown")
        is_seed = role == "seed" or _is_seed_feature(fid, seed_comp, seed_latent, n_kinds, kinds)
        nodes[str(node_uuid)] = NormalizedNode(
            uuid=str(node_uuid),
            layer=int(fid.layer),
            kind=str(fid.kind),
            latent_idx=int(fid.index),
            role=role,
            is_seed=is_seed,
        )

    if not nodes:
        return None

    edges: list[NormalizedEdge] = []
    for edge in circuit.edges:
        source_uuid = str(edge.source_uuid)
        target_uuid = str(edge.target_uuid)
        if source_uuid not in nodes or target_uuid not in nodes:
            continue
        weight = _numeric(edge.metadata.get("weight"), default=0.0)
        edges.append(
            NormalizedEdge(
                source_uuid=source_uuid,
                target_uuid=target_uuid,
                weight=weight,
                sign=_edge_sign(weight),
            )
        )

    summary = (summary_lookup or {}).get(circuit.uuid, {})
    post = _post_analysis(circuit.metadata, summary)
    return NormalizedCircuit(
        uuid=circuit.uuid,
        name=circuit.name,
        seed_comp=seed_comp,
        seed_latent=seed_latent,
        nodes=nodes,
        edges=edges,
        faithfulness=_first_metric(circuit.metadata, summary, ("evals", "counterfactual_faithfulness"), "counterfactual_faithfulness"),
        posctx_suppression_score=_first_metric(circuit.metadata, summary, ("evals", "posctx_suppression_score"), "posctx_suppression_score"),
        post_analysis=post,
    )


def mine_circuit_motifs(
    graph: NormalizedCircuit,
    *,
    max_edges_per_circuit: int = 32,
    max_fan_in_edges_per_target: int = 12,
    max_signed_pair_edges_per_role: int = 8,
) -> list[MotifInstance]:
    """Mine exact and typed 2-node/3-node motif instances from one graph."""

    instances: list[MotifInstance] = []
    incoming: dict[str, list[int]] = defaultdict(list)
    outgoing: dict[str, list[int]] = defaultdict(list)
    selected_edge_indices = _top_weighted_edge_indices(graph, list(range(len(graph.edges))), max_edges_per_circuit)
    for edge_idx in selected_edge_indices:
        edge = graph.edges[edge_idx]
        outgoing[edge.source_uuid].append(edge_idx)
        incoming[edge.target_uuid].append(edge_idx)
        for level in ("exact", "typed"):
            instances.append(_edge_motif(graph, edge_idx, level=level))

    for first_idx in selected_edge_indices:
        first = graph.edges[first_idx]
        for second_idx in outgoing.get(first.target_uuid, []):
            second = graph.edges[second_idx]
            if len({first.source_uuid, first.target_uuid, second.target_uuid}) < 3:
                continue
            for level in ("exact", "typed"):
                instances.append(_chain_motif(graph, first_idx, second_idx, level=level))

    for target_uuid, edge_indices in incoming.items():
        if len(edge_indices) < 2:
            continue
        fan_in_edges = _top_weighted_edge_indices(graph, edge_indices, max_fan_in_edges_per_target)
        for left_pos, left_idx in enumerate(fan_in_edges):
            for right_idx in fan_in_edges[left_pos + 1 :]:
                left = graph.edges[left_idx]
                right = graph.edges[right_idx]
                if left.source_uuid == right.source_uuid:
                    continue
                for level in ("exact", "typed"):
                    instances.append(_fan_in_motif(graph, left_idx, right_idx, level=level))

    seed_uuids = {node.uuid for node in graph.nodes.values() if node.is_seed}
    for seed_uuid in seed_uuids:
        direct_inputs = incoming.get(seed_uuid, [])
        activators = _top_weighted_edge_indices(
            graph,
            [idx for idx in direct_inputs if graph.nodes[graph.edges[idx].source_uuid].role in _ACTIVATOR_ROLES],
            max_signed_pair_edges_per_role,
        )
        inhibitors = _top_weighted_edge_indices(
            graph,
            [idx for idx in direct_inputs if graph.nodes[graph.edges[idx].source_uuid].role in _INHIBITOR_ROLES],
            max_signed_pair_edges_per_role,
        )
        for activator_idx in activators:
            for inhibitor_idx in inhibitors:
                if graph.edges[activator_idx].source_uuid == graph.edges[inhibitor_idx].source_uuid:
                    continue
                for level in ("exact", "typed"):
                    instances.append(_signed_seed_pair_motif(graph, activator_idx, inhibitor_idx, level=level))

    return instances


def _edge_motif(graph: NormalizedCircuit, edge_idx: int, *, level: str) -> MotifInstance:
    edge = graph.edges[edge_idx]
    target = graph.nodes[edge.target_uuid]
    motif_kind = "edge_to_seed" if target.is_seed else "edge"
    parts = (
        motif_kind,
        _node_key(graph.nodes[edge.source_uuid], level),
        edge.sign,
        _node_key(target, level),
    )
    exact_parts = (motif_kind, _node_key(graph.nodes[edge.source_uuid], "exact"), edge.sign, _node_key(target, "exact"))
    abstract_parts = (motif_kind, _node_key(graph.nodes[edge.source_uuid], "typed"), edge.sign, _node_key(target, "typed"))
    return _instance(graph, level, motif_kind, parts, exact_parts, abstract_parts, (edge.source_uuid, edge.target_uuid), (edge_idx,))


def _chain_motif(graph: NormalizedCircuit, first_idx: int, second_idx: int, *, level: str) -> MotifInstance:
    first = graph.edges[first_idx]
    second = graph.edges[second_idx]
    target = graph.nodes[second.target_uuid]
    motif_kind = "chain_to_seed" if target.is_seed else "chain"
    parts = (
        motif_kind,
        _node_key(graph.nodes[first.source_uuid], level),
        first.sign,
        _node_key(graph.nodes[first.target_uuid], level),
        second.sign,
        _node_key(target, level),
    )
    exact_parts = (
        motif_kind,
        _node_key(graph.nodes[first.source_uuid], "exact"),
        first.sign,
        _node_key(graph.nodes[first.target_uuid], "exact"),
        second.sign,
        _node_key(target, "exact"),
    )
    abstract_parts = (
        motif_kind,
        _node_key(graph.nodes[first.source_uuid], "typed"),
        first.sign,
        _node_key(graph.nodes[first.target_uuid], "typed"),
        second.sign,
        _node_key(target, "typed"),
    )
    return _instance(
        graph,
        level,
        motif_kind,
        parts,
        exact_parts,
        abstract_parts,
        (first.source_uuid, first.target_uuid, second.target_uuid),
        (first_idx, second_idx),
    )


def _fan_in_motif(graph: NormalizedCircuit, left_idx: int, right_idx: int, *, level: str) -> MotifInstance:
    left = graph.edges[left_idx]
    right = graph.edges[right_idx]
    target = graph.nodes[left.target_uuid]
    motif_kind = "fan_in_seed" if target.is_seed else "fan_in"
    input_parts = sorted(
        [
            (_node_key(graph.nodes[left.source_uuid], level), left.sign),
            (_node_key(graph.nodes[right.source_uuid], level), right.sign),
        ],
        key=_json_signature,
    )
    exact_inputs = sorted(
        [
            (_node_key(graph.nodes[left.source_uuid], "exact"), left.sign),
            (_node_key(graph.nodes[right.source_uuid], "exact"), right.sign),
        ],
        key=_json_signature,
    )
    abstract_inputs = sorted(
        [
            (_node_key(graph.nodes[left.source_uuid], "typed"), left.sign),
            (_node_key(graph.nodes[right.source_uuid], "typed"), right.sign),
        ],
        key=_json_signature,
    )
    parts = (motif_kind, tuple(input_parts), _node_key(target, level))
    exact_parts = (motif_kind, tuple(exact_inputs), _node_key(target, "exact"))
    abstract_parts = (motif_kind, tuple(abstract_inputs), _node_key(target, "typed"))
    return _instance(
        graph,
        level,
        motif_kind,
        parts,
        exact_parts,
        abstract_parts,
        (left.source_uuid, right.source_uuid, left.target_uuid),
        (left_idx, right_idx),
    )


def _signed_seed_pair_motif(graph: NormalizedCircuit, activator_idx: int, inhibitor_idx: int, *, level: str) -> MotifInstance:
    activator = graph.edges[activator_idx]
    inhibitor = graph.edges[inhibitor_idx]
    seed = graph.nodes[activator.target_uuid]
    motif_kind = "signed_seed_pair"
    parts = (
        motif_kind,
        (_node_key(graph.nodes[activator.source_uuid], level), activator.sign),
        (_node_key(graph.nodes[inhibitor.source_uuid], level), inhibitor.sign),
        _node_key(seed, level),
    )
    exact_parts = (
        motif_kind,
        (_node_key(graph.nodes[activator.source_uuid], "exact"), activator.sign),
        (_node_key(graph.nodes[inhibitor.source_uuid], "exact"), inhibitor.sign),
        _node_key(seed, "exact"),
    )
    abstract_parts = (
        motif_kind,
        (_node_key(graph.nodes[activator.source_uuid], "typed"), activator.sign),
        (_node_key(graph.nodes[inhibitor.source_uuid], "typed"), inhibitor.sign),
        _node_key(seed, "typed"),
    )
    return _instance(
        graph,
        level,
        motif_kind,
        parts,
        exact_parts,
        abstract_parts,
        (activator.source_uuid, inhibitor.source_uuid, activator.target_uuid),
        (activator_idx, inhibitor_idx),
    )


def _instance(
    graph: NormalizedCircuit,
    level: str,
    motif_kind: str,
    parts: object,
    exact_parts: object,
    abstract_parts: object,
    node_uuids: Iterable[str],
    edge_indices: tuple[int, ...],
) -> MotifInstance:
    unique_node_uuids = tuple(dict.fromkeys(str(uuid) for uuid in node_uuids))
    exact_keys = tuple(sorted((graph.nodes[uuid].exact_key for uuid in unique_node_uuids), key=_json_signature))
    edge_weights = tuple(float(graph.edges[idx].weight) for idx in edge_indices)
    return MotifInstance(
        circuit_uuid=graph.uuid,
        circuit_name=graph.name,
        motif_level=level,
        motif_kind=motif_kind,
        signature=_json_signature(parts),
        exact_signature=_json_signature(exact_parts),
        abstract_signature=_json_signature(abstract_parts),
        node_uuids=unique_node_uuids,
        edge_indices=edge_indices,
        exact_node_keys=exact_keys,
        edge_weights=edge_weights,
    )


def _motif_row(
    motif_id: str,
    group: list[MotifInstance],
    *,
    circuit_count: int,
    graphs: Mapping[str, NormalizedCircuit],
    high_circuit_uuids: set[str],
    latent_support: Mapping[tuple[int, str, int], int],
) -> tuple[dict[str, Any], float]:
    circuit_uuids = sorted({instance.circuit_uuid for instance in group})
    support_count = len(circuit_uuids)
    support_pct = _pct(support_count, circuit_count)
    exact_latents = {
        (layer, kind, latent_idx)
        for instance in group
        for layer, kind, latent_idx, _role in instance.exact_node_keys
    }
    max_latent_support = max((latent_support.get(latent, 0) for latent in exact_latents), default=0)
    max_latent_support_pct = _pct(max_latent_support, circuit_count)
    expected_pct = max(max_latent_support_pct, _pct(1, circuit_count))
    support_lift = support_pct / expected_pct if expected_pct > 0.0 else 0.0
    hub_penalty = 1.0 / (1.0 + max_latent_support_pct / 100.0)
    weights = [weight for instance in group for weight in instance.edge_weights]
    abs_weights = [abs(weight) for weight in weights]
    causal_score = float(mean(abs_weights)) if abs_weights else 1.0
    containing_high = sum(1 for uuid in circuit_uuids if uuid in high_circuit_uuids)
    high_base_rate = _pct(len(high_circuit_uuids), circuit_count)
    high_motif_rate = _pct(containing_high, support_count)
    high_enrichment = high_motif_rate / high_base_rate if high_base_rate > 0.0 else 0.0
    faithfulness = [graphs[uuid].faithfulness for uuid in circuit_uuids if uuid in graphs]
    row = {
        "motif_id": motif_id,
        "motif_size": max(len(instance.node_uuids) for instance in group),
        "motif_kind": group[0].motif_kind,
        "motif_level": group[0].motif_level,
        "exact_signature": group[0].exact_signature,
        "abstract_signature": group[0].abstract_signature,
        "support_count": support_count,
        "support_pct": support_pct,
        "support_lift": support_lift,
        "hub_penalty": hub_penalty,
        "max_latent_support_pct": max_latent_support_pct,
        "mean_faithfulness": float(mean(faithfulness)) if faithfulness else 0.0,
        "median_faithfulness": float(median(faithfulness)) if faithfulness else 0.0,
        "high_faithfulness_enrichment": high_enrichment,
        "mean_edge_weight": float(mean(weights)) if weights else 0.0,
        "mean_abs_edge_weight": causal_score,
        "motif_only_sufficiency": "",
        "motif_removal_drop": "",
    }
    score = support_lift * hub_penalty * (causal_score if causal_score > 0.0 else 1.0)
    return row, float(score)


def _membership_rows(
    graphs: Mapping[str, NormalizedCircuit],
    retained_groups: Mapping[str, list[MotifInstance]],
    retained_scores: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for motif_id, group in retained_groups.items():
        by_circuit: dict[str, list[MotifInstance]] = defaultdict(list)
        for instance in group:
            by_circuit[instance.circuit_uuid].append(instance)
        for circuit_uuid, instances in by_circuit.items():
            graph = graphs[circuit_uuid]
            covered_nodes = {uuid for instance in instances for uuid in instance.node_uuids}
            covered_edges = {idx for instance in instances for idx in instance.edge_indices}
            coverage_pct = _pct(len(covered_nodes), len(graph.nodes))
            membership_weight = (coverage_pct / 100.0) * float(retained_scores[motif_id])
            rows.append(
                {
                    "uuid": graph.uuid,
                    "name": graph.name,
                    "seed_comp": graph.seed_comp,
                    "seed_latent": graph.seed_latent,
                    "motif_id": motif_id,
                    "motif_role": instances[0].motif_kind,
                    "motif_level": instances[0].motif_level,
                    "motif_node_count": len(covered_nodes),
                    "motif_edge_count": len(covered_edges),
                    "motif_instance_count": len(instances),
                    "motif_coverage_pct": coverage_pct,
                    "membership_weight": membership_weight,
                }
            )
    rows.sort(key=lambda row: (str(row["uuid"]), str(row["motif_id"])))
    return rows


def _family_rows(
    graphs: Mapping[str, NormalizedCircuit],
    membership_rows: list[dict[str, Any]],
    retained_supports: Mapping[str, int],
    *,
    similarity_threshold: float,
    max_pair_motif_support: int,
) -> tuple[list[dict[str, Any]], int, int]:
    by_motif: dict[str, list[tuple[str, float]]] = defaultdict(list)
    by_circuit_family_weight: dict[str, Counter[str]] = defaultdict(Counter)
    for row in membership_rows:
        by_motif[str(row["motif_id"])].append((str(row["uuid"]), float(row["membership_weight"])))

    adjacency: dict[str, set[str]] = {uuid: set() for uuid in graphs}
    projected_edge_count = 0
    skipped_pair_motifs = 0
    for motif_id, members in by_motif.items():
        support = retained_supports.get(motif_id, len(members))
        if support > max_pair_motif_support:
            skipped_pair_motifs += 1
            continue
        for left_idx, (left_uuid, left_weight) in enumerate(members):
            for right_uuid, right_weight in members[left_idx + 1 :]:
                similarity = left_weight * right_weight
                if similarity >= similarity_threshold:
                    adjacency[left_uuid].add(right_uuid)
                    adjacency[right_uuid].add(left_uuid)
                    projected_edge_count += 1

    hard_family_by_uuid = _connected_components(adjacency)
    family_sizes = Counter(hard_family_by_uuid.values())
    for row in membership_rows:
        circuit_uuid = str(row["uuid"])
        fuzzy_family_id = f"motif:{row['motif_role']}"
        by_circuit_family_weight[circuit_uuid][fuzzy_family_id] += float(row["membership_weight"])

    rows: list[dict[str, Any]] = []
    for circuit_uuid, graph in graphs.items():
        hard_family_id = hard_family_by_uuid.get(circuit_uuid, f"F{len(rows) + 1:05d}")
        weights = by_circuit_family_weight.get(circuit_uuid) or Counter({hard_family_id: 1.0})
        total_weight = sum(weights.values()) or 1.0
        for family_id, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "uuid": graph.uuid,
                    "name": graph.name,
                    "seed_comp": graph.seed_comp,
                    "seed_latent": graph.seed_latent,
                    "hard_family_id": hard_family_id,
                    "hard_family_size": family_sizes.get(hard_family_id, 1),
                    "fuzzy_family_id": family_id,
                    "fuzzy_membership_weight": float(weight) / float(total_weight),
                    "raw_family_weight": float(weight),
                }
            )
    rows.sort(key=lambda row: (str(row["hard_family_id"]), str(row["uuid"]), str(row["fuzzy_family_id"])))
    return rows, projected_edge_count, skipped_pair_motifs


def _cohesion_row(graph: NormalizedCircuit, membership_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in membership_rows if row["uuid"] == graph.uuid]
    covered = sum(float(row["motif_node_count"]) for row in rows)
    motif_coverage_pct = min(100.0, _pct(int(covered), max(len(graph.nodes), 1)))
    layers = [node.layer for node in graph.nodes.values()]
    weights = [abs(edge.weight) for edge in graph.edges]
    return {
        "uuid": graph.uuid,
        "name": graph.name,
        "seed_comp": graph.seed_comp,
        "seed_latent": graph.seed_latent,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "counterfactual_faithfulness": graph.faithfulness,
        "posctx_suppression_score": graph.posctx_suppression_score,
        "internode_coact_density_pct": _post_float(graph.post_analysis, "internode_coact_density_pct"),
        "edge_weight_gini": _post_float(graph.post_analysis, "edge_weight_gini", default=_gini(weights)),
        "node_presence_pct_activators": _post_float(graph.post_analysis, "node_presence_pct_activators"),
        "node_presence_rate_mean": _post_float(graph.post_analysis, "node_presence_rate_mean"),
        "node_absence_pct_inhibitors": _post_float(graph.post_analysis, "node_absence_pct_inhibitors"),
        "node_inhibitor_rate_mean": _post_float(graph.post_analysis, "node_inhibitor_rate_mean"),
        "posctx_circuit_sufficiency": _post_float(graph.post_analysis, "posctx_circuit_sufficiency"),
        "motif_coverage_pct": motif_coverage_pct,
        "role_purity_score": _role_purity_score(graph),
        "causal_modularity": len({row["motif_id"] for row in rows}),
        "layer_span": (max(layers) - min(layers)) if layers else 0,
    }


def _connected_components(adjacency: Mapping[str, set[str]]) -> dict[str, str]:
    family_by_uuid: dict[str, str] = {}
    family_idx = 1
    for start_uuid in sorted(adjacency):
        if start_uuid in family_by_uuid:
            continue
        family_id = f"F{family_idx:05d}"
        family_idx += 1
        queue = deque([start_uuid])
        family_by_uuid[start_uuid] = family_id
        while queue:
            uuid = queue.popleft()
            for neighbor in sorted(adjacency.get(uuid, set())):
                if neighbor not in family_by_uuid:
                    family_by_uuid[neighbor] = family_id
                    queue.append(neighbor)
    return family_by_uuid


def _sorted_supported_groups(
    motif_groups: Mapping[tuple[str, str, str], list[MotifInstance]],
    *,
    min_support: int,
) -> list[tuple[tuple[str, str, str], list[MotifInstance]]]:
    supported = []
    for key, group in motif_groups.items():
        support = len({instance.circuit_uuid for instance in group})
        if support >= min_support:
            supported.append((key, group))
    supported.sort(key=lambda item: (-len({instance.circuit_uuid for instance in item[1]}), item[0]))
    return supported


def _exact_latent_support(graphs: Iterable[NormalizedCircuit]) -> Counter[tuple[int, str, int]]:
    support: Counter[tuple[int, str, int]] = Counter()
    for graph in graphs:
        circuit_latents = {(node.layer, node.kind, node.latent_idx) for node in graph.nodes.values()}
        support.update(circuit_latents)
    return support


def _node_key(node: NormalizedNode, level: str) -> tuple[int, str, int, str] | tuple[int, str, str]:
    if level == "exact":
        return node.exact_key
    if level == "typed":
        return node.abstract_key
    raise ValueError(f"unsupported motif level: {level}")


def _top_weighted_edge_indices(graph: NormalizedCircuit, edge_indices: Sequence[int], limit: int) -> list[int]:
    return sorted(edge_indices, key=lambda idx: (-abs(graph.edges[idx].weight), idx))[:limit]


def _eligible_feature(fid: FeatureID | None, kinds: Sequence[str]) -> bool:
    if fid is None:
        return False
    if fid.kind in ("logit", "token"):
        return False
    if fid.kind.endswith("_err"):
        return False
    return fid.kind in kinds


def _is_seed_feature(
    fid: FeatureID,
    seed_comp: object,
    seed_latent: object,
    n_kinds: int,
    kinds: Sequence[str],
) -> bool:
    if seed_comp is None or seed_latent is None:
        return False
    try:
        comp_idx, latent_idx = fid.to_component_id(n_kinds, kinds)
        return comp_idx == int(seed_comp) and latent_idx == int(seed_latent)
    except (TypeError, ValueError):
        return False


def _edge_sign(weight: float) -> str:
    if weight > 0.0:
        return "positive"
    if weight < 0.0:
        return "negative"
    return "zero"


def _summary_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    lookup = {}
    for row in rows:
        uuid = row.get("uuid")
        if uuid is not None:
            lookup[str(uuid)] = row
    return lookup


def _try_load_summary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_circuit_summary_rows(path)


def _post_analysis(metadata: Mapping[str, Any], summary: Mapping[str, Any]) -> Mapping[str, Any]:
    for source in (metadata, summary.get("metadata", {}) if isinstance(summary.get("metadata"), Mapping) else {}):
        post = source.get("post_analysis") if isinstance(source, Mapping) else None
        if isinstance(post, Mapping):
            return post
    return {}


def _first_metric(
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
    nested_path: tuple[str, str],
    fallback_key: str,
) -> float:
    for source in (metadata, summary.get("metadata", {}) if isinstance(summary.get("metadata"), Mapping) else {}):
        if not isinstance(source, Mapping):
            continue
        nested = source.get(nested_path[0])
        if isinstance(nested, Mapping) and isinstance(nested.get(nested_path[1]), (int, float)):
            return float(nested[nested_path[1]])
        if isinstance(source.get(fallback_key), (int, float)):
            return float(source[fallback_key])
    return 0.0


def _role_purity_score(graph: NormalizedCircuit) -> float:
    checked = 0
    compatible = 0
    for edge in graph.edges:
        source = graph.nodes[edge.source_uuid]
        target = graph.nodes[edge.target_uuid]
        if not target.is_seed:
            continue
        if source.role in _ACTIVATOR_ROLES:
            checked += 1
            compatible += int(edge.weight >= 0.0)
        elif source.role in _INHIBITOR_ROLES:
            checked += 1
            compatible += int(edge.weight <= 0.0)
    return _pct(compatible, checked)


def _write_plot(path: Path, stats: Mapping[str, object]) -> None:
    plt = configure_matplotlib()
    motif_rows = stats["motif_rows"]
    family_rows = stats["family_rows"]
    assert isinstance(motif_rows, list)
    assert isinstance(family_rows, list)
    fig, axes = plt.subplots(2, 2, figsize=panel_figsize(2, 2))

    supports = [int(row["support_count"]) for row in motif_rows]
    axes[0, 0].hist(supports, bins=40, color=BLUE)
    axes[0, 0].set_title("Motif Support Distribution")
    axes[0, 0].set_xlabel("Circuits containing motif")
    axes[0, 0].set_ylabel("Motif count")
    integer_ticks(axes[0, 0])

    kind_counts = Counter(str(row["motif_kind"]) for row in motif_rows)
    axes[0, 1].bar(list(kind_counts), list(kind_counts.values()), width=BAR_WIDTH, color=BLUE)
    axes[0, 1].set_title("Motifs By Kind")
    axes[0, 1].set_ylabel("Motif count")
    axes[0, 1].tick_params(axis="x", labelrotation=35)
    integer_ticks(axes[0, 1])

    axes[1, 0].scatter(
        [float(row["support_lift"]) for row in motif_rows],
        [float(row["high_faithfulness_enrichment"]) for row in motif_rows],
        s=20,
        color=BLUE,
        alpha=0.7,
        edgecolors="none",
    )
    axes[1, 0].set_title("Lift vs Faithfulness Enrichment")
    axes[1, 0].set_xlabel("Hub-aware support lift")
    axes[1, 0].set_ylabel("High-faithfulness enrichment")

    family_sizes = sorted({str(row["hard_family_id"]): int(row["hard_family_size"]) for row in family_rows}.values(), reverse=True)
    axes[1, 1].bar(range(len(family_sizes[:50])), family_sizes[:50], color=BLUE)
    axes[1, 1].set_title("Top Hard Family Sizes")
    axes[1, 1].set_xlabel("Family rank")
    axes[1, 1].set_ylabel("Circuit count")
    integer_ticks(axes[1, 1])

    round_bars(axes[0, 1])
    round_bars(axes[1, 1])
    style_suptitle(fig, "Circuit Motifs And Families")
    save_figure(fig, path)


def _write_motif_table(path: Path, stats: Mapping[str, object]) -> None:
    rows = stats["motif_rows"]
    assert isinstance(rows, list)
    write_csv(
        path,
        rows,
        [
            "motif_id",
            "motif_size",
            "motif_kind",
            "motif_level",
            "exact_signature",
            "abstract_signature",
            "support_count",
            "support_pct",
            "support_lift",
            "hub_penalty",
            "max_latent_support_pct",
            "mean_faithfulness",
            "median_faithfulness",
            "high_faithfulness_enrichment",
            "mean_edge_weight",
            "mean_abs_edge_weight",
            "motif_only_sufficiency",
            "motif_removal_drop",
        ],
    )


def _write_membership_table(path: Path, stats: Mapping[str, object]) -> None:
    rows = stats["membership_rows"]
    assert isinstance(rows, list)
    write_csv(
        path,
        rows,
        [
            "uuid",
            "name",
            "seed_comp",
            "seed_latent",
            "motif_id",
            "motif_role",
            "motif_level",
            "motif_node_count",
            "motif_edge_count",
            "motif_instance_count",
            "motif_coverage_pct",
            "membership_weight",
        ],
    )


def _write_cohesion_table(path: Path, stats: Mapping[str, object]) -> None:
    rows = stats["cohesion_rows"]
    assert isinstance(rows, list)
    write_csv(
        path,
        rows,
        [
            "uuid",
            "name",
            "seed_comp",
            "seed_latent",
            "nodes",
            "edges",
            "counterfactual_faithfulness",
            "posctx_suppression_score",
            "internode_coact_density_pct",
            "edge_weight_gini",
            "node_presence_pct_activators",
            "node_presence_rate_mean",
            "node_absence_pct_inhibitors",
            "node_inhibitor_rate_mean",
            "posctx_circuit_sufficiency",
            "motif_coverage_pct",
            "role_purity_score",
            "causal_modularity",
            "layer_span",
        ],
    )


def _write_family_table(path: Path, stats: Mapping[str, object]) -> None:
    rows = stats["family_rows"]
    assert isinstance(rows, list)
    write_csv(
        path,
        rows,
        [
            "uuid",
            "name",
            "seed_comp",
            "seed_latent",
            "hard_family_id",
            "hard_family_size",
            "fuzzy_family_id",
            "fuzzy_membership_weight",
            "raw_family_weight",
        ],
    )


def _build_summary(store_path: Path, stats: Mapping[str, object]) -> dict[str, object]:
    motif_rows = stats["motif_rows"]
    family_rows = stats["family_rows"]
    assert isinstance(motif_rows, list)
    assert isinstance(family_rows, list)
    family_ids = {str(row["hard_family_id"]) for row in family_rows}
    return {
        "circuit_store_path": str(store_path),
        "circuit_count": stats["circuit_count"],
        "raw_motif_instance_count": stats["raw_motif_instance_count"],
        "motif_count": stats["motif_count"],
        "hard_family_count": len(family_ids),
        "projected_edge_count": stats["projected_edge_count"],
        "skipped_pair_motif_count": stats["skipped_pair_motif_count"],
        "min_support": stats["min_support"],
        "high_faithfulness_quantile": stats["high_faithfulness_quantile"],
        "high_faithfulness_threshold": stats["high_faithfulness_threshold"],
        "similarity_threshold": stats["similarity_threshold"],
        "max_pair_motif_support": stats["max_pair_motif_support"],
        "max_edges_per_circuit": stats["max_edges_per_circuit"],
        "max_fan_in_edges_per_target": stats["max_fan_in_edges_per_target"],
        "max_signed_pair_edges_per_role": stats["max_signed_pair_edges_per_role"],
        "motif_support_summary": stats["motif_support_summary"],
        "membership_weight_summary": stats["membership_weight_summary"],
        "family_size_summary": stats["family_size_summary"],
        "top_motifs": motif_rows[:50],
        "note": (
            "This first pass mines exact and typed 2-node/3-node motifs from saved circuit graphs. "
            "Motif-only sufficiency and motif-removal scores are reserved for a later causal validation pass."
        ),
    }


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    return int(value) if isinstance(value, (int, float)) else -1


def _numeric(value: object, *, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else default


def _post_float(post: Mapping[str, Any], key: str, *, default: float = 0.0) -> float:
    return _numeric(post.get(key), default=default)


def _json_signature(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _summary(values: list[int] | list[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(clean)
    return {
        "count": len(clean),
        "mean": float(mean(clean)),
        "p50": float(median(clean)),
        "p90": float(_quantile(ordered, 0.90)),
        "max": float(max(clean)),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _pct(count: int, total: int) -> float:
    return (float(count) / float(total) * 100.0) if total else 0.0


def _gini(values: Sequence[float]) -> float:
    clean = sorted(abs(float(value)) for value in values if math.isfinite(float(value)))
    if not clean or sum(clean) == 0.0:
        return 0.0
    n = len(clean)
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(clean))
    return (2.0 * weighted_sum) / (n * sum(clean)) - (n + 1.0) / n
