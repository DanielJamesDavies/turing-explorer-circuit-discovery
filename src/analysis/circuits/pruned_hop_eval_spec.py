"""Prepare sampled full/pruned circuit variants for hop-based eval comparison."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from analysis.coactivation.data import load_top_coactivation
from analysis.coactivation.graph_utils import build_high_pmi_edges, high_pmi_in_degree
from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import configure_matplotlib
from store.circuits import Circuit
from .coact_overlap import SUITE_NAME
from .node_hop_overlap import (
    DEFAULT_KINDS,
    _circuit_node_sets,
    _metadata_float,
    _reachable_sets,
    load_circuit_store,
    resolve_circuit_store_path,
)


@dataclass(frozen=True)
class PrunedHopEvalSpecResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    spec_path: Path
    summary: dict[str, object]


def plot_pruned_hop_eval_spec(
    run_root: str | Path,
    *,
    circuit_store_path: str | Path | None = None,
    output_root: str | Path | None = None,
    sample_size: int = 128,
    max_hops: int = 6,
    threshold: float = 2.0,
    top_out_degree: int = 12,
    max_frontier: int = 256,
    hub_quantile: float = 0.99,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> PrunedHopEvalSpecResult:
    """Create and plot a deterministic sampled pruned-circuit eval spec."""

    root = resolve_run_root(run_root)
    store_path = resolve_circuit_store_path(root, circuit_store_path)
    circuits = load_circuit_store(store_path)
    artifact = load_top_coactivation(root)
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)

    spec = build_pruned_hop_eval_spec(
        circuits,
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        sample_size=sample_size,
        max_hops=max_hops,
        threshold=threshold,
        top_out_degree=top_out_degree,
        max_frontier=max_frontier,
        hub_quantile=hub_quantile,
        kinds=kinds,
    )
    figure_path = output_dirs["figures"] / "pruned-hop-eval-spec.png"
    table_path = output_dirs["tables"] / "pruned-hop-eval-spec.csv"
    summary_path = output_dirs["summaries"] / "pruned-hop-eval-spec.json"
    spec_path = output_dirs["root"] / "pruned-hop-eval-spec.pt"

    torch.save(spec["variants"], spec_path)
    _write_plot(figure_path, spec)
    _write_table(table_path, spec)
    summary = _build_summary(store_path, spec_path, spec)
    write_json(summary_path, summary)
    return PrunedHopEvalSpecResult(figure_path, summary_path, table_path, spec_path, summary)


def build_pruned_hop_eval_spec(
    circuits: Mapping[str, Circuit],
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    sample_size: int = 128,
    max_hops: int = 6,
    threshold: float = 2.0,
    top_out_degree: int = 12,
    max_frontier: int = 256,
    hub_quantile: float = 0.99,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> dict[str, object]:
    """Build full and hop-pruned circuit variants for a sampled eval run."""

    eligible = [
        circuit
        for circuit in circuits.values()
        if circuit.metadata.get("seed_comp") is not None and circuit.metadata.get("seed_latent") is not None
    ]
    sampled = _deterministic_circuit_sample(eligible, sample_size)
    values = top_values.detach().cpu().to(torch.float32).reshape(-1, top_values.shape[-1])
    indices = top_indices.detach().cpu().to(torch.int64).reshape(-1, top_indices.shape[-1])
    edges = build_high_pmi_edges(top_values, top_indices, threshold=threshold)
    in_degree = high_pmi_in_degree(edges)
    hub_cutoff = int(torch.quantile(in_degree.float(), torch.tensor(float(hub_quantile))).item())
    hub_cutoff = max(hub_cutoff, 1)
    n_kinds = len(kinds)
    cache: dict[int, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    variants: dict[str, dict[str, Circuit]] = {}

    for sample_index, circuit in enumerate(sampled):
        seed_comp = int(circuit.metadata["seed_comp"])
        seed_latent = int(circuit.metadata["seed_latent"])
        seed_global_id = seed_comp * int(d_sae) + seed_latent
        node_sets = _circuit_node_sets(circuit, n_kinds=n_kinds, d_sae=d_sae, kinds=kinds, seed_global_id=seed_global_id)
        if not node_sets["all"]:
            continue
        reachable = _reachable_sets(
            seed_global_id,
            values,
            indices,
            in_degree,
            hub_cutoff=hub_cutoff,
            threshold=threshold,
            top_out_degree=top_out_degree,
            max_frontier=max_frontier,
            max_hops=max_hops,
            cache=cache,
        )
        variant_group: dict[str, Circuit] = {"full": deepcopy(circuit)}
        rows.append(_variant_row(sample_index, circuit, "full", 0, seed_global_id, len(node_sets["all"]), len(circuit.nodes), len(circuit.edges), 100.0))
        for hop in range(1, int(max_hops) + 1):
            keep_gids = node_sets["all"] & reachable[hop]
            pruned = _prune_circuit_to_gids(circuit, keep_gids, n_kinds=n_kinds, d_sae=d_sae, kinds=kinds)
            variant_name = f"hop{hop}"
            variant_group[variant_name] = pruned
            rows.append(
                _variant_row(
                    sample_index,
                    circuit,
                    variant_name,
                    hop,
                    seed_global_id,
                    len(node_sets["all"]),
                    len(pruned.nodes),
                    len(pruned.edges),
                    (len(keep_gids) / len(node_sets["all"]) * 100.0) if node_sets["all"] else 0.0,
                )
            )
        variants[circuit.uuid] = variant_group

    return {
        "sample_size": int(sample_size),
        "actual_sample_size": len(variants),
        "max_hops": int(max_hops),
        "threshold": float(threshold),
        "top_out_degree": int(top_out_degree),
        "max_frontier": int(max_frontier),
        "hub_quantile": float(hub_quantile),
        "hub_cutoff_in_degree": int(hub_cutoff),
        "rows": rows,
        "variants": variants,
    }


def _deterministic_circuit_sample(circuits: list[Circuit], sample_size: int) -> list[Circuit]:
    circuits = sorted(circuits, key=lambda circuit: circuit.uuid)
    if len(circuits) <= int(sample_size):
        return circuits
    positions = torch.linspace(0, len(circuits) - 1, steps=int(sample_size), dtype=torch.float64).round().to(torch.int64).unique()
    return [circuits[int(position)] for position in positions.tolist()]


def _prune_circuit_to_gids(
    circuit: Circuit,
    keep_gids: set[int],
    *,
    n_kinds: int,
    d_sae: int,
    kinds: Sequence[str],
) -> Circuit:
    pruned = deepcopy(circuit)
    keep_uuids = set()
    for uuid, node in circuit.nodes.items():
        role = str(node.metadata.get("role", ""))
        if role == "seed":
            keep_uuids.add(uuid)
            continue
        fid = node.feature_id
        if fid is None or fid.kind not in kinds:
            continue
        if fid.to_global_id(n_kinds, d_sae, kinds) in keep_gids:
            keep_uuids.add(uuid)
    pruned.nodes = {uuid: node for uuid, node in pruned.nodes.items() if uuid in keep_uuids}
    pruned.edges = [
        edge for edge in pruned.edges if edge.source_uuid in keep_uuids and edge.target_uuid in keep_uuids
    ]
    pruned.metadata = dict(pruned.metadata)
    pruned.metadata["pruned_hop_variant"] = True
    return pruned


def _variant_row(
    sample_index: int,
    circuit: Circuit,
    variant: str,
    hop: int,
    seed_global_id: int,
    circuit_latent_count: int,
    retained_nodes: int,
    retained_edges: int,
    retained_circuit_latent_pct: float,
) -> dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "uuid": circuit.uuid,
        "name": circuit.name,
        "variant": variant,
        "hop": int(hop),
        "seed_comp": int(circuit.metadata.get("seed_comp", -1)),
        "seed_latent": int(circuit.metadata.get("seed_latent", -1)),
        "seed_global_id": int(seed_global_id),
        "full_nodes": int(len(circuit.nodes)),
        "full_edges": int(len(circuit.edges)),
        "retained_nodes": int(retained_nodes),
        "retained_edges": int(retained_edges),
        "circuit_latent_count": int(circuit_latent_count),
        "retained_circuit_latent_pct": float(retained_circuit_latent_pct),
        "full_counterfactual_faithfulness": _metadata_float(circuit.metadata, ("evals", "counterfactual_faithfulness")),
        "full_posctx_suppression_score": _metadata_float(circuit.metadata, ("evals", "posctx_suppression_score")),
    }


def _write_plot(path: Path, spec: dict[str, object]) -> None:
    plt = configure_matplotlib()
    rows = spec["rows"]
    max_hops = int(spec["max_hops"])
    assert isinstance(rows, list)
    hop_rows = [row for row in rows if int(row["hop"]) > 0]
    hop_labels = [f"hop{hop}" for hop in range(1, max_hops + 1)]
    means = []
    p90s = []
    node_means = []
    for hop in range(1, max_hops + 1):
        values = [float(row["retained_circuit_latent_pct"]) for row in hop_rows if int(row["hop"]) == hop]
        nodes = [float(row["retained_nodes"]) for row in hop_rows if int(row["hop"]) == hop]
        ordered = sorted(values)
        means.append(float(sum(values) / len(values)) if values else 0.0)
        p90s.append(_quantile(ordered, 0.9) if ordered else 0.0)
        node_means.append(float(sum(nodes) / len(nodes)) if nodes else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(hop_labels))
    axes[0].bar([pos - 0.18 for pos in x], means, width=0.36, color="#2f6f9f", alpha=0.85, label="mean")
    axes[0].bar([pos + 0.18 for pos in x], p90s, width=0.36, color="#b45f06", alpha=0.85, label="p90")
    axes[0].set_title("Circuit Latents Retained In Hop-Pruned Variants")
    axes[0].set_ylabel("Circuit latents retained (%)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(hop_labels)
    axes[0].legend(loc="upper left")

    axes[1].plot(hop_labels, node_means, marker="o", linewidth=2.0, color="#38761d")
    axes[1].set_title("Mean Pruned Circuit Size")
    axes[1].set_ylabel("Retained nodes including seed")
    axes[1].set_xlabel("Pruned variant")
    fig.suptitle("Pruned Hop Eval Spec: 128 Circuit Sample", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, spec: dict[str, object]) -> None:
    write_csv(
        path,
        spec["rows"],
        [
            "sample_index",
            "uuid",
            "name",
            "variant",
            "hop",
            "seed_comp",
            "seed_latent",
            "seed_global_id",
            "full_nodes",
            "full_edges",
            "retained_nodes",
            "retained_edges",
            "circuit_latent_count",
            "retained_circuit_latent_pct",
            "full_counterfactual_faithfulness",
            "full_posctx_suppression_score",
        ],
    )


def _build_summary(store_path: Path, spec_path: Path, spec: dict[str, object]) -> dict[str, object]:
    return {
        "circuit_store_path": str(store_path),
        "spec_path": str(spec_path),
        "sample_size": spec["sample_size"],
        "actual_sample_size": spec["actual_sample_size"],
        "max_hops": spec["max_hops"],
        "threshold": spec["threshold"],
        "top_out_degree": spec["top_out_degree"],
        "max_frontier": spec["max_frontier"],
        "hub_quantile": spec["hub_quantile"],
        "hub_cutoff_in_degree": spec["hub_cutoff_in_degree"],
        "notes": (
            "This is the stored full/pruned circuit variant spec. "
            "Intervention evals require running these variants through the live model/SAE evaluator."
        ),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

