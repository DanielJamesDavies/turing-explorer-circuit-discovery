"""Circuit seed multi-hop coactivation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import (
    SEQUENTIAL_CMAP,
    SERIES2,
    configure_matplotlib,
    panel_figsize,
    round_bars,
    save_figure,
    style_suptitle,
    styled_legend,
)
from analysis.coactivation.coact_degrees import _expand_frontier, _summary
from analysis.coactivation.data import TopCoactivationArtifact, load_top_coactivation
from analysis.coactivation.graph_utils import build_high_pmi_edges, high_pmi_in_degree
from .coact_overlap import SUITE_NAME, _parse_row, load_circuit_summary_rows


@dataclass(frozen=True)
class CircuitSeedCoactHopsResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_circuit_seed_coact_hops(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
    top_out_degree: int = 12,
    max_frontier: int = 256,
    hub_quantile: float = 0.99,
) -> CircuitSeedCoactHopsResult:
    """Generate 1/2/3-hop coactivation graphs for circuit seed latents."""

    root = resolve_run_root(run_root)
    summary_path = root / "circuits" / "summary.json"
    circuit_rows = load_circuit_summary_rows(summary_path)
    artifact = load_top_coactivation(root)
    if artifact.mode != "pmi":
        raise ValueError(f"circuit seed coact hops requires mode='pmi', got {artifact.mode!r}")

    stats = compute_circuit_seed_coact_hops(
        circuit_rows,
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        threshold=threshold,
        top_out_degree=top_out_degree,
        max_frontier=max_frontier,
        hub_quantile=hub_quantile,
    )
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "circuit-seed-coact-hops.png"
    table_path = output_dirs["tables"] / "circuit-seed-coact-hops.csv"
    output_summary_path = output_dirs["summaries"] / "circuit-seed-coact-hops.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(summary_path, artifact, stats)
    write_json(output_summary_path, summary)
    return CircuitSeedCoactHopsResult(figure_path, output_summary_path, table_path, summary)


def compute_circuit_seed_coact_hops(
    circuit_rows: list[dict[str, Any]],
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
    top_out_degree: int = 12,
    max_frontier: int = 256,
    hub_quantile: float = 0.99,
) -> dict[str, object]:
    """Compute 1/2/3-hop coact reachability for circuit seed latents."""

    parsed = [_parse_row(row) for row in circuit_rows]
    parsed = [row for row in parsed if row is not None and int(row["seed_comp"]) >= 0 and int(row["seed_latent"]) >= 0]
    values = top_values.detach().cpu().to(torch.float32).reshape(-1, top_values.shape[-1])
    indices = top_indices.detach().cpu().to(torch.int64).reshape(-1, top_indices.shape[-1])
    edges = build_high_pmi_edges(top_values, top_indices, threshold=threshold)
    in_degree = high_pmi_in_degree(edges)
    pruned_cutoff = int(torch.quantile(in_degree.float(), torch.tensor(float(hub_quantile))).item())
    pruned_cutoff = max(pruned_cutoff, 1)
    unpruned_cutoff = int(in_degree.max().item())
    pruned_cache: dict[int, torch.Tensor] = {}
    unpruned_cache: dict[int, torch.Tensor] = {}

    table_rows = []
    pruned_by_hop: dict[int, list[int]] = {1: [], 2: [], 3: []}
    unpruned_by_hop: dict[int, list[int]] = {1: [], 2: [], 3: []}
    faithfulness = []
    coact_overlap = []
    internode_density = []
    nodes = []

    for row in parsed:
        seed_global_id = int(row["seed_comp"]) * int(d_sae) + int(row["seed_latent"])
        if seed_global_id < 0 or seed_global_id >= values.shape[0]:
            continue
        pruned_reach = _reachability_for_source(
            seed_global_id,
            values,
            indices,
            in_degree,
            hub_cutoff=pruned_cutoff,
            threshold=threshold,
            top_out_degree=top_out_degree,
            max_frontier=max_frontier,
            cache=pruned_cache,
        )
        unpruned_reach = _reachability_for_source(
            seed_global_id,
            values,
            indices,
            in_degree,
            hub_cutoff=unpruned_cutoff,
            threshold=threshold,
            top_out_degree=top_out_degree,
            max_frontier=max_frontier,
            cache=unpruned_cache,
        )
        for hop in (1, 2, 3):
            pruned_by_hop[hop].append(pruned_reach[hop])
            unpruned_by_hop[hop].append(unpruned_reach[hop])
        faithfulness.append(float(row["counterfactual_faithfulness"]))
        coact_overlap.append(float(row["coact_overlap_pct"]))
        internode_density.append(float(row["internode_coact_density_pct"]))
        nodes.append(int(row["nodes"]))
        table_rows.append(
            {
                **row,
                "seed_global_id": seed_global_id,
                "pruned_hop1_reachable": pruned_reach[1],
                "pruned_hop2_reachable": pruned_reach[2],
                "pruned_hop3_reachable": pruned_reach[3],
                "unpruned_hop1_reachable": unpruned_reach[1],
                "unpruned_hop2_reachable": unpruned_reach[2],
                "unpruned_hop3_reachable": unpruned_reach[3],
            }
        )

    return {
        "threshold": float(threshold),
        "top_out_degree": int(top_out_degree),
        "max_frontier": int(max_frontier),
        "hub_quantile": float(hub_quantile),
        "hub_cutoff_in_degree": int(pruned_cutoff),
        "circuit_count": len(table_rows),
        "pruned_reach_by_hop": {str(hop): pruned_by_hop[hop] for hop in (1, 2, 3)},
        "unpruned_reach_by_hop": {str(hop): unpruned_by_hop[hop] for hop in (1, 2, 3)},
        "pruned_reach_summary": {str(hop): _summary(pruned_by_hop[hop]) for hop in (1, 2, 3)},
        "unpruned_reach_summary": {str(hop): _summary(unpruned_by_hop[hop]) for hop in (1, 2, 3)},
        "faithfulness": faithfulness,
        "coact_overlap": coact_overlap,
        "internode_density": internode_density,
        "nodes": nodes,
        "table_rows": table_rows,
        "correlations": _correlations(pruned_by_hop, unpruned_by_hop, faithfulness, coact_overlap, internode_density),
        "limitation": (
            "This uses circuit seed latents because this run exports aggregate circuit summaries, "
            "not full circuit node lists. Exact hop distance from seed to every circuit node requires discovered_circuits.pt."
        ),
    }


def _reachability_for_source(
    source: int,
    values: torch.Tensor,
    indices: torch.Tensor,
    in_degree: torch.Tensor,
    *,
    hub_cutoff: int,
    threshold: float,
    top_out_degree: int,
    max_frontier: int,
    cache: dict[int, torch.Tensor],
) -> dict[int, int]:
    seen = {int(source)}
    frontier = {int(source)}
    cumulative = set()
    reach = {}
    for hop in (1, 2, 3):
        next_nodes = _expand_frontier(
            frontier,
            values,
            indices,
            in_degree,
            hub_cutoff=hub_cutoff,
            threshold=threshold,
            top_out_degree=top_out_degree,
            max_frontier=max_frontier,
            cache=cache,
        )
        next_nodes.difference_update(seen)
        seen.update(next_nodes)
        cumulative.update(next_nodes)
        frontier = set(list(next_nodes)[:max_frontier])
        reach[hop] = len(cumulative)
    return reach


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=panel_figsize(2, 2))

    pruned_summary = stats["pruned_reach_summary"]
    unpruned_summary = stats["unpruned_reach_summary"]
    assert isinstance(pruned_summary, dict)
    assert isinstance(unpruned_summary, dict)
    hops = [1, 2, 3]
    labels = [f"{hop} hop" for hop in hops]
    x = range(len(hops))
    axes[0, 0].bar([pos - 0.18 for pos in x], [pruned_summary[str(hop)]["p90"] for hop in hops], width=0.3, color=SERIES2[0], label="hub-pruned p90")
    axes[0, 0].bar([pos + 0.18 for pos in x], [unpruned_summary[str(hop)]["p90"] for hop in hops], width=0.3, color=SERIES2[1], label="unpruned p90")
    axes[0, 0].set_title("Circuit Seed Reachability By Hop")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylabel("Reachable coact latents")
    styled_legend(axes[0, 0], loc="upper left")

    coact_overlap = stats["coact_overlap"]
    faithfulness = stats["faithfulness"]
    internode_density = stats["internode_density"]
    nodes = stats["nodes"]
    table_rows = stats["table_rows"]
    assert isinstance(coact_overlap, list)
    assert isinstance(faithfulness, list)
    assert isinstance(internode_density, list)
    assert isinstance(nodes, list)
    assert isinstance(table_rows, list)
    pruned_hop3 = [int(row["pruned_hop3_reachable"]) for row in table_rows]
    unpruned_hop3 = [int(row["unpruned_hop3_reachable"]) for row in table_rows]
    sizes = [max(8.0, min(float(node) / 8.0, 80.0)) for node in nodes]

    scatter = axes[0, 1].scatter(pruned_hop3, faithfulness, s=sizes, c=coact_overlap, cmap=SEQUENTIAL_CMAP, alpha=0.75, edgecolors="none")
    axes[0, 1].set_title("Pruned 3-Hop Reach vs Faithfulness")
    axes[0, 1].set_xlabel("Seed 3-hop reachable latents, hub-pruned")
    axes[0, 1].set_ylabel("Counterfactual faithfulness")
    fig.colorbar(scatter, ax=axes[0, 1], label="Direct circuit-node coact overlap (%)")

    axes[1, 0].scatter(unpruned_hop3, coact_overlap, s=sizes, c=faithfulness, cmap=SEQUENTIAL_CMAP, alpha=0.75, edgecolors="none")
    axes[1, 0].set_title("Unpruned 3-Hop Reach vs Direct Node Overlap")
    axes[1, 0].set_xlabel("Seed 3-hop reachable latents, unpruned")
    axes[1, 0].set_ylabel("Circuit nodes in seed top coacts (%)")

    axes[1, 1].scatter(unpruned_hop3, internode_density, s=sizes, c=coact_overlap, cmap=SEQUENTIAL_CMAP, alpha=0.75, edgecolors="none")
    axes[1, 1].set_title("Unpruned 3-Hop Reach vs Internode Coact Density")
    axes[1, 1].set_xlabel("Seed 3-hop reachable latents, unpruned")
    axes[1, 1].set_ylabel("Internode mutual coact density (%)")

    round_bars(axes[0, 0])
    style_suptitle(fig, "Circuit Seed Degrees Of Coactivation")
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    write_csv(
        path,
        stats["table_rows"],
        [
            "uuid",
            "name",
            "seed_global_id",
            "seed_comp",
            "seed_latent",
            "nodes",
            "edges",
            "n_activators",
            "n_inhibitors",
            "counterfactual_faithfulness",
            "coact_overlap_pct",
            "coact_overlap_pct_activators",
            "coact_overlap_pct_inhibitors",
            "internode_coact_density_pct",
            "pruned_hop1_reachable",
            "pruned_hop2_reachable",
            "pruned_hop3_reachable",
            "unpruned_hop1_reachable",
            "unpruned_hop2_reachable",
            "unpruned_hop3_reachable",
        ],
    )


def _build_summary(source_path: Path, artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "source_path": str(source_path),
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "top_out_degree": stats["top_out_degree"],
        "max_frontier": stats["max_frontier"],
        "hub_quantile": stats["hub_quantile"],
        "hub_cutoff_in_degree": stats["hub_cutoff_in_degree"],
        "circuit_count": stats["circuit_count"],
        "pruned_reach_summary": stats["pruned_reach_summary"],
        "unpruned_reach_summary": stats["unpruned_reach_summary"],
        "correlations": stats["correlations"],
        "limitation": stats["limitation"],
    }


def _correlations(
    pruned_by_hop: dict[int, list[int]],
    unpruned_by_hop: dict[int, list[int]],
    faithfulness: list[float],
    coact_overlap: list[float],
    internode_density: list[float],
) -> dict[str, float]:
    result = {}
    for prefix, by_hop in (("pruned", pruned_by_hop), ("unpruned", unpruned_by_hop)):
        for hop in (1, 2, 3):
            values = [float(value) for value in by_hop[hop]]
            result[f"{prefix}_hop{hop}_vs_faithfulness"] = _pearson(values, faithfulness)
            result[f"{prefix}_hop{hop}_vs_coact_overlap"] = _pearson(values, coact_overlap)
            result[f"{prefix}_hop{hop}_vs_internode_density"] = _pearson(values, internode_density)
    return result


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    left_norm = sum(value * value for value in left_centered) ** 0.5
    right_norm = sum(value * value for value in right_centered) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))

