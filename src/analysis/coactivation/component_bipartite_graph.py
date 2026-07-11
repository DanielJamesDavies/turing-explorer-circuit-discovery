"""Bipartite graph view of strongest component-pair coactivation edges."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import INK_MUTED, SERIES2, configure_matplotlib, save_figure
from .component_pair_heatmap import compute_component_pair_heatmap
from .data import TopCoactivationArtifact, load_top_coactivation
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class ComponentBipartiteGraphResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_component_bipartite_graph(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
    max_edges: int = 80,
) -> ComponentBipartiteGraphResult:
    """Generate a bipartite plot of strongest target->coact component edges."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"component bipartite graph requires mode='pmi', got {artifact.mode!r}")

    stats = compute_component_bipartite_graph(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        threshold=threshold,
        max_edges=max_edges,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "component-bipartite-graph.png"
    table_path = output_dirs["tables"] / "component-bipartite-graph.csv"
    summary_path = output_dirs["summaries"] / "component-bipartite-graph.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return ComponentBipartiteGraphResult(figure_path, summary_path, table_path, summary)


def compute_component_bipartite_graph(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
    max_edges: int = 80,
) -> dict[str, object]:
    """Select strongest component-pair edges by high-PMI count."""

    pair_stats = compute_component_pair_heatmap(
        top_values,
        top_indices,
        d_sae=d_sae,
        threshold=threshold,
    )
    high_counts = torch.tensor(pair_stats["high_counts"], dtype=torch.int64)
    high_rate = torch.tensor(pair_stats["high_rate"], dtype=torch.float32)
    mean_pmi = torch.tensor(pair_stats["mean_pmi"], dtype=torch.float32)
    pair_counts = torch.tensor(pair_stats["pair_counts"], dtype=torch.int64)
    edges = []
    for target_component in range(high_counts.shape[0]):
        for coact_component in range(high_counts.shape[1]):
            high_count = int(high_counts[target_component, coact_component].item())
            if high_count <= 0:
                continue
            edges.append(
                {
                    "target_component": target_component,
                    "coact_component": coact_component,
                    "high_count": high_count,
                    "pair_count": int(pair_counts[target_component, coact_component].item()),
                    "high_rate": float(high_rate[target_component, coact_component].item()),
                    "mean_pmi": float(mean_pmi[target_component, coact_component].item()),
                }
            )
    edges = sorted(edges, key=lambda row: (row["high_count"], row["high_rate"]), reverse=True)[:max_edges]
    return {
        "threshold": float(threshold),
        "max_edges": int(max_edges),
        "edges": edges,
        "target_components": sorted({int(edge["target_component"]) for edge in edges}),
        "coact_components": sorted({int(edge["coact_component"]) for edge in edges}),
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    edges = stats["edges"]
    threshold = stats["threshold"]
    assert isinstance(edges, list)

    target_components = list(range(36))
    coact_components = list(range(36))
    target_y = {component: 35 - component for component in target_components}
    coact_y = {component: 35 - component for component in coact_components}
    max_count = max((int(edge["high_count"]) for edge in edges), default=1)

    fig, ax = plt.subplots(figsize=(10, 12))
    for component in target_components:
        ax.scatter(0.0, target_y[component], s=55, color=SERIES2[0], zorder=3)
        ax.text(-0.05, target_y[component], str(component), ha="right", va="center", fontsize=8)
    for component in coact_components:
        ax.scatter(1.0, coact_y[component], s=55, color=SERIES2[1], zorder=3)
        ax.text(1.05, coact_y[component], str(component), ha="left", va="center", fontsize=8)

    for edge in edges:
        source_y = target_y[int(edge["target_component"])]
        dest_y = coact_y[int(edge["coact_component"])]
        width = 0.4 + 4.0 * (float(edge["high_count"]) / max_count)
        alpha = 0.15 + 0.65 * min(float(edge["high_rate"]), 1.0)
        ax.plot([0.0, 1.0], [source_y, dest_y], color=INK_MUTED, linewidth=width, alpha=alpha)

    ax.set_title(f"Strongest Component Pair Coactivation Edges (PMI > {threshold:g})")
    ax.text(0.0, 36.4, "Target component", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.text(1.0, 36.4, "Coacting component", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-1, 37)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    edges = stats["edges"]
    assert isinstance(edges, list)
    write_csv(
        path,
        edges,
        ["target_component", "coact_component", "high_count", "pair_count", "high_rate", "mean_pmi"],
    )


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "max_edges": stats["max_edges"],
        "edge_count": len(stats["edges"]),
        "target_components": stats["target_components"],
        "coact_components": stats["coact_components"],
        "top_edges": stats["edges"][:20],
    }

