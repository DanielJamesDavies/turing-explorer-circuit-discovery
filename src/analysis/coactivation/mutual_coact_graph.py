"""Reciprocal high-PMI coactivation graph analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import configure_matplotlib
from .data import TopCoactivationArtifact, load_top_coactivation
from .graph_utils import build_high_pmi_edges, edge_codes
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class MutualCoactGraphResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_mutual_coact_graph(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
    top_n: int = 100,
) -> MutualCoactGraphResult:
    """Generate reciprocal high-PMI latent pair summaries."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"mutual coact graph requires mode='pmi', got {artifact.mode!r}")

    stats = compute_mutual_coact_graph(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        threshold=threshold,
        top_n=top_n,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "mutual-coact-graph.png"
    table_path = output_dirs["tables"] / "mutual-coact-graph.csv"
    summary_path = output_dirs["summaries"] / "mutual-coact-graph.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return MutualCoactGraphResult(figure_path, summary_path, table_path, summary)


def compute_mutual_coact_graph(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
    top_n: int = 100,
) -> dict[str, object]:
    """Find unordered latent pairs where both directed edges exceed threshold."""

    edges = build_high_pmi_edges(top_values, top_indices, threshold=threshold)
    if edges.count == 0:
        return _empty_stats(threshold)

    codes = edge_codes(edges.source, edges.dest, edges.num_latents)
    sorted_codes, order = torch.sort(codes)
    sorted_scores = edges.score[order]
    reverse_codes = edge_codes(edges.dest, edges.source, edges.num_latents)
    positions = torch.searchsorted(sorted_codes, reverse_codes)
    in_bounds = positions < sorted_codes.numel()
    found = torch.zeros_like(in_bounds, dtype=torch.bool)
    found[in_bounds] = sorted_codes[positions[in_bounds]] == reverse_codes[in_bounds]
    unordered = edges.source < edges.dest
    mutual_mask = found & unordered
    mutual_positions = torch.nonzero(mutual_mask, as_tuple=False).flatten()
    if mutual_positions.numel() == 0:
        return _empty_stats(threshold) | {"directed_high_pmi_edges": edges.count}

    reverse_scores = sorted_scores[positions[mutual_positions]]
    forward_scores = edges.score[mutual_positions]
    pair_strength = torch.minimum(forward_scores, reverse_scores)
    mean_score = (forward_scores + reverse_scores) / 2.0
    top_count = min(int(top_n), int(mutual_positions.numel()))
    top_strength, top_order = torch.topk(pair_strength, k=top_count)

    top_pairs = []
    for rank, (strength, local_idx) in enumerate(zip(top_strength.tolist(), top_order.tolist()), start=1):
        pos = int(mutual_positions[local_idx].item())
        src = int(edges.source[pos].item())
        dst = int(edges.dest[pos].item())
        top_pairs.append(
            {
                "rank": rank,
                "source_global_id": src,
                "source_component": src // int(d_sae),
                "source_latent": src % int(d_sae),
                "dest_global_id": dst,
                "dest_component": dst // int(d_sae),
                "dest_latent": dst % int(d_sae),
                "forward_pmi": float(edges.score[pos].item()),
                "reverse_pmi": float(reverse_scores[local_idx].item()),
                "mutual_strength": float(strength),
                "mean_pmi": float(mean_score[local_idx].item()),
            }
        )

    component_pair_counts = torch.zeros((top_values.shape[0], top_values.shape[0]), dtype=torch.int64)
    src_components = (edges.source[mutual_positions] // int(d_sae)).to(torch.int64)
    dst_components = (edges.dest[mutual_positions] // int(d_sae)).to(torch.int64)
    for src_component, dst_component in zip(src_components.tolist(), dst_components.tolist()):
        component_pair_counts[src_component, dst_component] += 1
        if src_component != dst_component:
            component_pair_counts[dst_component, src_component] += 1

    return {
        "threshold": float(threshold),
        "directed_high_pmi_edges": edges.count,
        "mutual_pair_count": int(mutual_positions.numel()),
        "top_pairs": top_pairs,
        "strength_mean": float(pair_strength.mean().item()),
        "strength_p50": float(torch.quantile(pair_strength, torch.tensor(0.5)).item()),
        "strength_p90": float(torch.quantile(pair_strength, torch.tensor(0.9)).item()),
        "component_pair_counts": component_pair_counts.tolist(),
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    component_pair_counts = torch.tensor(stats["component_pair_counts"], dtype=torch.float32)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    image = axes[0].imshow(component_pair_counts.numpy(), cmap="viridis", aspect="auto")
    axes[0].set_title("Mutual Coact Pairs By Component")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Component")
    axes[0].set_xticks(range(component_pair_counts.shape[1]))
    axes[0].set_yticks(range(component_pair_counts.shape[0]))
    axes[0].tick_params(axis="x", labelrotation=90, labelsize=7)
    axes[0].tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=axes[0], label="Mutual pair count")

    top_pairs = stats["top_pairs"]
    labels = [f"{row['source_component']}:{row['source_latent']}<->{row['dest_component']}:{row['dest_latent']}" for row in top_pairs[:15]]
    strengths = [row["mutual_strength"] for row in top_pairs[:15]]
    axes[1].bar(range(len(strengths)), strengths, color="#2f6f9f", alpha=0.85)
    axes[1].set_title("Strongest Mutual Latent Pairs")
    axes[1].set_ylabel("min(PMI A->B, PMI B->A)")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    write_csv(
        path,
        stats["top_pairs"],
        [
            "rank",
            "source_global_id",
            "source_component",
            "source_latent",
            "dest_global_id",
            "dest_component",
            "dest_latent",
            "forward_pmi",
            "reverse_pmi",
            "mutual_strength",
            "mean_pmi",
        ],
    )


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "directed_high_pmi_edges": stats["directed_high_pmi_edges"],
        "mutual_pair_count": stats["mutual_pair_count"],
        "strength_mean": stats["strength_mean"],
        "strength_p50": stats["strength_p50"],
        "strength_p90": stats["strength_p90"],
        "top_pairs": stats["top_pairs"][:30],
    }


def _empty_stats(threshold: float) -> dict[str, object]:
    return {
        "threshold": float(threshold),
        "directed_high_pmi_edges": 0,
        "mutual_pair_count": 0,
        "top_pairs": [],
        "strength_mean": 0.0,
        "strength_p50": 0.0,
        "strength_p90": 0.0,
        "component_pair_counts": [],
    }

