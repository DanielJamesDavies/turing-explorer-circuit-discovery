"""Hub-corrected high-PMI coactivation edge analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import configure_matplotlib
from .data import TopCoactivationArtifact, load_top_coactivation
from .graph_utils import build_high_pmi_edges, high_pmi_in_degree, top_edges_by_score
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class HubCorrectedCoactsResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_hub_corrected_coacts(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
    alpha: float = 1.0,
    top_n: int = 100,
) -> HubCorrectedCoactsResult:
    """Generate top hub-corrected high-PMI edges."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"hub-corrected coacts require mode='pmi', got {artifact.mode!r}")

    stats = compute_hub_corrected_coacts(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        threshold=threshold,
        alpha=alpha,
        top_n=top_n,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "hub-corrected-coacts.png"
    table_path = output_dirs["tables"] / "hub-corrected-coacts.csv"
    summary_path = output_dirs["summaries"] / "hub-corrected-coacts.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return HubCorrectedCoactsResult(figure_path, summary_path, table_path, summary)


def compute_hub_corrected_coacts(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
    alpha: float = 1.0,
    top_n: int = 100,
) -> dict[str, object]:
    """Score edges as PMI minus a normalized in-degree hub penalty."""

    edges = build_high_pmi_edges(top_values, top_indices, threshold=threshold)
    if edges.count == 0:
        return _empty_stats(threshold, alpha)

    in_degree = high_pmi_in_degree(edges)
    dest_degree = in_degree[edges.dest].to(torch.float32)
    max_degree = float(in_degree.max().item())
    penalty = torch.log1p(dest_degree) / max(float(torch.log1p(torch.tensor(max_degree)).item()), 1e-12)
    corrected = edges.score - float(alpha) * penalty
    top_corrected = top_edges_by_score(
        edges.source,
        edges.dest,
        corrected,
        d_sae=d_sae,
        limit=top_n,
        extra_columns={
            "raw_pmi": edges.score,
            "dest_high_pmi_in_degree": dest_degree.to(torch.int64),
            "hub_penalty": penalty,
        },
    )
    top_raw = top_edges_by_score(
        edges.source,
        edges.dest,
        edges.score,
        d_sae=d_sae,
        limit=top_n,
        extra_columns={
            "corrected_score": corrected,
            "dest_high_pmi_in_degree": dest_degree.to(torch.int64),
            "hub_penalty": penalty,
        },
    )

    raw_top_dest_components = _component_counts([row["dest_component"] for row in top_raw], top_values.shape[0])
    corrected_top_dest_components = _component_counts([row["dest_component"] for row in top_corrected], top_values.shape[0])
    return {
        "threshold": float(threshold),
        "alpha": float(alpha),
        "edge_count": edges.count,
        "max_in_degree": int(max_degree),
        "top_corrected_edges": top_corrected,
        "top_raw_edges": top_raw,
        "raw_top_dest_component_counts": raw_top_dest_components,
        "corrected_top_dest_component_counts": corrected_top_dest_components,
        "corrected_score_mean": float(corrected.mean().item()),
        "corrected_score_p50": float(torch.quantile(corrected, torch.tensor(0.5)).item()),
        "corrected_score_p90": float(torch.quantile(corrected, torch.tensor(0.9)).item()),
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    raw_counts = stats["raw_top_dest_component_counts"]
    corrected_counts = stats["corrected_top_dest_component_counts"]
    assert isinstance(raw_counts, list)
    assert isinstance(corrected_counts, list)

    x = list(range(len(raw_counts)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x, raw_counts, color="#2f6f9f", alpha=0.85)
    axes[0].set_title("Top Raw PMI Edges: Destination Components")
    axes[0].set_xlabel("Destination component")
    axes[0].set_ylabel("Count in top edge list")
    axes[0].set_xticks(x)
    axes[0].tick_params(axis="x", labelsize=7)
    axes[1].bar(x, corrected_counts, color="#b45f06", alpha=0.85)
    axes[1].set_title("Top Hub-Corrected Edges: Destination Components")
    axes[1].set_xlabel("Destination component")
    axes[1].set_ylabel("Count in top edge list")
    axes[1].set_xticks(x)
    axes[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    rows = []
    for row in stats["top_corrected_edges"]:
        rows.append({"edge_list": "top_corrected", **row})
    for row in stats["top_raw_edges"]:
        rows.append({"edge_list": "top_raw", **row})
    write_csv(
        path,
        rows,
        [
            "edge_list",
            "rank",
            "source_global_id",
            "source_component",
            "source_latent",
            "dest_global_id",
            "dest_component",
            "dest_latent",
            "score",
            "raw_pmi",
            "corrected_score",
            "dest_high_pmi_in_degree",
            "hub_penalty",
        ],
    )


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "alpha": stats["alpha"],
        "edge_count": stats["edge_count"],
        "max_in_degree": stats["max_in_degree"],
        "corrected_score_mean": stats["corrected_score_mean"],
        "corrected_score_p50": stats["corrected_score_p50"],
        "corrected_score_p90": stats["corrected_score_p90"],
        "top_corrected_edges": stats["top_corrected_edges"][:30],
        "top_raw_edges": stats["top_raw_edges"][:30],
    }


def _component_counts(components: list[int], num_components: int) -> list[int]:
    counts = [0 for _ in range(int(num_components))]
    for component in components:
        counts[int(component)] += 1
    return counts


def _empty_stats(threshold: float, alpha: float) -> dict[str, object]:
    return {
        "threshold": float(threshold),
        "alpha": float(alpha),
        "edge_count": 0,
        "max_in_degree": 0,
        "top_corrected_edges": [],
        "top_raw_edges": [],
        "raw_top_dest_component_counts": [],
        "corrected_top_dest_component_counts": [],
        "corrected_score_mean": 0.0,
        "corrected_score_p50": 0.0,
        "corrected_score_p90": 0.0,
    }

