"""Component-pair heatmap for high-PMI coactivations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import FIGSIZE_SQUARE, SEQUENTIAL_CMAP, configure_matplotlib, save_figure
from .data import TopCoactivationArtifact, load_top_coactivation
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class ComponentPairHeatmapResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_component_pair_heatmap(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
) -> ComponentPairHeatmapResult:
    """Generate a heatmap of high-PMI coact rates by component pair."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"component-pair heatmap requires mode='pmi', got {artifact.mode!r}")

    stats = compute_component_pair_heatmap(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        threshold=threshold,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "component-pair-high-pmi-heatmap.png"
    table_path = output_dirs["tables"] / "component-pair-high-pmi-heatmap.csv"
    summary_path = output_dirs["summaries"] / "component-pair-high-pmi-heatmap.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return ComponentPairHeatmapResult(figure_path, summary_path, table_path, summary)


def compute_component_pair_heatmap(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
) -> dict[str, object]:
    """Compute high-PMI rate and mean PMI for each target/coact component pair."""

    if top_values.ndim != 3 or top_indices.ndim != 3:
        raise ValueError("top_values and top_indices must have shape [components, d_sae, top_k]")
    if tuple(top_values.shape) != tuple(top_indices.shape):
        raise ValueError("top_values and top_indices shapes must match")

    values = top_values.detach().cpu().to(torch.float32)
    indices = top_indices.detach().cpu().to(torch.int64)
    num_components = int(values.shape[0])
    pair_counts = torch.zeros((num_components, num_components), dtype=torch.int64)
    high_counts = torch.zeros((num_components, num_components), dtype=torch.int64)
    value_sums = torch.zeros((num_components, num_components), dtype=torch.float32)

    for target_component in range(num_components):
        coact_components = (indices[target_component].reshape(-1) // int(d_sae)).clamp(
            min=0,
            max=num_components - 1,
        )
        pmi_values = values[target_component].reshape(-1)
        pair_counts[target_component] = torch.bincount(
            coact_components,
            minlength=num_components,
        )[:num_components]
        high_counts[target_component] = torch.bincount(
            coact_components[pmi_values > float(threshold)],
            minlength=num_components,
        )[:num_components]
        value_sums[target_component].scatter_add_(0, coact_components, pmi_values)

    high_rate = high_counts.float() / pair_counts.clamp(min=1).float()
    mean_pmi = value_sums / pair_counts.clamp(min=1).float()
    top_pairs = _top_pairs(high_rate, high_counts, pair_counts, mean_pmi)

    return {
        "threshold": float(threshold),
        "pair_counts": pair_counts.tolist(),
        "high_counts": high_counts.tolist(),
        "high_rate": high_rate.tolist(),
        "mean_pmi": mean_pmi.tolist(),
        "top_pairs": top_pairs,
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    high_rate = torch.tensor(stats["high_rate"], dtype=torch.float32)
    threshold = stats["threshold"]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    image = ax.imshow(high_rate.numpy(), cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.grid(False)
    ax.set_title(f"Component Pair High-PMI Rate (PMI > {threshold:g})")
    ax.set_xlabel("Coacting component")
    ax.set_ylabel("Target component")
    ax.set_xticks(range(high_rate.shape[1]))
    ax.set_yticks(range(high_rate.shape[0]))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Fraction of stored coacts above threshold")
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    pair_counts = stats["pair_counts"]
    high_counts = stats["high_counts"]
    high_rate = stats["high_rate"]
    mean_pmi = stats["mean_pmi"]
    threshold = stats["threshold"]
    assert isinstance(pair_counts, list)
    assert isinstance(high_counts, list)
    assert isinstance(high_rate, list)
    assert isinstance(mean_pmi, list)

    rows = []
    for target_component, row in enumerate(pair_counts):
        for coact_component, count in enumerate(row):
            rows.append(
                {
                    "target_component": target_component,
                    "coact_component": coact_component,
                    "threshold": threshold,
                    "pair_count": count,
                    "high_count": high_counts[target_component][coact_component],
                    "high_rate": high_rate[target_component][coact_component],
                    "mean_pmi": mean_pmi[target_component][coact_component],
                }
            )
    write_csv(
        path,
        rows,
        [
            "target_component",
            "coact_component",
            "threshold",
            "pair_count",
            "high_count",
            "high_rate",
            "mean_pmi",
        ],
    )


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "top_pairs": stats["top_pairs"],
    }


def _top_pairs(
    high_rate: torch.Tensor,
    high_counts: torch.Tensor,
    pair_counts: torch.Tensor,
    mean_pmi: torch.Tensor,
    *,
    limit: int = 20,
    min_pair_count: int = 1000,
) -> list[dict[str, object]]:
    pairs = []
    for target_component in range(high_rate.shape[0]):
        for coact_component in range(high_rate.shape[1]):
            count = int(pair_counts[target_component, coact_component].item())
            if count < min_pair_count:
                continue
            pairs.append(
                {
                    "target_component": target_component,
                    "coact_component": coact_component,
                    "pair_count": count,
                    "high_count": int(high_counts[target_component, coact_component].item()),
                    "high_rate": float(high_rate[target_component, coact_component].item()),
                    "mean_pmi": float(mean_pmi[target_component, coact_component].item()),
                }
            )
    return sorted(pairs, key=lambda row: (row["high_rate"], row["high_count"]), reverse=True)[:limit]

