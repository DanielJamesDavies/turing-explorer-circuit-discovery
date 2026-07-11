"""Top coacting latent hub analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import BLUE, configure_matplotlib, integer_ticks, round_bars, save_figure
from .data import TopCoactivationArtifact, load_top_coactivation
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class TopCoactHubsResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_top_coact_hubs(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
    top_n: int = 50,
) -> TopCoactHubsResult:
    """Generate exact high-PMI coacting latent hub counts."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"top coact hub analysis requires mode='pmi', got {artifact.mode!r}")

    stats = compute_top_coact_hubs(
        artifact.top_values,
        artifact.top_indices,
        num_components=artifact.num_components,
        d_sae=artifact.d_sae,
        threshold=threshold,
        top_n=top_n,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "top-coact-hubs.png"
    table_path = output_dirs["tables"] / "top-coact-hubs.csv"
    summary_path = output_dirs["summaries"] / "top-coact-hubs.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return TopCoactHubsResult(figure_path, summary_path, table_path, summary)


def compute_top_coact_hubs(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    num_components: int,
    d_sae: int,
    threshold: float = 2.0,
    top_n: int = 50,
) -> dict[str, object]:
    """Count how often each coacting latent appears above a PMI threshold."""

    if top_values.ndim != 3 or top_indices.ndim != 3:
        raise ValueError("top_values and top_indices must have shape [components, d_sae, top_k]")
    if tuple(top_values.shape) != tuple(top_indices.shape):
        raise ValueError("top_values and top_indices shapes must match")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    values = top_values.detach().cpu().to(torch.float32).reshape(-1)
    indices = top_indices.detach().cpu().to(torch.int64).reshape(-1)
    total_latents = int(num_components) * int(d_sae)
    high_mask = values > float(threshold)
    high_indices = indices[high_mask].clamp(min=0, max=total_latents - 1)
    high_values = values[high_mask]

    high_counts = torch.bincount(high_indices, minlength=total_latents).to(torch.int64)
    high_sums = torch.zeros(total_latents, dtype=torch.float32)
    if high_indices.numel():
        high_sums.scatter_add_(0, high_indices, high_values)
    high_mean = high_sums / high_counts.clamp(min=1).to(torch.float32)
    top_count = min(int(top_n), total_latents)
    top_counts, top_global_ids = torch.topk(high_counts, k=top_count)

    top_hubs = []
    for rank, (global_id, count) in enumerate(zip(top_global_ids.tolist(), top_counts.tolist()), start=1):
        component = int(global_id) // int(d_sae)
        latent = int(global_id) % int(d_sae)
        top_hubs.append(
            {
                "rank": rank,
                "global_latent_id": int(global_id),
                "component": component,
                "latent": latent,
                "high_pmi_count": int(count),
                "mean_high_pmi": float(high_mean[int(global_id)].item()),
            }
        )

    component_counts = high_counts.reshape(int(num_components), int(d_sae)).sum(dim=1)
    component_rates = component_counts.float() / max(int(high_mask.sum().item()), 1)
    return {
        "threshold": float(threshold),
        "top_hubs": top_hubs,
        "component_high_counts": component_counts.tolist(),
        "component_high_fractions": component_rates.tolist(),
        "total_high_pmi_edges": int(high_mask.sum().item()),
        "total_stored_edges": int(values.numel()),
        "unique_high_pmi_coact_latents": int((high_counts > 0).sum().item()),
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    top_hubs = stats["top_hubs"]
    component_counts = stats["component_high_counts"]
    threshold = stats["threshold"]
    assert isinstance(top_hubs, list)
    assert isinstance(component_counts, list)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2.0, 1.2]})
    hub_labels = [f"{row['component']}:{row['latent']}" for row in top_hubs[:25]]
    hub_counts = [row["high_pmi_count"] for row in top_hubs[:25]]
    axes[0].bar(range(len(hub_counts)), hub_counts, width=0.72, color=BLUE)
    axes[0].set_title(f"Top Coacting Latent Hubs (PMI > {threshold:g})")
    axes[0].set_xlabel("Coacting latent (component:latent)")
    axes[0].set_ylabel("High-PMI appearances")
    axes[0].set_xticks(range(len(hub_labels)))
    axes[0].set_xticklabels(hub_labels, rotation=75, ha="right", fontsize=8)
    integer_ticks(axes[0])

    axes[1].bar(range(len(component_counts)), component_counts, width=0.72, color=BLUE)
    axes[1].set_title("High-PMI Coact Appearances By Coacting Component")
    axes[1].set_xlabel("Coacting component")
    axes[1].set_ylabel("High-PMI appearances")
    axes[1].set_xticks(range(len(component_counts)))
    axes[1].tick_params(axis="x", labelsize=8)
    integer_ticks(axes[1])
    round_bars(axes[0])
    round_bars(axes[1])
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    top_hubs = stats["top_hubs"]
    assert isinstance(top_hubs, list)
    write_csv(
        path,
        top_hubs,
        ["rank", "global_latent_id", "component", "latent", "high_pmi_count", "mean_high_pmi"],
    )


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "total_high_pmi_edges": stats["total_high_pmi_edges"],
        "total_stored_edges": stats["total_stored_edges"],
        "unique_high_pmi_coact_latents": stats["unique_high_pmi_coact_latents"],
        "top_hubs": stats["top_hubs"][:20],
        "component_high_counts": stats["component_high_counts"],
        "component_high_fractions": stats["component_high_fractions"],
    }

