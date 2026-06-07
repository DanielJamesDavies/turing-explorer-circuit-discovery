"""Same-component versus cross-component PMI distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import configure_matplotlib
from .data import TopCoactivationArtifact, load_top_coactivation
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class SameCrossDistributionResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_same_cross_distribution(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    bins: int = 120,
    value_range: tuple[float, float] = (-5.0, 10.0),
) -> SameCrossDistributionResult:
    """Generate same-component versus cross-component PMI histograms."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"same/cross distribution requires mode='pmi', got {artifact.mode!r}")

    stats = compute_same_cross_distribution(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        bins=bins,
        value_range=value_range,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "same-vs-cross-component-pmi.png"
    table_path = output_dirs["tables"] / "same-vs-cross-component-pmi.csv"
    summary_path = output_dirs["summaries"] / "same-vs-cross-component-pmi.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return SameCrossDistributionResult(figure_path, summary_path, table_path, summary)


def compute_same_cross_distribution(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    bins: int = 120,
    value_range: tuple[float, float] = (-5.0, 10.0),
) -> dict[str, object]:
    """Compute exact same/cross PMI histograms by streaming target components."""

    if top_values.ndim != 3 or top_indices.ndim != 3:
        raise ValueError("top_values and top_indices must have shape [components, d_sae, top_k]")
    if tuple(top_values.shape) != tuple(top_indices.shape):
        raise ValueError("top_values and top_indices shapes must match")
    if bins <= 0:
        raise ValueError("bins must be positive")

    min_value, max_value = value_range
    values = top_values.detach().cpu().to(torch.float32)
    indices = top_indices.detach().cpu().to(torch.int64)
    num_components = int(values.shape[0])
    same_hist = torch.zeros(bins, dtype=torch.int64)
    cross_hist = torch.zeros(bins, dtype=torch.int64)
    same_count = 0
    cross_count = 0
    same_sum = 0.0
    cross_sum = 0.0
    same_high2 = 0
    cross_high2 = 0
    same_high5 = 0
    cross_high5 = 0

    for target_component in range(num_components):
        pmi_values = values[target_component].reshape(-1)
        coact_components = (indices[target_component].reshape(-1) // int(d_sae)).clamp(
            min=0,
            max=num_components - 1,
        )
        same_mask = coact_components == target_component
        same_values = pmi_values[same_mask]
        cross_values = pmi_values[~same_mask]

        if same_values.numel():
            same_hist += torch.histc(same_values, bins=bins, min=min_value, max=max_value).to(torch.int64)
            same_count += int(same_values.numel())
            same_sum += float(same_values.sum().item())
            same_high2 += int((same_values > 2.0).sum().item())
            same_high5 += int((same_values > 5.0).sum().item())
        if cross_values.numel():
            cross_hist += torch.histc(cross_values, bins=bins, min=min_value, max=max_value).to(torch.int64)
            cross_count += int(cross_values.numel())
            cross_sum += float(cross_values.sum().item())
            cross_high2 += int((cross_values > 2.0).sum().item())
            cross_high5 += int((cross_values > 5.0).sum().item())

    edges = torch.linspace(min_value, max_value, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "bin_left": edges[:-1].tolist(),
        "bin_right": edges[1:].tolist(),
        "bin_center": centers.tolist(),
        "same_counts": same_hist.tolist(),
        "cross_counts": cross_hist.tolist(),
        "same_density": (same_hist.float() / max(same_count, 1)).tolist(),
        "cross_density": (cross_hist.float() / max(cross_count, 1)).tolist(),
        "same_summary": _relationship_summary(same_count, same_sum, same_high2, same_high5),
        "cross_summary": _relationship_summary(cross_count, cross_sum, cross_high2, cross_high5),
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    centers = stats["bin_center"]
    same_density = stats["same_density"]
    cross_density = stats["cross_density"]
    assert isinstance(centers, list)
    assert isinstance(same_density, list)
    assert isinstance(cross_density, list)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, same_density, linewidth=2.0, label="same component", color="#2f6f9f")
    ax.plot(centers, cross_density, linewidth=2.0, label="cross component", color="#b45f06")
    ax.axvline(0.0, color="#111111", linewidth=1.0, alpha=0.6)
    ax.set_title("Same-Component vs Cross-Component Coactivation PMI")
    ax.set_xlabel("PMI score")
    ax.set_ylabel("Density within relationship type")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    bin_left = stats["bin_left"]
    bin_right = stats["bin_right"]
    bin_center = stats["bin_center"]
    same_counts = stats["same_counts"]
    cross_counts = stats["cross_counts"]
    same_density = stats["same_density"]
    cross_density = stats["cross_density"]
    assert isinstance(bin_left, list)
    assert isinstance(bin_right, list)
    assert isinstance(bin_center, list)
    assert isinstance(same_counts, list)
    assert isinstance(cross_counts, list)
    assert isinstance(same_density, list)
    assert isinstance(cross_density, list)

    rows = []
    for idx, center in enumerate(bin_center):
        rows.append(
            {
                "bin_left": bin_left[idx],
                "bin_right": bin_right[idx],
                "bin_center": center,
                "same_count": same_counts[idx],
                "cross_count": cross_counts[idx],
                "same_density": same_density[idx],
                "cross_density": cross_density[idx],
            }
        )
    write_csv(
        path,
        rows,
        [
            "bin_left",
            "bin_right",
            "bin_center",
            "same_count",
            "cross_count",
            "same_density",
            "cross_density",
        ],
    )


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "same_summary": stats["same_summary"],
        "cross_summary": stats["cross_summary"],
    }


def _relationship_summary(count: int, total: float, high2: int, high5: int) -> dict[str, float | int]:
    return {
        "count": count,
        "mean": total / max(count, 1),
        "pmi_gt2_count": high2,
        "pmi_gt2_rate": high2 / max(count, 1),
        "pmi_gt5_count": high5,
        "pmi_gt5_rate": high5 / max(count, 1),
    }

