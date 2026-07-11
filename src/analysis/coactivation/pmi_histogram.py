"""Global PMI score histogram for top-coactivation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import BLUE, FIGSIZE_WIDE, INK_MUTED, configure_matplotlib, save_figure
from .data import TopCoactivationArtifact, load_top_coactivation
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class PmiHistogramResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_pmi_histogram(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    bins: int = 120,
    value_range: tuple[float, float] = (-5.0, 10.0),
) -> PmiHistogramResult:
    """Generate a global histogram of all stored coactivation PMI scores."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"PMI histogram requires mode='pmi', got {artifact.mode!r}")

    stats = compute_pmi_histogram(artifact.top_values, bins=bins, value_range=value_range)
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "pmi-histogram.png"
    table_path = output_dirs["tables"] / "pmi-histogram.csv"
    summary_path = output_dirs["summaries"] / "pmi-histogram.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return PmiHistogramResult(figure_path, summary_path, table_path, summary)


def compute_pmi_histogram(
    top_values: torch.Tensor,
    *,
    bins: int = 120,
    value_range: tuple[float, float] = (-5.0, 10.0),
) -> dict[str, object]:
    """Compute an exact histogram over all PMI scores."""

    if top_values.ndim != 3:
        raise ValueError("top_values must have shape [components, d_sae, top_k]")
    if bins <= 0:
        raise ValueError("bins must be positive")
    min_value, max_value = value_range
    if min_value >= max_value:
        raise ValueError("value_range min must be less than max")

    values = top_values.detach().cpu().to(torch.float32).reshape(-1)
    counts = torch.histc(values, bins=bins, min=min_value, max=max_value).to(torch.int64)
    edges = torch.linspace(min_value, max_value, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    total = int(values.numel())

    return {
        "bin_left": edges[:-1].tolist(),
        "bin_right": edges[1:].tolist(),
        "bin_center": centers.tolist(),
        "counts": counts.tolist(),
        "density": (counts.float() / max(total, 1)).tolist(),
        "total_scores": total,
        "score_min": float(values.min().item()),
        "score_max": float(values.max().item()),
        "score_mean": float(values.mean().item()),
        "score_std": float(values.std().item()),
        "clamp_min_count": int((values <= min_value).sum().item()),
        "clamp_max_count": int((values >= max_value).sum().item()),
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    centers = stats["bin_center"]
    density = stats["density"]
    bin_left = stats["bin_left"]
    bin_right = stats["bin_right"]
    assert isinstance(centers, list)
    assert isinstance(density, list)
    assert isinstance(bin_left, list)
    assert isinstance(bin_right, list)

    width = float(bin_right[0]) - float(bin_left[0])
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(centers, density, width=width, color=BLUE, align="center")
    ax.axvline(0.0, color=INK_MUTED, linewidth=1.0)
    ax.set_title("Coactivation PMI Score Distribution")
    ax.set_xlabel("PMI score")
    ax.set_ylabel("Fraction of stored coactivation scores")
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    bin_left = stats["bin_left"]
    bin_right = stats["bin_right"]
    bin_center = stats["bin_center"]
    counts = stats["counts"]
    density = stats["density"]
    assert isinstance(bin_left, list)
    assert isinstance(bin_right, list)
    assert isinstance(bin_center, list)
    assert isinstance(counts, list)
    assert isinstance(density, list)

    rows = [
        {
            "bin_left": left,
            "bin_right": right,
            "bin_center": center,
            "count": count,
            "density": density_value,
        }
        for left, right, center, count, density_value in zip(
            bin_left, bin_right, bin_center, counts, density
        )
    ]
    write_csv(path, rows, ["bin_left", "bin_right", "bin_center", "count", "density"])


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "total_scores": stats["total_scores"],
        "score_min": stats["score_min"],
        "score_max": stats["score_max"],
        "score_mean": stats["score_mean"],
        "score_std": stats["score_std"],
        "clamp_min_count": stats["clamp_min_count"],
        "clamp_max_count": stats["clamp_max_count"],
    }

