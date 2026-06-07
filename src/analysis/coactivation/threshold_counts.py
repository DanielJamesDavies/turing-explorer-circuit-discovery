"""Per-target counts of strong coactivations above PMI thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import configure_matplotlib
from .data import TopCoactivationArtifact, load_top_coactivation
from .sorted_pmi_decay import SUITE_NAME

DEFAULT_THRESHOLDS = (1.0, 2.0, 5.0)


@dataclass(frozen=True)
class ThresholdCountsResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_threshold_counts(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> ThresholdCountsResult:
    """Generate per-target strong-coactivation count distributions."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"threshold count plot requires mode='pmi', got {artifact.mode!r}")

    stats = compute_threshold_counts(artifact.top_values, thresholds=thresholds)
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "strong-coact-counts.png"
    table_path = output_dirs["tables"] / "strong-coact-counts.csv"
    summary_path = output_dirs["summaries"] / "strong-coact-counts.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return ThresholdCountsResult(figure_path, summary_path, table_path, summary)


def compute_threshold_counts(
    top_values: torch.Tensor,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    """Count, for each target latent, how many stored coacts exceed each threshold."""

    if top_values.ndim != 3:
        raise ValueError("top_values must have shape [components, d_sae, top_k]")
    if not thresholds:
        raise ValueError("at least one threshold is required")

    values = top_values.detach().cpu().to(torch.float32)
    top_k = int(values.shape[-1])
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float]] = {}

    for threshold in thresholds:
        counts_per_target = (values > float(threshold)).sum(dim=2).reshape(-1).to(torch.int64)
        histogram = torch.bincount(counts_per_target, minlength=top_k + 1)
        quantiles = torch.quantile(
            counts_per_target.to(torch.float32),
            torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99]),
        )
        threshold_key = _threshold_key(threshold)
        summaries[threshold_key] = {
            "threshold": float(threshold),
            "mean": float(counts_per_target.float().mean().item()),
            "max": float(counts_per_target.max().item()),
            "p01": float(quantiles[0].item()),
            "p10": float(quantiles[1].item()),
            "p50": float(quantiles[2].item()),
            "p90": float(quantiles[3].item()),
            "p99": float(quantiles[4].item()),
        }
        for count_value, target_count in enumerate(histogram.tolist()):
            rows.append(
                {
                    "threshold": float(threshold),
                    "threshold_key": threshold_key,
                    "strong_coact_count": count_value,
                    "target_count": int(target_count),
                    "target_fraction": float(target_count) / max(int(counts_per_target.numel()), 1),
                }
            )

    return {
        "rows": rows,
        "summaries": summaries,
        "num_targets": int(values.shape[0] * values.shape[1]),
        "top_k": top_k,
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    rows = stats["rows"]
    summaries = stats["summaries"]
    assert isinstance(rows, list)
    assert isinstance(summaries, dict)

    fig, ax = plt.subplots(figsize=(10, 6))
    for threshold_key, summary in summaries.items():
        threshold_rows = [row for row in rows if row["threshold_key"] == threshold_key]
        x = [row["strong_coact_count"] for row in threshold_rows]
        y = [row["target_fraction"] for row in threshold_rows]
        label = f"PMI > {summary['threshold']:g}"
        ax.plot(x, y, linewidth=1.8, label=label)

    ax.set_title("Strong Coactivation Counts Per Target")
    ax.set_xlabel("Number of stored coacts above PMI threshold")
    ax.set_ylabel("Fraction of target latents")
    ax.set_xlim(0, stats["top_k"])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    rows = stats["rows"]
    assert isinstance(rows, list)
    write_csv(
        path,
        rows,
        ["threshold", "threshold_key", "strong_coact_count", "target_count", "target_fraction"],
    )


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "num_targets": stats["num_targets"],
        "top_k": stats["top_k"],
        "threshold_summaries": stats["summaries"],
    }


def _threshold_key(threshold: float) -> str:
    return f"gt_{threshold:g}".replace(".", "_")

