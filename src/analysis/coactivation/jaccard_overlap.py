"""Sampled Jaccard overlap of target latents' top coact ID sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import FIGSIZE_WIDE, SERIES2, configure_matplotlib, save_figure, styled_legend
from .data import TopCoactivationArtifact, load_top_coactivation
from .profile_utils import deterministic_sample_indices
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class JaccardOverlapResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_jaccard_overlap(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    max_samples: int = 30_000,
    top_k: int = 32,
    bins: int = 80,
) -> JaccardOverlapResult:
    """Generate sampled same/cross Jaccard overlap distributions."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"Jaccard overlap requires mode='pmi', got {artifact.mode!r}")

    stats = compute_jaccard_overlap(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        max_samples=max_samples,
        top_k=top_k,
        bins=bins,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "topk-jaccard-overlap.png"
    table_path = output_dirs["tables"] / "topk-jaccard-overlap.csv"
    summary_path = output_dirs["summaries"] / "topk-jaccard-overlap.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return JaccardOverlapResult(figure_path, summary_path, table_path, summary)


def compute_jaccard_overlap(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    max_samples: int = 30_000,
    top_k: int = 32,
    bins: int = 80,
) -> dict[str, object]:
    """Compute sampled Jaccard overlap of top coact ID sets."""

    if top_values.ndim != 3 or top_indices.ndim != 3:
        raise ValueError("top_values and top_indices must have shape [components, d_sae, top_k]")
    if tuple(top_values.shape) != tuple(top_indices.shape):
        raise ValueError("top_values and top_indices shapes must match")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    num_targets = int(top_values.shape[0] * top_values.shape[1])
    sample_indices = deterministic_sample_indices(num_targets, max_samples)
    components = (sample_indices // int(d_sae)).to(torch.int64)
    sampled_ids = _sample_top_ids(top_values, top_indices, sample_indices=sample_indices, top_k=top_k)
    same_a, same_b = _same_component_pairs(components)
    cross_a, cross_b = _cross_component_pairs(components)
    same_jaccard = _pair_jaccard(sampled_ids, same_a, same_b)
    cross_jaccard = _pair_jaccard(sampled_ids, cross_a, cross_b)

    edges = torch.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    same_hist = torch.histc(same_jaccard, bins=bins, min=0.0, max=1.0).to(torch.int64)
    cross_hist = torch.histc(cross_jaccard, bins=bins, min=0.0, max=1.0).to(torch.int64)
    return {
        "bin_left": edges[:-1].tolist(),
        "bin_right": edges[1:].tolist(),
        "bin_center": centers.tolist(),
        "same_counts": same_hist.tolist(),
        "cross_counts": cross_hist.tolist(),
        "same_density": (same_hist.float() / max(int(same_jaccard.numel()), 1)).tolist(),
        "cross_density": (cross_hist.float() / max(int(cross_jaccard.numel()), 1)).tolist(),
        "same_summary": _summary(same_jaccard),
        "cross_summary": _summary(cross_jaccard),
        "sample_count": int(sample_indices.numel()),
        "same_pair_count": int(same_jaccard.numel()),
        "cross_pair_count": int(cross_jaccard.numel()),
        "top_k": int(top_k),
    }


def _sample_top_ids(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    sample_indices: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    flat_values = top_values.detach().cpu().to(torch.float32).reshape(-1, top_values.shape[-1])
    flat_indices = top_indices.detach().cpu().to(torch.int64).reshape(-1, top_indices.shape[-1])
    k = min(int(top_k), int(flat_values.shape[1]))
    rows = sample_indices.detach().cpu().to(torch.int64)
    sorted_positions = flat_values[rows].argsort(dim=1, descending=True)[:, :k]
    ids = flat_indices[rows].gather(1, sorted_positions)
    return ids.sort(dim=1).values


def _same_component_pairs(components: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left_parts = []
    right_parts = []
    for component in torch.unique(components).tolist():
        idx = torch.nonzero(components == int(component), as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        left_parts.append(idx[:-1])
        right_parts.append(idx[1:])
    if not left_parts:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty
    return torch.cat(left_parts), torch.cat(right_parts)


def _cross_component_pairs(components: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(components.numel(), dtype=torch.int64)
    for offset in (components.numel() // 3, components.numel() // 2, 1):
        rolled = torch.roll(indices, shifts=-int(offset))
        mask = components != components[rolled]
        if bool(mask.any()):
            return indices[mask], rolled[mask]
    empty = torch.empty(0, dtype=torch.int64)
    return empty, empty


def _pair_jaccard(sorted_ids: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    values = []
    for left_idx, right_idx in zip(left.tolist(), right.tolist()):
        a = sorted_ids[int(left_idx)]
        b = sorted_ids[int(right_idx)]
        intersection = torch.isin(a, b).sum().item()
        union = int(a.numel() + b.numel() - intersection)
        values.append(float(intersection) / max(union, 1))
    return torch.tensor(values, dtype=torch.float32)


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    centers = stats["bin_center"]
    same_density = stats["same_density"]
    cross_density = stats["cross_density"]
    top_k = stats["top_k"]
    assert isinstance(centers, list)
    assert isinstance(same_density, list)
    assert isinstance(cross_density, list)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(centers, same_density, linewidth=2.0, label="same target component", color=SERIES2[0])
    ax.plot(centers, cross_density, linewidth=2.0, label="cross target component", color=SERIES2[1])
    ax.set_title(f"Top-{top_k} Coact ID Jaccard Overlap")
    ax.set_xlabel("Jaccard overlap of sorted top coact ID sets")
    ax.set_ylabel("Density within sampled pair type")
    ax.set_xlim(0.0, 1.0)
    styled_legend(ax, loc="upper right")
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    rows = []
    for idx, center in enumerate(stats["bin_center"]):
        rows.append(
            {
                "bin_left": stats["bin_left"][idx],
                "bin_right": stats["bin_right"][idx],
                "bin_center": center,
                "same_count": stats["same_counts"][idx],
                "cross_count": stats["cross_counts"][idx],
                "same_density": stats["same_density"][idx],
                "cross_density": stats["cross_density"][idx],
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
        "sample_count": stats["sample_count"],
        "same_pair_count": stats["same_pair_count"],
        "cross_pair_count": stats["cross_pair_count"],
        "top_k": stats["top_k"],
        "same_summary": stats["same_summary"],
        "cross_summary": stats["cross_summary"],
    }


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    if values.numel() == 0:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    quantiles = torch.quantile(values, torch.tensor([0.10, 0.50, 0.90]))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }

