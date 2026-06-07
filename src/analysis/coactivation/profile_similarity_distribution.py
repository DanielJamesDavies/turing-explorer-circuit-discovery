"""Within- versus cross-component coactivation-profile similarity distributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import configure_matplotlib
from .data import TopCoactivationArtifact, load_top_coactivation
from .profile_utils import build_hashed_coact_profiles, deterministic_sample_indices
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class ProfileSimilarityDistributionResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_profile_similarity_distribution(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    max_samples: int = 50_000,
    hash_bins: int = 256,
    bins: int = 80,
) -> ProfileSimilarityDistributionResult:
    """Generate sampled same/cross component latent-profile cosine histograms."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"profile similarity distribution requires mode='pmi', got {artifact.mode!r}")

    stats = compute_profile_similarity_distribution(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        max_samples=max_samples,
        hash_bins=hash_bins,
        bins=bins,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "latent-coact-profile-similarity.png"
    table_path = output_dirs["tables"] / "latent-coact-profile-similarity.csv"
    summary_path = output_dirs["summaries"] / "latent-coact-profile-similarity.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return ProfileSimilarityDistributionResult(figure_path, summary_path, table_path, summary)


def compute_profile_similarity_distribution(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    max_samples: int = 50_000,
    hash_bins: int = 256,
    bins: int = 80,
) -> dict[str, object]:
    """Compute sampled same/cross component cosine similarity histograms."""

    num_targets = int(top_values.shape[0] * top_values.shape[1])
    sample_indices = deterministic_sample_indices(num_targets, max_samples)
    profiles = build_hashed_coact_profiles(
        top_values,
        top_indices,
        sample_indices=sample_indices,
        hash_bins=hash_bins,
    )
    components = (sample_indices // int(d_sae)).to(torch.int64)
    same_a, same_b = _same_component_pairs(components)
    cross_a, cross_b = _cross_component_pairs(components)

    same_similarity = _pair_cosines(profiles, same_a, same_b)
    cross_similarity = _pair_cosines(profiles, cross_a, cross_b)
    edges = torch.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    same_hist = torch.histc(same_similarity, bins=bins, min=0.0, max=1.0).to(torch.int64)
    cross_hist = torch.histc(cross_similarity, bins=bins, min=0.0, max=1.0).to(torch.int64)

    return {
        "bin_left": edges[:-1].tolist(),
        "bin_right": edges[1:].tolist(),
        "bin_center": centers.tolist(),
        "same_counts": same_hist.tolist(),
        "cross_counts": cross_hist.tolist(),
        "same_density": (same_hist.float() / max(int(same_similarity.numel()), 1)).tolist(),
        "cross_density": (cross_hist.float() / max(int(cross_similarity.numel()), 1)).tolist(),
        "same_summary": _similarity_summary(same_similarity),
        "cross_summary": _similarity_summary(cross_similarity),
        "sample_count": int(sample_indices.numel()),
        "same_pair_count": int(same_similarity.numel()),
        "cross_pair_count": int(cross_similarity.numel()),
        "hash_bins": int(hash_bins),
    }


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


def _pair_cosines(profiles: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    return (profiles[left] * profiles[right]).sum(dim=1).clamp(min=0.0, max=1.0)


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    centers = stats["bin_center"]
    same_density = stats["same_density"]
    cross_density = stats["cross_density"]
    assert isinstance(centers, list)
    assert isinstance(same_density, list)
    assert isinstance(cross_density, list)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centers, same_density, linewidth=2.0, label="same target component", color="#2f6f9f")
    ax.plot(centers, cross_density, linewidth=2.0, label="cross target component", color="#b45f06")
    ax.set_title("Latent Coact-Profile Similarity Distribution")
    ax.set_xlabel("Cosine similarity of hashed coact fingerprints")
    ax.set_ylabel("Density within sampled pair type")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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
        "hash_bins": stats["hash_bins"],
        "same_summary": stats["same_summary"],
        "cross_summary": stats["cross_summary"],
    }


def _similarity_summary(values: torch.Tensor) -> dict[str, float | int]:
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

