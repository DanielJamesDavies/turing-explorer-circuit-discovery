"""Sampled latent clustering on hashed coactivation fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import CLUSTER_CMAP, FIGSIZE_SQUARE, configure_matplotlib, save_figure
from .data import TopCoactivationArtifact, load_top_coactivation
from .profile_utils import build_hashed_coact_profiles, deterministic_sample_indices
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class LatentProfileClusteringResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_latent_profile_clustering(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    max_samples: int = 20_000,
    hash_bins: int = 256,
    cluster_count: int = 12,
) -> LatentProfileClusteringResult:
    """Cluster sampled latent coactivation fingerprints and plot PCA coordinates."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"latent coact-profile clustering requires mode='pmi', got {artifact.mode!r}")

    stats = compute_latent_profile_clustering(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        max_samples=max_samples,
        hash_bins=hash_bins,
        cluster_count=cluster_count,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "latent-coact-profile-clusters.png"
    table_path = output_dirs["tables"] / "latent-coact-profile-clusters.csv"
    summary_path = output_dirs["summaries"] / "latent-coact-profile-clusters.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return LatentProfileClusteringResult(figure_path, summary_path, table_path, summary)


def compute_latent_profile_clustering(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    max_samples: int = 20_000,
    hash_bins: int = 256,
    cluster_count: int = 12,
    iterations: int = 20,
) -> dict[str, object]:
    """Run deterministic k-means over sampled hashed coactivation profiles."""

    num_targets = int(top_values.shape[0] * top_values.shape[1])
    sample_indices = deterministic_sample_indices(num_targets, max_samples)
    profiles = build_hashed_coact_profiles(
        top_values,
        top_indices,
        sample_indices=sample_indices,
        hash_bins=hash_bins,
    )
    labels, centroids = _kmeans(profiles, cluster_count=cluster_count, iterations=iterations)
    coords, explained = _pca_coordinates(profiles)
    target_components = (sample_indices // int(d_sae)).to(torch.int64)
    target_latents = (sample_indices % int(d_sae)).to(torch.int64)
    cluster_summaries = _cluster_summaries(labels, target_components, int(top_values.shape[0]))

    return {
        "sample_indices": sample_indices.tolist(),
        "target_components": target_components.tolist(),
        "target_latents": target_latents.tolist(),
        "cluster_labels": labels.tolist(),
        "pc1": coords[:, 0].tolist(),
        "pc2": coords[:, 1].tolist(),
        "explained_variance": explained,
        "sample_count": int(sample_indices.numel()),
        "num_targets": num_targets,
        "hash_bins": int(hash_bins),
        "cluster_count": int(cluster_count),
        "cluster_summaries": cluster_summaries,
        "centroid_norms": centroids.norm(dim=1).tolist(),
    }


def _kmeans(profiles: torch.Tensor, *, cluster_count: int, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    k = min(int(cluster_count), int(profiles.shape[0]))
    initial_rows = torch.linspace(0, profiles.shape[0] - 1, steps=k, dtype=torch.float64).round().to(torch.int64)
    centroids = profiles[initial_rows].clone()
    labels = torch.zeros(profiles.shape[0], dtype=torch.int64)
    for _ in range(max(int(iterations), 1)):
        distances = torch.cdist(profiles, centroids)
        labels = distances.argmin(dim=1)
        new_centroids = centroids.clone()
        for cluster_idx in range(k):
            mask = labels == cluster_idx
            if bool(mask.any()):
                new_centroids[cluster_idx] = profiles[mask].mean(dim=0)
        centroids = new_centroids
    return labels, centroids


def _pca_coordinates(profiles: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    centered = profiles - profiles.mean(dim=0, keepdim=True)
    _, singular_values, components = torch.pca_lowrank(centered, q=2, center=False)
    coords = centered @ components[:, :2]
    total_variance = float((centered * centered).sum().item())
    explained = (singular_values[:2] ** 2 / max(total_variance, 1e-12)).tolist()
    return coords, explained


def _cluster_summaries(labels: torch.Tensor, components: torch.Tensor, num_components: int) -> list[dict[str, object]]:
    summaries = []
    for cluster_idx in torch.unique(labels).tolist():
        mask = labels == int(cluster_idx)
        component_counts = torch.bincount(components[mask], minlength=num_components)[:num_components]
        top_components = torch.topk(component_counts, k=min(5, num_components))
        summaries.append(
            {
                "cluster": int(cluster_idx),
                "size": int(mask.sum().item()),
                "top_components": [
                    {"component": int(component), "count": int(count)}
                    for count, component in zip(top_components.values.tolist(), top_components.indices.tolist())
                    if int(count) > 0
                ],
            }
        )
    return sorted(summaries, key=lambda row: row["cluster"])


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    pc1 = stats["pc1"]
    pc2 = stats["pc2"]
    labels = stats["cluster_labels"]
    explained = stats["explained_variance"]
    assert isinstance(pc1, list)
    assert isinstance(pc2, list)
    assert isinstance(labels, list)
    assert isinstance(explained, list)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    scatter = ax.scatter(pc1, pc2, c=labels, cmap=CLUSTER_CMAP, s=4, alpha=0.55, linewidths=0)
    ax.set_title("Sampled Latent Coact-Profile Clusters")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    rows = []
    for idx, sample_index in enumerate(stats["sample_indices"]):
        rows.append(
            {
                "sample_index": sample_index,
                "target_component": stats["target_components"][idx],
                "target_latent": stats["target_latents"][idx],
                "cluster": stats["cluster_labels"][idx],
                "pc1": stats["pc1"][idx],
                "pc2": stats["pc2"][idx],
            }
        )
    write_csv(path, rows, ["sample_index", "target_component", "target_latent", "cluster", "pc1", "pc2"])


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "sample_count": stats["sample_count"],
        "num_targets": stats["num_targets"],
        "hash_bins": stats["hash_bins"],
        "cluster_count": stats["cluster_count"],
        "explained_variance": stats["explained_variance"],
        "cluster_summaries": stats["cluster_summaries"],
    }

