"""PCA view of sampled latent coactivation-profile fingerprints."""

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
class LatentProfilePcaResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_latent_profile_pca(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    max_samples: int = 20_000,
    hash_bins: int = 256,
) -> LatentProfilePcaResult:
    """Generate a sampled PCA scatter of latent coactivation fingerprints."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"latent coact-profile PCA requires mode='pmi', got {artifact.mode!r}")

    stats = compute_latent_profile_pca(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        max_samples=max_samples,
        hash_bins=hash_bins,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "latent-coact-profile-pca.png"
    table_path = output_dirs["tables"] / "latent-coact-profile-pca.csv"
    summary_path = output_dirs["summaries"] / "latent-coact-profile-pca.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return LatentProfilePcaResult(figure_path, summary_path, table_path, summary)


def compute_latent_profile_pca(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    max_samples: int = 20_000,
    hash_bins: int = 256,
) -> dict[str, object]:
    """Compute PCA coordinates for sampled hashed coactivation profiles."""

    if top_values.ndim != 3:
        raise ValueError("top_values must have shape [components, d_sae, top_k]")
    num_targets = int(top_values.shape[0] * top_values.shape[1])
    sample_indices = deterministic_sample_indices(num_targets, max_samples)
    profiles = build_hashed_coact_profiles(
        top_values,
        top_indices,
        sample_indices=sample_indices,
        hash_bins=hash_bins,
    )
    centered = profiles - profiles.mean(dim=0, keepdim=True)
    _, singular_values, components = torch.pca_lowrank(centered, q=2, center=False)
    coords = centered @ components[:, :2]
    total_variance = float((centered * centered).sum().item())
    explained = (singular_values[:2] ** 2 / max(total_variance, 1e-12)).tolist()
    target_components = (sample_indices // int(d_sae)).to(torch.int64)
    target_latents = (sample_indices % int(d_sae)).to(torch.int64)

    return {
        "sample_indices": sample_indices.tolist(),
        "target_components": target_components.tolist(),
        "target_latents": target_latents.tolist(),
        "pc1": coords[:, 0].tolist(),
        "pc2": coords[:, 1].tolist(),
        "explained_variance": explained,
        "sample_count": int(sample_indices.numel()),
        "num_targets": num_targets,
        "hash_bins": int(hash_bins),
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    pc1 = stats["pc1"]
    pc2 = stats["pc2"]
    components = stats["target_components"]
    explained = stats["explained_variance"]
    assert isinstance(pc1, list)
    assert isinstance(pc2, list)
    assert isinstance(components, list)
    assert isinstance(explained, list)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        pc1,
        pc2,
        c=components,
        cmap="tab20",
        s=4,
        alpha=0.55,
        linewidths=0,
    )
    ax.set_title("Sampled Latent Coactivation-Profile PCA")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Target component")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    rows = []
    for idx, sample_index in enumerate(stats["sample_indices"]):
        rows.append(
            {
                "sample_index": sample_index,
                "target_component": stats["target_components"][idx],
                "target_latent": stats["target_latents"][idx],
                "pc1": stats["pc1"][idx],
                "pc2": stats["pc2"][idx],
            }
        )
    write_csv(path, rows, ["sample_index", "target_component", "target_latent", "pc1", "pc2"])


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "sample_count": stats["sample_count"],
        "num_targets": stats["num_targets"],
        "hash_bins": stats["hash_bins"],
        "explained_variance": stats["explained_variance"],
    }

