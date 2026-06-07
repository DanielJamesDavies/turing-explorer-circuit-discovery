"""Similarity between target components by outgoing coactivation signatures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import configure_matplotlib
from .component_pair_heatmap import compute_component_pair_heatmap
from .data import TopCoactivationArtifact, load_top_coactivation
from .profile_utils import normalize_rows
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class ComponentSignatureSimilarityResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_component_signature_similarity(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
) -> ComponentSignatureSimilarityResult:
    """Generate a target-component similarity heatmap from coact signatures."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"component signature similarity requires mode='pmi', got {artifact.mode!r}")

    stats = compute_component_signature_similarity(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        threshold=threshold,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "component-signature-similarity.png"
    table_path = output_dirs["tables"] / "component-signature-similarity.csv"
    summary_path = output_dirs["summaries"] / "component-signature-similarity.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return ComponentSignatureSimilarityResult(figure_path, summary_path, table_path, summary)


def compute_component_signature_similarity(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
) -> dict[str, object]:
    """Compute cosine similarity between target components' high-PMI coact signatures."""

    pair_stats = compute_component_pair_heatmap(
        top_values,
        top_indices,
        d_sae=d_sae,
        threshold=threshold,
    )
    signatures = torch.tensor(pair_stats["high_rate"], dtype=torch.float32)
    normalized = normalize_rows(signatures)
    similarity = normalized @ normalized.T
    top_pairs = _top_similar_components(similarity)
    return {
        "threshold": float(threshold),
        "signature_kind": "component_pair_high_pmi_rate",
        "signatures": signatures.tolist(),
        "similarity": similarity.tolist(),
        "top_similar_component_pairs": top_pairs,
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    similarity = torch.tensor(stats["similarity"], dtype=torch.float32)
    threshold = stats["threshold"]

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(similarity.numpy(), cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title(f"Target Component Coact-Signature Similarity (PMI > {threshold:g})")
    ax.set_xlabel("Target component")
    ax.set_ylabel("Target component")
    ax.set_xticks(range(similarity.shape[1]))
    ax.set_yticks(range(similarity.shape[0]))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Cosine similarity of coact-component signatures")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    similarity = stats["similarity"]
    assert isinstance(similarity, list)
    rows = []
    for component_a, row in enumerate(similarity):
        for component_b, score in enumerate(row):
            rows.append(
                {
                    "component_a": component_a,
                    "component_b": component_b,
                    "signature_similarity": score,
                }
            )
    write_csv(path, rows, ["component_a", "component_b", "signature_similarity"])


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "signature_kind": stats["signature_kind"],
        "top_similar_component_pairs": stats["top_similar_component_pairs"],
    }


def _top_similar_components(similarity: torch.Tensor, *, limit: int = 20) -> list[dict[str, object]]:
    rows = []
    for component_a in range(similarity.shape[0]):
        for component_b in range(component_a + 1, similarity.shape[1]):
            rows.append(
                {
                    "component_a": component_a,
                    "component_b": component_b,
                    "signature_similarity": float(similarity[component_a, component_b].item()),
                }
            )
    return sorted(rows, key=lambda row: row["signature_similarity"], reverse=True)[:limit]

