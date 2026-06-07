"""Hub-discounted shared-neighbor similarity for coactivation profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import configure_matplotlib
from .data import TopCoactivationArtifact, load_top_coactivation
from .graph_utils import build_high_pmi_edges, high_pmi_in_degree
from .profile_utils import deterministic_sample_indices
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class MutualNeighborSimilarityResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_mutual_neighbor_similarity(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
    max_samples: int = 30_000,
    bins: int = 80,
) -> MutualNeighborSimilarityResult:
    """Generate sampled hub-discounted shared-neighbor similarity distributions."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"mutual-neighbor similarity requires mode='pmi', got {artifact.mode!r}")

    stats = compute_mutual_neighbor_similarity(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        threshold=threshold,
        max_samples=max_samples,
        bins=bins,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "mutual-neighbor-similarity.png"
    table_path = output_dirs["tables"] / "mutual-neighbor-similarity.csv"
    summary_path = output_dirs["summaries"] / "mutual-neighbor-similarity.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    write_json(summary_path, summary)
    return MutualNeighborSimilarityResult(figure_path, summary_path, table_path, summary)


def compute_mutual_neighbor_similarity(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
    max_samples: int = 30_000,
    bins: int = 80,
) -> dict[str, object]:
    """Compare sampled target pairs by shared high-PMI coacting neighbors."""

    edges = build_high_pmi_edges(top_values, top_indices, threshold=threshold)
    in_degree = high_pmi_in_degree(edges).to(torch.float32)
    num_targets = int(top_values.shape[0] * top_values.shape[1])
    sample_indices = deterministic_sample_indices(num_targets, max_samples)
    components = (sample_indices // int(d_sae)).to(torch.int64)
    neighbor_ids, neighbor_scores = _sample_high_neighbors(
        top_values,
        top_indices,
        sample_indices=sample_indices,
        threshold=threshold,
    )
    same_a, same_b = _same_component_pairs(components)
    cross_a, cross_b = _cross_component_pairs(components)
    same_scores, same_common = _pair_scores(neighbor_ids, neighbor_scores, same_a, same_b, in_degree)
    cross_scores, cross_common = _pair_scores(neighbor_ids, neighbor_scores, cross_a, cross_b, in_degree)

    max_score = max(float(same_scores.max().item()) if same_scores.numel() else 0.0, float(cross_scores.max().item()) if cross_scores.numel() else 0.0, 1e-6)
    edges_for_hist = torch.linspace(0.0, max_score, bins + 1)
    centers = (edges_for_hist[:-1] + edges_for_hist[1:]) / 2
    same_hist = torch.histc(same_scores, bins=bins, min=0.0, max=max_score).to(torch.int64)
    cross_hist = torch.histc(cross_scores, bins=bins, min=0.0, max=max_score).to(torch.int64)
    top_pairs = _top_pair_rows(sample_indices, components, same_a, same_b, same_scores, same_common, d_sae, "same") + _top_pair_rows(
        sample_indices,
        components,
        cross_a,
        cross_b,
        cross_scores,
        cross_common,
        d_sae,
        "cross",
    )
    top_pairs = sorted(top_pairs, key=lambda row: row["shared_neighbor_score"], reverse=True)[:100]

    return {
        "threshold": float(threshold),
        "bin_left": edges_for_hist[:-1].tolist(),
        "bin_right": edges_for_hist[1:].tolist(),
        "bin_center": centers.tolist(),
        "same_counts": same_hist.tolist(),
        "cross_counts": cross_hist.tolist(),
        "same_density": (same_hist.float() / max(int(same_scores.numel()), 1)).tolist(),
        "cross_density": (cross_hist.float() / max(int(cross_scores.numel()), 1)).tolist(),
        "same_summary": _summary(same_scores, same_common),
        "cross_summary": _summary(cross_scores, cross_common),
        "sample_count": int(sample_indices.numel()),
        "same_pair_count": int(same_scores.numel()),
        "cross_pair_count": int(cross_scores.numel()),
        "top_pairs": top_pairs,
    }


def _sample_high_neighbors(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    sample_indices: torch.Tensor,
    threshold: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    flat_values = top_values.detach().cpu().to(torch.float32).reshape(-1, top_values.shape[-1])
    flat_indices = top_indices.detach().cpu().to(torch.int64).reshape(-1, top_indices.shape[-1])
    ids = []
    scores = []
    for row in sample_indices.tolist():
        values = flat_values[int(row)]
        mask = values > float(threshold)
        row_ids = flat_indices[int(row)][mask]
        row_scores = values[mask]
        order = torch.argsort(row_ids)
        ids.append(row_ids[order])
        scores.append(row_scores[order])
    return ids, scores


def _same_component_pairs(components: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left_parts = []
    right_parts = []
    for component in torch.unique(components).tolist():
        idx = torch.nonzero(components == int(component), as_tuple=False).flatten()
        if idx.numel() >= 2:
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


def _pair_scores(
    neighbor_ids: list[torch.Tensor],
    neighbor_scores: list[torch.Tensor],
    left: torch.Tensor,
    right: torch.Tensor,
    in_degree: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = []
    common_counts = []
    for left_idx, right_idx in zip(left.tolist(), right.tolist()):
        left_ids = neighbor_ids[int(left_idx)]
        right_ids = neighbor_ids[int(right_idx)]
        if left_ids.numel() == 0 or right_ids.numel() == 0:
            scores.append(0.0)
            common_counts.append(0)
            continue
        isin = torch.isin(left_ids, right_ids)
        if not bool(isin.any()):
            scores.append(0.0)
            common_counts.append(0)
            continue
        shared_ids = left_ids[isin]
        right_pos = torch.searchsorted(right_ids, shared_ids)
        shared_left_scores = neighbor_scores[int(left_idx)][isin]
        shared_right_scores = neighbor_scores[int(right_idx)][right_pos]
        weights = 1.0 / torch.log1p(in_degree[shared_ids].clamp(min=1.0))
        score = (torch.minimum(shared_left_scores, shared_right_scores) * weights).sum()
        scores.append(float(score.item()))
        common_counts.append(int(shared_ids.numel()))
    return torch.tensor(scores, dtype=torch.float32), torch.tensor(common_counts, dtype=torch.int64)


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
    ax.set_title("Hub-Discounted Shared Coact-Neighbor Similarity")
    ax.set_xlabel("Shared-neighbor score")
    ax.set_ylabel("Density within sampled pair type")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    write_csv(
        path,
        stats["top_pairs"],
        [
            "pair_type",
            "source_a_global_id",
            "source_a_component",
            "source_a_latent",
            "source_b_global_id",
            "source_b_component",
            "source_b_latent",
            "shared_neighbor_score",
            "shared_neighbor_count",
        ],
    )


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "sample_count": stats["sample_count"],
        "same_pair_count": stats["same_pair_count"],
        "cross_pair_count": stats["cross_pair_count"],
        "same_summary": stats["same_summary"],
        "cross_summary": stats["cross_summary"],
        "top_pairs": stats["top_pairs"][:30],
    }


def _summary(scores: torch.Tensor, common_counts: torch.Tensor) -> dict[str, float | int]:
    if scores.numel() == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "common_count_mean": 0.0}
    return {
        "count": int(scores.numel()),
        "mean": float(scores.mean().item()),
        "p50": float(torch.quantile(scores, torch.tensor(0.5)).item()),
        "p90": float(torch.quantile(scores, torch.tensor(0.9)).item()),
        "common_count_mean": float(common_counts.float().mean().item()),
    }


def _top_pair_rows(
    sample_indices: torch.Tensor,
    components: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    scores: torch.Tensor,
    common_counts: torch.Tensor,
    d_sae: int,
    pair_type: str,
    limit: int = 50,
) -> list[dict[str, object]]:
    if scores.numel() == 0:
        return []
    top_count = min(int(limit), int(scores.numel()))
    top_scores, order = torch.topk(scores, k=top_count)
    rows = []
    for score, pos in zip(top_scores.tolist(), order.tolist()):
        a_sample = int(left[pos].item())
        b_sample = int(right[pos].item())
        a = int(sample_indices[a_sample].item())
        b = int(sample_indices[b_sample].item())
        rows.append(
            {
                "pair_type": pair_type,
                "source_a_global_id": a,
                "source_a_component": int(components[a_sample].item()),
                "source_a_latent": a % int(d_sae),
                "source_b_global_id": b,
                "source_b_component": int(components[b_sample].item()),
                "source_b_latent": b % int(d_sae),
                "shared_neighbor_score": float(score),
                "shared_neighbor_count": int(common_counts[pos].item()),
            }
        )
    return rows

