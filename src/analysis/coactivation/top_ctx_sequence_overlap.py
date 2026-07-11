"""Exact top-context sequence overlap for coactivating latent pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import (
    FIGSIZE_WIDE,
    SERIES2,
    configure_matplotlib,
    panel_figsize,
    round_bars,
    save_figure,
    style_suptitle,
    styled_legend,
)
from .data import TopCoactivationArtifact, load_top_coactivation
from .graph_utils import build_high_pmi_edges, deterministic_edge_sample, load_top_context
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class TopCtxSequenceOverlapResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_top_ctx_sequence_overlap(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
    max_edge_samples: int = 50_000,
    bins: int = 65,
) -> TopCtxSequenceOverlapResult:
    """Generate exact top_ctx sequence-overlap distributions for coacting latents."""

    artifact = load_top_coactivation(run_root)
    ctx_artifact = load_top_context(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"top_ctx sequence overlap requires mode='pmi', got {artifact.mode!r}")
    if ctx_artifact.shape[:2] != artifact.shape[:2]:
        raise ValueError("top_ctx and top_coactivation component/d_sae dimensions must match")

    stats = compute_top_ctx_sequence_overlap(
        artifact.top_values,
        artifact.top_indices,
        ctx_artifact.ctx_seq_idx,
        d_sae=artifact.d_sae,
        threshold=threshold,
        max_edge_samples=max_edge_samples,
        bins=bins,
    )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "top-ctx-sequence-overlap.png"
    table_path = output_dirs["tables"] / "top-ctx-sequence-overlap.csv"
    summary_path = output_dirs["summaries"] / "top-ctx-sequence-overlap.json"
    readable_figure_path = output_dirs["figures"] / "top-ctx-sequence-overlap-readable.png"

    _write_plot(figure_path, stats)
    _write_readable_plot(readable_figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, ctx_artifact.path, stats)
    summary["readable_figure_path"] = str(readable_figure_path)
    write_json(summary_path, summary)
    return TopCtxSequenceOverlapResult(figure_path, summary_path, table_path, summary)


def compute_top_ctx_sequence_overlap(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    ctx_seq_idx: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
    max_edge_samples: int = 50_000,
    bins: int = 65,
) -> dict[str, object]:
    """Compare exact top_ctx sequence IDs for high-PMI edges and random pairs."""

    edges = build_high_pmi_edges(top_values, top_indices, threshold=threshold)
    if edges.count == 0:
        return _empty_stats(threshold)

    positions = deterministic_edge_sample(edges.count, max_edge_samples)
    coact_source = edges.source[positions]
    coact_dest = edges.dest[positions]
    coact_pmi = edges.score[positions]
    flat_ctx = ctx_seq_idx.detach().cpu().to(torch.int64).reshape(-1, ctx_seq_idx.shape[-1])
    coact_counts, coact_jaccard = _overlap_scores(flat_ctx, coact_source, coact_dest)
    random_source, random_dest = _random_baseline_pairs(edges.num_latents, int(positions.numel()))
    random_counts, random_jaccard = _overlap_scores(flat_ctx, random_source, random_dest)

    max_count = int(ctx_seq_idx.shape[-1])
    hist_bins = min(int(bins), max_count + 1)
    coact_hist = torch.histc(coact_counts.float(), bins=hist_bins, min=0.0, max=float(max_count)).to(torch.int64)
    random_hist = torch.histc(random_counts.float(), bins=hist_bins, min=0.0, max=float(max_count)).to(torch.int64)
    bin_edges = torch.linspace(0.0, float(max_count), hist_bins + 1)
    top_pairs = _top_pair_rows(coact_source, coact_dest, coact_pmi, coact_counts, coact_jaccard, d_sae)

    return {
        "threshold": float(threshold),
        "sample_count": int(positions.numel()),
        "top_ctx_k": int(ctx_seq_idx.shape[-1]),
        "bin_left": bin_edges[:-1].tolist(),
        "bin_right": bin_edges[1:].tolist(),
        "coact_counts": coact_hist.tolist(),
        "random_counts": random_hist.tolist(),
        "coact_density": (coact_hist.float() / max(int(coact_counts.numel()), 1)).tolist(),
        "random_density": (random_hist.float() / max(int(random_counts.numel()), 1)).tolist(),
        "coact_summary": _summary(coact_counts, coact_jaccard),
        "random_summary": _summary(random_counts, random_jaccard),
        "top_pairs": top_pairs,
    }


def _overlap_scores(ctx: torch.Tensor, source: torch.Tensor, dest: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = []
    jaccards = []
    for src, dst in zip(source.tolist(), dest.tolist()):
        left = ctx[int(src)]
        right = ctx[int(dst)]
        left = torch.unique(left[left >= 0])
        right = torch.unique(right[right >= 0])
        if left.numel() == 0 or right.numel() == 0:
            counts.append(0)
            jaccards.append(0.0)
            continue
        overlap = int(torch.isin(left, right).sum().item())
        union = int(left.numel() + right.numel() - overlap)
        counts.append(overlap)
        jaccards.append(float(overlap / union) if union else 0.0)
    return torch.tensor(counts, dtype=torch.int64), torch.tensor(jaccards, dtype=torch.float32)


def _random_baseline_pairs(num_latents: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.linspace(0, num_latents - 1, steps=count, dtype=torch.float64).round().to(torch.int64)
    stride = max(num_latents // 3, 1)
    dest = (source + stride + torch.arange(count, dtype=torch.int64)) % int(num_latents)
    dest = torch.where(dest == source, (dest + 1) % int(num_latents), dest)
    return source, dest


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    centers = [(left + right) / 2.0 for left, right in zip(stats["bin_left"], stats["bin_right"])]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(centers, stats["coact_density"], linewidth=2.0, color=SERIES2[0], label="high-PMI coacting pairs")
    ax.plot(centers, stats["random_density"], linewidth=2.0, color=SERIES2[1], label="random latent pairs")
    ax.set_title("Exact Top-Context Sequence Overlap")
    ax.set_xlabel("Shared top_ctx sequence count")
    ax.set_ylabel("Density within pair type")
    styled_legend(ax, loc="upper right")
    save_figure(fig, path)


def _write_readable_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    bin_left = stats["bin_left"]
    bin_right = stats["bin_right"]
    coact_counts = stats["coact_counts"]
    random_counts = stats["random_counts"]
    assert isinstance(bin_left, list)
    assert isinstance(bin_right, list)
    assert isinstance(coact_counts, list)
    assert isinstance(random_counts, list)

    centers = [(left + right) / 2.0 for left, right in zip(bin_left, bin_right)]
    coact_total = max(sum(int(count) for count in coact_counts), 1)
    random_total = max(sum(int(count) for count in random_counts), 1)
    coact_density = [int(count) / coact_total for count in coact_counts]
    random_density = [int(count) / random_total for count in random_counts]
    coact_survival = _survival_density(coact_counts)
    random_survival = _survival_density(random_counts)

    fig, axes = plt.subplots(2, 3, figsize=panel_figsize(2, 3))
    axes[0, 0].semilogy(centers, coact_density, linewidth=2.0, color=SERIES2[0], label="high-PMI coacting pairs")
    axes[0, 0].semilogy(centers, random_density, linewidth=2.0, color=SERIES2[1], label="random latent pairs")
    axes[0, 0].set_title("Full Distribution (Log Y)")
    axes[0, 0].set_xlabel("Shared top_ctx sequence count")
    axes[0, 0].set_ylabel("Density")
    styled_legend(axes[0, 0], loc="upper right")

    nonzero = [idx for idx, center in enumerate(centers) if center > 0.5]
    axes[0, 1].plot(
        [centers[idx] for idx in nonzero],
        [coact_density[idx] for idx in nonzero],
        linewidth=2.0,
        color=SERIES2[0],
        label="high-PMI coacting pairs",
    )
    axes[0, 1].plot(
        [centers[idx] for idx in nonzero],
        [random_density[idx] for idx in nonzero],
        linewidth=2.0,
        color=SERIES2[1],
        label="random latent pairs",
    )
    axes[0, 1].set_title("Nonzero Overlap Only")
    axes[0, 1].set_xlabel("Shared top_ctx sequence count")
    axes[0, 1].set_ylabel("Density")
    styled_legend(axes[0, 1], loc="upper right")

    axes[0, 2].semilogy(centers, coact_survival, linewidth=2.0, color=SERIES2[0], label="high-PMI coacting pairs")
    axes[0, 2].semilogy(centers, random_survival, linewidth=2.0, color=SERIES2[1], label="random latent pairs")
    axes[0, 2].set_title("Survival Curve: P(overlap >= x)")
    axes[0, 2].set_xlabel("Shared top_ctx sequence count")
    axes[0, 2].set_ylabel("Fraction of pairs")
    styled_legend(axes[0, 2], loc="upper right")

    coact_summary = stats["coact_summary"]
    random_summary = stats["random_summary"]
    assert isinstance(coact_summary, dict)
    assert isinstance(random_summary, dict)
    labels = ["mean overlap", "p90 overlap"]
    coact_values = [
        float(coact_summary["overlap_mean"]),
        float(coact_summary["overlap_p90"]),
    ]
    random_values = [
        float(random_summary["overlap_mean"]),
        float(random_summary["overlap_p90"]),
    ]
    x = range(len(labels))
    axes[1, 0].bar([pos - 0.18 for pos in x], coact_values, width=0.3, color=SERIES2[0], label="coacting")
    axes[1, 0].bar([pos + 0.18 for pos in x], random_values, width=0.3, color=SERIES2[1], label="random")
    axes[1, 0].set_title("Overlap Count Summary")
    axes[1, 0].set_xticks(list(x))
    axes[1, 0].set_xticklabels(labels, rotation=20, ha="right")
    styled_legend(axes[1, 0], loc="upper right")

    jaccard_labels = ["mean Jaccard"]
    jaccard_x = range(len(jaccard_labels))
    axes[1, 1].bar(
        [pos - 0.18 for pos in jaccard_x],
        [float(coact_summary["jaccard_mean"])],
        width=0.3,
        color=SERIES2[0],
        label="coacting",
    )
    axes[1, 1].bar(
        [pos + 0.18 for pos in jaccard_x],
        [float(random_summary["jaccard_mean"])],
        width=0.3,
        color=SERIES2[1],
        label="random",
    )
    axes[1, 1].set_title("Mean Jaccard Summary")
    axes[1, 1].set_xticks(list(jaccard_x))
    axes[1, 1].set_xticklabels(jaccard_labels, rotation=20, ha="right")
    styled_legend(axes[1, 1], loc="upper right")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.95,
        "\n".join(
            [
                "Sampled pair summary",
                f"Coacting mean overlap: {float(coact_summary['overlap_mean']):.2f}",
                f"Random mean overlap: {float(random_summary['overlap_mean']):.2f}",
                f"Coacting p90 overlap: {float(coact_summary['overlap_p90']):.0f}",
                f"Random p90 overlap: {float(random_summary['overlap_p90']):.0f}",
                f"Coacting mean Jaccard: {float(coact_summary['jaccard_mean']):.4f}",
                f"Random mean Jaccard: {float(random_summary['jaccard_mean']):.4f}",
            ]
        ),
        va="top",
        fontsize=11,
    )

    round_bars(axes[1, 0])
    round_bars(axes[1, 1])
    style_suptitle(fig, "Exact Top-Context Sequence Overlap: Readable Views")
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    write_csv(
        path,
        stats["top_pairs"],
        [
            "rank",
            "source_global_id",
            "source_component",
            "source_latent",
            "dest_global_id",
            "dest_component",
            "dest_latent",
            "pmi",
            "shared_top_ctx_sequence_count",
            "top_ctx_sequence_jaccard",
        ],
    )


def _build_summary(artifact: TopCoactivationArtifact, top_ctx_path: Path, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "top_ctx_path": str(top_ctx_path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "sample_count": stats["sample_count"],
        "top_ctx_k": stats["top_ctx_k"],
        "coact_summary": stats["coact_summary"],
        "random_summary": stats["random_summary"],
        "top_pairs": stats["top_pairs"][:30],
    }


def _summary(counts: torch.Tensor, jaccard: torch.Tensor) -> dict[str, float | int]:
    if counts.numel() == 0:
        return {"count": 0, "overlap_mean": 0.0, "overlap_p50": 0.0, "overlap_p90": 0.0, "jaccard_mean": 0.0}
    return {
        "count": int(counts.numel()),
        "overlap_mean": float(counts.float().mean().item()),
        "overlap_p50": float(torch.quantile(counts.float(), torch.tensor(0.5)).item()),
        "overlap_p90": float(torch.quantile(counts.float(), torch.tensor(0.9)).item()),
        "jaccard_mean": float(jaccard.mean().item()),
    }


def _survival_density(counts: list[int]) -> list[float]:
    total = max(sum(int(count) for count in counts), 1)
    running = 0
    survival = [0.0 for _ in counts]
    for idx in range(len(counts) - 1, -1, -1):
        running += int(counts[idx])
        survival[idx] = running / total
    return survival


def _top_pair_rows(
    source: torch.Tensor,
    dest: torch.Tensor,
    pmi: torch.Tensor,
    counts: torch.Tensor,
    jaccard: torch.Tensor,
    d_sae: int,
    limit: int = 100,
) -> list[dict[str, object]]:
    if counts.numel() == 0:
        return []
    order = torch.argsort(counts.float() + jaccard, descending=True)[: min(int(limit), int(counts.numel()))]
    rows = []
    for rank, pos in enumerate(order.tolist(), start=1):
        src = int(source[pos].item())
        dst = int(dest[pos].item())
        rows.append(
            {
                "rank": rank,
                "source_global_id": src,
                "source_component": src // int(d_sae),
                "source_latent": src % int(d_sae),
                "dest_global_id": dst,
                "dest_component": dst // int(d_sae),
                "dest_latent": dst % int(d_sae),
                "pmi": float(pmi[pos].item()),
                "shared_top_ctx_sequence_count": int(counts[pos].item()),
                "top_ctx_sequence_jaccard": float(jaccard[pos].item()),
            }
        )
    return rows


def _empty_stats(threshold: float) -> dict[str, object]:
    return {
        "threshold": float(threshold),
        "sample_count": 0,
        "top_ctx_k": 0,
        "bin_left": [],
        "bin_right": [],
        "coact_counts": [],
        "random_counts": [],
        "coact_density": [],
        "random_density": [],
        "coact_summary": {},
        "random_summary": {},
        "top_pairs": [],
    }

