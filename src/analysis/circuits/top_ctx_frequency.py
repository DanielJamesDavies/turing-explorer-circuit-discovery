"""Compare top-context frequency overlap for circuit nodes versus coact nodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import torch

from analysis.coactivation.data import load_top_coactivation
from analysis.coactivation.graph_utils import load_top_context
from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import configure_matplotlib
from store.circuits import Circuit
from .coact_overlap import SUITE_NAME
from .node_hop_overlap import DEFAULT_KINDS, _circuit_node_sets, load_circuit_store, resolve_circuit_store_path


@dataclass(frozen=True)
class TopCtxFrequencyResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_top_ctx_circuit_vs_coact_frequency(
    run_root: str | Path,
    *,
    circuit_store_path: str | Path | None = None,
    output_root: str | Path | None = None,
    sample_size: int = 1024,
    threshold: float = 2.0,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> TopCtxFrequencyResult:
    """Plot top-context sequence overlap counts for circuit and coact latents."""

    root = resolve_run_root(run_root)
    store_path = resolve_circuit_store_path(root, circuit_store_path)
    circuits = load_circuit_store(store_path)
    coacts = load_top_coactivation(root)
    top_ctx = load_top_context(root)
    stats = compute_top_ctx_circuit_vs_coact_frequency(
        circuits,
        coacts.top_values,
        coacts.top_indices,
        top_ctx.ctx_seq_idx,
        d_sae=coacts.d_sae,
        sample_size=sample_size,
        threshold=threshold,
        kinds=kinds,
    )
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "top-ctx-circuit-vs-coact-frequency.png"
    table_path = output_dirs["tables"] / "top-ctx-circuit-vs-coact-frequency.csv"
    summary_path = output_dirs["summaries"] / "top-ctx-circuit-vs-coact-frequency.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(store_path, coacts.path, top_ctx.path, stats)
    write_json(summary_path, summary)
    return TopCtxFrequencyResult(figure_path, summary_path, table_path, summary)


def compute_top_ctx_circuit_vs_coact_frequency(
    circuits: Mapping[str, Circuit],
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    top_ctx_indices: torch.Tensor,
    *,
    d_sae: int,
    sample_size: int = 1024,
    threshold: float = 2.0,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> dict[str, object]:
    """Compute top-context sequence overlap counts for circuit and coact node sets."""

    values = top_values.detach().cpu().to(torch.float32).reshape(-1, top_values.shape[-1])
    indices = top_indices.detach().cpu().to(torch.int64).reshape(-1, top_indices.shape[-1])
    ctx = top_ctx_indices.detach().cpu().to(torch.int64).reshape(-1, top_ctx_indices.shape[-1])
    n_kinds = len(kinds)
    sampled = _deterministic_circuit_sample(
        [
            circuit
            for circuit in circuits.values()
            if circuit.metadata.get("seed_comp") is not None and circuit.metadata.get("seed_latent") is not None
        ],
        sample_size,
    )

    circuit_counts: list[int] = []
    coact_counts: list[int] = []
    circuit_pcts: list[float] = []
    coact_pcts: list[float] = []
    rows: list[dict[str, Any]] = []

    for circuit in sampled:
        seed_comp = int(circuit.metadata["seed_comp"])
        seed_latent = int(circuit.metadata["seed_latent"])
        seed_gid = seed_comp * int(d_sae) + seed_latent
        if seed_gid < 0 or seed_gid >= ctx.shape[0]:
            continue
        seed_ctx = _ctx_set(ctx[seed_gid])
        if not seed_ctx:
            continue
        node_sets = _circuit_node_sets(circuit, n_kinds=n_kinds, d_sae=d_sae, kinds=kinds, seed_global_id=seed_gid)
        circuit_nodes = {gid for gid in node_sets["all"] if 0 <= gid < ctx.shape[0]}
        coact_nodes = _coact_nodes_for_seed(values, indices, seed_gid, threshold=threshold)

        circuit_seed_counts = [_overlap_count(seed_ctx, ctx[gid]) for gid in sorted(circuit_nodes)]
        coact_seed_counts = [_overlap_count(seed_ctx, ctx[gid]) for gid in sorted(coact_nodes)]
        circuit_counts.extend(circuit_seed_counts)
        coact_counts.extend(coact_seed_counts)
        denom = float(len(seed_ctx))
        circuit_seed_pcts = [(count / denom) * 100.0 for count in circuit_seed_counts]
        coact_seed_pcts = [(count / denom) * 100.0 for count in coact_seed_counts]
        circuit_pcts.extend(circuit_seed_pcts)
        coact_pcts.extend(coact_seed_pcts)
        intersection_count = len(circuit_nodes & coact_nodes)
        rows.append(
            {
                "uuid": circuit.uuid,
                "name": circuit.name,
                "seed_global_id": seed_gid,
                "seed_comp": seed_comp,
                "seed_latent": seed_latent,
                "seed_top_ctx_count": len(seed_ctx),
                "circuit_node_count": len(circuit_nodes),
                "coact_node_count": len(coact_nodes),
                "circuit_and_coact_node_count": intersection_count,
                "circuit_mean_overlap_count": _mean(circuit_seed_counts),
                "coact_mean_overlap_count": _mean(coact_seed_counts),
                "circuit_median_overlap_count": _median(circuit_seed_counts),
                "coact_median_overlap_count": _median(coact_seed_counts),
                "circuit_nonzero_overlap_pct": _nonzero_pct(circuit_seed_counts),
                "coact_nonzero_overlap_pct": _nonzero_pct(coact_seed_counts),
                "circuit_mean_overlap_pct": _mean(circuit_seed_pcts),
                "coact_mean_overlap_pct": _mean(coact_seed_pcts),
            }
        )

    return {
        "sample_size": int(sample_size),
        "actual_seed_count": len(rows),
        "threshold": float(threshold),
        "top_ctx_k": int(top_ctx_indices.shape[-1]),
        "circuit_counts": circuit_counts,
        "coact_counts": coact_counts,
        "circuit_pcts": circuit_pcts,
        "coact_pcts": coact_pcts,
        "circuit_summary": _summary(circuit_counts),
        "coact_summary": _summary(coact_counts),
        "circuit_pct_summary": _summary(circuit_pcts),
        "coact_pct_summary": _summary(coact_pcts),
        "rows": rows,
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    circuit_counts = [int(value) for value in stats["circuit_counts"]]
    coact_counts = [int(value) for value in stats["coact_counts"]]
    rows = stats["rows"]
    assert isinstance(rows, list)
    top_ctx_k = int(stats["top_ctx_k"])
    bins = list(range(0, top_ctx_k + 2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(circuit_counts, bins=bins, alpha=0.65, label="circuit nodes", color="#1f77b4")
    axes[0, 0].hist(coact_counts, bins=bins, alpha=0.55, label="coact nodes", color="#ff7f0e")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Top-ctx overlap count distribution")
    axes[0, 0].set_xlabel(f"Shared seed top-ctx sequences out of {top_ctx_k}")
    axes[0, 0].set_ylabel("Latent-pair count (log)")
    axes[0, 0].legend(loc="best")

    _plot_cdf(axes[0, 1], circuit_counts, "circuit nodes", "#1f77b4")
    _plot_cdf(axes[0, 1], coact_counts, "coact nodes", "#ff7f0e")
    axes[0, 1].set_title("Cumulative distribution")
    axes[0, 1].set_xlabel(f"Shared seed top-ctx sequences out of {top_ctx_k}")
    axes[0, 1].set_ylabel("Cumulative fraction")
    axes[0, 1].legend(loc="best")

    circuit_means = [float(row["circuit_mean_overlap_count"]) for row in rows]
    coact_means = [float(row["coact_mean_overlap_count"]) for row in rows]
    axes[1, 0].boxplot([circuit_means, coact_means], labels=["circuit", "coact"], showfliers=False)
    axes[1, 0].set_title("Per-seed mean overlap")
    axes[1, 0].set_ylabel("Mean shared top-ctx sequences")

    axes[1, 1].scatter(coact_means, circuit_means, s=14, alpha=0.45, color="#2ca02c")
    max_mean = max(circuit_means + coact_means + [1.0])
    axes[1, 1].plot([0, max_mean], [0, max_mean], linestyle="--", color="#666666", linewidth=1.0)
    axes[1, 1].set_title("Per-seed circuit vs coact mean")
    axes[1, 1].set_xlabel("Coact-node mean overlap")
    axes[1, 1].set_ylabel("Circuit-node mean overlap")

    fig.suptitle("Top-Context Frequency: Circuit Nodes vs Coact Nodes", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    rows = stats["rows"]
    assert isinstance(rows, list)
    fieldnames = [
        "uuid",
        "name",
        "seed_global_id",
        "seed_comp",
        "seed_latent",
        "seed_top_ctx_count",
        "circuit_node_count",
        "coact_node_count",
        "circuit_and_coact_node_count",
        "circuit_mean_overlap_count",
        "coact_mean_overlap_count",
        "circuit_median_overlap_count",
        "coact_median_overlap_count",
        "circuit_nonzero_overlap_pct",
        "coact_nonzero_overlap_pct",
        "circuit_mean_overlap_pct",
        "coact_mean_overlap_pct",
    ]
    write_csv(path, rows, fieldnames)


def _build_summary(store_path: Path, coact_path: Path, top_ctx_path: Path, stats: dict[str, object]) -> dict[str, object]:
    return {
        "circuit_store_path": str(store_path),
        "top_coactivation_path": str(coact_path),
        "top_ctx_path": str(top_ctx_path),
        "sample_size": stats["sample_size"],
        "actual_seed_count": stats["actual_seed_count"],
        "threshold": stats["threshold"],
        "top_ctx_k": stats["top_ctx_k"],
        "circuit_summary": stats["circuit_summary"],
        "coact_summary": stats["coact_summary"],
        "circuit_pct_summary": stats["circuit_pct_summary"],
        "coact_pct_summary": stats["coact_pct_summary"],
    }


def _deterministic_circuit_sample(circuits: list[Circuit], sample_size: int) -> list[Circuit]:
    circuits = sorted(circuits, key=lambda circuit: circuit.uuid)
    if len(circuits) <= int(sample_size):
        return circuits
    positions = torch.linspace(0, len(circuits) - 1, steps=int(sample_size), dtype=torch.float64).round().to(torch.int64).unique()
    return [circuits[int(position)] for position in positions.tolist()]


def _coact_nodes_for_seed(values: torch.Tensor, indices: torch.Tensor, seed_gid: int, *, threshold: float) -> set[int]:
    if seed_gid < 0 or seed_gid >= values.shape[0]:
        return set()
    mask = values[seed_gid] > float(threshold)
    return {
        int(gid)
        for gid in indices[seed_gid][mask].tolist()
        if int(gid) != int(seed_gid) and int(gid) >= 0
    }


def _ctx_set(values: torch.Tensor) -> set[int]:
    return {int(value) for value in values.tolist() if int(value) > 0}


def _overlap_count(seed_ctx: set[int], candidate_ctx: torch.Tensor) -> int:
    return len(seed_ctx & _ctx_set(candidate_ctx))


def _plot_cdf(axis: Any, values: list[int], label: str, color: str) -> None:
    if not values:
        return
    sorted_values = sorted(values)
    y = [(index + 1) / len(sorted_values) for index in range(len(sorted_values))]
    axis.plot(sorted_values, y, label=label, color=color, linewidth=2.0)


def _summary(values: list[int] | list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _mean(values: list[int] | list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _median(values: list[int] | list[float]) -> float:
    return float(median(values)) if values else 0.0


def _nonzero_pct(values: list[int]) -> float:
    return (sum(1 for value in values if value > 0) / len(values) * 100.0) if values else 0.0

