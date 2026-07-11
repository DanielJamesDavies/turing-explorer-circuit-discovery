"""Circuit summary plots for coactivation overlap metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import (
    BLUE,
    INK,
    INK_MUTED,
    SEQUENTIAL_CMAP,
    configure_matplotlib,
    panel_figsize,
    save_figure,
    style_suptitle,
    styled_boxplot,
)

SUITE_NAME = "circuit-coactivation"


@dataclass(frozen=True)
class CircuitCoactOverlapResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]
    paper_figure_path: Path | None = None


def plot_circuit_coact_overlap(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
) -> CircuitCoactOverlapResult:
    """Generate circuit-level coactivation overlap plots from `circuits/summary.json`."""

    root = resolve_run_root(run_root)
    summary_path = root / "circuits" / "summary.json"
    rows = load_circuit_summary_rows(summary_path)
    stats = compute_circuit_coact_overlap(rows)
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "circuit-coact-overlap.png"
    paper_figure_path = output_dirs["figures"] / "circuit-coact-overlap-paper.png"
    table_path = output_dirs["tables"] / "circuit-coact-overlap-top-circuits.csv"
    output_summary_path = output_dirs["summaries"] / "circuit-coact-overlap.json"

    _write_plot(figure_path, stats)
    _write_paper_plot(paper_figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(summary_path, stats)
    summary["paper_figure_path"] = str(paper_figure_path)
    write_json(output_summary_path, summary)
    return CircuitCoactOverlapResult(
        figure_path, output_summary_path, table_path, summary, paper_figure_path=paper_figure_path
    )


def load_circuit_summary_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load canonical circuit summary rows."""

    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f"circuit summary not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"circuit summary must be a list, got {type(payload).__name__}")
    return [row for row in payload if isinstance(row, dict)]


def compute_circuit_coact_overlap(rows: list[dict[str, Any]]) -> dict[str, object]:
    """Extract coactivation-vs-circuit summary metrics."""

    parsed = [_parse_row(row) for row in rows]
    parsed = [row for row in parsed if row is not None]
    coact_overlap = [float(row["coact_overlap_pct"]) for row in parsed]
    activator_overlap = [float(row["coact_overlap_pct_activators"]) for row in parsed]
    inhibitor_overlap = [float(row["coact_overlap_pct_inhibitors"]) for row in parsed]
    internode_density = [float(row["internode_coact_density_pct"]) for row in parsed]
    faithfulness = [float(row["counterfactual_faithfulness"]) for row in parsed]
    nodes = [int(row["nodes"]) for row in parsed]
    edges = [int(row["edges"]) for row in parsed]
    top_by_overlap = sorted(parsed, key=lambda row: float(row["coact_overlap_pct"]), reverse=True)[:100]
    top_by_density = sorted(parsed, key=lambda row: float(row["internode_coact_density_pct"]), reverse=True)[:100]

    return {
        "circuit_count": len(parsed),
        "coact_overlap": coact_overlap,
        "activator_overlap": activator_overlap,
        "inhibitor_overlap": inhibitor_overlap,
        "internode_density": internode_density,
        "faithfulness": faithfulness,
        "nodes": nodes,
        "edges": edges,
        "coact_summary": _summary(coact_overlap),
        "activator_summary": _summary(activator_overlap),
        "inhibitor_summary": _summary(inhibitor_overlap),
        "internode_density_summary": _summary(internode_density),
        "faithfulness_summary": _summary(faithfulness),
        "top_by_overlap": top_by_overlap,
        "top_by_density": top_by_density,
        "correlations": {
            "coact_overlap_vs_faithfulness": _pearson(coact_overlap, faithfulness),
            "internode_density_vs_faithfulness": _pearson(internode_density, faithfulness),
            "coact_overlap_vs_nodes": _pearson(coact_overlap, nodes),
            "internode_density_vs_nodes": _pearson(internode_density, nodes),
        },
    }


def _parse_row(row: dict[str, Any]) -> dict[str, object] | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    post = metadata.get("post_analysis")
    evals = metadata.get("evals")
    if not isinstance(post, dict) or not isinstance(evals, dict):
        return None
    faithfulness = evals.get("counterfactual_faithfulness")
    coact_overlap = post.get("coact_overlap_pct")
    internode_density = post.get("internode_coact_density_pct")
    if faithfulness is None or coact_overlap is None or internode_density is None:
        return None
    return {
        "uuid": str(row.get("uuid", "")),
        "name": str(row.get("name", "")),
        "seed_comp": int(metadata.get("seed_comp", -1)),
        "seed_latent": int(metadata.get("seed_latent", -1)),
        "nodes": int(row.get("nodes", metadata.get("n_nodes", 0))),
        "edges": int(row.get("edges", metadata.get("n_edges", 0))),
        "n_activators": int(metadata.get("n_activators", 0)),
        "n_inhibitors": int(metadata.get("n_inhibitors", 0)),
        "counterfactual_faithfulness": float(faithfulness),
        "coact_overlap_pct": float(coact_overlap),
        "coact_overlap_pct_activators": float(post.get("coact_overlap_pct_activators", 0.0)),
        "coact_overlap_pct_inhibitors": float(post.get("coact_overlap_pct_inhibitors", 0.0)),
        "internode_coact_density_pct": float(internode_density),
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    coact_overlap = stats["coact_overlap"]
    activator_overlap = stats["activator_overlap"]
    inhibitor_overlap = stats["inhibitor_overlap"]
    internode_density = stats["internode_density"]
    faithfulness = stats["faithfulness"]
    nodes = stats["nodes"]
    assert isinstance(coact_overlap, list)
    assert isinstance(activator_overlap, list)
    assert isinstance(inhibitor_overlap, list)
    assert isinstance(internode_density, list)
    assert isinstance(faithfulness, list)
    assert isinstance(nodes, list)

    fig, axes = plt.subplots(2, 2, figsize=panel_figsize(2, 2))
    bins = 50
    axes[0, 0].hist(coact_overlap, bins=bins, color=BLUE)
    axes[0, 0].set_title("Seed Coact Overlap With Circuit Nodes")
    axes[0, 0].set_xlabel("Circuit nodes in seed top coacts (%)")
    axes[0, 0].set_ylabel("Circuit count")

    styled_boxplot(
        axes[0, 1],
        [coact_overlap, activator_overlap, inhibitor_overlap],
        ["all nodes", "activators", "inhibitors"],
        [BLUE] * 3,
    )
    axes[0, 1].set_title("Coact Overlap By Circuit Node Role")
    axes[0, 1].set_ylabel("Nodes in seed top coacts (%)")

    axes[1, 0].hist(internode_density, bins=bins, color=BLUE)
    axes[1, 0].set_title("Mutual Coact Density Among Circuit Nodes")
    axes[1, 0].set_xlabel("Mutually coacting node pairs (%)")
    axes[1, 0].set_ylabel("Circuit count")

    sizes = [max(8.0, min(float(node) / 8.0, 80.0)) for node in nodes]
    scatter = axes[1, 1].scatter(
        coact_overlap,
        faithfulness,
        s=sizes,
        c=internode_density,
        cmap=SEQUENTIAL_CMAP,
        alpha=0.75,
        edgecolors="none",
    )
    axes[1, 1].set_title("Coact Overlap vs Counterfactual Faithfulness")
    axes[1, 1].set_xlabel("Circuit nodes in seed top coacts (%)")
    axes[1, 1].set_ylabel("Counterfactual faithfulness")
    fig.colorbar(scatter, ax=axes[1, 1], label="Internode mutual coact density (%)")

    style_suptitle(fig, "Coactivation Overlap With Discovered Circuit Nodes")
    save_figure(fig, path)


def _write_paper_plot(path: Path, stats: dict[str, object]) -> None:
    """Compact two-panel variant for the paper's non-recoverability claim."""

    plt = configure_matplotlib()
    coact_overlap = stats["coact_overlap"]
    faithfulness = stats["faithfulness"]
    coact_summary = stats["coact_summary"]
    correlations = stats["correlations"]
    assert isinstance(coact_overlap, list)
    assert isinstance(faithfulness, list)
    assert isinstance(coact_summary, dict)
    assert isinstance(correlations, dict)

    fig, axes = plt.subplots(1, 2, figsize=panel_figsize(1, 2))

    axes[0].hist(coact_overlap, bins=60, color=BLUE)
    mean_overlap = float(coact_summary["mean"])
    axes[0].axvline(mean_overlap, color=INK, linestyle="--", linewidth=1.4)
    axes[0].annotate(
        f"mean = {mean_overlap:.1f}%",
        (mean_overlap, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=10.5,
        fontweight="medium",
        color=INK,
    )
    axes[0].set_title("Seed Coact Overlap Is Near Zero")
    axes[0].set_xlabel("Circuit nodes in seed top coacts (%)")
    axes[0].set_ylabel("Circuit count")

    axes[1].scatter(coact_overlap, faithfulness, s=10, color=BLUE, alpha=0.35, edgecolors="none")
    pearson_r = float(correlations["coact_overlap_vs_faithfulness"])
    y_cap = 2.0
    clipped = sum(1 for value in faithfulness if float(value) > y_cap)
    axes[1].set_ylim(min(-0.1, min(float(v) for v in faithfulness)), y_cap)
    acceptance_floor = min(float(value) for value in faithfulness)
    axes[1].axhline(acceptance_floor, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    axes[1].annotate(
        f"acceptance threshold ({acceptance_floor:.2f})",
        (0.02, acceptance_floor),
        xycoords=("axes fraction", "data"),
        xytext=(0, -13),
        textcoords="offset points",
        fontsize=9.5,
        fontweight="medium",
        color=INK_MUTED,
    )
    note = f"Pearson r = {pearson_r:.2f} (all points)"
    if clipped:
        note += f"\n{clipped} points above axis"
    axes[1].annotate(
        note,
        (0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10.5,
        fontweight="medium",
        color=INK_MUTED,
    )
    axes[1].set_title("Overlap Does Not Predict Faithfulness")
    axes[1].set_xlabel("Circuit nodes in seed top coacts (%)")
    axes[1].set_ylabel("Counterfactual faithfulness")

    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    rows = []
    for row in stats["top_by_overlap"]:
        rows.append({"ranking": "top_seed_coact_overlap", **row})
    for row in stats["top_by_density"]:
        rows.append({"ranking": "top_internode_density", **row})
    write_csv(
        path,
        rows,
        [
            "ranking",
            "uuid",
            "name",
            "seed_comp",
            "seed_latent",
            "nodes",
            "edges",
            "n_activators",
            "n_inhibitors",
            "counterfactual_faithfulness",
            "coact_overlap_pct",
            "coact_overlap_pct_activators",
            "coact_overlap_pct_inhibitors",
            "internode_coact_density_pct",
        ],
    )


def _build_summary(source_path: Path, stats: dict[str, object]) -> dict[str, object]:
    return {
        "source_path": str(source_path),
        "circuit_count": stats["circuit_count"],
        "coact_summary": stats["coact_summary"],
        "activator_summary": stats["activator_summary"],
        "inhibitor_summary": stats["inhibitor_summary"],
        "internode_density_summary": stats["internode_density_summary"],
        "faithfulness_summary": stats["faithfulness_summary"],
        "correlations": stats["correlations"],
        "top_by_overlap": stats["top_by_overlap"][:20],
        "top_by_density": stats["top_by_density"][:20],
        "limitation": (
            "This run exposes aggregate circuit summary metrics, not full circuit node/edge lists. "
            "Exact coact-vs-causal-edge overlays require discovered_circuits.pt worker stores."
        ),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values]
    if not clean:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(clean)
    return {
        "count": len(clean),
        "mean": float(mean(clean)),
        "p50": float(median(clean)),
        "p90": float(_quantile(ordered, 0.90)),
        "max": float(max(clean)),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _pearson(left: list[float], right: list[float] | list[int]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = mean(float(value) for value in left)
    right_mean = mean(float(value) for value in right)
    left_centered = [float(value) - left_mean for value in left]
    right_centered = [float(value) - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    left_norm = sum(value * value for value in left_centered) ** 0.5
    right_norm = sum(value * value for value in right_centered) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))

