"""Analyze CF-vs-ablation source overlap in hybrid gradient circuits."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping, Sequence

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import (
    BAR_WIDTH,
    BLUE,
    FIGSIZE_SINGLE,
    INK_MUTED,
    NEG_MODE_COLORS,
    SERIES3,
    configure_matplotlib,
    round_bars,
    save_figure,
    styled_boxplot,
    styled_legend,
)

from .gradient_method_eval_distribution import _float, _is_finite, _resolve_grid_results_path
from .gradient_method_neg_mode_grid_runner import GRID_NEG_MODES

SUITE_NAME = "hybrid-source-overlap"
HYBRID_METHOD = "hybrid_gradient"
MODE_LABELS = {
    "close": "Close",
    "random": "Random",
    "distant": "Distant",
}
MODE_COLORS = NEG_MODE_COLORS
STACK_FIELDS = (
    ("cf_only", "CF-only", SERIES3[0]),
    ("ablation_only", "Ablation-only", SERIES3[1]),
    ("intersection", "Shared", SERIES3[2]),
)
MODE_PAIRS = (
    ("close", "random"),
    ("close", "distant"),
    ("random", "distant"),
)
BASE_REQUIRED_FIELDS = {
    "method",
    "neg_mode",
    "candidate_index",
    "status",
    "counterfactual_faithfulness",
    "posctx_suppression_score",
    "source_cf_only_node_count",
    "source_ablation_only_node_count",
    "source_intersection_node_count",
    "source_union_node_count",
    "source_jaccard",
    "post_prune_cf_only_node_count",
    "post_prune_ablation_only_node_count",
    "post_prune_intersection_node_count",
    "post_prune_union_node_count",
    "post_prune_jaccard",
}
SUMMARY_METRICS = (
    "source_cf_only_node_count",
    "source_ablation_only_node_count",
    "source_intersection_node_count",
    "source_union_node_count",
    "source_jaccard",
    "source_cf_only_ratio",
    "source_ablation_only_ratio",
    "source_intersection_ratio",
    "post_prune_cf_only_node_count",
    "post_prune_ablation_only_node_count",
    "post_prune_intersection_node_count",
    "post_prune_union_node_count",
    "post_prune_jaccard",
    "post_prune_cf_only_ratio",
    "post_prune_ablation_only_ratio",
    "post_prune_intersection_ratio",
    "counterfactual_faithfulness",
    "posctx_suppression_score",
)
AGGREGATE_TABLE_FIELDS = [
    "neg_mode",
    "count",
    *[
        f"{metric}_{stat}"
        for metric in SUMMARY_METRICS
        for stat in ("count", "mean", "std", "median", "min", "max")
    ],
]
PAIRED_DELTA_TABLE_FIELDS = [
    "candidate_index",
    "mode_pair",
    "left_mode",
    "right_mode",
    "source_jaccard_delta",
    "post_prune_jaccard_delta",
    "source_union_delta",
    "post_prune_union_delta",
    "source_intersection_delta",
    "post_prune_intersection_delta",
    "faithfulness_delta",
    "suppression_delta",
]


@dataclass(frozen=True)
class HybridSourceOverlapResult:
    figure_paths: list[Path]
    summary_path: Path
    aggregate_table_path: Path
    paired_delta_table_path: Path
    summary: dict[str, object]


def plot_hybrid_source_overlap(
    run_root: str | Path,
    *,
    results_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> HybridSourceOverlapResult:
    """Generate hybrid source-overlap figures and tables from a gradient grid CSV."""

    root = resolve_run_root(run_root)
    table_path = _resolve_grid_results_path(root, results_path)
    rows = load_hybrid_source_overlap_rows(table_path)
    aggregate_rows, stats = compute_hybrid_source_overlap_stats(rows)
    paired_rows, paired_summary = compute_hybrid_source_overlap_paired_deltas(rows)

    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figures_dir = output_dirs["figures"]
    aggregate_table_path = output_dirs["tables"] / "hybrid-source-overlap.csv"
    paired_delta_table_path = output_dirs["tables"] / "hybrid-source-overlap-paired-deltas.csv"
    summary_path = output_dirs["summaries"] / "hybrid-source-overlap.json"

    write_csv(aggregate_table_path, aggregate_rows, AGGREGATE_TABLE_FIELDS)
    write_csv(paired_delta_table_path, paired_rows, PAIRED_DELTA_TABLE_FIELDS)
    figure_paths = write_hybrid_source_overlap_figures(figures_dir, rows, stats, paired_summary)
    summary = {
        "results_path": str(table_path),
        "row_count": len(rows),
        "neg_modes": list(GRID_NEG_MODES),
        "by_mode": stats["by_mode"],
        "paired_delta_summary": paired_summary,
        "figure_paths": [str(path) for path in figure_paths],
        "aggregate_table_path": str(aggregate_table_path),
        "paired_delta_table_path": str(paired_delta_table_path),
    }
    write_json(summary_path, summary)
    return HybridSourceOverlapResult(
        figure_paths=figure_paths,
        summary_path=summary_path,
        aggregate_table_path=aggregate_table_path,
        paired_delta_table_path=paired_delta_table_path,
        summary=summary,
    )


def load_hybrid_source_overlap_rows(path: str | Path) -> list[dict[str, Any]]:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"gradient method grid results not found: {table_path}")
    with table_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"gradient method grid results table is empty: {table_path}")

    missing = BASE_REQUIRED_FIELDS - set(rows[0])
    if missing:
        raise ValueError(f"gradient method grid results missing columns: {sorted(missing)}")

    accepted = [
        _with_overlap_ratios(row)
        for row in rows
        if row.get("method") == HYBRID_METHOD
        and row.get("status") == "accepted"
        and not row.get("error")
        and all(_is_finite(row.get(field)) for field in _finite_required_fields())
    ]
    if not accepted:
        raise ValueError(f"gradient method grid results have no accepted hybrid overlap rows: {table_path}")
    return accepted


def compute_hybrid_source_overlap_stats(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_mode: dict[str, dict[str, object]] = {}
    aggregate_rows: list[dict[str, object]] = []
    for mode in GRID_NEG_MODES:
        mode_rows = [row for row in rows if row["neg_mode"] == mode]
        summary = _summarize_group(mode_rows)
        by_mode[mode] = summary
        aggregate_rows.append(_summary_table_row(mode, summary))
    return aggregate_rows, {"by_mode": by_mode}


def compute_hybrid_source_overlap_paired_deltas(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_candidate[row["candidate_index"]][row["neg_mode"]] = row

    delta_rows: list[dict[str, object]] = []
    for candidate_index, modes in by_candidate.items():
        for left_mode, right_mode in MODE_PAIRS:
            left = modes.get(left_mode)
            right = modes.get(right_mode)
            if left is None or right is None:
                continue
            delta_rows.append(
                {
                    "candidate_index": candidate_index,
                    "mode_pair": f"{left_mode}-{right_mode}",
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "source_jaccard_delta": _float(left["source_jaccard"])
                    - _float(right["source_jaccard"]),
                    "post_prune_jaccard_delta": _float(left["post_prune_jaccard"])
                    - _float(right["post_prune_jaccard"]),
                    "source_union_delta": _float(left["source_union_node_count"])
                    - _float(right["source_union_node_count"]),
                    "post_prune_union_delta": _float(left["post_prune_union_node_count"])
                    - _float(right["post_prune_union_node_count"]),
                    "source_intersection_delta": _float(left["source_intersection_node_count"])
                    - _float(right["source_intersection_node_count"]),
                    "post_prune_intersection_delta": _float(left["post_prune_intersection_node_count"])
                    - _float(right["post_prune_intersection_node_count"]),
                    "faithfulness_delta": _float(left["counterfactual_faithfulness"])
                    - _float(right["counterfactual_faithfulness"]),
                    "suppression_delta": _float(left["posctx_suppression_score"])
                    - _float(right["posctx_suppression_score"]),
                }
            )
    return delta_rows, _summarize_paired_deltas(delta_rows)


def write_hybrid_source_overlap_figures(
    figures_dir: Path,
    rows: list[dict[str, Any]],
    stats: Mapping[str, object],
    paired_summary: Mapping[str, object],
) -> list[Path]:
    return [
        _plot_box_by_mode(
            figures_dir / "hybrid-source-jaccard-by-mode.png",
            rows,
            "source_jaccard",
            "Hybrid Source Jaccard by Negctx Mode",
            "Source Jaccard",
        ),
        _plot_stacked_counts(
            figures_dir / "hybrid-source-stacked-counts-by-mode.png",
            stats,
            prefix="source",
            title="Hybrid Source Composition by Negctx Mode",
            ylabel="Mean node count",
        ),
        _plot_stacked_counts(
            figures_dir / "hybrid-post-prune-stacked-counts-by-mode.png",
            stats,
            prefix="post_prune",
            title="Hybrid Post-Prune Composition by Negctx Mode",
            ylabel="Mean node count",
        ),
        _plot_stacked_counts(
            figures_dir / "hybrid-source-composition-ratio-by-mode.png",
            stats,
            prefix="source",
            title="Hybrid Source Composition Ratios by Negctx Mode",
            ylabel="Mean share of union",
            ratio=True,
            ylim=(0.0, 1.0),
        ),
        _plot_scatter_by_mode(
            figures_dir / "hybrid-overlap-vs-faithfulness.png",
            rows,
            x_metric="source_jaccard",
            y_metric="counterfactual_faithfulness",
            title="Hybrid Source Overlap vs Faithfulness",
            xlabel="Source Jaccard",
            ylabel="Counterfactual Faithfulness",
        ),
        _plot_scatter_by_mode(
            figures_dir / "hybrid-overlap-vs-suppression.png",
            rows,
            x_metric="source_jaccard",
            y_metric="posctx_suppression_score",
            title="Hybrid Source Overlap vs Suppression",
            xlabel="Source Jaccard",
            ylabel="Posctx Suppression Score",
        ),
        _plot_paired_delta_summary(
            figures_dir / "hybrid-paired-jaccard-deltas.png",
            paired_summary,
            metric="source_jaccard_delta",
            title="Paired Source Jaccard Deltas by Negctx Mode",
        ),
        _plot_pre_vs_post_jaccard(figures_dir / "hybrid-pre-vs-post-jaccard.png", rows),
    ]


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, object]:
    summary: dict[str, object] = {"count": len(rows)}
    for metric in SUMMARY_METRICS:
        summary[metric] = _distribution_summary(_metric_values(rows, metric))
    return summary


def _summary_table_row(mode: str, summary: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "neg_mode": mode,
        "count": summary["count"],
    }
    for metric in SUMMARY_METRICS:
        metric_summary = summary[metric]
        assert isinstance(metric_summary, dict)
        for stat in ("count", "mean", "std", "median", "min", "max"):
            row[f"{metric}_{stat}"] = metric_summary[stat]
    return row


def _summarize_paired_deltas(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for pair in [f"{left}-{right}" for left, right in MODE_PAIRS]:
        pair_rows = [row for row in rows if row["mode_pair"] == pair]
        summary[pair] = {
            "count": len(pair_rows),
            "source_jaccard_delta": _distribution_summary(_object_values(pair_rows, "source_jaccard_delta")),
            "post_prune_jaccard_delta": _distribution_summary(
                _object_values(pair_rows, "post_prune_jaccard_delta")
            ),
            "source_union_delta": _distribution_summary(_object_values(pair_rows, "source_union_delta")),
            "post_prune_union_delta": _distribution_summary(_object_values(pair_rows, "post_prune_union_delta")),
            "source_intersection_delta": _distribution_summary(
                _object_values(pair_rows, "source_intersection_delta")
            ),
            "post_prune_intersection_delta": _distribution_summary(
                _object_values(pair_rows, "post_prune_intersection_delta")
            ),
            "faithfulness_delta": _distribution_summary(_object_values(pair_rows, "faithfulness_delta")),
            "suppression_delta": _distribution_summary(_object_values(pair_rows, "suppression_delta")),
        }
    return summary


def _plot_box_by_mode(path: Path, rows: list[dict[str, Any]], metric: str, title: str, ylabel: str) -> Path:
    plt = configure_matplotlib()
    data = [_metric_values([row for row in rows if row["neg_mode"] == mode], metric) for mode in GRID_NEG_MODES]
    fig, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    styled_boxplot(
        axis,
        data,
        [MODE_LABELS[mode] for mode in GRID_NEG_MODES],
        [MODE_COLORS[mode] for mode in GRID_NEG_MODES],
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    return save_figure(fig, path)


def _plot_stacked_counts(
    path: Path,
    stats: Mapping[str, object],
    *,
    prefix: str,
    title: str,
    ylabel: str,
    ratio: bool = False,
    ylim: tuple[float, float] | None = None,
) -> Path:
    plt = configure_matplotlib()
    by_mode = stats["by_mode"]
    assert isinstance(by_mode, dict)
    labels = [MODE_LABELS[mode] for mode in GRID_NEG_MODES]
    bottoms = [0.0] * len(GRID_NEG_MODES)
    fig, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    for key, label, color in STACK_FIELDS:
        metric = f"{prefix}_{key}_ratio" if ratio else f"{prefix}_{key}_node_count"
        values = [float(by_mode[mode][metric]["mean"]) for mode in GRID_NEG_MODES]
        axis.bar(labels, values, width=0.55, bottom=bottoms, color=color, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    styled_legend(axis, loc="best")
    return save_figure(fig, path)


def _plot_scatter_by_mode(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    plt = configure_matplotlib()
    fig, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    for mode in GRID_NEG_MODES:
        mode_rows = [row for row in rows if row["neg_mode"] == mode]
        axis.scatter(
            _metric_values(mode_rows, x_metric),
            _metric_values(mode_rows, y_metric),
            color=MODE_COLORS[mode],
            alpha=0.75,
            edgecolors="none",
            label=MODE_LABELS[mode],
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    styled_legend(axis, loc="best")
    return save_figure(fig, path)


def _plot_paired_delta_summary(
    path: Path,
    paired_summary: Mapping[str, object],
    *,
    metric: str,
    title: str,
) -> Path:
    plt = configure_matplotlib()
    pairs = [f"{left}-{right}" for left, right in MODE_PAIRS]
    values = [
        float(paired_summary[pair][metric]["mean"])
        if int(paired_summary[pair][metric]["count"]) > 0
        else 0.0
        for pair in pairs
    ]
    fig, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    axis.bar(pairs, values, width=BAR_WIDTH, color=BLUE)
    axis.axhline(0.0, color=INK_MUTED, linewidth=1.0)
    axis.set_title(title)
    axis.set_ylabel("Mean delta")
    round_bars(axis)
    return save_figure(fig, path)


def _plot_pre_vs_post_jaccard(path: Path, rows: list[dict[str, Any]]) -> Path:
    plt = configure_matplotlib()
    fig, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    for mode in GRID_NEG_MODES:
        mode_rows = [row for row in rows if row["neg_mode"] == mode]
        axis.scatter(
            _metric_values(mode_rows, "source_jaccard"),
            _metric_values(mode_rows, "post_prune_jaccard"),
            color=MODE_COLORS[mode],
            alpha=0.75,
            edgecolors="none",
            label=MODE_LABELS[mode],
        )
    axis.plot([0.0, 1.0], [0.0, 1.0], color=INK_MUTED, linewidth=1.0, linestyle="--")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Hybrid Pre-Prune vs Post-Prune Jaccard")
    axis.set_xlabel("Pre-prune source Jaccard")
    axis.set_ylabel("Post-prune source Jaccard")
    styled_legend(axis, loc="best")
    return save_figure(fig, path)


def _with_overlap_ratios(row: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    for prefix in ("source", "post_prune"):
        union = _float(copied[f"{prefix}_union_node_count"])
        for key in ("cf_only", "ablation_only", "intersection"):
            count = _float(copied[f"{prefix}_{key}_node_count"])
            copied[f"{prefix}_{key}_ratio"] = count / union if union else 0.0
    return copied


def _finite_required_fields() -> tuple[str, ...]:
    return (
        "counterfactual_faithfulness",
        "posctx_suppression_score",
        "source_cf_only_node_count",
        "source_ablation_only_node_count",
        "source_intersection_node_count",
        "source_union_node_count",
        "source_jaccard",
        "post_prune_cf_only_node_count",
        "post_prune_ablation_only_node_count",
        "post_prune_intersection_node_count",
        "post_prune_union_node_count",
        "post_prune_jaccard",
    )


def _metric_values(rows: Iterable[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if _is_finite(value):
            values.append(_float(value))
    return values


def _object_values(rows: Iterable[Mapping[str, object]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if _is_finite(value):
            values.append(float(value))
    return values


def _distribution_summary(values: Sequence[float]) -> dict[str, object]:
    samples = [float(value) for value in values if math.isfinite(float(value))]
    if not samples:
        return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(samples),
        "mean": float(mean(samples)),
        "std": float(pstdev(samples)) if len(samples) > 1 else 0.0,
        "median": float(median(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
    }


__all__ = [
    "HybridSourceOverlapResult",
    "compute_hybrid_source_overlap_paired_deltas",
    "compute_hybrid_source_overlap_stats",
    "load_hybrid_source_overlap_rows",
    "plot_hybrid_source_overlap",
]
