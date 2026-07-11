"""Compare negative-context modes in gradient discovery grid results."""

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
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    INK_MUTED,
    METHOD_COLORS,
    NEG_MODE_COLORS,
    annotate_bars,
    configure_matplotlib,
    grouped_bar_geometry,
    panel_figsize,
    round_bars,
    save_figure,
    style_suptitle,
    styled_boxplot,
    styled_legend,
)

from .coact_overlap import SUITE_NAME
from .gradient_method_eval_distribution import (
    METHOD_LABELS,
    _float,
    _is_finite,
    _resolve_grid_results_path,
)
from .gradient_method_neg_mode_grid_runner import GRID_METHODS, GRID_NEG_MODES

METRIC_LABELS = {
    "counterfactual_faithfulness": "Counterfactual Faithfulness",
    "posctx_suppression_score": "Posctx Suppression Score",
    "n_nodes": "Circuit Nodes",
}
MODE_LABELS = {
    "close": "Close",
    "random": "Random",
    "distant": "Distant",
}
MODE_COLORS = NEG_MODE_COLORS
MODE_PAIRS = (
    ("close", "random"),
    ("close", "distant"),
    ("random", "distant"),
)
SUMMARY_METRICS = (
    "counterfactual_faithfulness",
    "posctx_suppression_score",
    "n_nodes",
    "n_edges",
)
AGGREGATE_TABLE_FIELDS = [
    "scope",
    "method",
    "neg_mode",
    "count",
    "accepted_count",
    "acceptance_rate",
    *[
        f"{metric}_{stat}"
        for metric in SUMMARY_METRICS
        for stat in ("count", "mean", "std", "median", "min", "max")
    ],
]
PAIRED_DELTA_TABLE_FIELDS = [
    "method",
    "candidate_index",
    "mode_pair",
    "left_mode",
    "right_mode",
    "faithfulness_delta",
    "suppression_delta",
    "nodes_delta",
    "edges_delta",
]


@dataclass(frozen=True)
class GradientNegModeComparisonResult:
    figure_paths: list[Path]
    summary_path: Path
    aggregate_table_path: Path
    paired_delta_table_path: Path
    summary: dict[str, object]


def plot_gradient_neg_mode_comparison(
    run_root: str | Path,
    *,
    results_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> GradientNegModeComparisonResult:
    """Generate mode-first, method-aware negctx comparison plots."""

    root = resolve_run_root(run_root)
    table_path = _resolve_grid_results_path(root, results_path)
    rows = load_gradient_neg_mode_rows(table_path)
    aggregate_rows, stats = compute_gradient_neg_mode_stats(rows)
    paired_rows, paired_summary = compute_paired_mode_deltas(rows)

    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figures_dir = output_dirs["figures"]
    aggregate_table_path = output_dirs["tables"] / "gradient-neg-mode-comparison.csv"
    paired_delta_table_path = output_dirs["tables"] / "gradient-neg-mode-paired-deltas.csv"
    summary_path = output_dirs["summaries"] / "gradient-neg-mode-comparison.json"

    write_csv(aggregate_table_path, aggregate_rows, AGGREGATE_TABLE_FIELDS)
    write_csv(paired_delta_table_path, paired_rows, PAIRED_DELTA_TABLE_FIELDS)
    figure_paths = write_gradient_neg_mode_figures(figures_dir, rows, stats, paired_summary)
    summary = {
        "results_path": str(table_path),
        "row_count": len(rows),
        "methods": list(GRID_METHODS),
        "neg_modes": list(GRID_NEG_MODES),
        "by_mode": stats["by_mode"],
        "by_method_mode": stats["by_method_mode"],
        "best_mode_by_method": stats["best_mode_by_method"],
        "paired_delta_summary": paired_summary,
        "figure_paths": [str(path) for path in figure_paths],
        "aggregate_table_path": str(aggregate_table_path),
        "paired_delta_table_path": str(paired_delta_table_path),
    }
    write_json(summary_path, summary)
    return GradientNegModeComparisonResult(
        figure_paths=figure_paths,
        summary_path=summary_path,
        aggregate_table_path=aggregate_table_path,
        paired_delta_table_path=paired_delta_table_path,
        summary=summary,
    )


def load_gradient_neg_mode_rows(path: str | Path) -> list[dict[str, Any]]:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"gradient method grid results not found: {table_path}")
    with table_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"gradient method grid results table is empty: {table_path}")
    required = {
        "method",
        "neg_mode",
        "candidate_index",
        "status",
        "n_nodes",
        "n_edges",
        "duration_s",
        "counterfactual_faithfulness",
        "posctx_suppression_score",
        "error",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"gradient method grid results missing columns: {sorted(missing)}")
    return rows


def compute_gradient_neg_mode_stats(rows: list[dict[str, Any]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    aggregate_rows: list[dict[str, object]] = []
    by_mode: dict[str, dict[str, object]] = {}
    by_method_mode: dict[str, dict[str, dict[str, object]]] = {}

    for mode in GRID_NEG_MODES:
        mode_rows = [row for row in rows if row["neg_mode"] == mode]
        summary = _summarize_group(mode_rows)
        by_mode[mode] = summary
        aggregate_rows.append(_summary_table_row("mode", "all", mode, summary))

    for method in GRID_METHODS:
        by_method_mode[method] = {}
        for mode in GRID_NEG_MODES:
            group_rows = [row for row in rows if row["method"] == method and row["neg_mode"] == mode]
            summary = _summarize_group(group_rows)
            by_method_mode[method][mode] = summary
            aggregate_rows.append(_summary_table_row("method_mode", method, mode, summary))

    stats = {
        "by_mode": by_mode,
        "by_method_mode": by_method_mode,
        "best_mode_by_method": _best_modes_by_method(by_method_mode),
    }
    return aggregate_rows, stats


def compute_paired_mode_deltas(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if _accepted_finite(row):
            by_key[(row["method"], row["candidate_index"])][row["neg_mode"]] = row

    delta_rows: list[dict[str, object]] = []
    for (method, candidate_index), modes in by_key.items():
        for left_mode, right_mode in MODE_PAIRS:
            left = modes.get(left_mode)
            right = modes.get(right_mode)
            if left is None or right is None:
                continue
            delta_rows.append(
                {
                    "method": method,
                    "candidate_index": candidate_index,
                    "mode_pair": f"{left_mode}-{right_mode}",
                    "left_mode": left_mode,
                    "right_mode": right_mode,
                    "faithfulness_delta": _float(left["counterfactual_faithfulness"])
                    - _float(right["counterfactual_faithfulness"]),
                    "suppression_delta": _float(left["posctx_suppression_score"])
                    - _float(right["posctx_suppression_score"]),
                    "nodes_delta": _float(left["n_nodes"]) - _float(right["n_nodes"]),
                    "edges_delta": _float(left["n_edges"]) - _float(right["n_edges"]),
                }
            )
    return delta_rows, _summarize_paired_deltas(delta_rows)


def write_gradient_neg_mode_figures(
    figures_dir: Path,
    rows: list[dict[str, Any]],
    stats: Mapping[str, object],
    paired_summary: Mapping[str, object],
) -> list[Path]:
    paths = [
        _plot_acceptance_rate(figures_dir / "gradient-neg-mode-acceptance-rate.png", stats),
        _plot_acceptance_rate_by_method(figures_dir / "gradient-neg-mode-acceptance-rate-by-method.png", stats),
        _plot_box_by_mode(
            figures_dir / "gradient-neg-mode-faithfulness-distribution.png",
            rows,
            "counterfactual_faithfulness",
            "Counterfactual Faithfulness by Negctx Mode",
            accepted_only=True,
        ),
        _plot_box_by_method_mode(
            figures_dir / "gradient-neg-mode-faithfulness-by-method.png",
            rows,
            "counterfactual_faithfulness",
            "Counterfactual Faithfulness by Method and Mode",
            accepted_only=True,
        ),
        _plot_box_by_mode(
            figures_dir / "gradient-neg-mode-suppression-distribution.png",
            rows,
            "posctx_suppression_score",
            "Posctx Suppression by Negctx Mode",
            accepted_only=True,
        ),
        _plot_box_by_method_mode(
            figures_dir / "gradient-neg-mode-suppression-by-method.png",
            rows,
            "posctx_suppression_score",
            "Posctx Suppression by Method and Mode",
            accepted_only=True,
        ),
        _plot_box_by_mode(
            figures_dir / "gradient-neg-mode-circuit-size.png",
            rows,
            "n_nodes",
            "Circuit Size by Negctx Mode",
            accepted_only=True,
        ),
        _plot_box_by_method_mode(
            figures_dir / "gradient-neg-mode-circuit-size-by-method.png",
            rows,
            "n_nodes",
            "Circuit Size by Method and Mode",
            accepted_only=True,
            box_edge="match",
        ),
        _plot_paired_delta_summary(
            figures_dir / "gradient-neg-mode-paired-deltas-faithfulness.png",
            paired_summary,
            "faithfulness_delta",
            "Paired Faithfulness Deltas by Method",
        ),
        _plot_paired_delta_summary(
            figures_dir / "gradient-neg-mode-paired-deltas-suppression.png",
            paired_summary,
            "suppression_delta",
            "Paired Suppression Deltas by Method",
        ),
    ]
    return paths


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, object]:
    accepted = [row for row in rows if _accepted_finite(row)]
    summary: dict[str, object] = {
        "count": len(rows),
        "accepted_count": len(accepted),
        "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
    }
    metric_sources = {
        "counterfactual_faithfulness": accepted,
        "posctx_suppression_score": accepted,
        "n_nodes": accepted,
        "n_edges": accepted,
    }
    for metric, source_rows in metric_sources.items():
        summary[metric] = _distribution_summary(_metric_values(source_rows, metric))
    return summary


def _summary_table_row(scope: str, method: str, mode: str, summary: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "scope": scope,
        "method": method,
        "neg_mode": mode,
        "count": summary["count"],
        "accepted_count": summary["accepted_count"],
        "acceptance_rate": summary["acceptance_rate"],
    }
    for metric in SUMMARY_METRICS:
        metric_summary = summary[metric]
        assert isinstance(metric_summary, dict)
        for stat in ("count", "mean", "std", "median", "min", "max"):
            row[f"{metric}_{stat}"] = metric_summary[stat]
    return row


def _best_modes_by_method(by_method_mode: Mapping[str, Mapping[str, Mapping[str, object]]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    metric_specs = {
        "acceptance_rate": (lambda summary: float(summary["acceptance_rate"]), True),
        "mean_faithfulness": (lambda summary: float(_nested(summary, "counterfactual_faithfulness", "mean")), True),
        "mean_suppression": (lambda summary: float(_nested(summary, "posctx_suppression_score", "mean")), True),
    }
    for method, by_mode in by_method_mode.items():
        best[method] = {}
        for metric_name, (getter, larger_is_better) in metric_specs.items():
            ordered = sorted(
                GRID_NEG_MODES,
                key=lambda mode: getter(by_mode[mode]),
                reverse=larger_is_better,
            )
            best[method][metric_name] = ordered[0]
    return best


def _summarize_paired_deltas(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"overall": {}, "by_method": {}}
    for pair in [f"{left}-{right}" for left, right in MODE_PAIRS]:
        pair_rows = [row for row in rows if row["mode_pair"] == pair]
        summary["overall"][pair] = _delta_metric_summary(pair_rows)
    by_method = summary["by_method"]
    assert isinstance(by_method, dict)
    for method in GRID_METHODS:
        by_method[method] = {}
        method_rows = [row for row in rows if row["method"] == method]
        for pair in [f"{left}-{right}" for left, right in MODE_PAIRS]:
            by_method[method][pair] = _delta_metric_summary(
                [row for row in method_rows if row["mode_pair"] == pair]
            )
    return summary


def _delta_metric_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "count": len(rows),
        "faithfulness_delta": _distribution_summary(_object_values(rows, "faithfulness_delta")),
        "suppression_delta": _distribution_summary(_object_values(rows, "suppression_delta")),
        "nodes_delta": _distribution_summary(_object_values(rows, "nodes_delta")),
        "edges_delta": _distribution_summary(_object_values(rows, "edges_delta")),
    }


def _plot_acceptance_rate(path: Path, stats: Mapping[str, object]) -> Path:
    plt = configure_matplotlib()
    by_mode = stats["by_mode"]
    assert isinstance(by_mode, dict)
    values = [float(by_mode[mode]["acceptance_rate"]) for mode in GRID_NEG_MODES]
    fig, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    axis.bar(
        [MODE_LABELS[mode] for mode in GRID_NEG_MODES],
        values,
        width=BAR_WIDTH,
        color=[MODE_COLORS[mode] for mode in GRID_NEG_MODES],
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Acceptance rate")
    axis.set_title("Acceptance Rate by Negctx Mode")
    annotate_bars(axis)
    round_bars(axis)
    return save_figure(fig, path)


def _plot_acceptance_rate_by_method(path: Path, stats: Mapping[str, object]) -> Path:
    by_method_mode = stats["by_method_mode"]
    assert isinstance(by_method_mode, dict)
    values = {
        method: [float(by_method_mode[method][mode]["acceptance_rate"]) for mode in GRID_NEG_MODES]
        for method in GRID_METHODS
    }
    return _grouped_bar_plot(path, values, "Acceptance Rate by Method and Mode", "Acceptance rate", ylim=(0.0, 1.0))


def _plot_box_by_mode(
    path: Path,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    *,
    accepted_only: bool,
) -> Path:
    plt = configure_matplotlib()
    data = [_metric_values(_filter_metric_rows(rows, mode=mode, accepted_only=accepted_only), metric) for mode in GRID_NEG_MODES]
    fig, axis = plt.subplots(figsize=FIGSIZE_SINGLE)
    styled_boxplot(axis, data, [MODE_LABELS[mode] for mode in GRID_NEG_MODES], [MODE_COLORS[mode] for mode in GRID_NEG_MODES])
    axis.set_title(title)
    axis.set_ylabel(METRIC_LABELS.get(metric, metric))
    return save_figure(fig, path)


def _plot_box_by_method_mode(
    path: Path,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    *,
    accepted_only: bool,
    box_edge: str = "ink",
) -> Path:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, len(GRID_METHODS), figsize=panel_figsize(1, len(GRID_METHODS)), sharey=True)
    for axis, method in zip(axes, GRID_METHODS):
        data = [
            _metric_values(
                _filter_metric_rows(rows, method=method, mode=mode, accepted_only=accepted_only),
                metric,
            )
            for mode in GRID_NEG_MODES
        ]
        styled_boxplot(
            axis,
            data,
            [MODE_LABELS[mode] for mode in GRID_NEG_MODES],
            [MODE_COLORS[mode] for mode in GRID_NEG_MODES],
            edge=box_edge,
        )
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Negative Context Mode")
    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric))
    style_suptitle(fig, title)
    return save_figure(fig, path)


def _plot_paired_delta_summary(path: Path, paired_summary: Mapping[str, object], metric: str, title: str) -> Path:
    by_method = paired_summary["by_method"]
    assert isinstance(by_method, dict)
    pairs = [f"{left}-{right}" for left, right in MODE_PAIRS]
    values = {
        method: [
            float(by_method[method][pair][metric]["mean"])
            if int(by_method[method][pair][metric]["count"]) > 0
            else 0.0
            for pair in pairs
        ]
        for method in GRID_METHODS
    }
    return _grouped_bar_plot(path, values, title, "Mean delta", x_labels=pairs, zero_line=True)


def _grouped_bar_plot(
    path: Path,
    values_by_method: Mapping[str, Sequence[float]],
    title: str,
    ylabel: str,
    *,
    x_labels: Sequence[str] | None = None,
    ylim: tuple[float, float] | None = None,
    zero_line: bool = False,
) -> Path:
    plt = configure_matplotlib()
    labels = list(x_labels or [MODE_LABELS[mode] for mode in GRID_NEG_MODES])
    x_positions = list(range(len(labels)))
    width, offsets = grouped_bar_geometry(len(GRID_METHODS))
    fig, axis = plt.subplots(figsize=FIGSIZE_WIDE)
    for method, offset in zip(GRID_METHODS, offsets):
        values = values_by_method[method]
        axis.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    if zero_line:
        axis.axhline(0.0, color=INK_MUTED, linewidth=1.0)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)
    styled_legend(axis, loc="best")
    round_bars(axis)
    return save_figure(fig, path)


def _filter_metric_rows(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    accepted_only: bool,
    method: str | None = None,
) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row["neg_mode"] == mode and (method is None or row["method"] == method)]
    if accepted_only:
        filtered = [row for row in filtered if _accepted_finite(row)]
    return filtered


def _metric_values(rows: Iterable[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _derived_metric(row, metric)
        if math.isfinite(value):
            values.append(value)
    return values


def _object_values(rows: Iterable[Mapping[str, object]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if _is_finite(value):
            values.append(float(value))
    return values


def _derived_metric(row: Mapping[str, Any], metric: str) -> float:
    if _is_finite(row.get(metric)):
        return _float(row.get(metric))
    return math.nan


def _accepted_finite(row: Mapping[str, Any]) -> bool:
    return (
        row.get("status") == "accepted"
        and not row.get("error")
        and _is_finite(row.get("counterfactual_faithfulness"))
        and _is_finite(row.get("posctx_suppression_score"))
    )


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


def _nested(summary: Mapping[str, object], metric: str, stat: str) -> object:
    metric_summary = summary[metric]
    assert isinstance(metric_summary, dict)
    return metric_summary[stat]
