"""Median faithfulness summary bars for the gradient method x negmode grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import (
    FIGSIZE_WIDE,
    INK,
    METHOD_COLORS,
    SURFACE,
    configure_matplotlib,
    grouped_bar_geometry,
    round_bars,
    save_figure,
    styled_legend,
    tint,
)

from .gradient_method_eval_distribution import METHOD_LABELS, _float, _resolve_grid_results_path
from .gradient_method_neg_mode_grid_runner import GRID_METHODS, GRID_NEG_MODES, SUITE_NAME as GRID_SUITE_NAME
from .gradient_neg_mode_comparison import MODE_LABELS, _accepted_finite, load_gradient_neg_mode_rows

METRIC = "counterfactual_faithfulness"
TABLE_FIELDS = [
    "method",
    "neg_mode",
    "accepted_count",
    "median_counterfactual_faithfulness",
    "top_n_median_counterfactual_faithfulness",
]


@dataclass(frozen=True)
class GradientGridMedianResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_gradient_grid_median_faithfulness(
    run_root: str | Path,
    *,
    results_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> GradientGridMedianResult:
    """Plot median counterfactual faithfulness by method and negctx mode.

    Bars are two-tone: the solid segment reaches the group's median over all
    accepted circuits; the light segment extends to the median of the group's
    top-n circuits by faithfulness, with n set to the smallest accepted count
    across the method x mode groups (equal-population comparison).
    """

    root = resolve_run_root(run_root)
    table_path = _resolve_grid_results_path(root, results_path)
    rows = load_gradient_neg_mode_rows(table_path)

    values_by_group: dict[str, dict[str, list[float]]] = {}
    for method in GRID_METHODS:
        values_by_group[method] = {}
        for mode in GRID_NEG_MODES:
            values_by_group[method][mode] = sorted(
                _float(row[METRIC])
                for row in rows
                if row["method"] == method and row["neg_mode"] == mode and _accepted_finite(row)
            )
    group_sizes = [len(values) for by_mode in values_by_group.values() for values in by_mode.values()]
    top_n = min((size for size in group_sizes if size > 0), default=0)

    medians: dict[str, dict[str, float]] = {}
    top_medians: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    table_rows: list[dict[str, object]] = []
    for method in GRID_METHODS:
        medians[method] = {}
        top_medians[method] = {}
        counts[method] = {}
        for mode in GRID_NEG_MODES:
            values = values_by_group[method][mode]
            group_median = float(median(values)) if values else 0.0
            top_slice = values[-top_n:] if top_n > 0 and values else []
            top_median = float(median(top_slice)) if top_slice else 0.0
            medians[method][mode] = group_median
            top_medians[method][mode] = top_median
            counts[method][mode] = len(values)
            table_rows.append(
                {
                    "method": method,
                    "neg_mode": mode,
                    "accepted_count": len(values),
                    "median_counterfactual_faithfulness": group_median,
                    "top_n_median_counterfactual_faithfulness": top_median,
                }
            )

    output_dirs = analysis_output_dirs(root, GRID_SUITE_NAME, output_root=output_root)
    figure_path = _write_plot(output_dirs["figures"] / "median-cf-faithfulness.png", medians, top_medians, top_n)
    table_path_out = write_csv(output_dirs["tables"] / "median-cf-faithfulness.csv", table_rows, TABLE_FIELDS)
    summary = {
        "results_path": str(table_path),
        "row_count": len(rows),
        "metric": METRIC,
        "top_n": top_n,
        "medians": medians,
        "top_n_medians": top_medians,
        "accepted_counts": counts,
        "figure_path": str(figure_path),
        "table_path": str(table_path_out),
    }
    summary_path = write_json(output_dirs["summaries"] / "median-cf-faithfulness.json", summary)
    return GradientGridMedianResult(
        figure_path=figure_path,
        summary_path=summary_path,
        table_path=table_path_out,
        summary=summary,
    )


def _write_plot(
    path: Path,
    medians: dict[str, dict[str, float]],
    top_medians: dict[str, dict[str, float]],
    top_n: int,
) -> Path:
    plt = configure_matplotlib()
    fig, axis = plt.subplots(figsize=FIGSIZE_WIDE)
    x_positions = list(range(len(GRID_NEG_MODES)))
    width, offsets = grouped_bar_geometry(len(GRID_METHODS))

    max_top = 0.0
    for method, offset in zip(GRID_METHODS, offsets):
        color = METHOD_COLORS[method]
        for x, mode in zip(x_positions, GRID_NEG_MODES):
            group_median = medians[method][mode]
            top_median = top_medians[method][mode]
            max_top = max(max_top, top_median, group_median)
            axis.bar(
                x + offset,
                group_median,
                width=width,
                color=color,
                label=METHOD_LABELS[method] if x == 0 else None,
            )
            cap = max(top_median - group_median, 0.0)
            has_cap = cap > 0.005
            if has_cap:
                axis.bar(x + offset, cap, width=width, bottom=group_median, color=tint(color))
                axis.annotate(
                    f"{group_median:.2f}",
                    (x + offset, group_median),
                    xytext=(0, -4),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=9,
                    fontweight="medium",
                    color=SURFACE,
                )
            top_of_stack = group_median + cap if has_cap else group_median
            axis.annotate(
                f"{top_median:.2f}" if has_cap else f"{group_median:.2f}",
                (x + offset, top_of_stack),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="medium",
                color=INK,
            )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([MODE_LABELS[mode] for mode in GRID_NEG_MODES])
    axis.set_xlabel("Negative Context Mode")
    axis.set_ylim(0.0, max(1.0, max_top + 0.1))
    axis.set_ylabel("Median counterfactual faithfulness")
    axis.set_title("Median Counterfactual Faithfulness by Method and Negative Context Mode")
    styled_legend(axis, loc="upper left")
    round_bars(axis)
    return save_figure(fig, path)
