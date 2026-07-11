"""Plot top-N eval score distributions per gradient discovery method."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json

from .coact_overlap import SUITE_NAME
from .gradient_method_neg_mode_grid_runner import GRID_METHODS
from .gradient_method_eval_distribution import (
    METRICS,
    _distribution_stats_from_values,
    _float,
    _resolve_grid_results_path,
    _values_by_method,
    _write_distribution_plot,
    load_gradient_method_eval_rows,
)

RANK_METRICS = (
    "counterfactual_faithfulness",
    "posctx_suppression_score",
)
TOP_N_TABLE_FIELDS = [
    "method",
    "metric",
    "top_n",
    "available_count",
    "count",
    "mean",
    "std",
    "median",
    "q1",
    "q3",
    "min",
    "max",
]


@dataclass(frozen=True)
class GradientMethodEvalTopNDistributionResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_gradient_method_eval_top_n_distribution(
    run_root: str | Path,
    *,
    results_path: str | Path | None = None,
    output_root: str | Path | None = None,
    top_n: int | None = None,
) -> GradientMethodEvalTopNDistributionResult:
    """Plot eval KDE curves for the top-N circuits per discovery method."""

    root = resolve_run_root(run_root)
    table_path = _resolve_grid_results_path(root, results_path)
    rows = load_gradient_method_eval_rows(table_path)
    selected_rows, available_counts, resolved_top_n = select_top_n_per_method(rows, top_n=top_n)
    stats = compute_gradient_method_eval_top_n_distribution_stats(
        selected_rows,
        top_n=resolved_top_n,
        available_counts=available_counts,
    )
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "gradient-method-eval-top-n-distribution.png"
    output_table_path = output_dirs["tables"] / "gradient-method-eval-top-n-distribution.csv"
    summary_path = output_dirs["summaries"] / "gradient-method-eval-top-n-distribution.json"

    title = f"Gradient Method Eval Distributions (Top {resolved_top_n} per Method)"
    _write_distribution_plot(figure_path, stats, title=title)
    _write_table(output_table_path, stats)
    summary = _build_summary(table_path, stats)
    write_json(summary_path, summary)
    return GradientMethodEvalTopNDistributionResult(figure_path, summary_path, output_table_path, summary)


def select_top_n_per_method(
    rows: list[dict[str, Any]],
    *,
    top_n: int | None = None,
    rank_metrics: Sequence[str] = RANK_METRICS,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Keep the top-N accepted circuits per method ranked by eval quality."""

    available_counts = {
        method: len([row for row in rows if row["method"] == method])
        for method in GRID_METHODS
    }
    if not all(available_counts.values()):
        raise ValueError("each method must have at least one accepted finite eval row")

    resolved_top_n = min(available_counts.values()) if top_n is None else int(top_n)
    if resolved_top_n <= 0:
        raise ValueError("top_n must be positive")
    if resolved_top_n > min(available_counts.values()):
        raise ValueError(
            f"top_n={resolved_top_n} exceeds the smallest method count "
            f"({min(available_counts.values())})"
        )

    selected: list[dict[str, Any]] = []
    for method in GRID_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        ranked = sorted(
            method_rows,
            key=lambda row: tuple(-_float(row[metric]) for metric in rank_metrics),
        )
        selected.extend(ranked[:resolved_top_n])
    return selected, available_counts, resolved_top_n


def compute_gradient_method_eval_top_n_distribution_stats(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    available_counts: Mapping[str, int],
    rank_metrics: Sequence[str] = RANK_METRICS,
) -> dict[str, object]:
    """Compute distribution stats for top-N circuits per method."""

    by_method = _values_by_method(rows)
    return _distribution_stats_from_values(
        by_method,
        row_count=len(rows),
        top_n=top_n,
        rank_metrics=rank_metrics,
        available_counts=available_counts,
    )


def _write_table(path: Path, stats: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    by_method = stats["by_method"]
    assert isinstance(by_method, dict)
    top_n = int(stats["top_n"])
    available_counts = stats["available_counts"]
    assert isinstance(available_counts, dict)
    for method in GRID_METHODS:
        method_stats = by_method[method]
        assert isinstance(method_stats, dict)
        for metric in METRICS:
            summary = method_stats[metric]
            assert isinstance(summary, dict)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "top_n": top_n,
                    "available_count": available_counts[method],
                    **summary,
                }
            )
    write_csv(path, rows, TOP_N_TABLE_FIELDS)


def _build_summary(table_path: Path, stats: dict[str, object]) -> dict[str, object]:
    return {
        "results_path": str(table_path),
        "row_count": stats["row_count"],
        "top_n": stats["top_n"],
        "rank_metrics": stats["rank_metrics"],
        "available_counts": stats["available_counts"],
        "methods": stats["methods"],
        "metrics": stats["metrics"],
        "by_method": stats["by_method"],
        "x_ranges": stats["x_ranges"],
    }
