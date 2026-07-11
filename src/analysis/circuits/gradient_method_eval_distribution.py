"""Plot eval score distributions by gradient discovery method."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev, quantiles
from typing import Any, Mapping, Sequence

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import METHOD_COLORS, configure_matplotlib, panel_figsize, save_figure, style_suptitle, styled_legend

from .coact_overlap import SUITE_NAME
from .gradient_method_neg_mode_grid_runner import GRID_METHODS, SUITE_NAME as GRID_SUITE_NAME

METRICS = (
    "counterfactual_faithfulness",
    "posctx_suppression_score",
)
METRIC_LABELS = {
    "counterfactual_faithfulness": "Counterfactual Faithfulness",
    "posctx_suppression_score": "Posctx Suppression Score",
}
METHOD_LABELS = {
    "counterfactual_gradient": "Counterfactual",
    "ablation_gradient": "Ablation",
    "hybrid_gradient": "Hybrid",
}
TABLE_FIELDS = [
    "population",
    "method",
    "metric",
    "count",
    "mean",
    "std",
    "median",
    "q1",
    "q3",
    "min",
    "max",
]
DEFAULT_GRID_CSV_CANDIDATES = (
    Path("analysis") / "5" / GRID_SUITE_NAME / "tables" / "gradient-method-neg-mode-grid.csv",
    Path("analysis") / GRID_SUITE_NAME / "tables" / "gradient-method-neg-mode-grid.csv",
)


@dataclass(frozen=True)
class GradientMethodEvalDistributionResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_gradient_method_eval_distribution(
    run_root: str | Path,
    *,
    results_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> GradientMethodEvalDistributionResult:
    """Plot eval score KDE curves grouped by discovery method."""

    root = resolve_run_root(run_root)
    table_path = _resolve_grid_results_path(root, results_path)
    rows = load_gradient_method_eval_rows(table_path)
    best_rows = select_best_per_seed(rows)
    stats = compute_gradient_method_eval_distribution_stats(rows)
    # The best-per-seed values are a subset of the full population, so the
    # full x-ranges cover them; sharing ranges keeps each row comparable.
    best_stats = _distribution_stats_from_values(
        _values_by_method(best_rows),
        row_count=len(best_rows),
        x_ranges_override=stats["x_ranges"],
    )
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "gradient-method-eval-distribution.png"
    output_table_path = output_dirs["tables"] / "gradient-method-eval-distribution.csv"
    summary_path = output_dirs["summaries"] / "gradient-method-eval-distribution.json"

    _write_distribution_grid(
        figure_path, stats, best_stats, title="Gradient Method Eval Distributions"
    )
    _write_table(output_table_path, stats, best_stats)
    summary = _build_summary(table_path, stats, best_stats)
    write_json(summary_path, summary)
    return GradientMethodEvalDistributionResult(figure_path, summary_path, output_table_path, summary)


def select_best_per_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep each method's best accepted circuit per seed.

    Every method ran the same 3 negative-mode attempts per seed, so taking
    the top circuit per (method, seed) --- ranked by counterfactual
    faithfulness, ties broken by suppression --- is an equal-budget
    comparison that does not reward a larger accepted pool.
    """

    best: dict[tuple[str, str, str], tuple[tuple[float, float], dict[str, Any]]] = {}
    for row in rows:
        key = (row["method"], row["comp_idx"], row["latent_idx"])
        score = (
            _float(row["counterfactual_faithfulness"]),
            _float(row["posctx_suppression_score"]),
        )
        current = best.get(key)
        if current is None or score > current[0]:
            best[key] = (score, row)
    return [row for _, row in best.values()]


def _resolve_grid_results_path(root: Path, results_path: str | Path | None) -> Path:
    if results_path is not None:
        path = Path(results_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"gradient method grid results not found: {path}")
        return path

    for relative in DEFAULT_GRID_CSV_CANDIDATES:
        candidate = root / relative
        if candidate.exists():
            return candidate

    searched = ", ".join(str(root / relative) for relative in DEFAULT_GRID_CSV_CANDIDATES)
    raise FileNotFoundError(
        f"gradient method grid results not found. Expected one of: {searched}. "
        "Run the gradient-method grid first or pass --results-path."
    )


def load_gradient_method_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load accepted grid rows with finite eval scores."""

    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"gradient method grid results not found: {table_path}")
    with table_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"gradient method grid results table is empty: {table_path}")

    required = {"method", "neg_mode", "status", *METRICS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"gradient method grid results missing columns: {sorted(missing)}")

    accepted = [
        row
        for row in rows
        if row.get("status") == "accepted"
        and not row.get("error")
        and _is_finite(row.get("counterfactual_faithfulness"))
        and _is_finite(row.get("posctx_suppression_score"))
    ]
    if not accepted:
        raise ValueError(f"gradient method grid results have no accepted finite eval rows: {table_path}")
    return accepted


def compute_gradient_method_eval_distribution_stats(rows: list[dict[str, Any]]) -> dict[str, object]:
    """Compute per-method eval distribution stats and KDE curves."""

    by_method = _values_by_method(rows)
    return _distribution_stats_from_values(by_method, row_count=len(rows))


def _values_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    by_method: dict[str, dict[str, list[float]]] = {}
    for method in GRID_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        by_method[method] = {
            metric: [_float(row[metric]) for row in method_rows if _is_finite(row.get(metric))]
            for metric in METRICS
        }
    return by_method


def _distribution_stats_from_values(
    by_method: dict[str, dict[str, list[float]]],
    *,
    row_count: int,
    top_n: int | None = None,
    rank_metrics: Sequence[str] | None = None,
    available_counts: Mapping[str, int] | None = None,
    x_ranges_override: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    method_stats: dict[str, dict[str, object]] = {}
    for method in GRID_METHODS:
        method_stats[method] = {}
        for metric in METRICS:
            values = by_method[method][metric]
            method_stats[method][metric] = _distribution_summary(values)

    if x_ranges_override is not None:
        x_ranges = {metric: tuple(x_ranges_override[metric]) for metric in METRICS}
    else:
        x_ranges = {
            metric: _metric_x_range(
                [value for method in GRID_METHODS for value in by_method[method][metric]],
                method_stats=method_stats,
                metric=metric,
            )
            for metric in METRICS
        }

    curves: dict[str, dict[str, object]] = {}
    for metric in METRICS:
        x_grid = _linspace(x_ranges[metric][0], x_ranges[metric][1], 200)
        curves[metric] = {"x": x_grid, "by_method": {}}
        for method in GRID_METHODS:
            values = by_method[method][metric]
            curves[metric]["by_method"][method] = gaussian_kde(values, x_grid)

    stats: dict[str, object] = {
        "methods": list(GRID_METHODS),
        "metrics": list(METRICS),
        "row_count": row_count,
        "by_method": method_stats,
        "x_ranges": x_ranges,
        "curves": curves,
    }
    if top_n is not None:
        stats["top_n"] = int(top_n)
    if rank_metrics is not None:
        stats["rank_metrics"] = list(rank_metrics)
    if available_counts is not None:
        stats["available_counts"] = dict(available_counts)
    return stats


def gaussian_kde(values: Sequence[float], x_grid: Sequence[float], *, bandwidth: float | None = None) -> list[float]:
    """Evaluate a Gaussian kernel density estimate on a grid."""

    samples = [float(value) for value in values]
    if not samples or not x_grid:
        return [0.0] * len(x_grid)

    n = len(samples)
    if bandwidth is None:
        std = pstdev(samples) if n > 1 else 1.0
        if std <= 0.0:
            std = 1.0
        bandwidth = 1.06 * std * (n ** (-0.2))

    scale = 1.0 / (n * bandwidth * math.sqrt(2.0 * math.pi))
    densities: list[float] = []
    for x in x_grid:
        total = sum(math.exp(-0.5 * ((x - sample) / bandwidth) ** 2) for sample in samples)
        densities.append(scale * total)
    return densities


def _distribution_summary(values: Sequence[float]) -> dict[str, object]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    if len(samples) > 1:
        q1, _, q3 = quantiles(samples, n=4, method="inclusive")
    else:
        q1 = q3 = samples[0]
    return {
        "count": len(samples),
        "mean": float(mean(samples)),
        "std": float(pstdev(samples)) if len(samples) > 1 else 0.0,
        "median": float(median(samples)),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(min(samples)),
        "max": float(max(samples)),
    }


def _metric_x_range(
    values: Sequence[float],
    *,
    method_stats: Mapping[str, Mapping[str, Mapping[str, object]]],
    metric: str,
) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)

    global_min = min(values)
    global_max = max(values)
    max_std = max(float(method_stats[method][metric]["std"]) for method in GRID_METHODS)
    padding = max(max_std, (global_max - global_min) * 0.05, 0.05)
    return (global_min - padding, global_max + padding)


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    step = (end - start) / (count - 1)
    return [start + step * index for index in range(count)]


def _draw_metric_panel(axis: Any, stats: dict[str, object], metric: str, *, legend: bool) -> None:
    curves = stats["curves"]
    by_method = stats["by_method"]
    assert isinstance(curves, dict)
    assert isinstance(by_method, dict)
    metric_curves = curves[metric]
    assert isinstance(metric_curves, dict)
    x_grid = metric_curves["x"]
    assert isinstance(x_grid, list)
    method_curves = metric_curves["by_method"]
    assert isinstance(method_curves, dict)

    for method in GRID_METHODS:
        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        y_grid = method_curves[method]
        assert isinstance(y_grid, list)
        method_summary = by_method[method][metric]
        assert isinstance(method_summary, dict)
        method_median = float(method_summary["median"])

        axis.plot(x_grid, y_grid, linewidth=2.0, color=color, label=label)
        axis.axvspan(
            float(method_summary["q1"]),
            float(method_summary["q3"]),
            color=color,
            alpha=0.12,
        )
        axis.axvline(method_median, color=color, linestyle="--", linewidth=1.2, alpha=0.85)

    if legend:
        styled_legend(axis, loc="best")


def _write_distribution_plot(path: Path, stats: dict[str, object], *, title: str) -> None:
    plt = configure_matplotlib()
    metrics = stats["metrics"]
    assert isinstance(metrics, list)

    fig, axes = plt.subplots(1, 2, figsize=panel_figsize(1, 2))
    for axis, metric in zip(axes, metrics):
        _draw_metric_panel(axis, stats, metric, legend=True)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Eval score")
        axis.set_ylabel("Density")

    style_suptitle(fig, title)
    save_figure(fig, path)


def _write_distribution_grid(
    path: Path,
    stats: dict[str, object],
    best_stats: dict[str, object],
    *,
    title: str,
) -> None:
    """2x2 grid: rows = metrics, columns = all accepted vs best per seed."""

    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=panel_figsize(2, 2), sharex="row", sharey="row")
    columns = ((stats, ""), (best_stats, " (Best per Seed)"))
    for row_index, metric in enumerate(METRICS):
        for col_index, (column_stats, suffix) in enumerate(columns):
            axis = axes[row_index][col_index]
            _draw_metric_panel(
                axis,
                column_stats,
                metric,
                legend=(row_index == 0 and col_index == 0),
            )
            axis.set_title(METRIC_LABELS[metric] + suffix)
            if row_index == len(METRICS) - 1:
                axis.set_xlabel("Eval score")
            if col_index == 0:
                axis.set_ylabel("Density")

    style_suptitle(fig, title)
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object], best_stats: dict[str, object] | None = None) -> None:
    rows: list[dict[str, object]] = []
    populations = [("all_accepted", stats)]
    if best_stats is not None:
        populations.append(("best_per_seed", best_stats))
    for population, population_stats in populations:
        by_method = population_stats["by_method"]
        assert isinstance(by_method, dict)
        for method in GRID_METHODS:
            method_stats = by_method[method]
            assert isinstance(method_stats, dict)
            for metric in METRICS:
                summary = method_stats[metric]
                assert isinstance(summary, dict)
                rows.append(
                    {
                        "population": population,
                        "method": method,
                        "metric": metric,
                        **summary,
                    }
                )
    write_csv(path, rows, TABLE_FIELDS)


def _build_summary(
    table_path: Path, stats: dict[str, object], best_stats: dict[str, object] | None = None
) -> dict[str, object]:
    summary: dict[str, object] = {
        "results_path": str(table_path),
        "row_count": stats["row_count"],
        "methods": stats["methods"],
        "metrics": stats["metrics"],
        "by_method": stats["by_method"],
        "x_ranges": stats["x_ranges"],
    }
    if best_stats is not None:
        summary["best_per_seed"] = {
            "row_count": best_stats["row_count"],
            "by_method": best_stats["by_method"],
        }
    return summary


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
