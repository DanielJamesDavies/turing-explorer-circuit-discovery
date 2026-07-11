"""Plot median counterfactual faithfulness against circuit size per method."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import (
    INK_MUTED,
    METHOD_COLORS,
    configure_matplotlib,
    panel_figsize,
    save_figure,
    style_suptitle,
    styled_legend,
)

from .gradient_method_eval_distribution import METHOD_LABELS, _float, _is_finite
from .gradient_method_neg_mode_grid_runner import GRID_METHODS
from .gradient_size_sweep_runner import SUITE_NAME

TABLE_FIELDS = [
    "method",
    "nodes",
    "count",
    "mean_counterfactual_faithfulness",
    "median_counterfactual_faithfulness",
    "mean_posctx_suppression_score",
    "median_posctx_suppression_score",
    "mean_ablation_faithfulness",
    "median_ablation_faithfulness",
]
# Grid x-positions covered by fewer seed curves than this are dropped: the
# extremes are composition-biased (only seeds with many upstream sites reach
# the largest node counts, only sparse seeds the smallest).
MIN_SEEDS = 25
GRID_POINTS = 48


@dataclass(frozen=True)
class GradientSizeCurveResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]
    ablation_figure_path: Path | None = None


def plot_gradient_size_curve(
    run_root: str | Path,
    *,
    results_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> GradientSizeCurveResult:
    root = resolve_run_root(run_root)
    table_path = _resolve_sweep_results_path(root, results_path)
    rows = _load_sweep_rows(table_path)
    has_ablation = _is_finite(rows[0].get("ablation_faithfulness"))
    points = _curve_points(rows, with_ablation=has_ablation)

    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "gradient-size-curve.png"
    output_table_path = output_dirs["tables"] / "gradient-size-curve.csv"
    summary_path = output_dirs["summaries"] / "gradient-size-curve.json"

    _write_curve_plot(
        figure_path,
        points,
        metric="counterfactual_faithfulness",
        metric_label="Faithfulness",
        ylabel="Counterfactual faithfulness",
    )
    ablation_figure_path = None
    if has_ablation:
        ablation_figure_path = output_dirs["figures"] / "gradient-size-curve-ablation.png"
        _write_curve_plot(
            ablation_figure_path,
            points,
            metric="ablation_faithfulness",
            metric_label="Ablation Faithfulness",
            ylabel="Ablation faithfulness (circuit-only, winsorised to [-1, 2])",
        )
    write_csv(output_table_path, points, TABLE_FIELDS)
    summary = {
        "results_path": str(table_path),
        "row_count": len(rows),
        "methods": list(GRID_METHODS),
        "has_ablation_faithfulness": has_ablation,
        "points": points,
    }
    write_json(summary_path, summary)
    return GradientSizeCurveResult(
        figure_path, summary_path, output_table_path, summary, ablation_figure_path
    )


def _resolve_sweep_results_path(root: Path, results_path: str | Path | None) -> Path:
    if results_path is not None:
        path = Path(results_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"size sweep results not found: {path}")
        return path
    candidates = (
        root / "analysis" / SUITE_NAME / "tables" / "gradient-size-sweep.csv",
        root / "analysis" / "5" / SUITE_NAME / "tables" / "gradient-size-sweep.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"size sweep results not found. Expected one of: {searched}. "
        "Run gradient-size-sweep-run first or pass --results-path."
    )


def _load_sweep_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable = [
        row
        for row in rows
        if row.get("status") == "ok"
        and _is_finite(row.get("counterfactual_faithfulness"))
        and _is_finite(row.get("nodes_used"))
    ]
    if not usable:
        raise ValueError(f"size sweep results have no usable rows: {path}")
    return usable


def _curve_points(rows: list[dict[str, Any]], *, with_ablation: bool = False) -> list[dict[str, object]]:
    """Aggregate per-seed size-faithfulness curves onto a common node grid.

    Each seed's truncation trajectory (its 8 evaluated sizes) is linearly
    interpolated in log-node space onto a shared log-spaced grid, and the
    curves are aggregated across seeds pointwise --- the SFC convention.
    Interpolation is only within a seed's observed size range, so no seed is
    extrapolated, and grid points covered by too few seeds are dropped.
    """

    sizes = [_float(row["nodes_used"]) for row in rows if _float(row["nodes_used"]) > 0]
    if not sizes:
        raise ValueError("size sweep rows contain no positive node counts")
    x_min, x_max = min(sizes), max(sizes)
    grid = [
        x_min * (x_max / x_min) ** (index / (GRID_POINTS - 1))
        for index in range(GRID_POINTS)
    ]

    points: list[dict[str, object]] = []
    for method in GRID_METHODS:
        seed_curves: dict[tuple, list[tuple[float, float, float]]] = {}
        for row in rows:
            if row["method"] != method or _float(row["nodes_used"]) <= 0:
                continue
            key = (row["comp_idx"], row["latent_idx"])
            # Circuit-only execution occasionally detonates the seed far
            # off-distribution (values in the hundreds); winsorise so a few
            # exploding seeds cannot dominate the mean curve.
            abl = min(max(_float(row["ablation_faithfulness"]), -1.0), 2.0) if with_ablation else 0.0
            seed_curves.setdefault(key, []).append(
                (
                    _float(row["nodes_used"]),
                    _float(row["counterfactual_faithfulness"]),
                    _float(row["posctx_suppression_score"]),
                    abl,
                )
            )
        curves = []
        for samples in seed_curves.values():
            samples.sort(key=lambda sample: sample[0])
            deduped = [
                sample
                for index, sample in enumerate(samples)
                if index == 0 or sample[0] > samples[index - 1][0]
            ]
            if len(deduped) >= 2:
                curves.append(deduped)

        for x in grid:
            cf_values = []
            sup_values = []
            abl_values = []
            for samples in curves:
                interpolated = _log_interp(samples, x)
                if interpolated is not None:
                    cf_values.append(interpolated[0])
                    sup_values.append(interpolated[1])
                    abl_values.append(interpolated[2])
            if len(cf_values) < MIN_SEEDS:
                continue
            points.append(
                {
                    "method": method,
                    "nodes": x,
                    "count": len(cf_values),
                    "mean_counterfactual_faithfulness": mean(cf_values),
                    "median_counterfactual_faithfulness": median(cf_values),
                    "mean_posctx_suppression_score": mean(sup_values),
                    "median_posctx_suppression_score": median(sup_values),
                    "mean_ablation_faithfulness": mean(abl_values) if with_ablation else "",
                    "median_ablation_faithfulness": median(abl_values) if with_ablation else "",
                }
            )
    return points


def _log_interp(
    samples: list[tuple[float, ...]], x: float
) -> tuple[float, ...] | None:
    """Linear interpolation in log-x space over every metric column;
    None outside the sample range."""

    if x < samples[0][0] or x > samples[-1][0]:
        return None
    for sample0, sample1 in zip(samples, samples[1:]):
        x0, x1 = sample0[0], sample1[0]
        if x0 <= x <= x1:
            if x1 == x0:
                return sample0[1:]
            weight = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return tuple(
                value0 + weight * (value1 - value0)
                for value0, value1 in zip(sample0[1:], sample1[1:])
            )
    return samples[-1][1:]


def _write_curve_plot(
    path: Path,
    points: list[dict[str, object]],
    *,
    metric: str,
    metric_label: str,
    ylabel: str,
) -> None:
    plt = configure_matplotlib()
    from matplotlib.ticker import LogLocator, ScalarFormatter

    fig, axes = plt.subplots(1, 2, figsize=panel_figsize(1, 2), sharey=True)
    panels = (
        (f"mean_{metric}", f"Mean {metric_label}"),
        (f"median_{metric}", f"Median {metric_label}"),
    )
    for axis, (field, title) in zip(axes, panels):
        axis.axhline(1.0, color=INK_MUTED, linestyle=":", linewidth=1.0, alpha=0.7)
        for method in GRID_METHODS:
            method_points = [point for point in points if point["method"] == method]
            if not method_points:
                continue
            method_points.sort(key=lambda point: float(point["nodes"]))
            xs = [float(point["nodes"]) for point in method_points]
            ys = [float(point[field]) for point in method_points]
            axis.plot(
                xs,
                ys,
                linewidth=2.0,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )

        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_locator(LogLocator(base=2))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.minorticks_off()
        axis.set_xlabel("Circuit nodes")
        axis.set_title(title)

    axes[0].set_ylabel(ylabel)
    styled_legend(axes[0], loc="lower right")
    style_suptitle(fig, f"{metric_label} by Circuit Size")
    save_figure(fig, path)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Plot the faithfulness-by-size curve.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    result = plot_gradient_size_curve(
        args.run_root, results_path=args.results_path, output_root=args.output_root
    )
    print(result.figure_path)
    print(result.table_path)
    print(result.summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
