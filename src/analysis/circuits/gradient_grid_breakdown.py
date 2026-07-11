"""Faithfulness across seed layers and component kinds for the gradient grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import (
    BLUE,
    configure_matplotlib,
    ordinal_blues,
    panel_figsize,
    save_figure,
    styled_boxplot,
)

from .gradient_method_eval_distribution import _float, _resolve_grid_results_path
from .gradient_method_neg_mode_grid_runner import SUITE_NAME as GRID_SUITE_NAME
from .gradient_neg_mode_comparison import _accepted_finite, load_gradient_neg_mode_rows

METRIC = "counterfactual_faithfulness"
KINDS = ("attn", "mlp", "resid")
N_LAYERS = 12
TABLE_FIELDS = ["scope", "group", "count", "mean", "median", "min", "max"]


@dataclass(frozen=True)
class GradientGridBreakdownResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_gradient_grid_seed_breakdown(
    run_root: str | Path,
    *,
    results_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> GradientGridBreakdownResult:
    """Plot faithfulness distributions by seed layer and seed component kind."""

    root = resolve_run_root(run_root)
    table_path = _resolve_grid_results_path(root, results_path)
    rows = [row for row in load_gradient_neg_mode_rows(table_path) if _accepted_finite(row)]

    by_layer: dict[int, list[float]] = {layer: [] for layer in range(N_LAYERS)}
    by_kind: dict[str, list[float]] = {kind: [] for kind in KINDS}
    for row in rows:
        value = _float(row[METRIC])
        layer = row.get("layer")
        kind = row.get("kind")
        if layer is not None and str(layer).lstrip("-").isdigit() and int(layer) in by_layer:
            by_layer[int(layer)].append(value)
        if kind in by_kind:
            by_kind[kind].append(value)

    output_dirs = analysis_output_dirs(root, GRID_SUITE_NAME, output_root=output_root)
    figure_path = _write_plot(output_dirs["figures"] / "gradient-grid-seed-breakdown.png", by_layer, by_kind)
    table_rows = [
        {"scope": "layer", "group": str(layer), **_group_summary(values)}
        for layer, values in sorted(by_layer.items())
    ] + [{"scope": "kind", "group": kind, **_group_summary(by_kind[kind])} for kind in KINDS]
    table_path_out = write_csv(output_dirs["tables"] / "gradient-grid-seed-breakdown.csv", table_rows, TABLE_FIELDS)
    summary = {
        "results_path": str(table_path),
        "metric": METRIC,
        "accepted_row_count": len(rows),
        "by_layer": {str(layer): _group_summary(values) for layer, values in sorted(by_layer.items())},
        "by_kind": {kind: _group_summary(by_kind[kind]) for kind in KINDS},
        "figure_path": str(figure_path),
        "table_path": str(table_path_out),
    }
    summary_path = write_json(output_dirs["summaries"] / "gradient-grid-seed-breakdown.json", summary)
    return GradientGridBreakdownResult(
        figure_path=figure_path,
        summary_path=summary_path,
        table_path=table_path_out,
        summary=summary,
    )


def _write_plot(path: Path, by_layer: dict[int, list[float]], by_kind: dict[str, list[float]]) -> Path:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=panel_figsize(1, 2), sharey=True)

    layers = sorted(by_layer)
    styled_boxplot(
        axes[0],
        [by_layer[layer] for layer in layers],
        [str(layer) for layer in layers],
        ordinal_blues(len(layers)),
        edge="match",
    )
    axes[0].set_title("By Seed Layer")
    axes[0].set_xlabel("Seed layer")
    axes[0].set_ylabel("Counterfactual faithfulness")

    styled_boxplot(axes[1], [by_kind[kind] for kind in KINDS], list(KINDS), [BLUE] * len(KINDS), edge="match")
    axes[1].set_title("By Component Kind")
    axes[1].set_xlabel("Seed component kind")

    return save_figure(fig, path)


def _group_summary(values: list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }
