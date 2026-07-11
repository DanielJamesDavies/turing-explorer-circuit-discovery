"""Descriptive statistics for the gradient method x negmode grid runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json

from .gradient_method_eval_distribution import _float, _resolve_grid_results_path
from .gradient_method_neg_mode_grid_runner import GRID_METHODS, GRID_NEG_MODES, SUITE_NAME as GRID_SUITE_NAME
from .gradient_neg_mode_comparison import _accepted_finite, load_gradient_neg_mode_rows

TABLE_FIELDS = [
    "scope",
    "method",
    "neg_mode",
    "runs",
    "accepted",
    "acceptance_rate",
    "cf_median",
    "cf_median_top_n",
    "n_nodes_median",
    "n_nodes_p25",
    "n_nodes_p75",
    "n_edges_median",
    "duration_s_median",
]


@dataclass(frozen=True)
class GradientGridDescriptivesResult:
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def compute_gradient_grid_descriptives(
    run_root: str | Path,
    *,
    results_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> GradientGridDescriptivesResult:
    """Summarise run counts, acceptance, circuit sizes, and runtime per method."""

    root = resolve_run_root(run_root)
    table_path = _resolve_grid_results_path(root, results_path)
    rows = load_gradient_neg_mode_rows(table_path)

    # Equal-population comparison: each method's top-n circuits by faithfulness,
    # with n set to the smallest per-method accepted count (matches the top-n
    # distribution figure's default).
    accepted_counts = [
        sum(1 for row in rows if row["method"] == method and _accepted_finite(row)) for method in GRID_METHODS
    ]
    top_n = min((count for count in accepted_counts if count > 0), default=0)

    table_rows: list[dict[str, object]] = []
    by_method: dict[str, dict[str, object]] = {}
    for method in GRID_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        stats = _group_stats(method_rows, top_n=top_n)
        by_method[method] = stats
        table_rows.append({"scope": "method", "method": method, "neg_mode": "all", **stats})
        for mode in GRID_NEG_MODES:
            group = [row for row in method_rows if row["neg_mode"] == mode]
            table_rows.append(
                {"scope": "method_mode", "method": method, "neg_mode": mode, **_group_stats(group, top_n=top_n)}
            )
    overall = _group_stats(rows, top_n=top_n)
    table_rows.append({"scope": "overall", "method": "all", "neg_mode": "all", **overall})

    output_dirs = analysis_output_dirs(root, GRID_SUITE_NAME, output_root=output_root)
    table_path_out = write_csv(output_dirs["tables"] / "gradient-grid-descriptives.csv", table_rows, TABLE_FIELDS)
    summary = {
        "results_path": str(table_path),
        "top_n": top_n,
        "overall": overall,
        "by_method": by_method,
        "table_path": str(table_path_out),
    }
    summary_path = write_json(output_dirs["summaries"] / "gradient-grid-descriptives.json", summary)
    return GradientGridDescriptivesResult(summary_path=summary_path, table_path=table_path_out, summary=summary)


def _group_stats(rows: list[dict[str, Any]], *, top_n: int = 0) -> dict[str, object]:
    accepted = [row for row in rows if _accepted_finite(row)]
    nodes = sorted(_float(row["n_nodes"]) for row in accepted)
    edges = sorted(_float(row["n_edges"]) for row in accepted)
    durations = sorted(_float(row["duration_s"]) for row in rows)
    faithfulness = sorted(_float(row["counterfactual_faithfulness"]) for row in accepted)
    top_slice = faithfulness[-top_n:] if top_n > 0 else []
    return {
        "runs": len(rows),
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
        "cf_median": float(median(faithfulness)) if faithfulness else 0.0,
        "cf_median_top_n": float(median(top_slice)) if top_slice else 0.0,
        "n_nodes_median": float(median(nodes)) if nodes else 0.0,
        "n_nodes_p25": _quantile(nodes, 0.25),
        "n_nodes_p75": _quantile(nodes, 0.75),
        "n_edges_median": float(median(edges)) if edges else 0.0,
        "duration_s_median": float(median(durations)) if durations else 0.0,
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
