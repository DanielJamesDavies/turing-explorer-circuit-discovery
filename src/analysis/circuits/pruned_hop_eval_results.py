"""Plot pruned-hop circuit eval comparison results."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from analysis.io import analysis_output_dirs, resolve_run_root, write_json
from analysis.style import configure_matplotlib
from .coact_overlap import SUITE_NAME


@dataclass(frozen=True)
class PrunedHopEvalResultsPlot:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_pruned_hop_eval_results(
    run_root: str | Path,
    *,
    results_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> PrunedHopEvalResultsPlot:
    """Plot eval deltas between full circuits and hop-pruned variants."""

    root = resolve_run_root(run_root)
    table_path = (
        Path(results_path).expanduser().resolve()
        if results_path is not None
        else root / "analysis" / SUITE_NAME / "tables" / "pruned-hop-eval-results.csv"
    )
    rows = load_pruned_hop_eval_results(table_path)
    stats = compute_pruned_hop_eval_result_stats(rows)
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "pruned-hop-eval-comparison.png"
    summary_path = output_dirs["summaries"] / "pruned-hop-eval-comparison.json"

    _write_plot(figure_path, stats)
    summary = _build_summary(table_path, stats)
    write_json(summary_path, summary)
    return PrunedHopEvalResultsPlot(figure_path, summary_path, table_path, summary)


def load_pruned_hop_eval_results(path: str | Path) -> list[dict[str, Any]]:
    """Load pruned-hop eval results CSV."""

    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"pruned eval results not found: {table_path}. "
            "Run the intervention eval pass first and write pruned-hop-eval-results.csv."
        )
    with table_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if not row.get("error") and _is_finite(row.get("counterfactual_faithfulness")) and _is_finite(row.get("posctx_suppression_score"))]
    if not rows:
        raise ValueError(f"pruned eval results table has no successful finite rows: {table_path}")
    required = {
        "variant",
        "hop",
        "counterfactual_faithfulness",
        "posctx_suppression_score",
        "full_counterfactual_faithfulness",
        "full_posctx_suppression_score",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"pruned eval results missing columns: {sorted(missing)}")
    return rows


def compute_pruned_hop_eval_result_stats(rows: list[dict[str, Any]]) -> dict[str, object]:
    """Aggregate eval result deltas by hop-pruned variant."""

    hop_rows = [row for row in rows if str(row["variant"]).startswith("hop")]
    if not hop_rows:
        raise ValueError("results must include hop-pruned variants")
    hops = sorted({int(row["hop"]) for row in hop_rows})
    faithfulness_mean = []
    suppression_mean = []
    faithfulness_delta_mean = []
    suppression_delta_mean = []
    for hop in hops:
        selected = [row for row in hop_rows if int(row["hop"]) == hop]
        cf = [_float(row["counterfactual_faithfulness"]) for row in selected]
        sup = [_float(row["posctx_suppression_score"]) for row in selected]
        full_cf = [_float(row["full_counterfactual_faithfulness"]) for row in selected]
        full_sup = [_float(row["full_posctx_suppression_score"]) for row in selected]
        faithfulness_mean.append(float(mean(cf)) if cf else 0.0)
        suppression_mean.append(float(mean(sup)) if sup else 0.0)
        faithfulness_delta_mean.append(float(mean([value - base for value, base in zip(cf, full_cf)])) if cf else 0.0)
        suppression_delta_mean.append(float(mean([value - base for value, base in zip(sup, full_sup)])) if sup else 0.0)

    full_rows = [row for row in rows if str(row["variant"]) == "full"]
    return {
        "hops": hops,
        "variant_count": len(rows),
        "circuit_count": len({row.get("uuid", row.get("sample_index", "")) for row in rows}),
        "full_counterfactual_faithfulness_mean": float(mean([_float(row["full_counterfactual_faithfulness"]) for row in hop_rows])),
        "full_posctx_suppression_score_mean": float(mean([_float(row["full_posctx_suppression_score"]) for row in hop_rows])),
        "faithfulness_mean": faithfulness_mean,
        "suppression_mean": suppression_mean,
        "faithfulness_delta_mean": faithfulness_delta_mean,
        "suppression_delta_mean": suppression_delta_mean,
        "full_rows_present": bool(full_rows),
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    hops = stats["hops"]
    assert isinstance(hops, list)
    labels = [f"hop{hop}" for hop in hops]
    x = range(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(labels, stats["faithfulness_mean"], marker="o", linewidth=2.0, label="pruned cf faith")
    axes[0].axhline(
        float(stats["full_counterfactual_faithfulness_mean"]),
        color="#b45f06",
        linestyle="--",
        linewidth=2.0,
        label="full circuit mean",
    )
    axes[0].set_title("Counterfactual Faithfulness")
    axes[0].set_ylabel("Eval score")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].legend(loc="best")

    axes[1].plot(labels, stats["suppression_mean"], marker="o", linewidth=2.0, label="pruned suppression")
    axes[1].axhline(
        float(stats["full_posctx_suppression_score_mean"]),
        color="#b45f06",
        linestyle="--",
        linewidth=2.0,
        label="full circuit mean",
    )
    axes[1].set_title("Posctx Suppression Score")
    axes[1].set_ylabel("Eval score")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].legend(loc="best")
    fig.suptitle("Pruned-Hop Eval Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _build_summary(table_path: Path, stats: dict[str, object]) -> dict[str, object]:
    return {
        "results_path": str(table_path),
        "circuit_count": stats["circuit_count"],
        "variant_count": stats["variant_count"],
        "hops": stats["hops"],
        "full_counterfactual_faithfulness_mean": stats["full_counterfactual_faithfulness_mean"],
        "full_posctx_suppression_score_mean": stats["full_posctx_suppression_score_mean"],
        "faithfulness_mean": stats["faithfulness_mean"],
        "suppression_mean": stats["suppression_mean"],
        "faithfulness_delta_mean": stats["faithfulness_delta_mean"],
        "suppression_delta_mean": stats["suppression_delta_mean"],
    }


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

