from __future__ import annotations

import csv
from pathlib import Path

from analysis.circuits.gradient_neg_mode_comparison import (
    compute_gradient_neg_mode_stats,
    compute_paired_mode_deltas,
    load_gradient_neg_mode_rows,
    plot_gradient_neg_mode_comparison,
)


FIELDS = [
    "method",
    "neg_mode",
    "run_index",
    "candidate_index",
    "kind",
    "layer",
    "comp_idx",
    "latent_idx",
    "status",
    "n_nodes",
    "n_edges",
    "duration_s",
    "peak_gb",
    "counterfactual_faithfulness",
    "posctx_suppression_score",
    "source_counterfactual_returned",
    "source_ablation_returned",
    "error",
]


def _write_grid_csv(path: Path) -> None:
    rows = []
    for method_index, method in enumerate(
        ("counterfactual_gradient", "ablation_gradient", "hybrid_gradient")
    ):
        for candidate_index in range(2):
            for mode_index, mode in enumerate(("close", "random", "distant")):
                accepted = not (
                    method == "counterfactual_gradient"
                    and mode == "distant"
                    and candidate_index == 1
                )
                rows.append(
                    {
                        "method": method,
                        "neg_mode": mode,
                        "run_index": candidate_index,
                        "candidate_index": candidate_index,
                        "kind": "mlp",
                        "layer": method_index,
                        "comp_idx": method_index,
                        "latent_idx": 10 + candidate_index,
                        "status": "accepted" if accepted else "none",
                        "n_nodes": 10 + method_index + mode_index,
                        "n_edges": 9 + method_index + mode_index,
                        "duration_s": 1.0 + mode_index,
                        "peak_gb": "",
                        "counterfactual_faithfulness": 0.8 - 0.1 * mode_index + 0.01 * method_index
                        if accepted
                        else "",
                        "posctx_suppression_score": 0.9 - 0.05 * mode_index if accepted else "",
                        "source_counterfactual_returned": "",
                        "source_ablation_returned": "",
                        "error": "",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_mode_stats_include_overall_and_method_views(tmp_path):
    csv_path = tmp_path / "grid.csv"
    _write_grid_csv(csv_path)
    rows = load_gradient_neg_mode_rows(csv_path)

    aggregate_rows, stats = compute_gradient_neg_mode_stats(rows)

    assert len(aggregate_rows) == 12
    assert stats["by_mode"]["close"]["accepted_count"] == 6
    assert stats["by_mode"]["distant"]["accepted_count"] == 5
    assert stats["by_method_mode"]["counterfactual_gradient"]["distant"]["accepted_count"] == 1
    assert stats["best_mode_by_method"]["hybrid_gradient"]["mean_faithfulness"] == "close"


def test_paired_deltas_are_method_and_candidate_matched(tmp_path):
    csv_path = tmp_path / "grid.csv"
    _write_grid_csv(csv_path)
    rows = load_gradient_neg_mode_rows(csv_path)

    paired_rows, summary = compute_paired_mode_deltas(rows)

    assert all(row["method"] in {"counterfactual_gradient", "ablation_gradient", "hybrid_gradient"} for row in paired_rows)
    cf_close_distant = [
        row
        for row in paired_rows
        if row["method"] == "counterfactual_gradient" and row["mode_pair"] == "close-distant"
    ]
    assert len(cf_close_distant) == 1
    assert "duration_delta" not in cf_close_distant[0]
    assert summary["by_method"]["counterfactual_gradient"]["close-distant"]["count"] == 1


def test_plot_gradient_neg_mode_comparison_writes_outputs(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    csv_path = tmp_path / "grid.csv"
    _write_grid_csv(csv_path)

    result = plot_gradient_neg_mode_comparison(
        run_root,
        results_path=csv_path,
        output_root=tmp_path / "analysis-out",
    )

    assert len(result.figure_paths) == 10
    assert all(path.exists() for path in result.figure_paths)
    assert all("runtime" not in path.name and "per-second" not in path.name for path in result.figure_paths)
    assert result.summary_path.exists()
    assert result.aggregate_table_path.exists()
    assert result.paired_delta_table_path.exists()
    assert result.summary["by_method_mode"]["ablation_gradient"]["random"]["accepted_count"] == 2
