from __future__ import annotations

import csv

import pytest

from analysis.circuits.hybrid_source_overlap import (
    compute_hybrid_source_overlap_paired_deltas,
    compute_hybrid_source_overlap_stats,
    load_hybrid_source_overlap_rows,
    plot_hybrid_source_overlap,
)


FIELDS = [
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
    "error",
]

MODE_VALUES = {
    "close": {
        "source_cf_only_node_count": 2,
        "source_ablation_only_node_count": 3,
        "source_intersection_node_count": 5,
        "source_union_node_count": 10,
        "source_jaccard": 0.5,
        "post_prune_cf_only_node_count": 1,
        "post_prune_ablation_only_node_count": 2,
        "post_prune_intersection_node_count": 5,
        "post_prune_union_node_count": 8,
        "post_prune_jaccard": 0.625,
    },
    "random": {
        "source_cf_only_node_count": 4,
        "source_ablation_only_node_count": 4,
        "source_intersection_node_count": 4,
        "source_union_node_count": 12,
        "source_jaccard": 1 / 3,
        "post_prune_cf_only_node_count": 4,
        "post_prune_ablation_only_node_count": 3,
        "post_prune_intersection_node_count": 3,
        "post_prune_union_node_count": 10,
        "post_prune_jaccard": 0.3,
    },
    "distant": {
        "source_cf_only_node_count": 1,
        "source_ablation_only_node_count": 2,
        "source_intersection_node_count": 5,
        "source_union_node_count": 8,
        "source_jaccard": 0.625,
        "post_prune_cf_only_node_count": 1,
        "post_prune_ablation_only_node_count": 1,
        "post_prune_intersection_node_count": 4,
        "post_prune_union_node_count": 6,
        "post_prune_jaccard": 2 / 3,
    },
}


def _write_grid_csv(path, *, omit: set[str] | None = None) -> None:
    omit = omit or set()
    rows = []
    for candidate_index in range(2):
        for mode_index, (mode, values) in enumerate(MODE_VALUES.items()):
            rows.append(
                {
                    "method": "hybrid_gradient",
                    "neg_mode": mode,
                    "candidate_index": candidate_index,
                    "status": "accepted",
                    "counterfactual_faithfulness": 0.8 - 0.1 * mode_index,
                    "posctx_suppression_score": 0.9 - 0.05 * mode_index,
                    **values,
                    "error": "",
                }
            )
    rows.append(
        {
            "method": "counterfactual_gradient",
            "neg_mode": "close",
            "candidate_index": 99,
            "status": "accepted",
            "counterfactual_faithfulness": 1.0,
            "posctx_suppression_score": 1.0,
            **MODE_VALUES["close"],
            "error": "",
        }
    )
    fieldnames = [field for field in FIELDS if field not in omit]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_loader_filters_to_accepted_hybrid_overlap_rows(tmp_path):
    csv_path = tmp_path / "grid.csv"
    _write_grid_csv(csv_path)

    rows = load_hybrid_source_overlap_rows(csv_path)

    assert len(rows) == 6
    assert {row["method"] for row in rows} == {"hybrid_gradient"}
    assert rows[0]["source_cf_only_ratio"] == pytest.approx(0.2)


def test_loader_rejects_missing_required_columns(tmp_path):
    csv_path = tmp_path / "grid.csv"
    _write_grid_csv(csv_path, omit={"source_jaccard"})

    with pytest.raises(ValueError, match="missing columns"):
        load_hybrid_source_overlap_rows(csv_path)


def test_overlap_stats_summarize_counts_ratios_and_jaccard_by_mode(tmp_path):
    csv_path = tmp_path / "grid.csv"
    _write_grid_csv(csv_path)
    rows = load_hybrid_source_overlap_rows(csv_path)

    aggregate_rows, stats = compute_hybrid_source_overlap_stats(rows)

    assert len(aggregate_rows) == 3
    assert stats["by_mode"]["close"]["count"] == 2
    assert stats["by_mode"]["close"]["source_jaccard"]["mean"] == pytest.approx(0.5)
    assert stats["by_mode"]["close"]["source_cf_only_ratio"]["mean"] == pytest.approx(0.2)
    assert stats["by_mode"]["distant"]["post_prune_jaccard"]["mean"] == pytest.approx(2 / 3)


def test_overlap_paired_deltas_are_candidate_matched(tmp_path):
    csv_path = tmp_path / "grid.csv"
    _write_grid_csv(csv_path)
    rows = load_hybrid_source_overlap_rows(csv_path)

    paired_rows, summary = compute_hybrid_source_overlap_paired_deltas(rows)

    close_random = [row for row in paired_rows if row["mode_pair"] == "close-random"]
    assert len(close_random) == 2
    assert close_random[0]["source_jaccard_delta"] == pytest.approx(0.5 - 1 / 3)
    assert summary["close-random"]["count"] == 2


def test_plot_hybrid_source_overlap_writes_outputs(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    csv_path = tmp_path / "grid.csv"
    _write_grid_csv(csv_path)

    result = plot_hybrid_source_overlap(
        run_root,
        results_path=csv_path,
        output_root=tmp_path / "analysis-out",
    )

    assert len(result.figure_paths) == 8
    assert all(path.exists() for path in result.figure_paths)
    assert result.summary_path.exists()
    assert result.aggregate_table_path.exists()
    assert result.paired_delta_table_path.exists()
    assert result.summary["row_count"] == 6
    assert result.summary["by_mode"]["random"]["source_union_node_count"]["mean"] == pytest.approx(12)
