from __future__ import annotations

import torch

from analysis.circuits.gradient_method_neg_mode_grid_runner import ROW_FIELDS, _run_candidate
from store.circuits import Circuit, CircuitNode


class FakeBank:
    kinds = ["attn", "mlp", "resid"]


class FakeMethod:
    def discover(self, comp_idx: int, latent_idx: int):
        circuit = Circuit(name="fake")
        circuit.add_node(CircuitNode(metadata={"role": "seed"}))
        circuit.metadata.update(
            {
                "counterfactual_faithfulness": 0.7,
                "posctx_suppression_score": 0.9,
                "source_counterfactual_returned": True,
                "source_ablation_returned": True,
                "source_cf_node_count": 3,
                "source_ablation_node_count": 4,
                "source_intersection_node_count": 2,
                "source_union_node_count": 5,
                "source_cf_only_node_count": 1,
                "source_ablation_only_node_count": 2,
                "source_jaccard": 0.4,
                "post_prune_cf_only_node_count": 1,
                "post_prune_ablation_only_node_count": 1,
                "post_prune_intersection_node_count": 2,
                "post_prune_union_node_count": 4,
                "post_prune_jaccard": 0.5,
            }
        )
        return circuit


def test_grid_row_fields_include_hybrid_source_overlap_columns():
    for field in (
        "source_cf_node_count",
        "source_ablation_node_count",
        "source_intersection_node_count",
        "source_union_node_count",
        "source_cf_only_node_count",
        "source_ablation_only_node_count",
        "source_jaccard",
        "post_prune_cf_only_node_count",
        "post_prune_ablation_only_node_count",
        "post_prune_intersection_node_count",
        "post_prune_union_node_count",
        "post_prune_jaccard",
    ):
        assert field in ROW_FIELDS


def test_run_candidate_extracts_hybrid_source_overlap_metadata():
    row = _run_candidate(
        FakeMethod(),
        method_name="hybrid_gradient",
        neg_mode="close",
        run_index=0,
        candidate={"candidate_index": 7, "comp_idx": 1, "latent_idx": 2},
        bank=FakeBank(),
        device=torch.device("cpu"),
    )

    assert row["source_cf_node_count"] == 3
    assert row["source_ablation_node_count"] == 4
    assert row["source_intersection_node_count"] == 2
    assert row["source_union_node_count"] == 5
    assert row["source_jaccard"] == 0.4
    assert row["post_prune_union_node_count"] == 4
    assert row["post_prune_jaccard"] == 0.5
