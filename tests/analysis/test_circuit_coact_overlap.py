from analysis.circuits.coact_overlap import compute_circuit_coact_overlap
from analysis.circuits.latent_commonality import compute_circuit_latent_commonality
from analysis.circuits.node_hop_overlap import compute_circuit_node_hop_overlap
from analysis.circuits.pruned_hop_eval_results import compute_pruned_hop_eval_result_stats
from analysis.circuits.pruned_hop_eval_spec import build_pruned_hop_eval_spec
from analysis.circuits.seed_coact_hops import compute_circuit_seed_coact_hops
from analysis.circuits.top_ctx_frequency import compute_top_ctx_circuit_vs_coact_frequency

import torch
from circuit.types.feature_id import FeatureID
from store.circuits import Circuit, CircuitNode


def test_compute_circuit_coact_overlap_extracts_summary_metrics():
    rows = [
        {
            "uuid": "a",
            "name": "circuit-a",
            "nodes": 10,
            "edges": 9,
            "metadata": {
                "seed_comp": 1,
                "seed_latent": 2,
                "n_activators": 4,
                "n_inhibitors": 5,
                "evals": {"counterfactual_faithfulness": 0.8},
                "post_analysis": {
                    "coact_overlap_pct": 20.0,
                    "coact_overlap_pct_activators": 5.0,
                    "coact_overlap_pct_inhibitors": 30.0,
                    "internode_coact_density_pct": 10.0,
                },
            },
        },
        {
            "uuid": "b",
            "name": "circuit-b",
            "nodes": 20,
            "edges": 19,
            "metadata": {
                "seed_comp": 3,
                "seed_latent": 4,
                "n_activators": 8,
                "n_inhibitors": 10,
                "evals": {"counterfactual_faithfulness": 0.4},
                "post_analysis": {
                    "coact_overlap_pct": 40.0,
                    "coact_overlap_pct_activators": 10.0,
                    "coact_overlap_pct_inhibitors": 60.0,
                    "internode_coact_density_pct": 5.0,
                },
            },
        },
    ]

    stats = compute_circuit_coact_overlap(rows)

    assert stats["circuit_count"] == 2
    assert stats["coact_summary"]["mean"] == 30.0
    assert stats["inhibitor_summary"]["max"] == 60.0
    assert stats["top_by_overlap"][0]["uuid"] == "b"
    assert "coact_overlap_vs_faithfulness" in stats["correlations"]


def test_compute_circuit_seed_coact_hops_expands_seed_neighborhoods():
    rows = [
        {
            "uuid": "a",
            "name": "circuit-a",
            "nodes": 10,
            "edges": 9,
            "metadata": {
                "seed_comp": 0,
                "seed_latent": 0,
                "n_activators": 4,
                "n_inhibitors": 5,
                "evals": {"counterfactual_faithfulness": 0.8},
                "post_analysis": {
                    "coact_overlap_pct": 20.0,
                    "coact_overlap_pct_activators": 5.0,
                    "coact_overlap_pct_inhibitors": 30.0,
                    "internode_coact_density_pct": 10.0,
                },
            },
        }
    ]
    values = torch.tensor([[[3.0], [3.0], [3.0], [0.0]]], dtype=torch.float32)
    indices = torch.tensor([[[1], [2], [3], [0]]], dtype=torch.int64)

    stats = compute_circuit_seed_coact_hops(
        rows,
        values,
        indices,
        d_sae=4,
        threshold=2.0,
        top_out_degree=1,
        max_frontier=4,
        hub_quantile=1.0,
    )

    assert stats["circuit_count"] == 1
    assert stats["unpruned_reach_summary"]["1"]["max"] == 1.0
    assert stats["unpruned_reach_summary"]["3"]["max"] == 3.0
    assert stats["table_rows"][0]["seed_global_id"] == 0


def test_compute_circuit_node_hop_overlap_recovers_actual_circuit_nodes():
    circuit = Circuit(name="test-circuit")
    circuit.metadata = {
        "seed_comp": 0,
        "seed_latent": 0,
        "evals": {"counterfactual_faithfulness": 0.75},
    }
    seed = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 0), "role": "seed"}))
    first = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 1), "role": "counterfactual_activator"}))
    second = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 2), "role": "counterfactual_inhibitor"}))
    third = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 3), "role": "counterfactual_inhibitor"}))
    circuit.add_edge(seed.uuid, first.uuid)
    circuit.add_edge(first.uuid, second.uuid)
    circuit.add_edge(second.uuid, third.uuid)

    values = torch.tensor([[[3.0], [3.0], [3.0], [0.0]]], dtype=torch.float32)
    indices = torch.tensor([[[1], [2], [3], [0]]], dtype=torch.int64)

    stats = compute_circuit_node_hop_overlap(
        {"c": circuit},
        values,
        indices,
        d_sae=4,
        threshold=2.0,
        top_out_degree=1,
        max_frontier=4,
        hub_quantile=1.0,
        max_hops=6,
        kinds=("attn",),
    )

    row = stats["rows"][0]
    assert stats["circuit_count"] == 1
    assert row["all_hop1_count"] == 1
    assert row["all_hop2_count"] == 2
    assert row["all_hop3_count"] == 3
    assert row["all_hop3_pct"] == 100.0
    assert row["all_hop6_pct"] == 100.0
    assert row["inhibitor_hop2_pct"] == 50.0
    assert stats["all_pct_summary"]["6"]["max"] == 100.0


def test_compute_circuit_latent_commonality_counts_latent_reuse():
    circuits = {}
    for idx in range(3):
        circuit = Circuit(name=f"test-circuit-{idx}")
        circuit.metadata = {
            "seed_comp": 0,
            "seed_latent": 0,
            "evals": {"counterfactual_faithfulness": 0.5 + idx},
        }
        circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 0), "role": "seed"}))
        circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 1), "role": "counterfactual_activator"}))
        if idx != 1:
            circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 2), "role": "counterfactual_inhibitor"}))
        circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 3 + idx), "role": "counterfactual_activator"}))
        circuits[circuit.uuid] = circuit

    stats = compute_circuit_latent_commonality(
        circuits,
        rare_max_pct=50.0,
        common_min_pct=90.0,
        kinds=("attn",),
    )

    assert stats["circuit_count"] == 3
    assert stats["unique_latent_count"] == 5
    assert stats["latent_rows"][0]["latent"] == "L0.attn.f1"
    assert stats["latent_rows"][0]["circuit_count"] == 3
    assert stats["latent_rows"][0]["circuit_pct"] == 100.0
    assert stats["rare_max_pct"] == 50.0
    assert stats["common_min_pct"] == 90.0
    assert stats["bucket_latent_counts"] == {"singleton": 3, "rare": 0, "shared": 1, "common": 1}

    rows_by_name = {row["name"]: row for row in stats["circuit_rows"]}
    assert abs(rows_by_name["test-circuit-0"]["common_latent_pct"] - (100.0 / 3.0)) < 1e-12
    assert abs(rows_by_name["test-circuit-0"]["shared_latent_pct"] - (100.0 / 3.0)) < 1e-12
    assert rows_by_name["test-circuit-1"]["singleton_latent_pct"] == 50.0


def test_build_pruned_hop_eval_spec_stores_full_and_pruned_variants():
    circuit = Circuit(name="test-circuit")
    circuit.metadata = {
        "seed_comp": 0,
        "seed_latent": 0,
        "evals": {"counterfactual_faithfulness": 0.75, "posctx_suppression_score": 0.5},
    }
    seed = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 0), "role": "seed"}))
    first = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 1), "role": "counterfactual_activator"}))
    second = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 2), "role": "counterfactual_inhibitor"}))
    circuit.add_edge(seed.uuid, first.uuid)
    circuit.add_edge(first.uuid, second.uuid)
    values = torch.tensor([[[3.0], [3.0], [0.0]]], dtype=torch.float32)
    indices = torch.tensor([[[1], [2], [0]]], dtype=torch.int64)

    spec = build_pruned_hop_eval_spec(
        {"c": circuit},
        values,
        indices,
        d_sae=3,
        sample_size=1,
        max_hops=2,
        threshold=2.0,
        top_out_degree=1,
        max_frontier=4,
        hub_quantile=1.0,
        kinds=("attn",),
    )

    rows = spec["rows"]
    variants = spec["variants"][circuit.uuid]
    assert spec["actual_sample_size"] == 1
    assert len(rows) == 3
    assert set(variants) == {"full", "hop1", "hop2"}
    assert len(variants["hop1"].nodes) == 2
    assert len(variants["hop2"].nodes) == 3


def test_compute_pruned_hop_eval_result_stats_aggregates_by_hop():
    rows = [
        {
            "uuid": "a",
            "variant": "hop1",
            "hop": "1",
            "counterfactual_faithfulness": "0.2",
            "posctx_suppression_score": "0.3",
            "full_counterfactual_faithfulness": "1.0",
            "full_posctx_suppression_score": "0.9",
        },
        {
            "uuid": "a",
            "variant": "hop2",
            "hop": "2",
            "counterfactual_faithfulness": "0.4",
            "posctx_suppression_score": "0.5",
            "full_counterfactual_faithfulness": "1.0",
            "full_posctx_suppression_score": "0.9",
        },
    ]

    stats = compute_pruned_hop_eval_result_stats(rows)

    assert stats["hops"] == [1, 2]
    assert stats["faithfulness_mean"] == [0.2, 0.4]
    assert stats["suppression_delta_mean"] == [-0.6000000000000001, -0.4]


def test_compute_top_ctx_frequency_compares_circuit_and_coact_nodes():
    circuit = Circuit(name="test-circuit")
    circuit.metadata = {"seed_comp": 0, "seed_latent": 0}
    seed = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 0), "role": "seed"}))
    first = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 1), "role": "counterfactual_activator"}))
    second = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", 2), "role": "counterfactual_inhibitor"}))
    circuit.add_edge(seed.uuid, first.uuid)
    circuit.add_edge(first.uuid, second.uuid)
    values = torch.tensor([[[3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    indices = torch.tensor([[[2, 3], [0, 0], [0, 0], [0, 0]]], dtype=torch.int64)
    top_ctx = torch.tensor(
        [
            [
                [10, 11, 12],
                [10, 20, 21],
                [30, 31, 32],
                [10, 11, 40],
            ]
        ],
        dtype=torch.int64,
    )

    stats = compute_top_ctx_circuit_vs_coact_frequency(
        {"c": circuit},
        values,
        indices,
        top_ctx,
        d_sae=4,
        sample_size=1,
        threshold=2.0,
        kinds=("attn",),
    )

    assert stats["actual_seed_count"] == 1
    assert sorted(stats["circuit_counts"]) == [0, 1]
    assert sorted(stats["coact_counts"]) == [0, 2]
    assert stats["rows"][0]["circuit_and_coact_node_count"] == 1

