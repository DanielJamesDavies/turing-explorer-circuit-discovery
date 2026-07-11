import pytest
import torch
from pydantic import ValidationError

from circuit.discovery.hybrid_gradient import (
    HybridGradientDiscovery,
    compute_source_overlap,
    fuse_circuits_by_feature_id,
    prune_non_minimal_nodes_both,
    prune_sfc_threshold,
)
from circuit.discovery_window import METHOD_REGISTRY
from circuit.types.feature_id import FeatureID
from config import HybridGradientConfig
from store.circuits import Circuit, CircuitNode


KINDS = ["attn", "mlp", "resid"]


def _node(fid: FeatureID, role: str, score: float | None = None) -> CircuitNode:
    metadata = {"feature_id": fid, "role": role}
    if score is not None:
        metadata["attribution_score"] = score
    return CircuitNode(metadata=metadata)


def _circuit(name: str, seed: FeatureID, upstream: list[tuple[FeatureID, str, float]]) -> Circuit:
    circuit = Circuit(name=name)
    seed_node = circuit.add_node(_node(seed, "seed"))
    for fid, role, score in upstream:
        upstream_node = circuit.add_node(_node(fid, role, score))
        circuit.add_edge(upstream_node.uuid, seed_node.uuid, weight=score)
    circuit.metadata["discovery_method"] = name
    return circuit


def test_hybrid_config_accepts_defaults_and_valid_modes():
    default_cfg = HybridGradientConfig()
    cfg = HybridGradientConfig(
        acceptance_mode="both",
        pruning_objective="cf",
        pruning_method="sfc_threshold",
    )

    assert default_cfg.pruning_method == "leave_one_out"
    assert cfg.run_counterfactual is True
    assert cfg.run_ablation is True
    assert cfg.acceptance_mode == "both"
    assert cfg.pruning_objective == "cf"
    assert cfg.pruning_method == "sfc_threshold"
    assert cfg.sfc_score_mode == "abs"


def test_hybrid_config_rejects_invalid_modes():
    with pytest.raises(ValidationError):
        HybridGradientConfig(acceptance_mode="invalid")

    with pytest.raises(ValidationError):
        HybridGradientConfig(pruning_objective="invalid")

    with pytest.raises(ValidationError):
        HybridGradientConfig(pruning_method="invalid")

    with pytest.raises(ValidationError):
        HybridGradientConfig(sfc_score_mode="invalid")


def test_hybrid_gradient_is_registered():
    assert METHOD_REGISTRY["hybrid_gradient"] is HybridGradientDiscovery


def test_fuse_circuits_deduplicates_nodes_by_feature_id_and_remaps_edges():
    seed = FeatureID(0, "attn", 10)
    shared = FeatureID(0, "mlp", 20)
    cf_only = FeatureID(0, "resid", 30)
    ablation_only = FeatureID(1, "mlp", 40)

    cf = _circuit(
        "counterfactual_gradient",
        seed,
        [
            (shared, "counterfactual_activator", 0.5),
            (cf_only, "counterfactual_inhibitor", -0.7),
        ],
    )
    ablation = _circuit(
        "ablation_gradient",
        seed,
        [
            (shared, "ablation_support", 0.9),
            (ablation_only, "ablation_support", 0.3),
        ],
    )

    fused = fuse_circuits_by_feature_id(
        [("counterfactual_gradient", cf), ("ablation_gradient", ablation)],
        seed_comp_idx=0,
        seed_latent_idx=10,
        kinds=KINDS,
    )

    fids = {node.feature_id for node in fused.nodes.values()}
    assert fids == {seed, shared, cf_only, ablation_only}
    assert len(fused.nodes) == 4
    assert len(fused.edges) == 3

    shared_nodes = [node for node in fused.nodes.values() if node.feature_id == shared]
    assert len(shared_nodes) == 1
    shared_node = shared_nodes[0]
    assert shared_node.metadata["source_methods"] == [
        "ablation_gradient",
        "counterfactual_gradient",
    ]
    assert set(shared_node.metadata["roles"]) == {
        "ablation_support",
        "counterfactual_activator",
    }
    assert shared_node.metadata["attribution_scores"] == {
        "counterfactual_gradient": 0.5,
        "ablation_gradient": 0.9,
    }
    assert shared_node.metadata["attribution_score"] == 0.9

    for edge in fused.edges:
        assert edge.source_uuid in fused.nodes
        assert edge.target_uuid in fused.nodes


def test_compute_source_overlap_counts_source_sets_and_excludes_seed():
    seed = FeatureID(0, "attn", 10)
    shared = FeatureID(0, "mlp", 20)
    cf_only = FeatureID(0, "resid", 30)
    ablation_only = FeatureID(1, "mlp", 40)
    cf = _circuit(
        "counterfactual_gradient",
        seed,
        [
            (shared, "counterfactual_activator", 0.5),
            (cf_only, "counterfactual_inhibitor", -0.7),
        ],
    )
    ablation = _circuit(
        "ablation_gradient",
        seed,
        [
            (shared, "ablation_support", 0.9),
            (ablation_only, "ablation_support", 0.3),
        ],
    )
    fused = fuse_circuits_by_feature_id(
        [("counterfactual_gradient", cf), ("ablation_gradient", ablation)],
        seed_comp_idx=0,
        seed_latent_idx=10,
        kinds=KINDS,
    )

    overlap = compute_source_overlap(
        fused,
        seed_comp_idx=0,
        seed_latent_idx=10,
        kinds=KINDS,
    )

    assert overlap["seed_node_count"] == 1
    assert overlap["cf_node_count"] == 2
    assert overlap["ablation_node_count"] == 2
    assert overlap["cf_only_node_count"] == 1
    assert overlap["ablation_only_node_count"] == 1
    assert overlap["intersection_node_count"] == 1
    assert overlap["union_node_count"] == 3
    assert overlap["jaccard"] == pytest.approx(1 / 3)
    assert overlap["by_kind"]["mlp"]["intersection_node_count"] == 1
    assert overlap["by_layer"]["0"]["cf_only_node_count"] == 1


def test_fuse_circuits_rejects_mismatched_seed():
    expected_seed = FeatureID(0, "attn", 10)
    wrong_seed = FeatureID(0, "attn", 11)
    circuit = _circuit("counterfactual_gradient", wrong_seed, [])

    with pytest.raises(ValueError, match="does not contain expected seed"):
        fuse_circuits_by_feature_id(
            [("counterfactual_gradient", circuit)],
            seed_comp_idx=0,
            seed_latent_idx=expected_seed.index,
            kinds=KINDS,
        )


def test_hybrid_acceptance_modes():
    method = HybridGradientDiscovery.__new__(HybridGradientDiscovery)
    method.min_counterfactual_faithfulness = 0.2
    method.min_suppression_score = 0.2

    method.acceptance_mode = "cf"
    assert method._passes_acceptance(0.3, 0.0) is True
    assert method._passes_acceptance(0.1, 1.0) is False

    method.acceptance_mode = "suppression"
    assert method._passes_acceptance(0.0, 0.3) is True
    assert method._passes_acceptance(1.0, 0.1) is False

    method.acceptance_mode = "both"
    assert method._passes_acceptance(0.3, 0.3) is True
    assert method._passes_acceptance(0.3, 0.1) is False

    method.acceptance_mode = "either"
    assert method._passes_acceptance(0.3, 0.1) is True
    assert method._passes_acceptance(0.1, 0.3) is True
    assert method._passes_acceptance(0.1, 0.1) is False


def test_prune_sfc_threshold_filters_nodes_edges_and_isolated_nodes():
    seed = FeatureID(0, "attn", 10)
    strong = FeatureID(0, "mlp", 20)
    weak_node = FeatureID(0, "resid", 30)
    weak_edge = FeatureID(1, "mlp", 40)
    circuit = _circuit(
        "hybrid",
        seed,
        [
            (strong, "ablation_support", 0.5),
            (weak_node, "counterfactual_activator", 0.001),
            (weak_edge, "counterfactual_inhibitor", 0.4),
        ],
    )
    weak_edge_uuid = next(
        node.uuid for node in circuit.nodes.values() if node.feature_id == weak_edge
    )
    for edge in circuit.edges:
        if edge.source_uuid == weak_edge_uuid:
            edge.metadata["weight"] = 0.001

    removed = prune_sfc_threshold(circuit, node_threshold=0.01, edge_threshold=0.01)

    remaining_fids = {node.feature_id for node in circuit.nodes.values()}
    assert remaining_fids == {seed, strong}
    assert len(circuit.edges) == 1
    assert all(edge.weight is not None and abs(edge.weight) >= 0.01 for edge in circuit.edges)
    assert len(removed) == 2


def test_hybrid_prune_dispatches_sfc_threshold_without_leave_one_out(monkeypatch):
    seed = FeatureID(0, "attn", 10)
    weak = FeatureID(0, "mlp", 20)
    circuit = _circuit("hybrid", seed, [(weak, "ablation_support", 0.001)])

    def fail_leave_one_out(*args, **kwargs):
        raise AssertionError("leave-one-out pruning should not run for sfc_threshold")

    monkeypatch.setattr(
        "circuit.discovery.hybrid_gradient.prune_non_minimal_nodes_cf",
        fail_leave_one_out,
    )

    method = HybridGradientDiscovery.__new__(HybridGradientDiscovery)
    method.pruning_method = "sfc_threshold"
    method.pruning_objective = "cf"
    method.sfc_node_threshold = 0.01
    method.sfc_edge_threshold = 0.01

    removed = method._prune(
        circuit,
        neg_tokens_eval=torch.empty(1, 1, dtype=torch.long),
        pos_tokens_eval=torch.empty(1, 1, dtype=torch.long),
        seed_layer=0,
        seed_kind="attn",
        seed_latent_idx=10,
        pos_argmax_eval=torch.empty(1, dtype=torch.long),
        circuit_layers={0},
    )

    assert len(removed) == 1
    assert {node.feature_id for node in circuit.nodes.values()} == {seed}


def test_both_pruning_removes_only_nodes_weak_for_both_scores(monkeypatch):
    seed = FeatureID(0, "attn", 10)
    weak = FeatureID(0, "mlp", 20)
    strong = FeatureID(0, "resid", 30)
    circuit = _circuit(
        "hybrid",
        seed,
        [
            (weak, "ablation_support", 0.01),
            (strong, "counterfactual_activator", 1.0),
        ],
    )
    weak_uuid = next(node.uuid for node in circuit.nodes.values() if node.feature_id == weak)
    strong_uuid = next(node.uuid for node in circuit.nodes.values() if node.feature_id == strong)

    def fake_eval(*args, **kwargs):
        nodes = args[3].nodes
        if strong_uuid not in nodes:
            return 0.5, 0.99
        if weak_uuid not in nodes:
            return 0.99, 0.99
        return 1.0, 1.0

    monkeypatch.setattr(
        "circuit.discovery.hybrid_gradient.evaluate_counterfactual_faithfulness",
        fake_eval,
    )

    removed = prune_non_minimal_nodes_both(
        inference=None,
        sae_bank=None,
        avg_acts=torch.empty(0),
        circuit=circuit,
        neg_tokens=torch.empty(1, 1, dtype=torch.long),
        pos_tokens=torch.empty(1, 1, dtype=torch.long),
        seed_layer=0,
        seed_kind="attn",
        seed_latent_idx=10,
        threshold=0.05,
    )

    assert removed == [weak_uuid]
    assert weak_uuid not in circuit.nodes
    assert strong_uuid in circuit.nodes
