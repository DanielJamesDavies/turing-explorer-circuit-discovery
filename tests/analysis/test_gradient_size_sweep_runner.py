"""Unit tests for gradient_size_sweep_runner helpers (no GPU)."""

import pytest

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config,
    _circuit_depth_stats,
    _restoration_round_prefixes,
    _restore_sweep_config,
    _round_prefix_circuit,
)
from circuit.types.feature_id import FeatureID
from config import config
from store.circuits import Circuit, CircuitNode


def _node(circuit, layer, kind, index, role):
    return circuit.add_node(
        CircuitNode(metadata={"feature_id": FeatureID(layer, kind, index), "role": role})
    )


def test_star_circuit_has_depth_one_everywhere():
    circuit = Circuit(name="star")
    seed = _node(circuit, 5, "mlp", 1, "seed")
    for index in range(4):
        node = _node(circuit, index, "attn", index, "ablation_support")
        circuit.add_edge(node.uuid, seed.uuid, weight=1.0)
    stats = _circuit_depth_stats(circuit)
    assert stats["node_depth_max"] == 1
    assert stats["node_depth_mean"] == pytest.approx(1.0)
    assert stats["n_internal_edges"] == 0


def test_chain_circuit_reports_hop_depths():
    """seed <- a <- b <- c gives depths 1, 2, 3 and two internal edges."""
    circuit = Circuit(name="chain")
    seed = _node(circuit, 6, "resid", 0, "seed")
    a = _node(circuit, 4, "mlp", 1, "ablation_support")
    b = _node(circuit, 2, "mlp", 2, "ablation_support")
    c = _node(circuit, 0, "attn", 3, "ablation_support")
    circuit.add_edge(a.uuid, seed.uuid, weight=1.0)
    circuit.add_edge(b.uuid, a.uuid, weight=1.0)
    circuit.add_edge(c.uuid, b.uuid, weight=1.0)
    stats = _circuit_depth_stats(circuit)
    assert stats["node_depth_max"] == 3
    assert stats["node_depth_mean"] == pytest.approx((1 + 2 + 3) / 3)
    assert stats["n_internal_edges"] == 2


def test_seed_only_circuit_is_zero_depth():
    circuit = Circuit(name="empty")
    _node(circuit, 3, "attn", 0, "seed")
    stats = _circuit_depth_stats(circuit)
    assert stats == {"node_depth_max": 0, "node_depth_mean": 0.0, "n_internal_edges": 0}


def test_apply_sweep_config_restoration_budget_override_and_restore():
    cf = config.discovery.counterfactual_gradient
    ab = config.discovery.ablation_gradient
    original_cf_rounds, original_ab_rounds = cf.restoration.rounds, ab.restoration.rounds
    original_cf_k, original_ab_k = cf.restoration.per_round_k, ab.restoration.per_round_k

    saved = _apply_sweep_config(max_per_site=32, restoration_rounds=16, restoration_per_round_k=128)
    try:
        assert cf.restoration.rounds == 16
        assert ab.restoration.rounds == 16
        assert cf.restoration.per_round_k == 128
        assert ab.restoration.per_round_k == 128
    finally:
        _restore_sweep_config(saved)

    assert cf.restoration.rounds == original_cf_rounds
    assert ab.restoration.rounds == original_ab_rounds
    assert cf.restoration.per_round_k == original_cf_k
    assert ab.restoration.per_round_k == original_ab_k


def test_apply_sweep_config_leaves_restoration_budget_untouched_by_default():
    cf = config.discovery.counterfactual_gradient
    original_rounds, original_k = cf.restoration.rounds, cf.restoration.per_round_k

    saved = _apply_sweep_config(max_per_site=32)
    try:
        assert cf.restoration.rounds == original_rounds
        assert cf.restoration.per_round_k == original_k
    finally:
        _restore_sweep_config(saved)


def test_apply_sweep_config_final_ig_polish_override_and_restore():
    cf = config.discovery.counterfactual_gradient
    ab = config.discovery.ablation_gradient
    original_cf, original_ab = cf.restoration.final_ig_polish, ab.restoration.final_ig_polish

    saved = _apply_sweep_config(max_per_site=32, restoration_final_ig_polish=True)
    try:
        assert cf.restoration.final_ig_polish is True
        assert ab.restoration.final_ig_polish is True
    finally:
        _restore_sweep_config(saved)

    assert cf.restoration.final_ig_polish == original_cf
    assert ab.restoration.final_ig_polish == original_ab


def test_run_rejects_bogus_restoration_truncation():
    from analysis.circuits.gradient_size_sweep_runner import run_gradient_size_sweep

    with pytest.raises(ValueError, match="restoration_truncation"):
        run_gradient_size_sweep("nonexistent", restoration_truncation="bogus")


def _restoration_circuit():
    """seed + two round-1 nodes + one round-2 node + one unstamped node."""
    circuit = Circuit(name="restored")
    seed = _node(circuit, 6, "resid", 0, "seed")
    r1a = _node(circuit, 2, "mlp", 1, "counterfactual_support")
    r1b = _node(circuit, 3, "attn", 2, "counterfactual_support")
    r2 = _node(circuit, 1, "mlp", 3, "counterfactual_support")
    unstamped = _node(circuit, 0, "attn", 4, "counterfactual_support")
    for node, round_index in ((r1a, 1), (r1b, 1), (r2, 2)):
        node.metadata["selected_round"] = round_index
    for node in (r1a, r1b, r2, unstamped):
        circuit.add_edge(node.uuid, seed.uuid, weight=1.0)
    circuit.metadata["restoration_rounds_used"] = 2
    return circuit, seed, {1: {r1a.uuid, r1b.uuid}, 2: {r2.uuid}}


def test_round_prefix_circuit_is_nested_by_selected_round():
    circuit, seed, by_round = _restoration_circuit()

    sub1, n1 = _round_prefix_circuit(circuit, 1)
    assert n1 == 2
    assert set(sub1.nodes) == {seed.uuid} | by_round[1]
    # Edges into the seed from kept nodes survive; others are dropped.
    assert {e.source_uuid for e in sub1.edges} == by_round[1]

    sub2, n2 = _round_prefix_circuit(circuit, 2)
    assert n2 == 3
    assert set(sub2.nodes) == {seed.uuid} | by_round[1] | by_round[2]
    # Unstamped nodes are never included in a round prefix.
    assert set(sub1.nodes) < set(sub2.nodes)


def test_restoration_round_prefixes_require_provenance():
    circuit, _, _ = _restoration_circuit()
    assert _restoration_round_prefixes(circuit) == [1, 2]

    bare = Circuit(name="bare")
    _node(bare, 3, "mlp", 0, "seed")
    assert _restoration_round_prefixes(bare) == []

    # rounds_used without stamped nodes (e.g. legacy circuit) -> no prefixes.
    bare.metadata["restoration_rounds_used"] = 4
    assert _restoration_round_prefixes(bare) == []
