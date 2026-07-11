"""Unit tests for gradient_size_sweep_runner helpers (no GPU)."""

import pytest

from analysis.circuits.gradient_size_sweep_runner import _circuit_depth_stats
from circuit.types.feature_id import FeatureID
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
