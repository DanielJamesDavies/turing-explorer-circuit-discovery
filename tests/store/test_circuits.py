"""
Phase 1 — Circuit data structure tests.

Pure Python / pure PyTorch: no model, no SAE, no hooks required.
Tests CircuitNode, CircuitEdge, Circuit, and CircuitStore in isolation.
"""
import os
import tempfile

import pytest
import torch

from store.circuits import Circuit, CircuitEdge, CircuitNode, CircuitStore


# ---------------------------------------------------------------------------
# CircuitNode
# ---------------------------------------------------------------------------

class TestCircuitNode:
    def test_uuid_auto_generated(self):
        node = CircuitNode()
        assert isinstance(node.uuid, str)
        assert len(node.uuid) > 0

    def test_two_nodes_have_different_uuids(self):
        assert CircuitNode().uuid != CircuitNode().uuid

    def test_metadata_defaults_to_empty_dict(self):
        node = CircuitNode()
        assert node.metadata == {}

    def test_metadata_stored_correctly(self):
        node = CircuitNode(metadata={"layer_idx": 3, "kind": "mlp", "latent_idx": 42})
        assert node.metadata["layer_idx"] == 3
        assert node.metadata["kind"] == "mlp"
        assert node.metadata["latent_idx"] == 42

    def test_weight_property_reads_from_metadata(self):
        node = CircuitNode(metadata={"weight": 0.75})
        assert node.weight == 0.75

    def test_weight_property_none_when_absent(self):
        node = CircuitNode()
        assert node.weight is None

    def test_source_property_reads_from_metadata(self):
        node = CircuitNode(metadata={"source": "logit_attribution"})
        assert node.source == "logit_attribution"

    def test_source_property_none_when_absent(self):
        node = CircuitNode()
        assert node.source is None

    def test_explicit_uuid_respected(self):
        node = CircuitNode(uuid="fixed-uuid")
        assert node.uuid == "fixed-uuid"


# ---------------------------------------------------------------------------
# CircuitEdge
# ---------------------------------------------------------------------------

class TestCircuitEdge:
    def test_uuid_auto_generated(self):
        edge = CircuitEdge(source_uuid="a", target_uuid="b")
        assert isinstance(edge.uuid, str)
        assert len(edge.uuid) > 0

    def test_two_edges_have_different_uuids(self):
        e1 = CircuitEdge(source_uuid="a", target_uuid="b")
        e2 = CircuitEdge(source_uuid="a", target_uuid="b")
        assert e1.uuid != e2.uuid

    def test_source_and_target_stored(self):
        edge = CircuitEdge(source_uuid="src-uuid", target_uuid="tgt-uuid")
        assert edge.source_uuid == "src-uuid"
        assert edge.target_uuid == "tgt-uuid"

    def test_metadata_defaults_to_empty_dict(self):
        edge = CircuitEdge(source_uuid="a", target_uuid="b")
        assert edge.metadata == {}

    def test_weight_property_reads_from_metadata(self):
        edge = CircuitEdge(source_uuid="a", target_uuid="b", metadata={"weight": 0.3})
        assert edge.weight == pytest.approx(0.3)

    def test_weight_property_none_when_absent(self):
        edge = CircuitEdge(source_uuid="a", target_uuid="b")
        assert edge.weight is None


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------

class TestCircuit:
    def _make_node(self, layer_idx=0, kind="mlp", latent_idx=0, role="upstream"):
        return CircuitNode(metadata={
            "layer_idx": layer_idx,
            "kind": kind,
            "latent_idx": latent_idx,
            "role": role,
        })

    def test_nodes_defaults_to_empty_dict(self):
        c = Circuit(name="test")
        assert c.nodes == {}

    def test_edges_defaults_to_empty_list(self):
        c = Circuit(name="test")
        assert c.edges == []

    def test_metadata_defaults_to_empty_dict(self):
        c = Circuit(name="test")
        assert c.metadata == {}

    def test_uuid_auto_generated(self):
        c = Circuit(name="test")
        assert isinstance(c.uuid, str) and len(c.uuid) > 0

    def test_two_circuits_have_different_uuids(self):
        assert Circuit(name="a").uuid != Circuit(name="b").uuid

    def test_add_node_stores_by_uuid(self):
        c = Circuit(name="test")
        node = self._make_node()
        returned = c.add_node(node)
        assert returned is node
        assert c.nodes[node.uuid] is node

    def test_add_multiple_nodes(self):
        c = Circuit(name="test")
        n1 = self._make_node(layer_idx=0)
        n2 = self._make_node(layer_idx=1)
        c.add_node(n1)
        c.add_node(n2)
        assert len(c.nodes) == 2
        assert n1.uuid in c.nodes
        assert n2.uuid in c.nodes

    def test_add_edge_appends_to_edges(self):
        c = Circuit(name="test")
        n1 = c.add_node(self._make_node(layer_idx=0))
        n2 = c.add_node(self._make_node(layer_idx=1))
        edge = c.add_edge(n1.uuid, n2.uuid, weight=0.5)
        assert len(c.edges) == 1
        assert c.edges[0] is edge

    def test_add_edge_source_target_correct(self):
        c = Circuit(name="test")
        n1 = c.add_node(self._make_node())
        n2 = c.add_node(self._make_node())
        edge = c.add_edge(n1.uuid, n2.uuid)
        assert edge.source_uuid == n1.uuid
        assert edge.target_uuid == n2.uuid

    def test_add_edge_stores_metadata(self):
        c = Circuit(name="test")
        n1 = c.add_node(self._make_node())
        n2 = c.add_node(self._make_node())
        edge = c.add_edge(n1.uuid, n2.uuid, weight=0.99)
        assert edge.weight == pytest.approx(0.99)

    def test_add_edge_does_not_require_nodes_to_exist(self):
        """Edges can be added before or without corresponding nodes."""
        c = Circuit(name="test")
        edge = c.add_edge("phantom-src", "phantom-tgt", weight=1.0)
        assert edge.source_uuid == "phantom-src"
        assert edge.target_uuid == "phantom-tgt"

    def test_add_multiple_edges(self):
        c = Circuit(name="test")
        n1 = c.add_node(self._make_node())
        n2 = c.add_node(self._make_node())
        n3 = c.add_node(self._make_node())
        c.add_edge(n1.uuid, n2.uuid)
        c.add_edge(n2.uuid, n3.uuid)
        assert len(c.edges) == 2

    def test_circuit_name_stored(self):
        c = Circuit(name="my-circuit")
        assert c.name == "my-circuit"

    def test_circuit_metadata_stored(self):
        c = Circuit(name="test", metadata={"faithfulness": 0.85})
        assert c.metadata["faithfulness"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# CircuitStore
# ---------------------------------------------------------------------------

class TestCircuitStore:
    def test_circuits_defaults_to_empty_dict(self):
        store = CircuitStore()
        assert store.circuits == {}

    def test_add_circuit_stores_by_uuid(self):
        store = CircuitStore()
        c = Circuit(name="test")
        returned = store.add_circuit(c)
        assert returned is c
        assert store.circuits[c.uuid] is c

    def test_get_circuit_retrieves_by_uuid(self):
        store = CircuitStore()
        c = Circuit(name="test")
        store.add_circuit(c)
        assert store.get_circuit(c.uuid) is c

    def test_get_circuit_returns_none_for_unknown_uuid(self):
        store = CircuitStore()
        assert store.get_circuit("does-not-exist") is None

    def test_add_multiple_circuits(self):
        store = CircuitStore()
        c1 = Circuit(name="a")
        c2 = Circuit(name="b")
        store.add_circuit(c1)
        store.add_circuit(c2)
        assert len(store.circuits) == 2

    def test_save_load_roundtrip_preserves_circuit_uuids(self):
        store = CircuitStore()
        c = Circuit(name="roundtrip-test")
        store.add_circuit(c)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "circuits.pt")
            store.save(path)

            store2 = CircuitStore()
            store2.load(path)
            assert c.uuid in store2.circuits

    def test_save_load_roundtrip_preserves_node_uuids(self):
        store = CircuitStore()
        c = Circuit(name="node-test")
        node = c.add_node(CircuitNode(metadata={"layer_idx": 2, "kind": "attn", "latent_idx": 7}))
        store.add_circuit(c)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "circuits.pt")
            store.save(path)

            store2 = CircuitStore()
            store2.load(path)
            loaded_circuit = store2.circuits[c.uuid]
            assert node.uuid in loaded_circuit.nodes

    def test_save_load_roundtrip_preserves_node_metadata(self):
        store = CircuitStore()
        c = Circuit(name="meta-test")
        c.add_node(CircuitNode(uuid="fixed-node", metadata={"layer_idx": 5, "kind": "resid", "latent_idx": 100}))
        store.add_circuit(c)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "circuits.pt")
            store.save(path)

            store2 = CircuitStore()
            store2.load(path)
            loaded_node = store2.circuits[c.uuid].nodes["fixed-node"]
            assert loaded_node.metadata["layer_idx"] == 5
            assert loaded_node.metadata["kind"] == "resid"
            assert loaded_node.metadata["latent_idx"] == 100

    def test_save_load_roundtrip_preserves_edges(self):
        store = CircuitStore()
        c = Circuit(name="edge-test")
        n1 = c.add_node(CircuitNode())
        n2 = c.add_node(CircuitNode())
        c.add_edge(n1.uuid, n2.uuid, weight=0.42)
        store.add_circuit(c)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "circuits.pt")
            store.save(path)

            store2 = CircuitStore()
            store2.load(path)
            loaded = store2.circuits[c.uuid]
            assert len(loaded.edges) == 1
            assert loaded.edges[0].source_uuid == n1.uuid
            assert loaded.edges[0].target_uuid == n2.uuid
            assert loaded.edges[0].weight == pytest.approx(0.42)

    def test_save_creates_parent_directories(self):
        store = CircuitStore()
        store.add_circuit(Circuit(name="dir-test"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "deep", "circuits.pt")
            store.save(path)
            assert os.path.exists(path)

    def test_load_missing_path_is_graceful(self, capsys):
        """Loading a non-existent path should warn but not raise."""
        store = CircuitStore()
        store.load("/this/path/does/not/exist.pt")
        captured = capsys.readouterr()
        assert "Warning" in captured.out or store.circuits == {}
