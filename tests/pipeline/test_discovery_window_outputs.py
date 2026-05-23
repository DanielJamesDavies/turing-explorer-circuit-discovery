import json

from circuit.discovery_window import DiscoveryWindow
from store.circuits import Circuit, CircuitStore, circuit_store


def test_attach_candidate_metadata_preserves_circuit_and_adds_provenance():
    circuit = Circuit(name="candidate-circuit")
    candidate = {
        "comp_idx": 4,
        "latent_idx": 12,
        "candidate_index": 7,
        "run_id": "20260517-002500-abcdef12",
        "worker_id": 3,
        "config_hash": "abcdef1234567890",
        "artifact_hashes": {"top_ctx": "a" * 64},
        "criteria_scores": {"connectivity": 1.0},
    }

    DiscoveryWindow._attach_candidate_metadata(circuit, candidate)

    assert circuit.uuid
    assert circuit.metadata["seed_comp"] == 4
    assert circuit.metadata["seed_latent"] == 12
    assert circuit.metadata["candidate_index"] == 7
    assert circuit.metadata["run_id"] == "20260517-002500-abcdef12"
    assert circuit.metadata["worker_id"] == 3
    assert circuit.metadata["config_hash"] == "abcdef1234567890"
    assert circuit.metadata["artifact_hashes"] == {"top_ctx": "a" * 64}


def test_discovery_window_save_store_round_trips_atomically(monkeypatch, tmp_path):
    circuit_store.circuits.clear()
    try:
        circuit = Circuit(name="worker-circuit")
        circuit.metadata.update(
            {
                "run_id": "20260517-002500-abcdef12",
                "worker_id": 0,
                "candidate_index": 1,
                "seed_comp": 2,
                "seed_latent": 3,
                "discovery_method": "mock_method",
            }
        )
        circuit_store.add_circuit(circuit)
        window = DiscoveryWindow.__new__(DiscoveryWindow)
        window.output_dir = str(tmp_path)
        monkeypatch.setattr(DiscoveryWindow, "_save_summary_xlsx", lambda _self: None)

        DiscoveryWindow.save_store(window)

        loaded = CircuitStore()
        loaded.load(str(tmp_path / "discovered_circuits.pt"))
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    finally:
        circuit_store.circuits.clear()

    assert list(loaded.circuits) == [circuit.uuid]
    assert loaded.circuits[circuit.uuid].metadata["worker_id"] == 0
    assert summary[0]["metadata"]["candidate_index"] == 1
    assert not (tmp_path / "discovered_circuits.pt.tmp").exists()
    assert not (tmp_path / "summary.json.tmp").exists()
