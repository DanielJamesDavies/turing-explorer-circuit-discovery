from analysis.circuits.circuit_motifs import compute_circuit_motifs
from circuit.types.feature_id import FeatureID
from store.circuits import Circuit, CircuitNode


def _base_circuit(name: str, *, seed_idx: int = 0, faithfulness: float = 1.0) -> tuple[Circuit, CircuitNode]:
    circuit = Circuit(name=name)
    circuit.metadata = {
        "seed_comp": 0,
        "seed_latent": seed_idx,
        "evals": {
            "counterfactual_faithfulness": faithfulness,
            "posctx_suppression_score": 0.25,
        },
        "post_analysis": {
            "internode_coact_density_pct": 12.5,
            "node_presence_pct_activators": 50.0,
        },
    }
    seed = circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(0, "attn", seed_idx), "role": "seed"}))
    return circuit, seed


def _add_node(circuit: Circuit, layer: int, latent_idx: int, role: str, *, kind: str = "attn") -> CircuitNode:
    return circuit.add_node(CircuitNode(metadata={"feature_id": FeatureID(layer, kind, latent_idx), "role": role}))


def test_compute_circuit_motifs_counts_shared_exact_edge_to_seed():
    circuits = {}
    for idx in range(2):
        circuit, seed = _base_circuit(f"circuit-{idx}", faithfulness=0.8 + idx)
        activator = _add_node(circuit, 1, 10, "counterfactual_activator")
        circuit.add_edge(activator.uuid, seed.uuid, weight=1.5)
        circuits[circuit.uuid] = circuit

    stats = compute_circuit_motifs(circuits, min_support=2, kinds=("attn",))

    edge_rows = [
        row
        for row in stats["motif_rows"]
        if row["motif_level"] == "exact" and row["motif_kind"] == "edge_to_seed"
    ]
    assert edge_rows
    assert edge_rows[0]["support_count"] == 2
    assert edge_rows[0]["mean_faithfulness"] == 1.3
    assert len(stats["membership_rows"]) >= 2


def test_compute_circuit_motifs_keeps_exact_and_typed_signatures_separate():
    first, first_seed = _base_circuit("first")
    first_activator = _add_node(first, 1, 10, "counterfactual_activator")
    first.add_edge(first_activator.uuid, first_seed.uuid, weight=1.0)

    second, second_seed = _base_circuit("second")
    second_activator = _add_node(second, 1, 11, "counterfactual_activator")
    second.add_edge(second_activator.uuid, second_seed.uuid, weight=1.0)

    stats = compute_circuit_motifs({first.uuid: first, second.uuid: second}, min_support=2, kinds=("attn",))

    typed_rows = [
        row
        for row in stats["motif_rows"]
        if row["motif_level"] == "typed" and row["motif_kind"] == "edge_to_seed"
    ]
    exact_rows = [
        row
        for row in stats["motif_rows"]
        if row["motif_level"] == "exact" and row["motif_kind"] == "edge_to_seed"
    ]
    assert typed_rows
    assert not exact_rows


def test_compute_circuit_motifs_canonicalizes_fan_in_ordering():
    first, first_seed = _base_circuit("first")
    first_a = _add_node(first, 1, 10, "counterfactual_activator")
    first_b = _add_node(first, 2, 20, "counterfactual_inhibitor")
    first.add_edge(first_a.uuid, first_seed.uuid, weight=1.0)
    first.add_edge(first_b.uuid, first_seed.uuid, weight=-1.0)

    second, second_seed = _base_circuit("second")
    second_b = _add_node(second, 2, 20, "counterfactual_inhibitor")
    second_a = _add_node(second, 1, 10, "counterfactual_activator")
    second.add_edge(second_b.uuid, second_seed.uuid, weight=-1.0)
    second.add_edge(second_a.uuid, second_seed.uuid, weight=1.0)

    stats = compute_circuit_motifs({first.uuid: first, second.uuid: second}, min_support=2, kinds=("attn",))

    rows = [
        row
        for row in stats["motif_rows"]
        if row["motif_level"] == "exact" and row["motif_kind"] == "fan_in_seed"
    ]
    assert len(rows) == 1
    assert rows[0]["support_count"] == 2


def test_compute_circuit_motifs_projects_shared_motifs_to_family_rows():
    first, first_seed = _base_circuit("first")
    first_a = _add_node(first, 1, 10, "counterfactual_activator")
    first.add_edge(first_a.uuid, first_seed.uuid, weight=1.0)

    second, second_seed = _base_circuit("second")
    second_a = _add_node(second, 1, 10, "counterfactual_activator")
    second.add_edge(second_a.uuid, second_seed.uuid, weight=1.0)

    isolated, _isolated_seed = _base_circuit("isolated")

    stats = compute_circuit_motifs(
        {first.uuid: first, second.uuid: second, isolated.uuid: isolated},
        min_support=2,
        similarity_threshold=0.0,
        kinds=("attn",),
    )

    family_by_uuid = {row["uuid"]: row for row in stats["family_rows"]}
    assert family_by_uuid[first.uuid]["hard_family_id"] == family_by_uuid[second.uuid]["hard_family_id"]
    assert family_by_uuid[first.uuid]["hard_family_size"] == 2
    assert family_by_uuid[isolated.uuid]["hard_family_size"] == 1
    assert family_by_uuid[first.uuid]["fuzzy_membership_weight"] == 1.0


def test_compute_circuit_motifs_emits_cohesion_rows_with_post_metrics():
    circuit, seed = _base_circuit("cohesion")
    activator = _add_node(circuit, 1, 10, "counterfactual_activator")
    inhibitor = _add_node(circuit, 2, 20, "counterfactual_inhibitor")
    circuit.add_edge(activator.uuid, seed.uuid, weight=2.0)
    circuit.add_edge(inhibitor.uuid, seed.uuid, weight=-1.0)

    stats = compute_circuit_motifs({circuit.uuid: circuit}, min_support=1, kinds=("attn",))

    row = stats["cohesion_rows"][0]
    assert row["counterfactual_faithfulness"] == 1.0
    assert row["posctx_suppression_score"] == 0.25
    assert row["internode_coact_density_pct"] == 12.5
    assert row["node_presence_pct_activators"] == 50.0
    assert row["role_purity_score"] == 100.0
    assert row["motif_coverage_pct"] > 0.0
