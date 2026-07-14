"""
Tests for the global magnitude-bisection prune (eval/magnitude_prune.py).

The prune ranks non-seed members by |attribution_score| and keeps the smallest
top-K prefix whose free-φ meets a floor, found by binary search over K. We drive
a synthetic monotone φ(K) by monkeypatching the two eval calls, and check the
prune against the real Circuit data structures.

φ(K) := min(1, K/6):  φ(5)=0.833, φ(6)=1.0, base φ(10)=1.0.
"""

import pytest

from eval import magnitude_prune as mp
from store.circuits import Circuit, CircuitNode
from circuit.types.feature_id import FeatureID


def _build_circuit(n_members=10):
    """Seed + n members with scores n, n-1, ..., 1 (strongest first has index 0)."""
    c = Circuit(name="t")
    seed = c.add_node(CircuitNode(metadata={"feature_id": FeatureID(5, "mlp", 999), "role": "seed"}))
    uuids = []
    for i in range(n_members):
        node = c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(i % 5, "mlp", i),  # distinct latent index i
            "role": "counterfactual_activator",
            "attribution_score": float(n_members - i),  # 10,9,...,1
        }))
        uuids.append(node.uuid)
        c.add_edge(node.uuid, seed.uuid, weight=float(n_members - i))
    return c, seed, uuids


@pytest.fixture(autouse=True)
def patch_eval(monkeypatch):
    # a_posctx = 1.0; a_empty (K=0) = 0.0; a_C(K) = min(1, K/6) -> denom 1.0
    monkeypatch.setattr(mp, "measure_seed_activation", lambda *a, **k: 1.0)
    monkeypatch.setattr(mp, "upstream_sites", lambda *a, **k: {(0, "mlp"), (1, "mlp")})

    def fake_activation(inference, bank, keep_indices, in_scope, *a, **k):
        K = sum(len(v) for v in keep_indices.values())
        return min(1.0, K / 6.0)

    monkeypatch.setattr(mp, "circuit_only_activation", fake_activation)


def _call(circuit, **kw):
    return mp.prune_by_magnitude_bisection(
        inference=None, sae_bank=None, circuit=circuit,
        pos_tokens=None, seed_layer=5, seed_kind="mlp", seed_latent_idx=999,
        **kw,
    )


# ---------------------------------------------------------------- tolerance mode
def test_tolerance_keeps_smallest_prefix_above_floor():
    c, seed, uuids = _build_circuit(10)
    removed = _call(c, tolerance=0.05)  # floor = base(1.0) - 0.05 = 0.95 -> need K=6
    assert len(removed) == 4                     # dropped the 4 weakest
    assert set(removed) == set(uuids[6:])        # weakest by |score|
    assert seed.uuid in c.nodes                  # seed never pruned
    assert len(c.nodes) == 7                     # seed + 6 kept


def test_removed_edges_are_dropped():
    c, seed, uuids = _build_circuit(10)
    removed = _call(c, tolerance=0.05)
    rm = set(removed)
    for e in c.edges:
        assert e.source_uuid not in rm and e.target_uuid not in rm
    assert len(c.edges) == 6                      # only kept members' edges remain


# ---------------------------------------------------------------- target mode
def test_absolute_target_floor():
    c, seed, uuids = _build_circuit(10)
    removed = _call(c, target=0.5)  # floor 0.5 -> smallest K with K/6 >= 0.5 is K=3
    assert len(c.nodes) == 4                       # seed + 3
    assert set(removed) == set(uuids[3:])


def test_target_above_base_keeps_all():
    c, seed, uuids = _build_circuit(10)
    removed = _call(c, target=2.0)  # unreachable -> keep everything
    assert removed == []
    assert len(c.nodes) == 11


# ---------------------------------------------------------------- guards
def test_min_keep_respected():
    c, seed, uuids = _build_circuit(10)
    removed = _call(c, tolerance=0.05, min_keep=8)  # knee is 6, but floor at 8
    assert len(removed) == 2
    assert len(c.nodes) == 9                          # seed + 8


def test_noop_when_at_or_below_min_keep():
    c, seed, uuids = _build_circuit(3)
    removed = _call(c, tolerance=0.05, min_keep=3)
    assert removed == []
    assert len(c.nodes) == 4


def test_bad_objective_raises():
    c, seed, uuids = _build_circuit(10)
    with pytest.raises(ValueError):
        _call(c, tolerance=0.05, objective="both")


def test_pinned_objective_collects_pins_and_prunes(monkeypatch):
    # objective="pinned" pulls position-specific pins and passes them through; the
    # fake activation ignores pins (so the K-curve is unchanged) but the pinned path
    # must run end-to-end and prune to the same knee.
    called = {"n": 0}
    def fake_anchors(*a, **k):
        called["n"] += 1
        return {}, {(0, "mlp"): None, (1, "mlp"): None}
    monkeypatch.setattr("eval.floors.collect_site_anchors", fake_anchors)
    c, seed, uuids = _build_circuit(10)
    removed = _call(c, tolerance=0.05, objective="pinned")
    assert called["n"] == 1               # pins collected exactly once
    assert len(removed) == 4              # same knee as the free objective here


def test_config_objective_validation():
    from config import DiscoveryConfig
    assert DiscoveryConfig(magnitude_prune_objective="pinned").magnitude_prune_objective == "pinned"
    with pytest.raises(ValueError):
        DiscoveryConfig(magnitude_prune_objective="drivers")


def test_result_provably_meets_floor():
    # Whatever K is chosen, the kept circuit's φ must be >= floor (correctness of
    # the *result*, independent of monotonicity).
    c, seed, uuids = _build_circuit(10)
    _call(c, tolerance=0.05)
    kept_members = sum(1 for u, n in c.nodes.items() if n.metadata.get("role") != "seed")
    assert min(1.0, kept_members / 6.0) >= 0.95
