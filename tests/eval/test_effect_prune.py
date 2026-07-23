"""
Tests for the effect-threshold prune (eval/effect_prune.py) — the SFC-inspired
fixed-cut stopping rule (keep iff |score| > T). No forward passes are involved,
so these run entirely on real Circuit structures.

Fixture: member i carries |attribution_score| = n - i, so thresholds map
directly onto survivor counts.
"""

import pytest

from circuit.types.feature_id import FeatureID
from eval.effect_prune import prune_by_effect_threshold
from store.circuits import Circuit, CircuitNode


def _build_circuit(n_members=6, score_key="attribution_score"):
    c = Circuit(name="t")
    seed = c.add_node(CircuitNode(metadata={
        "feature_id": FeatureID(5, "mlp", 999), "role": "seed"}))
    uuids = []
    for i in range(n_members):
        node = c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(0, "mlp", i),
            "role": "ablation_support",
            score_key: float(n_members - i),
        }))
        uuids.append(node.uuid)
        c.add_edge(node.uuid, seed.uuid, weight=1.0)
    return c, seed, uuids


class TestAbsThreshold:
    def test_drops_members_at_or_below_threshold(self):
        # Scores 6,5,4,3,2,1; T=3 keeps STRICTLY above (SFC's rule is >, not >=)
        c, seed, uuids = _build_circuit(6)
        removed = prune_by_effect_threshold(c, threshold=3.0)
        assert set(removed) == {uuids[3], uuids[4], uuids[5]}
        assert seed.uuid in c.nodes
        assert len(c.nodes) == 4                     # 3 survivors + seed

    def test_edges_of_removed_members_are_dropped(self):
        c, seed, uuids = _build_circuit(4)
        prune_by_effect_threshold(c, threshold=3.0)
        endpoints = {e.source_uuid for e in c.edges} | {e.target_uuid for e in c.edges}
        assert endpoints <= set(c.nodes)

    def test_threshold_above_all_scores_falls_back_to_min_keep(self):
        c, _, uuids = _build_circuit(6)
        removed = prune_by_effect_threshold(c, threshold=100.0, min_keep=2)
        # top-2 by |score| survive despite the cut
        assert uuids[0] in c.nodes and uuids[1] in c.nodes
        assert len(removed) == 4

    def test_all_above_threshold_removes_nothing(self):
        c, _, _ = _build_circuit(6)
        removed = prune_by_effect_threshold(c, threshold=0.5)
        assert removed == []
        assert len(c.nodes) == 7

    def test_tiny_circuit_is_left_alone(self):
        c, _, _ = _build_circuit(1)
        assert prune_by_effect_threshold(c, threshold=100.0, min_keep=1) == []
        assert "n_members_pre_effect_prune" not in c.metadata


class TestPctlThreshold:
    def test_pctl_keeps_top_fraction(self):
        # 100 members, p90 cut -> ~top 10% survive
        c, _, uuids = _build_circuit(100)
        prune_by_effect_threshold(c, threshold=90.0, threshold_mode="pctl")
        survivors = [u for u in uuids if u in c.nodes]
        assert survivors == uuids[:len(survivors)]   # a strict top-prefix
        assert 5 <= len(survivors) <= 15

    def test_invalid_mode_raises(self):
        c, _, _ = _build_circuit(4)
        with pytest.raises(ValueError, match="threshold_mode"):
            prune_by_effect_threshold(c, threshold=1.0, threshold_mode="topk")


class TestMetadataAndChaining:
    def test_stamps_are_namespaced_and_quantiles_logged(self):
        c, _, _ = _build_circuit(6)
        # simulate a prior magnitude-bisection prune's stamps
        c.metadata["n_members_pre_prune"] = 999
        c.metadata["prune_phi_base"] = 0.5
        prune_by_effect_threshold(c, threshold=3.0)
        assert c.metadata["n_members_pre_effect_prune"] == 6
        assert c.metadata["effect_prune_threshold"] == 3.0
        assert c.metadata["effect_prune_mode"] == "abs"
        assert c.metadata["effect_prune_score_q"]["max"] == 6.0
        # the bisection prune's stamps are untouched
        assert c.metadata["n_members_pre_prune"] == 999
        assert c.metadata["prune_phi_base"] == 0.5

    def test_score_fallback_chain(self):
        # effect_score preferred over attribution_score; weight as last resort
        c = Circuit(name="t")
        seed = c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(5, "mlp", 999), "role": "seed"}))
        hi = c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(0, "mlp", 0),
            "effect_score": 10.0, "attribution_score": 0.001}))
        lo = c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(0, "mlp", 1), "weight": 0.01}))
        removed = prune_by_effect_threshold(c, threshold=5.0)
        assert hi.uuid in c.nodes                    # judged on effect_score
        assert removed == [lo.uuid]

    def test_negative_scores_judged_by_magnitude(self):
        c = Circuit(name="t")
        c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(5, "mlp", 999), "role": "seed"}))
        neg = c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(0, "mlp", 0), "attribution_score": -8.0}))
        pos = c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(0, "mlp", 1), "attribution_score": 2.0}))
        removed = prune_by_effect_threshold(c, threshold=5.0)
        assert neg.uuid in c.nodes                   # |-8| > 5 survives
        assert removed == [pos.uuid]
