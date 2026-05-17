"""
tests/circuit/discovery/test_layerwise_gradient_upstream.py

Tests for LayerwiseGradientUpstreamDiscovery and the supporting
get_all_upstream_components helper.

Coverage targets (Phase 5 plan):
  - get_all_upstream_components: correct comp indices for several (layer, kind)
    scenarios, min_layer limiting, include_same_layer flag, deduplication.
  - LayerwiseGradientUpstreamDiscovery._discover: role assignment, score-sign
    contracts, rejection cases (empty probe / seed-only / low faithfulness),
    heap termination, and — crucially — that nodes from non-adjacent layers
    can be discovered directly (the key property absent from GradientUpstreamDiscovery).

The heavy infrastructure (_run_node, eval functions, latent_stats, CircuitLogger)
is mocked out so the test suite runs without a real model or GPU.
"""

import pytest
import torch
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from pipeline.component_index import (
    get_all_upstream_components,
    component_idx,
)
from circuit.discovery.layerwise_gradient_upstream import LayerwiseGradientUpstreamDiscovery
from circuit.instrument.attribution import UpstreamScores
from circuit.types.feature_id import FeatureID


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B, T    = 2, 4
D_MODEL = 16
D_SAE   = 32
K_SAE   = 4
N_LAYERS = 2
KINDS    = ["attn", "mlp", "resid"]
N_KINDS  = len(KINDS)
N_COMP   = N_LAYERS * N_KINDS

# Default seed: layer=1, kind="attn" → comp_idx = 1*3 + 0 = 3
SEED_COMP = 3
SEED_LAT  = 0


# ---------------------------------------------------------------------------
# TestGetAllUpstreamComponents
# ---------------------------------------------------------------------------

class TestGetAllUpstreamComponents:
    """Unit tests for get_all_upstream_components in component_index.py."""

    def test_layer_0_attn_returns_empty(self):
        """attn@L0 has no preceding layers and no same-layer preds → empty list."""
        comp = component_idx(0, 0, N_KINDS)  # (layer=0, kind_idx=0 "attn")
        result = get_all_upstream_components(comp, N_KINDS, KINDS)
        assert result == []

    def test_layer_0_resid_only_same_layer(self):
        """resid@L0 has no preceding layers; include_same_layer=True adds attn@L0, mlp@L0."""
        comp = component_idx(0, 2, N_KINDS)  # (layer=0, kind_idx=2 "resid")
        result = get_all_upstream_components(comp, N_KINDS, KINDS, include_same_layer=True)
        # Only same-layer causal preds: attn@L0 and mlp@L0
        attn_l0 = component_idx(0, 0, N_KINDS)
        mlp_l0  = component_idx(0, 1, N_KINDS)
        assert attn_l0 in result
        assert mlp_l0  in result
        # No layer -1 components (would be negative)
        assert all(c >= 0 for c in result)

    def test_resid_layer1_contains_all_kinds_at_layer0(self):
        """resid@L1 must include attn@L0, mlp@L0, resid@L0 as strictly-preceding."""
        comp = component_idx(1, 2, N_KINDS)  # resid@L1
        result = get_all_upstream_components(comp, N_KINDS, KINDS, include_same_layer=False)
        for kind_idx in range(N_KINDS):
            expected = component_idx(0, kind_idx, N_KINDS)
            assert expected in result, f"Expected {KINDS[kind_idx]}@L0 in result"

    def test_include_same_layer_true_adds_causal_preds(self):
        """resid@L1 with include_same_layer=True adds attn@L1 and mlp@L1."""
        comp = component_idx(1, 2, N_KINDS)  # resid@L1
        result = get_all_upstream_components(comp, N_KINDS, KINDS, include_same_layer=True)
        attn_l1 = component_idx(1, 0, N_KINDS)
        mlp_l1  = component_idx(1, 1, N_KINDS)
        assert attn_l1 in result
        assert mlp_l1  in result

    def test_include_same_layer_false_excludes_same_layer(self):
        """With include_same_layer=False, same-layer components are absent."""
        comp = component_idx(1, 2, N_KINDS)  # resid@L1
        result = get_all_upstream_components(comp, N_KINDS, KINDS, include_same_layer=False)
        attn_l1 = component_idx(1, 0, N_KINDS)
        mlp_l1  = component_idx(1, 1, N_KINDS)
        assert attn_l1 not in result
        assert mlp_l1  not in result

    def test_min_layer_limits_depth(self):
        """With min_layer=1 for a target at layer=3, layer 0 must be absent."""
        n_kinds = 3
        kinds   = ["attn", "mlp", "resid"]
        comp    = component_idx(3, 2, n_kinds)  # resid@L3 in a 4-layer model
        result  = get_all_upstream_components(comp, n_kinds, kinds, min_layer=1,
                                              include_same_layer=False)
        # Only L1 and L2 expected
        for kind_idx in range(n_kinds):
            assert component_idx(0, kind_idx, n_kinds) not in result, \
                f"Layer 0 comp appeared with min_layer=1"
            assert component_idx(1, kind_idx, n_kinds) in result
            assert component_idx(2, kind_idx, n_kinds) in result

    def test_no_duplicates_in_result(self):
        """The returned list must not contain duplicate component indices."""
        comp = component_idx(1, 2, N_KINDS)  # resid@L1
        result = get_all_upstream_components(comp, N_KINDS, KINDS, include_same_layer=True)
        assert len(result) == len(set(result)), "Duplicate component indices found"

    def test_no_self_reference(self):
        """The target component itself must never appear in the upstream list."""
        for layer in range(N_LAYERS):
            for kind_idx in range(N_KINDS):
                comp = component_idx(layer, kind_idx, N_KINDS)
                result = get_all_upstream_components(comp, N_KINDS, KINDS, include_same_layer=True)
                assert comp not in result, \
                    f"Self-reference: comp {comp} appeared in its own upstream list"

    def test_result_count_for_multi_layer(self):
        """
        attn@L2 (no same-layer preds, include_same_layer=False):
          expected = 2 layers × 3 kinds = 6 components.
        """
        n_kinds = 3
        kinds   = ["attn", "mlp", "resid"]
        comp    = component_idx(2, 0, n_kinds)  # attn@L2 in a 3-layer model
        result  = get_all_upstream_components(comp, n_kinds, kinds, include_same_layer=False)
        assert len(result) == 6, f"Expected 6, got {len(result)}: {result}"


# ---------------------------------------------------------------------------
# Helpers for LayerwiseGradientUpstreamDiscovery tests
# ---------------------------------------------------------------------------

class _MockSAEBankForDiscovery:
    """Minimal SAEBank with the interface used by LayerwiseGradientUpstreamDiscovery."""

    def __init__(self, n_layers=N_LAYERS):
        self.kinds = KINDS
        self.d_sae = D_SAE
        self.n_layer = n_layers
        self.device = torch.device("cpu")
        self.layer_device_map = {l: torch.device("cpu") for l in range(n_layers)}
        self.saes = {
            kind: [MagicMock() for _ in range(n_layers)]
            for kind in KINDS
        }

    def encode(self, x, kind, layer):
        shape = x.shape[:-1]
        return torch.zeros(*shape, K_SAE), torch.zeros(*shape, K_SAE, dtype=torch.long)

    def decode(self, latents, kind, layer):
        return torch.zeros(*latents.shape[:-1], D_MODEL)

    def pin_decoders(self):
        return nullcontext()


def _make_probe_data(n_pos=4, seq_len=T):
    data = MagicMock()
    data.pos_tokens    = torch.zeros(n_pos, seq_len, dtype=torch.long)
    data.neg_tokens    = torch.zeros(n_pos, seq_len, dtype=torch.long)
    data.pos_argmax    = torch.zeros(n_pos, dtype=torch.long)
    data.target_tokens = torch.zeros(n_pos, seq_len, dtype=torch.long)
    return data


def _make_algo(n_layers=N_LAYERS, min_faithfulness=0.0, attribution_threshold=0.0,
               max_layers_back=0, absent_inhibitor_top_k=0):
    sae_bank = _MockSAEBankForDiscovery(n_layers=n_layers)
    n_comp   = n_layers * N_KINDS
    avg_acts = torch.zeros(n_comp, D_SAE)
    return LayerwiseGradientUpstreamDiscovery(
        inference=MagicMock(),
        sae_bank=sae_bank,
        avg_acts=avg_acts,
        probe_builder=MagicMock(),
        top_k_per_node=8,
        attribution_threshold=attribution_threshold,
        min_active_count=1,
        max_ctx_sequences=4,
        hop_batch_size=4,
        absent_inhibitor_top_k=absent_inhibitor_top_k,
        absent_inhibitor_threshold=0.01,
        max_layers_back=max_layers_back,
        include_same_layer=True,
        pruning_threshold=0.0,
        min_faithfulness=min_faithfulness,
    )


def _run_discover(
    algo,
    probe_data,
    run_node_side_effect,
    seed_comp=SEED_COMP,
    seed_lat=SEED_LAT,
    up_faith=0.8,
    faith=0.8,
    suff=0.8,
    comp=0.8,
):
    """
    Runs algo.discover(seed_comp, seed_lat) with all heavy dependencies mocked.

    run_node_side_effect: callable or list passed to MagicMock(side_effect=...).
      Each call corresponds to one _run_node invocation from _discover.
    """
    n_comp   = algo.sae_bank.n_layer * N_KINDS
    mock_active_count = torch.full((n_comp, D_SAE), 100, dtype=torch.long)

    algo.build_probe_dataset = MagicMock(return_value=probe_data)

    _mod = "circuit.discovery.layerwise_gradient_upstream"
    with patch(f"{_mod}.latent_stats") as mock_ls, \
         patch(f"{_mod}.evaluate_upstream_faithfulness", return_value=up_faith), \
         patch(f"{_mod}.evaluate_faithfulness",          return_value=faith), \
         patch(f"{_mod}.evaluate_sufficiency",           return_value=suff), \
         patch(f"{_mod}.evaluate_completeness",          return_value=comp), \
         patch(f"{_mod}.prune_non_minimal_nodes"), \
         patch("observability.circuit_logger.CircuitLogger.save"):

        mock_ls.active_count = mock_active_count
        algo._run_node = MagicMock(side_effect=run_node_side_effect)

        circuit = algo.discover(seed_comp, seed_lat)

    return circuit


# Convenience: UpstreamScores with a single activator
def _activator_scores(fid: FeatureID, score: float = 0.5) -> UpstreamScores:
    return UpstreamScores(attribution={fid: score}, absent_gradient={})


def _inhibitor_scores(fid: FeatureID, score: float = -0.5) -> UpstreamScores:
    return UpstreamScores(attribution={fid: score}, absent_gradient={})


def _absent_scores(fid: FeatureID, score: float = -0.5) -> UpstreamScores:
    return UpstreamScores(attribution={}, absent_gradient={fid: score})


def _empty_scores() -> UpstreamScores:
    return UpstreamScores(attribution={}, absent_gradient={})


# ---------------------------------------------------------------------------
# TestLayerwiseGradientUpstreamRejections
# ---------------------------------------------------------------------------

class TestLayerwiseGradientUpstreamRejections:

    def test_rejects_empty_probe_dataset(self):
        """discover must return None when pos_tokens is empty."""
        algo  = _make_algo()
        empty = _make_probe_data(n_pos=0)
        algo.build_probe_dataset = MagicMock(return_value=empty)

        with patch("circuit.discovery.layerwise_gradient_upstream.latent_stats"), \
             patch("observability.circuit_logger.CircuitLogger.save"):
            result = algo.discover(SEED_COMP, SEED_LAT)

        assert result is None

    def test_rejects_seed_only_circuit(self):
        """When _run_node returns no upstream nodes, circuit has only the seed → None."""
        algo = _make_algo()
        probe = _make_probe_data()
        # _run_node always returns empty → no nodes added → circuit has only seed
        circuit = _run_discover(algo, probe, run_node_side_effect=[_empty_scores()])
        assert circuit is None

    def test_rejects_below_upstream_faithfulness(self):
        """Circuit is rejected when upstream_faithfulness < min_faithfulness."""
        algo  = _make_algo(min_faithfulness=0.5)
        probe = _make_probe_data()
        fid   = FeatureID(0, "attn", 1)

        # First call (seed): returns activator; second call (activator node): returns empty
        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_activator_scores(fid), _empty_scores()],
            up_faith=0.1,   # below 0.5
        )
        assert circuit is None

    def test_attribution_threshold_filters_low_score_nodes(self):
        """Nodes with |score| < attribution_threshold must not be added."""
        algo  = _make_algo(min_faithfulness=0.0, attribution_threshold=1.0)
        probe = _make_probe_data()
        fid   = FeatureID(0, "attn", 2)

        # Score 0.1 is below threshold of 1.0 → node filtered → seed-only → None
        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[
                UpstreamScores(attribution={fid: 0.1}, absent_gradient={}),
            ],
        )
        assert circuit is None


# ---------------------------------------------------------------------------
# TestLayerwiseGradientUpstreamNodeRoles
# ---------------------------------------------------------------------------

class TestLayerwiseGradientUpstreamNodeRoles:

    def test_seed_node_has_seed_role(self):
        """The seed FeatureID must be present with role='seed'."""
        algo  = _make_algo()
        probe = _make_probe_data()
        fid   = FeatureID(0, "attn", 5)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_activator_scores(fid), _empty_scores()],
        )

        assert circuit is not None
        seed_nodes = [n for n in circuit.nodes.values() if n.metadata["role"] == "seed"]
        assert len(seed_nodes) == 1
        assert seed_nodes[0].feature_id == FeatureID(1, "attn", SEED_LAT)

    def test_positive_score_gets_activator_role(self):
        """Upstream nodes with score > 0 receive role='activator'."""
        algo  = _make_algo()
        probe = _make_probe_data()
        fid   = FeatureID(0, "mlp", 3)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_activator_scores(fid, score=0.7), _empty_scores()],
        )

        assert circuit is not None
        activators = [n for n in circuit.nodes.values() if n.metadata["role"] == "activator"]
        assert any(n.feature_id == fid for n in activators)

    def test_negative_score_gets_active_inhibitor_role(self):
        """Upstream nodes with score < 0 receive role='active_inhibitor'."""
        algo  = _make_algo()
        probe = _make_probe_data()
        fid   = FeatureID(0, "resid", 7)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_inhibitor_scores(fid, score=-0.4)],
        )

        assert circuit is not None
        inhibitors = [n for n in circuit.nodes.values()
                      if n.metadata["role"] == "active_inhibitor"]
        assert any(n.feature_id == fid for n in inhibitors)

    def test_absent_inhibitor_role_when_enabled(self):
        """With absent_inhibitor_top_k > 0, absent_gradient entries become 'absent_inhibitor'."""
        algo  = _make_algo(absent_inhibitor_top_k=4)
        probe = _make_probe_data()
        fid   = FeatureID(0, "attn", 9)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_absent_scores(fid, score=-0.6)],
        )

        assert circuit is not None
        absent = [n for n in circuit.nodes.values()
                  if n.metadata["role"] == "absent_inhibitor"]
        assert any(n.feature_id == fid for n in absent)

    def test_activator_attribution_score_positive_in_metadata(self):
        """attribution_score stored on an activator node must be > 0."""
        algo  = _make_algo()
        probe = _make_probe_data()
        fid   = FeatureID(0, "attn", 4)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_activator_scores(fid, score=0.9), _empty_scores()],
        )

        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata["role"] == "activator":
                assert node.metadata["attribution_score"] > 0


# ---------------------------------------------------------------------------
# TestLayerwiseGradientUpstreamCircuitStructure
# ---------------------------------------------------------------------------

class TestLayerwiseGradientUpstreamCircuitStructure:

    def test_no_duplicate_nodes(self):
        """Each (layer, kind, index) triple must appear at most once in the circuit."""
        algo  = _make_algo()
        probe = _make_probe_data()
        fid   = FeatureID(0, "attn", 2)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_activator_scores(fid), _empty_scores()],
        )

        assert circuit is not None
        seen: set = set()
        for node in circuit.nodes.values():
            key = (node.feature_id.layer, node.feature_id.kind, node.feature_id.index)
            assert key not in seen, f"Duplicate node: {key}"
            seen.add(key)

    def test_edges_connect_upstream_to_downstream(self):
        """Every edge must go from an upstream (lower-layer) node to the seed/downstream node."""
        algo  = _make_algo()
        probe = _make_probe_data()
        fid   = FeatureID(0, "resid", 6)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_activator_scores(fid), _empty_scores()],
        )

        assert circuit is not None
        # Build uuid → node map
        uuid_to_node = circuit.nodes
        for edge in circuit.edges:
            src = uuid_to_node[edge.source_uuid]
            tgt = uuid_to_node[edge.target_uuid]
            assert src.feature_id.layer <= tgt.feature_id.layer, (
                f"Edge goes from higher layer {src.feature_id.layer} "
                f"to lower layer {tgt.feature_id.layer}"
            )

    def test_active_inhibitors_not_expanded(self):
        """Active inhibitors must not be added to the work-queue (no further expansion)."""
        algo  = _make_algo()
        probe = _make_probe_data()

        # Return an active inhibitor (negative score) from the seed
        inh_fid = FeatureID(0, "attn", 1)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_inhibitor_scores(inh_fid, score=-0.5)],
        )

        assert circuit is not None
        # _run_node should have been called exactly once (seed only — inhibitor not enqueued)
        assert algo._run_node.call_count == 1

    def test_activators_trigger_further_expansion(self):
        """Activator nodes must be enqueued and their _run_node must be called."""
        algo  = _make_algo()
        probe = _make_probe_data()
        fid   = FeatureID(0, "mlp", 8)

        # First call (seed): returns activator → enqueues (0, "mlp")
        # Second call (activator): returns empty → work-queue drains
        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_activator_scores(fid), _empty_scores()],
        )

        assert circuit is not None
        # _run_node called twice: once for seed, once for the discovered activator
        assert algo._run_node.call_count == 2

    def test_accepted_circuit_metadata_keys(self):
        """Accepted circuit must contain all required metadata keys."""
        algo  = _make_algo()
        probe = _make_probe_data()
        fid   = FeatureID(0, "attn", 3)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[_activator_scores(fid), _empty_scores()],
            up_faith=0.9, faith=0.85, suff=0.75, comp=0.65,
        )

        assert circuit is not None
        meta = circuit.metadata
        required = {
            "faithfulness", "sufficiency", "completeness",
            "upstream_faithfulness", "seed_comp", "seed_latent",
            "n_nodes", "n_edges", "discovery_method",
            "max_layers_back", "top_k_per_node",
        }
        assert required.issubset(meta.keys()), (
            f"Missing keys: {required - set(meta.keys())}"
        )
        assert meta["discovery_method"]       == "layerwise_gradient_upstream"
        assert meta["seed_comp"]              == SEED_COMP
        assert meta["seed_latent"]            == SEED_LAT
        assert meta["n_nodes"]                == len(circuit.nodes)
        assert meta["n_edges"]                == len(circuit.edges)
        assert meta["upstream_faithfulness"]  == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TestNonAdjacentLayerDiscovery
# ---------------------------------------------------------------------------

class TestNonAdjacentLayerDiscovery:
    """
    Key correctness test: verifies that the algorithm can discover nodes from
    non-adjacent layers (e.g. layer 0 from a seed at layer 2), which is the
    primary feature that distinguishes this method from GradientUpstreamDiscovery.

    A 3-layer model is used so the seed can sit at layer 2 and discover
    a node at layer 0 (two hops back).
    """

    def _setup_3layer(self):
        n_layers = 3
        algo  = _make_algo(n_layers=n_layers, min_faithfulness=0.0)
        probe = _make_probe_data()
        # Seed at layer 2, kind "attn" → comp_idx = 2*3 + 0 = 6
        seed_comp = component_idx(2, 0, N_KINDS)
        seed_lat  = 0
        return algo, probe, seed_comp, seed_lat

    def test_discovers_node_two_layers_back(self):
        """
        Seed at (2, attn). _run_node returns an activator at (0, attn, 7).
        That node should be present in the circuit, demonstrating that attribution
        reached across all upstream layers rather than just the direct predecessor.
        """
        algo, probe, seed_comp, seed_lat = self._setup_3layer()
        non_adj_fid = FeatureID(0, "attn", 7)   # 2 layers back from seed@L2

        # Call sequence: seed → discovers non_adj_fid; non_adj_fid → empty
        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[
                _activator_scores(non_adj_fid, score=0.6),
                _empty_scores(),
            ],
            seed_comp=seed_comp,
            seed_lat=seed_lat,
        )

        assert circuit is not None
        fids_in_circuit = {n.feature_id for n in circuit.nodes.values()}
        assert non_adj_fid in fids_in_circuit, (
            f"Non-adjacent node {non_adj_fid} was not discovered from seed at layer 2"
        )

    def test_non_adjacent_activator_is_further_expanded(self):
        """
        The non-adjacent activator at (0, attn, 7) must be enqueued for expansion,
        proving the heap correctly schedules it for a second _run_node call.
        """
        algo, probe, seed_comp, seed_lat = self._setup_3layer()
        non_adj_fid = FeatureID(0, "attn", 7)

        _run_discover(
            algo, probe,
            run_node_side_effect=[
                _activator_scores(non_adj_fid, score=0.6),
                _empty_scores(),  # second call for the non-adjacent activator
            ],
            seed_comp=seed_comp,
            seed_lat=seed_lat,
        )

        # Two _run_node calls: one for seed, one for the non-adjacent activator
        assert algo._run_node.call_count == 2

    def test_max_layers_back_zero_reaches_layer_0(self):
        """
        max_layers_back=0 means no limit — all layers back to 0 are in scope.
        Verify the discovered node at layer 0 is not filtered out.
        """
        algo, probe, seed_comp, seed_lat = self._setup_3layer()
        far_fid = FeatureID(0, "resid", 11)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[
                _activator_scores(far_fid, score=0.4),
                _empty_scores(),
            ],
            seed_comp=seed_comp,
            seed_lat=seed_lat,
        )

        assert circuit is not None
        fids = {n.feature_id for n in circuit.nodes.values()}
        assert far_fid in fids

    def test_max_layers_back_1_excludes_layer_0(self):
        """
        max_layers_back=1 from seed at layer 2 gives effective_min_layer=1.
        A node discovered at layer 0 must not be enqueued (below min_layer).
        """
        n_layers = 3
        algo  = _make_algo(n_layers=n_layers, min_faithfulness=0.0, max_layers_back=1)
        probe = _make_probe_data()
        seed_comp = component_idx(2, 0, N_KINDS)
        seed_lat  = 0

        layer0_fid = FeatureID(0, "attn", 5)

        circuit = _run_discover(
            algo, probe,
            run_node_side_effect=[
                _activator_scores(layer0_fid, score=0.5),
            ],
            seed_comp=seed_comp,
            seed_lat=seed_lat,
        )

        # The node at layer 0 IS added to the circuit (attribution found it),
        # but it must NOT be enqueued for expansion — so _run_node is called only once.
        assert algo._run_node.call_count == 1
