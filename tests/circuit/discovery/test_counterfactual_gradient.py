"""
Tests for CounterfactualGradientDiscovery and its supporting components.

Covers Phase 5 validation targets:
  - compute_latent_counterfactual_scores: oracle tests with hand-computable expected
    values, sign contracts (activators > 0, inhibitors < 0), guard conditions,
    seed_layer boundary, top-k limits, and min_active_count filtering.
  - SeedProjectionInstrument: seed_pre_act populated after forward, correct shape
    [B, T], differentiability w.r.t. upstream leaf anchors.
  - CounterfactualGradientDiscovery: role assignment, score sign contracts, rejection
    on empty negctx and low faithfulness.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from circuit.instrument.attribution import compute_latent_counterfactual_scores
from circuit.instrument.sae_graph import FeatureGraph, SAEGraphInstrument
from circuit.discovery.counterfactual_gradient import (
    CounterfactualGradientDiscovery,
    SeedProjectionInstrument,
)
from circuit.types.sparse_act import SparseAct
from circuit.types.feature_id import FeatureID

# ---------------------------------------------------------------------------
# Shared constants (mirror conftest D_MODEL / D_SAE / KINDS)
# ---------------------------------------------------------------------------

B, T    = 2, 4
D_MODEL = 16
D_SAE   = 32
K_SAE   = 4
N_LAYERS = 2
KINDS    = ["attn", "mlp", "resid"]
N_COMP   = N_LAYERS * len(KINDS)

# Oracle graph uses its own tiny D_SAE to keep hand-computation tractable.
_ORACLE_D_SAE  = 3
_N_KINDS       = 3
_KINDS         = ["attn", "mlp", "resid"]
_ACTIVE_COUNT  = torch.ones(N_LAYERS * _N_KINDS, _ORACLE_D_SAE, dtype=torch.long)


# ---------------------------------------------------------------------------
# Oracle graph helpers
# ---------------------------------------------------------------------------

def _build_single_layer_oracle():
    """
    Graph with one upstream component: layer 0, kind 'attn', D_SAE=3.

    f0_vals = [2.0, 0.0, 1.0]  (B=1, T=1)

    target_scalar = f0_grad[0,0,0] - f0_grad[0,0,2]
      → d(target_scalar)/d(f0_grad[0,0,:]) = [+1.0, 0.0, -1.0]

    Activator scores  (raw grad > 0):
      latent 0: +1.0   ← absent activator
      latent 2: -1.0   → excluded (negative)

    Inhibitor scores  (acts * grad < 0):
      latent 0: 2.0 * (+1.0) = +2.0  → excluded (positive)
      latent 2: 1.0 * (-1.0) = -1.0  ← present inhibitor
    """
    f0_vals   = torch.tensor([[[2.0, 0.0, 1.0]]])            # [1, 1, 3]
    f0_grad   = f0_vals.detach().clone().requires_grad_(True)

    graph = FeatureGraph(torch.device("cpu"))
    graph.add(
        0, "attn",
        SparseAct(act=f0_grad),
        SparseAct(act=f0_vals.clone()),
        torch.tensor([[[0, 1, 2]]]),
    )

    target_scalar = f0_grad[0, 0, 0] - f0_grad[0, 0, 2]
    return graph, target_scalar


def _build_two_layer_oracle():
    """
    Graph with two layers so we can test the seed_layer boundary.

    Layer 0 'attn': f0_vals = [3.0, 0.0, 2.0]
      gradients when differentiating w.r.t. target_scalar below: [+1, 0, -1]
    Layer 1 'attn': f1_vals = [1.0, 0.0]  (connected to f0_grad via M)

    target_scalar = f0_grad[0,0,0] - f0_grad[0,0,2]
    """
    M = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -1.0]])
    f0_vals = torch.tensor([[[3.0, 0.0, 2.0]]])
    f0_grad = f0_vals.detach().clone().requires_grad_(True)

    f1_connected = f0_grad @ M                                # [1, 1, 2]
    f1_grad      = f1_connected.detach().clone().requires_grad_(True)

    graph = FeatureGraph(torch.device("cpu"))
    graph.add(
        0, "attn",
        SparseAct(act=f0_grad),
        SparseAct(act=f0_vals.clone()),
        torch.tensor([[[0, 1, 2]]]),
    )
    graph.add(
        1, "attn",
        SparseAct(act=f1_grad),
        SparseAct(act=f1_connected),
        torch.tensor([[[0, 1]]]),
    )

    target_scalar = f0_grad[0, 0, 0] - f0_grad[0, 0, 2]
    return graph, target_scalar


# ---------------------------------------------------------------------------
# TestComputeLatentCounterfactualScores
# ---------------------------------------------------------------------------

class TestComputeLatentCounterfactualScores:
    """Oracle and contract tests for compute_latent_counterfactual_scores."""

    def _call(self, graph, target_scalar, seed_layer=0,
              top_k_activators=10, top_k_inhibitors=10,
              min_active_count=1, active_count=None):
        if active_count is None:
            active_count = _ACTIVE_COUNT
        return compute_latent_counterfactual_scores(
            graph=graph,
            target_scalar=target_scalar,
            seed_layer=seed_layer,
            n_kinds=_N_KINDS,
            kinds=_KINDS,
            top_k_activators=top_k_activators,
            top_k_inhibitors=top_k_inhibitors,
            min_active_count=min_active_count,
            active_count=active_count,
        )

    # --- Sign contracts --------------------------------------------------------

    def test_activator_scores_are_positive(self):
        """Every score in activator_scores must be > 0."""
        graph, ts = _build_single_layer_oracle()
        activators, _ = self._call(graph, ts)
        for fid, score in activators.items():
            assert score > 0, f"{fid} has non-positive activator score {score}"

    def test_inhibitor_scores_are_negative(self):
        """Every score in inhibitor_scores must be < 0."""
        graph, ts = _build_single_layer_oracle()
        _, inhibitors = self._call(graph, ts)
        for fid, score in inhibitors.items():
            assert score < 0, f"{fid} has non-negative inhibitor score {score}"

    def test_activators_and_inhibitors_are_disjoint(self):
        """A latent cannot appear in both output dicts simultaneously."""
        graph, ts = _build_single_layer_oracle()
        activators, inhibitors = self._call(graph, ts)
        overlap = set(activators.keys()) & set(inhibitors.keys())
        assert overlap == set(), f"Overlap between activators and inhibitors: {overlap}"

    # --- Exact oracle values ---------------------------------------------------

    def test_exact_activator_latent_and_score(self):
        """
        Oracle: latent 0 gets raw gradient +1.0 → activator score 1.0.
        Latent 2 gets raw gradient -1.0 → not an activator.
        """
        graph, ts = _build_single_layer_oracle()
        activators, _ = self._call(graph, ts)

        assert FeatureID(0, "attn", 0) in activators
        assert activators[FeatureID(0, "attn", 0)] == pytest.approx(1.0, abs=1e-5)
        assert FeatureID(0, "attn", 2) not in activators  # negative gradient → excluded

    def test_exact_inhibitor_latent_and_score(self):
        """
        Oracle: latent 2 has acts=1.0, grad=-1.0 → acts*grad=-1.0 < 0 → inhibitor.
        Latent 0 has acts=2.0, grad=+1.0 → acts*grad=+2.0 > 0 → not an inhibitor.
        """
        graph, ts = _build_single_layer_oracle()
        _, inhibitors = self._call(graph, ts)

        assert FeatureID(0, "attn", 2) in inhibitors
        assert inhibitors[FeatureID(0, "attn", 2)] == pytest.approx(-1.0, abs=1e-5)
        assert FeatureID(0, "attn", 0) not in inhibitors  # positive product → excluded

    def test_zero_gradient_latent_excluded_from_both(self):
        """Latent 1 has gradient 0.0 and acts 0.0 → appears in neither dict."""
        graph, ts = _build_single_layer_oracle()
        activators, inhibitors = self._call(graph, ts)

        assert FeatureID(0, "attn", 1) not in activators
        assert FeatureID(0, "attn", 1) not in inhibitors

    # --- Guard: no grad_fn ----------------------------------------------------

    def test_no_grad_fn_returns_empty_dicts(self):
        """
        When target_scalar has no grad_fn (detached scalar), both dicts must be empty.
        """
        graph, _ = _build_single_layer_oracle()
        detached_scalar = torch.tensor(1.0)   # no grad_fn
        assert detached_scalar.grad_fn is None

        activators, inhibitors = self._call(graph, detached_scalar)
        assert activators == {}
        assert inhibitors == {}

    # --- seed_layer boundary --------------------------------------------------

    def test_seed_layer_includes_equal_and_lower_layers(self):
        """With seed_layer=0, layer 0 is scored; with seed_layer=1, both 0 and 1 are."""
        graph, ts = _build_two_layer_oracle()

        # seed_layer=0: only layer 0 can appear
        act_l0, inh_l0 = self._call(graph, ts, seed_layer=0)
        for fid in list(act_l0.keys()) + list(inh_l0.keys()):
            assert fid.layer <= 0, f"Layer {fid.layer} appeared with seed_layer=0"

        # seed_layer=1: layers 0 and 1 both eligible
        act_l1, inh_l1 = self._call(graph, ts, seed_layer=1)
        # Layer 0 activator must still appear when seed_layer=1
        assert FeatureID(0, "attn", 0) in act_l1

    def test_layer_above_seed_layer_is_excluded(self):
        """Entries from layer 1 must not appear when seed_layer=0."""
        graph, ts = _build_two_layer_oracle()
        activators, inhibitors = self._call(graph, ts, seed_layer=0)

        for fid in list(activators.keys()) + list(inhibitors.keys()):
            assert fid.layer <= 0

    # --- top_k limits ---------------------------------------------------------

    def test_top_k_activators_limit_respected(self):
        """With top_k_activators=1, at most 1 activator is returned."""
        graph, ts = _build_single_layer_oracle()
        activators, _ = self._call(graph, ts, top_k_activators=1)
        assert len(activators) <= 1

    def test_top_k_inhibitors_limit_respected(self):
        """With top_k_inhibitors=1, at most 1 inhibitor is returned."""
        graph, ts = _build_single_layer_oracle()
        _, inhibitors = self._call(graph, ts, top_k_inhibitors=1)
        assert len(inhibitors) <= 1

    def test_top_k_zero_returns_empty_for_that_type(self):
        """top_k_activators=0 must yield an empty activator dict (and vice-versa)."""
        graph, ts = _build_single_layer_oracle()

        act_zero, inh_ok = self._call(graph, ts, top_k_activators=0, top_k_inhibitors=10)
        assert act_zero == {}

        act_ok, inh_zero = self._call(graph, ts, top_k_activators=10, top_k_inhibitors=0)
        assert inh_zero == {}

    # --- min_active_count filtering -------------------------------------------

    def test_min_active_count_zero_blocks_low_count_latents(self):
        """
        Set active_count to 0 for all latents: min_active_count=1 must block everything.
        """
        graph, ts = _build_single_layer_oracle()
        dead_count = torch.zeros(N_LAYERS * _N_KINDS, _ORACLE_D_SAE, dtype=torch.long)

        activators, inhibitors = self._call(graph, ts, min_active_count=1,
                                            active_count=dead_count)
        assert activators == {}
        assert inhibitors == {}

    def test_min_active_count_passes_when_count_sufficient(self):
        """Active_count=100 for all latents → no filtering, same result as default."""
        graph, ts = _build_single_layer_oracle()
        full_count = torch.full((N_LAYERS * _N_KINDS, _ORACLE_D_SAE), 100, dtype=torch.long)

        act_full, inh_full = self._call(graph, ts, min_active_count=1, active_count=full_count)
        act_def,  inh_def  = self._call(graph, ts, min_active_count=1, active_count=_ACTIVE_COUNT)

        assert set(act_full.keys()) == set(act_def.keys())
        assert set(inh_full.keys()) == set(inh_def.keys())

    # --- Return types ---------------------------------------------------------

    def test_return_types(self):
        """Both outputs must be dicts mapping FeatureID → float."""
        graph, ts = _build_single_layer_oracle()
        activators, inhibitors = self._call(graph, ts)

        assert isinstance(activators, dict)
        assert isinstance(inhibitors, dict)
        for fid, score in {**activators, **inhibitors}.items():
            assert isinstance(fid, FeatureID)
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# TestSeedProjectionInstrument
# ---------------------------------------------------------------------------

class TestSeedProjectionInstrument:
    """Tests for SeedProjectionInstrument using conftest mock_model / mock_sae_bank."""

    def _run_forward(self, mock_sae_bank, mock_model, seed_layer=0, seed_kind="attn"):
        torch.manual_seed(0)
        w_seed = torch.randn(D_MODEL)
        b_seed = torch.zeros(())

        instrument = SeedProjectionInstrument(
            mock_sae_bank, seed_layer, seed_kind, w_seed, b_seed
        )
        x = torch.randn(B, T, D_MODEL, requires_grad=True)

        with torch.enable_grad():
            with instrument(mock_model):
                mock_model(x)

        return instrument, x, w_seed, b_seed

    def test_seed_pre_act_is_populated_after_forward(self, mock_sae_bank, mock_model):
        """seed_pre_act must not be None after a forward pass."""
        instrument, _, _, _ = self._run_forward(mock_sae_bank, mock_model)
        assert instrument.seed_pre_act is not None

    def test_seed_pre_act_shape(self, mock_sae_bank, mock_model):
        """seed_pre_act must have shape [B, T]."""
        instrument, _, _, _ = self._run_forward(mock_sae_bank, mock_model)
        assert instrument.seed_pre_act.shape == (B, T)

    def test_seed_pre_act_is_differentiable(self, mock_sae_bank, mock_model):
        """seed_pre_act must have a grad_fn (differentiable w.r.t. upstream anchors)."""
        instrument, _, _, _ = self._run_forward(mock_sae_bank, mock_model)
        assert instrument.seed_pre_act.grad_fn is not None

    def test_seed_pre_act_responds_to_w_seed(self, mock_sae_bank, mock_model):
        """
        Changing w_seed changes the seed_pre_act values — confirms x is being
        projected onto w_seed, not some other direction.
        """
        torch.manual_seed(0)
        w_a = torch.randn(D_MODEL)
        w_b = w_a.clone()
        w_b[0] += 10.0
        b_seed = torch.zeros(())
        x = torch.randn(B, T, D_MODEL, requires_grad=True)

        inst_a = SeedProjectionInstrument(mock_sae_bank, 0, "attn", w_a, b_seed)
        inst_b = SeedProjectionInstrument(mock_sae_bank, 0, "attn", w_b, b_seed)

        with torch.enable_grad():
            with inst_a(mock_model):
                mock_model(x)

        with torch.enable_grad():
            with inst_b(mock_model):
                mock_model(x)

        assert not torch.allclose(inst_a.seed_pre_act, inst_b.seed_pre_act), (
            "Changing w_seed should change seed_pre_act values"
        )

    def test_none_for_non_seed_layer_and_kind(self, mock_sae_bank, mock_model):
        """
        seed_pre_act must remain None for (layer, kind) combinations that don't
        match the specified seed_layer / seed_kind.
        Test: create instrument with seed_layer=99 (not in mock model) → None.
        """
        w_seed = torch.randn(D_MODEL)
        b_seed = torch.zeros(())
        instrument = SeedProjectionInstrument(
            mock_sae_bank, seed_layer=99, seed_kind="attn", w_seed=w_seed, b_seed=b_seed
        )
        x = torch.randn(B, T, D_MODEL)

        with torch.enable_grad():
            with instrument(mock_model):
                mock_model(x)

        assert instrument.seed_pre_act is None

    def test_gradient_flows_to_upstream_anchors(self, mock_sae_bank, mock_model):
        """
        A scalar derived from seed_pre_act must propagate gradients to the graph's
        leaf anchors — confirming the computation graph is connected end-to-end.

        seed_layer=1 is required: with seed_layer=0 there are no upstream layers,
        so no upstream leaf anchors exist for gradients to flow to.  Using layer 1
        gives us layer 0's anchors as upstream targets.
        """
        instrument, x, w_seed, _ = self._run_forward(mock_sae_bank, mock_model,
                                                      seed_layer=1)

        target = instrument.seed_pre_act.sum()
        assert target.grad_fn is not None

        anchors = instrument.graph.all_anchors()
        assert len(anchors) > 0, "No leaf anchors found in graph"

        grads = torch.autograd.grad(
            target, anchors, retain_graph=True, allow_unused=True
        )
        non_none_grads = [g for g in grads if g is not None]
        assert len(non_none_grads) > 0, (
            "No gradients flowed from seed_pre_act to any upstream leaf anchor"
        )


# ---------------------------------------------------------------------------
# TestCounterfactualGradientDiscovery  (mock-based integration tests)
# ---------------------------------------------------------------------------

class MockSAEBankForDiscovery:
    """Minimal SAEBank mock with the interface used by CounterfactualGradientDiscovery."""

    def __init__(self):
        self.kinds = KINDS
        self.d_sae = D_SAE
        self.n_layer = N_LAYERS
        self.device = torch.device("cpu")
        self.layer_device_map = {l: torch.device("cpu") for l in range(N_LAYERS)}

        # Mock SAE modules with .encoder.weight and ._get_bias_eff()
        class _MockSAEModule:
            def __init__(self):
                self.encoder = MagicMock()
                self.encoder.weight = torch.randn(D_SAE, D_MODEL)
                self._bias_eff = torch.zeros(D_SAE)

            def _get_bias_eff(self):
                return self._bias_eff

        self.saes = {
            kind: [_MockSAEModule() for _ in range(N_LAYERS)]
            for kind in KINDS
        }

    def encode(self, x, kind, layer):
        B_T = x.shape[:-1]
        top_acts = torch.zeros(*B_T, K_SAE)
        top_indices = torch.zeros(*B_T, K_SAE, dtype=torch.long)
        return top_acts, top_indices

    def decode(self, latents, kind, layer):
        return torch.zeros(*latents.shape[:-1], D_MODEL)


def _make_discovery(min_faithfulness=0.0, activator_threshold=0.0,
                    inhibitor_threshold=0.0, max_neg_sequences=4,
                    top_k_activators=8, top_k_inhibitors=8):
    """Helper: build a CounterfactualGradientDiscovery with fully mocked infrastructure."""
    inference    = MagicMock()
    inference._compiled = False
    sae_bank     = MockSAEBankForDiscovery()
    avg_acts     = torch.zeros(N_COMP, D_SAE)
    probe_builder = MagicMock()

    return CounterfactualGradientDiscovery(
        inference, sae_bank, avg_acts, probe_builder,
        top_k_activators=top_k_activators,
        top_k_inhibitors=top_k_inhibitors,
        activator_threshold=activator_threshold,
        inhibitor_threshold=inhibitor_threshold,
        min_active_count=1,
        max_neg_sequences=max_neg_sequences,
        pruning_threshold=0.0,
        min_faithfulness=min_faithfulness,
    )


def _make_probe_data(n_pos=4, n_neg=4, seq_len=8):
    """Build a minimal ProbeDataset-like object."""
    data = MagicMock()
    data.pos_tokens  = torch.zeros(n_pos, seq_len, dtype=torch.long)
    data.neg_tokens  = torch.zeros(n_neg, seq_len, dtype=torch.long)
    data.pos_argmax  = torch.zeros(n_pos, dtype=torch.long)
    data.target_tokens = torch.zeros(n_pos, seq_len, dtype=torch.long)
    return data


@pytest.fixture
def discovery_patched():
    """
    Returns a factory that builds a CounterfactualGradientDiscovery with
    latent_stats and CircuitLogger.save both patched out.
    """
    def _factory(**kwargs):
        d = _make_discovery(**kwargs)
        return d
    return _factory


class TestCounterfactualGradientDiscovery:

    def _run_discover(
        self,
        algo,
        probe_data,
        activator_fids_and_scores,
        inhibitor_fids_and_scores,
        up_faith=0.8,
        faith=0.8, suff=0.8, comp=0.8,
    ):
        """
        Runs `algo.discover(seed_comp=3, seed_latent=0)` with all heavy
        dependencies mocked out.

        - probe_data is returned by build_probe_dataset.
        - _get_posctx_activation returns 1.0.
        - _run_negctx_hop returns (activator_fids_and_scores, inhibitor_fids_and_scores).
        - All eval functions return the supplied floats.
        - latent_stats.active_count always passes min_active_count=1.
        - CircuitLogger.save is patched to prevent disk I/O.
        """
        SEED_COMP, SEED_LAT = 3, 0  # layer 1, kind "resid"

        algo.build_probe_dataset = MagicMock(return_value=probe_data)

        mock_active_count = torch.full((N_COMP, D_SAE), 100, dtype=torch.long)

        with patch("circuit.discovery.counterfactual_gradient.latent_stats") as mock_ls, \
             patch("circuit.discovery.counterfactual_gradient.evaluate_upstream_faithfulness",
                   return_value=up_faith), \
             patch("circuit.discovery.counterfactual_gradient.evaluate_faithfulness",
                   return_value=faith), \
             patch("circuit.discovery.counterfactual_gradient.evaluate_sufficiency",
                   return_value=suff), \
             patch("circuit.discovery.counterfactual_gradient.evaluate_completeness",
                   return_value=comp), \
             patch("circuit.discovery.counterfactual_gradient.prune_non_minimal_nodes"), \
             patch("observability.circuit_logger.CircuitLogger.save"):

            mock_ls.active_count = mock_active_count

            algo._get_posctx_activation = MagicMock(return_value=1.0)
            algo._run_negctx_hop = MagicMock(
                return_value=(activator_fids_and_scores, inhibitor_fids_and_scores)
            )

            circuit = algo.discover(SEED_COMP, SEED_LAT)

        return circuit

    # -----------------------------------------------------------------------

    def test_rejects_empty_neg_tokens(self):
        """When negctx is empty, discover must return None."""
        algo = _make_discovery()
        probe_data = _make_probe_data(n_neg=0)

        circuit = self._run_discover(algo, probe_data, {}, {})
        assert circuit is None

    def test_rejects_empty_pos_tokens(self):
        """When posctx is empty, discover must return None."""
        algo = _make_discovery()
        probe_data = _make_probe_data(n_pos=0)

        circuit = self._run_discover(algo, probe_data, {}, {})
        assert circuit is None

    def test_rejects_no_activators_or_inhibitors(self):
        """Empty score dicts from _run_negctx_hop → circuit has only seed → rejected."""
        algo = _make_discovery()
        probe_data = _make_probe_data()

        circuit = self._run_discover(algo, probe_data, activator_fids_and_scores={},
                                     inhibitor_fids_and_scores={})
        assert circuit is None

    def test_rejects_below_upstream_faithfulness_threshold(self):
        """Circuit with good nodes but low upstream_faithfulness is rejected."""
        algo = _make_discovery(min_faithfulness=0.5)
        probe_data = _make_probe_data()

        fid = FeatureID(0, "attn", 5)
        circuit = self._run_discover(
            algo, probe_data,
            activator_fids_and_scores={fid: 0.9},
            inhibitor_fids_and_scores={},
            up_faith=0.1,   # below threshold
        )
        assert circuit is None

    def test_activator_node_role_assigned_correctly(self):
        """Nodes from activator_scores must carry role='counterfactual_activator'."""
        algo = _make_discovery(min_faithfulness=0.0)
        probe_data = _make_probe_data()

        fid = FeatureID(0, "attn", 7)
        circuit = self._run_discover(
            algo, probe_data,
            activator_fids_and_scores={fid: 0.5},
            inhibitor_fids_and_scores={},
        )

        assert circuit is not None
        roles = {n.metadata.get("role") for n in circuit.nodes.values()}
        assert "counterfactual_activator" in roles

    def test_inhibitor_node_role_assigned_correctly(self):
        """Nodes from inhibitor_scores must carry role='counterfactual_inhibitor'."""
        algo = _make_discovery(min_faithfulness=0.0)
        probe_data = _make_probe_data()

        fid = FeatureID(0, "mlp", 3)
        circuit = self._run_discover(
            algo, probe_data,
            activator_fids_and_scores={},
            inhibitor_fids_and_scores={fid: -0.7},
        )

        assert circuit is not None
        roles = {n.metadata.get("role") for n in circuit.nodes.values()}
        assert "counterfactual_inhibitor" in roles

    def test_both_role_types_present_when_both_returned(self):
        """Circuit contains both activator and inhibitor nodes when both are returned."""
        algo = _make_discovery(min_faithfulness=0.0)
        probe_data = _make_probe_data()

        act_fid = FeatureID(0, "attn", 2)
        inh_fid = FeatureID(0, "mlp",  9)
        circuit = self._run_discover(
            algo, probe_data,
            activator_fids_and_scores={act_fid: 0.4},
            inhibitor_fids_and_scores={inh_fid: -0.3},
        )

        assert circuit is not None
        roles = {n.metadata.get("role") for n in circuit.nodes.values()}
        assert "counterfactual_activator" in roles
        assert "counterfactual_inhibitor" in roles

    def test_activator_score_positive_in_metadata(self):
        """attribution_score in activator node metadata must be positive."""
        algo = _make_discovery(min_faithfulness=0.0)
        probe_data = _make_probe_data()

        fid = FeatureID(0, "resid", 1)
        circuit = self._run_discover(
            algo, probe_data,
            activator_fids_and_scores={fid: 0.6},
            inhibitor_fids_and_scores={},
        )

        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata.get("role") == "counterfactual_activator":
                assert node.metadata["attribution_score"] > 0

    def test_inhibitor_score_negative_in_metadata(self):
        """attribution_score in inhibitor node metadata must be negative."""
        algo = _make_discovery(min_faithfulness=0.0)
        probe_data = _make_probe_data()

        fid = FeatureID(1, "attn", 4)
        circuit = self._run_discover(
            algo, probe_data,
            activator_fids_and_scores={},
            inhibitor_fids_and_scores={fid: -0.5},
        )

        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata.get("role") == "counterfactual_inhibitor":
                assert node.metadata["attribution_score"] < 0

    def test_accepted_circuit_metadata_fields(self):
        """Accepted circuit must include all expected metadata keys."""
        algo = _make_discovery(min_faithfulness=0.0)
        probe_data = _make_probe_data()

        fid = FeatureID(0, "attn", 10)
        circuit = self._run_discover(
            algo, probe_data,
            activator_fids_and_scores={fid: 0.3},
            inhibitor_fids_and_scores={},
            up_faith=0.9, faith=0.8, suff=0.7, comp=0.6,
        )

        assert circuit is not None
        meta = circuit.metadata
        for key in ("faithfulness", "sufficiency", "completeness",
                    "upstream_faithfulness", "seed_comp", "seed_latent",
                    "n_nodes", "n_edges", "discovery_method"):
            assert key in meta, f"Missing metadata key: {key}"

        assert meta["discovery_method"] == "counterfactual_gradient"
        assert meta["upstream_faithfulness"] == pytest.approx(0.9)

    def test_threshold_filtering_blocks_low_score_nodes(self):
        """Nodes with |score| < threshold must not appear in the circuit."""
        algo = _make_discovery(
            min_faithfulness=0.0,
            activator_threshold=0.5,
            inhibitor_threshold=0.5,
        )
        probe_data = _make_probe_data()

        act_fid = FeatureID(0, "attn", 0)
        inh_fid = FeatureID(0, "mlp",  1)
        circuit = self._run_discover(
            algo, probe_data,
            activator_fids_and_scores={act_fid: 0.1},   # below threshold
            inhibitor_fids_and_scores={inh_fid: -0.1},  # below threshold
        )

        # All scored nodes filtered out → only seed → circuit rejected
        assert circuit is None
