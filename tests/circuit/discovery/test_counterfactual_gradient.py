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


def _build_two_site_oracle():
    """
    Two upstream sites at layer 0, both carrying REAL gradients (unlike the
    two-layer oracle, whose layer-1 anchor is detached and so scores None).
    Used to test posctx_values coverage across sites.

      layer 0 'attn': vals [2.0, 0.0, 1.0], grads [+1, 0, -1]
      layer 0 'mlp' : vals [1.0, 3.0, 0.0], grads [ 0, +1,  0]
    """
    f_attn_vals = torch.tensor([[[2.0, 0.0, 1.0]]])
    f_attn = f_attn_vals.detach().clone().requires_grad_(True)
    f_mlp_vals = torch.tensor([[[1.0, 3.0, 0.0]]])
    f_mlp = f_mlp_vals.detach().clone().requires_grad_(True)

    graph = FeatureGraph(torch.device("cpu"))
    graph.add(
        0, "attn",
        SparseAct(act=f_attn), SparseAct(act=f_attn_vals.clone()),
        torch.tensor([[[0, 1, 2]]]),
    )
    graph.add(
        0, "mlp",
        SparseAct(act=f_mlp), SparseAct(act=f_mlp_vals.clone()),
        torch.tensor([[[0, 1, 2]]]),
    )

    target_scalar = f_attn[0, 0, 0] - f_attn[0, 0, 2] + f_mlp[0, 0, 1]
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
# TestActivatorSignalPosctxScaling
# ---------------------------------------------------------------------------

class TestActivatorSignalPosctxScaling:
    """`posctx_values` switches the activator signal from the raw gradient
    (the seed's per-unit sensitivity) to grad x posctx target — the first-order
    effect of the injection that counterfactual faithfulness actually performs.
    Selected by config's activator_signal="gradient_x_posctx".

    Single-layer oracle: vals [2.0, 0.0, 1.0], grads [+1.0, 0.0, -1.0].
    """

    def _call(self, graph, target_scalar, posctx_values=None,
              position_aware=None, seed_layer=0):
        return compute_latent_counterfactual_scores(
            graph=graph,
            target_scalar=target_scalar,
            seed_layer=seed_layer,
            n_kinds=_N_KINDS,
            kinds=_KINDS,
            top_k_activators=10,
            top_k_inhibitors=10,
            min_active_count=1,
            active_count=_ACTIVE_COUNT,
            position_aware=position_aware,
            posctx_values=posctx_values,
        )

    # --- The scaling itself ----------------------------------------------------

    def test_activator_score_is_gradient_times_posctx_value(self):
        """latent 0: grad +1.0 x posctx 3.0 = 3.0 (raw-gradient signal gives 1.0)."""
        graph, ts = _build_single_layer_oracle()
        pv = {(0, "attn"): torch.tensor([3.0, 5.0, 7.0])}
        activators, _ = self._call(graph, ts, posctx_values=pv)

        assert activators[FeatureID(0, "attn", 0)] == pytest.approx(3.0, abs=1e-5)

    def test_none_posctx_values_leaves_the_classic_signal_untouched(self):
        """The default path must stay bit-identical — every existing result
        depends on it."""
        graph, ts = _build_single_layer_oracle()
        activators, _ = self._call(graph, ts, posctx_values=None)

        assert activators[FeatureID(0, "attn", 0)] == pytest.approx(1.0, abs=1e-5)

    def test_zero_posctx_value_drops_a_high_gradient_latent(self):
        """The correction that matters: the seed may be highly sensitive to a
        latent, but if that latent has no posctx value to inject, the eval can
        never cash the sensitivity in — so it must not be selected."""
        graph, ts = _build_single_layer_oracle()
        pv = {(0, "attn"): torch.tensor([0.0, 5.0, 7.0])}   # latent 0 absent on posctx

        scaled, _ = self._call(graph, ts, posctx_values=pv)
        raw, _ = self._call(graph, ts)

        assert FeatureID(0, "attn", 0) not in scaled
        assert FeatureID(0, "attn", 0) in raw, "raw-gradient signal should still pick it"

    def test_activator_sign_contract_survives_scaling(self):
        """Scaling by non-negative posctx values must not smuggle in negatives."""
        graph, ts = _build_single_layer_oracle()
        pv = {(0, "attn"): torch.tensor([3.0, 5.0, 7.0])}
        activators, _ = self._call(graph, ts, posctx_values=pv)

        for fid, score in activators.items():
            assert score > 0, f"{fid} has non-positive activator score {score}"

    def test_inhibitors_are_unaffected_by_posctx_values(self):
        """acts x grad is already an effect rather than a sensitivity, so the
        knob is activator-only."""
        graph, ts = _build_single_layer_oracle()
        pv = {(0, "attn"): torch.tensor([3.0, 5.0, 7.0])}

        _, scaled = self._call(graph, ts, posctx_values=pv)
        _, raw = self._call(graph, ts)

        assert scaled == raw
        assert scaled[FeatureID(0, "attn", 2)] == pytest.approx(-1.0, abs=1e-5)

    # --- Site coverage ---------------------------------------------------------

    def test_each_site_is_scaled_by_its_own_values(self):
        graph, ts = _build_two_site_oracle()
        pv = {
            (0, "attn"): torch.tensor([3.0, 5.0, 7.0]),
            (0, "mlp"):  torch.tensor([2.0, 4.0, 6.0]),
        }
        activators, _ = self._call(graph, ts, posctx_values=pv)

        assert activators[FeatureID(0, "attn", 0)] == pytest.approx(3.0, abs=1e-5)  # +1 x 3
        assert activators[FeatureID(0, "mlp", 1)] == pytest.approx(4.0, abs=1e-5)   # +1 x 4

    def test_missing_scored_site_raises_rather_than_mixing_scales(self):
        """Falling back to the raw gradient for an uncovered site would put two
        incommensurable scales into one global ranking — fail loudly instead."""
        graph, ts = _build_two_site_oracle()
        pv = {(0, "attn"): torch.tensor([3.0, 5.0, 7.0])}   # (0, "mlp") missing

        with pytest.raises(KeyError, match="missing scored site"):
            self._call(graph, ts, posctx_values=pv)

    # --- Reaches both reductions ----------------------------------------------

    def test_position_aware_branch_also_scales(self):
        """A toggle that silently applied to only one of the two position
        reductions would be a footgun."""
        from circuit.instrument.position_aware import PositionAwareSpec

        graph, ts = _build_single_layer_oracle()
        spec = PositionAwareSpec(peaks=torch.zeros(1, dtype=torch.long), top_n=3)
        pv = {(0, "attn"): torch.tensor([3.0, 5.0, 7.0])}

        scaled, _ = self._call(graph, ts, posctx_values=pv, position_aware=spec)
        raw, _ = self._call(graph, ts, position_aware=spec)

        assert scaled[FeatureID(0, "attn", 0)] == pytest.approx(3.0, abs=1e-5)
        assert raw[FeatureID(0, "attn", 0)] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# TestContrastiveIgMode
# ---------------------------------------------------------------------------

class TestContrastiveIgMode:
    """Contract tests for attribution_mode="ig_negctx": config gating
    (cf-only), dispatch, and the negctx anchor helper. The path arithmetic and
    completeness live in tests/circuit/test_ig_baseline.py."""

    def test_config_accepts_ig_negctx_for_cf_only(self):
        """The first attribution mode the two gradient methods do NOT share:
        cf accepts it; ablation (no contrast input) must reject it."""
        from config import AblationGradientConfig, CounterfactualGradientConfig

        assert CounterfactualGradientConfig(
            attribution_mode="ig_negctx"
        ).attribution_mode == "ig_negctx"
        with pytest.raises(ValueError, match="attribution_mode"):
            AblationGradientConfig(attribution_mode="ig_negctx")

    def test_config_objective_validator(self):
        from config import CounterfactualGradientConfig

        assert CounterfactualGradientConfig().ig_negctx_objective == "drive"
        assert CounterfactualGradientConfig(
            ig_negctx_objective="gap"
        ).ig_negctx_objective == "gap"
        with pytest.raises(ValueError, match="ig_negctx_objective"):
            CounterfactualGradientConfig(ig_negctx_objective="banana")

    def test_discover_dispatches_to_contrastive_hop(self):
        """attribution_mode="ig_negctx" must route to the contrastive hop
        and NOT the local contrast hop, and the circuit must carry the roles."""
        algo = _make_discovery(min_faithfulness=0.0)
        algo.attribution_mode = "ig_negctx"
        probe_data = _make_probe_data()
        fid = FeatureID(0, "attn", 5)

        algo.build_probe_dataset = MagicMock(return_value=probe_data)
        mock_active_count = torch.full((N_COMP, D_SAE), 100, dtype=torch.long)
        with patch("circuit.discovery.counterfactual_gradient.latent_stats") as mock_ls, \
             patch("circuit.discovery.counterfactual_gradient.evaluate_counterfactual_faithfulness",
                   return_value=(0.9, 0.25)), \
             patch("observability.circuit_logger.CircuitLogger.save"):
            mock_ls.active_count = mock_active_count
            algo._get_posctx_activation = MagicMock(return_value=1.0)
            algo._get_neg_tokens = MagicMock(return_value=probe_data.neg_tokens)
            algo._run_ig_negctx_hop = MagicMock(return_value=({fid: 0.7}, {}))
            algo._run_contrast_hop = MagicMock()

            circuit = algo.discover(3, 0)

        algo._run_ig_negctx_hop.assert_called_once()
        algo._run_contrast_hop.assert_not_called()
        assert circuit is not None
        roles = {n.metadata.get("role") for n in circuit.nodes.values()}
        assert "counterfactual_activator" in roles

    def test_negctx_anchor_returns_preact_argmax(self):
        """The anchor must be the seed's would-be-firing position — the
        pre-activation argmax per sequence — from a no-grad forward."""
        algo = _make_discovery()

        pre_act = torch.tensor([[0.1, 3.0, 0.2, -1.0],
                                [2.0, 0.0, 0.0, 5.0]])  # argmax -> [1, 3]

        def fake_forward(tokens, patcher=None, **kwargs):
            patcher.seed_pre_act = pre_act

        algo.inference.forward = MagicMock(side_effect=fake_forward)
        anchor = algo._negctx_anchor(1, "resid", 0, torch.zeros(2, 4, dtype=torch.long))

        assert torch.equal(anchor, torch.tensor([1, 3]))
        assert algo.inference.forward.call_args.kwargs["grad_enabled"] is False

    def test_negctx_anchor_raises_when_capture_fails(self):
        algo = _make_discovery()
        algo.inference.forward = MagicMock()  # never sets seed_pre_act

        with pytest.raises(RuntimeError, match="not captured"):
            algo._negctx_anchor(1, "resid", 0, torch.zeros(2, 4, dtype=torch.long))

    def test_capture_patcher_matches_graph_instrument(self, mock_sae_bank, mock_model):
        """SeedPreActCapture must produce the same seed pre-activation (and
        therefore the same anchor argmax) as the graph instrument it replaced
        — without densifying any site. The graph instrument built two dense
        [B, T, d_sae] copies at EVERY upstream site (~15.4GB at full width on
        a deep seed) as a side effect of this capture."""
        from circuit.discovery.counterfactual_gradient import SeedPreActCapture

        torch.manual_seed(0)
        w_seed = torch.randn(D_MODEL)
        b_seed = torch.tensor(0.25)
        x = torch.randn(B, T, D_MODEL)

        graph_inst = SeedProjectionInstrument(mock_sae_bank, 1, "attn", w_seed, b_seed)
        capture = SeedPreActCapture(1, "attn", w_seed, b_seed)
        with torch.enable_grad():
            with graph_inst(mock_model):
                mock_model(x)
        with torch.no_grad():
            with capture(mock_model):
                mock_model(x)

        assert capture.seed_pre_act is not None
        assert capture.seed_pre_act.shape == (B, T)
        # Same values up to the graph instrument's reassociation rounding.
        assert torch.allclose(capture.seed_pre_act, graph_inst.seed_pre_act,
                              atol=1e-4, rtol=1e-4)
        assert torch.equal(capture.seed_pre_act.argmax(dim=-1),
                           graph_inst.seed_pre_act.argmax(dim=-1))

    def test_ig_negctx_batch_is_depth_adaptive(self):
        """Deep seeds (> threshold upstream sites) drop the neg microbatch —
        per-site residency is batch-proportional and held across all sites,
        so B=8 crosses a 16GB card near 29 sites; shallow seeds keep the
        configured batch and pay no extra chunk overhead."""
        algo = _make_discovery()
        algo.neg_batch_size = 8
        algo.ig_negctx_deep_site_threshold = 25
        algo.ig_negctx_deep_neg_batch = 4

        assert algo._ig_negctx_batch(10) == 8    # shallow: unchanged
        assert algo._ig_negctx_batch(25) == 8    # at threshold: unchanged
        assert algo._ig_negctx_batch(26) == 4    # deep: halved
        assert algo._ig_negctx_batch(35) == 4

        # Never RAISE the batch: if the configured batch is already smaller,
        # keep it.
        algo.neg_batch_size = 2
        assert algo._ig_negctx_batch(35) == 2

    def test_ig_negctx_batch_config_defaults(self):
        from config import CounterfactualGradientConfig

        cfg = CounterfactualGradientConfig()
        assert cfg.ig_negctx_deep_site_threshold == 21
        assert cfg.ig_negctx_deep_neg_batch == 4
        with pytest.raises(ValueError):
            CounterfactualGradientConfig(ig_negctx_deep_neg_batch=0)

    def test_capture_patcher_leaves_stream_untouched(self, mock_sae_bank):
        from circuit.discovery.counterfactual_gradient import SeedPreActCapture

        capture = SeedPreActCapture(1, "attn", torch.randn(D_MODEL), torch.tensor(0.0))
        x = torch.randn(B, T, D_MODEL)
        out = capture.transform(0, "mlp", x)   # non-seed site: identity, no capture
        assert out is x
        assert capture.seed_pre_act is None
        out_seed = capture.transform(1, "attn", x)
        assert out_seed is x                   # seed site: capture, still identity
        assert capture.seed_pre_act is not None


# ---------------------------------------------------------------------------
# TestCollectPosctxValues
# ---------------------------------------------------------------------------

class TestCollectPosctxValues:
    """Gating contract for the activator_signal toggle: under the default
    signal the collector must not run at all — no extra forward pass, and the
    scorer receives None, so the classic behaviour is bit-identical."""

    def _algo(self, activator_signal):
        algo = _make_discovery()
        algo.activator_signal = activator_signal
        return algo

    def test_default_gradient_signal_collects_nothing(self):
        algo = self._algo("gradient")

        out = algo._collect_posctx_values(
            1, "resid", torch.zeros(2, 8, dtype=torch.long), None, MagicMock()
        )

        assert out is None
        algo.inference.forward.assert_not_called()

    def test_enabled_signal_collects_the_collapsed_pins(self):
        algo = self._algo("gradient_x_posctx")
        pins = {(0, "attn"): torch.ones(D_SAE)}

        with patch("eval.floors.collect_site_anchors", return_value=({}, pins)) as m:
            out = algo._collect_posctx_values(
                1, "resid",
                torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, dtype=torch.long),
                MagicMock(),
            )

        assert out is pins
        # Must be the COLLAPSED pin: that is the value the cf eval injects.
        assert m.call_args.kwargs["pin_position_specific"] is False

    def test_seed_with_no_upstream_sites_returns_empty(self):
        """Layer-0 attn has nothing upstream — no site is scored, so an empty
        map is complete rather than missing."""
        algo = self._algo("gradient_x_posctx")

        out = algo._collect_posctx_values(
            0, "attn", torch.zeros(2, 8, dtype=torch.long), None, MagicMock()
        )

        assert out == {}

    def test_enabled_signal_without_pos_tokens_raises(self):
        algo = self._algo("gradient_x_posctx")

        with pytest.raises(ValueError, match="needs pos_tokens"):
            algo._collect_posctx_values(1, "resid", None, None, MagicMock())


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
        cf_faith=0.8,
    ):
        """
        Runs `algo.discover(seed_comp=3, seed_latent=0)` with all heavy
        dependencies mocked out.

        - probe_data is returned by build_probe_dataset.
        - _get_neg_tokens is mocked to represent centralized selector output.
        - _get_posctx_activation returns 1.0.
        - _run_contrast_hop returns (activator_fids_and_scores, inhibitor_fids_and_scores).
        - evaluate_counterfactual_faithfulness returns (cf_faith, suppression_score).
        - latent_stats.active_count always passes min_active_count=1.
        - CircuitLogger.save is patched to prevent disk I/O.
        """
        SEED_COMP, SEED_LAT = 3, 0  # layer 1, kind "resid"

        algo.build_probe_dataset = MagicMock(return_value=probe_data)
        algo.neg_mode = "close"

        mock_active_count = torch.full((N_COMP, D_SAE), 100, dtype=torch.long)

        with patch("circuit.discovery.counterfactual_gradient.latent_stats") as mock_ls, \
             patch("circuit.discovery.counterfactual_gradient.evaluate_counterfactual_faithfulness",
                   return_value=(cf_faith, 0.25)), \
             patch("circuit.discovery.counterfactual_gradient.prune_non_minimal_nodes_cf"), \
             patch("observability.circuit_logger.CircuitLogger.save"):

            mock_ls.active_count = mock_active_count

            algo._get_posctx_activation = MagicMock(return_value=1.0)
            algo._get_neg_tokens = MagicMock(
                return_value=probe_data.neg_tokens if probe_data.neg_tokens.shape[0] > 0 else None
            )
            algo._run_contrast_hop = MagicMock(
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
        """Empty score dicts from _run_contrast_hop → circuit has only seed → rejected."""
        algo = _make_discovery()
        probe_data = _make_probe_data()

        circuit = self._run_discover(algo, probe_data, activator_fids_and_scores={},
                                     inhibitor_fids_and_scores={})
        assert circuit is None

    def test_rejects_below_counterfactual_faithfulness_threshold(self):
        """Circuit with good nodes but low counterfactual_faithfulness is rejected."""
        algo = _make_discovery(min_faithfulness=0.5)
        probe_data = _make_probe_data()

        fid = FeatureID(0, "attn", 5)
        circuit = self._run_discover(
            algo, probe_data,
            activator_fids_and_scores={fid: 0.9},
            inhibitor_fids_and_scores={},
            cf_faith=0.1,   # below threshold
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
            cf_faith=0.9,
        )

        assert circuit is not None
        meta = circuit.metadata
        for key in ("counterfactual_faithfulness", "seed_comp", "seed_latent",
                    "n_nodes", "n_edges", "discovery_method"):
            assert key in meta, f"Missing metadata key: {key}"

        assert meta["discovery_method"] == "counterfactual_gradient"
        assert meta["counterfactual_faithfulness"] == pytest.approx(0.9)

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
