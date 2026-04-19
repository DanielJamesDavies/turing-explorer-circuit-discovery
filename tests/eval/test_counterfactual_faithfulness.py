"""
Unit tests for CounterfactualInterventionPatcher and evaluate_counterfactual_faithfulness.

Structure
---------
Part 1  TestCounterfactualInterventionPatcher
            Runs a real forward pass through MockModel (from conftest) with the
            patcher applied as a context manager.  Each test inspects the patcher's
            side-effects (captured_activation, transform return value).

Part 2  TestEvaluateCounterfactualFaithfulness
            Tests the score formula and call protocol using a stateful ControlledSAEBank
            and a StubInference that drives all three forward passes.  No real model
            runs; the stub calls activations_callback / patcher.transform directly at
            the seed layer.

Formula under test:
    score = (a_intervened - a_baseline) / (a_posctx - a_baseline)
"""

import pytest
import torch
from unittest.mock import MagicMock

from eval.counterfactual_faithfulness import (
    CounterfactualInterventionPatcher,
    evaluate_counterfactual_faithfulness,
)
from store.circuits import Circuit, CircuitNode
from circuit.types.feature_id import FeatureID


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B, T     = 2, 4
D_MODEL  = 16
D_SAE    = 32
K_SAE    = 4
N_LAYERS = 2
KINDS    = ["attn", "mlp", "resid"]

SEED_LAYER  = 1
SEED_KIND   = "resid"
SEED_LATENT = 5


# ---------------------------------------------------------------------------
# ControlledSAEBank
#
# At (seed_layer, seed_kind) encode always places SEED_LATENT in slot 0 of
# top_indices with act = current_seed_act (updated externally before each pass).
# All other encodes return all-zero acts / indices.
# Decode always returns a zero tensor of the right shape.
# ---------------------------------------------------------------------------

class ControlledSAEBank:
    d_sae            = D_SAE
    kinds            = KINDS
    n_layer          = N_LAYERS
    layer_device_map = {l: torch.device("cpu") for l in range(N_LAYERS)}

    def __init__(
        self,
        seed_layer: int  = SEED_LAYER,
        seed_kind:  str  = SEED_KIND,
        seed_latent: int = SEED_LATENT,
        seed_act:   float = 0.0,
    ) -> None:
        self.seed_layer       = seed_layer
        self.seed_kind        = seed_kind
        self.seed_latent      = seed_latent
        self.current_seed_act = seed_act   # caller sets this before each forward

    def encode(self, x: torch.Tensor, kind: str, layer: int):
        Bx, Tx = x.shape[:2]
        top_acts    = torch.zeros(Bx, Tx, K_SAE)
        top_indices = torch.zeros(Bx, Tx, K_SAE, dtype=torch.long)
        if layer == self.seed_layer and kind == self.seed_kind:
            top_indices[..., 0] = self.seed_latent
            top_acts[..., 0]    = self.current_seed_act
        return top_acts, top_indices

    def decode(self, latents: torch.Tensor, kind: str, layer: int) -> torch.Tensor:
        return torch.zeros(*latents.shape[:-1], D_MODEL)


# ---------------------------------------------------------------------------
# Stub inference
#
# Drives the three forward passes expected by evaluate_counterfactual_faithfulness:
#   call 1 → posctx  (activations_callback)
#   call 2 → baseline negctx (activations_callback)
#   call 3 → intervened negctx (patcher)
#
# Before each pass the bank's current_seed_act is updated so that encode returns
# the desired activation level for that pass.
# ---------------------------------------------------------------------------

def _make_stub_inference(bank: ControlledSAEBank, pass_acts: tuple, seed_layer: int = SEED_LAYER):
    """
    pass_acts: (posctx_seed_act, baseline_seed_act, intervened_seed_act)
    Returns (mock_inference, call_counter_list).
    """
    call_num = [0]

    def forward_fn(tokens, activations_callback=None, patcher=None, **kwargs):
        call_num[0] += 1
        bank.current_seed_act = pass_acts[call_num[0] - 1]

        x         = torch.zeros(B, T, D_MODEL)
        acts_tuple = tuple(x.clone() for _ in KINDS)

        if activations_callback is not None:
            activations_callback(seed_layer, acts_tuple)
        if patcher is not None:
            for kind in KINDS:
                patcher.transform(seed_layer, kind, x.clone())

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    return inf, call_num


# ---------------------------------------------------------------------------
# Circuit builder helpers
# ---------------------------------------------------------------------------

def _make_circuit(activator_fids=(), inhibitor_fids=()):
    c = Circuit(name="test")
    c.add_node(CircuitNode(metadata={
        "feature_id": FeatureID(SEED_LAYER, SEED_KIND, SEED_LATENT),
        "role": "seed",
    }))
    for fid in activator_fids:
        c.add_node(CircuitNode(metadata={
            "feature_id": fid,
            "role": "counterfactual_activator",
        }))
    for fid in inhibitor_fids:
        c.add_node(CircuitNode(metadata={
            "feature_id": fid,
            "role": "counterfactual_inhibitor",
        }))
    return c


def _avg_acts() -> torch.Tensor:
    return torch.zeros(N_LAYERS * len(KINDS), D_SAE)


# ============================================================================
# Part 1 — TestCounterfactualInterventionPatcher
# ============================================================================

class TestCounterfactualInterventionPatcher:
    """
    Exercises CounterfactualInterventionPatcher by running it as a context manager
    on MockModel (the conftest fixture), then inspecting side-effects.
    """

    def _make_patcher(
        self,
        bank,
        activator_targets=None,
        inhibitor_indices=None,
        seed_layer=SEED_LAYER,
        seed_kind=SEED_KIND,
        seed_latent=SEED_LATENT,
        pos_argmax=None,
        circuit_layers=None,
    ) -> CounterfactualInterventionPatcher:
        return CounterfactualInterventionPatcher(
            bank=bank,
            activator_targets=activator_targets or {},
            inhibitor_indices=inhibitor_indices or {},
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent,
            pos_argmax=pos_argmax,
            circuit_layers=circuit_layers,
        )

    # ── Capture ──────────────────────────────────────────────────────────────

    def test_captured_activation_set_after_forward(self, mock_model):
        """captured_activation must not be None after a forward pass."""
        bank    = ControlledSAEBank(seed_act=0.0)
        patcher = self._make_patcher(bank)
        x = torch.zeros(B, T, D_MODEL)
        with patcher(mock_model):
            mock_model(x)
        assert patcher.captured_activation is not None

    def test_captured_activation_reflects_seed_latent_act(self, mock_model):
        """captured_activation should equal the configured seed latent activation."""
        expected = 4.5
        bank     = ControlledSAEBank(seed_act=expected)
        pos_argmax = torch.zeros(B, dtype=torch.long)   # measure at position 0
        patcher  = self._make_patcher(bank, pos_argmax=pos_argmax)
        x = torch.zeros(B, T, D_MODEL)
        with patcher(mock_model):
            mock_model(x)
        assert patcher.captured_activation == pytest.approx(expected, abs=1e-5)

    def test_capture_when_pos_argmax_shorter_than_batch(self, mock_model):
        """
        pos_argmax with fewer entries than B must not raise IndexError.
        The fix (actual_B = min(B, pos_argmax.shape[0])) is exercised here.
        """
        bank       = ControlledSAEBank(seed_act=2.0)
        pos_argmax = torch.zeros(1, dtype=torch.long)   # B=2 but only 1 argmax entry
        patcher    = self._make_patcher(bank, pos_argmax=pos_argmax)
        x = torch.zeros(B, T, D_MODEL)
        with patcher(mock_model):
            mock_model(x)
        # Must not raise; capture should succeed over the shared single entry
        assert patcher.captured_activation is not None

    def test_no_capture_when_seed_layer_not_reached(self, mock_model):
        """captured_activation stays None when seed_layer exceeds the model depth."""
        bank    = ControlledSAEBank(seed_layer=99, seed_act=5.0)
        patcher = self._make_patcher(bank, seed_layer=99)
        x = torch.zeros(B, T, D_MODEL)
        with patcher(mock_model):
            mock_model(x)
        assert patcher.captured_activation is None

    # ── Intervention ─────────────────────────────────────────────────────────

    def test_no_intervention_outside_circuit_layers(self):
        """
        When circuit_layers restricts to a layer not reached, transform must
        return the input tensor unchanged.
        """
        bank = ControlledSAEBank(seed_act=0.0)
        activator_targets = {(0, "attn"): {3: 5.0}}
        patcher = self._make_patcher(
            bank,
            activator_targets=activator_targets,
            circuit_layers={99},   # layer 0 is excluded
        )
        x   = torch.randn(B, T, D_MODEL)
        out = patcher.transform(0, "attn", x)
        assert torch.allclose(out, x)

    def test_no_intervention_at_layer_with_no_targets(self):
        """
        transform at a (layer, kind) pair absent from both activator_targets and
        inhibitor_indices must return x unchanged (no encoding or decoding).
        """
        bank    = ControlledSAEBank(seed_act=0.0)
        patcher = self._make_patcher(
            bank,
            activator_targets={(1, "attn"): {0: 1.0}},   # only layer 1 attn
        )
        x   = torch.randn(B, T, D_MODEL)
        out = patcher.transform(0, "mlp", x)   # layer 0 mlp — no target
        assert torch.allclose(out, x)


# ============================================================================
# Part 2 — TestEvaluateCounterfactualFaithfulness
# ============================================================================

class TestEvaluateCounterfactualFaithfulness:
    """
    Tests the score formula and call protocol of evaluate_counterfactual_faithfulness.

    All heavy infrastructure is replaced by ControlledSAEBank + _make_stub_inference,
    so the tests exercise only the measurement + formula logic.
    """

    POSCTX_ACT   = 2.0
    BASELINE_ACT = 0.0

    def _default_circuit(self):
        return _make_circuit(activator_fids=[FeatureID(0, "attn", 7)])

    def _run(
        self,
        circuit,
        posctx_act   = None,
        baseline_act = None,
        intervened_act = None,
        pos_argmax   = None,
    ) -> float:
        posctx_act    = self.POSCTX_ACT   if posctx_act    is None else posctx_act
        baseline_act  = self.BASELINE_ACT if baseline_act  is None else baseline_act
        intervened_act = posctx_act       if intervened_act is None else intervened_act

        bank = ControlledSAEBank()
        inf, _ = _make_stub_inference(bank, (posctx_act, baseline_act, intervened_act))

        pos_tokens = torch.zeros(B, T, dtype=torch.long)
        neg_tokens = torch.zeros(B, T, dtype=torch.long)
        pos_argmax = torch.zeros(B, dtype=torch.long) if pos_argmax is None else pos_argmax

        return evaluate_counterfactual_faithfulness(
            inf, bank, _avg_acts(), circuit,
            neg_tokens=neg_tokens,
            pos_tokens=pos_tokens,
            seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT,
            pos_argmax=pos_argmax,
            circuit_layers={SEED_LAYER, 0},
        )

    # ── Score formula ────────────────────────────────────────────────────────

    def test_score_is_one_when_intervention_fully_recovers(self):
        """(a_intervened == a_posctx) → score = 1.0."""
        score = self._run(self._default_circuit(),
                          posctx_act=2.0, baseline_act=0.0, intervened_act=2.0)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_score_is_zero_when_intervention_has_no_effect(self):
        """(a_intervened == a_baseline) → score = 0.0."""
        score = self._run(self._default_circuit(),
                          posctx_act=2.0, baseline_act=0.0, intervened_act=0.0)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_score_is_half_when_halfway_recovered(self):
        """a_intervened halfway between baseline and posctx → score = 0.5."""
        # (2 - 0) / (4 - 0) = 0.5
        score = self._run(self._default_circuit(),
                          posctx_act=4.0, baseline_act=0.0, intervened_act=2.0)
        assert score == pytest.approx(0.5, abs=1e-5)

    def test_score_is_negative_when_intervention_worsens_activation(self):
        """a_intervened below a_baseline → score < 0."""
        # (0 - 1) / (2 - 1) = -1.0
        score = self._run(self._default_circuit(),
                          posctx_act=2.0, baseline_act=1.0, intervened_act=0.0)
        assert score < 0.0

    def test_score_above_one_when_intervention_overshoots(self):
        """a_intervened exceeds a_posctx → score > 1.0 (formula is unclamped)."""
        # (3 - 0) / (2 - 0) = 1.5
        score = self._run(self._default_circuit(),
                          posctx_act=2.0, baseline_act=0.0, intervened_act=3.0)
        assert score > 1.0

    # ── Empty circuit ────────────────────────────────────────────────────────

    def test_empty_circuit_returns_zero_without_forward(self):
        """A circuit with only a seed node must return 0.0 with no forward pass."""
        seed_only = _make_circuit()   # no activators, no inhibitors
        bank      = ControlledSAEBank()
        inf       = MagicMock()

        score = evaluate_counterfactual_faithfulness(
            inf, bank, _avg_acts(), seed_only,
            neg_tokens=torch.zeros(B, T, dtype=torch.long),
            pos_tokens=torch.zeros(B, T, dtype=torch.long),
            seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT,
        )

        assert score == pytest.approx(0.0, abs=1e-5)
        inf.forward.assert_not_called()

    # ── Small-denominator guard ───────────────────────────────────────────────

    def test_small_denom_matching_intervened_returns_one(self):
        """
        When |a_posctx - a_baseline| < 1e-9 and a_intervened ≈ a_posctx → score = 1.0.
        """
        eps   = 1e-12
        score = self._run(self._default_circuit(),
                          posctx_act=eps, baseline_act=0.0, intervened_act=eps)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_small_denom_nonmatching_intervened_returns_zero(self):
        """
        When |a_posctx - a_baseline| < 1e-9 and a_intervened ≠ a_posctx → score = 0.0.
        """
        score = self._run(self._default_circuit(),
                          posctx_act=0.0, baseline_act=0.0, intervened_act=1.0)
        assert score == pytest.approx(0.0, abs=1e-5)

    # ── Call protocol ────────────────────────────────────────────────────────

    def test_exactly_three_forward_passes(self):
        """evaluate_counterfactual_faithfulness must call inference.forward exactly 3 times."""
        bank = ControlledSAEBank()
        inf, counter = _make_stub_inference(bank, (2.0, 0.0, 1.5))

        evaluate_counterfactual_faithfulness(
            inf, bank, _avg_acts(), self._default_circuit(),
            neg_tokens=torch.zeros(B, T, dtype=torch.long),
            pos_tokens=torch.zeros(B, T, dtype=torch.long),
            seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT,
            pos_argmax=torch.zeros(B, dtype=torch.long),
        )

        assert counter[0] == 3

    def test_return_type_is_float(self):
        """evaluate_counterfactual_faithfulness must return a Python float."""
        score = self._run(self._default_circuit())
        assert isinstance(score, float)
