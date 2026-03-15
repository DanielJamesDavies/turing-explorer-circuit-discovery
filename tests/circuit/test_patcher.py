"""
Phase 3 — CircuitPatcher tests.

Tests the core ablation/intervention logic without loading real model weights.
Uses MockSAEBank from conftest.py (tiny d_model=16, d_sae=32, k=4).

Key algebraic identity (derived in patcher.py docstring):
    patched = circuit_recon + bg_linear + error
            = x + W_dec @ (circuit_e + avg_non_circuit - full_e)

All tests call patcher.transform() directly so the hook system is not
involved — that was already verified in Phase 2.

Notation used in test docstrings:
    full_e          dense top-k activation vector [B, T, D_SAE]
    circuit_e       full_e masked to circuit-latent indices
    avg_non_circuit avg_acts masked to non-circuit indices (0 for circuit indices)
    b_dec           decoder_bias [D_MODEL]
    decode(v)       v @ W_dec.T + b_dec
"""
import pytest
import torch

from circuit.patcher import CircuitPatcher
from store.circuits import Circuit, CircuitNode

# Dimensions match conftest constants.
B, T = 2, 4          # batch, sequence length
D_MODEL = 16
D_SAE   = 32
K_SAE   = 4
KINDS   = ["attn", "mlp", "resid"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kind_idx(kind: str) -> int:
    return KINDS.index(kind)


def _comp_idx(layer: int, kind: str) -> int:
    return layer * len(KINDS) + _kind_idx(kind)


def _scatter(top_acts: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
    """Scatter top-k activations into a dense [B, T, D_SAE] tensor."""
    dense = torch.zeros(*top_acts.shape[:-1], D_SAE)
    dense.scatter_(-1, top_indices.long(), top_acts.float())
    return dense


def _make_circuit_at(layer: int, kind: str, latent_idx: int, role: str = "upstream") -> Circuit:
    c = Circuit(name="test")
    c.add_node(CircuitNode(metadata={
        "layer_idx": layer, "kind": kind, "latent_idx": latent_idx, "role": role
    }))
    return c


def _make_full_circuit(layer: int, kind: str, latent_indices) -> Circuit:
    """Circuit containing every latent in `latent_indices` at (layer, kind)."""
    c = Circuit(name="full-circuit")
    for idx in latent_indices:
        c.add_node(CircuitNode(metadata={
            "layer_idx": layer, "kind": kind, "latent_idx": int(idx), "role": "upstream"
        }))
    return c


# ---------------------------------------------------------------------------
# TestCircuitPatcherInit — mask construction and background tensor
# ---------------------------------------------------------------------------

class TestCircuitPatcherInit:
    def test_circuit_masks_cover_all_layer_kind_pairs(self, mock_sae_bank, avg_acts_zero, make_circuit):
        circuit = make_circuit([(0, "attn", 5, "seed")])
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        for l in range(mock_sae_bank.n_layer):
            for kind in KINDS:
                assert (l, kind) in patcher.circuit_masks

    def test_mask_sets_correct_latent_to_true(self, mock_sae_bank, avg_acts_zero):
        circuit = _make_circuit_at(0, "attn", 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        assert patcher.circuit_masks[(0, "attn")][5].item() is True

    def test_mask_non_circuit_latents_are_false(self, mock_sae_bank, avg_acts_zero):
        circuit = _make_circuit_at(0, "attn", 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        mask = patcher.circuit_masks[(0, "attn")]
        assert not mask[:5].any()
        assert not mask[6:].any()

    def test_mask_all_false_for_unrelated_layers(self, mock_sae_bank, avg_acts_zero):
        """Node at (0, attn) should leave (1, attn) mask all-False."""
        circuit = _make_circuit_at(0, "attn", 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        assert not patcher.circuit_masks[(1, "attn")].any()

    def test_mask_all_false_for_unrelated_kinds(self, mock_sae_bank, avg_acts_zero):
        """Node at (0, attn) should leave (0, mlp) and (0, resid) masks all-False."""
        circuit = _make_circuit_at(0, "attn", 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        assert not patcher.circuit_masks[(0, "mlp")].any()
        assert not patcher.circuit_masks[(0, "resid")].any()

    def test_null_circuit_all_masks_false(self, mock_sae_bank, avg_acts_zero):
        patcher = CircuitPatcher(mock_sae_bank, None, avg_acts_zero)
        for l in range(mock_sae_bank.n_layer):
            for kind in KINDS:
                assert not patcher.circuit_masks[(l, kind)].any()

    def test_background_tensor_shape_is_d_model(self, mock_sae_bank, avg_acts_zero):
        circuit = _make_circuit_at(0, "attn", 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        for l in range(mock_sae_bank.n_layer):
            for kind in KINDS:
                assert patcher.background_tensors[(l, kind)].shape == (D_MODEL,)

    def test_background_excludes_decoder_bias(self, mock_sae_bank, avg_acts_nonzero):
        """
        background_tensors[(l, k)] must equal decode(avg_non_circuit) - b_dec.
        This exercises the single most subtle invariant in CircuitPatcher:
        the linear part only (W @ avg_non_circuit) is stored, not the full decode.
        """
        layer, kind = 0, "attn"
        circuit = _make_circuit_at(layer, kind, 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_nonzero)

        comp = _comp_idx(layer, kind)
        sae  = mock_sae_bank.saes[kind][layer]

        # Reconstruct the non-circuit average that __init__ used for bg.
        bg_latents = avg_acts_nonzero[comp].clone()
        bg_latents[5] = 0.0   # circuit latent zeroed for forward mode

        expected_bg = (
            mock_sae_bank.decode(bg_latents.view(1, 1, -1), kind, layer).squeeze()
            - sae.decoder_bias
        )
        actual_bg = patcher.background_tensors[(layer, kind)]
        assert torch.allclose(actual_bg, expected_bg, atol=1e-5)

    def test_background_is_zero_when_avg_is_zero(self, mock_sae_bank, avg_acts_zero):
        """
        With avg_acts = 0:
            bg = decode(0) - b_dec = (0 @ W_dec.T + b_dec) - b_dec = 0
        """
        patcher = CircuitPatcher(mock_sae_bank, None, avg_acts_zero)
        for l in range(mock_sae_bank.n_layer):
            for kind in KINDS:
                assert torch.allclose(
                    patcher.background_tensors[(l, kind)],
                    torch.zeros(D_MODEL),
                    atol=1e-6,
                )


# ---------------------------------------------------------------------------
# TestPatchingForwardMode — algebraic correctness (inverse=False, no pos_argmax)
# ---------------------------------------------------------------------------

class TestPatchingForwardMode:
    def test_full_circuit_returns_original_input(self, mock_sae_bank, avg_acts_zero):
        """
        When circuit contains ALL active latents at (layer, kind) and avg = 0:
            circuit_e = full_e  →  delta = 0  →  patched = x + decode(0) - b_dec = x
        """
        layer, kind = 0, "attn"
        x = torch.randn(B, T, D_MODEL)

        top_acts, top_indices = mock_sae_bank.encode(x, kind, layer)
        active_indices = top_indices.unique().tolist()

        circuit = _make_full_circuit(layer, kind, active_indices)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        result = patcher.transform(layer, kind, x)

        assert torch.allclose(result, x, atol=1e-4)

    def test_algebraic_identity_patched_equals_x_plus_linear_delta(
        self, mock_sae_bank, avg_acts_zero
    ):
        """
        Core formula test (with avg = 0):
            patched = x + W_dec @ (circuit_e - full_e)
                    = x + decode(circuit_e - full_e) - b_dec
        Verified numerically for a single-node circuit at latent index 3.
        """
        layer, kind = 0, "attn"
        circuit_latent = 3

        circuit = _make_circuit_at(layer, kind, circuit_latent)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)

        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)

        # Manual computation
        sae = mock_sae_bank.saes[kind][layer]
        top_acts, top_indices = mock_sae_bank.encode(x, kind, layer)
        full_e = _scatter(top_acts, top_indices)

        # circuit_e: full_e masked to latent 3 only
        mask = torch.zeros(D_SAE, dtype=torch.bool)
        mask[circuit_latent] = True
        circuit_e = full_e * mask

        delta = circuit_e - full_e   # avg = 0 so avg_non_circuit = 0
        expected = x + mock_sae_bank.decode(delta, kind, layer) - sae.decoder_bias

        assert torch.allclose(result, expected, atol=1e-4)

    def test_algebraic_identity_with_nonzero_avg(self, mock_sae_bank, avg_acts_nonzero):
        """
        Same formula but with avg ≠ 0:
            patched = x + W_dec @ (circuit_e + avg_non_circuit - full_e)
                    = x + decode(circuit_e + avg_non_circuit - full_e) - b_dec
        """
        layer, kind = 0, "mlp"
        circuit_latent = 7

        circuit = _make_circuit_at(layer, kind, circuit_latent)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_nonzero)

        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)

        sae  = mock_sae_bank.saes[kind][layer]
        comp = _comp_idx(layer, kind)
        top_acts, top_indices = mock_sae_bank.encode(x, kind, layer)
        full_e = _scatter(top_acts, top_indices)

        circuit_mask = torch.zeros(D_SAE, dtype=torch.bool)
        circuit_mask[circuit_latent] = True

        circuit_e       = full_e * circuit_mask
        avg_non_circuit = avg_acts_nonzero[comp] * (~circuit_mask)
        delta           = circuit_e + avg_non_circuit - full_e

        expected = x + mock_sae_bank.decode(delta, kind, layer) - sae.decoder_bias
        assert torch.allclose(result, expected, atol=1e-4)

    def test_null_circuit_formula_with_zero_avg(self, mock_sae_bank, avg_acts_zero):
        """
        With circuit=None and avg=0:
            patched = x - W_dec @ full_e  = x - (decode(full_e) - b_dec)
        """
        layer, kind = 0, "resid"
        patcher = CircuitPatcher(mock_sae_bank, None, avg_acts_zero)

        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)

        sae = mock_sae_bank.saes[kind][layer]
        top_acts, top_indices = mock_sae_bank.encode(x, kind, layer)
        full_e = _scatter(top_acts, top_indices)

        # With avg=0: delta = 0 + 0 - full_e = -full_e
        expected = x + mock_sae_bank.decode(-full_e, kind, layer) - sae.decoder_bias
        assert torch.allclose(result, expected, atol=1e-4)

    def test_null_circuit_differs_from_original(self, mock_sae_bank, avg_acts_zero):
        """
        With circuit=None (total ablation), the output is not equal to x
        because the SAE reconstruction is subtracted from x.
        """
        layer, kind = 0, "attn"
        patcher = CircuitPatcher(mock_sae_bank, None, avg_acts_zero)
        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)
        assert not torch.allclose(result, x, atol=1e-3)

    def test_partial_circuit_output_between_null_and_full(
        self, mock_sae_bank, avg_acts_zero
    ):
        """
        A single-latent circuit sits between the full circuit (→ x) and
        the null circuit (→ most ablated). The output must differ from both.
        """
        layer, kind = 0, "attn"
        x = torch.randn(B, T, D_MODEL)

        top_acts, top_indices = mock_sae_bank.encode(x, kind, layer)
        active_indices = top_indices.unique().tolist()

        null_patcher = CircuitPatcher(mock_sae_bank, None, avg_acts_zero)
        full_circuit = _make_full_circuit(layer, kind, active_indices)
        full_patcher = CircuitPatcher(mock_sae_bank, full_circuit, avg_acts_zero)

        # Single-latent circuit using one of the active latents
        partial_circuit = _make_circuit_at(layer, kind, active_indices[0])
        partial_patcher = CircuitPatcher(mock_sae_bank, partial_circuit, avg_acts_zero)

        null_out    = null_patcher.transform(layer, kind, x)
        full_out    = full_patcher.transform(layer, kind, x)
        partial_out = partial_patcher.transform(layer, kind, x)

        assert not torch.allclose(partial_out, null_out, atol=1e-4)
        assert not torch.allclose(partial_out, full_out, atol=1e-4)

    def test_non_circuit_latents_zeroed_in_live_acts(self, mock_sae_bank, avg_acts_zero):
        """
        With circuit at latent j, any active latent i ≠ j should have its
        contribution to circuit_recon zeroed.  We verify this by comparing
        circuit_recon from the patcher against an explicit single-latent decode.
        Equivalently: patched ≠ patched_with_j_zeroed_too (i.e., j matters).
        """
        layer, kind = 0, "attn"
        x = torch.randn(B, T, D_MODEL)

        top_acts, top_indices = mock_sae_bank.encode(x, kind, layer)
        active_indices = top_indices.unique().tolist()

        # Circuit contains only the first active latent.
        j = active_indices[0]
        circuit = _make_circuit_at(layer, kind, j)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        result  = patcher.transform(layer, kind, x)

        # Reference: circuit also contains the second active latent.
        circuit2 = _make_full_circuit(layer, kind, active_indices[:2])
        patcher2 = CircuitPatcher(mock_sae_bank, circuit2, avg_acts_zero)
        result2  = patcher2.transform(layer, kind, x)

        # Adding a second latent to the circuit should change the output
        # (the extra latent had a non-zero activation contribution).
        assert not torch.allclose(result, result2, atol=1e-5)


# ---------------------------------------------------------------------------
# TestPatchingInverseMode — inverse=True
# ---------------------------------------------------------------------------

class TestPatchingInverseMode:
    def test_inverse_empty_circuit_returns_original_input(self, mock_sae_bank, avg_acts_zero):
        """
        inverse=True + circuit with no active latents + avg=0:
            live_acts = top_acts (all non-circuit latents kept)
            patched   = decode(full_e) + 0 + error = x
        """
        layer, kind = 0, "attn"
        # Use a latent index that is very unlikely to appear in top-k.
        # Choose an index outside realistic top-k range by making the circuit
        # contain a latent that never fires for this specific x.
        x = torch.ones(B, T, D_MODEL) * 0.01   # small → encoder pre-acts near 0
        top_acts, top_indices = mock_sae_bank.encode(x, kind, layer)
        active_set = set(top_indices.unique().tolist())

        # Find a latent NOT in the active set
        dormant = next(i for i in range(D_SAE) if i not in active_set)

        circuit = _make_circuit_at(layer, kind, dormant)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero, inverse=True)
        result  = patcher.transform(layer, kind, x)

        assert torch.allclose(result, x, atol=1e-4)

    def test_inverse_full_circuit_ablates_all_active_latents(
        self, mock_sae_bank, avg_acts_zero
    ):
        """
        inverse=True + circuit contains ALL active latents + avg=0:
            live_acts = zeros (all circuit latents are ablated)
            patched   = decode(0) + 0 + error = b_dec + x - decode(full_e)
                      = x - W_dec @ full_e
        """
        layer, kind = 0, "attn"
        x = torch.randn(B, T, D_MODEL)

        top_acts, top_indices = mock_sae_bank.encode(x, kind, layer)
        active_indices = top_indices.unique().tolist()

        circuit = _make_full_circuit(layer, kind, active_indices)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero, inverse=True)
        result  = patcher.transform(layer, kind, x)

        sae    = mock_sae_bank.saes[kind][layer]
        full_e = _scatter(top_acts, top_indices)
        # With avg=0: live_acts=0, circuit_recon=b_dec, bg=0, error=x-decode(full_e)
        expected = sae.decoder_bias + x - mock_sae_bank.decode(full_e, kind, layer)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_inverse_mode_differs_from_forward_mode(self, mock_sae_bank, avg_acts_zero):
        """
        forward and inverse modes should produce different outputs for
        the same circuit (unless the circuit is trivially empty or full).
        """
        layer, kind = 0, "attn"
        x = torch.randn(B, T, D_MODEL)

        top_acts, top_indices = mock_sae_bank.encode(x, kind, layer)
        active_indices = top_indices.unique().tolist()
        # Use only half the active latents so neither mode is trivial
        partial_indices = active_indices[: len(active_indices) // 2 + 1]
        circuit = _make_full_circuit(layer, kind, partial_indices)

        fwd = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero, inverse=False)
        inv = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero, inverse=True)

        assert not torch.allclose(
            fwd.transform(layer, kind, x),
            inv.transform(layer, kind, x),
            atol=1e-4,
        )

    def test_inverse_mask_is_complement_of_forward_mask(self, mock_sae_bank, avg_acts_zero):
        """
        The same circuit should produce complementary live_acts selections
        in forward vs inverse mode.  We verify via the stored circuit_masks —
        both patchers use the same mask tensor (same circuit) but live_acts
        logic is inverted.
        The masks themselves should be identical (mask represents the circuit,
        not the live selection).
        """
        layer, kind = 0, "attn"
        circuit = _make_circuit_at(layer, kind, 10)
        fwd = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero, inverse=False)
        inv = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero, inverse=True)
        assert torch.equal(
            fwd.circuit_masks[(layer, kind)],
            inv.circuit_masks[(layer, kind)],
        )


# ---------------------------------------------------------------------------
# TestPositionSelectiveMode — pos_argmax
# ---------------------------------------------------------------------------

class TestPositionSelectiveMode:
    def test_probe_position_is_patched(self, mock_sae_bank, avg_acts_zero):
        """
        At pos_argmax[b], the output should differ from x (assuming the
        null circuit actually changes values, which it does for non-trivial x).
        """
        layer, kind = 0, "attn"
        pos_argmax = torch.tensor([1, 2])            # probe positions
        patcher = CircuitPatcher(
            mock_sae_bank, None, avg_acts_zero, pos_argmax=pos_argmax
        )

        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)

        # At each batch item's probe position, result ≠ x
        for b in range(B):
            p = pos_argmax[b].item()
            assert not torch.allclose(result[b, p], x[b, p], atol=1e-4), (
                f"Probe position {p} for batch {b} was not patched"
            )

    def test_non_probe_positions_unchanged(self, mock_sae_bank, avg_acts_zero):
        """
        At positions other than pos_argmax[b], result must equal x exactly.
        """
        layer, kind = 0, "attn"
        pos_argmax = torch.tensor([1, 2])
        patcher = CircuitPatcher(
            mock_sae_bank, None, avg_acts_zero, pos_argmax=pos_argmax
        )

        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)

        for b in range(B):
            probe = pos_argmax[b].item()
            for t in range(T):
                if t != probe:
                    assert torch.allclose(result[b, t], x[b, t], atol=1e-6), (
                        f"Position {t} (non-probe) was modified for batch {b}"
                    )

    def test_pos_argmax_boundary_position_zero(self, mock_sae_bank, avg_acts_zero):
        """pos_argmax = 0 (first token) should work correctly."""
        layer, kind = 0, "attn"
        pos_argmax = torch.zeros(B, dtype=torch.long)
        patcher = CircuitPatcher(
            mock_sae_bank, None, avg_acts_zero, pos_argmax=pos_argmax
        )
        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)

        # Non-probe positions (1, 2, 3) must be unchanged
        assert torch.allclose(result[:, 1:, :], x[:, 1:, :], atol=1e-6)

    def test_pos_argmax_last_position(self, mock_sae_bank, avg_acts_zero):
        """pos_argmax = T-1 (last token) should work correctly."""
        layer, kind = 0, "attn"
        pos_argmax = torch.full((B,), T - 1, dtype=torch.long)
        patcher = CircuitPatcher(
            mock_sae_bank, None, avg_acts_zero, pos_argmax=pos_argmax
        )
        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)

        # Non-probe positions (0..T-2) must be unchanged
        assert torch.allclose(result[:, :-1, :], x[:, :-1, :], atol=1e-6)

    def test_each_batch_item_has_independent_probe_position(
        self, mock_sae_bank, avg_acts_zero
    ):
        """
        With pos_argmax=[0, 3], batch item 0 is patched at position 0
        and unchanged at position 3; batch item 1 is patched at position 3
        and unchanged at position 0.
        """
        layer, kind = 0, "attn"
        pos_argmax = torch.tensor([0, 3])
        patcher = CircuitPatcher(
            mock_sae_bank, None, avg_acts_zero, pos_argmax=pos_argmax
        )
        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)

        # Batch 0: unchanged at position 3
        assert torch.allclose(result[0, 3], x[0, 3], atol=1e-6)
        # Batch 1: unchanged at position 0
        assert torch.allclose(result[1, 0], x[1, 0], atol=1e-6)

    def test_no_pos_argmax_all_positions_are_patched(self, mock_sae_bank, avg_acts_zero):
        """
        Without pos_argmax, the intervention applies at every token position,
        so result should differ from x across all positions.
        """
        layer, kind = 0, "attn"
        patcher = CircuitPatcher(mock_sae_bank, None, avg_acts_zero)  # no pos_argmax
        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)
        assert not torch.allclose(result, x, atol=1e-4)


# ---------------------------------------------------------------------------
# TestEdgeCases — passthrough, dtypes, small batch/seq
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_layer_returns_x_unchanged(self, mock_sae_bank, avg_acts_zero):
        """
        transform() for a (layer, kind) not in circuit_masks returns x as-is.
        Layer 99 is not in the bank, so it has no mask entry.
        """
        circuit = _make_circuit_at(0, "attn", 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(99, "attn", x)
        assert torch.equal(result, x)

    def test_output_dtype_matches_input_dtype_float32(self, mock_sae_bank, avg_acts_zero):
        layer, kind = 0, "attn"
        circuit = _make_circuit_at(layer, kind, 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        x = torch.randn(B, T, D_MODEL, dtype=torch.float32)
        result = patcher.transform(layer, kind, x)
        assert result.dtype == torch.float32

    def test_output_shape_matches_input_shape(self, mock_sae_bank, avg_acts_zero):
        layer, kind = 0, "attn"
        circuit = _make_circuit_at(layer, kind, 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        x = torch.randn(B, T, D_MODEL)
        result = patcher.transform(layer, kind, x)
        assert result.shape == x.shape

    def test_batch_size_one(self, mock_sae_bank, avg_acts_zero):
        layer, kind = 0, "attn"
        circuit = _make_circuit_at(layer, kind, 2)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        x = torch.randn(1, T, D_MODEL)
        result = patcher.transform(layer, kind, x)
        assert result.shape == (1, T, D_MODEL)

    def test_seq_len_one(self, mock_sae_bank, avg_acts_zero):
        layer, kind = 0, "attn"
        circuit = _make_circuit_at(layer, kind, 2)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)
        x = torch.randn(B, 1, D_MODEL)
        result = patcher.transform(layer, kind, x)
        assert result.shape == (B, 1, D_MODEL)

    def test_patcher_callable_returns_context_manager(
        self, mock_model, mock_sae_bank, avg_acts_zero
    ):
        """
        CircuitPatcher.__call__(model) must return a context manager
        (the multi_patch context).  Entering and exiting it should leave
        the model in its original state.
        """
        circuit = _make_circuit_at(0, "attn", 5)
        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)

        x = torch.randn(B, T, D_MODEL)
        with torch.no_grad():
            ref = mock_model(x).clone()

        ctx = patcher(mock_model)
        assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")

        with torch.no_grad():
            with ctx:
                mock_model(x)   # run under patching (side effects don't matter here)

        # Model should be restored after context exits
        with torch.no_grad():
            after = mock_model(x).clone()
        assert torch.allclose(ref, after)

    def test_multiple_nodes_same_layer_kind(self, mock_sae_bank, avg_acts_zero):
        """A circuit with several nodes at the same (layer, kind) is handled."""
        layer, kind = 0, "attn"
        circuit = Circuit(name="multi")
        for idx in [2, 5, 10, 20]:
            circuit.add_node(CircuitNode(metadata={
                "layer_idx": layer, "kind": kind,
                "latent_idx": idx, "role": "upstream"
            }))

        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)

        # All four latents should be True in the mask
        mask = patcher.circuit_masks[(layer, kind)]
        for idx in [2, 5, 10, 20]:
            assert mask[idx].item() is True

        # All other latents False
        other = [i for i in range(D_SAE) if i not in [2, 5, 10, 20]]
        assert not mask[other].any()

    def test_nodes_across_different_layers_and_kinds(self, mock_sae_bank, avg_acts_zero):
        """Nodes at different (layer, kind) pairs each set their own mask bit."""
        circuit = Circuit(name="cross")
        circuit.add_node(CircuitNode(metadata={"layer_idx": 0, "kind": "attn",  "latent_idx": 3,  "role": "upstream"}))
        circuit.add_node(CircuitNode(metadata={"layer_idx": 0, "kind": "mlp",   "latent_idx": 7,  "role": "upstream"}))
        circuit.add_node(CircuitNode(metadata={"layer_idx": 1, "kind": "resid", "latent_idx": 15, "role": "seed"}))

        patcher = CircuitPatcher(mock_sae_bank, circuit, avg_acts_zero)

        assert patcher.circuit_masks[(0, "attn")][3].item()  is True
        assert patcher.circuit_masks[(0, "mlp")][7].item()   is True
        assert patcher.circuit_masks[(1, "resid")][15].item() is True

        # Cross-contamination check
        assert not patcher.circuit_masks[(0, "attn")][7].item()
        assert not patcher.circuit_masks[(1, "attn")][3].item()
