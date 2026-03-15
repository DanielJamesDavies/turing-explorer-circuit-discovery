"""
Phase 2 — Hook system tests.

Tests capture_activations, patch, and multi_patch using MockModel from conftest.py.
No model weights, no SAE, no CUDA required — all tests run on CPU.

MockBlock closed-form values (used for precise algebraic assertions):
    attn_out  = x * 0.1
    mlp_out   = x * 0.2   (norm_2 is identity passthrough)
    resid_out = x * 1.3   (x + attn_out + mlp_out)
"""
import pytest
import torch

from model.hooks import capture_activations, patch, multi_patch

# Tensor dimensions shared across all tests — match D_MODEL from conftest.
B, T, N = 2, 4, 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(model, x):
    """Deterministic no-grad forward pass; returns final residual."""
    with torch.no_grad():
        return model(x)


# ---------------------------------------------------------------------------
# capture_activations
# ---------------------------------------------------------------------------

class TestCaptureActivations:
    def test_output_shape_is_B_L_K_T_N(self, mock_model):
        x = torch.randn(B, T, N)
        with capture_activations(mock_model) as acts:
            _run(mock_model, x)
        n_layers = len(mock_model.transformer.h)
        assert acts.tensor.shape == (B, n_layers, 3, T, N)

    def test_layer_dimension_matches_block_count(self, mock_model):
        x = torch.randn(B, T, N)
        with capture_activations(mock_model) as acts:
            _run(mock_model, x)
        assert acts.tensor.shape[1] == len(mock_model.transformer.h)

    def test_kind_dimension_is_three(self, mock_model):
        """K=3: one slot for attn, mlp, and resid outputs."""
        x = torch.randn(B, T, N)
        with capture_activations(mock_model) as acts:
            _run(mock_model, x)
        assert acts.tensor.shape[2] == 3

    def test_stored_tensor_is_detached(self, mock_model):
        x = torch.randn(B, T, N)
        with capture_activations(mock_model) as acts:
            _run(mock_model, x)
        assert acts.tensor.grad_fn is None

    def test_capture_false_yields_none_tensor(self, mock_model):
        x = torch.randn(B, T, N)
        with capture_activations(mock_model, capture=False) as acts:
            _run(mock_model, x)
        assert acts.tensor is None

    def test_callback_called_once_per_layer(self, mock_model):
        calls = []
        x = torch.randn(B, T, N)
        with capture_activations(mock_model, callback=lambda l, o: calls.append(l)):
            _run(mock_model, x)
        assert len(calls) == len(mock_model.transformer.h)

    def test_callback_receives_ascending_layer_indices(self, mock_model):
        calls = []
        x = torch.randn(B, T, N)
        with capture_activations(mock_model, callback=lambda l, o: calls.append(l)):
            _run(mock_model, x)
        assert calls == list(range(len(mock_model.transformer.h)))

    def test_callback_output_is_a_three_tuple(self, mock_model):
        outputs = []
        x = torch.randn(B, T, N)
        with capture_activations(mock_model, callback=lambda l, o: outputs.append(o)):
            _run(mock_model, x)
        for out in outputs:
            assert len(out) == 3

    def test_hooks_removed_after_context_model_runs_identically(self, mock_model):
        x = torch.randn(B, T, N)
        ref = _run(mock_model, x).clone()
        with capture_activations(mock_model):
            _run(mock_model, x)
        after = _run(mock_model, x)
        assert torch.allclose(ref, after)

    def test_captured_attn_slot_matches_mockblock_attn_formula(self, mock_model):
        """
        acts[:, 0, 0, :, :] (layer 0, kind 0 = attn) should equal x * 0.1.
        Layer 0 receives the raw input x, so attn_out = MockAttn(x) = x * 0.1.
        """
        x = torch.randn(B, T, N)
        with capture_activations(mock_model) as acts:
            _run(mock_model, x)
        assert torch.allclose(acts.tensor[:, 0, 0, :, :], x * 0.1, atol=1e-5)

    def test_captured_mlp_slot_matches_mockblock_mlp_formula(self, mock_model):
        """acts[:, 0, 1, :, :] (layer 0, kind 1 = mlp) should equal x * 0.2."""
        x = torch.randn(B, T, N)
        with capture_activations(mock_model) as acts:
            _run(mock_model, x)
        assert torch.allclose(acts.tensor[:, 0, 1, :, :], x * 0.2, atol=1e-5)

    def test_captured_resid_slot_matches_mockblock_resid_formula(self, mock_model):
        """acts[:, 0, 2, :, :] (layer 0, kind 2 = resid) should equal x * 1.3."""
        x = torch.randn(B, T, N)
        with capture_activations(mock_model) as acts:
            _run(mock_model, x)
        assert torch.allclose(acts.tensor[:, 0, 2, :, :], x * 1.3, atol=1e-5)

    def test_layer1_captures_propagated_residual(self, mock_model):
        """
        Layer 1 receives layer 0's residual (x * 1.3) as input, so
        its attn output should be x * 1.3 * 0.1 = x * 0.13.
        """
        x = torch.randn(B, T, N)
        with capture_activations(mock_model) as acts:
            _run(mock_model, x)
        assert torch.allclose(acts.tensor[:, 1, 0, :, :], x * 0.13, atol=1e-5)


# ---------------------------------------------------------------------------
# patch (single static replacement)
# ---------------------------------------------------------------------------

class TestPatch:
    def test_attn_kind_replaces_block_attn_output(self, mock_model):
        """
        patch(model, 0, 'attn', value) hooks block.attn so it returns `value`.
        The block captures this in its 3-tuple return as o[0].
        """
        value = torch.ones(B, T, N) * 42.0
        x = torch.randn(B, T, N)
        block_outputs = []

        with torch.no_grad():
            with patch(mock_model, 0, "attn", value):
                handle = mock_model.transformer.h[0].register_forward_hook(
                    lambda m, i, o: block_outputs.append(o)
                )
                mock_model(x)
                handle.remove()

        attn_out, _, _ = block_outputs[0]
        assert torch.allclose(attn_out, value)

    def test_attn_patch_propagates_into_residual(self, mock_model):
        """
        With attn replaced by `value`, resid = x + value + mlp_out.
        mlp_out is still x * 0.2 (mlp is not patched), so
        resid = x + value + x * 0.2 = x * 1.2 + value.
        """
        value = torch.zeros(B, T, N)  # zeroed attn
        x = torch.randn(B, T, N)
        block_outputs = []

        with torch.no_grad():
            with patch(mock_model, 0, "attn", value):
                handle = mock_model.transformer.h[0].register_forward_hook(
                    lambda m, i, o: block_outputs.append(o)
                )
                mock_model(x)
                handle.remove()

        _, _, resid = block_outputs[0]
        expected = x * 1.2 + value  # x + 0 + x*0.2
        assert torch.allclose(resid, expected, atol=1e-5)

    def test_mlp_kind_replaces_block_mlp_output(self, mock_model):
        value = torch.ones(B, T, N) * 99.0
        x = torch.randn(B, T, N)
        block_outputs = []

        with torch.no_grad():
            with patch(mock_model, 0, "mlp", value):
                handle = mock_model.transformer.h[0].register_forward_hook(
                    lambda m, i, o: block_outputs.append(o)
                )
                mock_model(x)
                handle.remove()

        _, mlp_out, _ = block_outputs[0]
        assert torch.allclose(mlp_out, value)

    def test_resid_kind_replaces_norm2_input_via_prehook(self, mock_model):
        """
        patch(model, 0, 'resid', value) installs a pre-hook on block.norm_2
        that injects `value` as its input.  Since MockNorm2 is identity,
        norm_2 then returns `value`, and the MLP receives `value`.
        We verify by adding a second pre-hook that records norm_2's received input.
        """
        value = torch.ones(B, T, N) * 7.0
        x = torch.randn(B, T, N)
        norm2_received = []

        with torch.no_grad():
            with patch(mock_model, 0, "resid", value):
                # Second pre-hook sees whatever the first hook passed in.
                handle = mock_model.transformer.h[0].norm_2.register_forward_pre_hook(
                    lambda m, i: norm2_received.append(i[0].clone())
                )
                mock_model(x)
                handle.remove()

        assert torch.allclose(norm2_received[0], value)

    def test_patch_only_affects_specified_layer(self, mock_model):
        """A patch at layer 0 must not directly alter layer 1's attn output."""
        value = torch.ones(B, T, N) * 50.0
        x = torch.randn(B, T, N)
        layer1_attn_outputs = []

        with torch.no_grad():
            with patch(mock_model, 0, "attn", value):
                handle = mock_model.transformer.h[1].attn.register_forward_hook(
                    lambda m, i, o: layer1_attn_outputs.append(o.clone())
                )
                mock_model(x)
                handle.remove()

        # Layer 1's attn output is not the same as `value` (it was not patched directly)
        assert not torch.allclose(layer1_attn_outputs[0], value)

    def test_context_modifies_output_then_restores_it(self, mock_model):
        """Output should change inside the context and match original after."""
        x = torch.randn(B, T, N)
        value = torch.zeros(B, T, N)

        ref = _run(mock_model, x).clone()

        with torch.no_grad():
            with patch(mock_model, 0, "attn", value):
                during = mock_model(x).clone()

        after = _run(mock_model, x).clone()

        assert not torch.allclose(ref, during), "patch had no effect inside context"
        assert torch.allclose(ref, after), "hook was not removed after context"

    def test_hooks_removed_after_context(self, mock_model):
        """After an empty context entry/exit, model output is unchanged."""
        x = torch.randn(B, T, N)
        ref = _run(mock_model, x).clone()

        with torch.no_grad():
            with patch(mock_model, 0, "attn", torch.ones(B, T, N) * 999.0):
                pass  # enter and immediately exit

        after = _run(mock_model, x).clone()
        assert torch.allclose(ref, after)


# ---------------------------------------------------------------------------
# multi_patch (per-layer-per-kind transform)
# ---------------------------------------------------------------------------

class TestMultiPatch:
    def test_transform_called_for_every_layer_and_kind(self, mock_model):
        calls = []
        x = torch.randn(B, T, N)
        with torch.no_grad():
            with multi_patch(mock_model, lambda l, k, t: calls.append((l, k)) or None):
                mock_model(x)
        n_layers = len(mock_model.transformer.h)
        assert len(calls) == n_layers * 3

    def test_transform_receives_all_three_kinds(self, mock_model):
        kinds_seen = set()
        x = torch.randn(B, T, N)
        with torch.no_grad():
            with multi_patch(mock_model, lambda l, k, t: kinds_seen.add(k) or None):
                mock_model(x)
        assert kinds_seen == {"attn", "mlp", "resid"}

    def test_transform_receives_all_layer_indices(self, mock_model):
        layers_seen = set()
        x = torch.randn(B, T, N)
        with torch.no_grad():
            with multi_patch(mock_model, lambda l, k, t: layers_seen.add(l) or None):
                mock_model(x)
        assert layers_seen == set(range(len(mock_model.transformer.h)))

    def test_transform_receives_correct_tensor_shape(self, mock_model):
        """Each (layer, kind) tensor passed to transform should be [B, T, N]."""
        shapes = []
        x = torch.randn(B, T, N)
        with torch.no_grad():
            with multi_patch(mock_model, lambda l, k, t: shapes.append(t.shape) or None):
                mock_model(x)
        for s in shapes:
            assert tuple(s) == (B, T, N)

    def test_none_return_passes_tensor_through_unchanged(self, mock_model):
        x = torch.randn(B, T, N)
        ref = _run(mock_model, x).clone()
        with torch.no_grad():
            with multi_patch(mock_model, lambda l, k, t: None):
                result = mock_model(x).clone()
        assert torch.allclose(ref, result)

    def test_zeroing_attn_changes_model_output(self, mock_model):
        """Replacing all attn outputs with zeros must change the final residual."""
        x = torch.randn(B, T, N)
        ref = _run(mock_model, x).clone()

        def transform(l, k, t):
            return torch.zeros_like(t) if k == "attn" else None

        with torch.no_grad():
            with multi_patch(mock_model, transform):
                patched = mock_model(x).clone()

        assert not torch.allclose(ref, patched)

    def test_zeroing_mlp_changes_model_output(self, mock_model):
        x = torch.randn(B, T, N)
        ref = _run(mock_model, x).clone()

        def transform(l, k, t):
            return torch.zeros_like(t) if k == "mlp" else None

        with torch.no_grad():
            with multi_patch(mock_model, transform):
                patched = mock_model(x).clone()

        assert not torch.allclose(ref, patched)

    def test_resid_patch_feeds_into_next_block_as_input(self, mock_model):
        """
        Replacing layer 0's residual with zeros means block 1 receives zeros
        as its input.  Verified by capturing block 1's forward pre-hook.
        """
        zero_resid = torch.zeros(B, T, N)
        layer1_inputs = []

        def transform(l, k, t):
            return zero_resid if (l == 0 and k == "resid") else None

        x = torch.randn(B, T, N)
        with torch.no_grad():
            with multi_patch(mock_model, transform):
                handle = mock_model.transformer.h[1].register_forward_pre_hook(
                    lambda m, i: layer1_inputs.append(i[0].clone())
                )
                mock_model(x)
                handle.remove()

        assert torch.allclose(layer1_inputs[0], zero_resid)

    def test_attn_transform_value_appears_in_block_output_tuple(self, mock_model):
        """
        When transform returns `constant` for 'attn' at layer 0,
        the block's returned o[0] (attn slot) should equal `constant`.
        """
        constant = torch.ones(B, T, N) * 5.0
        block0_outputs = []

        def transform(l, k, t):
            return constant if (l == 0 and k == "attn") else None

        x = torch.randn(B, T, N)
        with torch.no_grad():
            with multi_patch(mock_model, transform):
                handle = mock_model.transformer.h[0].register_forward_hook(
                    lambda m, i, o: block0_outputs.append(o)
                )
                mock_model(x)
                handle.remove()

        # o[0] is attn slot; multi_patch's block hook replaces o[2] (resid),
        # but attn_out is baked into o[0] from inside block.forward.
        attn_out, _, _ = block0_outputs[0]
        assert torch.allclose(attn_out, constant)

    def test_resid_transform_produces_expected_algebraic_output(self, mock_model):
        """
        Replace layer 0 resid with zeros.  Layer 1 then computes on zeros:
            attn_out_1  = 0 * 0.1 = 0
            mlp_out_1   = 0 * 0.2 = 0
            resid_1     = 0 + 0 + 0 = 0
        Final model output should be all zeros.
        """
        def transform(l, k, t):
            return torch.zeros_like(t) if (l == 0 and k == "resid") else None

        x = torch.randn(B, T, N)
        with torch.no_grad():
            with multi_patch(mock_model, transform):
                result = mock_model(x)

        assert torch.allclose(result, torch.zeros(B, T, N), atol=1e-6)

    def test_layer_specific_transform_only_modifies_target(self, mock_model):
        """Transform for (layer=0, kind='resid') should fire exactly once."""
        fired = []

        def transform(l, k, t):
            if l == 0 and k == "resid":
                fired.append(True)
                return t * 0.0
            return None

        x = torch.randn(B, T, N)
        with torch.no_grad():
            with multi_patch(mock_model, transform):
                mock_model(x)

        assert len(fired) == 1

    def test_hooks_removed_after_context(self, mock_model):
        """After the context, destructive patching has no lasting effect."""
        x = torch.randn(B, T, N)
        ref = _run(mock_model, x).clone()

        with torch.no_grad():
            with multi_patch(mock_model, lambda l, k, t: torch.zeros_like(t)):
                mock_model(x)

        after = _run(mock_model, x).clone()
        assert torch.allclose(ref, after)

    def test_no_hooks_remain_after_context_via_hook_count(self, mock_model):
        """
        After the context, no extra forward hooks should be attached to any
        block or its submodules.  We count hooks before and after.
        """
        def _count_hooks(model):
            count = 0
            for block in model.transformer.h:
                count += len(block._forward_hooks)
                count += len(block.attn._forward_hooks)
                count += len(block.mlp._forward_hooks)
                count += len(block.norm_2._forward_pre_hooks)
            return count

        x = torch.randn(B, T, N)
        before = _count_hooks(mock_model)

        with torch.no_grad():
            with multi_patch(mock_model, lambda l, k, t: None):
                pass

        assert _count_hooks(mock_model) == before
