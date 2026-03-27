"""
Phase 4 — SAE Graph & Gradient Infrastructure tests.

Tests FeatureGraph and SAEGraphInstrument using MockSAEBank from conftest.py.
No model weights or CUDA required; all tests run on CPU.

Key invariants under test:

  top_acts_grad      — detached leaf (requires_grad=True, is_leaf=True)
                       gradient accumulates here during attribution backward passes
  top_acts_connected — original encoder output, still in computation graph;
                       has grad_fn when called under torch.enable_grad() with
                       a tensor that requires grad
  top_indices        — always detached (discrete, no gradient through indices)
  output of transform — reconstruction + error == x  (exact identity, not approximate)
  stop_error_grad=False — backward from output reaches x via error = x - full_recon
  stop_error_grad=True  — error is detached; backward from output cannot reach x
"""
import pytest
import torch

from circuit.sae_graph import FeatureGraph, SAEGraphInstrument

# Dimensions shared with conftest constants.
B, T    = 2, 4
D_MODEL = 16
D_SAE   = 32
K_SAE   = 4
N_LAYERS = 2
KINDS    = ["attn", "mlp", "resid"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaf(shape=None):
    """Create a detached leaf tensor with requires_grad=True."""
    if shape is None:
        shape = (B, T, K_SAE)
    return torch.randn(*shape).requires_grad_(True)


def _conn(shape=None):
    """Create a graph-connected tensor (has grad_fn) via a simple operation."""
    if shape is None:
        shape = (B, T, K_SAE)
    root = torch.randn(*shape, requires_grad=True)
    return root * 1.0   # trivial op gives it a grad_fn while keeping values


def _indices(shape=None):
    if shape is None:
        shape = (B, T, K_SAE)
    return torch.randint(0, D_SAE, shape)


# ---------------------------------------------------------------------------
# TestFeatureGraph
# ---------------------------------------------------------------------------

class TestFeatureGraph:
    def test_add_and_get_returns_original_tuple(self):
        graph = FeatureGraph(torch.device("cpu"))
        ag, ac, idx = _leaf(), _conn(), _indices()
        graph.add(0, "attn", ag, ac, idx)

        got_ag, got_ac, got_idx = graph.get_latents(0, "attn", step=0)
        assert got_ag.act is ag
        assert got_ac.act is ac
        assert got_idx is idx

    def test_multiple_steps_same_key_appended_in_order(self):
        graph = FeatureGraph(torch.device("cpu"))
        ag0, ag1 = _leaf(), _leaf()
        ac0, ac1 = _conn(), _conn()
        idx0, idx1 = _indices(), _indices()

        graph.add(0, "attn", ag0, ac0, idx0)
        graph.add(0, "attn", ag1, ac1, idx1)

        r0_ag, _, _ = graph.get_latents(0, "attn", step=0)
        r1_ag, _, _ = graph.get_latents(0, "attn", step=1)
        assert r0_ag.act is ag0
        assert r1_ag.act is ag1

    def test_different_keys_stored_independently(self):
        graph = FeatureGraph(torch.device("cpu"))
        ag_a, ac_a, idx_a = _leaf(), _conn(), _indices()
        ag_m, ac_m, idx_m = _leaf(), _conn(), _indices()

        graph.add(0, "attn", ag_a, ac_a, idx_a)
        graph.add(0, "mlp",  ag_m, ac_m, idx_m)

        got_ag_a, _, _ = graph.get_latents(0, "attn")
        got_ag_m, _, _ = graph.get_latents(0, "mlp")
        assert got_ag_a.act is ag_a
        assert got_ag_m.act is ag_m

    def test_different_layers_stored_independently(self):
        graph = FeatureGraph(torch.device("cpu"))
        ag0, ag1 = _leaf(), _leaf()
        ac, idx = _conn(), _indices()

        graph.add(0, "resid", ag0, ac, idx)
        graph.add(1, "resid", ag1, ac, idx)

        r0, _, _ = graph.get_latents(0, "resid")
        r1, _, _ = graph.get_latents(1, "resid")
        assert r0.act is ag0
        assert r1.act is ag1

    def test_all_anchors_empty_graph_returns_empty_list(self):
        graph = FeatureGraph(torch.device("cpu"))
        assert graph.all_anchors() == []

    def test_all_anchors_returns_only_the_leaf_tensors_not_connected(self):
        """
        all_anchors() must return top_acts_grad (slot 0 of each 3-tuple),
        not top_acts_connected (slot 1).  Both have requires_grad=True in
        this test; identity is verified with `is`.
        """
        graph = FeatureGraph(torch.device("cpu"))
        ag = _leaf()
        ac = _leaf()   # also has requires_grad — identity check is critical
        graph.add(0, "attn", ag, ac, _indices())

        anchors = graph.all_anchors()
        assert len(anchors) == 1
        assert anchors[0] is ag
        assert anchors[0] is not ac

    def test_all_anchors_count_equals_total_add_calls(self):
        """One anchor per add() call, regardless of which (layer, kind) it belongs to."""
        graph = FeatureGraph(torch.device("cpu"))
        for l in range(2):
            for kind in ["attn", "mlp", "resid"]:
                graph.add(l, kind, _leaf(), _conn(), _indices())
        assert len(graph.all_anchors()) == 6   # 2 layers × 3 kinds

    def test_all_anchors_are_leaves_with_requires_grad(self):
        graph = FeatureGraph(torch.device("cpu"))
        for kind in KINDS:
            graph.add(0, kind, _leaf(), _conn(), _indices())

        for anchor in graph.all_anchors():
            assert anchor.is_leaf, "anchor must be a leaf tensor"
            assert anchor.requires_grad, "anchor must require grad"

    def test_all_anchors_accumulate_across_multiple_steps(self):
        """Two add() calls for the same key yield two anchors."""
        graph = FeatureGraph(torch.device("cpu"))
        graph.add(0, "attn", _leaf(), _conn(), _indices())
        graph.add(0, "attn", _leaf(), _conn(), _indices())
        assert len(graph.all_anchors()) == 2


# ---------------------------------------------------------------------------
# TestSAEGraphInstrument — transform() behaviour
# ---------------------------------------------------------------------------

class TestSAEGraphInstrumentTransform:
    def test_leaf_anchor_is_detached_and_requires_grad(self, mock_sae_bank):
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)

        with torch.no_grad():
            instrument.transform(0, "attn", x)

        acts_grad, _, _ = instrument.graph.get_latents(0, "attn")
        assert acts_grad.act.is_leaf
        assert acts_grad.act.requires_grad

    def test_leaf_anchor_has_same_values_as_top_acts(self, mock_sae_bank):
        """
        top_acts_grad = top_acts.detach().requires_grad_(True)
        Values must be identical; only the computation-graph connection differs.
        """
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)

        with torch.enable_grad():
            instrument.transform(0, "attn", x)

        acts_grad, acts_conn, _ = instrument.graph.get_latents(0, "attn")
        assert torch.allclose(acts_grad.act.detach(), acts_conn.act.detach(), atol=1e-6)

    def test_connected_acts_have_grad_fn_under_enable_grad(self, mock_sae_bank):
        """
        top_acts_connected = the raw encoder output, still in the computation graph.
        Under torch.enable_grad() with x that has requires_grad, it must have grad_fn.
        """
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL, requires_grad=True)

        with torch.enable_grad():
            instrument.transform(0, "attn", x)

        _, acts_conn, _ = instrument.graph.get_latents(0, "attn")
        assert acts_conn.act.grad_fn is not None

    def test_stored_indices_are_detached(self, mock_sae_bank):
        """top_indices must always be detached (no gradient through discrete indices)."""
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL, requires_grad=True)

        with torch.enable_grad():
            instrument.transform(0, "attn", x)

        _, _, indices = instrument.graph.get_latents(0, "attn")
        assert indices.grad_fn is None

    def test_output_equals_x_exactly(self, mock_sae_bank):
        """
        reconstruction + error == x exactly, because:
            reconstruction = decode(scatter(top_acts_grad))
            error          = x - decode(scatter(top_acts))
        and top_acts_grad has identical values to top_acts (just detached),
        so decode(scatter(top_acts_grad)) == decode(scatter(top_acts)) == full_recon.
        Therefore output = full_recon + (x - full_recon) = x.
        """
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)

        with torch.no_grad():
            output = instrument.transform(0, "attn", x)

        assert torch.allclose(output, x, atol=1e-5)

    def test_output_shape_matches_input_shape(self, mock_sae_bank):
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)
        with torch.no_grad():
            output = instrument.transform(0, "mlp", x)
        assert output.shape == x.shape

    def test_transform_adds_entry_to_graph(self, mock_sae_bank):
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)
        assert (0, "attn") not in instrument.graph.activations

        with torch.no_grad():
            instrument.transform(0, "attn", x)

        assert (0, "attn") in instrument.graph.activations
        assert len(instrument.graph.activations[(0, "attn")]) == 1

    def test_repeated_calls_accumulate_entries_for_same_key(self, mock_sae_bank):
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)
        with torch.no_grad():
            instrument.transform(0, "attn", x)
            instrument.transform(0, "attn", x)
        assert len(instrument.graph.activations[(0, "attn")]) == 2

    def test_different_layer_kind_pairs_add_separate_entries(self, mock_sae_bank):
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)
        with torch.no_grad():
            instrument.transform(0, "attn", x)
            instrument.transform(0, "mlp",  x)
            instrument.transform(1, "resid", x)
        assert len(instrument.graph.activations) == 3


# ---------------------------------------------------------------------------
# TestSAEGraphInstrumentGradients — gradient flow correctness
# ---------------------------------------------------------------------------

class TestSAEGraphInstrumentGradients:
    def test_gradient_flows_to_leaf_anchor_after_backward(self, mock_sae_bank):
        """
        Backpropagating from output.sum() must populate acts_grad.grad.
        Path: output.sum() ← output ← reconstruction ← decode(scatter(acts_grad)) ← acts_grad
        """
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)

        with torch.enable_grad():
            output = instrument.transform(0, "attn", x)
            output.sum().backward()

        acts_grad, _, _ = instrument.graph.get_latents(0, "attn")
        assert acts_grad.act.grad is not None
        assert acts_grad.act.grad.shape == acts_grad.act.shape

    def test_leaf_anchor_grad_is_nonzero(self, mock_sae_bank):
        """The gradient at acts_grad should be non-zero for generic random x."""
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)

        with torch.enable_grad():
            output = instrument.transform(0, "attn", x)
            output.sum().backward()

        acts_grad, _, _ = instrument.graph.get_latents(0, "attn")
        assert acts_grad.act.grad.abs().sum().item() > 0.0

    def test_stop_error_grad_false_allows_gradient_to_reach_x(self, mock_sae_bank):
        """
        stop_error_grad=False (default): error = x - full_recon contains x directly,
        so backward from output reaches x.grad via the error-term path.
        """
        instrument = SAEGraphInstrument(mock_sae_bank, stop_error_grad=False)
        x = torch.randn(B, T, D_MODEL, requires_grad=True)

        with torch.enable_grad():
            output = instrument.transform(0, "attn", x)
            output.sum().backward()

        assert x.grad is not None, (
            "Gradient should flow through the error term (x - full_recon) to x"
        )

    def test_stop_error_grad_true_blocks_gradient_to_x(self, mock_sae_bank):
        """
        stop_error_grad=True: error is detached before return.
        output = reconstruction + detached_error
        reconstruction depends only on acts_grad (leaf), NOT on x.
        Therefore backward from output cannot reach x.grad.
        """
        instrument = SAEGraphInstrument(mock_sae_bank, stop_error_grad=True)
        x = torch.randn(B, T, D_MODEL, requires_grad=True)

        with torch.enable_grad():
            output = instrument.transform(0, "attn", x)
            output.sum().backward()

        assert x.grad is None, (
            "Gradient must be blocked when stop_error_grad=True"
        )

    def test_stop_error_grad_false_output_has_grad_fn_through_x(self, mock_sae_bank):
        """
        With stop_error_grad=False and x.requires_grad=True, the output
        should have a grad_fn that traces back through x (via error = x - full_recon).
        """
        instrument = SAEGraphInstrument(mock_sae_bank, stop_error_grad=False)
        x = torch.randn(B, T, D_MODEL, requires_grad=True)

        with torch.enable_grad():
            output = instrument.transform(0, "attn", x)

        assert output.grad_fn is not None

    def test_stop_error_grad_true_output_still_has_grad_fn_via_reconstruction(
        self, mock_sae_bank
    ):
        """
        Even with stop_error_grad=True, the output has grad_fn because
        reconstruction = decode(scatter(acts_grad)) is still in the graph
        (acts_grad is a leaf with requires_grad=True).
        """
        instrument = SAEGraphInstrument(mock_sae_bank, stop_error_grad=True)
        x = torch.randn(B, T, D_MODEL)

        with torch.enable_grad():
            output = instrument.transform(0, "attn", x)

        assert output.grad_fn is not None

    def test_multiple_anchors_all_receive_gradients(self, mock_sae_bank):
        """
        After calling transform for several (layer, kind) pairs and doing
        a single backward, every leaf anchor should have a non-None .grad.
        (Uses stop_error_grad=False so all paths are live.)
        """
        instrument = SAEGraphInstrument(mock_sae_bank, stop_error_grad=False)
        x = torch.randn(B, T, D_MODEL)

        with torch.enable_grad():
            total = torch.zeros(1)
            for kind in KINDS:
                total = total + instrument.transform(0, kind, x).sum()
            total.backward()

        for anchor in instrument.graph.all_anchors():
            assert anchor.grad is not None


# ---------------------------------------------------------------------------
# TestSAEGraphInstrumentContextManager — __call__ and model integration
# ---------------------------------------------------------------------------

class TestSAEGraphInstrumentContextManager:
    def test_call_returns_object_with_enter_and_exit(self, mock_sae_bank, mock_model):
        instrument = SAEGraphInstrument(mock_sae_bank)
        ctx = instrument(mock_model)
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")

    def test_graph_populated_for_all_layers_and_kinds_after_forward(
        self, mock_sae_bank, mock_model
    ):
        """
        Running the mock model under the instrument context should call transform
        for every (layer, kind) combination, populating the graph completely.
        MockModel has N_LAYERS=2 and 3 kinds → 6 (layer, kind) entries expected.
        """
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)

        with torch.no_grad():
            with instrument(mock_model):
                mock_model(x)

        for l in range(N_LAYERS):
            for kind in KINDS:
                assert (l, kind) in instrument.graph.activations, (
                    f"Missing graph entry for layer {l}, kind {kind}"
                )

    def test_each_layer_kind_has_exactly_one_step_after_single_forward(
        self, mock_sae_bank, mock_model
    ):
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)

        with torch.no_grad():
            with instrument(mock_model):
                mock_model(x)

        for key, steps in instrument.graph.activations.items():
            assert len(steps) == 1, f"Expected 1 step for {key}, got {len(steps)}"

    def test_model_output_unchanged_after_context_exits(self, mock_sae_bank, mock_model):
        """
        After the context manager exits, model hooks are removed and
        the model runs identically to the pre-instrumentation reference.
        """
        x = torch.randn(B, T, D_MODEL)
        with torch.no_grad():
            ref = mock_model(x).clone()

        instrument = SAEGraphInstrument(mock_sae_bank)
        with torch.no_grad():
            with instrument(mock_model):
                mock_model(x)

        with torch.no_grad():
            after = mock_model(x).clone()

        assert torch.allclose(ref, after)

    def test_all_leaf_anchors_receive_grad_after_backward_through_context(
        self, mock_sae_bank, mock_model
    ):
        """
        End-to-end gradient test:
        Run model under instrument context with enable_grad.
        Every leaf anchor in the populated graph must receive a non-None .grad
        after backward from the model's final output.

        Uses stop_error_grad=False so gradients can flow across layers via error terms.
        """
        instrument = SAEGraphInstrument(mock_sae_bank, stop_error_grad=False)
        x = torch.randn(B, T, D_MODEL)

        with torch.enable_grad():
            with instrument(mock_model):
                output = mock_model(x)
            output.sum().backward()

        anchors = instrument.graph.all_anchors()
        # Each layer/kind now produces 2 anchors (activations and residual error)
        # unless stop_error_grad=True, but here it's False.
        assert len(anchors) == N_LAYERS * len(KINDS) * 2, (
            "Expected two anchors per (layer, kind) pair (act + res)"
        )
        for anchor in anchors:
            assert anchor.grad is not None, "Every leaf anchor must receive a gradient"

    def test_leaf_anchors_are_leaves_in_populated_graph(
        self, mock_sae_bank, mock_model
    ):
        """After context forward, every stored acts_grad must still be a leaf."""
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)

        with torch.no_grad():
            with instrument(mock_model):
                mock_model(x)

        for anchor in instrument.graph.all_anchors():
            assert anchor.is_leaf
            assert anchor.requires_grad
