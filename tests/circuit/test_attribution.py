"""
Phase 5 — Attribution function tests.

Tests compute_logit_attribution, compute_feature_attribution, and the legacy
compute_attribution using a fully wired synthetic computation graph produced by
SAEGraphInstrument.  All tests run on CPU without loading real model weights.

Setup strategy
--------------
attr_setup — forward pass under SAEGraphInstrument (stop_error_grad=False) with
             x.requires_grad=True.  This ensures:
               • top_acts_connected has grad_fn at every (layer, kind) — needed for
                 compute_feature_attribution's target_sum.grad_fn guard.
               • output depends on every leaf anchor via the error-term path —
                 needed for compute_logit_attribution to return non-zero scores.
             logits [B, T, V] = output @ W_logit are then also in the graph.
"""
import pytest
import torch

from circuit.attribution import (
    compute_logit_attribution,
    compute_feature_attribution,
    compute_attribution,
)
from circuit.sae_graph import FeatureGraph, SAEGraphInstrument

# ---------------------------------------------------------------------------
# Local constants
# ---------------------------------------------------------------------------

B, T    = 2, 4          # batch size, sequence length
D_MODEL = 16
D_SAE   = 32
K_SAE   = 4
N_LAYERS = 2
KINDS   = ["attn", "mlp", "resid"]
V       = 20            # tiny vocabulary for logit projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_valid_target(graph: FeatureGraph, layer: int, kind: str,
                      pos_argmax: torch.Tensor) -> int:
    """Return the first top-k latent at pos_argmax[0] for batch item 0."""
    _, _, indices = graph.get_latents(layer, kind)
    return int(indices[0, int(pos_argmax[0].item()), 0].item())


def _get_dormant_latent(graph: FeatureGraph, layer: int, kind: str,
                        pos_argmax: torch.Tensor) -> int:
    """Return a latent index NOT present in top_indices at pos_argmax for any batch item."""
    _, _, indices = graph.get_latents(layer, kind)
    batch_idx = torch.arange(B)
    active = set(int(v.item()) for v in indices[batch_idx, pos_argmax].flatten())
    return next(i for i in range(D_SAE) if i not in active)


# ---------------------------------------------------------------------------
# Fixture — fully wired computation graph
# ---------------------------------------------------------------------------

@pytest.fixture
def attr_setup(mock_sae_bank, mock_model):
    """
    Returns (graph, logits, pos_argmax, target_tokens) where:

    - graph          FeatureGraph populated with stop_error_grad=False so that
                     all leaf anchors are reachable from logits via error terms.
    - logits         [B, T, V] in the computation graph (logits = output @ W_logit).
    - pos_argmax     [B] probe token positions.
    - target_tokens  [B, T] valid next-token indices.
    """
    torch.manual_seed(42)
    W_logit    = torch.randn(D_MODEL, V)
    pos_argmax = torch.tensor([1, 2])

    instrument = SAEGraphInstrument(mock_sae_bank, stop_error_grad=False)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)

    with torch.enable_grad():
        with instrument(mock_model):
            output = mock_model(x)
        logits = output @ W_logit          # [B, T, V], still in computation graph

    target_tokens = torch.randint(0, V, (B, T))
    return instrument.graph, logits, pos_argmax, target_tokens


# ---------------------------------------------------------------------------
# TestComputeLogitAttribution
# ---------------------------------------------------------------------------

class TestComputeLogitAttribution:

    def test_returns_dict_with_layer_kind_latent_keys(self, attr_setup):
        graph, logits, pos_argmax, target_tokens = attr_setup
        result = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)
        for key in result.keys():
            assert isinstance(key, tuple) and len(key) == 3
            layer, kind, latent = key
            assert isinstance(layer, int)
            assert isinstance(kind, str) and kind in KINDS
            assert isinstance(latent, int)

    def test_nonempty_result_for_connected_graph(self, attr_setup):
        """At least one latent should receive a non-zero attribution score."""
        graph, logits, pos_argmax, target_tokens = attr_setup
        result = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)
        assert len(result) > 0

    def test_empty_graph_returns_empty_dict(self, mock_sae_bank):
        empty_graph   = FeatureGraph(torch.device("cpu"))
        logits        = torch.randn(B, T, V, requires_grad=True)
        pos_argmax    = torch.zeros(B, dtype=torch.long)
        target_tokens = torch.randint(0, V, (B, T))
        result = compute_logit_attribution(empty_graph, logits, pos_argmax, target_tokens)
        assert result == {}

    def test_all_scores_are_python_floats(self, attr_setup):
        graph, logits, pos_argmax, target_tokens = attr_setup
        result = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)
        for key, score in result.items():
            assert isinstance(score, float), (
                f"Score for {key} is {type(score).__name__}, expected float"
            )

    def test_result_latents_are_subset_of_active_top_k(self, attr_setup):
        """Every returned latent index must exist somewhere in the graph's top_indices."""
        graph, logits, pos_argmax, target_tokens = attr_setup
        result = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)

        all_active: set = set()
        for steps in graph.activations.values():
            for _, _, indices in steps:
                all_active.update(int(v.item()) for v in indices.flatten())

        for (layer, kind, i) in result.keys():
            assert i in all_active, (
                f"Latent {i} at ({layer},{kind}) is not in any top_indices entry"
            )

    def test_changing_target_token_changes_scores(self, attr_setup):
        """Swapping the target tokens at the probe position changes the scores."""
        graph, logits, pos_argmax, target_tokens = attr_setup
        result_a = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)

        alt_tokens = (target_tokens + 1) % V
        result_b   = compute_logit_attribution(graph, logits, pos_argmax, alt_tokens)

        shared = set(result_a) & set(result_b)
        keys_differ = set(result_a) != set(result_b)
        vals_differ = any(result_a[k] != result_b[k] for k in shared)
        assert keys_differ or vals_differ, (
            "Changing target tokens had no effect on any attribution score"
        )

    def test_changing_pos_argmax_changes_scores(self, attr_setup):
        """Swapping probe positions changes the backward target and therefore the scores."""
        graph, logits, pos_argmax, target_tokens = attr_setup
        result_a = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)

        alt_pos  = torch.tensor([2, 1])        # swap batch-0 / batch-1 probe positions
        result_b = compute_logit_attribution(graph, logits, alt_pos, target_tokens)

        shared = set(result_a) & set(result_b)
        keys_differ = set(result_a) != set(result_b)
        vals_differ = any(result_a[k] != result_b[k] for k in shared)
        assert keys_differ or vals_differ, (
            "Swapping pos_argmax had no effect on any attribution score"
        )


# ---------------------------------------------------------------------------
# TestComputeFeatureAttribution
# ---------------------------------------------------------------------------

class TestComputeFeatureAttribution:

    def test_returns_nonzero_for_valid_target_latent(self, attr_setup):
        graph, _, pos_argmax, _ = attr_setup
        target_layer, target_kind = 1, "resid"
        target_latent = _get_valid_target(graph, target_layer, target_kind, pos_argmax)

        result = compute_feature_attribution(
            graph, target_layer, target_kind, target_latent, pos_argmax
        )
        assert len(result) > 0

    def test_no_match_at_probe_position_returns_empty_dict(self, attr_setup):
        graph, _, pos_argmax, _ = attr_setup
        target_layer, target_kind = 1, "attn"
        dormant = _get_dormant_latent(graph, target_layer, target_kind, pos_argmax)

        result = compute_feature_attribution(
            graph, target_layer, target_kind, dormant, pos_argmax
        )
        assert result == {}

    def test_no_grad_fn_on_connected_acts_returns_empty_dict(self, mock_sae_bank, mock_model):
        """top_acts_connected captured under no_grad has no grad_fn → return {}."""
        instrument = SAEGraphInstrument(mock_sae_bank)
        x = torch.randn(B, T, D_MODEL)

        with torch.no_grad():
            with instrument(mock_model):
                mock_model(x)

        graph = instrument.graph
        layer, kind = 0, "attn"
        _, acts_conn, indices = graph.get_latents(layer, kind)
        assert acts_conn.grad_fn is None, "Precondition failed: expected grad_fn=None"

        pos_argmax    = torch.zeros(B, dtype=torch.long)
        target_latent = int(indices[0, 0, 0].item())

        result = compute_feature_attribution(graph, layer, kind, target_latent, pos_argmax)
        assert result == {}

    def test_skips_layers_above_target_layer(self, attr_setup):
        """With target_layer=0, no entries for layer 1 should appear."""
        graph, _, pos_argmax, _ = attr_setup
        target_layer, target_kind = 0, "resid"
        target_latent = _get_valid_target(graph, target_layer, target_kind, pos_argmax)

        result = compute_feature_attribution(
            graph, target_layer, target_kind, target_latent, pos_argmax
        )

        for (l, _k, _i) in result.keys():
            assert l <= target_layer, (
                f"Layer {l} > target_layer {target_layer} appeared in result"
            )

    def test_candidate_nodes_restricts_output_keys(self, attr_setup):
        """Only the explicitly listed candidate nodes should appear in the result."""
        graph, _, pos_argmax, _ = attr_setup
        target_layer, target_kind = 1, "mlp"
        target_latent = _get_valid_target(graph, target_layer, target_kind, pos_argmax)

        cand_layer, cand_kind = 0, "attn"
        cand_latent = _get_valid_target(graph, cand_layer, cand_kind, pos_argmax)

        result = compute_feature_attribution(
            graph, target_layer, target_kind, target_latent, pos_argmax,
            candidate_nodes=[(cand_layer, cand_kind, cand_latent)],
        )

        allowed = {(cand_layer, cand_kind, cand_latent)}
        for key in result.keys():
            assert key in allowed, f"Unexpected key {key} outside candidate_nodes"

    def test_none_candidates_is_superset_of_restricted_result(self, attr_setup):
        """candidate_nodes=None returns at least as many scored nodes as any restriction."""
        graph, _, pos_argmax, _ = attr_setup
        target_layer, target_kind = 1, "resid"
        target_latent = _get_valid_target(graph, target_layer, target_kind, pos_argmax)

        result_none = compute_feature_attribution(
            graph, target_layer, target_kind, target_latent, pos_argmax,
            candidate_nodes=None,
        )

        cand_layer, cand_kind = 0, "attn"
        cand_latent = _get_valid_target(graph, cand_layer, cand_kind, pos_argmax)
        result_restricted = compute_feature_attribution(
            graph, target_layer, target_kind, target_latent, pos_argmax,
            candidate_nodes=[(cand_layer, cand_kind, cand_latent)],
        )

        # The unrestricted result must cover all keys the restricted version found.
        assert set(result_restricted.keys()) <= set(result_none.keys())
        assert len(result_none) >= len(result_restricted)

    def test_empty_candidate_list_returns_empty_dict(self, attr_setup):
        """candidate_nodes=[] iterates nothing and must return {}."""
        graph, _, pos_argmax, _ = attr_setup
        target_layer, target_kind = 1, "resid"
        target_latent = _get_valid_target(graph, target_layer, target_kind, pos_argmax)

        result = compute_feature_attribution(
            graph, target_layer, target_kind, target_latent, pos_argmax,
            candidate_nodes=[],
        )
        assert result == {}


# ---------------------------------------------------------------------------
# TestComputeAttributionLegacy
# ---------------------------------------------------------------------------

class TestComputeAttributionLegacy:

    def test_returns_attribution_for_valid_target_latent(self, attr_setup):
        graph, _, pos_argmax, _ = attr_setup
        layer, kind = 1, "attn"
        target_latent = _get_valid_target(graph, layer, kind, pos_argmax)

        result = compute_attribution(graph, layer, kind, target_latent, pos_argmax)

        assert len(result) > 0
        assert (layer, kind, target_latent) in result

    def test_no_match_at_probe_position_returns_empty_dict(self, attr_setup):
        graph, _, pos_argmax, _ = attr_setup
        layer, kind = 0, "mlp"
        dormant = _get_dormant_latent(graph, layer, kind, pos_argmax)

        result = compute_attribution(graph, layer, kind, dormant, pos_argmax)
        assert result == {}

    def test_score_for_target_latent_equals_target_activation_sum(self, attr_setup):
        """
        Algebraic identity for the legacy function.

        target_sum = acts_grad[b, pos_argmax[b], k].sum()  (over k matching target_latent)
        d(target_sum)/d(acts_grad[b,t,k]) = 1  iff t==pos_argmax[b] and index matches, else 0
        attr_tensor[b,t,k]               = acts_grad[b,t,k] * grad[b,t,k]
        score for target_latent           = target_sum.item()
        """
        graph, _, pos_argmax, _ = attr_setup
        layer, kind = 1, "resid"
        acts_grad, _, indices = graph.get_latents(layer, kind)

        batch_idx     = torch.arange(B)
        target_latent = int(indices[batch_idx, pos_argmax][0, 0].item())

        vals_at_pos   = indices[batch_idx, pos_argmax]   # [B, K]
        matches       = (vals_at_pos == target_latent)   # [B, K]
        expected_score = acts_grad.data[batch_idx, pos_argmax][matches].sum().item()

        result = compute_attribution(graph, layer, kind, target_latent, pos_argmax)

        assert (layer, kind, target_latent) in result
        assert abs(result[(layer, kind, target_latent)] - expected_score) < 1e-5

    def test_cross_layer_anchors_absent_for_same_layer_backward(self, attr_setup):
        """
        The legacy backward target is top_acts_grad — a detached leaf with no path
        to earlier layers.  Therefore only same-layer entries should appear in the result.
        """
        graph, _, pos_argmax, _ = attr_setup
        layer, kind = 1, "mlp"
        target_latent = _get_valid_target(graph, layer, kind, pos_argmax)

        result = compute_attribution(graph, layer, kind, target_latent, pos_argmax)

        for (l, _k, _i) in result.keys():
            assert l == layer, (
                f"Legacy attribution returned entry for layer {l}, expected only layer {layer}"
            )
