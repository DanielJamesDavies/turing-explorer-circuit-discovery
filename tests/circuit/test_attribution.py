import pytest
import torch

from circuit.instrument.attribution import (
    compute_logit_attribution,
    compute_feature_attribution,
    compute_feature_gradient,
    compute_latent_upstream_scores,
    UpstreamScores,
)
from circuit.instrument.sae_graph import FeatureGraph, SAEGraphInstrument
from circuit.types.sparse_act import SparseAct
from circuit.types.feature_id import FeatureID

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

    def test_returns_dict_with_feature_id_keys(self, attr_setup):
        graph, logits, pos_argmax, target_tokens = attr_setup
        result = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)
        for key in result.keys():
            assert isinstance(key, FeatureID)
            assert key.layer < N_LAYERS
            assert key.kind in KINDS
            assert key.index < D_SAE

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

        for fid in result.keys():
            assert fid.index in all_active, (
                f"Latent {fid.index} at ({fid.layer},{fid.kind}) is not in any top_indices entry"
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

        for fid in result.keys():
            assert fid.layer <= target_layer, (
                f"Layer {fid.layer} > target_layer {target_layer} appeared in result"
            )

    def test_candidate_nodes_restricts_output_keys(self, attr_setup):
        """Only the explicitly listed candidate nodes should appear in the result."""
        graph, _, pos_argmax, _ = attr_setup
        target_layer, target_kind = 1, "mlp"
        target_latent = _get_valid_target(graph, target_layer, target_kind, pos_argmax)

        cand_layer, cand_kind = 0, "attn"
        cand_latent = _get_valid_target(graph, cand_layer, cand_kind, pos_argmax)
        cand_fid = FeatureID(cand_layer, cand_kind, cand_latent)

        result = compute_feature_attribution(
            graph, target_layer, target_kind, target_latent, pos_argmax,
            candidate_nodes=[cand_fid],
        )

        allowed = {cand_fid}
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
        cand_fid = FeatureID(cand_layer, cand_kind, cand_latent)
        result_restricted = compute_feature_attribution(
            graph, target_layer, target_kind, target_latent, pos_argmax,
            candidate_nodes=[cand_fid],
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
# Oracle helpers — build graphs with analytically known gradients
# ---------------------------------------------------------------------------

def _build_logit_oracle_graph():
    """
    Minimal graph for logit attribution with hand-computable scores.

    Structure (B=1, T=1, V=2, D_SAE=3):
      - Single entry: layer 0, kind 'attn'
      - f_grad = [2.0, 3.0, 1.0] (leaf anchor)
      - logits = f_grad @ W,  W = [[1, 0.5], [0, 1], [2, -1]]
      - target token = 0, pos_argmax = 0

    Hand-computed:
      target_scalar = logits[0, 0, 0] = 2*1 + 3*0 + 1*2 = 4.0
      grad = W[:, 0] = [1.0, 0.0, 2.0]
      attribution[i] = f_vals[i] * grad[i]
        i=0: 2.0 * 1.0 = 2.0
        i=1: 3.0 * 0.0 = 0.0  (skipped)
        i=2: 1.0 * 2.0 = 2.0
    """
    W = torch.tensor([[1.0, 0.5], [0.0, 1.0], [2.0, -1.0]])
    f_vals = torch.tensor([[[2.0, 3.0, 1.0]]])            # [1, 1, 3]
    f_grad = f_vals.detach().clone().requires_grad_(True)
    logits = f_grad @ W                                    # [1, 1, 2]

    graph = FeatureGraph(torch.device("cpu"))
    graph.add(0, "attn",
              SparseAct(act=f_grad),
              SparseAct(act=f_vals.clone()),
              torch.tensor([[[0, 1, 2]]]))

    pos_argmax    = torch.tensor([0])
    target_tokens = torch.tensor([[0]])

    expected = {
        FeatureID(0, "attn", 0): 2.0,
        FeatureID(0, "attn", 2): 2.0,
    }
    return graph, logits, pos_argmax, target_tokens, expected


def _build_multibatch_logit_oracle_graph():
    """
    Multi-batch logit oracle (B=2, T=1, V=2, D_SAE=3).

    f_grad = [[[2, 3, 1]], [[1, 0, 4]]]
    W = [[1, 0.5], [0, 1], [2, -1]]
    Both target token = 0, pos_argmax = [0, 0].

    target_scalar = logits[0,0,0] + logits[1,0,0]
    grad[b, 0, :] = W[:, 0] = [1, 0, 2]  for both batches

    attribution[i] = sum_b( f_grad[b,0,i] * W[i,0] )
      i=0: 2*1 + 1*1 = 3.0
      i=1: 3*0 + 0*0 = 0.0  (skipped)
      i=2: 1*2 + 4*2 = 10.0
    """
    W = torch.tensor([[1.0, 0.5], [0.0, 1.0], [2.0, -1.0]])
    f_vals = torch.tensor([[[2.0, 3.0, 1.0]], [[1.0, 0.0, 4.0]]])  # [2, 1, 3]
    f_grad = f_vals.detach().clone().requires_grad_(True)
    logits = f_grad @ W

    graph = FeatureGraph(torch.device("cpu"))
    graph.add(0, "attn",
              SparseAct(act=f_grad),
              SparseAct(act=f_vals.clone()),
              torch.tensor([[[0, 1, 2]], [[0, 1, 2]]]))

    pos_argmax    = torch.tensor([0, 0])
    target_tokens = torch.tensor([[0], [0]])

    expected = {
        FeatureID(0, "attn", 0): 3.0,
        FeatureID(0, "attn", 2): 10.0,
    }
    return graph, logits, pos_argmax, target_tokens, expected


def _build_feature_oracle_graph():
    """
    Two-layer graph for feature / gradient attribution with known scores.

    Layer 0 ('attn', D_SAE=3) — upstream leaf anchor:
      f0_grad = [2.0, 3.0, 1.0]  (requires_grad leaf)

    Layer 1 ('attn', D_SAE=2) — target, connected to f0_grad:
      f1_connected.act = f0_grad @ M,  M = [[1, 0], [0, 1], [0.5, -1]]
      f1_grad = detached leaf (not connected to target_sum)

    target_latent_idx = 0, pos_argmax = [0]

    target_sum = (f0_grad @ M)[0, 0, 0]
               = 2*1 + 3*0 + 1*0.5 = 2.5

    grad = d(target_sum)/d(f0_grad) = M[:, 0] = [1.0, 0.0, 0.5]

    Feature attribution (act * grad):
      i=0: 2.0 * 1.0 = 2.0
      i=1: 3.0 * 0.0 = 0.0  (skipped)
      i=2: 1.0 * 0.5 = 0.5

    Feature gradient (raw grad):
      i=0: 1.0
      i=1: 0.0  (skipped)
      i=2: 0.5
    """
    M = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -1.0]])  # [3, 2]
    f0_vals = torch.tensor([[[2.0, 3.0, 1.0]]])               # [1, 1, 3]
    f0_grad = f0_vals.detach().clone().requires_grad_(True)

    f1_act_connected = f0_grad @ M                             # [1, 1, 2]
    f1_grad_act = f1_act_connected.detach().clone().requires_grad_(True)

    graph = FeatureGraph(torch.device("cpu"))
    graph.add(0, "attn",
              SparseAct(act=f0_grad),
              SparseAct(act=f0_vals.clone()),
              torch.tensor([[[0, 1, 2]]]))
    graph.add(1, "attn",
              SparseAct(act=f1_grad_act),
              SparseAct(act=f1_act_connected),
              torch.tensor([[[0, 1]]]))

    pos_argmax = torch.tensor([0])

    expected_attr = {
        FeatureID(0, "attn", 0): 2.0,
        FeatureID(0, "attn", 2): 0.5,
    }
    expected_grad = {
        FeatureID(0, "attn", 0): 1.0,
        FeatureID(0, "attn", 2): 0.5,
    }
    return graph, pos_argmax, expected_attr, expected_grad


# ---------------------------------------------------------------------------
# TestLogitAttributionOracle — numerically exact verification
# ---------------------------------------------------------------------------

class TestLogitAttributionOracle:
    """Verify attribution scores against hand-computed values."""

    def test_single_batch_exact_scores(self):
        graph, logits, pos_argmax, target_tokens, expected = _build_logit_oracle_graph()
        result = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)

        assert set(result.keys()) == set(expected.keys())
        for fid, exp_score in expected.items():
            assert result[fid] == pytest.approx(exp_score, abs=1e-6), (
                f"{fid}: expected {exp_score}, got {result[fid]}"
            )

    def test_multi_batch_sums_correctly(self):
        graph, logits, pos_argmax, target_tokens, expected = _build_multibatch_logit_oracle_graph()
        result = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)

        assert set(result.keys()) == set(expected.keys())
        for fid, exp_score in expected.items():
            assert result[fid] == pytest.approx(exp_score, abs=1e-6), (
                f"{fid}: expected {exp_score}, got {result[fid]}"
            )

    def test_zero_score_latents_excluded(self):
        """Latent 1 has activation 3.0 but gradient 0.0 → must not appear."""
        graph, logits, pos_argmax, target_tokens, _ = _build_logit_oracle_graph()
        result = compute_logit_attribution(graph, logits, pos_argmax, target_tokens)
        assert FeatureID(0, "attn", 1) not in result

    def test_different_target_token_changes_scores(self):
        """Switching target token 0→1 uses W[:, 1] = [0.5, 1.0, -1.0] as gradient."""
        graph, logits, pos_argmax, _, _ = _build_logit_oracle_graph()
        target_tokens_alt = torch.tensor([[1]])

        result = compute_logit_attribution(graph, logits, pos_argmax, target_tokens_alt)

        # grad = W[:, 1] = [0.5, 1.0, -1.0]
        # score[0] = 2.0 * 0.5  = 1.0
        # score[1] = 3.0 * 1.0  = 3.0
        # score[2] = 1.0 * -1.0 = -1.0
        assert result[FeatureID(0, "attn", 0)] == pytest.approx(1.0, abs=1e-6)
        assert result[FeatureID(0, "attn", 1)] == pytest.approx(3.0, abs=1e-6)
        assert result[FeatureID(0, "attn", 2)] == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestFeatureAttributionOracle — numerically exact verification
# ---------------------------------------------------------------------------

class TestFeatureAttributionOracle:

    def test_exact_scores_no_candidates(self):
        graph, pos_argmax, expected_attr, _ = _build_feature_oracle_graph()
        result = compute_feature_attribution(
            graph, target_layer=1, target_kind="attn",
            target_latent_idx=0, pos_argmax=pos_argmax,
            candidate_nodes=None,
        )
        assert set(result.keys()) == set(expected_attr.keys())
        for fid, exp in expected_attr.items():
            assert result[fid] == pytest.approx(exp, abs=1e-6)

    def test_exact_scores_with_candidates(self):
        graph, pos_argmax, expected_attr, _ = _build_feature_oracle_graph()
        candidates = [FeatureID(0, "attn", 0), FeatureID(0, "attn", 2)]
        result = compute_feature_attribution(
            graph, target_layer=1, target_kind="attn",
            target_latent_idx=0, pos_argmax=pos_argmax,
            candidate_nodes=candidates,
        )
        assert set(result.keys()) == set(expected_attr.keys())
        for fid, exp in expected_attr.items():
            assert result[fid] == pytest.approx(exp, abs=1e-6)

    def test_candidate_filtering_restricts_keys(self):
        graph, pos_argmax, _, _ = _build_feature_oracle_graph()
        candidates = [FeatureID(0, "attn", 0)]
        result = compute_feature_attribution(
            graph, target_layer=1, target_kind="attn",
            target_latent_idx=0, pos_argmax=pos_argmax,
            candidate_nodes=candidates,
        )
        assert set(result.keys()) == {FeatureID(0, "attn", 0)}
        assert result[FeatureID(0, "attn", 0)] == pytest.approx(2.0, abs=1e-6)

    def test_zero_gradient_latent_excluded(self):
        """Latent 1 has grad=0 → must not appear even when explicitly listed."""
        graph, pos_argmax, _, _ = _build_feature_oracle_graph()
        candidates = [FeatureID(0, "attn", 1)]
        result = compute_feature_attribution(
            graph, target_layer=1, target_kind="attn",
            target_latent_idx=0, pos_argmax=pos_argmax,
            candidate_nodes=candidates,
        )
        assert result == {}

    def test_returns_empty_when_act_is_none(self):
        """Graph with act=None connected acts → early return {}."""
        graph = FeatureGraph(torch.device("cpu"))
        graph.add(0, "attn",
                  SparseAct(act=torch.zeros(1, 1, 3, requires_grad=True)),
                  SparseAct(act=None),  # connected has no act
                  torch.tensor([[[0, 1, 2]]]))
        result = compute_feature_attribution(
            graph, target_layer=0, target_kind="attn",
            target_latent_idx=0, pos_argmax=torch.tensor([0]),
        )
        assert result == {}


# ---------------------------------------------------------------------------
# TestComputeFeatureGradient — raw gradient oracle
# ---------------------------------------------------------------------------

class TestComputeFeatureGradient:

    def test_exact_gradient_values(self):
        graph, pos_argmax, _, expected_grad = _build_feature_oracle_graph()
        candidates = [
            FeatureID(0, "attn", 0),
            FeatureID(0, "attn", 1),
            FeatureID(0, "attn", 2),
        ]
        result = compute_feature_gradient(
            graph, target_layer=1, target_kind="attn",
            target_latent_idx=0, pos_argmax=pos_argmax,
            candidate_nodes=candidates,
        )
        assert set(result.keys()) == set(expected_grad.keys())
        for fid, exp in expected_grad.items():
            assert result[fid] == pytest.approx(exp, abs=1e-6)

    def test_zero_gradient_latent_excluded(self):
        """Latent 1 has gradient 0.0 → not in result."""
        graph, pos_argmax, _, _ = _build_feature_oracle_graph()
        result = compute_feature_gradient(
            graph, target_layer=1, target_kind="attn",
            target_latent_idx=0, pos_argmax=pos_argmax,
            candidate_nodes=[FeatureID(0, "attn", 1)],
        )
        assert result == {}

    def test_gradient_differs_from_attribution(self):
        """Raw gradient ≠ act * grad (unless act is 1)."""
        graph, pos_argmax, expected_attr, expected_grad = _build_feature_oracle_graph()
        candidates = [FeatureID(0, "attn", 0), FeatureID(0, "attn", 2)]

        attr_result = compute_feature_attribution(
            graph, target_layer=1, target_kind="attn",
            target_latent_idx=0, pos_argmax=pos_argmax,
            candidate_nodes=candidates,
        )
        grad_result = compute_feature_gradient(
            graph, target_layer=1, target_kind="attn",
            target_latent_idx=0, pos_argmax=pos_argmax,
            candidate_nodes=candidates,
        )
        for fid in candidates:
            if fid in attr_result and fid in grad_result:
                # attr = act * grad, grad = raw → they differ when act ≠ 1
                # fid(0,'attn',0): attr=2.0, grad=1.0 (act=2.0)
                # fid(0,'attn',2): attr=0.5, grad=0.5 (act=1.0 → same here)
                pass
        assert attr_result[FeatureID(0, "attn", 0)] != grad_result[FeatureID(0, "attn", 0)]

    def test_missing_layer_returns_empty(self):
        graph = FeatureGraph(torch.device("cpu"))
        result = compute_feature_gradient(
            graph, target_layer=5, target_kind="attn",
            target_latent_idx=0, pos_argmax=torch.tensor([0]),
            candidate_nodes=[FeatureID(0, "attn", 0)],
        )
        assert result == {}

    def test_no_grad_fn_returns_empty(self):
        """Detached connected acts (no grad_fn) → return {}."""
        f_grad = torch.zeros(1, 1, 3, requires_grad=True)
        f_connected = torch.zeros(1, 1, 3)  # no grad_fn

        graph = FeatureGraph(torch.device("cpu"))
        graph.add(0, "attn", SparseAct(act=f_grad), SparseAct(act=f_connected),
                  torch.tensor([[[0, 1, 2]]]))

        result = compute_feature_gradient(
            graph, target_layer=0, target_kind="attn",
            target_latent_idx=0, pos_argmax=torch.tensor([0]),
            candidate_nodes=[FeatureID(0, "attn", 0)],
        )
        assert result == {}

    def test_act_none_on_connected_returns_empty(self):
        graph = FeatureGraph(torch.device("cpu"))
        graph.add(0, "attn",
                  SparseAct(act=torch.zeros(1, 1, 3, requires_grad=True)),
                  SparseAct(act=None),
                  torch.tensor([[[0, 1, 2]]]))
        result = compute_feature_gradient(
            graph, target_layer=0, target_kind="attn",
            target_latent_idx=0, pos_argmax=torch.tensor([0]),
            candidate_nodes=[FeatureID(0, "attn", 0)],
        )
        assert result == {}


# ---------------------------------------------------------------------------
# TestComputeLatentUpstreamScores — UpstreamScores return type + split logic
# ---------------------------------------------------------------------------

def _build_upstream_oracle_graph(f0_vals_data):
    """
    Two-layer oracle graph for compute_latent_upstream_scores.

    Layer 0 'attn' (D_SAE=3) — upstream leaf anchor with controllable activations.
    Layer 1 'attn' (D_SAE=2) — target whose connected acts depend on layer-0 leaf.

    M = [[1, 0], [0, 1], [0.5, -1]]  →  target_latent=0 has gradient M[:,0]=[1,0,0.5]
                                         target_latent=1 has gradient M[:,1]=[0,1,-1]

    Args:
        f0_vals_data: list/tensor shape [3] specifying layer-0 activations.
    """
    M = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -1.0]])
    f0_vals = torch.tensor([[[f0_vals_data[0], f0_vals_data[1], f0_vals_data[2]]]])  # [1,1,3]
    f0_grad = f0_vals.detach().clone().requires_grad_(True)

    f1_act_connected = f0_grad @ M                              # [1,1,2]
    f1_grad_act = f1_act_connected.detach().clone().requires_grad_(True)

    graph = FeatureGraph(torch.device("cpu"))
    graph.add(0, "attn",
              SparseAct(act=f0_grad),
              SparseAct(act=f0_vals.clone()),
              torch.tensor([[[0, 1, 2]]]))
    graph.add(1, "attn",
              SparseAct(act=f1_grad_act),
              SparseAct(act=f1_act_connected),
              torch.tensor([[[0, 1]]]))

    return graph, f0_grad


# n_kinds=3 matches conftest KINDS=["attn","mlp","resid"]; comp_idx(layer=0, kind_idx=0) = 0
_N_KINDS = 3
_KINDS   = ["attn", "mlp", "resid"]
# Predecessor of (layer=1, kind="attn") in our tiny setup: just (layer=0, kind="attn") → comp_idx=0
_PREDECESSOR_COMP_INDICES = [0]
# Oracle graph has D_SAE=3 latents per layer (not conftest's D_SAE=32)
_ORACLE_D_SAE = 3
_ACTIVE_COUNT = torch.ones(2 * _N_KINDS, _ORACLE_D_SAE, dtype=torch.long)  # everything passes min_active_count=1


class TestComputeLatentUpstreamScores:
    """
    Tests for the refactored compute_latent_upstream_scores which now returns
    UpstreamScores(attribution, absent_gradient) instead of a plain dict.
    """

    def test_returns_upstream_scores_object(self):
        """Return type must be UpstreamScores, not a dict."""
        graph, _ = _build_upstream_oracle_graph([2.0, 3.0, 1.0])
        result = compute_latent_upstream_scores(
            graph,
            target_layer=1, target_kind="attn", target_latent_idx=0,
            pos_argmax=torch.tensor([0]),
            predecessor_comp_indices=_PREDECESSOR_COMP_INDICES,
            n_kinds=_N_KINDS, kinds=_KINDS,
            top_k=3,
            min_active_count=1, active_count=_ACTIVE_COUNT,
        )
        assert isinstance(result, UpstreamScores)
        assert isinstance(result.attribution, dict)
        assert isinstance(result.absent_gradient, dict)

    def test_empty_predecessors_returns_empty_upstream_scores(self):
        """No predecessor components → both dicts are empty."""
        graph, _ = _build_upstream_oracle_graph([2.0, 3.0, 1.0])
        result = compute_latent_upstream_scores(
            graph,
            target_layer=1, target_kind="attn", target_latent_idx=0,
            pos_argmax=torch.tensor([0]),
            predecessor_comp_indices=[],
            n_kinds=_N_KINDS, kinds=_KINDS,
            top_k=3,
            min_active_count=1, active_count=_ACTIVE_COUNT,
        )
        assert isinstance(result, UpstreamScores)
        assert result.attribution == {}
        assert result.absent_gradient == {}

    def test_attribution_scores_match_oracle(self):
        """
        With f0=[2,3,1] and target latent 0, expected attribution scores are:
          acts * grad = [2,3,1] * M[:,0] = [2,3,1] * [1,0,0.5] = [2,0,0.5]
        Latent 1 is excluded (score=0); latents 0 and 2 appear.
        """
        graph, _ = _build_upstream_oracle_graph([2.0, 3.0, 1.0])
        result = compute_latent_upstream_scores(
            graph,
            target_layer=1, target_kind="attn", target_latent_idx=0,
            pos_argmax=torch.tensor([0]),
            predecessor_comp_indices=_PREDECESSOR_COMP_INDICES,
            n_kinds=_N_KINDS, kinds=_KINDS,
            top_k=3,
            min_active_count=1, active_count=_ACTIVE_COUNT,
        )
        assert FeatureID(0, "attn", 0) in result.attribution
        assert FeatureID(0, "attn", 2) in result.attribution
        assert result.attribution[FeatureID(0, "attn", 0)] == pytest.approx(2.0, abs=1e-5)
        assert result.attribution[FeatureID(0, "attn", 2)] == pytest.approx(0.5, abs=1e-5)
        # Latent 1 has zero score → excluded
        assert FeatureID(0, "attn", 1) not in result.attribution

    def test_absent_gradient_empty_when_top_k_is_zero(self):
        """absent_inhibitor_top_k=0 (default) → absent_gradient must always be {}."""
        graph, _ = _build_upstream_oracle_graph([0.0, 0.0, 0.0])
        result = compute_latent_upstream_scores(
            graph,
            target_layer=1, target_kind="attn", target_latent_idx=1,
            pos_argmax=torch.tensor([0]),
            predecessor_comp_indices=_PREDECESSOR_COMP_INDICES,
            n_kinds=_N_KINDS, kinds=_KINDS,
            top_k=3,
            min_active_count=1, active_count=_ACTIVE_COUNT,
            absent_inhibitor_top_k=0,
        )
        assert result.absent_gradient == {}

    def test_absent_gradient_finds_inactive_suppressor(self):
        """
        With f0≈0 (all inactive) and target latent=1:
          gradient = M[:,1] = [0, 1, -1]
        Latent 2 has raw gradient -1 < 0 → should appear in absent_gradient.
        Latent 0 has gradient 0 → excluded. Latent 1 has gradient +1 → not inhibitory.
        """
        graph, _ = _build_upstream_oracle_graph([0.0, 0.0, 0.0])
        result = compute_latent_upstream_scores(
            graph,
            target_layer=1, target_kind="attn", target_latent_idx=1,
            pos_argmax=torch.tensor([0]),
            predecessor_comp_indices=_PREDECESSOR_COMP_INDICES,
            n_kinds=_N_KINDS, kinds=_KINDS,
            top_k=3,
            min_active_count=1, active_count=_ACTIVE_COUNT,
            absent_inhibitor_top_k=3,
            absent_inhibitor_threshold=0.5,
        )
        # Latent 2: raw gradient = -1.0 < -0.5 threshold → present as absent inhibitor
        assert FeatureID(0, "attn", 2) in result.absent_gradient
        assert result.absent_gradient[FeatureID(0, "attn", 2)] == pytest.approx(-1.0, abs=1e-5)
        # Latent 1: gradient = +1.0 → not a suppressor
        assert FeatureID(0, "attn", 1) not in result.absent_gradient
        # Latent 0: gradient = 0.0 → excluded
        assert FeatureID(0, "attn", 0) not in result.absent_gradient

    def test_absent_gradient_respects_threshold(self):
        """
        With a high absent_inhibitor_threshold, the -1.0 gradient at latent 2
        is only captured when threshold < 1.0.
        """
        graph_a, _ = _build_upstream_oracle_graph([0.0, 0.0, 0.0])
        result_captured = compute_latent_upstream_scores(
            graph_a,
            target_layer=1, target_kind="attn", target_latent_idx=1,
            pos_argmax=torch.tensor([0]),
            predecessor_comp_indices=_PREDECESSOR_COMP_INDICES,
            n_kinds=_N_KINDS, kinds=_KINDS,
            top_k=3,
            min_active_count=1, active_count=_ACTIVE_COUNT,
            absent_inhibitor_top_k=3,
            absent_inhibitor_threshold=0.5,  # -1.0 passes (abs > 0.5)
        )
        graph_b, _ = _build_upstream_oracle_graph([0.0, 0.0, 0.0])
        result_filtered = compute_latent_upstream_scores(
            graph_b,
            target_layer=1, target_kind="attn", target_latent_idx=1,
            pos_argmax=torch.tensor([0]),
            predecessor_comp_indices=_PREDECESSOR_COMP_INDICES,
            n_kinds=_N_KINDS, kinds=_KINDS,
            top_k=3,
            min_active_count=1, active_count=_ACTIVE_COUNT,
            absent_inhibitor_top_k=3,
            absent_inhibitor_threshold=2.0,  # -1.0 does NOT pass (abs < 2.0)
        )
        assert FeatureID(0, "attn", 2) in result_captured.absent_gradient
        assert FeatureID(0, "attn", 2) not in result_filtered.absent_gradient

    def test_attribution_and_absent_gradient_are_disjoint(self):
        """
        With active f0=[2,3,1] targeting latent 0, active latents produce
        attribution scores.  Absent gradient for the same run with absent_inhibitor_top_k>0
        should not overlap with attribution keys (active latents are not in absent_mask).
        """
        graph, _ = _build_upstream_oracle_graph([2.0, 3.0, 1.0])
        result = compute_latent_upstream_scores(
            graph,
            target_layer=1, target_kind="attn", target_latent_idx=0,
            pos_argmax=torch.tensor([0]),
            predecessor_comp_indices=_PREDECESSOR_COMP_INDICES,
            n_kinds=_N_KINDS, kinds=_KINDS,
            top_k=3,
            min_active_count=1, active_count=_ACTIVE_COUNT,
            absent_inhibitor_top_k=3,
            absent_inhibitor_threshold=0.01,
        )
        # Active latents (2.0, 3.0, 1.0 >> 1e-6) must not appear in absent_gradient
        for fid in result.attribution:
            assert fid not in result.absent_gradient, (
                f"Active latent {fid} appeared in both attribution and absent_gradient"
            )
