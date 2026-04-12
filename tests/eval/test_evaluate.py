"""
Phase 6 — End-to-end evaluation metric tests.

Tests evaluate_faithfulness, evaluate_completeness, evaluate_minimality,
prune_non_minimal_nodes, and evaluate_sufficiency using a MockInference that
runs the real MockModel under CircuitPatcher hooks.

Key algebraic identities exploited
------------------------------------
Full circuit (all D_SAE latents in every layer/kind), forward mode:
  • CircuitPatcher keeps all active latents → patched = x
    (circuit_recon = full_recon, bg = 0 since no non-circuit latents)
  → circuit_logits = original_logits → faithfulness = 1.0, sufficiency = 1.0

Empty circuit (no nodes), forward mode:
  • CircuitPatcher zeroes all active latents → patched = decode(0) + error = b_dec + error
  • baseline patcher is identical (circuit=None) → circuit_logits = baseline_logits
  → faithfulness = 0.0

Empty circuit, inverse mode (complement pass for completeness):
  • Nothing to ablate → patched = x → complement_logits = original_logits
  → faithfulness(complement) = 1.0 → completeness = 0.0

Full circuit, inverse mode:
  • All latents ablated → patched = b_dec + error = baseline result
  → faithfulness(complement) = 0.0 → completeness = 1.0
"""
import math

import pytest
import torch

from eval.faithfulness import evaluate_faithfulness, evaluate_kind_local_faithfulness
from eval.completeness import evaluate_completeness
from eval.minimality  import evaluate_minimality, prune_non_minimal_nodes
from eval.sufficiency import evaluate_sufficiency
from store.circuits import Circuit, CircuitNode

# ---------------------------------------------------------------------------
# Local constants  (mirror conftest but kept local for readability)
# ---------------------------------------------------------------------------

B_EVAL,  T_EVAL  = 2, 4
D_MODEL          = 16
D_SAE            = 32
N_LAYERS         = 2
KINDS            = ["attn", "mlp", "resid"]
V_EVAL           = 20

TOKENS = torch.zeros(B_EVAL, T_EVAL, dtype=torch.long)
_g = torch.Generator().manual_seed(99)
TARGET_TOKENS = torch.randint(0, V_EVAL, (B_EVAL, T_EVAL), generator=_g)
POS_ARGMAX    = torch.tensor([1, 2])

# ---------------------------------------------------------------------------
# MockInference
# ---------------------------------------------------------------------------

class MockInference:
    """
    Minimal stand-in for Inference.forward() that runs MockModel under
    CircuitPatcher hooks.  A fixed seeded input is reused on every call so
    that the same patcher configuration always produces identical logits.
    """
    def __init__(self, model, d_model: int = D_MODEL, d_vocab: int = V_EVAL):
        torch.manual_seed(7)
        self.model   = model
        self.W_logit = torch.randn(d_model, d_vocab)
        torch.manual_seed(13)
        self._x = torch.randn(B_EVAL, T_EVAL, d_model)   # fixed input

    _compiled = False
    def disable_compile(self): pass
    def enable_compile(self):  pass

    def forward(
        self,
        tokens,
        num_gen:            int  = 1,
        tokenize_final:     bool = False,
        return_activations: bool = False,
        all_logits:         bool = False,
        patcher             = None,
    ):
        with torch.no_grad():
            if patcher is not None:
                with patcher(self.model):
                    output = self.model(self._x)
            else:
                output = self.model(self._x)

        if all_logits:
            logits = output @ self.W_logit              # [B, T, V]
        else:
            logits = (output[:, -1:, :]) @ self.W_logit  # [B, 1, V]

        return tokens, logits, None


class FixedLogitInference:
    """
    Returns pre-set logit tensors for oracle tests.
    Original logits when no patcher, circuit logits when patcher is present.
    The patcher object itself is never invoked.
    """
    _compiled = False
    def disable_compile(self): pass
    def enable_compile(self): pass

    def __init__(self, original_logits: torch.Tensor, circuit_logits: torch.Tensor):
        self._original = original_logits
        self._circuit = circuit_logits

    def forward(self, tokens, num_gen=1, tokenize_final=False,
                return_activations=False, all_logits=False, patcher=None):
        logits = self._original if patcher is None else self._circuit
        return tokens, logits, None


# ---------------------------------------------------------------------------
# Circuit helpers
# ---------------------------------------------------------------------------

def _make_full_circuit() -> Circuit:
    """All D_SAE latents at every (layer, kind) — circuit output equals original."""
    c = Circuit(name="full-circuit")
    for layer in range(N_LAYERS):
        for kind in KINDS:
            for latent_idx in range(D_SAE):
                c.add_node(CircuitNode(metadata={
                    "layer_idx":  layer,
                    "kind":       kind,
                    "latent_idx": latent_idx,
                    "role":       "latent",
                }))
    return c


def _make_empty_circuit() -> Circuit:
    return Circuit(name="empty-circuit")


def _make_kind_full_circuit(kind: str) -> Circuit:
    """All D_SAE latents for a single kind across all layers."""
    c = Circuit(name=f"full-{kind}-circuit")
    for layer in range(N_LAYERS):
        for latent_idx in range(D_SAE):
            c.add_node(CircuitNode(metadata={
                "layer_idx": layer, "kind": kind,
                "latent_idx": latent_idx, "role": "latent",
            }))
    return c


def _make_two_node_circuit():
    """Returns (circuit, uuid_a, uuid_b) for LOO algebraic identity tests."""
    node_a = CircuitNode(metadata={"layer_idx": 0, "kind": "attn", "latent_idx": 5,  "role": "latent"})
    node_b = CircuitNode(metadata={"layer_idx": 0, "kind": "mlp",  "latent_idx": 11, "role": "latent"})
    c = Circuit(name="two-node-circuit")
    c.add_node(node_a)
    c.add_node(node_b)
    return c, node_a.uuid, node_b.uuid


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_inference(mock_model):
    return MockInference(mock_model)


# ---------------------------------------------------------------------------
# TestEvaluateFaithfulness
# ---------------------------------------------------------------------------

class TestEvaluateFaithfulness:

    def test_returns_float(self, mock_inference, mock_sae_bank, avg_acts_zero):
        result = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_full_circuit(), TOKENS
        )
        assert isinstance(result, float)

    def test_full_circuit_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """Full circuit keeps all active latents → patched = x → score 1.0."""
        score = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_full_circuit(), TOKENS
        )
        assert abs(score - 1.0) < 1e-4

    def test_empty_circuit_returns_zero(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """Empty circuit ablates everything → circuit_logits = baseline_logits → score 0.0."""
        score = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_empty_circuit(), TOKENS
        )
        assert abs(score - 0.0) < 1e-4

    def test_circuit_none_returns_zero(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """circuit=None creates an empty-mask patcher identical to an empty circuit."""
        score = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, None, TOKENS
        )
        assert abs(score - 0.0) < 1e-4

    def test_full_circuit_scores_higher_than_empty(self, mock_inference, mock_sae_bank, avg_acts_zero):
        score_full  = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_full_circuit(), TOKENS
        )
        score_empty = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_empty_circuit(), TOKENS
        )
        assert score_full > score_empty

    def test_with_pos_argmax_returns_float(self, mock_inference, mock_sae_bank, avg_acts_zero):
        result = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, pos_argmax=POS_ARGMAX
        )
        assert isinstance(result, float)

    def test_full_circuit_with_pos_argmax_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        score = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, pos_argmax=POS_ARGMAX
        )
        assert abs(score - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# TestEvaluateKindLocalFaithfulness
# ---------------------------------------------------------------------------

class TestEvaluateKindLocalFaithfulness:
    """
    evaluate_kind_local_faithfulness patches only the listed kinds.
    Unlisted kinds pass through unchanged (CircuitPatcher.transform returns x).

    Algebraic identities:
      full circuit + target_kinds=all → same as global faithfulness → 1.0
      full circuit + target_kinds=[k] → circuit covers all latents for k,
        non-k kinds unchanged → circuit_logits = original_logits → 1.0
      empty circuit + any target_kinds → circuit has no nodes for those kinds,
        patched = baseline for those kinds → circuit_logits = baseline_logits → 0.0
    """

    def test_full_circuit_all_kinds_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """All kinds listed → equivalent to global faithfulness with full circuit."""
        score = evaluate_kind_local_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, target_kinds=KINDS,
        )
        assert abs(score - 1.0) < 1e-4

    def test_full_circuit_single_kind_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Full circuit with target_kinds=["mlp"]:
        Only MLP is patched. Circuit covers all MLP latents → patched MLP = original.
        Attn and resid pass through unchanged → circuit_logits = original_logits → 1.0.
        """
        score = evaluate_kind_local_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, target_kinds=["mlp"],
        )
        assert abs(score - 1.0) < 1e-4

    def test_empty_circuit_single_kind_returns_zero(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Empty circuit with target_kinds=["mlp"]:
        MLP patched with no circuit nodes → all MLP latents ablated → same as baseline.
        circuit_logits = baseline_logits → 0.0.
        """
        score = evaluate_kind_local_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_empty_circuit(), TOKENS, target_kinds=["mlp"],
        )
        assert abs(score - 0.0) < 1e-4

    def test_circuit_missing_target_kind_returns_zero(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Circuit has only MLP nodes, but target_kinds=["attn"]:
        Attn patched with no matching circuit nodes → all attn latents ablated → same as baseline.
        """
        mlp_circuit = _make_kind_full_circuit("mlp")
        score = evaluate_kind_local_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            mlp_circuit, TOKENS, target_kinds=["attn"],
        )
        assert abs(score - 0.0) < 1e-4

    def test_kind_circuit_matches_target_kind_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Circuit has only MLP nodes (full coverage), target_kinds=["mlp"]:
        MLP patched with full MLP coverage → patched MLP = original MLP → 1.0.
        """
        mlp_circuit = _make_kind_full_circuit("mlp")
        score = evaluate_kind_local_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            mlp_circuit, TOKENS, target_kinds=["mlp"],
        )
        assert abs(score - 1.0) < 1e-4

    def test_with_pos_argmax_full_circuit_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        score = evaluate_kind_local_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, target_kinds=["resid"],
            pos_argmax=POS_ARGMAX,
        )
        assert abs(score - 1.0) < 1e-4

    def test_with_pos_argmax_empty_circuit_returns_zero(self, mock_inference, mock_sae_bank, avg_acts_zero):
        score = evaluate_kind_local_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_empty_circuit(), TOKENS, target_kinds=["resid"],
            pos_argmax=POS_ARGMAX,
        )
        assert abs(score - 0.0) < 1e-4


# ---------------------------------------------------------------------------
# TestEvaluateCompleteness
# ---------------------------------------------------------------------------

class TestEvaluateCompleteness:

    def test_returns_float(self, mock_inference, mock_sae_bank, avg_acts_zero):
        result = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_full_circuit(), TOKENS
        )
        assert isinstance(result, float)

    def test_full_circuit_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Complement of full circuit (inverse mode on all-True mask) ablates everything
        → complement_logits = baseline_logits → f_complement = 0.0 → completeness = 1.0.
        """
        score = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_full_circuit(), TOKENS
        )
        assert abs(score - 1.0) < 1e-4

    def test_empty_circuit_returns_zero(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Complement of empty circuit (inverse mode on all-False mask) passes through
        unchanged → complement_logits = original_logits → f_complement = 1.0 → completeness = 0.0.
        """
        score = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_empty_circuit(), TOKENS
        )
        assert abs(score - 0.0) < 1e-4

    def test_full_circuit_scores_higher_than_empty(self, mock_inference, mock_sae_bank, avg_acts_zero):
        score_full  = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_full_circuit(), TOKENS
        )
        score_empty = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_empty_circuit(), TOKENS
        )
        assert score_full > score_empty

    def test_with_pos_argmax_returns_float(self, mock_inference, mock_sae_bank, avg_acts_zero):
        result = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, pos_argmax=POS_ARGMAX
        )
        assert isinstance(result, float)

    def test_full_circuit_with_pos_argmax_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Full circuit + pos_argmax: complement ablates all latents at probe position
        → complement_logits = baseline_logits → f_complement = 0.0 → completeness = 1.0.
        """
        score = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, pos_argmax=POS_ARGMAX,
        )
        assert abs(score - 1.0) < 1e-4

    def test_empty_circuit_with_pos_argmax_returns_zero(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Empty circuit + pos_argmax: complement passes through at probe position
        → complement_logits = original_logits → f_complement = 1.0 → completeness = 0.0.
        """
        score = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_empty_circuit(), TOKENS, pos_argmax=POS_ARGMAX,
        )
        assert abs(score - 0.0) < 1e-4

    def test_full_scores_higher_than_empty_with_pos_argmax(self, mock_inference, mock_sae_bank, avg_acts_zero):
        score_full = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, pos_argmax=POS_ARGMAX,
        )
        score_empty = evaluate_completeness(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_empty_circuit(), TOKENS, pos_argmax=POS_ARGMAX,
        )
        assert score_full > score_empty


# ---------------------------------------------------------------------------
# TestEvaluateMinimality
# ---------------------------------------------------------------------------

class TestEvaluateMinimality:

    def test_returns_dict_keyed_by_node_uuid_strings(self, mock_inference, mock_sae_bank, avg_acts_zero):
        c, uuid_a, uuid_b = _make_two_node_circuit()
        result = evaluate_minimality(mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS)
        assert set(result.keys()) == {uuid_a, uuid_b}
        for key in result:
            assert isinstance(key, str)

    def test_empty_circuit_returns_empty_dict(self, mock_inference, mock_sae_bank, avg_acts_zero):
        result = evaluate_minimality(
            mock_inference, mock_sae_bank, avg_acts_zero, _make_empty_circuit(), TOKENS
        )
        assert result == {}

    def test_single_node_circuit_has_one_key(self, mock_inference, mock_sae_bank, avg_acts_zero):
        node = CircuitNode(metadata={"layer_idx": 0, "kind": "attn", "latent_idx": 3, "role": "latent"})
        c = Circuit(name="one-node")
        c.add_node(node)
        result = evaluate_minimality(mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS)
        assert len(result) == 1
        assert node.uuid in result

    def test_importance_values_are_floats(self, mock_inference, mock_sae_bank, avg_acts_zero):
        c, _, _ = _make_two_node_circuit()
        result = evaluate_minimality(mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS)
        for v in result.values():
            assert isinstance(v, float)

    def test_importance_equals_base_minus_loo_faithfulness(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Algebraic identity:
          importance[A] = faithfulness(circuit_AB) - faithfulness(circuit_B_only)
          importance[B] = faithfulness(circuit_AB) - faithfulness(circuit_A_only)
        """
        c, uuid_a, uuid_b = _make_two_node_circuit()
        f_base = evaluate_faithfulness(mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS)

        c_b_only = Circuit(name="b-only")
        c_b_only.add_node(c.nodes[uuid_b])
        f_without_a = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, c_b_only, TOKENS
        )

        c_a_only = Circuit(name="a-only")
        c_a_only.add_node(c.nodes[uuid_a])
        f_without_b = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, c_a_only, TOKENS
        )

        importances = evaluate_minimality(mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS)

        assert abs(importances[uuid_a] - (f_base - f_without_a)) < 1e-5
        assert abs(importances[uuid_b] - (f_base - f_without_b)) < 1e-5

    def test_loo_identity_with_pos_argmax(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """Same LOO algebraic identity holds when pos_argmax is supplied."""
        c, uuid_a, uuid_b = _make_two_node_circuit()
        f_base = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS, pos_argmax=POS_ARGMAX
        )

        c_b_only = Circuit(name="b-only")
        c_b_only.add_node(c.nodes[uuid_b])
        f_without_a = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, c_b_only, TOKENS, pos_argmax=POS_ARGMAX
        )

        c_a_only = Circuit(name="a-only")
        c_a_only.add_node(c.nodes[uuid_a])
        f_without_b = evaluate_faithfulness(
            mock_inference, mock_sae_bank, avg_acts_zero, c_a_only, TOKENS, pos_argmax=POS_ARGMAX
        )

        importances = evaluate_minimality(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS, pos_argmax=POS_ARGMAX
        )

        assert abs(importances[uuid_a] - (f_base - f_without_a)) < 1e-5
        assert abs(importances[uuid_b] - (f_base - f_without_b)) < 1e-5


# ---------------------------------------------------------------------------
# TestPruneNonMinimalNodes
# ---------------------------------------------------------------------------

class TestPruneNonMinimalNodes:

    def test_returns_list_of_uuid_strings(self, mock_inference, mock_sae_bank, avg_acts_zero):
        c, _, _ = _make_two_node_circuit()
        removed = prune_non_minimal_nodes(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS, threshold=2.0
        )
        assert isinstance(removed, list)
        for uid in removed:
            assert isinstance(uid, str)

    def test_high_threshold_removes_all_non_seed_nodes(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        threshold=2.0 exceeds any achievable importance (faithfulness ≤ 1.0,
        so importance = base - loo ≤ 1.0 < 2.0).  Every non-seed node is pruned.
        """
        node = CircuitNode(metadata={"layer_idx": 0, "kind": "attn", "latent_idx": 2, "role": "latent"})
        c = Circuit(name="test")
        c.add_node(node)

        removed = prune_non_minimal_nodes(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS, threshold=2.0
        )

        assert node.uuid in removed
        assert node.uuid not in c.nodes

    def test_seed_node_is_never_pruned(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """role='seed' nodes must survive regardless of their importance score."""
        seed = CircuitNode(metadata={"layer_idx": 0, "kind": "attn", "latent_idx": 0, "role": "seed"})
        c = Circuit(name="test")
        c.add_node(seed)

        removed = prune_non_minimal_nodes(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS, threshold=2.0
        )

        assert seed.uuid not in removed
        assert seed.uuid in c.nodes

    def test_pruning_removes_associated_edges(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """When a non-seed node is pruned, every edge referencing it must be removed."""
        seed   = CircuitNode(metadata={"layer_idx": 0, "kind": "attn", "latent_idx": 0, "role": "seed"})
        latent = CircuitNode(metadata={"layer_idx": 0, "kind": "mlp",  "latent_idx": 1, "role": "latent"})
        c = Circuit(name="test")
        c.add_node(seed)
        c.add_node(latent)
        c.add_edge(seed.uuid, latent.uuid)   # one edge: seed → latent

        assert len(c.edges) == 1  # precondition

        prune_non_minimal_nodes(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS, threshold=2.0
        )

        assert latent.uuid not in c.nodes
        assert not any(
            e.source_uuid == latent.uuid or e.target_uuid == latent.uuid
            for e in c.edges
        )

    def test_high_threshold_with_pos_argmax_prunes_non_seed(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """pos_argmax does not prevent pruning — non-seed node is still removed."""
        node = CircuitNode(metadata={"layer_idx": 0, "kind": "attn", "latent_idx": 2, "role": "latent"})
        c = Circuit(name="test")
        c.add_node(node)

        removed = prune_non_minimal_nodes(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS,
            pos_argmax=POS_ARGMAX, threshold=2.0,
        )

        assert node.uuid in removed
        assert node.uuid not in c.nodes

    def test_multi_step_both_non_seeds_pruned(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Three-node circuit (seed + 2 latents) with threshold=2.0:
        Both non-seed nodes are pruned iteratively, only seed survives.
        """
        seed  = CircuitNode(metadata={"layer_idx": 0, "kind": "attn",  "latent_idx": 0,  "role": "seed"})
        lat_a = CircuitNode(metadata={"layer_idx": 0, "kind": "mlp",   "latent_idx": 5,  "role": "latent"})
        lat_b = CircuitNode(metadata={"layer_idx": 1, "kind": "resid", "latent_idx": 10, "role": "latent"})
        c = Circuit(name="three-node")
        c.add_node(seed)
        c.add_node(lat_a)
        c.add_node(lat_b)

        removed = prune_non_minimal_nodes(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS, threshold=2.0,
        )

        assert len(removed) == 2
        assert lat_a.uuid in removed
        assert lat_b.uuid in removed
        assert set(c.nodes.keys()) == {seed.uuid}

    def test_multi_step_least_important_pruned_first(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        Iterative pruning removes the least important node first.
        Verify removed[0] matches the initially-least-important non-seed node.
        """
        seed  = CircuitNode(metadata={"layer_idx": 0, "kind": "attn",  "latent_idx": 0,  "role": "seed"})
        lat_a = CircuitNode(metadata={"layer_idx": 0, "kind": "mlp",   "latent_idx": 5,  "role": "latent"})
        lat_b = CircuitNode(metadata={"layer_idx": 1, "kind": "resid", "latent_idx": 10, "role": "latent"})

        c_for_imp = Circuit(name="importance-check")
        c_for_imp.add_node(seed)
        c_for_imp.add_node(lat_a)
        c_for_imp.add_node(lat_b)

        importances = evaluate_minimality(
            mock_inference, mock_sae_bank, avg_acts_zero, c_for_imp, TOKENS,
        )
        non_seed_imps = {k: v for k, v in importances.items() if k != seed.uuid}
        least_uuid = min(non_seed_imps, key=non_seed_imps.get)

        c = Circuit(name="three-node")
        c.add_node(seed)
        c.add_node(lat_a)
        c.add_node(lat_b)

        removed = prune_non_minimal_nodes(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS, threshold=2.0,
        )

        assert len(removed) == 2
        assert removed[0] == least_uuid

    def test_selective_threshold_preserves_important_node(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """
        With a threshold between the two nodes' standalone importances,
        the weaker node is pruned and the stronger node survives.

        Uses standalone (seed + one node) importances to pick a threshold
        that discriminates reliably even after the re-evaluation step.
        """
        seed  = CircuitNode(metadata={"layer_idx": 0, "kind": "attn",  "latent_idx": 0,  "role": "seed"})
        lat_a = CircuitNode(metadata={"layer_idx": 0, "kind": "mlp",   "latent_idx": 5,  "role": "latent"})
        lat_b = CircuitNode(metadata={"layer_idx": 1, "kind": "resid", "latent_idx": 10, "role": "latent"})

        c_sa = Circuit(name="seed-a")
        c_sa.add_node(seed); c_sa.add_node(lat_a)
        imp_a = evaluate_minimality(
            mock_inference, mock_sae_bank, avg_acts_zero, c_sa, TOKENS,
        )[lat_a.uuid]

        c_sb = Circuit(name="seed-b")
        c_sb.add_node(seed); c_sb.add_node(lat_b)
        imp_b = evaluate_minimality(
            mock_inference, mock_sae_bank, avg_acts_zero, c_sb, TOKENS,
        )[lat_b.uuid]

        if abs(imp_a - imp_b) < 1e-8:
            pytest.skip("Nodes have identical standalone importance — cannot discriminate")

        if imp_a < imp_b:
            weak, strong = lat_a, lat_b
            threshold = (imp_a + imp_b) / 2
        else:
            weak, strong = lat_b, lat_a
            threshold = (imp_b + imp_a) / 2

        c = Circuit(name="three-node")
        c.add_node(seed)
        c.add_node(lat_a)
        c.add_node(lat_b)

        removed = prune_non_minimal_nodes(
            mock_inference, mock_sae_bank, avg_acts_zero, c, TOKENS, threshold=threshold,
        )

        assert weak.uuid in removed
        assert strong.uuid in c.nodes


# ---------------------------------------------------------------------------
# TestEvaluateSufficiency
# ---------------------------------------------------------------------------

class TestEvaluateSufficiency:

    def test_returns_float(self, mock_inference, mock_sae_bank, avg_acts_zero):
        result = evaluate_sufficiency(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, TARGET_TOKENS
        )
        assert isinstance(result, float)

    def test_full_circuit_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """Full circuit → circuit_logits = original_logits → exp(0) = 1.0."""
        score = evaluate_sufficiency(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, TARGET_TOKENS
        )
        assert abs(score - 1.0) < 1e-4

    def test_score_is_positive(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """exp(·) is always positive regardless of the log-prob difference."""
        score = evaluate_sufficiency(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_empty_circuit(), TOKENS, TARGET_TOKENS
        )
        assert score > 0.0

    def test_with_pos_argmax_returns_float(self, mock_inference, mock_sae_bank, avg_acts_zero):
        result = evaluate_sufficiency(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, TARGET_TOKENS, pos_argmax=POS_ARGMAX
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TestSufficiencyOracle — hand-computed exp(Δ log-prob)
# ---------------------------------------------------------------------------

class TestSufficiencyOracle:
    """
    Oracle tests for evaluate_sufficiency using FixedLogitInference.

    The formula is:
        sufficiency = mean_b( exp(log_softmax(circ)[target] - log_softmax(orig)[target]) )

    Single-batch oracle (B=1, V=3, target=0):
        orig logits = [log(6), 0, 0]
            softmax  = [6/8, 1/8, 1/8]
            log_softmax[0] = log(6/8) = log(3/4)
        circ logits = [0, 0, 0]
            softmax  = [1/3, 1/3, 1/3]
            log_softmax[0] = -log(3)

        sufficiency = exp(-log(3) - log(3/4))
                    = exp(log(4/9))
                    = 4/9 ≈ 0.4444
    """

    def test_hand_computed_single_batch(self, mock_sae_bank, avg_acts_zero):
        """Verify exact sufficiency value against hand-computed 4/9."""
        orig = torch.tensor([[[math.log(6), 0.0, 0.0]]])  # [1, 1, 3]
        circ = torch.tensor([[[0.0, 0.0, 0.0]]])           # [1, 1, 3]
        tokens = torch.zeros(1, 1, dtype=torch.long)
        target_tokens = torch.zeros(1, 1, dtype=torch.long)

        inference = FixedLogitInference(orig, circ)
        score = evaluate_sufficiency(
            inference, mock_sae_bank, avg_acts_zero,
            _make_empty_circuit(), tokens, target_tokens,
        )
        assert score == pytest.approx(4.0 / 9.0, abs=1e-5)

    def test_identical_logits_returns_one(self, mock_sae_bank, avg_acts_zero):
        """When circuit logits = original logits, exp(0) = 1.0 exactly."""
        logits = torch.randn(1, 1, 10)
        tokens = torch.zeros(1, 1, dtype=torch.long)
        target_tokens = torch.zeros(1, 1, dtype=torch.long)

        inference = FixedLogitInference(logits, logits.clone())
        score = evaluate_sufficiency(
            inference, mock_sae_bank, avg_acts_zero,
            _make_empty_circuit(), tokens, target_tokens,
        )
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_hand_computed_multi_batch_with_pos_argmax(self, mock_sae_bank, avg_acts_zero):
        """
        B=2, T=3, V=3, pos_argmax=[0, 2].

        Batch 0 at pos 0: orig=[log(6), 0, 0], circ=[0, 0, 0], target=0
            → suff_0 = 4/9  (same derivation as single-batch test)

        Batch 1 at pos 2: orig=[0, 0, 0], circ=[0, 0, 0], target=0
            → identical logits → suff_1 = 1.0

        Expected: mean(4/9, 1.0) = 13/18 ≈ 0.72222
        """
        orig = torch.zeros(2, 3, 3)
        orig[0, 0, :] = torch.tensor([math.log(6), 0.0, 0.0])

        circ = torch.zeros(2, 3, 3)

        tokens = torch.zeros(2, 3, dtype=torch.long)
        target_tokens = torch.zeros(2, 3, dtype=torch.long)
        pos_argmax = torch.tensor([0, 2])

        inference = FixedLogitInference(orig, circ)
        score = evaluate_sufficiency(
            inference, mock_sae_bank, avg_acts_zero,
            _make_empty_circuit(), tokens, target_tokens,
            pos_argmax=pos_argmax,
        )
        assert score == pytest.approx(13.0 / 18.0, abs=1e-5)

    def test_full_circuit_with_pos_argmax_returns_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """Full circuit + pos_argmax → circuit_logits = original_logits → 1.0."""
        score = evaluate_sufficiency(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_full_circuit(), TOKENS, TARGET_TOKENS,
            pos_argmax=POS_ARGMAX,
        )
        assert abs(score - 1.0) < 1e-4

    def test_empty_circuit_differs_from_one(self, mock_inference, mock_sae_bank, avg_acts_zero):
        """Empty circuit changes logits → sufficiency ≠ 1.0 (positive, but shifted)."""
        score = evaluate_sufficiency(
            mock_inference, mock_sae_bank, avg_acts_zero,
            _make_empty_circuit(), TOKENS, TARGET_TOKENS,
        )
        assert score > 0.0
        assert abs(score - 1.0) > 1e-4
