"""
Phase 1 — Unit tests for _calculate_faithfulness_score.

Tests the pure mathematical formula in isolation, using hand-crafted logit
tensors. No model, no SAE, no hooks, no inference required.

Formula: score = 1 - MSE(intervened, original) / MSE(baseline, original)
"""
import pytest
import torch

from eval.faithfulness import _calculate_faithfulness_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logits(B: int, T: int, V: int, fill: float) -> torch.Tensor:
    """Returns a [B, T, V] logit tensor filled with a constant value."""
    return torch.full((B, T, V), fill)


def _lerp_logits(orig: torch.Tensor, base: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation: t=0 → orig, t=1 → base."""
    return orig + t * (base - orig)


# ---------------------------------------------------------------------------
# Edge cases: MSE(baseline, original) ≈ 0
# ---------------------------------------------------------------------------

class TestScoreEdgeCases:
    def test_baseline_equals_original_and_circuit_also_equals_original(self):
        """All three identical → perfect match → score = 1.0."""
        logits = _make_logits(2, 4, 10, 0.0)
        score = _calculate_faithfulness_score(logits, logits.clone(), logits.clone())
        assert score == pytest.approx(1.0)

    def test_baseline_equals_original_but_circuit_differs(self):
        """MSE_base < 1e-9 but circuit is different → score = 0.0."""
        orig = _make_logits(2, 4, 10, 0.0)
        base = _make_logits(2, 4, 10, 0.0)
        interv = _make_logits(2, 4, 10, 1.0)
        score = _calculate_faithfulness_score(orig, interv, base)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Core formula correctness (no pos_argmax — uses last token position)
# ---------------------------------------------------------------------------

class TestScoreFormulaLastToken:
    def test_perfect_circuit_returns_one(self):
        """circuit_logits == original_logits → score = 1.0."""
        B, T, V = 3, 8, 50
        orig = torch.randn(B, T, V)
        base = torch.randn(B, T, V)
        score = _calculate_faithfulness_score(orig, orig.clone(), base)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_zero_circuit_returns_zero(self):
        """circuit_logits == baseline_logits → score = 0.0."""
        B, T, V = 3, 8, 50
        orig = torch.randn(B, T, V)
        base = torch.randn(B, T, V)
        score = _calculate_faithfulness_score(orig, base.clone(), base)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_halfway_circuit_is_between_zero_and_one(self):
        """Interpolated circuit → score strictly in (0, 1)."""
        B, T, V = 2, 6, 20
        orig = torch.zeros(B, T, V)
        base = torch.ones(B, T, V) * 2.0
        # circuit is 50% of the way from orig to base at last token
        interv = _lerp_logits(orig, base, 0.5)
        score = _calculate_faithfulness_score(orig, interv, base)
        assert 0.0 < score < 1.0

    def test_halfway_circuit_correct_value(self):
        """
        At t=0.5 interpolation (using last token):
            interv = orig + 0.5*(base - orig)
            MSE(interv, orig) = 0.5^2 * MSE(base, orig) = 0.25 * MSE(base, orig)
            score = 1 - 0.25 = 0.75
        """
        B, T, V = 2, 4, 10
        orig = torch.zeros(B, T, V)
        base = torch.ones(B, T, V) * 2.0
        interv = _lerp_logits(orig, base, 0.5)
        score = _calculate_faithfulness_score(orig, interv, base)
        assert score == pytest.approx(0.75, abs=1e-5)

    def test_circuit_worse_than_baseline_returns_negative(self):
        """Circuit further from original than baseline → score < 0."""
        B, T, V = 2, 4, 10
        orig = torch.zeros(B, T, V)
        base = torch.ones(B, T, V)
        interv = torch.ones(B, T, V) * 3.0  # much further than base
        score = _calculate_faithfulness_score(orig, interv, base)
        assert score < 0.0

    def test_score_uses_last_token_position(self):
        """
        Without pos_argmax, only the last token position (T-1) contributes.
        Make orig == interv == base everywhere except position T-1,
        where interv differs from orig. Score should be 0.0 (interv == base there).
        """
        B, T, V = 2, 6, 10
        orig = torch.zeros(B, T, V)
        base = torch.ones(B, T, V)
        # interv matches base at all positions (including last)
        interv = base.clone()
        score = _calculate_faithfulness_score(orig, interv, base)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_score_ignores_non_last_token_positions(self):
        """
        Middle positions should not affect the score when pos_argmax is None.
        Set orig == interv == base at the last token, but differ at earlier tokens.
        Score should be 1.0.
        """
        B, T, V = 2, 6, 10
        orig = torch.zeros(B, T, V)
        base = torch.ones(B, T, V) * 5.0
        interv = base.clone()
        # Override last token: make interv match orig there
        interv[:, -1, :] = orig[:, -1, :]
        score = _calculate_faithfulness_score(orig, interv, base)
        assert score == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# pos_argmax — position-selective extraction
# ---------------------------------------------------------------------------

class TestScoreFormulaWithPosArgmax:
    def test_perfect_circuit_with_pos_argmax_returns_one(self):
        B, T, V = 3, 8, 50
        orig = torch.randn(B, T, V)
        base = torch.randn(B, T, V)
        pos_argmax = torch.randint(0, T, (B,))
        score = _calculate_faithfulness_score(orig, orig.clone(), base, pos_argmax)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_zero_circuit_with_pos_argmax_returns_zero(self):
        B, T, V = 3, 8, 50
        orig = torch.randn(B, T, V)
        base = torch.randn(B, T, V)
        pos_argmax = torch.randint(0, T, (B,))
        score = _calculate_faithfulness_score(orig, base.clone(), base, pos_argmax)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_halfway_circuit_with_pos_argmax_correct_value(self):
        B, T, V = 2, 8, 10
        orig = torch.zeros(B, T, V)
        base = torch.ones(B, T, V) * 2.0
        interv = _lerp_logits(orig, base, 0.5)
        pos_argmax = torch.tensor([2, 5])
        score = _calculate_faithfulness_score(orig, interv, base, pos_argmax)
        assert score == pytest.approx(0.75, abs=1e-5)

    def test_pos_argmax_extracts_correct_position_not_last(self):
        """
        Set interv == orig at pos_argmax positions, but interv == base at last token.
        Score should be 1.0 because only pos_argmax positions are used.
        """
        B, T, V = 2, 8, 10
        orig = torch.zeros(B, T, V)
        base = torch.ones(B, T, V) * 4.0
        interv = base.clone()
        pos_argmax = torch.tensor([1, 3])
        # At probe positions, match orig perfectly
        for b, p in enumerate(pos_argmax.tolist()):
            interv[b, p, :] = orig[b, p, :]
        score = _calculate_faithfulness_score(orig, interv, base, pos_argmax)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_pos_argmax_different_positions_per_batch_item(self):
        """Each batch item can have a different probe position."""
        B, T, V = 3, 10, 20
        orig = torch.randn(B, T, V)
        base = torch.randn(B, T, V)
        pos_argmax = torch.tensor([0, 5, 9])
        # interv matches base at all probe positions → score = 0.0
        interv = orig.clone()
        for b, p in enumerate(pos_argmax.tolist()):
            interv[b, p, :] = base[b, p, :]
        score = _calculate_faithfulness_score(orig, interv, base, pos_argmax)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_pos_argmax_first_token(self):
        """pos_argmax = 0 should work (boundary check)."""
        B, T, V = 2, 6, 10
        orig = torch.zeros(B, T, V)
        base = torch.ones(B, T, V)
        pos_argmax = torch.zeros(B, dtype=torch.long)
        score = _calculate_faithfulness_score(orig, orig.clone(), base, pos_argmax)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_pos_argmax_last_token_matches_no_pos_argmax(self):
        """When pos_argmax points to T-1 for all items, result equals no-pos_argmax."""
        B, T, V = 2, 6, 10
        orig = torch.randn(B, T, V)
        base = torch.randn(B, T, V)
        interv = _lerp_logits(orig, base, 0.3)
        pos_argmax = torch.full((B,), T - 1, dtype=torch.long)
        score_with = _calculate_faithfulness_score(orig, interv, base, pos_argmax)
        score_without = _calculate_faithfulness_score(orig, interv, base, None)
        assert score_with == pytest.approx(score_without, abs=1e-5)


# ---------------------------------------------------------------------------
# Batch and shape invariance
# ---------------------------------------------------------------------------

class TestScoreShapeInvariance:
    @pytest.mark.parametrize("B", [1, 4, 16])
    def test_various_batch_sizes(self, B):
        T, V = 8, 100
        orig = torch.randn(B, T, V)
        base = torch.randn(B, T, V)
        score = _calculate_faithfulness_score(orig, orig.clone(), base)
        assert score == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("V", [10, 1000, 50304])
    def test_various_vocab_sizes(self, V):
        B, T = 2, 4
        orig = torch.randn(B, T, V)
        base = torch.randn(B, T, V)
        score = _calculate_faithfulness_score(orig, orig.clone(), base)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_single_token_sequence(self):
        """T=1 — seq_len of 1 with last-token fallback."""
        B, T, V = 2, 1, 20
        orig = torch.randn(B, T, V)
        base = torch.randn(B, T, V)
        score = _calculate_faithfulness_score(orig, orig.clone(), base)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_return_type_is_python_float(self):
        B, T, V = 2, 4, 10
        orig = torch.zeros(B, T, V)
        base = torch.ones(B, T, V)
        score = _calculate_faithfulness_score(orig, orig.clone(), base)
        assert isinstance(score, float)
