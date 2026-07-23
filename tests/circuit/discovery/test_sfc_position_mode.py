"""SFC replication: the position-aggregation rules.

`_aggregate_effect` is the ONLY place npa and pa_union differ, so it is pinned
here directly without needing a model.

The provenance assertions matter as much as the arithmetic. `npa` must stay
EXACTLY SFC's published rule — sum over token positions, then example-wise mean
(their App. C; feature-circuits circuit.py:69-70) — because the cross-method
table presents it as their algorithm. `pa_union` is ours and is asserted to be
distinguishable from it. A third value, `pa_peak`, was removed on 2026-07-22:
it was our invention rather than SFC's, and is now asserted to be REJECTED so
it cannot quietly return under a mode list headed "SFC".
"""
import pytest
import torch

from circuit.instrument.position_aware import PositionAwareSpec
from circuit.types.sparse_act import SparseAct
from config import SFCAttributionPatchingConfig


class _Agg:
    """The aggregation surface of SFCAttributionPatching, without its __init__
    (which needs a model, SAE bank and probe builder)."""

    from circuit.discovery.sfc_attribution_patching import (  # noqa: E501
        SFCAttributionPatching as _cls,
    )
    _aggregate_effect = _cls._aggregate_effect

    def __init__(self, position_mode="npa", pa_select="abs_pctl",
                 pa_top_n=64, pa_threshold=90.0):
        self.position_mode = position_mode
        self.pa_select = pa_select
        self.pa_top_n = pa_top_n
        self.pa_threshold = pa_threshold


def _effect(act, resc=None):
    return SparseAct(act=act, resc=resc)


@pytest.fixture
def sparse_effect():
    """[B=2, T=4, d_sae=6]. Latent 0 fires at ONE position (the sparse-support
    majority); latent 1 spreads across three; latent 2 has mixed signs that
    cancel exactly in the sum; latents 3-5 are dead."""
    a = torch.zeros(2, 4, 6)
    a[:, 2, 0] = 5.0                      # single position, strong
    a[:, 0, 1] = 1.0
    a[:, 1, 1] = 1.0
    a[:, 2, 1] = 1.0                      # three positions, weak each
    a[:, 0, 2] = 3.0
    a[:, 1, 2] = -3.0                     # cancels in the sum, not at the peak
    return a


class TestNPA:
    def test_npa_is_sum_over_positions(self, sparse_effect):
        scores, _ = _Agg("npa")._aggregate_effect(_effect(sparse_effect), None)
        assert scores[0].item() == pytest.approx(5.0)
        assert scores[1].item() == pytest.approx(3.0)
        assert scores[2].item() == pytest.approx(0.0)   # exact cancellation

    def test_npa_ignores_pa_spec(self, sparse_effect):
        """npa must be untouched by the PA knobs — it is SFC's own rule."""
        spec = PositionAwareSpec(peaks=torch.tensor([3, 3]), top_n=1, select="top_n")
        a, _ = _Agg("npa")._aggregate_effect(_effect(sparse_effect), spec)
        b, _ = _Agg("npa")._aggregate_effect(_effect(sparse_effect), None)
        assert torch.equal(a, b)


class TestSFCProvenance:
    """npa is presented in our results as SFC's own algorithm, so its arithmetic
    is pinned to what their paper and code specify — nothing more."""

    def test_npa_matches_the_published_reduction_exactly(self, sparse_effect):
        """feature-circuits circuit.py:69-70 — `.sum(dim=1)` over token position,
        then `.mean(dim=0)` over examples."""
        expected = sparse_effect.sum(dim=1).mean(dim=0)
        scores, _ = _Agg("npa")._aggregate_effect(_effect(sparse_effect), None)
        assert torch.allclose(scores, expected)

    def test_no_peak_reduction_is_reachable(self):
        """`pa_peak` (score at the strongest position) is in neither SFC's paper
        nor their reference implementation — it was ours. It must not be
        selectable from a config whose other value is presented as theirs."""
        with pytest.raises(ValueError, match="position_mode must be one of"):
            SFCAttributionPatchingConfig(position_mode="pa_peak")


class TestPAUnion:
    def test_per_position_selection_admits_by_rank_not_magnitude(self, sparse_effect):
        """The distinguishing property: with top_n=1 per position, a latent that
        is weak globally still qualifies where it is locally strongest. A global
        cut on a position-collapsed score cannot express this."""
        spec = PositionAwareSpec(peaks=torch.tensor([3, 3]), top_n=1, select="top_n")
        scores, _ = _Agg("pa_union", pa_select="top_n", pa_top_n=1)._aggregate_effect(
            _effect(sparse_effect), spec)
        # position 0 -> latent 2 (3.0) wins; position 1 -> latent 2 (-3.0);
        # position 2 -> latent 0 (5.0). Latent 1 never wins a position.
        assert scores[1].item() == pytest.approx(0.0)
        assert scores[0].item() == pytest.approx(5.0)
        assert abs(scores[2].item()) == pytest.approx(3.0)

    def test_is_distinguishable_from_sfcs_own_rule(self, sparse_effect):
        """Guard against a regression that quietly collapses our extension back
        onto SFC's reduction — which would make a comparison meaningless."""
        spec = PositionAwareSpec(peaks=torch.tensor([3, 3]), top_n=1, select="top_n")
        union, _ = _Agg("pa_union", pa_select="top_n", pa_top_n=1)._aggregate_effect(
            _effect(sparse_effect), spec)
        npa, _ = _Agg("npa")._aggregate_effect(_effect(sparse_effect), None)
        assert not torch.equal(union, npa)

    def test_respects_causal_prefix(self, sparse_effect):
        """Positions after the metric's argmax must not contribute."""
        a = torch.zeros(2, 4, 6)
        a[:, 3, 4] = 9.0                                  # only after the anchor
        spec = PositionAwareSpec(peaks=torch.tensor([1, 1]), top_n=8, select="top_n")
        scores, _ = _Agg("pa_union", pa_select="top_n", pa_top_n=8)._aggregate_effect(
            _effect(a), spec)
        assert scores[4].item() == pytest.approx(0.0)

    def test_requires_a_resolved_spec(self, sparse_effect):
        with pytest.raises(ValueError, match="pa_union requires"):
            _Agg("pa_union")._aggregate_effect(_effect(sparse_effect), None)


class TestErrorTerm:
    def test_npa_sums_error_over_positions(self, sparse_effect):
        resc = torch.zeros(2, 4, 1)
        resc[:, 0, 0] = 2.0
        resc[:, 1, 0] = 1.0
        _, res = _Agg("npa")._aggregate_effect(_effect(sparse_effect, resc), None)
        assert res.reshape(-1)[0].item() == pytest.approx(3.0)

    def test_pa_union_takes_error_at_its_strongest_position(self, sparse_effect):
        """One error node per site, so there is no union to take."""
        resc = torch.zeros(2, 4, 1)
        resc[:, 0, 0] = 2.0
        resc[:, 1, 0] = 1.0
        spec = PositionAwareSpec(peaks=torch.tensor([3, 3]), top_n=8, select="top_n")
        _, res = _Agg("pa_union", pa_select="top_n", pa_top_n=8)._aggregate_effect(
            _effect(sparse_effect, resc), spec)
        assert res.reshape(-1)[0].item() == pytest.approx(2.0)


class TestConfig:
    @pytest.mark.parametrize("mode", ["npa", "pa_union"])
    def test_accepts_the_two_rules(self, mode):
        assert SFCAttributionPatchingConfig(position_mode=mode).position_mode == mode

    def test_rejects_the_retired_pa_alias(self):
        """"pa" was the old binary flag; it must fail loudly rather than
        silently resolving to one of the rules."""
        with pytest.raises(ValueError, match="position_mode must be one of"):
            SFCAttributionPatchingConfig(position_mode="pa")

    def test_default_is_sfcs_own_rule(self):
        assert SFCAttributionPatchingConfig().position_mode == "npa"

    def test_pa_select_validated(self):
        assert SFCAttributionPatchingConfig(pa_select="top_n").pa_select == "top_n"
        with pytest.raises(ValueError, match="pa_select must be one of"):
            SFCAttributionPatchingConfig(pa_select="nonsense")

    def test_pa_defaults_match_our_validated_settings(self):
        c = SFCAttributionPatchingConfig()
        assert (c.pa_select, c.pa_threshold) == ("abs_pctl", 90.0)
