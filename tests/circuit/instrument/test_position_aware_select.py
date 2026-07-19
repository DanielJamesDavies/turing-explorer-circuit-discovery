"""
Tests for position-aware allowed-set per-position selection rules
(`_selection_mask`) and the `select` / `threshold` toggle in
`position_aware_membership`.

The four rules (config `position_aware_select`):
  top_n    -- fixed count per position (position_aware_top_n).
  abs      -- |attr| >= threshold (global absolute cut).
  relative -- |attr| >= threshold * max|attr| at that position (scale-free).
  mass     -- smallest set covering `threshold` of the position's |attr| mass.

Part 1  _selection_mask semantics on a peaked and a flat row.
Part 2  Invariants (top-1 always kept for mass; f=0 keeps all; threshold monotonicity).
Part 3  Config validation of the two new fields.
"""

import pytest
import torch

from circuit.instrument.position_aware import _selection_mask, SELECT_MODES


# Row 0: peaked (one big, one medium, long thin tail). Row 1: perfectly flat.
BLOCK = torch.tensor([
    [10.0, -4.0, 1.0, 0.5, 0.2, 0.1],
    [2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
])
BA = BLOCK.abs()


def kept(mask, row):
    return set(mask[row].nonzero().flatten().tolist())


# ---------------------------------------------------------------- Part 1
def test_top_n_fixed_count():
    m = _selection_mask(BA, "top_n", 0.0, 2)
    assert kept(m, 0) == {0, 1}          # the two largest |attr|
    assert m[0].sum().item() == 2
    assert m[1].sum().item() == 2        # fixed count regardless of ties/flatness


def test_abs_global_cut():
    m = _selection_mask(BA, "abs", 1.0, 0)
    assert kept(m, 0) == {0, 1, 2}       # 10, 4, 1 pass; 0.5/0.2/0.1 do not
    assert kept(m, 1) == {0, 1, 2, 3, 4}  # all the 2.0s pass; the 0.0 does not


def test_relative_scale_free():
    # f = 0.5 -> keep |attr| >= 0.5 * rowmax.
    m = _selection_mask(BA, "relative", 0.5, 0)
    assert kept(m, 0) == {0}             # rowmax 10 -> cut 5 -> only the 10
    assert kept(m, 1) == {0, 1, 2, 3, 4}  # rowmax 2 -> cut 1 -> every 2.0


def test_mass_cumulative():
    # Row 0 total |attr| = 15.8; 10/15.8 = 63%, +4 = 88.6%.
    assert kept(_selection_mask(BA, "mass", 0.8, 0), 0) == {0, 1}
    # Row 1 total 10; need 80% -> four of the 2.0s (the fifth would sit exactly
    # at the boundary and is excluded by the exclusive-prefix rule).
    assert len(kept(_selection_mask(BA, "mass", 0.8, 0), 1)) == 4


# ---------------------------------------------------------------- Part 2
def test_mass_always_keeps_top_one():
    # Even a vanishingly small mass threshold keeps at least the largest latent.
    m = _selection_mask(BA, "mass", 1e-6, 0)
    assert kept(m, 0) == {0}
    assert kept(m, 1) and len(kept(m, 1)) == 1


def test_relative_zero_keeps_all_nonzero_positions():
    # f = 0 -> keep everything with |attr| >= 0, i.e. all latents.
    m = _selection_mask(BA, "relative", 0.0, 0)
    assert m[0].all() and m[1].all()


def test_threshold_monotonicity():
    # Raising an abs threshold can only shrink the kept set.
    low = _selection_mask(BA, "abs", 0.5, 0)
    high = _selection_mask(BA, "abs", 2.0, 0)
    for r in range(BA.shape[0]):
        assert kept(high, r) <= kept(low, r)


def test_mass_threshold_monotonicity():
    small = _selection_mask(BA, "mass", 0.5, 0)
    large = _selection_mask(BA, "mass", 0.95, 0)
    for r in range(BA.shape[0]):
        assert kept(small, r) <= kept(large, r)


def test_dead_position_selects_nothing():
    # A block with one live (peaked) row and one dead (all-zero) row: the dead
    # row must contribute NOTHING under relative/mass (not the whole dictionary),
    # while the live row is unaffected.
    block = torch.tensor([
        [10.0, 4.0, 1.0, 0.5, 0.2, 0.1],   # live
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # dead
    ]).abs()
    for select, thr in [("relative", 0.1), ("mass", 0.9)]:
        m = _selection_mask(block, select, thr, 0)
        assert m[1].sum().item() == 0, f"{select}: dead row must select nothing"
        assert m[0].sum().item() >= 1, f"{select}: live row must still select"


def test_dead_position_guard_does_not_touch_abs():
    # abs needs no guard and must be untouched: 0 >= positive threshold is False.
    block = torch.zeros(2, 6)
    block[0, 0] = 5.0
    m = _selection_mask(block.abs(), "abs", 1.0, 0)
    assert kept(m, 0) == {0} and m[1].sum().item() == 0


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        _selection_mask(BA, "bogus", 0.0, 4)


def test_all_modes_listed():
    assert set(SELECT_MODES) == {"top_n", "abs", "relative", "mass", "abs_pctl"}


def test_config_abs_pctl_threshold_must_be_percentile():
    from config import DiscoveryConfig
    c = DiscoveryConfig(position_aware_select="abs_pctl", position_aware_threshold=90)
    assert c.position_aware_threshold == 90
    with pytest.raises(ValueError, match="percentile"):
        DiscoveryConfig(position_aware_select="abs_pctl", position_aware_threshold=0.0)
    with pytest.raises(ValueError, match="percentile"):
        DiscoveryConfig(position_aware_select="abs_pctl", position_aware_threshold=150)


# ---------------------------------------------------------------- Part 3
def test_config_accepts_valid_select_and_threshold():
    from config import DiscoveryConfig
    c = DiscoveryConfig(position_aware=True, position_aware_select="mass",
                        position_aware_threshold=0.9)
    assert c.position_aware_select == "mass"
    assert c.position_aware_threshold == 0.9


def test_config_rejects_bad_select():
    from config import DiscoveryConfig
    with pytest.raises(ValueError):
        DiscoveryConfig(position_aware_select="topk")


def test_config_rejects_negative_threshold():
    from config import DiscoveryConfig
    with pytest.raises(ValueError):
        DiscoveryConfig(position_aware_threshold=-0.1)


# ------------------------------------- shared selector backend / spec
def test_select_position_aware_unions_over_prefix():
    """The backend keeps the position axis: each position in the seed's causal
    prefix selects its own top-N and the union is taken (vs classic, which would
    .sum(dim=(0,1)) the axis away)."""
    from circuit.instrument.position_aware import select_position_aware
    B, T, D = 1, 4, 6
    attr = torch.zeros(B, T, D)
    attr[0, 0, 1] = 5.0     # position 0 -> latent 1
    attr[0, 1, 3] = 4.0     # position 1 -> latent 3
    attr[0, 2, 5] = 3.0     # position 2 -> latent 5
    attr[0, 3, 0] = 9.0     # position 3 is AFTER the peak -> must be excluded
    peaks = torch.tensor([2])  # causal prefix = positions 0..2
    sel = select_position_aware(attr, peaks, top_n=1)
    assert set(sel) == {1, 3, 5}          # one per prefix position, unioned
    assert 0 not in sel                    # post-peak position excluded


def test_select_position_aware_rejects_wrong_shape():
    from circuit.instrument.position_aware import select_position_aware
    with pytest.raises(ValueError):
        select_position_aware(torch.zeros(4, 6), torch.tensor([1]), top_n=1)


def test_resolve_role_delivery_semantics():
    """PA keeps both signs as members (union invariant across the toggle);
    NPA-exclude drops negatives. include labels inhibitors; PA-exclude folds
    them into supports as |score|."""
    from circuit.instrument.position_aware import resolve_role_delivery
    from circuit.types.feature_id import FeatureID
    sup = {FeatureID(0, "attn", 0): 2.0}
    neg = {FeatureID(1, "mlp", 3): -5.0}

    # include: negatives come back as inhibitors, signed.
    s, i = resolve_role_delivery(sup, neg, position_aware=True, include_negatives=True)
    assert s == sup and i == neg

    # PA + exclude: same MEMBERSHIP (union of keys), negatives folded into
    # supports as |score|, no inhibitors.
    s, i = resolve_role_delivery(sup, neg, position_aware=True, include_negatives=False)
    assert i == {}
    assert set(s) == {FeatureID(0, "attn", 0), FeatureID(1, "mlp", 3)}
    assert s[FeatureID(1, "mlp", 3)] == 5.0            # |score|, kept as support
    # include vs exclude cover the identical member set for PA
    inc_s, inc_i = resolve_role_delivery(sup, neg, position_aware=True, include_negatives=True)
    assert set(s) == set(inc_s) | set(inc_i)

    # NPA + exclude: negatives genuinely dropped (classic seed-driving select).
    s, i = resolve_role_delivery(sup, neg, position_aware=False, include_negatives=False)
    assert s == sup and i == {}


def test_select_values_tensor_matches_dict_path():
    """select_position_aware_values is the tensor twin of select_position_aware:
    identical membership and identical signed max-|score| values, across rules
    and random inputs (zeros = unselected)."""
    from circuit.instrument.position_aware import (
        select_position_aware, select_position_aware_values,
    )
    g = torch.Generator().manual_seed(7)
    B, T, D = 3, 6, 24
    attr = torch.randn(B, T, D, generator=g)
    attr[attr.abs() < 0.6] = 0.0          # realistic sparsity + exact zeros
    peaks = torch.tensor([2, 5, 0])
    for select, thr, n in [("top_n", 0.0, 3), ("abs", 0.7, 0),
                           ("relative", 0.4, 0), ("mass", 0.8, 0)]:
        want = select_position_aware(attr, peaks, top_n=n, select=select,
                                     threshold=thr)
        got = select_position_aware_values(attr, peaks, top_n=n, select=select,
                                           threshold=thr)
        got_dict = {int(i): float(v) for i, v in
                    zip(got.nonzero(as_tuple=True)[0].tolist(),
                        got[got != 0].tolist())}
        want_nonzero = {k: v for k, v in want.items() if v != 0.0}
        assert got_dict.keys() == want_nonzero.keys(), select
        for k in want_nonzero:
            assert got_dict[k] == pytest.approx(want_nonzero[k], abs=1e-6), select


def test_position_aware_spec_carries_selection_rule():
    from circuit.instrument.position_aware import PositionAwareSpec
    B, T, D = 1, 3, 6
    attr = torch.zeros(B, T, D)
    attr[0, 0, 2] = 7.0
    attr[0, 1, 4] = 6.0
    spec = PositionAwareSpec(peaks=torch.tensor([1]), top_n=1)
    assert set(spec.select_from(attr)) == {2, 4}


def test_activation_gradient_is_a_method_not_a_mode():
    """Promoted to a top-level method: neither gradient method accepts it as an
    attribution_mode any more, and it is registered as its own method."""
    from config import AblationGradientConfig, CounterfactualGradientConfig
    from circuit.discovery_window import METHOD_REGISTRY
    from circuit.discovery.activation_gradient import ActivationGradientDiscovery

    with pytest.raises(ValueError):
        CounterfactualGradientConfig(attribution_mode="activation_gradient")
    with pytest.raises(ValueError):
        AblationGradientConfig(attribution_mode="activation_gradient")
    with pytest.raises(ValueError):
        CounterfactualGradientConfig(attribution_mode="bogus_mode")

    assert METHOD_REGISTRY["activation_gradient"] is ActivationGradientDiscovery
    assert ActivationGradientDiscovery.method_name == "activation_gradient"
    # A SIBLING of ablation gradient on the shared pipeline base (2026-07-18
    # refactor) — it shares assembly/eval via GradientDiscoveryBase but is NOT
    # an ablation gradient: no attribution_mode dispatch, its own hop.
    from circuit.discovery.ablation_gradient import AblationGradientDiscovery
    from circuit.discovery.gradient_base import GradientDiscoveryBase
    assert issubclass(ActivationGradientDiscovery, GradientDiscoveryBase)
    assert not issubclass(ActivationGradientDiscovery, AblationGradientDiscovery)
    assert issubclass(AblationGradientDiscovery, GradientDiscoveryBase)


def test_config_accepts_scope_and_position_weight():
    from config import DiscoveryConfig
    c = DiscoveryConfig(position_aware=True, position_aware_scope="per_instance",
                        position_aware_position_weight=True)
    assert c.position_aware_scope == "per_instance"
    assert c.position_aware_position_weight is True


def test_config_rejects_bad_scope():
    from config import DiscoveryConfig
    with pytest.raises(ValueError):
        DiscoveryConfig(position_aware_scope="single")


# ---------------------------------------------------------------- Part 4
# batch_size chunking in position_aware_membership: chunked == single pass
# (the objective is a SUM over sequences, and cross-chunk merge uses the same
# max-|score| rule the selection applies across sequences within a pass).

def _membership_arms(batch_size):
    import types
    from unittest.mock import MagicMock

    from circuit.instrument.position_aware import position_aware_membership
    from tests.conftest import MockSAEBank

    torch.manual_seed(3)
    bank = MockSAEBank()
    # position_aware_membership reads the production SAE surface
    # (sae.encoder.weight / _get_bias_eff); graft it onto the mock modules.
    for kind in bank.kinds:
        for mod in bank.saes[kind]:
            mod.encoder = types.SimpleNamespace(weight=mod.W_enc)
            mod._get_bias_eff = lambda m=mod: m.b_enc

    n, t = 6, 4
    x0 = torch.randn(n, t, bank.d_model)

    def forward_fn(tokens, patcher=None, **kwargs):
        # Row 0 of each sequence carries its index into x0, so chunked calls
        # see exactly their own sequences.
        x = x0[tokens[:, 0]]
        out0 = patcher.transform(0, "attn", x.clone())
        patcher.transform(1, "resid", out0)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    tokens = torch.zeros(n, t, dtype=torch.long)
    tokens[:, 0] = torch.arange(n)

    members, _ = position_aware_membership(
        inf, bank,
        tokens=tokens, sites={(0, "attn")},
        seed_layer=1, seed_kind="resid", seed_latent_idx=0,
        pos_argmax=torch.full((n,), t - 1, dtype=torch.long),
        top_n=4, batch_size=batch_size,
    )
    return members


def test_membership_chunked_matches_single_pass():
    single = _membership_arms(batch_size=None)
    chunked = _membership_arms(batch_size=2)
    assert single, "stub produced no members — test harness broken"
    assert set(chunked) == set(single)
    for fid, val in single.items():
        assert chunked[fid] == pytest.approx(val, rel=1e-4, abs=1e-5)


def test_membership_uneven_chunks_match_single_pass():
    single = _membership_arms(batch_size=None)
    chunked = _membership_arms(batch_size=4)  # 6 seqs -> chunks of 4 + 2
    assert set(chunked) == set(single)


# ---------------------------------------------------------------- Part 5
# "abs_pctl": abs whose cut is the pctl-th percentile of the pass's POOLED
# nonzero |attr| across sites — resolved by the frontends, never by
# _selection_mask directly.

def test_pooled_abs_threshold_quantile():
    from circuit.instrument.position_aware import pooled_abs_threshold
    # 1..100 pooled across two tensors (zeros excluded from the distribution)
    a = torch.arange(0, 51, dtype=torch.float32).view(1, 1, -1)   # 0 dropped
    b = torch.arange(51, 101, dtype=torch.float32).view(1, 1, -1)
    th = pooled_abs_threshold([a, b], 90)
    assert th == pytest.approx(90.1, abs=0.5)


def test_pooled_abs_threshold_pools_across_sites():
    """One global cut: a strong site must raise the bar for a weak site."""
    from circuit.instrument.position_aware import pooled_abs_threshold
    weak = torch.full((1, 1, 10), 1.0)
    strong = torch.full((1, 1, 90), 100.0)
    th = pooled_abs_threshold([weak, strong], 50)
    assert th > 1.0  # the weak site's values sit below the pooled median


def test_pooled_abs_threshold_empty_admits_nothing():
    from circuit.instrument.position_aware import pooled_abs_threshold
    assert pooled_abs_threshold([torch.zeros(1, 1, 4)], 90) == float("inf")


def test_selection_mask_rejects_unresolved_abs_pctl():
    with pytest.raises(ValueError, match="resolved"):
        _selection_mask(BA, "abs_pctl", 90.0, 0)


def test_spec_resolved_for_becomes_abs():
    from circuit.instrument.position_aware import PositionAwareSpec, pooled_abs_threshold
    attrs = [torch.arange(1, 101, dtype=torch.float32).view(1, 1, -1)]
    spec = PositionAwareSpec(peaks=torch.tensor([0]), select="abs_pctl", threshold=90)
    resolved = spec.resolved_for(attrs)
    assert resolved.select == "abs"
    assert resolved.threshold == pytest.approx(pooled_abs_threshold(attrs, 90), rel=1e-6)
    # non-pctl specs pass through untouched
    plain = PositionAwareSpec(peaks=torch.tensor([0]), select="abs", threshold=0.5)
    assert plain.resolved_for(attrs) is plain


def test_membership_abs_pctl_equals_abs_at_derived_cut():
    """The equivalence contract: abs_pctl(p) == abs(threshold=pooled p-quantile
    of the same attribution). Verified through the production membership path."""
    from circuit.instrument.position_aware import pooled_abs_threshold

    # Capture the pooled threshold the pctl run derives by spying the samples:
    # simpler — run abs_pctl, then reconstruct the cut from the same stub attrs
    # via a plain-abs run at that value and compare memberships.
    members_pctl = _membership_arms_select("abs_pctl", 75.0)
    # derive the same pooled cut offline from the captured attrs
    attrs = _membership_arms_select.last_attrs
    th = pooled_abs_threshold(attrs, 75.0)
    members_abs = _membership_arms_select("abs", th)
    assert set(members_pctl) == set(members_abs)
    for fid, val in members_abs.items():
        assert members_pctl[fid] == pytest.approx(val, rel=1e-5)


def _membership_arms_select(select, threshold):
    """Membership run with a chosen select rule on a deterministic stub;
    stashes the attribution tensors on the function for offline checks."""
    import types
    from unittest.mock import MagicMock

    import circuit.instrument.position_aware as pam
    from tests.conftest import MockSAEBank

    torch.manual_seed(11)
    bank = MockSAEBank()
    for kind in bank.kinds:
        for mod in bank.saes[kind]:
            mod.encoder = types.SimpleNamespace(weight=mod.W_enc)
            mod._get_bias_eff = lambda m=mod: m.b_enc

    n, t = 4, 4
    x0 = torch.randn(n, t, bank.d_model)

    def forward_fn(tokens, patcher=None, **kwargs):
        x = x0[tokens[:, 0]]
        out0 = patcher.transform(0, "attn", x.clone())
        patcher.transform(1, "resid", out0)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    tokens = torch.zeros(n, t, dtype=torch.long)
    tokens[:, 0] = torch.arange(n)

    captured = []
    orig = pam.select_position_aware

    def spy(attr, peaks, **kw):
        captured.append(attr.detach().clone())
        return orig(attr, peaks, **kw)

    pam.select_position_aware = spy
    try:
        members, _ = pam.position_aware_membership(
            inf, bank,
            tokens=tokens, sites={(0, "attn")},
            seed_layer=1, seed_kind="resid", seed_latent_idx=0,
            pos_argmax=torch.full((n,), t - 1, dtype=torch.long),
            top_n=4, select=select, threshold=threshold,
        )
    finally:
        pam.select_position_aware = orig
    _membership_arms_select.last_attrs = captured
    return members
