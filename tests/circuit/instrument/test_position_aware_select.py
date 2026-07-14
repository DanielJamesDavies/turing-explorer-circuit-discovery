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
    assert set(SELECT_MODES) == {"top_n", "abs", "relative", "mass"}


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
