"""
Unit tests for FeatureID — round-trip conversions with hand-computed expected values.

Global ID layout: (layer * n_kinds + kind_idx) * d_sae + latent_idx
Component ID:     comp_idx = layer * n_kinds + kind_idx
"""
import pytest
from circuit.types.feature_id import FeatureID

KINDS = ["attn", "mlp", "resid"]
N_KINDS = 3
D_SAE = 100


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestFeatureIDBasics:

    def test_frozen_dataclass(self):
        fid = FeatureID(layer=1, kind="mlp", index=42)
        with pytest.raises(AttributeError):
            fid.layer = 2  # type: ignore[misc]

    def test_key_property(self):
        fid = FeatureID(layer=2, kind="resid", index=7)
        assert fid.key == (2, "resid", 7)

    def test_repr(self):
        fid = FeatureID(layer=0, kind="attn", index=10)
        assert repr(fid) == "L0.attn.f10"

    def test_equality(self):
        a = FeatureID(1, "mlp", 5)
        b = FeatureID(1, "mlp", 5)
        assert a == b

    def test_inequality(self):
        a = FeatureID(0, "attn", 0)
        b = FeatureID(0, "attn", 1)
        assert a != b

    def test_hashable_as_dict_key(self):
        a = FeatureID(0, "attn", 0)
        b = FeatureID(0, "attn", 0)
        d = {a: 1.0}
        assert d[b] == 1.0

    def test_usable_in_set(self):
        s = {FeatureID(0, "attn", 0), FeatureID(0, "attn", 0), FeatureID(1, "mlp", 5)}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Global ID conversions — hand-computed expected values
#
# comp_idx = layer * 3 + kind_idx
# global   = comp_idx * 100 + latent_idx
# ---------------------------------------------------------------------------

class TestGlobalIdConversions:

    @pytest.mark.parametrize("layer, kind, index, expected_global", [
        (0, "attn",  0,   0),        # comp=0
        (0, "attn",  99,  99),       # comp=0, last latent
        (0, "mlp",   0,   100),      # comp=1
        (0, "resid", 0,   200),      # comp=2
        (1, "attn",  0,   300),      # comp=3
        (1, "mlp",   42,  442),      # comp=4, 4*100+42
        (2, "resid", 50,  850),      # comp=8, 8*100+50
    ])
    def test_to_global_id(self, layer, kind, index, expected_global):
        fid = FeatureID(layer, kind, index)
        assert fid.to_global_id(N_KINDS, D_SAE, KINDS) == expected_global

    @pytest.mark.parametrize("expected_global, layer, kind, index", [
        (0,   0, "attn",  0),
        (99,  0, "attn",  99),
        (100, 0, "mlp",   0),
        (200, 0, "resid", 0),
        (300, 1, "attn",  0),
        (442, 1, "mlp",   42),
        (850, 2, "resid", 50),
    ])
    def test_from_global_id(self, expected_global, layer, kind, index):
        fid = FeatureID.from_global_id(expected_global, N_KINDS, D_SAE, KINDS)
        assert fid == FeatureID(layer, kind, index)

    @pytest.mark.parametrize("layer, kind, index", [
        (0, "attn",  0),
        (0, "mlp",   50),
        (1, "resid", 99),
        (3, "mlp",   1),
    ])
    def test_round_trip_global(self, layer, kind, index):
        fid = FeatureID(layer, kind, index)
        gid = fid.to_global_id(N_KINDS, D_SAE, KINDS)
        assert FeatureID.from_global_id(gid, N_KINDS, D_SAE, KINDS) == fid


# ---------------------------------------------------------------------------
# Component ID conversions — hand-computed expected values
#
# comp_idx = layer * 3 + kind_idx
# ---------------------------------------------------------------------------

class TestComponentIdConversions:

    @pytest.mark.parametrize("layer, kind, index, expected_comp", [
        (0, "attn",  5,  0),
        (0, "mlp",   5,  1),
        (0, "resid", 5,  2),
        (1, "attn",  10, 3),
        (2, "mlp",   0,  7),    # 2*3+1=7
    ])
    def test_to_component_id(self, layer, kind, index, expected_comp):
        fid = FeatureID(layer, kind, index)
        comp_idx, latent_idx = fid.to_component_id(N_KINDS, KINDS)
        assert comp_idx == expected_comp
        assert latent_idx == index

    @pytest.mark.parametrize("comp_idx, latent_idx, expected_layer, expected_kind", [
        (0, 5,  0, "attn"),
        (1, 10, 0, "mlp"),
        (2, 0,  0, "resid"),
        (3, 7,  1, "attn"),
        (7, 0,  2, "mlp"),
    ])
    def test_from_component_id(self, comp_idx, latent_idx, expected_layer, expected_kind):
        fid = FeatureID.from_component_id(comp_idx, latent_idx, N_KINDS, KINDS)
        assert fid.layer == expected_layer
        assert fid.kind == expected_kind
        assert fid.index == latent_idx

    @pytest.mark.parametrize("layer, kind, index", [
        (0, "attn",  0),
        (1, "mlp",   42),
        (2, "resid", 99),
    ])
    def test_round_trip_component(self, layer, kind, index):
        fid = FeatureID(layer, kind, index)
        comp, lat = fid.to_component_id(N_KINDS, KINDS)
        assert FeatureID.from_component_id(comp, lat, N_KINDS, KINDS) == fid


# ---------------------------------------------------------------------------
# Cross-conversion round-trip: global ↔ component
# ---------------------------------------------------------------------------

class TestCrossConversion:

    @pytest.mark.parametrize("layer, kind, index", [
        (0, "attn", 0),
        (1, "mlp", 50),
        (2, "resid", 99),
    ])
    def test_global_and_component_agree(self, layer, kind, index):
        fid = FeatureID(layer, kind, index)
        gid = fid.to_global_id(N_KINDS, D_SAE, KINDS)
        comp, lat = fid.to_component_id(N_KINDS, KINDS)
        assert gid == comp * D_SAE + lat
