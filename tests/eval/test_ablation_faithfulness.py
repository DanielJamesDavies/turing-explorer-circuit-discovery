"""
Unit tests for CircuitOnlyPatcher and evaluate_ablation_faithfulness.

Structure
---------
Part 1  TestUpstreamSites
            Pure-function tests for the in-scope site enumeration.

Part 2  TestCircuitOnlyPatcher
            Exact-arithmetic invariants of the transform, using MockSAEBank
            (real linear encode/decode from conftest):
              keep-all  → identity (decode(all) + error == x)
              keep-none → decode(ablation floor) + error, computed manually
              mean mode → mean vector replaces the floor
              out-of-scope sites untouched

Part 3  TestEvaluateAblationFaithfulness
            Score formula and call protocol with a stateful ControlledSAEBank
            and a stub inference driving the forward passes.

Formula under test:
    score = (a_circuit_only - a_empty) / (a_posctx - a_empty)
"""

import pytest
import torch
from unittest.mock import MagicMock

from eval.ablation_faithfulness import (
    CircuitOnlyPatcher,
    circuit_only_activation,
    collect_site_anchors,
    collect_site_means,
    evaluate_ablation_faithfulness,
    measure_seed_activation,
    upstream_sites,
)
from sae.dense import sparse_topk_to_dense
from store.circuits import Circuit, CircuitNode
from circuit.types.feature_id import FeatureID

from tests.conftest import D_MODEL, D_SAE, K_SAE, KINDS, N_LAYERS, MockSAEBank


B, T = 2, 4

SEED_LAYER = 1
SEED_KIND = "resid"
SEED_LATENT = 5


# ---------------------------------------------------------------------------
# Part 1 — upstream_sites
# ---------------------------------------------------------------------------


class TestUpstreamSites:
    def test_layer0_attn_has_no_upstream(self, mock_sae_bank):
        assert upstream_sites(mock_sae_bank, 0, "attn") == set()

    def test_layer0_mlp_sees_same_layer_attn(self, mock_sae_bank):
        assert upstream_sites(mock_sae_bank, 0, "mlp") == {(0, "attn")}

    def test_layer1_resid_sees_all_lower_and_same_layer_earlier(self, mock_sae_bank):
        expected = {
            (0, "attn"),
            (0, "mlp"),
            (0, "resid"),
            (1, "attn"),
            (1, "mlp"),
        }
        assert upstream_sites(mock_sae_bank, 1, "resid") == expected

    def test_seed_site_never_in_scope(self, mock_sae_bank):
        for layer in range(N_LAYERS):
            for kind in KINDS:
                assert (layer, kind) not in upstream_sites(mock_sae_bank, layer, kind)


# ---------------------------------------------------------------------------
# Part 2 — CircuitOnlyPatcher transform invariants
# ---------------------------------------------------------------------------


def _natural_dense(bank, x, kind, layer):
    ta, ti = bank.encode(x, kind, layer)
    return sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)


def _make_patcher(bank, keep=None, in_scope=None, site_means=None, pin_values=None):
    return CircuitOnlyPatcher(
        bank=bank,
        keep_indices=keep or {},
        in_scope=in_scope if in_scope is not None else {(0, "attn")},
        seed_layer=SEED_LAYER,
        seed_kind=SEED_KIND,
        seed_latent_idx=SEED_LATENT,
        site_means=site_means,
        pin_values=pin_values,
    )


class TestCircuitOnlyPatcher:
    def test_out_of_scope_site_untouched(self, mock_sae_bank):
        patcher = _make_patcher(mock_sae_bank, in_scope={(0, "attn")})
        x = torch.randn(B, T, D_MODEL)
        out = patcher.transform(0, "mlp", x)
        assert torch.equal(out, x)

    def test_keep_all_is_identity(self, mock_sae_bank):
        """Keeping every latent must reproduce x exactly: decode(all) + (x -
        decode(all)) == x. Guards the error-preservation arithmetic."""
        keep = {(0, "attn"): set(range(D_SAE))}
        patcher = _make_patcher(mock_sae_bank, keep=keep, in_scope={(0, "attn")})
        x = torch.randn(B, T, D_MODEL)
        out = patcher.transform(0, "attn", x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_keep_none_zero_mode_leaves_floor_plus_error(self, mock_sae_bank):
        """With nothing kept (zero mode) the output must equal
        decode(zeros) + (x - decode(natural)), computed independently."""
        patcher = _make_patcher(mock_sae_bank, keep={}, in_scope={(0, "attn")})
        x = torch.randn(B, T, D_MODEL)
        dense = _natural_dense(mock_sae_bank, x, "attn", 0)
        expected = (
            mock_sae_bank.decode(torch.zeros_like(dense), "attn", 0)
            + x
            - mock_sae_bank.decode(dense, "attn", 0)
        )
        out = patcher.transform(0, "attn", x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_keep_none_mean_mode_uses_mean_floor(self, mock_sae_bank):
        """With nothing kept (mean mode) the output must equal
        decode(mean vector) + error."""
        mean_vector = torch.rand(D_SAE) * 0.5
        patcher = _make_patcher(
            mock_sae_bank,
            keep={},
            in_scope={(0, "attn")},
            site_means={(0, "attn"): mean_vector},
        )
        x = torch.randn(B, T, D_MODEL)
        dense = _natural_dense(mock_sae_bank, x, "attn", 0)
        mean_dense = mean_vector.expand_as(dense)
        expected = (
            mock_sae_bank.decode(mean_dense, "attn", 0)
            + x
            - mock_sae_bank.decode(dense, "attn", 0)
        )
        out = patcher.transform(0, "attn", x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_partial_keep_restores_kept_latents_over_mean_floor(self, mock_sae_bank):
        """Kept latents must carry their natural values; the rest the mean."""
        x = torch.randn(B, T, D_MODEL)
        dense = _natural_dense(mock_sae_bank, x, "attn", 0)
        kept = {3, 11}
        mean_vector = torch.rand(D_SAE) * 0.5
        patcher = _make_patcher(
            mock_sae_bank,
            keep={(0, "attn"): kept},
            in_scope={(0, "attn")},
            site_means={(0, "attn"): mean_vector},
        )
        expected_latents = mean_vector.expand_as(dense).clone()
        keep_tensor = torch.tensor(sorted(kept))
        expected_latents[:, :, keep_tensor] = dense[:, :, keep_tensor]
        expected = (
            mock_sae_bank.decode(expected_latents, "attn", 0)
            + x
            - mock_sae_bank.decode(dense, "attn", 0)
        )
        out = patcher.transform(0, "attn", x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_pinned_keep_overrides_natural_values(self, mock_sae_bank):
        """In pinned mode, kept latents must carry the pin values (not the
        natural encoding); the rest keep the mean floor."""
        x = torch.randn(B, T, D_MODEL)
        dense = _natural_dense(mock_sae_bank, x, "attn", 0)
        kept = {3, 11}
        mean_vector = torch.rand(D_SAE) * 0.5
        pin_vector = torch.rand(D_SAE) * 3.0
        patcher = _make_patcher(
            mock_sae_bank,
            keep={(0, "attn"): kept},
            in_scope={(0, "attn")},
            site_means={(0, "attn"): mean_vector},
            pin_values={(0, "attn"): pin_vector},
        )
        expected_latents = mean_vector.expand_as(dense).clone()
        keep_tensor = torch.tensor(sorted(kept))
        expected_latents[:, :, keep_tensor] = pin_vector[keep_tensor]
        expected = (
            mock_sae_bank.decode(expected_latents, "attn", 0)
            + x
            - mock_sae_bank.decode(dense, "attn", 0)
        )
        out = patcher.transform(0, "attn", x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_pinned_with_empty_keep_matches_free(self, mock_sae_bank):
        """With nothing kept, pin values are irrelevant: pinned and free
        empty-circuit floors must be identical (shared a_empty)."""
        x = torch.randn(B, T, D_MODEL)
        mean_vector = torch.rand(D_SAE) * 0.5
        free = _make_patcher(
            mock_sae_bank, keep={}, in_scope={(0, "attn")},
            site_means={(0, "attn"): mean_vector},
        )
        pinned = _make_patcher(
            mock_sae_bank, keep={}, in_scope={(0, "attn")},
            site_means={(0, "attn"): mean_vector},
            pin_values={(0, "attn"): torch.rand(D_SAE) * 9.0},
        )
        assert torch.allclose(free.transform(0, "attn", x), pinned.transform(0, "attn", x.clone()), atol=1e-6)

    def test_capture_at_seed_site(self, mock_sae_bank):
        """captured_activation must equal the seed latent's dense value at
        the argmax positions, computed independently."""
        x = torch.randn(B, T, D_MODEL)
        dense = _natural_dense(mock_sae_bank, x, SEED_KIND, SEED_LAYER)
        argmax = torch.tensor([1, 3])
        expected = dense[torch.arange(B), argmax, SEED_LATENT].mean().item()

        patcher = _make_patcher(mock_sae_bank, in_scope=set())
        patcher.pos_argmax = argmax
        out = patcher.transform(SEED_LAYER, SEED_KIND, x)
        assert torch.equal(out, x)  # seed site never modified
        assert patcher.captured_activation == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Part 3 — evaluate_ablation_faithfulness formula and protocol
# ---------------------------------------------------------------------------


class ControlledSAEBank:
    """encode returns the configured seed activation at the seed site and
    zeros elsewhere; decode returns zeros (so transforms are inert)."""

    d_sae = D_SAE
    kinds = KINDS
    n_layer = N_LAYERS

    def __init__(self, seed_act: float = 0.0):
        self.current_seed_act = seed_act

    def encode(self, x, kind, layer):
        Bx, Tx = x.shape[:2]
        top_acts = torch.zeros(Bx, Tx, K_SAE)
        top_indices = torch.zeros(Bx, Tx, K_SAE, dtype=torch.long)
        if layer == SEED_LAYER and kind == SEED_KIND:
            top_indices[..., 0] = SEED_LATENT
            top_acts[..., 0] = self.current_seed_act
        return top_acts, top_indices

    def decode(self, latents, kind, layer):
        return torch.zeros(*latents.shape[:-1], D_MODEL)


def _make_stub_inference(bank, pass_acts):
    """Each forward sets the bank's seed act for that pass, then drives the
    callback (all layers) or the patcher (all layers x kinds)."""

    call_num = [0]

    def forward_fn(tokens, activations_callback=None, patcher=None, **kwargs):
        call_num[0] += 1
        bank.current_seed_act = pass_acts[call_num[0] - 1]
        x = torch.zeros(B, T, D_MODEL)
        acts_tuple = tuple(x.clone() for _ in KINDS)
        for layer in range(N_LAYERS):
            if activations_callback is not None:
                activations_callback(layer, acts_tuple)
            if patcher is not None:
                for kind in KINDS:
                    patcher.transform(layer, kind, x.clone())

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    return inf, call_num


def _make_circuit():
    c = Circuit(name="test")
    c.add_node(CircuitNode(metadata={
        "feature_id": FeatureID(SEED_LAYER, SEED_KIND, SEED_LATENT),
        "role": "seed",
    }))
    c.add_node(CircuitNode(metadata={
        "feature_id": FeatureID(0, "attn", 7),
        "role": "ablation_support",
    }))
    return c


def _avg_acts():
    return torch.zeros(N_LAYERS * len(KINDS), D_SAE)


class TestEvaluateAblationFaithfulness:
    def _run(self, *, circ_act, a_posctx=2.0, a_empty=0.5, ablation="mean"):
        bank = ControlledSAEBank()
        inf, counter = _make_stub_inference(bank, (circ_act,))
        sites = upstream_sites(bank, SEED_LAYER, SEED_KIND)
        site_means = {site: torch.zeros(D_SAE) for site in sites}
        score = evaluate_ablation_faithfulness(
            inf, bank, _avg_acts(), _make_circuit(),
            pos_tokens=torch.zeros(B, T, dtype=torch.long),
            seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT,
            pos_argmax=torch.zeros(B, dtype=torch.long),
            ablation=ablation,
            site_means=site_means,
            a_posctx=a_posctx,
            a_empty=a_empty,
        )
        return score, counter

    def test_score_one_when_circuit_recovers_posctx(self):
        (score, _), _ = self._run(circ_act=2.0, a_posctx=2.0, a_empty=0.5)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_score_zero_at_empty_floor(self):
        (score, _), _ = self._run(circ_act=0.5, a_posctx=2.0, a_empty=0.5)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_score_half_midway(self):
        (score, _), _ = self._run(circ_act=1.25, a_posctx=2.0, a_empty=0.5)
        assert score == pytest.approx(0.5, abs=1e-5)

    def test_score_unclamped_above_one(self):
        (score, _), _ = self._run(circ_act=3.5, a_posctx=2.0, a_empty=0.5)
        assert score == pytest.approx(2.0, abs=1e-5)

    def test_single_pass_when_anchors_supplied(self):
        _, counter = self._run(circ_act=1.0)
        assert counter[0] == 1

    def test_four_passes_when_nothing_supplied_mean_mode(self):
        """mean mode with no anchors: site means, a_posctx, a_empty,
        circuit-only — exactly four forwards."""
        bank = ControlledSAEBank()
        inf, counter = _make_stub_inference(bank, (0.0, 2.0, 0.5, 1.25))
        score, _ = evaluate_ablation_faithfulness(
            inf, bank, _avg_acts(), _make_circuit(),
            pos_tokens=torch.zeros(B, T, dtype=torch.long),
            seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT,
            pos_argmax=torch.zeros(B, dtype=torch.long),
            ablation="mean",
        )
        assert counter[0] == 4
        assert score == pytest.approx((1.25 - 0.5) / (2.0 - 0.5), abs=1e-5)

    def test_three_passes_zero_mode(self):
        """zero mode skips the site-means pass."""
        bank = ControlledSAEBank()
        inf, counter = _make_stub_inference(bank, (2.0, 0.5, 1.25))
        evaluate_ablation_faithfulness(
            inf, bank, _avg_acts(), _make_circuit(),
            pos_tokens=torch.zeros(B, T, dtype=torch.long),
            seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT,
            pos_argmax=torch.zeros(B, dtype=torch.long),
            ablation="zero",
        )
        assert counter[0] == 3

    def test_small_denominator_guard(self):
        (score, _), _ = self._run(circ_act=1.0, a_posctx=0.5, a_empty=0.5)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_invalid_ablation_mode_raises(self):
        with pytest.raises(ValueError):
            self._run(circ_act=1.0, ablation="banana")


# ---------------------------------------------------------------------------
# collect_site_means / measure_seed_activation protocol
# ---------------------------------------------------------------------------


class TestAnchors:
    def test_collect_site_anchors_returns_means_and_pins(self):
        bank = ControlledSAEBank(seed_act=4.0)
        inf, _ = _make_stub_inference(bank, (4.0,))
        sites = {(SEED_LAYER, SEED_KIND)}
        means, pins = collect_site_anchors(
            inf, bank, torch.zeros(B, T, dtype=torch.long), sites,
            argmax=torch.zeros(B, dtype=torch.long),
        )
        # Seed latent fires at 4.0 everywhere: position-mean and probe-position
        # values coincide, and both anchors must report it.
        assert means[(SEED_LAYER, SEED_KIND)][SEED_LATENT] == pytest.approx(4.0, abs=1e-5)
        assert pins[(SEED_LAYER, SEED_KIND)][SEED_LATENT] == pytest.approx(4.0, abs=1e-5)

    def test_collect_site_means_raises_on_missing_site(self):
        bank = ControlledSAEBank()
        inf, _ = _make_stub_inference(bank, (0.0,))
        with pytest.raises(RuntimeError):
            collect_site_means(inf, bank, torch.zeros(B, T, dtype=torch.long), {(99, "attn")})

    def test_measure_seed_activation_reads_configured_value(self):
        bank = ControlledSAEBank()
        inf, _ = _make_stub_inference(bank, (3.25,))
        value = measure_seed_activation(
            inf, bank, torch.zeros(B, T, dtype=torch.long),
            SEED_LAYER, SEED_KIND, SEED_LATENT,
            torch.zeros(B, dtype=torch.long),
        )
        assert value == pytest.approx(3.25, abs=1e-5)

    def test_circuit_only_activation_empty_keep(self):
        bank = ControlledSAEBank()
        inf, counter = _make_stub_inference(bank, (0.75,))
        sites = upstream_sites(bank, SEED_LAYER, SEED_KIND)
        value = circuit_only_activation(
            inf, bank, {}, sites,
            torch.zeros(B, T, dtype=torch.long),
            SEED_LAYER, SEED_KIND, SEED_LATENT,
            torch.zeros(B, dtype=torch.long),
        )
        assert value == pytest.approx(0.75, abs=1e-5)
        assert counter[0] == 1
