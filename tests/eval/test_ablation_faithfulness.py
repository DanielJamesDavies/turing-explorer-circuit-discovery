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


def _make_patcher(bank, keep=None, in_scope=None, site_means=None, pin_values=None,
                  respect_topk=False, topk=128):
    return CircuitOnlyPatcher(
        bank=bank,
        keep_indices=keep or {},
        in_scope=in_scope if in_scope is not None else {(0, "attn")},
        seed_layer=SEED_LAYER,
        seed_kind=SEED_KIND,
        seed_latent_idx=SEED_LATENT,
        site_means=site_means,
        pin_values=pin_values,
        respect_topk=respect_topk,
        topk=topk,
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

    def test_respect_topk_fill_empty_keep_is_k_sparse(self, mock_sae_bank):
        """respect_topk with nothing kept: the patched latent vector has at
        most `topk` nonzero entries per position, all from the top-mean
        latents at their mean value."""
        K = 4
        patcher = _make_patcher(mock_sae_bank, keep={}, respect_topk=True, topk=K)
        all_latents = torch.zeros(B, T, D_SAE)  # empty keep -> kept_values None
        mean_vector = torch.zeros(D_SAE)
        mean_vector[[3, 7, 11, 19, 23, 29]] = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        patched = patcher._respect_topk_fill(all_latents, mean_vector, None, None)
        # at most K nonzero per (batch, position)
        assert int((patched != 0).sum(-1).max().item()) == K
        # the K active latents are the top-K by mean (3,7,11,19), at their means
        active = (patched[0, 0] != 0).nonzero().squeeze(1).tolist()
        assert sorted(active) == [3, 7, 11, 19]
        assert torch.isclose(patched[0, 0, 3], torch.tensor(0.9))

    def test_respect_topk_keep_all_is_identity(self, mock_sae_bank):
        """respect_topk keep-all: kept fills the whole budget, no mean fill,
        patched == natural latents (so the stream reconstructs to x)."""
        keep = {(0, "attn"): set(range(D_SAE))}
        patcher = _make_patcher(
            mock_sae_bank, keep=keep, in_scope={(0, "attn")},
            site_means={(0, "attn"): torch.rand(D_SAE)}, respect_topk=True, topk=4,
        )
        x = torch.randn(B, T, D_MODEL)
        out = patcher.transform(0, "attn", x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_respect_topk_budget_leaves_room_for_kept(self, mock_sae_bank):
        """Active kept latents consume the budget: total active == topk when
        fewer than topk kept latents fire."""
        K = 4
        patcher = _make_patcher(mock_sae_bank, keep={}, respect_topk=True, topk=K)
        all_latents = torch.zeros(B, T, D_SAE)
        all_latents[:, :, 5] = 2.0  # one kept latent fires everywhere
        mean_vector = torch.zeros(D_SAE)
        mean_vector[[1, 2, 3, 8, 9]] = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        keep_tensor = torch.tensor([5])
        kept_values = all_latents[:, :, keep_tensor]
        patched = patcher._respect_topk_fill(all_latents, mean_vector, keep_tensor, kept_values)
        assert int((patched != 0).sum(-1).max().item()) == K  # 1 kept + 3 fill
        assert torch.isclose(patched[0, 0, 5], torch.tensor(2.0))  # kept preserved
        assert patched[0, 0, 5] != 0

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

    def test_pinned_position_specific_pins_per_position(self, mock_sae_bank):
        """A 3-D pin tensor [B, T, d_sae] pins kept latents to their own
        per-position value (not a single broadcast vector); non-kept keep the
        mean floor."""
        x = torch.randn(B, T, D_MODEL)
        dense = _natural_dense(mock_sae_bank, x, "attn", 0)
        kept = {3, 11}
        mean_vector = torch.rand(D_SAE) * 0.5
        pin_pos = torch.rand(B, T, D_SAE) * 3.0  # distinct value per (b, t, latent)
        patcher = _make_patcher(
            mock_sae_bank,
            keep={(0, "attn"): kept},
            in_scope={(0, "attn")},
            site_means={(0, "attn"): mean_vector},
            pin_values={(0, "attn"): pin_pos},
        )
        expected_latents = mean_vector.expand_as(dense).clone()
        keep_tensor = torch.tensor(sorted(kept))
        expected_latents[:, :, keep_tensor] = pin_pos[:, :, keep_tensor]
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


class TestFloorSource:
    def _loader(self, tokens):
        loader = MagicMock()
        loader.get_batches.return_value = iter([(None, tokens)])
        return loader

    def test_posctx_source_returns_means_unchanged(self, monkeypatch):
        from config import config as cfg
        from eval.ablation_faithfulness import resolve_site_floors

        monkeypatch.setattr(cfg.discovery, "floor_source", "posctx")
        means = {(0, "attn"): torch.ones(D_SAE)}
        out = resolve_site_floors(MagicMock(), ControlledSAEBank(), {(0, "attn")}, posctx_means=means)
        assert out is means

    # ---- floor_source="negctx" ------------------------------------------
    # The floor built from the seed's NEGATIVE contexts: sequences retrieved
    # because the seed is silent on them. Unlike posctx (whose mean carries the
    # seed's own firing signature by construction) it strips seed-specific
    # content while keeping generic stream content.

    def _negctx_setup(self, monkeypatch, seed_act=4.0):
        from config import config as cfg

        monkeypatch.setattr(cfg.discovery, "floor_source", "negctx")
        bank = ControlledSAEBank(seed_act=seed_act)
        bank.device = torch.device("cpu")
        inf, _ = _make_stub_inference(bank, (seed_act,))
        return bank, inf

    def test_negctx_source_reads_neg_tokens_not_posctx(self, monkeypatch):
        """The floor must come from a forward on neg_tokens. Guards the whole
        point of the source: reading pos_tokens here would silently reproduce
        the posctx floor under a negctx label."""
        import eval.ablation_faithfulness as module

        bank, inf = self._negctx_setup(monkeypatch)
        sites = {(SEED_LAYER, SEED_KIND)}
        # Distinctive fill so the assertion is on identity, not coincidence.
        neg_tokens = torch.full((B, T), 7, dtype=torch.long)
        posctx_means = {(SEED_LAYER, SEED_KIND): torch.zeros(D_SAE)}

        out = module.resolve_site_floors(
            inf, bank, sites, posctx_means=posctx_means, neg_tokens=neg_tokens,
        )

        assert inf.forward.call_count == 1
        assert torch.equal(inf.forward.call_args[0][0], neg_tokens)
        assert out is not posctx_means
        assert out[(SEED_LAYER, SEED_KIND)][SEED_LATENT] == pytest.approx(4.0, abs=1e-5)

    def test_negctx_source_needs_no_loader(self, monkeypatch):
        """Seed-specific, so unlike global/diverse it never touches the corpus."""
        import eval.ablation_faithfulness as module

        bank, inf = self._negctx_setup(monkeypatch)
        out = module.resolve_site_floors(
            inf, bank, {(SEED_LAYER, SEED_KIND)},
            posctx_means={(SEED_LAYER, SEED_KIND): torch.zeros(D_SAE)},
            loader=None, neg_tokens=torch.zeros(B, T, dtype=torch.long),
        )
        assert (SEED_LAYER, SEED_KIND) in out

    def test_negctx_source_is_not_cached_across_seeds(self, monkeypatch):
        """global/diverse cache per process because they are seed-independent;
        negctx must NOT, or seed 2 would inherit seed 1's negatives."""
        import eval.ablation_faithfulness as module

        from config import config as cfg
        monkeypatch.setattr(cfg.discovery, "floor_source", "negctx")
        bank = ControlledSAEBank(seed_act=1.0)
        bank.device = torch.device("cpu")
        inf, _ = _make_stub_inference(bank, (1.0, 9.0))
        sites = {(SEED_LAYER, SEED_KIND)}
        pm = {(SEED_LAYER, SEED_KIND): torch.zeros(D_SAE)}

        first = module.resolve_site_floors(
            inf, bank, sites, posctx_means=pm,
            neg_tokens=torch.zeros(B, T, dtype=torch.long))
        second = module.resolve_site_floors(
            inf, bank, sites, posctx_means=pm,
            neg_tokens=torch.ones(B, T, dtype=torch.long))

        assert inf.forward.call_count == 2          # recomputed, not cached
        assert first[(SEED_LAYER, SEED_KIND)][SEED_LATENT] == pytest.approx(1.0, abs=1e-5)
        assert second[(SEED_LAYER, SEED_KIND)][SEED_LATENT] == pytest.approx(9.0, abs=1e-5)

    @pytest.mark.parametrize("bad", [None, torch.zeros(0, 8, dtype=torch.long)])
    def test_negctx_source_raises_rather_than_falling_back(self, monkeypatch, bad):
        """A seed with no negatives must fail loudly. A silent fallback would
        label another floor's numbers 'negctx' and poison the comparison."""
        import eval.ablation_faithfulness as module

        bank, inf = self._negctx_setup(monkeypatch)
        with pytest.raises(ValueError, match="negctx"):
            module.resolve_site_floors(
                inf, bank, {(SEED_LAYER, SEED_KIND)},
                posctx_means={(SEED_LAYER, SEED_KIND): torch.zeros(D_SAE)},
                neg_tokens=bad,
            )
        assert inf.forward.call_count == 0

    @pytest.mark.parametrize("source", ["posctx", "zero"])
    def test_neg_tokens_ignored_under_other_sources(self, monkeypatch, source):
        """Inertness: threading neg_tokens everywhere must not perturb any
        existing floor. This is what keeps the default path byte-identical."""
        from config import config as cfg
        import eval.ablation_faithfulness as module

        monkeypatch.setattr(cfg.discovery, "floor_source", source)
        inf = MagicMock()
        means = {(0, "attn"): torch.ones(D_SAE)}
        out = module.resolve_site_floors(
            inf, ControlledSAEBank(), {(0, "attn")}, posctx_means=means,
            neg_tokens=torch.full((B, T), 7, dtype=torch.long),
        )
        if source == "posctx":
            assert out is means
        else:
            assert torch.all(out[(0, "attn")] == 0.0)
        assert inf.forward.call_count == 0

    def test_config_validator_accepts_negctx_and_rejects_unknown(self):
        from config import DiscoveryConfig

        assert DiscoveryConfig(floor_source="negctx").floor_source == "negctx"
        for known in ("posctx", "zero", "global", "diverse"):
            assert DiscoveryConfig(floor_source=known).floor_source == known
        with pytest.raises(ValueError, match="floor_source"):
            DiscoveryConfig(floor_source="negctxx")

    def test_floor_negctx_mode_validator(self):
        """Which negatives define the negctx floor. Defaults to the neg_ctx KNN
        store (close/hard); the other modes re-retrieve so the floor's negative
        hardness can be swept independently of a method's own neg_mode."""
        from config import DiscoveryConfig

        assert DiscoveryConfig().floor_negctx_mode == "store"
        for known in ("store", "close", "random", "distant"):
            assert DiscoveryConfig(floor_negctx_mode=known).floor_negctx_mode == known
        with pytest.raises(ValueError, match="floor_negctx_mode"):
            DiscoveryConfig(floor_negctx_mode="closest")

    def test_global_source_uses_corpus_sample_and_caches(self, monkeypatch):
        from config import config as cfg
        import eval.ablation_faithfulness as module
        import eval.floors as floors_module

        monkeypatch.setattr(cfg.discovery, "floor_source", "global")
        monkeypatch.setattr(floors_module, "_GLOBAL_FLOOR_CACHE", {})
        bank = ControlledSAEBank(seed_act=3.0)
        bank.device = torch.device("cpu")
        inf, _ = _make_stub_inference(bank, (3.0, 99.0))
        loader = self._loader(torch.zeros(B, T, dtype=torch.long))

        sites = {(SEED_LAYER, SEED_KIND)}
        floors = module.resolve_site_floors(
            inf, bank, sites, posctx_means={}, loader=loader,
        )
        assert floors[(SEED_LAYER, SEED_KIND)][SEED_LATENT] == pytest.approx(3.0, abs=1e-5)
        # Second call hits the cache: no new batch, no new forward.
        floors2 = module.resolve_site_floors(inf, bank, sites, posctx_means={}, loader=loader)
        assert loader.get_batches.call_count == 1
        assert inf.forward.call_count == 1
        assert torch.allclose(floors2[(SEED_LAYER, SEED_KIND)], floors[(SEED_LAYER, SEED_KIND)])

    def test_farthest_point_sample_picks_spread(self):
        from eval.ablation_faithfulness import _farthest_point_sample

        # Three clusters on the unit circle; FPS from index 0 must visit the
        # two other clusters before returning to cluster A.
        reprs = torch.tensor([
            [1.0, 0.0], [0.999, 0.01],            # cluster A
            [-1.0, 0.0], [-0.999, 0.01],          # cluster B (opposite)
            [0.0, 1.0], [0.01, 0.999],            # cluster C (orthogonal)
        ])
        chosen = _farthest_point_sample(reprs, 3)
        assert chosen[0] == 0
        assert chosen[1] in (2, 3)
        assert chosen[2] in (4, 5)
        assert len(_farthest_point_sample(reprs, 99)) == 6  # capped at pool

    def test_diverse_source_samples_pool_and_caches(self, monkeypatch):
        from config import config as cfg
        import eval.ablation_faithfulness as module
        import eval.floors as floors_module
        import store.seq_repr as seq_repr_module

        monkeypatch.setattr(cfg.discovery, "floor_source", "diverse")
        monkeypatch.setattr(floors_module, "_GLOBAL_FLOOR_CACHE", {})

        repr_store = MagicMock()
        repr_store.get_repr.side_effect = lambda ids: torch.randn(len(ids), 8)
        monkeypatch.setattr(seq_repr_module, "seq_repr", repr_store)

        bank = ControlledSAEBank(seed_act=2.5)
        bank.device = torch.device("cpu")
        inf, _ = _make_stub_inference(bank, (2.5, 99.0))
        tokens = torch.zeros(B, T, dtype=torch.long)
        loader = MagicMock()
        loader.get_batches.side_effect = lambda **k: iter([(torch.arange(B), tokens)])
        loader.get_batches_by_ids.side_effect = lambda ids, **k: iter([(torch.tensor(ids), tokens)])

        sites = {(SEED_LAYER, SEED_KIND)}
        floors = module.resolve_site_floors(inf, bank, sites, posctx_means={}, loader=loader)
        assert floors[(SEED_LAYER, SEED_KIND)][SEED_LATENT] == pytest.approx(2.5, abs=1e-5)
        # Cached under its own source key, separate from "global".
        assert "diverse" in floors_module._GLOBAL_FLOOR_CACHE
        module.resolve_site_floors(inf, bank, sites, posctx_means={}, loader=loader)
        assert inf.forward.call_count == 1

    def test_global_source_without_loader_raises(self, monkeypatch):
        from config import config as cfg
        from eval.ablation_faithfulness import resolve_site_floors

        monkeypatch.setattr(cfg.discovery, "floor_source", "global")
        with pytest.raises(ValueError):
            resolve_site_floors(MagicMock(), ControlledSAEBank(), {(0, "attn")}, posctx_means={})


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


class TestEvalBatchChunking:
    """Sequence count vs batch size: pos_tokens may carry more sequences than
    one forward pass; chunks merge with B_chunk/B_total weights, equal to the
    single-pass per-sequence mean."""

    def test_circuit_only_activation_chunked_weighted_mean(self):
        """4 seqs in chunks of 2 -> two passes whose per-pass means (1.0, 3.0)
        combine to the sequence-weighted mean 2.0."""
        bank = ControlledSAEBank()
        inf, counter = _make_stub_inference(bank, (1.0, 3.0))
        sites = upstream_sites(bank, SEED_LAYER, SEED_KIND)
        value = circuit_only_activation(
            inf, bank, {}, sites,
            torch.zeros(4, T, dtype=torch.long),
            SEED_LAYER, SEED_KIND, SEED_LATENT,
            torch.zeros(4, dtype=torch.long),
            batch_size=2,
        )
        assert counter[0] == 2
        assert value == pytest.approx(2.0, abs=1e-5)

    def test_circuit_only_activation_uneven_chunks_weighting(self):
        """3 seqs in chunks of 2 -> weights 2/3 and 1/3, not 1/2 each."""
        bank = ControlledSAEBank()
        inf, counter = _make_stub_inference(bank, (1.0, 4.0))
        sites = upstream_sites(bank, SEED_LAYER, SEED_KIND)
        value = circuit_only_activation(
            inf, bank, {}, sites,
            torch.zeros(3, T, dtype=torch.long),
            SEED_LAYER, SEED_KIND, SEED_LATENT,
            torch.zeros(3, dtype=torch.long),
            batch_size=2,
        )
        assert counter[0] == 2
        assert value == pytest.approx((2 * 1.0 + 1 * 4.0) / 3, abs=1e-5)

    def test_circuit_only_activation_default_single_pass(self):
        """batch_size=None must stay one forward pass (historical behaviour)."""
        bank = ControlledSAEBank()
        inf, counter = _make_stub_inference(bank, (0.5,))
        sites = upstream_sites(bank, SEED_LAYER, SEED_KIND)
        circuit_only_activation(
            inf, bank, {}, sites,
            torch.zeros(4, T, dtype=torch.long),
            SEED_LAYER, SEED_KIND, SEED_LATENT,
            torch.zeros(4, dtype=torch.long),
        )
        assert counter[0] == 1

    def test_measure_seed_activation_chunked_weighted_mean(self):
        bank = ControlledSAEBank()
        inf, counter = _make_stub_inference(bank, (1.0, 3.0))
        value = measure_seed_activation(
            inf, bank, torch.zeros(4, T, dtype=torch.long),
            SEED_LAYER, SEED_KIND, SEED_LATENT,
            torch.zeros(4, dtype=torch.long),
            batch_size=2,
        )
        assert counter[0] == 2
        assert value == pytest.approx(2.0, abs=1e-5)


class TestKeepScale:
    """keep_scale: the amplitude intervention (redundancy probe). Default 1.0
    must be bit-identical to the historical transform."""

    def _x(self):
        torch.manual_seed(3)
        return torch.randn(B, T, D_MODEL)

    def _patcher(self, bank, scale, keep=None, pin_values=None):
        return CircuitOnlyPatcher(
            bank=bank, keep_indices=keep or {(0, "attn"): {1, 3}},
            in_scope={(0, "attn")}, seed_layer=1, seed_kind="resid",
            seed_latent_idx=5, pin_values=pin_values, keep_scale=scale)

    def test_default_scale_is_identity(self, mock_sae_bank):
        x = self._x()
        a = self._patcher(mock_sae_bank, 1.0).transform(0, "attn", x)
        b = CircuitOnlyPatcher(bank=mock_sae_bank,
                               keep_indices={(0, "attn"): {1, 3}},
                               in_scope={(0, "attn")}, seed_layer=1,
                               seed_kind="resid",
                               seed_latent_idx=5).transform(0, "attn", x)
        assert torch.equal(a, b)

    def test_scale_doubles_kept_contribution(self, mock_sae_bank):
        """decode is linear in the code, so scaling kept values by 2 must move
        the output by exactly the kept latents' decode contribution."""
        x = self._x()
        out1 = self._patcher(mock_sae_bank, 1.0).transform(0, "attn", x)
        out2 = self._patcher(mock_sae_bank, 2.0).transform(0, "attn", x)
        ta, ti = mock_sae_bank.encode(x, "attn", 0)
        dense = sparse_topk_to_dense(ta, ti, D_SAE)
        kept_only = torch.zeros_like(dense)
        kept_only[:, :, [1, 3]] = dense[:, :, [1, 3]]
        contrib = (mock_sae_bank.decode(kept_only, "attn", 0)
                   - mock_sae_bank.decode(torch.zeros_like(dense), "attn", 0))
        assert torch.allclose(out2 - out1, contrib, atol=1e-5)

    def test_scale_applies_to_pins_too(self, mock_sae_bank):
        x = self._x()
        pins = {(0, "attn"): torch.full((D_SAE,), 2.0)}
        out1 = self._patcher(mock_sae_bank, 1.0, pin_values=pins).transform(0, "attn", x)
        out3 = self._patcher(mock_sae_bank, 3.0, pin_values=pins).transform(0, "attn", x)
        assert not torch.allclose(out1, out3, atol=1e-4)

    def test_threading_reaches_patcher(self, mock_sae_bank, monkeypatch):
        from eval import ablation_faithfulness as af
        captured = {}

        class FakePatcher:
            def __init__(self, **kw):
                captured.update(kw)
                self.captured_activation = 1.0

        monkeypatch.setattr(af, "CircuitOnlyPatcher", FakePatcher)
        af.circuit_only_activation(
            MagicMock(), mock_sae_bank, {}, {(0, "attn")},
            torch.zeros(B, T, dtype=torch.long), 1, "resid", 5, keep_scale=4.0)
        assert captured["keep_scale"] == 4.0


class TestPreactCapture:
    """capture_preact / preact=True: the uncensored seed measurement.

    The default post-top-k read is floored at 0 (target_latent_activations
    returns 0 when the seed misses the top-k), so it cannot distinguish "no
    drive" from "below threshold" nor show the sign of changes underneath.
    """

    def _bank_with_seed_sae(self, w_row, bias=0.0):
        """Attach the real SAE's encoder API (encoder.weight / _get_bias_eff)
        to the mock module — replacing the module would break bank.encode()."""
        from types import SimpleNamespace
        bank = MockSAEBank()
        mod = bank.saes[SEED_KIND][SEED_LAYER]
        W = torch.zeros(D_SAE, D_MODEL)
        W[SEED_LATENT] = w_row
        mod.encoder = SimpleNamespace(weight=W)
        mod._get_bias_eff = lambda: torch.full((D_SAE,), float(bias))
        return bank

    def test_preact_matches_manual_dot_product(self):
        bank = self._bank_with_seed_sae(torch.eye(D_MODEL)[0], bias=0.5)
        p = CircuitOnlyPatcher(
            bank=bank, keep_indices={}, in_scope=set(),
            seed_layer=SEED_LAYER, seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT, capture_preact=True)
        x = torch.zeros(B, T, D_MODEL)
        x[:, :, 0] = 3.0
        p.transform(SEED_LAYER, SEED_KIND, x)
        assert p.captured_preactivation == pytest.approx(3.5, abs=1e-5)

    def test_preact_can_be_negative_where_activation_floors_at_zero(self):
        """The whole point: post-top-k reads 0, pre-act reads the real value."""
        bank = self._bank_with_seed_sae(torch.eye(D_MODEL)[0], bias=0.0)
        p = CircuitOnlyPatcher(
            bank=bank, keep_indices={}, in_scope=set(),
            seed_layer=SEED_LAYER, seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT, capture_preact=True)
        x = torch.zeros(B, T, D_MODEL)
        x[:, :, 0] = -2.0
        p.transform(SEED_LAYER, SEED_KIND, x)
        assert p.captured_preactivation == pytest.approx(-2.0, abs=1e-5)
        assert p.captured_activation == pytest.approx(0.0, abs=1e-6)

    def test_preact_respects_pos_argmax(self):
        bank = self._bank_with_seed_sae(torch.eye(D_MODEL)[0])
        pa = torch.full((B,), 2, dtype=torch.long)
        p = CircuitOnlyPatcher(
            bank=bank, keep_indices={}, in_scope=set(),
            seed_layer=SEED_LAYER, seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT, pos_argmax=pa, capture_preact=True)
        x = torch.zeros(B, T, D_MODEL)
        x[:, 2, 0] = 7.0          # only the probe position carries signal
        p.transform(SEED_LAYER, SEED_KIND, x)
        assert p.captured_preactivation == pytest.approx(7.0, abs=1e-5)

    def test_default_off_leaves_preact_none(self, mock_sae_bank):
        p = CircuitOnlyPatcher(
            bank=mock_sae_bank, keep_indices={}, in_scope=set(),
            seed_layer=SEED_LAYER, seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT)
        p.transform(SEED_LAYER, SEED_KIND, torch.randn(B, T, D_MODEL))
        assert p.captured_preactivation is None
        assert p.captured_activation is not None

    def test_circuit_only_activation_returns_preact(self, mock_sae_bank, monkeypatch):
        from eval import ablation_faithfulness as af
        captured = {}

        class FakePatcher:
            def __init__(self, **kw):
                captured.update(kw)
                self.captured_activation = 1.0
                self.captured_preactivation = -4.0

        monkeypatch.setattr(af, "CircuitOnlyPatcher", FakePatcher)
        out = af.circuit_only_activation(
            MagicMock(), mock_sae_bank, {}, {(0, "attn")},
            torch.zeros(B, T, dtype=torch.long), 1, "resid", 5, preact=True)
        assert captured["capture_preact"] is True
        assert out == pytest.approx(-4.0)
