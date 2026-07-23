"""SAE error terms as ABLATABLE nodes (CircuitOnlyPatcher).

By default the error term is preserved at every site — the historical behaviour
every existing metric was measured under. Passing ``keep_error_sites`` turns the
error into a first-class circuit node, as in SFC: a site's error survives only
if that site is in the set, otherwise it is replaced by its mean, or zeroed when
no mean is supplied.

The load-bearing property is the DEFAULT: `keep_error_sites=None` must leave the
transform byte-identical, because every φ number in the results docs was
measured under it. A regression there would silently re-baseline the whole
project, so it is pinned first and hardest.

Why the feature exists: with the error always preserved, the EMPTY circuit
already retains most of the model's predictive signal, which collapses the
faithfulness denominator (SFC's own finding that residual error nodes are
load-bearing).
"""
import pytest
import torch
from unittest.mock import MagicMock

from eval import ablation_faithfulness as af
from eval.ablation_faithfulness import CircuitOnlyPatcher
from eval.floors import collect_site_error_means
from sae.dense import sparse_topk_to_dense
from tests.conftest import D_MODEL, D_SAE, KINDS, N_LAYERS, MockSAEBank

B, T = 2, 4
SITE = (0, "attn")
OTHER = (0, "mlp")


def _patcher(bank, *, keep=None, keep_error_sites=None, error_means=None,
             in_scope=None, site_means=None):
    return CircuitOnlyPatcher(
        bank=bank,
        keep_indices=keep or {},
        in_scope=in_scope if in_scope is not None else {SITE, OTHER},
        seed_layer=1, seed_kind="resid", seed_latent_idx=5,
        site_means=site_means,
        keep_error_sites=keep_error_sites,
        error_means=error_means,
    )


class TestDefaultIsUnchanged:
    """`keep_error_sites=None` must behave exactly as before the feature."""

    def test_keep_all_is_still_identity(self, mock_sae_bank):
        """decode(all) + (x - decode(all)) == x — holds only while the error is
        preserved, so this is the sharpest guard on the default."""
        x = torch.randn(B, T, D_MODEL)
        keep = {SITE: set(range(D_SAE))}
        out = _patcher(mock_sae_bank, keep=keep).transform(*SITE, x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_default_preserves_error_exactly(self, mock_sae_bank):
        """Empty circuit, no means: output is decode(0) + error, where the error
        is the SAE's own reconstruction residual x - decode(all_latents). Note
        this is NOT x — only keep-all recovers x."""
        x = torch.randn(B, T, D_MODEL)
        top_acts, top_idx = mock_sae_bank.encode(x, SITE[1], SITE[0])
        all_latents = sparse_topk_to_dense(top_acts, top_idx, D_SAE)
        error = x - mock_sae_bank.decode(all_latents, SITE[1], SITE[0])
        expected = mock_sae_bank.decode(torch.zeros_like(all_latents),
                                        SITE[1], SITE[0]) + error
        out = _patcher(mock_sae_bank, keep={}).transform(*SITE, x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_explicit_none_matches_omitted(self, mock_sae_bank):
        x = torch.randn(B, T, D_MODEL)
        a = _patcher(mock_sae_bank, keep={}).transform(*SITE, x)
        b = _patcher(mock_sae_bank, keep={}, keep_error_sites=None).transform(*SITE, x)
        assert torch.equal(a, b)


class TestErrorAsAblatableNode:
    def test_site_in_keep_set_retains_its_error(self, mock_sae_bank):
        """A site whose error node IS in the circuit behaves like the default."""
        x = torch.randn(B, T, D_MODEL)
        default = _patcher(mock_sae_bank, keep={}).transform(*SITE, x)
        kept = _patcher(mock_sae_bank, keep={},
                        keep_error_sites={SITE}).transform(*SITE, x)
        assert torch.allclose(default, kept, atol=1e-6)

    def test_site_outside_keep_set_loses_its_error(self, mock_sae_bank):
        """Error node absent and no mean supplied -> error zeroed, so the output
        is decode(patched) alone. With an empty circuit that is decode(0)."""
        x = torch.randn(B, T, D_MODEL)
        out = _patcher(mock_sae_bank, keep={},
                       keep_error_sites={OTHER}).transform(*SITE, x)
        expected = mock_sae_bank.decode(torch.zeros(B, T, D_SAE), SITE[1], SITE[0])
        assert torch.allclose(out, expected, atol=1e-6)
        assert not torch.allclose(out, x, atol=1e-3)      # the error really went

    def test_error_mean_substitutes_when_supplied(self, mock_sae_bank):
        """With error_means the dropped error is replaced by its mean, not zero —
        the mean-ablation analogue for error nodes."""
        x = torch.randn(B, T, D_MODEL)
        mean = torch.full((D_MODEL,), 0.25)
        out = _patcher(mock_sae_bank, keep={}, keep_error_sites={OTHER},
                       error_means={SITE: mean}).transform(*SITE, x)
        expected = mock_sae_bank.decode(torch.zeros(B, T, D_SAE), SITE[1], SITE[0]) + mean
        assert torch.allclose(out, expected, atol=1e-6)

    def test_mean_only_applies_to_dropped_sites(self, mock_sae_bank):
        """A site in keep_error_sites keeps its REAL error even if a mean exists,
        so it must match the default exactly and show no trace of the mean."""
        x = torch.randn(B, T, D_MODEL)
        mean = torch.full((D_MODEL,), 99.0)               # absurd, to be visible
        default = _patcher(mock_sae_bank, keep={}).transform(*SITE, x)
        kept = _patcher(mock_sae_bank, keep={}, keep_error_sites={SITE},
                        error_means={SITE: mean}).transform(*SITE, x)
        dropped = _patcher(mock_sae_bank, keep={}, keep_error_sites=set(),
                           error_means={SITE: mean}).transform(*SITE, x)
        assert torch.allclose(kept, default, atol=1e-6)   # real error retained
        assert not torch.allclose(kept, dropped, atol=1e-3)  # mean never applied

    def test_missing_mean_for_that_site_falls_back_to_zero(self, mock_sae_bank):
        """error_means given but not covering this site -> zeroed, not crashed."""
        x = torch.randn(B, T, D_MODEL)
        out = _patcher(mock_sae_bank, keep={}, keep_error_sites=set(),
                       error_means={OTHER: torch.zeros(D_MODEL)}).transform(*SITE, x)
        expected = mock_sae_bank.decode(torch.zeros(B, T, D_SAE), SITE[1], SITE[0])
        assert torch.allclose(out, expected, atol=1e-6)


class TestScopeInteraction:
    def test_out_of_scope_site_untouched_regardless(self, mock_sae_bank):
        """Error ablation must never reach outside the evaluated scope."""
        x = torch.randn(B, T, D_MODEL)
        out = _patcher(mock_sae_bank, keep={}, in_scope={OTHER},
                       keep_error_sites=set()).transform(*SITE, x)
        assert torch.equal(out, x)

    def test_empty_keep_set_drops_every_in_scope_error(self, mock_sae_bank):
        """`keep_error_sites=set()` is meaningfully different from None: it means
        'no error node is in the circuit', not 'feature disabled'."""
        x = torch.randn(B, T, D_MODEL)
        off = _patcher(mock_sae_bank, keep={}, keep_error_sites=None).transform(*SITE, x)
        none_kept = _patcher(mock_sae_bank, keep={},
                             keep_error_sites=set()).transform(*SITE, x)
        assert not torch.allclose(off, none_kept, atol=1e-3)


class TestCollectSiteErrorMeans:
    """The error-space floor collector (eval/floors.py)."""

    def _stub_inference(self, acts):
        inf = MagicMock()

        def forward_fn(tokens, activations_callback=None, **kw):
            for layer in range(N_LAYERS):
                activations_callback(layer, tuple(acts.clone() for _ in KINDS))

        inf.forward.side_effect = forward_fn
        return inf

    def test_matches_manual_mean_error(self, mock_sae_bank):
        torch.manual_seed(0)
        acts = torch.randn(B, T, D_MODEL)
        inf = self._stub_inference(acts)
        means = collect_site_error_means(
            inf, mock_sae_bank, torch.zeros(B, T, dtype=torch.long), {SITE})

        ta, ti = mock_sae_bank.encode(acts, SITE[1], SITE[0])
        dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=torch.float32)
        expected = (acts - mock_sae_bank.decode(dense, SITE[1], SITE[0])).mean(dim=(0, 1))
        assert means[SITE].shape == (D_MODEL,)
        assert torch.allclose(means[SITE], expected, atol=1e-5)

    def test_only_requested_sites_returned(self, mock_sae_bank):
        inf = self._stub_inference(torch.randn(B, T, D_MODEL))
        means = collect_site_error_means(
            inf, mock_sae_bank, torch.zeros(B, T, dtype=torch.long), {SITE})
        assert set(means) == {SITE}

    def test_unreachable_site_raises(self, mock_sae_bank):
        inf = self._stub_inference(torch.randn(B, T, D_MODEL))
        with pytest.raises(RuntimeError, match="error means missing"):
            collect_site_error_means(
                inf, mock_sae_bank, torch.zeros(B, T, dtype=torch.long),
                {(99, "attn")})


class TestCircuitOnlyActivationThreading:
    """circuit_only_activation must hand the error-node kwargs to the patcher
    unchanged — and default to None (the historical behaviour)."""

    def _run(self, mock_sae_bank, monkeypatch, **kwargs):
        captured = {}

        class FakePatcher:
            def __init__(self, **kw):
                captured.update(kw)
                self.captured_activation = 1.0

        monkeypatch.setattr(af, "CircuitOnlyPatcher", FakePatcher)
        af.circuit_only_activation(
            MagicMock(), mock_sae_bank, {}, {SITE},
            torch.zeros(B, T, dtype=torch.long), 1, "resid", 5, **kwargs)
        return captured

    def test_error_kwargs_reach_patcher(self, mock_sae_bank, monkeypatch):
        emeans = {SITE: torch.zeros(D_MODEL)}
        captured = self._run(mock_sae_bank, monkeypatch,
                             keep_error_sites={SITE}, error_means=emeans)
        assert captured["keep_error_sites"] == {SITE}
        assert captured["error_means"] is emeans

    def test_default_is_none(self, mock_sae_bank, monkeypatch):
        captured = self._run(mock_sae_bank, monkeypatch)
        assert captured["keep_error_sites"] is None
        assert captured["error_means"] is None
