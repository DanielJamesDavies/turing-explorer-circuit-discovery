"""Position-aware selection under chunking: every chunk's causal-prefix cut must
use THAT chunk's own anchors (bug found 2026-10-23 by the exp 083 build).

The bug: ``integrated_baseline_scores`` (the ig_mean / ig_negctx attribution of
the ablation- and counterfactual-gradient methods, and so of hybrid) received a
``PositionAwareSpec`` whose ``peaks`` were the FULL anchor list and selected each
chunk with it; ``select_position_aware`` indexes peaks from 0, so every chunk was
cut at chunk 1's anchors. Every earlier chunking test used uniform anchors
(all 0 or all T-1), which cannot see this.

Harness: a position-wise linear pipeline (position t only affects the seed at
position t), so a sequence's attribution sits entirely on its anchor row. A
chunk cut at another chunk's anchors therefore loses its members, and IG on the
"drive" objective is exact, so the union has an analytic reference.

Part 1  ig_baseline (the buggy path): chunked == analytic == single pass, both
        IG paths and both objectives; single-chunk output unchanged.
Part 2  the other position-aware chunk loops (already per-chunk; pinned here):
        position_aware_membership (activation_gradient), restoration
        _round_scores, the abl local hop and the cf local contrast hop.
"""

import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from circuit.instrument.ig_baseline import integrated_baseline_scores
from circuit.instrument.position_aware import (
    PositionAwareSpec,
    position_aware_membership,
    select_position_aware,
)
from sae.dense import sparse_topk_to_dense

from tests.conftest import D_MODEL, D_SAE, KINDS, N_LAYERS, MockSAEBank

SEED_LAYER, SEED_KIND, SITE = 1, "resid", (0, "attn")
N, T = 6, 5
# In chunks of 2: chunk 1 anchors at 0, chunk 2 at the last position, chunk 3
# mixed. In chunks of 4: [0, 0, 4, 4] then [1, 3].
ANCHORS = torch.tensor([0, 0, 4, 4, 1, 3])
TOP_N = 3


def _stub(x0, w_map):
    """Row 0 of each token sequence indexes x0 (chunked calls see exactly their
    own sequences); the site output maps position-wise and linearly to the seed."""

    def forward_fn(tokens, activations_callback=None, patcher=None, **kwargs):
        x = x0[tokens[:, 0]]
        if activations_callback is not None:
            acts = tuple(x.clone() for _ in KINDS)
            for layer in range(N_LAYERS):
                activations_callback(layer, acts)
            return
        out0 = patcher.transform(*SITE, x.clone())
        patcher.transform(SEED_LAYER, SEED_KIND, out0 @ w_map)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    return inf


def _tokens(n=N):
    tokens = torch.zeros(n, T, dtype=torch.long)
    tokens[:, 0] = torch.arange(n)
    return tokens


def _setup(seed=11):
    torch.manual_seed(seed)
    return dict(
        bank=MockSAEBank(),
        x0=torch.randn(N, T, D_MODEL),
        w_map=torch.randn(D_MODEL, D_MODEL) * 0.2,
        w_seed=torch.randn(D_MODEL),
        b_seed=torch.tensor(0.1),
        endpoint=torch.rand(D_SAE) * 0.3,
    )


def _spec(select="top_n", threshold=0.0, position_weight=False, peaks=ANCHORS):
    # Exactly what GradientDiscoveryBase._position_aware_spec(pos_argmax) builds:
    # the spec carries the FULL anchor list, the call chunks at batch_size.
    return PositionAwareSpec(peaks=peaks, top_n=TOP_N, select=select,
                             threshold=threshold, position_weight=position_weight)


def _ig(s, *, batch_size, path="to_natural", objective="drive", spec=None):
    scores, _, _ = integrated_baseline_scores(
        _stub(s["x0"], s["w_map"]), s["bank"],
        tokens=_tokens(), substitute_sites={SITE},
        site_floors={SITE: s["endpoint"]},
        seed_layer=SEED_LAYER, seed_kind=SEED_KIND,
        w_seed=s["w_seed"], b_seed=s["b_seed"],
        pos_argmax=ANCHORS, objective=objective, target_act=0.7, ig_steps=4,
        position_aware=spec if spec is not None else _spec(),
        path=path, batch_size=batch_size,
    )
    return scores[SITE]


def _analytic_union(s, path, anchors=ANCHORS, n=N):
    """The exact position-aware union for the "drive" objective on the stub:
    attr[b, t] = [t == anchor_b] * g * delta[b, t] / N (the global-mean scale),
    g = d pre / d f = W_dec^T (w_map w_seed), delta = end - start of the path."""
    bank = s["bank"]
    ta, ti = bank.encode(s["x0"][:n], "attn", 0)
    natural = sparse_topk_to_dense(ta, ti, D_SAE, dtype=torch.float32)
    delta = natural - s["endpoint"] if path == "to_natural" else s["endpoint"] - natural
    g = bank.saes["attn"][0].W_dec.T @ (s["w_map"] @ s["w_seed"])
    attr = torch.zeros(n, T, D_SAE)
    for b in range(n):
        t = int(anchors[b])
        attr[b, t] = g * delta[b, t] / N
    out = torch.zeros(D_SAE, dtype=torch.float64)
    for lat, val in select_position_aware(attr, anchors[:n], top_n=TOP_N).items():
        out[lat] = val
    return out


# =============================================================================
# Part 1: integrated_baseline_scores (ig_mean / ig_negctx)
# =============================================================================

class TestIgBaselineChunkAnchors:
    @pytest.mark.parametrize("batch_size", [2, 4])
    @pytest.mark.parametrize("path", ["to_natural", "from_natural"])
    def test_union_uses_each_chunks_own_anchors(self, path, batch_size):
        """The failing case: with chunk 1's anchors applied to every chunk,
        chunks 2+ are cut at the wrong prefix and their members vanish."""
        s = _setup()
        ref = _analytic_union(s, path)
        first_chunk_only = _analytic_union(s, path, n=2)
        # Meaningful: later chunks contribute members chunk 1 does not have.
        assert ((ref != 0) & (first_chunk_only == 0)).any()
        got = _ig(s, batch_size=batch_size, path=path)
        assert torch.equal(got != 0, ref != 0)
        assert torch.allclose(got, ref, rtol=1e-4, atol=1e-7)

    @pytest.mark.parametrize("objective", ["drive", "gap"])
    @pytest.mark.parametrize("select,threshold,position_weight", [
        ("top_n", 0.0, False),
        ("relative", 0.5, False),
        ("mass", 0.5, False),
        ("top_n", 0.0, True),
    ])
    def test_chunked_matches_single_pass(self, objective, select, threshold, position_weight):
        """Per-sequence attributions are batch-independent, so the chunked union
        must equal the single-pass union for any per-position rule (abs_pctl is
        excluded by design: its cut is resolved per chunk)."""
        s = _setup()
        spec = _spec(select, threshold, position_weight)
        single = _ig(s, batch_size=None, objective=objective, spec=spec)
        chunked = _ig(s, batch_size=2, objective=objective, spec=spec)
        assert single.abs().sum() > 0
        assert torch.equal(chunked != 0, single != 0)
        assert torch.allclose(chunked, single, rtol=1e-4, atol=1e-7)

    @pytest.mark.parametrize("path", ["to_natural", "from_natural"])
    def test_single_chunk_unchanged(self, path):
        """Non-batched behaviour: batch_size None, == N and > N are one chunk,
        give bit-identical output, and match the analytic union (this test
        passes on both the pre-fix and the fixed code)."""
        s = _setup()
        none = _ig(s, batch_size=None, path=path)
        exact = _ig(s, batch_size=N, path=path)
        over = _ig(s, batch_size=N + 3, path=path)
        assert torch.equal(none, exact) and torch.equal(none, over)
        assert torch.allclose(none, _analytic_union(s, path), rtol=1e-4, atol=1e-7)

    def test_classic_path_ignores_position_aware(self):
        """position_aware=None (the default) keeps the collapsed scores; the
        chunk-anchor change cannot touch it."""
        s = _setup()
        inf = _stub(s["x0"], s["w_map"])
        kw = dict(tokens=_tokens(), substitute_sites={SITE}, site_floors={SITE: s["endpoint"]},
                  seed_layer=SEED_LAYER, seed_kind=SEED_KIND, w_seed=s["w_seed"],
                  b_seed=s["b_seed"], pos_argmax=ANCHORS, objective="drive", ig_steps=4)
        single, _, _ = integrated_baseline_scores(inf, s["bank"], **kw)
        chunked, _, _ = integrated_baseline_scores(inf, s["bank"], batch_size=2, **kw)
        assert torch.allclose(chunked[SITE], single[SITE], atol=1e-6)


# =============================================================================
# Part 2: the other position-aware chunk loops
# =============================================================================

def _graft_encoder_surface(bank):
    """position_aware_membership reads sae.encoder.weight / _get_bias_eff."""
    for kind in bank.kinds:
        for mod in bank.saes[kind]:
            mod.encoder = types.SimpleNamespace(weight=mod.W_enc)
            mod._get_bias_eff = lambda m=mod: m.b_enc
    return bank


class TestPositionAwareMembershipChunkAnchors:
    """activation_gradient's PA hop: slices pos_argmax per chunk (correct)."""

    def _members(self, batch_size):
        s = _setup()
        bank = _graft_encoder_surface(s["bank"])
        members, inhibitors = position_aware_membership(
            _stub(s["x0"], s["w_map"]), bank,
            tokens=_tokens(), sites={SITE},
            seed_layer=SEED_LAYER, seed_kind=SEED_KIND, seed_latent_idx=0,
            pos_argmax=ANCHORS, top_n=TOP_N, batch_size=batch_size,
        )
        return {**members, **inhibitors}

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_chunked_matches_single_pass(self, batch_size):
        single = self._members(None)
        chunked = self._members(batch_size)
        assert single
        assert set(chunked) == set(single)
        for fid, val in single.items():
            assert chunked[fid] == pytest.approx(val, rel=1e-4, abs=1e-6)


class TestRestorationRoundChunkAnchors:
    """Restoration's per-round PA selection: dataclass_replace(peaks=argmax_chunk)
    per chunk (correct)."""

    def _selected(self, batch_size):
        from circuit.instrument.restoration import _round_scores

        s = _setup()
        bank = s["bank"]
        x = s["x0"]
        ta, ti = bank.encode(x, "attn", 0)
        dense = sparse_topk_to_dense(ta, ti, D_SAE, dtype=torch.float32)
        residuals = {SITE: (x - bank.decode(dense, "attn", 0)).detach()}   # [N, T, d]
        _, _, selected = _round_scores(
            _stub(x, s["w_map"]), bank,
            tokens=_tokens(), substitute_sites={SITE}, residuals=residuals,
            site_floors={SITE: torch.zeros(D_SAE)},
            natural_dense={SITE: dense.mean(dim=(0, 1))},
            masks={SITE: torch.zeros(D_SAE, dtype=torch.bool)},
            seed_layer=SEED_LAYER, seed_kind=SEED_KIND,
            w_seed=s["w_seed"], b_seed=s["b_seed"],
            pos_argmax=ANCHORS, target_act=0.7, alphas=[0.0],
            batch_size=batch_size, position_select=_spec(),
        )
        return selected

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_chunked_matches_single_pass(self, batch_size):
        single = self._selected(None)
        chunked = self._selected(batch_size)
        assert single
        assert set(chunked) == set(single)
        for key, val in single.items():
            assert chunked[key] == pytest.approx(val, rel=1e-4, abs=1e-7)


class _FakeSeedInstrument:
    """Stands in for SeedProjectionInstrument: the stub forward stamps a
    per-sequence seed pre-activation whose argmax is known."""

    def __init__(self, *args, **kwargs):
        self.seed_pre_act = None
        self.graph = None

    def release(self):
        pass


def _fake_pre_act(tokens):
    # Sequence i peaks at position ANCHORS[i]; differentiable leaf.
    pre = torch.zeros(tokens.shape[0], T)
    for row, seq_id in enumerate(tokens[:, 0].tolist()):
        pre[row, int(ANCHORS[seq_id])] = 1.0 + seq_id
    return pre.requires_grad_(True)


def _local_inference():
    inf = MagicMock()
    inf._compiled = False

    def forward_fn(tokens, patcher=None, **kwargs):
        patcher.seed_pre_act = _fake_pre_act(tokens)

    inf.forward.side_effect = forward_fn
    return inf


class _DiscoveryBank:
    kinds = KINDS
    d_sae = D_SAE
    n_layer = N_LAYERS

    def __init__(self):
        mod = types.SimpleNamespace(encoder=types.SimpleNamespace(weight=torch.zeros(D_SAE, D_MODEL)),
                                    _get_bias_eff=lambda: torch.zeros(D_SAE))
        self.saes = {kind: [mod for _ in range(N_LAYERS)] for kind in KINDS}


class TestLocalHopChunkAnchors:
    """The local (single-point gradient) hops build a spec per chunk from that
    chunk's own anchors (correct). Pinned by recording the peaks each chunk's
    scorer receives."""

    def test_abl_local_hop_passes_each_chunks_anchors(self):
        from circuit.discovery import ablation_gradient as mod

        algo = mod.AblationGradientDiscovery(
            _local_inference(), _DiscoveryBank(), torch.zeros(1, D_SAE), MagicMock(),
            attribution_mode="local",
        )
        algo.position_aware = True
        algo.probe_batch_size = 2
        seen = []

        def record(**kwargs):
            seen.append(kwargs["position_aware"].peaks.clone())
            return {}

        with patch.object(mod, "SeedProjectionInstrument", _FakeSeedInstrument), \
                patch.object(mod, "compute_latent_ablation_scores", side_effect=record), \
                patch.object(mod.AblationGradientDiscovery, "_active_count", return_value=None):
            algo._run_ablation_hop(SEED_LAYER * len(KINDS) + KINDS.index(SEED_KIND), 0,
                                   _tokens(), ANCHORS, MagicMock())
        assert [p.tolist() for p in seen] == [[0, 0], [4, 4], [1, 3]]

    def test_cf_local_contrast_hop_anchors_each_batch_on_itself(self):
        from circuit.discovery import counterfactual_gradient as mod

        algo = mod.CounterfactualGradientDiscovery(
            _local_inference(), _DiscoveryBank(), torch.zeros(1, D_SAE), MagicMock(),
            min_active_count=1,
        )
        algo.position_aware = True
        algo.neg_batch_size = 2
        algo.activator_signal = "gradient"
        seen = []

        def record(**kwargs):
            seen.append(kwargs["position_aware"].peaks.clone())
            return {}, {}

        with patch.object(mod, "SeedProjectionInstrument", _FakeSeedInstrument), \
                patch.object(mod, "compute_latent_counterfactual_scores", side_effect=record), \
                patch.object(mod, "latent_stats", MagicMock()):
            algo._run_contrast_hop(SEED_LAYER * len(KINDS) + KINDS.index(SEED_KIND), 0,
                                   _tokens(), 5.0, MagicMock())
        assert [p.tolist() for p in seen] == [[0, 0], [4, 4], [1, 3]]
