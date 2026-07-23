"""Mean-ablation floors and site anchors — shared, method-agnostic.

Single home for every consumer of mean-ablation baselines: the
ablation-faithfulness evals (free/pinned, a_empty), the ig_mean and
restoration attribution modes, and the sweep runner. One knob
(config.discovery.floor_source) selects the floor semantics everywhere,
keeping the discovery/evaluation pact intact:

  - "posctx"  — per-seed means over the seed's positive probe batch
                (SFC's distribution-matched baseline; Marks et al., 2025).
  - "negctx"  — per-seed means over the seed's retrieved NEGATIVE contexts:
                sequences selected because the seed is silent on them.
  - "global"  — seed-independent means over a random corpus sample
                (density-weighted corpus expectation; colder, no
                evaluation-distribution leakage).
  - "diverse" — seed-independent means over a farthest-point sample of the
                corpus by stored sequence representation (coverage-weighted:
                a DIFFERENT estimand from "global", overweighting rare
                context regions by design — not a variance-reduced corpus
                mean).

Seed-independent floors ("global"/"diverse") are cached per process under
their source key. "posctx"/"negctx" are seed-specific and never cached.

On "negctx". Order the floors by what they REMOVE from the stream:

  zero    removes everything, preserves nothing  — a sparser-than-k code the
          model never produces.
  posctx  removes nothing specific, preserves everything INCLUDING the seed's
          own firing signature. The mean over contexts where a latent fires
          carries that firing by construction, so the floor drives the seed
          before any circuit exists: measured at 23%/30%/26% of a_pos on the
          L8/L9/L10 validation seeds (0% at L2 — the leak grows with depth).
          See dev-notes/data/floor-diagnostic-2026-07-23.
  negctx  removes seed-specific content, preserves generic stream content.

That middle column is the point: "what makes this latent fire?" wants a
baseline that strips what is specific to firing while keeping the ordinary
background a real sequence carries. It also matches the field's convention —
ACDC resamples from a corrupted prompt, and SFC's contrastive mode patches
from the counterfactual distribution — where our posctx default is the
outlier, being conditioned on the very event it is used to explain. The
asymmetry is that for a BEHAVIOURAL endpoint the task mean does not trivially
reproduce the behaviour, whereas for a LATENT endpoint it does.

Measured: a_empty(negctx) == 0.0000 on all four validation seeds, so it
reaches free0's clean denominator (den == a_pos) through a real non-firing
state rather than an empty code. Note freeN != free0 even so — the numerators
are different forward passes over a shared denominator.

Which negatives. The floor uses ProbeDataset.neg_tokens, which comes straight
from the per-latent neg_ctx store: the 64 NEAREST non-activating sequences to
the latent's positive-context centroid (cosine over mean-pooled sequence
representations, exact KNN over a 512-neighbour pool, positives excluded —
store/neg_ctx/component.py). These are CLOSE / hard negatives.

Note this does NOT co-vary with config.discovery.*.neg_mode: that knob selects
the cf method's own negatives (ig_negctx, restoration_negctx, phi_cf) via
_get_neg_tokens, a path the floor never takes. The floor is pinned to close
negatives by construction, so no separate floor_negctx_mode knob is needed.

Close negatives are the HARDEST case for this floor — being semantically
nearest the positive contexts, their mean is the closest to the posctx mean
that any negative mode could give, so they remove the least. The measured
a_empty == 0.0 therefore holds a fortiori: random or distant negatives would
only separate further. It is also the sharpest contrast, since what survives
is what makes the latent fire rather than a generic topic difference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import torch

from sae.dense import sparse_topk_to_dense

Site = Tuple[int, str]

# Process-level cache for seed-independent floors: {source: {site: [d_sae]}}.
_GLOBAL_FLOOR_CACHE: Dict[str, Dict[Site, torch.Tensor]] = {}


@torch.no_grad()
def collect_site_anchors(
    inference: Any,
    sae_bank: Any,
    tokens: torch.Tensor,
    sites: Set[Site],
    argmax: Optional[torch.Tensor] = None,
    pin_position_specific: bool = False,
) -> Tuple[Dict[Site, torch.Tensor], Dict[Site, torch.Tensor]]:
    """One clean forward pass returning, per site, both anchors:

    - means [d_sae]: mean dense latent vector over all (batch, position)
      slots, zeros included — the "average value over data from D" of SFC's
      mean ablation.
    - pins: the clean-run values used to pin kept latents. Two shapes:
      * collapsed (default) — [d_sae], the mean dense vector at the probe
        positions (phi_cf injection semantics; falls back to the position
        mean when argmax is None). One value per latent, applied at every
        position.
      * position-specific (``pin_position_specific=True``) — [B, T, d_sae],
        the full clean dense stream, so kept latents are pinned to their
        actual per-position clean value. This is the correct pin for
        position-aware / allowed-set circuits (a collapsed pin discards the
        position axis the seed reads); stored on CPU to bound memory.
    """

    kind_to_idx = {k: i for i, k in enumerate(sae_bank.kinds)}
    means: Dict[Site, torch.Tensor] = {}
    pins: Dict[Site, torch.Tensor] = {}
    argmax_cpu = argmax.detach().cpu() if argmax is not None else None

    def hook(layer_idx: int, activations: tuple) -> None:
        for kind in sae_bank.kinds:
            if (layer_idx, kind) not in sites:
                continue
            act = activations[kind_to_idx[kind]]
            ta, ti = sae_bank.encode(act, kind, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, sae_bank.d_sae, dtype=torch.float32)
            means[(layer_idx, kind)] = dense.mean(dim=(0, 1)).detach()
            if pin_position_specific:
                # full [B, T, d_sae] clean stream (CPU-resident to bound memory);
                # the patcher moves each site to device as it runs.
                pins[(layer_idx, kind)] = dense.detach().cpu()
            elif argmax_cpu is not None:
                Bx, Tx = dense.shape[:2]
                actual_B = min(Bx, argmax_cpu.shape[0])
                pa = argmax_cpu[:actual_B].to(dense.device).clamp(0, Tx - 1)
                pins[(layer_idx, kind)] = (
                    dense[:actual_B][torch.arange(actual_B, device=dense.device), pa]
                    .mean(dim=0)
                    .detach()
                )
            else:
                pins[(layer_idx, kind)] = means[(layer_idx, kind)]

    inference.disable_compile()
    try:
        inference.forward(
            tokens,
            activations_callback=hook,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()

    missing = sites - set(means)
    if missing:
        raise RuntimeError(f"site anchors missing for sites: {sorted(missing)}")
    return means, pins


@torch.no_grad()
def collect_site_means(
    inference: Any,
    sae_bank: Any,
    tokens: torch.Tensor,
    sites: Set[Site],
) -> Dict[Site, torch.Tensor]:
    """Mean dense latent vector [d_sae] per site over a clean forward pass."""

    means, _ = collect_site_anchors(inference, sae_bank, tokens, sites)
    return means


@torch.no_grad()
def collect_site_error_means(
    inference: Any,
    sae_bank: Any,
    tokens: torch.Tensor,
    sites: Set[Site],
) -> Dict[Site, torch.Tensor]:
    """Mean SAE reconstruction error [d_model] per site over a clean forward
    pass: mean over (batch, position) of ``x - decode(encode(x))``.

    The mean-ablation floor for ERROR NODES — the error-space analogue of
    ``collect_site_means``. Feed to ``CircuitOnlyPatcher(error_means=...)`` so
    a non-member site's error is replaced by its distribution mean rather than
    zeroed (the same preserve/mean/zero ladder latents already have)."""

    kind_to_idx = {k: i for i, k in enumerate(sae_bank.kinds)}
    means: Dict[Site, torch.Tensor] = {}

    def hook(layer_idx: int, activations: tuple) -> None:
        for kind in sae_bank.kinds:
            if (layer_idx, kind) not in sites:
                continue
            act = activations[kind_to_idx[kind]]
            ta, ti = sae_bank.encode(act, kind, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, sae_bank.d_sae, dtype=torch.float32)
            recon = sae_bank.decode(dense, kind, layer_idx)
            means[(layer_idx, kind)] = (act.float() - recon.float()).mean(dim=(0, 1)).detach()

    inference.disable_compile()
    try:
        inference.forward(
            tokens,
            activations_callback=hook,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()

    missing = sites - set(means)
    if missing:
        raise RuntimeError(f"site error means missing for sites: {sorted(missing)}")
    return means


def _farthest_point_sample(reprs: torch.Tensor, n: int) -> list[int]:
    """Greedy k-center over l2-normalised representations (cosine distance).
    Deliberately coverage-weighted: overweights rare regions relative to a
    density-weighted uniform sample — a different floor semantics, not a
    variance-reduced estimate of the corpus mean."""

    reprs = torch.nn.functional.normalize(reprs.to(torch.float32), dim=-1)
    chosen = [0]
    # max_sim[i] = highest cosine similarity of candidate i to any chosen
    # sequence; the farthest point is the one whose best match is worst.
    max_sim = reprs @ reprs[0]
    for _ in range(min(n, reprs.shape[0]) - 1):
        candidate = int(max_sim.argmin().item())
        chosen.append(candidate)
        max_sim = torch.maximum(max_sim, reprs @ reprs[candidate])
    return chosen


@torch.no_grad()
def collect_global_site_floors(
    inference: Any,
    sae_bank: Any,
    loader: Any,
    sites: Set[Site],
    *,
    max_sequences: int = 128,
) -> Dict[Site, torch.Tensor]:
    """Seed-independent floors: mean dense latent vector per site over a
    sample of random corpus sequences ("average over D" with D = corpus).

    A colder baseline than posctx floors — no evaluation-distribution
    information leaks into it. Computed once per process and cached; cache
    misses trigger one pass over the missing sites.
    """

    cache = _GLOBAL_FLOOR_CACHE.setdefault("global", {})
    missing = {site for site in sites if site not in cache}
    if missing:
        _, tokens = next(loader.get_batches(device=sae_bank.device))
        tokens = tokens[:max_sequences]
        means, _ = collect_site_anchors(inference, sae_bank, tokens, missing)
        for site, mean in means.items():
            cache[site] = mean.detach().cpu()
    return {site: cache[site] for site in sites}


@torch.no_grad()
def collect_diverse_site_floors(
    inference: Any,
    sae_bank: Any,
    loader: Any,
    sites: Set[Site],
    *,
    n_sequences: int = 128,
    pool_batches: int = 4,
) -> Dict[Site, torch.Tensor]:
    """Coverage-weighted floors: farthest-point sample (greedy k-center over
    the stored sequence representations) of a corpus candidate pool, then
    the same anchors pass. Cached per process like the global floors."""

    from store.seq_repr import seq_repr

    cache = _GLOBAL_FLOOR_CACHE.setdefault("diverse", {})
    missing = {site for site in sites if site not in cache}
    if missing:
        if seq_repr is None:
            raise RuntimeError("floor_source='diverse' requires the seq_repr store to be loaded")
        pool_ids: list[int] = []
        batch_iter = loader.get_batches(device=sae_bank.device)
        for _ in range(pool_batches):
            try:
                ids, _tokens = next(batch_iter)
            except StopIteration:
                break
            pool_ids.extend(int(i) for i in ids.tolist())
        if not pool_ids:
            raise RuntimeError("diverse floors: corpus pool is empty")
        id_tensor = torch.tensor(pool_ids, dtype=torch.long)
        reprs = seq_repr.get_repr(id_tensor)
        chosen = _farthest_point_sample(reprs, n_sequences)
        chosen_ids = [pool_ids[i] for i in chosen]
        _, tokens = next(loader.get_batches_by_ids(chosen_ids, device=sae_bank.device))
        means, _ = collect_site_anchors(inference, sae_bank, tokens, missing)
        for site, mean in means.items():
            cache[site] = mean.detach().cpu()
    return {site: cache[site] for site in sites}


def resolve_site_floors(
    inference: Any,
    sae_bank: Any,
    sites: Set[Site],
    *,
    posctx_means: Dict[Site, torch.Tensor],
    loader: Any = None,
    neg_tokens: Optional[torch.Tensor] = None,
) -> Dict[Site, torch.Tensor]:
    """Apply the shared config.discovery.floor_source knob: return the posctx
    means unchanged, or swap in zero / negctx / cached global / diverse floors.

    ``neg_tokens`` is required only by floor_source="negctx" and ignored
    otherwise, so callers that cannot supply negatives are unaffected under
    every other source (including the "posctx" default)."""

    from config import config

    source = str(config.discovery.floor_source)
    if source == "posctx":
        return posctx_means
    if source == "zero":
        # The zero-ablation counterfactual (free0's own): every latent's floor
        # value is 0. Shapes/dtypes/devices mirror the posctx means so every
        # consumer is drop-in.
        return {site: torch.zeros_like(means) for site, means in posctx_means.items()}
    if source == "negctx":
        # Seed-specific, so never cached: these are THIS seed's negatives.
        if neg_tokens is None or int(neg_tokens.shape[0]) == 0:
            raise ValueError(
                "floor_source='negctx' requires neg_tokens, but none were "
                "supplied — either this seed retrieved no negative contexts, "
                "or this caller does not thread them. Refusing to fall back: a "
                "result labelled 'negctx' that silently used another floor is "
                "worse than a visible failure."
            )
        return collect_site_means(inference, sae_bank, neg_tokens, sites)
    if loader is None:
        raise ValueError(f"floor_source={source!r} requires a data loader")
    if source == "diverse":
        return collect_diverse_site_floors(inference, sae_bank, loader, sites)
    return collect_global_site_floors(inference, sae_bank, loader, sites)


__all__ = [
    "collect_site_anchors",
    "collect_site_means",
    "collect_site_error_means",
    "collect_global_site_floors",
    "collect_diverse_site_floors",
    "resolve_site_floors",
]
