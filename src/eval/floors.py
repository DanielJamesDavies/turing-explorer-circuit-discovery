"""Mean-ablation floors and site anchors — shared, method-agnostic.

Single home for every consumer of mean-ablation baselines: the
ablation-faithfulness evals (free/pinned, a_empty), the ig_baseline and
restoration attribution modes, and the sweep runner. One knob
(config.discovery.floor_source) selects the floor semantics everywhere,
keeping the discovery/evaluation pact intact:

  - "posctx"  — per-seed means over the seed's positive probe batch
                (SFC's distribution-matched baseline; Marks et al., 2025).
  - "global"  — seed-independent means over a random corpus sample
                (density-weighted corpus expectation; colder, no
                evaluation-distribution leakage).
  - "diverse" — seed-independent means over a farthest-point sample of the
                corpus by stored sequence representation (coverage-weighted:
                a DIFFERENT estimand from "global", overweighting rare
                context regions by design — not a variance-reduced corpus
                mean).

Seed-independent floors are cached per process under their source key.
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
) -> Dict[Site, torch.Tensor]:
    """Apply the shared config.discovery.floor_source knob: return the
    posctx means unchanged, or swap in cached global/diverse floors."""

    from config import config

    source = str(config.discovery.floor_source)
    if source == "posctx":
        return posctx_means
    if loader is None:
        raise ValueError(f"floor_source={source!r} requires a data loader")
    if source == "diverse":
        return collect_diverse_site_floors(inference, sae_bank, loader, sites)
    return collect_global_site_floors(inference, sae_bank, loader, sites)


__all__ = [
    "collect_site_anchors",
    "collect_site_means",
    "collect_global_site_floors",
    "collect_diverse_site_floors",
    "resolve_site_floors",
]
