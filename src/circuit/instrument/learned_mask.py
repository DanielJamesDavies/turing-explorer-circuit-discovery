"""Learned continuous-mask circuit discovery (the loss-following mode family).

Motivated by the 2026-07-24 gradient-observability findings: instantaneous
gradient signs are state-dependent to near-independence (corr(g_natural,
g_intervened) = 0.05 at L8), and the activator/inhibitor populations are
indistinguishable by value — so no single gradient, and no one-shot sign
split, is a reliable membership criterion. This mode never asks either
question: it optimises a soft membership m in (0,1)^N directly against a
reconstruction loss and lets the loss decide what stays.

Three objectives, one engine (mode names follow the ig_mean/ig_negctx
precedent — the _negctx suffix marks the counterfactual-distribution worker):

  "pos"      (abl-mask)          L = mse(pre_pos(m), pre_pos(natural)) + l1.mean(m)
             The sparsest set reproducing natural firing.
  "contrast" (cf-mask_contrast)  + beta * mse(pre_neg(m), pre_neg(natural))
             ... that also keeps the seed silent on negctx (selectivity).
  "negctx"   (cf-mask_negctx)    L = mse(pre_neg(m), target_pos) + l1.mean(1 - m)
             The minimal EDIT to the natural negctx stream that fires the
             seed: a gate-opening search over present-on-negctx latents.
             Sparsity flips to the edit (1 - m) because the reference state
             is the natural stream, not the empty one. Reported residual gap
             measures suppression-gated vs drive-absent silence.

Design rules carried over from the week's findings:
  * losses target the seed PRE-activation (the post-top-k read is censored to
    exactly 0 below the cutoff — zero gradient exactly where deep seeds live);
  * "reproduce, don't maximise": every target is a natural level, so the
    optimiser cannot win by super-stimulus amplification;
  * held-out probe split: optimisation uses the train slice only, and the
    final data loss is also reported on the held-out slice (a learned method
    can overfit probes in a way one-shot attribution cannot).

The masked transform is the free-execution semantics made differentiable:
decode(m * dense) + error, error preserved from the unmodified encoding.
m = 1 is exactly the identity (decode(all) + error == x).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID
from model.hooks import multi_patch
from sae.dense import sparse_topk_to_dense

Site = Tuple[int, str]

OBJECTIVES = ("pos", "contrast", "negctx")


class LearnedMaskPatcher:
    """Differentiable masked forward: at every masked site the dense code is
    multiplied by sigmoid(theta) before decode (error term preserved); at the
    seed's site the pre-activation (w.x + b) is captured and x passes
    untouched. thetas are shared across forwards — the optimiser owns them."""

    def __init__(self, bank: Any, thetas: Dict[Site, torch.Tensor],
                 seed_layer: int, seed_kind: str,
                 w_seed: torch.Tensor, b_seed: torch.Tensor) -> None:
        self.bank = bank
        self.thetas = thetas
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.w_seed = w_seed
        self.b_seed = b_seed
        self.seed_pre: Optional[torch.Tensor] = None

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        theta = self.thetas.get((layer_idx, kind))
        if theta is None:
            return x
        ta, ti = self.bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, self.bank.d_sae, dtype=torch.float32)
        recon = self.bank.decode(dense, kind, layer_idx)
        error = x - recon.to(x.dtype)
        m = torch.sigmoid(theta).to(device=dense.device, dtype=dense.dtype)
        out = self.bank.decode(dense * m, kind, layer_idx)
        return out.to(x.dtype) + error


def _forward_preact(inference: Any, patcher: LearnedMaskPatcher,
                    tokens: torch.Tensor, grad: bool) -> torch.Tensor:
    patcher.seed_pre = None
    inference.forward(tokens, patcher=patcher, grad_enabled=grad,
                      return_activations=False, tokenize_final=False)
    if patcher.seed_pre is None:
        raise RuntimeError("learned_mask: seed pre-activation was not captured")
    return patcher.seed_pre


def _at(pre: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    B = min(pre.shape[0], anchors.shape[0])
    idx = torch.arange(B, device=pre.device)
    pa = anchors[:B].to(pre.device).clamp(0, pre.shape[1] - 1)
    return pre[:B][idx, pa]


@torch.no_grad()
def _natural(inference: Any, bank: Any, tokens: torch.Tensor,
             seed_layer: int, seed_kind: str,
             w_seed: torch.Tensor, b_seed: torch.Tensor) -> torch.Tensor:
    """Natural (m=1 == untouched) seed pre-activation [B, T]."""
    p = LearnedMaskPatcher(bank, {}, seed_layer, seed_kind, w_seed, b_seed)
    return _forward_preact(inference, p, tokens, grad=False).detach()


def run_learned_mask(
    inference: Any,
    bank: Any,
    *,
    objective: str,
    sites: List[Site],
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    pos_tokens: torch.Tensor,
    pos_argmax: torch.Tensor,
    neg_tokens: Optional[torch.Tensor] = None,
    target_act: Optional[float] = None,
    steps: int = 200,
    lr: float = 0.05,
    l1_lambda: float = 1e-3,
    beta: float = 1.0,
    keep_threshold: float = 0.5,
    batch_size: int = 4,
    holdout_frac: float = 0.25,
    theta_init: float = 4.0,
    log_every: int = 50,
    deep_site_threshold: Optional[int] = None,
    deep_batch_size: Optional[int] = None,
    logger: Any = None,
) -> Tuple[Dict[FeatureID, float], Dict[str, Any]]:
    """Optimise the mask and return (scores, provenance).

    Scores: "pos"/"contrast" -> {fid: m} for kept members (m > keep_threshold,
    all positive). "negctx" -> {fid: -(1 - m)} for edited members
    ((1 - m) > keep_threshold, negative — they are delivered as inhibitors:
    the latents whose PRESENCE holds the seed off).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective in ("contrast", "negctx"):
        if neg_tokens is None or int(neg_tokens.shape[0]) == 0:
            raise ValueError(f"objective={objective!r} requires neg_tokens")
    if objective == "negctx" and target_act is None:
        raise ValueError("objective='negctx' requires target_act (posctx level)")

    # Depth-adaptive VRAM guard (the sites x per-site-tensors law): every
    # backward holds dense-code graphs at ALL masked sites simultaneously,
    # so deep seeds shrink the MICRO-batch — but keep the effective batch via
    # GRADIENT ACCUMULATION (micro-chunks backwarded separately, one
    # opt.step()). The optimisation regime is therefore identical across
    # depths; only the peak VRAM changes. Without this, deep seeds either
    # spill into WDDM shared memory at PCIe speed (measured, L10 at batch 4)
    # or pay batch-2 gradient noise (the pre-accumulation version).
    micro_bs = max(1, int(batch_size))
    if (deep_site_threshold is not None and deep_batch_size is not None
            and len(sites) > int(deep_site_threshold)):
        micro_bs = max(1, int(deep_batch_size))
        if logger is not None:
            logger.note(f"learned_mask: {len(sites)} sites > "
                        f"{deep_site_threshold} — micro-batch "
                        f"{batch_size} -> {micro_bs} with gradient "
                        f"accumulation (effective batch unchanged)")
    accum = max(1, -(-int(batch_size) // micro_bs))   # ceil division

    sae = bank.saes[seed_kind][seed_layer]
    w_seed = sae.encoder.weight[seed_latent_idx].detach()
    b_seed = sae._get_bias_eff()[seed_latent_idx].detach()
    device = getattr(bank, "device", pos_tokens.device)

    thetas: Dict[Site, torch.Tensor] = {
        s: torch.full((bank.d_sae,), float(theta_init), device=device,
                      requires_grad=True)
        for s in sites
    }
    params = list(thetas.values())
    opt = torch.optim.Adam(params, lr=lr)

    def split(tokens, anchors):
        n = int(tokens.shape[0])
        n_hold = int(round(n * holdout_frac))
        n_train = max(1, n - n_hold)
        return (tokens[:n_train], anchors[:n_train],
                tokens[n_train:], anchors[n_train:])

    # ---- targets from the NATURAL stream (reproduce, don't maximise) -------
    pos_nat = _natural(inference, bank, pos_tokens, seed_layer, seed_kind,
                       w_seed, b_seed)
    pos_tgt_all = _at(pos_nat, pos_argmax)
    if objective in ("contrast", "negctx"):
        neg_nat = _natural(inference, bank, neg_tokens, seed_layer, seed_kind,
                           w_seed, b_seed)
        # would-be-firing anchor per negctx sequence (pre-act argmax) — the
        # same anchor the anchored cf eval and ig_negctx use.
        neg_anchors = neg_nat.argmax(dim=-1).cpu()
        neg_tgt_all = _at(neg_nat, neg_anchors)

    pt_tr, pa_tr, pt_ho, pa_ho = split(pos_tokens, pos_argmax)
    ptgt_tr = pos_tgt_all[:pt_tr.shape[0]]
    ptgt_ho = pos_tgt_all[pt_tr.shape[0]:]
    if objective in ("contrast", "negctx"):
        nt_tr, na_tr, nt_ho, na_ho = split(neg_tokens, neg_anchors)
        ntgt_tr = neg_tgt_all[:nt_tr.shape[0]]
        ntgt_ho = neg_tgt_all[nt_tr.shape[0]:]

    patcher = LearnedMaskPatcher(bank, thetas, seed_layer, seed_kind,
                                 w_seed, b_seed)

    def mask_mean() -> torch.Tensor:
        return torch.stack([torch.sigmoid(t).mean() for t in params]).mean()

    def mask_sum() -> torch.Tensor:
        # Penalty in PER-LATENT units (sum, not mean): mean-normalising over
        # ~3e5 latents makes the per-latent L1 gradient ~lambda*sigma'/N ~ 5e-10,
        # BELOW Adam's eps (1e-8) — the epsilon floor then swallows the
        # sparsity pressure and nothing prunes (measured: the L2 spike kept
        # all 327,680 latents). With sum, every latent feels lambda*sigma'(theta)
        # regardless of dictionary size, so lambda is a per-latent price.
        return torch.stack([torch.sigmoid(t).sum() for t in params]).sum()

    def edit_sum() -> torch.Tensor:
        return torch.stack([(1.0 - torch.sigmoid(t)).sum() for t in params]).sum()

    def data_loss(tokens, anchors, targets, micro_index) -> torch.Tensor:
        s = (micro_index * micro_bs) % max(int(tokens.shape[0]), 1)
        tk, an = tokens[s:s + micro_bs], anchors[s:s + micro_bs]
        tg = targets[s:s + micro_bs].to(device)
        if tk.shape[0] == 0:
            tk, an, tg = tokens[:micro_bs], anchors[:micro_bs], targets[:micro_bs].to(device)
        pre = _forward_preact(inference, patcher, tk, grad=True)
        vals = _at(pre, an)
        if objective == "negctx":
            tgt = torch.full_like(vals, float(target_act))
        else:
            tgt = tg[:vals.shape[0]].to(vals.device, vals.dtype)
        return ((vals - tgt) ** 2).mean()

    losses: List[float] = []
    inference.disable_compile()
    try:
        for step in range(int(steps)):
            opt.zero_grad()
            # Gradient accumulation: each micro-chunk's scaled loss is
            # backwarded on its own (freeing that chunk's graph before the
            # next forward), so peak VRAM is one micro-chunk while the
            # STEP's gradient equals the full effective-batch gradient.
            step_total = 0.0
            for j in range(accum):
                mi = step * accum + j
                if objective == "pos":
                    part = data_loss(pt_tr, pa_tr, ptgt_tr, mi) / accum
                elif objective == "contrast":
                    part = (data_loss(pt_tr, pa_tr, ptgt_tr, mi)
                            + beta * data_loss(nt_tr, na_tr, ntgt_tr, mi)) / accum
                else:  # negctx — sparsity on the EDIT (1 - m): reference is
                    # the natural stream, so cost accrues for turning DOWN.
                    part = data_loss(nt_tr, na_tr, ntgt_tr, mi) / accum
                part.backward()
                step_total += float(part.detach())
            penalty = (l1_lambda * edit_sum() if objective == "negctx"
                       else l1_lambda * mask_sum())
            penalty.backward()
            opt.step()
            losses.append(step_total + float(penalty.detach()))
            if logger is not None and log_every and step % int(log_every) == 0:
                logger.note(f"learned_mask[{objective}] step {step} "
                            f"loss {losses[-1]:.5f} mean_m {float(mask_mean()):.4f}")

        # held-out data loss at the final mask (no grad)
        with torch.no_grad():
            if objective == "pos" and pt_ho.shape[0]:
                pre = _forward_preact(inference, patcher, pt_ho, grad=False)
                ho = float(((_at(pre, pa_ho) - ptgt_ho.to(device)) ** 2).mean())
            elif objective == "contrast" and pt_ho.shape[0] and nt_ho.shape[0]:
                pre_p = _forward_preact(inference, patcher, pt_ho, grad=False)
                pre_n = _forward_preact(inference, patcher, nt_ho, grad=False)
                ho = float(((_at(pre_p, pa_ho) - ptgt_ho.to(device)) ** 2).mean()
                           + beta * ((_at(pre_n, na_ho) - ntgt_ho.to(device)) ** 2).mean())
            elif objective == "negctx" and nt_ho.shape[0]:
                pre = _forward_preact(inference, patcher, nt_ho, grad=False)
                v = _at(pre, na_ho)
                ho = float(((v - float(target_act)) ** 2).mean())
            else:
                ho = None
    finally:
        inference.enable_compile()

    # ---- selection ----------------------------------------------------------
    scores: Dict[FeatureID, float] = {}
    kept_m: List[float] = []
    with torch.no_grad():
        for (layer, kind), theta in thetas.items():
            m = torch.sigmoid(theta)
            if objective == "negctx":
                edit = 1.0 - m
                idx = (edit > keep_threshold).nonzero(as_tuple=True)[0]
                for i in idx.tolist():
                    scores[FeatureID(layer, kind, i)] = -float(edit[i])
                    kept_m.append(float(m[i]))
            else:
                idx = (m > keep_threshold).nonzero(as_tuple=True)[0]
                for i in idx.tolist():
                    scores[FeatureID(layer, kind, i)] = float(m[i])
                    kept_m.append(float(m[i]))

    provenance = {
        "objective": objective,
        "batch_size_used": int(batch_size),      # effective batch (unchanged by guard)
        "micro_batch": int(micro_bs),
        "accum_chunks": int(accum),
        "steps": int(steps),
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "holdout_data_loss": ho,
        "n_kept": len(scores),
        "mean_m_final": float(mask_mean().detach()),
    }
    if logger is not None:
        logger.note(f"learned_mask[{objective}]: kept {len(scores)} "
                    f"(loss {provenance['loss_initial']:.5f} -> "
                    f"{provenance['loss_final']:.5f}, holdout {ho})")
    return scores, provenance


__all__ = ["LearnedMaskPatcher", "run_learned_mask", "OBJECTIVES"]
