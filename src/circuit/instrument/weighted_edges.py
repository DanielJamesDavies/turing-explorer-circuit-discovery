"""Direct-effect edges for WEIGHTED circuits (WCM), linearised around the circuit's own run.

`edge_attribution.attach_direct_edges` linearises around the UNMODIFIED model, which describes the full model, not a
weighted circuit: a WCM circuit is only faithful in its circuit-only run, where every circuit latent carries
alpha x its live value, every other upstream latent sits at the ablation value (zero, or the A / C mean with the
code kept k-sparse), and the SAE error is recomputed from the edited stream. This module runs exactly that
computation (the same fill as `eval.ablation_faithfulness.CircuitOnlyPatcher` with `keep_scales`) and exposes the
circuit nodes to autograd in three modes:

  total   node = alpha * live + tap, tap a zero leaf: d m / d tap_u = g_u, the TOTAL gradient of the metric
          (the target's pre-activation at its anchor) with respect to node u, through every path.
  stop    node = leaf holding alpha * live: circuit nodes are gradient terminals, while model ops, the fill and the
          SAE errors stay live. d live_d / d leaf_u is then the DIRECT Jacobian (paths through no other circuit
          node), and d m / d leaf_u = h_u is u's direct effect on the target.
  pin     no gradients: every node is fixed to given values except one `live` node, and one node can be overridden
          (set to its floor). Used for the causal edge check.

With v_u the node's value and b_u its floor (the value latent u takes when dropped from the circuit):

  node attribution   A_u          = sum_{b,t} g_u . (v_u - b_u)
  edge u -> d        w(u -> d)    = sum_{b,t} g_d . (d live_d / d leaf_u) . (v_u - b_u)
  edge u -> target   w(u -> tgt)  = sum_{b,t} h_u . (v_u - b_u)

and, because every path from u to the metric either reaches it directly or has a FIRST circuit node d,

  A_u = w(u -> tgt) + sum_d w(u -> d)          (exact chain rule at one linearisation point)

which is the implementation check. The coefficients enter where they should: alpha_u through v_u - b_u, and
alpha_d through the Jacobian of live_d = alpha_d * f_d(x_d). Non-members are constants in this run, so edges only
ever connect circuit nodes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from model.hooks import multi_patch
from sae.dense import sparse_topk_to_dense

Site = Tuple[int, str]


class WeightedCircuitGraph:
    """Forward-pass instrument for one weighted circuit under one ablation method.

    keep:    site -> LongTensor of member latent indices (sites with no members are fully ablated)
    alphas:  site -> FloatTensor of the members' coefficients, aligned with keep[site]
    in_scope: every upstream site (the evaluator's `upstream_sites`)
    seed_site, seed_vec (w, b): the target's site and encoder row; the metric is its pre-activation at `anchors`
    site_means: site -> [d_sae] ablation values (None = zero ablation); respect_topk as in the evaluator
    """

    def __init__(self, bank, keep: Dict[Site, torch.Tensor], alphas: Dict[Site, torch.Tensor], in_scope,
                 seed_site: Site, seed_vec, anchors: torch.Tensor, site_means=None, respect_topk: bool = False,
                 topk: int = 128, mode: str = "total", pins: Optional[Dict[Site, torch.Tensor]] = None,
                 live: Optional[Tuple[Site, int]] = None, override: Optional[Tuple[Site, int, torch.Tensor]] = None):
        assert mode in ("total", "stop", "pin")
        self.bank, self.keep, self.alphas, self.in_scope = bank, keep, alphas, set(in_scope)
        self.seed_site, self.seed_vec, self.anchors = seed_site, seed_vec, anchors
        self.site_means, self.respect_topk, self.topk = site_means, respect_topk, topk
        self.mode, self.pins, self.live_node, self.override = mode, pins, live, override
        self.taps: Dict[Site, torch.Tensor] = {}      # total mode
        self.leaves: Dict[Site, torch.Tensor] = {}    # stop mode
        self.live: Dict[Site, torch.Tensor] = {}      # connected alpha * f(x) at member columns [B, T, n]
        self.values: Dict[Site, torch.Tensor] = {}    # node values actually written [B, T, n]
        self.floors: Dict[Site, torch.Tensor] = {}    # value each member takes if dropped [B, T, n]
        self.metric_vec: Optional[torch.Tensor] = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    # -- helpers -----------------------------------------------------------------------------------------------
    def _fill(self, site: Site, node_vals: torch.Tensor, like: torch.Tensor):
        """The ablation code with member columns zeroed (constant), and each member's floor. Same fill as
        CircuitOnlyPatcher: dense mean, zero, or (respect_topk) the top (k - #active members) non-members by
        mean."""
        kt = self.keep.get(site)
        mv = self.site_means[site].to(like.device, like.dtype) if self.site_means is not None else None
        B, T, D = like.shape
        n = 0 if kt is None else int(kt.numel())
        if mv is None:
            fill = torch.zeros_like(like)
            floors = torch.zeros(B, T, n, device=like.device, dtype=like.dtype)
        elif not self.respect_topk:
            fill = mv.expand_as(like).clone()
            if kt is not None:
                fill[:, :, kt] = 0
            floors = mv[kt].view(1, 1, n).expand(B, T, n).clone() if kt is not None else fill[:, :, :0]
        else:
            fill = torch.zeros_like(like)
            active = (node_vals != 0) if kt is not None else torch.zeros(B, T, 0, dtype=torch.bool, device=like.device)
            budget = (self.topk - active.sum(-1)).clamp(min=0)                               # [B, T]
            mean_rank = mv.clone().float()
            if kt is not None:
                mean_rank[kt] = float("-inf")
            max_b = int(budget.max().item())
            if max_b > 0:
                ranked = torch.argsort(mean_rank, descending=True)[:max_b]
                on = torch.arange(max_b, device=like.device).view(1, 1, max_b) < budget.unsqueeze(-1)
                fill[:, :, ranked] = (mv[ranked].view(1, 1, -1) * on.to(mv.dtype)).to(fill.dtype)
            if kt is not None:
                # dropping member u frees one budget slot where u was active; u is then filled iff its mean ranks
                # inside the (enlarged) budget among the non-members
                nonkept = mean_rank[torch.isfinite(mean_rank)]
                rank_u = (nonkept.view(1, -1) > mv[kt].float().view(-1, 1)).sum(1)          # [n]
                slot = budget.unsqueeze(-1) + active.long()                                  # [B, T, n]
                floors = (mv[kt].view(1, 1, n) * (rank_u.view(1, 1, n) < slot).to(mv.dtype)).to(like.dtype)
            else:
                floors = fill[:, :, :0]
        return fill, floors

    # -- the hook ----------------------------------------------------------------------------------------------
    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        site = (layer_idx, kind)
        if site == self.seed_site:
            w, b = self.seed_vec
            pre = x.float() @ w.to(x.device).float() + b.to(x.device).float()               # [B, T]
            B = min(pre.shape[0], self.anchors.shape[0])
            pa = self.anchors[:B].to(pre.device).clamp(0, pre.shape[1] - 1)
            self.metric_vec = pre[:B][torch.arange(B, device=pre.device), pa]
            return x
        if site not in self.in_scope:
            return x

        top_acts, top_idx = self.bank.encode(x, kind, layer_idx)
        f = sparse_topk_to_dense(top_acts, top_idx, self.bank.d_sae, dtype=x.dtype)
        error = x - self.bank.decode(f, kind, layer_idx)                                    # live SAE error
        kt = self.keep.get(site)
        if kt is None:
            fill, _ = self._fill(site, None, f.detach())
            return self.bank.decode(fill, kind, layer_idx) + error

        live = f[:, :, kt] * self.alphas[site].to(f.device, f.dtype)                        # [B, T, n]
        if self.mode == "total":
            tap = torch.zeros_like(live, requires_grad=True)
            self.taps[site] = tap
            node = live + tap
        elif self.mode == "stop":
            node = live.detach().requires_grad_(True)
            self.leaves[site] = node
        else:
            node = self.pins[site].to(live.device, live.dtype).clone()
            if self.live_node is not None and self.live_node[0] == site:
                node[:, :, self.live_node[1]] = live[:, :, self.live_node[1]]
        if self.override is not None and self.override[0] == site:
            node = node.clone()
            node[:, :, self.override[1]] = self.override[2].to(node.device, node.dtype)
        self.live[site] = live
        self.values[site] = node.detach()
        fill, floors = self._fill(site, node.detach(), f.detach())
        self.floors[site] = floors
        patched = fill.clone()
        patched[:, :, kt] = node
        return self.bank.decode(patched, kind, layer_idx) + error


__all__ = ["WeightedCircuitGraph"]
