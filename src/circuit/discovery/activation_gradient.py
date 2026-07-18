"""Activation-gradient discovery — a top-level method.

Attributes the seed's own firing on its POSITIVE contexts: one total-effect
gradient pass, scored ``grad(seed peak) x natural value`` per upstream latent,
then a per-position select + union over the seed's causal prefix (the
position-aware allowed set). It answers *"which active upstream latents is the
seed's firing built from?"* — the same question as ablation gradient, but with
no baseline and no path: a single-point ``grad x value`` at the natural posctx
state.

Why a METHOD and not a mode. The others (local / ig_baseline / restoration /
ig_restoration / contrastive_ig) vary *where the gradient is linearised* along a
baseline->target path while sharing an input regime and objective. Activation
gradient shares none of that axis: it has its own input regime (posctx, seed
present), its own attribution (``grad x natural``, no baseline — so it cannot
find absent latents, which is exactly why it is not a counterfactual mode), and
its own endpoint (the seed's real firing peak). It was briefly reachable as an
``attribution_mode`` on both gradient methods for implementation reuse; hosting
it under counterfactual gradient was misleading (the discovery ignored negctx
entirely), so it is promoted here.

Implementation: it is ablation gradient's posctx support-discovery with the
attribution hop fixed to the position-aware ``grad x natural`` union, so it
inherits that method's assembly, evaluation (posctx suppression + cf
faithfulness) and pruning unchanged. The reusable algorithm itself lives in
``circuit.instrument.position_aware.position_aware_membership``.
"""

from typing import Dict, Tuple

import torch

from circuit.types.feature_id import FeatureID
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx

from .ablation_gradient import AblationGradientDiscovery


class ActivationGradientDiscovery(AblationGradientDiscovery):
    """Position-aware ``grad x natural`` support discovery on posctx."""

    method_name = "activation_gradient"

    def _run_ablation_hop(
        self,
        seed_comp_idx: int,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], float, float]:
        """This method IS the position-aware posctx hop — no attribution_mode
        dispatch. (target_loss / target_pre_act are unused by the position-aware
        path; returned as 0.0 for the shared assembly's signature.)"""
        n_kinds = len(self.sae_bank.kinds)
        seed_layer, seed_kind_idx = split_component_idx(seed_comp_idx, n_kinds)
        seed_kind = self.sae_bank.kinds[seed_kind_idx]
        return self._run_position_aware_hop(
            seed_layer, seed_kind, seed_latent_idx, pos_tokens, pos_argmax, logger
        )

    def _run_position_aware_hop(
        self,
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        pos_tokens: torch.Tensor,
        pos_argmax: torch.Tensor,
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], float, float]:
        """Union each prefix position's selected ``grad x natural`` latents
        (supports); negative-attribution latents are stashed as inhibitors for
        the shared assembly, exactly as the ablation hops deliver them."""

        from circuit.instrument.position_aware import position_aware_membership
        from eval.ablation_faithfulness import upstream_sites

        sites = upstream_sites(self.sae_bank, seed_layer, seed_kind)
        supports, inhibitors = position_aware_membership(
            self.inference, self.sae_bank,
            tokens=pos_tokens, sites=sites,
            seed_layer=seed_layer, seed_kind=seed_kind, seed_latent_idx=seed_latent_idx,
            pos_argmax=pos_argmax, top_n=self.position_aware_top_n,
            select=self.position_aware_select, threshold=self.position_aware_threshold,
            position_weight=self.position_aware_position_weight, scope=self.position_aware_scope,
            negative_roles=self.negative_roles == "include",
            batch_size=self.probe_batch_size,
        )
        self._pending_inhibitors = inhibitors if self.negative_roles == "include" else {}
        sel = (f"top_n={self.position_aware_top_n}" if self.position_aware_select == "top_n"
               else f"select={self.position_aware_select} threshold={self.position_aware_threshold}")
        sel += f" scope={self.position_aware_scope}" + (" +posw" if self.position_aware_position_weight else "")
        logger.note(
            f"activation-gradient: {len(supports)} supports + "
            f"{len(self._pending_inhibitors)} inhibitors ({sel})"
        )
        return supports, 0.0, 0.0


__all__ = ["ActivationGradientDiscovery"]
