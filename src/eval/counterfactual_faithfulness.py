"""
Counterfactual faithfulness evaluation for CounterfactualGradientDiscovery.

Two paired scores are computed, each using the same shared forward passes.

─── Score 1: negctx activation score  (counterfactual_faithfulness) ───────────
Asks: "can we *activate* the seed on negctx by injecting activators and
removing inhibitors?"

  Intervention on negctx:
    - Absent activators → injected at their posctx mean values
    - Present inhibitors → suppressed to zero

  Score = (a_intervened_neg − a_baseline) / (a_posctx − a_baseline)

  Near 1.0: the discovered nodes fully explain why the seed is absent on negctx.

─── Score 2: posctx suppression score  (posctx_suppression_score) ─────────────
Asks: "can we *suppress* the seed on posctx by removing activators and
injecting inhibitors?"

  Intervention on posctx:
    - Present activators → suppressed to zero
    - Absent inhibitors → injected at their negctx mean values

  Score = (a_posctx − a_intervened_pos) / (a_posctx − a_baseline)

  Near 1.0: removing activators and adding inhibitors fully silences the seed on
  posctx — confirming that these nodes are causally necessary drivers.

─── Shared variables ───────────────────────────────────────────────────────────
  a_posctx   = seed activation on posctx without intervention
  a_baseline = seed activation on negctx without intervention (≈ 0)

Four no-grad forward passes in total:
  Pass 1 — posctx    : collect a_posctx + per-activator posctx targets
  Pass 2 — negctx    : collect a_baseline + per-inhibitor negctx targets
  Pass 3 — negctx    : intervened (inject activators, suppress inhibitors)
  Pass 4 — posctx    : intervened (suppress activators, inject inhibitors)

Returns:
  Tuple[float, float] = (counterfactual_faithfulness, posctx_suppression_score)
"""

import sys
import torch
from typing import Any, Dict, List, Optional, Set, Tuple

from store.circuits import Circuit
from model.hooks import multi_patch


class CounterfactualInterventionPatcher:
    """
    A forward-pass hook that injects activator activations and suppresses
    inhibitor activations on negctx sequences, then captures the seed
    latent's response at the probe position.

    At each (layer, kind) that has activators or inhibitors:
      1. Encode the arriving residual stream into SAE sparse activations.
      2. Build a dense latent tensor from top-k.
      3. Compute error = x - decode(natural_latents)  (always preserved).
      4. Override: activator latents ← posctx mean value; inhibitors ← 0.
      5. Return decode(patched_latents) + error.

    At (seed_layer, seed_kind): capture the seed latent's activation from x
    (which already reflects upstream interventions) and return x unchanged.
    The capture and the intervention are mutually exclusive by construction —
    the seed's own (layer, kind) never appears in activator_targets or
    inhibitor_indices (the seed node has role="seed").
    """

    def __init__(
        self,
        bank: Any,
        activator_targets: Dict[Tuple[int, str], Dict[int, float]],
        inhibitor_indices: Dict[Tuple[int, str], Set[int]],
        seed_layer: int,
        seed_kind: str,
        seed_latent_idx: int,
        pos_argmax: Optional[torch.Tensor] = None,
        circuit_layers: Optional[Set[int]] = None,
    ) -> None:
        self.bank = bank
        self.activator_targets = activator_targets
        self.inhibitor_indices = inhibitor_indices
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.seed_latent_idx = seed_latent_idx
        self.pos_argmax = pos_argmax.detach().cpu() if pos_argmax is not None else None
        self.circuit_layers = circuit_layers
        self.captured_activation: Optional[float] = None

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        target_dtype = x.dtype

        # ── Capture seed latent activation ────────────────────────────────
        # x at (seed_layer, seed_kind) already reflects all upstream
        # interventions because the forward pass is sequential through layers.
        # We capture before returning so the measurement is the seed's
        # natural response to the modified upstream residual stream.
        if layer_idx == self.seed_layer and kind == self.seed_kind:
            top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
            is_seed = (top_indices == self.seed_latent_idx)
            seed_dense = torch.where(is_seed, top_acts, torch.zeros_like(top_acts)).sum(-1)  # [B, T]
            n_active = int((seed_dense > 0).sum().item())
            print(
                f"      [CF-Capture] Layer {layer_idx} {kind} "
                f"| Seed {self.seed_latent_idx} | Active in {n_active}/{B * T}"
            )
            sys.stdout.flush()
            if self.pos_argmax is not None:
                actual_B = min(B, self.pos_argmax.shape[0])
                pa = self.pos_argmax[:actual_B].to(x.device).clamp(0, T - 1)
                val = seed_dense[:actual_B][torch.arange(actual_B, device=x.device), pa].mean().item()
            else:
                val = seed_dense.mean().item()
            self.captured_activation = val
            print(f"      [CF-Capture] Mean at probe pos: {val:.4f}")
            sys.stdout.flush()

        # ── Intervention ──────────────────────────────────────────────────
        if self.circuit_layers is not None and layer_idx not in self.circuit_layers:
            return x

        has_act = (layer_idx, kind) in self.activator_targets
        has_inh = (layer_idx, kind) in self.inhibitor_indices
        if not has_act and not has_inh:
            return x

        # Build dense latent tensor from natural top-k encoding
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        all_latents = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=target_dtype)
        all_latents.scatter_(-1, top_indices.long(), top_acts.to(target_dtype))

        # Preserve the SAE error term from the unmodified encoding
        full_recon = self.bank.decode(all_latents, kind, layer_idx)
        error = x - full_recon

        # Apply the interventions to a copy of the latent tensor
        patched = all_latents.clone()
        if has_act:
            for idx, val in self.activator_targets[(layer_idx, kind)].items():
                patched[:, :, idx] = float(val)
        if has_inh:
            for idx in self.inhibitor_indices[(layer_idx, kind)]:
                patched[:, :, idx] = 0.0

        return self.bank.decode(patched, kind, layer_idx) + error


@torch.no_grad()
def evaluate_counterfactual_faithfulness(
    inference: Any,
    sae_bank: Any,
    avg_acts: torch.Tensor,  # kept for API symmetry; not used internally
    circuit: Circuit,
    neg_tokens: torch.Tensor,
    pos_tokens: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    pos_argmax: Optional[torch.Tensor] = None,
    circuit_layers: Optional[Set[int]] = None,
) -> Tuple[float, float]:
    """
    Measure how well the discovered activators and inhibitors causally explain
    the seed's firing behaviour in both directions.

    Four no-grad forward passes are run (see module docstring for details):

      Pass 1 (posctx)           : a_posctx + per-activator posctx targets
      Pass 2 (negctx baseline)  : a_baseline + per-inhibitor negctx targets
      Pass 3 (intervened negctx): inject activators, suppress inhibitors
      Pass 4 (intervened posctx): suppress activators, inject inhibitors

    Returns:
        (counterfactual_faithfulness, posctx_suppression_score)

        counterfactual_faithfulness:
            (a_intervened_neg - a_baseline) / (a_posctx - a_baseline)
            Near 1.0 → injecting activators + removing inhibitors on negctx
            recovers posctx-level seed activation.

        posctx_suppression_score:
            (a_posctx - a_intervened_pos) / (a_posctx - a_baseline)
            Near 1.0 → suppressing activators + injecting inhibitors on posctx
            fully silences the seed.

    Args:
        avg_acts:       Not used; present for API symmetry with other evals.
        neg_tokens:     Negctx token sequences [B_neg, T].
        pos_tokens:     Posctx token sequences [B_pos, T].
        pos_argmax:     [B_pos] int tensor — probe token position per sequence.
        circuit_layers: Layer indices at which to apply the intervention.
                        When None, all (layer, kind) pairs with nodes are used.
    """
    kinds = sae_bank.kinds
    kind_to_idx: Dict[str, int] = {k: i for i, k in enumerate(kinds)}

    # ── Parse circuit nodes by role ───────────────────────────────────────
    activator_fids: Dict[Tuple[int, str], List[int]] = {}
    inhibitor_fids: Dict[Tuple[int, str], Set[int]] = {}

    for node in circuit.nodes.values():
        fid = node.feature_id
        if fid is None:
            continue
        role = node.metadata.get("role", "")
        key = (fid.layer, fid.kind)
        # Never intervene at the seed's own (layer, kind) — only the seed node
        # should occupy that slot; patching other latents there would modify the
        # residual stream at the measurement point.
        if key == (seed_layer, seed_kind):
            continue
        if role == "counterfactual_activator":
            activator_fids.setdefault(key, []).append(fid.index)
        elif role == "counterfactual_inhibitor":
            inhibitor_fids.setdefault(key, set()).add(fid.index)

    n_act = sum(len(v) for v in activator_fids.values())
    n_inh = sum(len(v) for v in inhibitor_fids.values())

    if n_act == 0 and n_inh == 0:
        print("  [CFaithfulness] No activator or inhibitor nodes — returning (0.0, 0.0)")
        sys.stdout.flush()
        return 0.0, 0.0

    print(
        f"  [CFaithfulness] Seed L{seed_layer} {seed_kind} idx {seed_latent_idx} | "
        f"{n_act} activators, {n_inh} inhibitors"
    )
    sys.stdout.flush()

    # ── Pass 1: posctx ────────────────────────────────────────────────────
    # Collect a_posctx and per-activator mean activation at probe positions.
    activator_targets: Dict[Tuple[int, str], Dict[int, float]] = {}
    a_posctx_buf: List[float] = []

    def posctx_hook(layer_idx: int, activations: tuple) -> None:
        # Seed activation on posctx
        if layer_idx == seed_layer and seed_kind in kind_to_idx:
            act = activations[kind_to_idx[seed_kind]]
            ta, ti = sae_bank.encode(act, seed_kind, layer_idx)
            is_s = (ti == seed_latent_idx)
            s_dense = torch.where(is_s, ta, torch.zeros_like(ta)).sum(-1)  # [B, T]
            Bx = s_dense.shape[0]
            if pos_argmax is not None:
                pa = pos_argmax[:Bx].to(s_dense.device).clamp(0, s_dense.shape[1] - 1)
                val = s_dense[torch.arange(Bx, device=s_dense.device), pa].mean().item()
            else:
                val = s_dense.mean().item()
            a_posctx_buf.append(val)

        # Per-activator target activations at probe positions
        for kind in kinds:
            key = (layer_idx, kind)
            if key not in activator_fids:
                continue
            act = activations[kind_to_idx[kind]]
            ta, ti = sae_bank.encode(act, kind, layer_idx)
            Bx, Tx = ta.shape[:2]
            targets = activator_targets.setdefault(key, {})
            for latent_idx in activator_fids[key]:
                is_t = (ti == latent_idx)
                t_dense = torch.where(is_t, ta, torch.zeros_like(ta)).sum(-1)  # [B, T]
                if pos_argmax is not None:
                    pa = pos_argmax[:Bx].to(t_dense.device).clamp(0, Tx - 1)
                    tval = t_dense[torch.arange(Bx, device=t_dense.device), pa].mean().item()
                else:
                    tval = t_dense.mean().item()
                targets[latent_idx] = tval

    inference.disable_compile()
    try:
        inference.forward(
            pos_tokens,
            activations_callback=posctx_hook,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()

    a_posctx = a_posctx_buf[0] if a_posctx_buf else 0.0

    # ── Pass 2: baseline negctx (no intervention) ─────────────────────────
    # Also collects per-inhibitor negctx mean activations — these are the
    # injection targets used in Pass 4 (posctx suppression).
    neg_B = neg_tokens.shape[0]
    neg_argmax = pos_argmax[:neg_B] if pos_argmax is not None else None
    a_baseline_buf: List[float] = []
    inhibitor_targets: Dict[Tuple[int, str], Dict[int, float]] = {}

    def baseline_hook(layer_idx: int, activations: tuple) -> None:
        # Seed baseline activation on negctx
        if layer_idx == seed_layer and seed_kind in kind_to_idx:
            act = activations[kind_to_idx[seed_kind]]
            ta, ti = sae_bank.encode(act, seed_kind, layer_idx)
            is_s = (ti == seed_latent_idx)
            s_dense = torch.where(is_s, ta, torch.zeros_like(ta)).sum(-1)
            Bx = s_dense.shape[0]
            if neg_argmax is not None:
                actual_B = min(Bx, neg_argmax.shape[0])
                pa = neg_argmax[:actual_B].to(s_dense.device).clamp(0, s_dense.shape[1] - 1)
                val = s_dense[:actual_B][torch.arange(actual_B, device=s_dense.device), pa].mean().item()
            else:
                val = s_dense.mean().item()
            a_baseline_buf.append(val)

        # Per-inhibitor target activations at negctx probe positions
        for kind in kinds:
            key = (layer_idx, kind)
            if key not in inhibitor_fids:
                continue
            act = activations[kind_to_idx[kind]]
            ta, ti = sae_bank.encode(act, kind, layer_idx)
            Bx, Tx = ta.shape[:2]
            targets = inhibitor_targets.setdefault(key, {})
            for latent_idx in inhibitor_fids[key]:
                is_t = (ti == latent_idx)
                t_dense = torch.where(is_t, ta, torch.zeros_like(ta)).sum(-1)
                if neg_argmax is not None:
                    actual_B = min(Bx, neg_argmax.shape[0])
                    pa = neg_argmax[:actual_B].to(t_dense.device).clamp(0, Tx - 1)
                    tval = t_dense[:actual_B][torch.arange(actual_B, device=t_dense.device), pa].mean().item()
                else:
                    tval = t_dense.mean().item()
                targets[latent_idx] = tval

    inference.disable_compile()
    try:
        inference.forward(
            neg_tokens,
            activations_callback=baseline_hook,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()

    a_baseline = a_baseline_buf[0] if a_baseline_buf else 0.0

    # ── Pass 3: intervened negctx ─────────────────────────────────────────
    patcher = CounterfactualInterventionPatcher(
        bank=sae_bank,
        activator_targets=activator_targets,
        inhibitor_indices=inhibitor_fids,
        seed_layer=seed_layer,
        seed_kind=seed_kind,
        seed_latent_idx=seed_latent_idx,
        pos_argmax=neg_argmax,
        circuit_layers=circuit_layers,
    )

    inference.disable_compile()
    try:
        inference.forward(
            neg_tokens,
            patcher=patcher,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()

    a_intervened_neg = patcher.captured_activation or 0.0

    # ── Pass 4: intervened posctx (suppress activators, inject inhibitors) ──
    # Swap roles relative to Pass 3:
    #   activators → inhibitor_indices (suppress to 0)
    #   inhibitors → activator_targets (inject at negctx mean values)
    activator_suppress: Dict[Tuple[int, str], Set[int]] = {
        key: set(idxs) for key, idxs in activator_fids.items()
    }
    reverse_patcher = CounterfactualInterventionPatcher(
        bank=sae_bank,
        activator_targets=inhibitor_targets,
        inhibitor_indices=activator_suppress,
        seed_layer=seed_layer,
        seed_kind=seed_kind,
        seed_latent_idx=seed_latent_idx,
        pos_argmax=pos_argmax,
        circuit_layers=circuit_layers,
    )

    inference.disable_compile()
    try:
        inference.forward(
            pos_tokens,
            patcher=reverse_patcher,
            return_activations=False,
            tokenize_final=False,
        )
    finally:
        inference.enable_compile()

    a_intervened_pos = reverse_patcher.captured_activation or 0.0

    # ── Scores ────────────────────────────────────────────────────────────
    denom = a_posctx - a_baseline
    print(
        f"  [CFaithfulness] a_posctx: {a_posctx:.4f} | "
        f"a_baseline: {a_baseline:.4f} | "
        f"a_intervened_neg: {a_intervened_neg:.4f} | "
        f"a_intervened_pos: {a_intervened_pos:.4f} | "
        f"denom: {denom:.4f}"
    )
    sys.stdout.flush()

    if abs(denom) < 1e-9:
        cf_score = 1.0 if abs(a_intervened_neg - a_posctx) < 1e-9 else 0.0
        sup_score = 1.0 if abs(a_intervened_pos - a_baseline) < 1e-9 else 0.0
        print(f"  [CFaithfulness] cf={cf_score:.4f}  sup={sup_score:.4f}  (small denom)")
        sys.stdout.flush()
        return float(cf_score), float(sup_score)

    cf_score = (a_intervened_neg - a_baseline) / denom
    sup_score = (a_posctx - a_intervened_pos) / denom
    print(f"  [CFaithfulness] cf={cf_score:.4f}  sup={sup_score:.4f}")
    sys.stdout.flush()
    return float(cf_score), float(sup_score)
