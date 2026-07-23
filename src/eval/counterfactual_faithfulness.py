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
from sae.dense import sparse_topk_to_dense, target_latent_activations


def _batch_latent_targets(
    ta: torch.Tensor,
    ti: torch.Tensor,
    latent_ids: List[int],
    argmax: Optional[torch.Tensor],
    d_sae: int,
) -> Dict[int, float]:
    """Mean activation at the probe positions for MANY latents in one op.

    Equivalent to looping ``target_latent_activations`` + probe-mean per
    latent, but with a single scatter and a single GPU->CPU sync — the
    per-latent loop launched one kernel AND one blocking .item() per member,
    which at PA-circuit sizes (100k+ members) was ~1 ms/member of pure
    launch/sync overhead (measured: the cf eval scaled linearly to 386 s at
    362k members with GPU utilisation collapsed to ~14%).

    The scatter uses amax reduction against a zero base: identical semantics
    to sparse_topk_to_dense / target_latent_activations (top-k activations
    are non-negative; padded index-0 slots carry 0 and lose the max to any
    genuine activation).
    """
    Bx, Tx = ta.shape[:2]
    device = ta.device
    if argmax is not None:
        actual_B = min(Bx, argmax.shape[0])
        pa = argmax[:actual_B].to(device).clamp(0, Tx - 1)
        rows = torch.arange(actual_B, device=device)
        rows_a = ta[:actual_B][rows, pa]  # [B, k]
        rows_i = ti[:actual_B][rows, pa]  # [B, k]
    else:
        rows_a = ta.reshape(-1, ta.shape[-1])
        rows_i = ti.reshape(-1, ti.shape[-1])
    dense = torch.zeros(rows_a.shape[0], d_sae, device=device, dtype=torch.float32)
    dense.scatter_reduce_(1, rows_i.long(), rows_a.to(torch.float32),
                          reduce="amax", include_self=True)
    idx = torch.tensor(latent_ids, device=device, dtype=torch.long)
    vals = dense[:, idx].mean(dim=0)
    return {int(l): float(v) for l, v in zip(latent_ids, vals.cpu().tolist())}


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
        capture_max: bool = False,
    ) -> None:
        self.bank = bank
        self.activator_targets = activator_targets
        self.inhibitor_indices = inhibitor_indices
        self.seed_layer = seed_layer
        self.seed_kind = seed_kind
        self.seed_latent_idx = seed_latent_idx
        self.pos_argmax = pos_argmax.detach().cpu() if pos_argmax is not None else None
        self.circuit_layers = circuit_layers
        self.capture_max = capture_max
        self.captured_activation: Optional[float] = None
        # Max-over-positions capture (set when capture_max): "did the seed fire
        # ANYWHERE?" — diagnostic for anchor placement, never a score input.
        self.captured_activation_max: Optional[float] = None
        # Precomputed per-site index/value tensors so the intervention is one
        # advanced-index write per site instead of one kernel launch per
        # member (the per-latent loop was ~1 ms/member at PA-circuit sizes).
        self._act_tensors: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        for key, targets in activator_targets.items():
            if targets:
                self._act_tensors[key] = (
                    torch.tensor(list(targets.keys()), dtype=torch.long),
                    torch.tensor(list(targets.values()), dtype=torch.float32),
                )
        self._inh_tensors: Dict[Tuple[int, str], torch.Tensor] = {
            key: torch.tensor(sorted(idxs), dtype=torch.long)
            for key, idxs in inhibitor_indices.items() if idxs
        }

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
            seed_dense = target_latent_activations(top_acts, top_indices, self.seed_latent_idx)  # [B, T]
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
            if self.capture_max:
                self.captured_activation_max = seed_dense.max(dim=1).values.mean().item()
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
        all_latents = sparse_topk_to_dense(top_acts, top_indices, self.bank.d_sae, dtype=target_dtype)

        # Preserve the SAE error term from the unmodified encoding
        full_recon = self.bank.decode(all_latents, kind, layer_idx)
        error = x - full_recon

        # Apply the interventions to a copy of the latent tensor — one
        # advanced-index write per role per site (vals [n] broadcasts over
        # [B, T, n]), not one kernel per member.
        patched = all_latents.clone()
        act_pair = self._act_tensors.get((layer_idx, kind))
        if act_pair is not None:
            act_idx, act_vals = act_pair
            patched[:, :, act_idx.to(x.device)] = act_vals.to(x.device, target_dtype)
        inh_idx = self._inh_tensors.get((layer_idx, kind))
        if inh_idx is not None:
            patched[:, :, inh_idx.to(x.device)] = 0.0

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
    anchor_mode: str = "legacy",
    return_details: bool = False,
):
    """
    Measure how well the discovered activators and inhibitors causally explain
    the seed's firing behaviour in both directions.

    ``anchor_mode`` selects where the negctx-side seed measurement is taken:

      * ``"legacy"`` (default, byte-identical to the historical eval): the
        posctx argmax positions are reused on the negctx sequences —
        ``neg_argmax = pos_argmax[:B_neg]``. Position i's probe index came
        from a DIFFERENT sequence, so on negctx it is an arbitrary position.
        This is the position-collapse defect that censors cf at depth (a
        circuit with free0 = 0.988 scored cf = 0.039 at L9): deep seeds read
        specific positions, and the measurement looks at the wrong one.
      * ``"negctx_preact"``: per-sequence anchors at the seed's own
        WOULD-BE-FIRING position on each negctx sequence — the argmax of its
        pre-activation (w_seed·x + b_seed) on the natural negctx run, computed
        in Pass 2 at no extra forward cost. The same anchor ig_negctx
        integrates at (counterfactual_gradient._negctx_anchor). a_baseline is
        measured at the same anchors, so numerator and denominator share their
        positions. Also captures max-over-positions of the intervened run
        (``a_intervened_neg_maxpos``) as an anchor-placement diagnostic.
        Inhibitor injection TARGETS (pass 2 → pass 4) keep legacy positions:
        they are collected at upstream sites before the seed's layer runs, so
        the anchors do not exist yet when they are needed — and they feed the
        posctx-side score, which has correct anchors already.

    ``return_details``: when True, returns ``(cf, sup, details)`` where
    details carries every anchor (a_posctx, a_baseline, a_intervened_neg,
    a_intervened_pos, denom, anchor_mode, and in anchored mode
    a_intervened_neg_maxpos) plus ``cf_bounded`` =
    1 − |a_intervened_neg − a_posctx| / |denom| — a variant that treats
    overshoot as error rather than success (cf rose 1.63→2.49 at L10 while
    free0 collapsed to 0.005; the unbounded ratio rewards uncompensated
    drive). Logged so scores can be recomputed post hoc without a rerun.

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
    if anchor_mode not in ("legacy", "negctx_preact"):
        raise ValueError(
            f"anchor_mode must be 'legacy' or 'negctx_preact', got {anchor_mode!r}")
    anchored = anchor_mode == "negctx_preact"

    kinds = sae_bank.kinds
    kind_to_idx: Dict[str, int] = {k: i for i, k in enumerate(kinds)}

    # Anchored mode: the seed's encoder row, for the pre-activation argmax.
    w_seed = b_seed = None
    if anchored:
        sae = sae_bank.saes[seed_kind][seed_layer]
        w_seed = sae.encoder.weight[seed_latent_idx].detach()
        b_seed = sae._get_bias_eff()[seed_latent_idx].detach()

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
        if role in ("counterfactual_activator", "ablation_support"):
            activator_fids.setdefault(key, []).append(fid.index)
        elif role == "counterfactual_inhibitor":
            inhibitor_fids.setdefault(key, set()).add(fid.index)

    n_act = sum(len(v) for v in activator_fids.values())
    n_inh = sum(len(v) for v in inhibitor_fids.values())

    if n_act == 0 and n_inh == 0:
        print("  [CFaithfulness] No activator or inhibitor nodes — returning (0.0, 0.0)")
        sys.stdout.flush()
        if return_details:
            return 0.0, 0.0, {"anchor_mode": anchor_mode, "empty_circuit": True}
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
            s_dense = target_latent_activations(ta, ti, seed_latent_idx)  # [B, T]
            Bx = s_dense.shape[0]
            if pos_argmax is not None:
                pa = pos_argmax[:Bx].to(s_dense.device).clamp(0, s_dense.shape[1] - 1)
                val = s_dense[torch.arange(Bx, device=s_dense.device), pa].mean().item()
            else:
                val = s_dense.mean().item()
            a_posctx_buf.append(val)

        # Per-activator target activations at probe positions — one scatter +
        # one sync per site for ALL of the site's activators.
        for kind in kinds:
            key = (layer_idx, kind)
            if key not in activator_fids:
                continue
            act = activations[kind_to_idx[kind]]
            ta, ti = sae_bank.encode(act, kind, layer_idx)
            activator_targets.setdefault(key, {}).update(
                _batch_latent_targets(ta, ti, activator_fids[key], pos_argmax,
                                      sae_bank.d_sae)
            )

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
    neg_anchors_buf: List[torch.Tensor] = []
    inhibitor_targets: Dict[Tuple[int, str], Dict[int, float]] = {}

    def baseline_hook(layer_idx: int, activations: tuple) -> None:
        # Seed baseline activation on negctx
        if layer_idx == seed_layer and seed_kind in kind_to_idx:
            act = activations[kind_to_idx[seed_kind]]
            ta, ti = sae_bank.encode(act, seed_kind, layer_idx)
            s_dense = target_latent_activations(ta, ti, seed_latent_idx)
            Bx = s_dense.shape[0]
            if anchored:
                # Per-sequence would-be-firing anchor: pre-activation argmax on
                # the natural negctx run. Baseline is measured AT the anchors so
                # numerator and denominator share their positions.
                pre = act.float() @ w_seed.to(act.device).float() + float(b_seed)  # [B, T]
                anchors = pre.argmax(dim=-1)                                       # [B]
                neg_anchors_buf.append(anchors.detach().cpu())
                val = s_dense[torch.arange(Bx, device=s_dense.device),
                              anchors.to(s_dense.device)].mean().item()
            elif neg_argmax is not None:
                actual_B = min(Bx, neg_argmax.shape[0])
                pa = neg_argmax[:actual_B].to(s_dense.device).clamp(0, s_dense.shape[1] - 1)
                val = s_dense[:actual_B][torch.arange(actual_B, device=s_dense.device), pa].mean().item()
            else:
                val = s_dense.mean().item()
            a_baseline_buf.append(val)

        # Per-inhibitor target activations at negctx probe positions — one
        # scatter + one sync per site for ALL of the site's inhibitors.
        for kind in kinds:
            key = (layer_idx, kind)
            if key not in inhibitor_fids:
                continue
            act = activations[kind_to_idx[kind]]
            ta, ti = sae_bank.encode(act, kind, layer_idx)
            inhibitor_targets.setdefault(key, {}).update(
                _batch_latent_targets(ta, ti, sorted(inhibitor_fids[key]), neg_argmax,
                                      sae_bank.d_sae)
            )

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
    neg_anchors = neg_anchors_buf[0] if neg_anchors_buf else None
    if anchored and neg_anchors is None:
        # The hook never reached the seed's layer — surface it rather than
        # silently measuring at legacy positions under an "anchored" label.
        raise RuntimeError("anchor_mode='negctx_preact': no anchors captured "
                           "(seed layer not reached in the negctx pass)")

    # ── Pass 3: intervened negctx ─────────────────────────────────────────
    patcher = CounterfactualInterventionPatcher(
        bank=sae_bank,
        activator_targets=activator_targets,
        inhibitor_indices=inhibitor_fids,
        seed_layer=seed_layer,
        seed_kind=seed_kind,
        seed_latent_idx=seed_latent_idx,
        pos_argmax=neg_anchors if anchored else neg_argmax,
        circuit_layers=circuit_layers,
        capture_max=anchored,
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
        f"denom: {denom:.4f} | anchor_mode: {anchor_mode}"
    )
    sys.stdout.flush()

    def _details(cf_score: float, sup_score: float) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "anchor_mode": anchor_mode,
            "a_posctx": float(a_posctx),
            "a_baseline": float(a_baseline),
            "a_intervened_neg": float(a_intervened_neg),
            "a_intervened_pos": float(a_intervened_pos),
            "denom": float(denom),
            "cf": float(cf_score),
            "sup": float(sup_score),
            # Overshoot-as-error variant: 1 at perfect restoration, falls off
            # symmetrically in both directions. The raw ratio rewards
            # uncompensated drive (cf can RISE as a circuit is gutted).
            "cf_bounded": (1.0 - abs(a_intervened_neg - a_posctx) / abs(denom)
                           if abs(denom) > 1e-9 else None),
        }
        if patcher.captured_activation_max is not None:
            d["a_intervened_neg_maxpos"] = float(patcher.captured_activation_max)
        return d

    if abs(denom) < 1e-9:
        cf_score = 1.0 if abs(a_intervened_neg - a_posctx) < 1e-9 else 0.0
        sup_score = 1.0 if abs(a_intervened_pos - a_baseline) < 1e-9 else 0.0
        print(f"  [CFaithfulness] cf={cf_score:.4f}  sup={sup_score:.4f}  (small denom)")
        sys.stdout.flush()
        if return_details:
            return float(cf_score), float(sup_score), _details(cf_score, sup_score)
        return float(cf_score), float(sup_score)

    cf_score = (a_intervened_neg - a_baseline) / denom
    sup_score = (a_posctx - a_intervened_pos) / denom
    print(f"  [CFaithfulness] cf={cf_score:.4f}  sup={sup_score:.4f}")
    sys.stdout.flush()
    if return_details:
        return float(cf_score), float(sup_score), _details(cf_score, sup_score)
    return float(cf_score), float(sup_score)
