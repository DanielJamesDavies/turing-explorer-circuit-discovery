"""Cross-sequence recurrence prune: drop members that fire on only one input.

Position-aware membership is a union over the causal prefix, so a latent joins
the circuit if it is selected *anywhere* — including on a single probe sequence.
Measured over 14 circuits (arm abl-ig_mean PA, 64 probe sequences), members
divide sharply by how many sequences they actually fire in:

    in ALL sequences   :  0.9% of members -> 43.0% of attribution mass
    in >50%            :  6.1% of members -> 77.8% of mass
    in exactly ONE     :  9.3% of members ->  0.4% of mass

The single-sequence tail is per-input scaffolding: ~10% of nodes carrying ~0.4%
of mass. Removing it costs a median 0.010 free-phi (worst 0.036; 11/14 circuits
under 0.02) and holds at every depth band — L0-3 0.949->0.938, L4-7
0.951->0.942, L8-11 0.949->0.938.

Unlike the magnitude prune this needs no bisection and no sufficiency search:
recurrence is a property of the probe pass, so ONE forward pass answers it for
every member at once. It composes with (and runs before) the magnitude prune,
which then bisects over a smaller ranked set.

Deliberately conservative by default (``min_sequences=2``, i.e. drop only the
exactly-one tail). Higher thresholds compress much harder but cost real closure
— keeping members present in >=8 of 64 sequences retains 34.5% of nodes at
free-phi 0.839, and deep seeds degrade faster than shallow ones (L8+ falls to
0.560 there against 0.872 for L0-3), so raising it is a per-depth trade rather
than a free win.

ROLE-SPLIT MEASUREMENT
----------------------
Members are judged on the distribution they act on, not on one shared pass:

    supports / activators  ->  recurrence over POSCTX
    inhibitors             ->  recurrence over NEGCTX

Roughly half of a PA circuit's members are ``counterfactual_inhibitor`` —
latents present on negctx that suppress the seed. Counting those over posctx
would penalise them for doing exactly what they are for, and the ordering makes
that consequential: both prunes run BEFORE the counterfactual faithfulness eval,
and cf faithfulness (measured on negctx) is the acceptance gate for cf methods,
so mis-pruning inhibitors can change which circuits are ACCEPTED rather than
merely how large they are. When negctx is unavailable the inhibitors are exempt
rather than judged on the wrong pass.

SCOPE OF THE EVIDENCE
---------------------
The percentages and costs above were measured on ``abl-ig_mean PA`` circuits,
under a posctx-only count, before the role split existed. They should be
re-measured per arm — particularly on the counterfactual family, whose members
are selected on negctx — before the toggle is trusted there.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

Site = Tuple[int, str]


@torch.no_grad()
def collect_sequence_recurrence(
    inference: Any,
    sae_bank: Any,
    tokens: torch.Tensor,
    sites: Set[Site],
) -> Dict[Site, torch.Tensor]:
    """Per site, a [d_sae] count of how many sequences each latent fires in.

    Works from the top-k indices rather than a dense [B, T, d_sae] tensor: only
    "did this latent fire anywhere in this sequence" is needed, so the per-site
    footprint is [B, d_sae] booleans instead of the ~0.7 GB dense stream.
    """

    kind_to_idx = {k: i for i, k in enumerate(sae_bank.kinds)}
    counts: Dict[Site, torch.Tensor] = {}

    def hook(layer_idx: int, activations: tuple) -> None:
        for kind in sae_bank.kinds:
            if (layer_idx, kind) not in sites:
                continue
            act = activations[kind_to_idx[kind]]
            top_acts, top_idx = sae_bank.encode(act, kind, layer_idx)
            batch = top_idx.shape[0]
            seen = torch.zeros(
                batch, sae_bank.d_sae, dtype=torch.bool, device=top_idx.device
            )
            live = top_acts > 0
            if bool(live.any()):
                rows = (
                    torch.arange(batch, device=top_idx.device)
                    .view(batch, 1, 1)
                    .expand_as(top_idx)
                )
                seen[rows[live], top_idx[live]] = True
            # Move to CPU HERE, once per site. Downstream scoring indexes these
            # once per member, and a scalar index into a CUDA tensor forces a
            # device sync each time (~81us measured) - 40s on a 500k-member
            # circuit, dwarfing the forward passes that produce the counts.
            counts[(layer_idx, kind)] = seen.sum(dim=0).detach().cpu()

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

    missing = sites - set(counts)
    if missing:
        raise RuntimeError(f"recurrence counts missing for sites: {sorted(missing)}")
    return counts


#: Roles whose members do their work on the NEGATIVE contexts. An inhibitor is
#: present on negctx and suppresses the seed, so its relevance is how
#: consistently it recurs THERE - counting it over posctx would penalise it for
#: doing exactly what it is supposed to do.
NEGATIVE_ROLES = ("counterfactual_inhibitor",)


@torch.no_grad()
def prune_by_sequence_recurrence(
    inference: Any,
    sae_bank: Any,
    circuit: Any,
    *,
    pos_tokens: torch.Tensor,
    neg_tokens: Optional[torch.Tensor] = None,
    min_sequences: int = 2,
    min_keep: int = 1,
    sides: Tuple[str, ...] = ("pos", "neg"),
    negative_roles: Tuple[str, ...] = NEGATIVE_ROLES,
    logger: Any = None,
) -> List[str]:
    """Prune ``circuit`` in place, dropping members that recur in fewer than
    ``min_sequences`` sequences OF THE DISTRIBUTION THEY ACT ON. Returns the
    removed uuids.

    Members are split by role sign and judged separately:
      * supports / activators -> recurrence over ``pos_tokens``
      * inhibitors            -> recurrence over ``neg_tokens``

    That split is what makes the prune sound for the counterfactual family.
    Judging everything on posctx would measure inhibitors on a distribution
    where they are not expected to fire, and since both prunes run before the
    counterfactual faithfulness eval - the acceptance gate for cf methods -
    mis-pruning them can change which circuits are accepted, not merely their
    size.

    If ``neg_tokens`` is None, inhibitors are EXEMPT rather than judged on the
    wrong distribution: a prune that cannot measure a member correctly should
    leave it alone.

    The seed node is never removed. If the threshold would leave fewer than
    ``min_keep`` members, the most-recurrent members are retained instead (ties
    broken by ``|attribution_score|``), so the prune degrades to a no-op rather
    than emptying a circuit whose members all happen to be input-specific.
    """

    pos_members: List[Tuple[str, Site, int]] = []
    neg_members: List[Tuple[str, Site, int]] = []
    pos_sites: Set[Site] = set()
    neg_sites: Set[Site] = set()
    n_exempt = 0
    for uuid, node in circuit.nodes.items():
        if node.metadata.get("role") == "seed":
            continue
        fid = node.feature_id
        if fid is None:
            n_exempt += 1                     # unscored node: survives untouched
            continue
        site = (fid.layer, fid.kind)
        if node.metadata.get("role") in negative_roles:
            if neg_tokens is None or "neg" not in sides:
                n_exempt += 1                 # exempt: unmeasurable, or side off
                continue
            neg_members.append((uuid, site, int(fid.index)))
            neg_sites.add(site)
        else:
            if "pos" not in sides:
                n_exempt += 1
                continue
            pos_members.append((uuid, site, int(fid.index)))
            pos_sites.add(site)

    members = pos_members + neg_members
    if not members or min_sequences <= 1:
        return []

    counts: Dict[Site, torch.Tensor] = {}
    if pos_sites:
        counts.update({
            ("pos", site): c
            for site, c in collect_sequence_recurrence(
                inference, sae_bank, pos_tokens, pos_sites).items()
        })
    if neg_sites and neg_tokens is not None:
        counts.update({
            ("neg", site): c
            for site, c in collect_sequence_recurrence(
                inference, sae_bank, neg_tokens, neg_sites).items()
        })

    # Gather counts per SITE rather than per member: one vectorised lookup per
    # site instead of one scalar index per member. At 500k members the scalar
    # form measured 40s (GPU) / 0.4s (CPU) against 0.15s vectorised, and the
    # cost is linear in members - invisible on a small circuit, dominant on a
    # real one.
    scored: List[Tuple[str, int, float]] = []
    for side, group in (("pos", pos_members), ("neg", neg_members)):
        by_site: Dict[Site, List[Tuple[str, int]]] = defaultdict(list)
        for uuid, site, index in group:
            by_site[site].append((uuid, index))
        for site, entries in by_site.items():
            table = counts[(side, site)]
            idx = torch.as_tensor([i for _, i in entries], dtype=torch.long)
            n_seqs = table[idx].tolist()
            for (uuid, _), n_seq in zip(entries, n_seqs):
                weight = circuit.nodes[uuid].metadata.get("attribution_score")
                if weight is None:
                    weight = circuit.nodes[uuid].metadata.get("weight") or 0.0
                scored.append((uuid, int(n_seq), abs(float(weight))))

    doomed = {uuid for uuid, n_seq, _ in scored if n_seq < min_sequences}
    # Exempt members still survive, so they count toward min_keep - otherwise the
    # guard would rescue a scored member to "save" a circuit that was never at
    # risk of emptying.
    n_keep = len(members) - len(doomed) + n_exempt
    if n_keep < min_keep:
        # retain the most-recurrent members up to min_keep rather than empty out
        rescue = sorted(
            (s for s in scored if s[0] in doomed),
            key=lambda s: (s[1], s[2]),
            reverse=True,
        )[: min_keep - n_keep]
        doomed -= {uuid for uuid, _, _ in rescue}

    if not doomed:
        if logger is not None:
            logger.note(
                f"recurrence prune: no members below min_sequences={min_sequences}"
            )
        return []

    n_before = len(circuit.nodes)
    for uuid in doomed:
        circuit.nodes.pop(uuid, None)
    circuit.edges = [
        edge
        for edge in circuit.edges
        if edge.source_uuid not in doomed and edge.target_uuid not in doomed
    ]
    circuit.metadata["n_members_pre_recurrence_prune"] = n_before
    circuit.metadata["recurrence_prune_min_sequences"] = min_sequences

    if logger is not None:
        n_pos_cut = sum(1 for uuid, _, _ in pos_members if uuid in doomed)
        n_neg_cut = len(doomed) - n_pos_cut
        neg_note = (
            f"{n_neg_cut} inhibitors of {int(neg_tokens.shape[0])} negctx"
            if neg_tokens is not None else "inhibitors exempt (no negctx)"
        )
        logger.stage(
            "after recurrence prune",
            len(circuit.nodes),
            len(circuit.edges),
            note=(
                f"removed {len(doomed)} members recurring in <{min_sequences} "
                f"sequences: {n_pos_cut} supports of "
                f"{int(pos_tokens.shape[0])} posctx, {neg_note}"
            ),
        )
    return sorted(doomed)
