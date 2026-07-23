"""Effect-threshold prune — the fixed-cut stopping rule, SFC-inspired.

Keeps a member iff ``|score| > threshold``; drops the rest. This is the
stopping rule of Marks et al.'s node selection (keep iff |IE| > T_N, default
0.1), transplanted onto our circuits as a third post-hoc prune alongside
recurrence and magnitude-bisection.

The scientifically useful part is what it CONTROLS FOR: our magnitude prune
(eval/magnitude_prune.py) already ranks members by |attribution_score| — the
same ranking an effect threshold uses. The two prunes therefore differ ONLY in
the stopping rule:

  * magnitude-bisection — adaptive: smallest top-K prefix that still meets a
    validated faithfulness floor (~log2(N) forward passes).
  * effect threshold   — fixed: everything above T survives, faithfulness is
    whatever falls out (ZERO forward passes).

Running both, and both chained (threshold after bisection), isolates the value
of faithfulness-validated stopping — with the ranking held identical.

Scale caveat: T is in the units of the circuit's own scores. Our IG scores are
not calibrated to Marks et al.'s IE scale, so T=0.1 (their default) is a
starting point, not an equivalence; ``threshold_mode="pctl"`` sidesteps scale
entirely by cutting at a percentile of the circuit's own |score| distribution.
Score distribution quantiles are stamped into circuit.metadata either way, so a
run's thresholds can be calibrated afterwards from any single row.
"""

from __future__ import annotations

from typing import Any, List

__all__ = ["prune_by_effect_threshold"]


def _member_score(node: Any) -> float:
    """|score| under the same fallback chain the runners' members_of uses."""
    sc = node.metadata.get("effect_score")
    if sc is None:
        sc = node.metadata.get("attribution_score")
    if sc is None:
        sc = node.metadata.get("weight") or 0.0
    return abs(float(sc))


def prune_by_effect_threshold(
    circuit: Any,
    *,
    threshold: float = 0.1,
    threshold_mode: str = "abs",
    min_keep: int = 1,
    logger: Any = None,
) -> List[str]:
    """Prune ``circuit`` in place to the members with ``|score| > T``. Returns
    removed uuids. Needs no model — no forward passes are run.

    ``threshold_mode``:
      * ``"abs"``  — T = ``threshold`` in raw score units (SFC's rule; their
        node default is 0.1 in IE units — see the scale caveat above).
      * ``"pctl"`` — T = the ``threshold``-th percentile of the circuit's own
        |score| distribution (0..100), i.e. keep the top (100-threshold)%.

    ``min_keep``: if the cut would leave fewer members, the top-``min_keep`` by
    |score| are kept instead (an empty circuit evaluates degenerately, and the
    other prunes hold the same guarantee).

    Chaining: safe after recurrence/magnitude pruning — metadata keys are
    namespaced (``*_effect_prune``) so they never clobber the bisection's
    ``n_members_pre_prune``/``prune_phi_*`` stamps.
    """

    if threshold_mode not in ("abs", "pctl"):
        raise ValueError(
            f"threshold_mode must be 'abs' or 'pctl', got {threshold_mode!r}")

    members = []
    for uuid, node in circuit.nodes.items():
        if node.metadata.get("role") == "seed":
            continue
        if node.feature_id is None:
            continue
        members.append((_member_score(node), uuid))
    n_total = len(members)
    if n_total <= min_keep:
        return []
    members.sort(key=lambda m: m[0], reverse=True)

    scores = [s for s, _ in members]

    def q(p: float) -> float:  # simple percentile on the sorted-desc list
        idx = min(n_total - 1, max(0, round((100.0 - p) / 100.0 * (n_total - 1))))
        return scores[idx]

    cut = q(threshold) if threshold_mode == "pctl" else float(threshold)
    survivors = [(s, u) for s, u in members if s > cut]
    if len(survivors) < min_keep:
        survivors = members[:min_keep]

    keep_uuids = {u for _, u in survivors}
    removed = [u for _, u in members if u not in keep_uuids]

    # Namespaced stamps — never clobber the bisection prune's metadata.
    circuit.metadata["n_members_pre_effect_prune"] = n_total
    circuit.metadata["effect_prune_threshold"] = float(cut)
    circuit.metadata["effect_prune_mode"] = threshold_mode
    # Distribution quantiles in raw units, for post-hoc threshold calibration.
    circuit.metadata["effect_prune_score_q"] = {
        "p50": q(50.0), "p90": q(90.0), "p99": q(99.0), "max": scores[0],
    }
    if not removed:
        return []
    rm = set(removed)
    for uuid in removed:
        circuit.nodes.pop(uuid, None)
    circuit.edges = [
        e for e in circuit.edges
        if e.source_uuid not in rm and e.target_uuid not in rm
    ]
    if logger is not None:
        logger.stage(
            "after effect-threshold prune",
            len(circuit.nodes), len(circuit.edges),
            note=(f"kept {len(survivors)}/{n_total} with |score| > {cut:.4g} "
                  f"({threshold_mode}), removed {len(removed)}"),
        )
    return removed
