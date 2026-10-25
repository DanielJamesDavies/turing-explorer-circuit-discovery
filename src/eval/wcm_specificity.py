"""Protocol-v1 specificity diagnostic, amplifier flag and near-threshold stratum (DAN-73; protocol-v1.md §6, §7.1).

The 062 run's ``experiments/056-specificity/specificity.py: score`` and the per-target flags of
``experiments/062-h100-protocol-v1/merge.py`` / ``pass_rule.py``, moved into src with an explicit interface.

Specificity asks whether a circuit is a mechanism for its target or a concept amplifier: it reads the WHOLE code at
the target's site (every latent's pre-activation and the SAE's Top-K set, at the target's anchor) on the held-out
activating contexts, in three runs per ablation method pi in {Z, A, C} (A and C sparsity-preserving, fitted alpha):
clean (unmodified), empty (every upstream latent ablated per pi) and circuit. Per (circuit, pi) row: the target's
rank among all site latents (1 = top), Top-K share, Jaccard with the clean Top-K, latents switched on above the clean
Top-K cut, and the faithfulness of the target and of its siblings (latents in the clean Top-K at the anchor on
>= sib_frac of the TRAIN contexts) in the target's own form on the pre-activation.

  amplifier flag   target_faith_pre >= 0.5 AND sibling_faith_median >= 0.8 x target_faith_pre (per pi);
                   amp_any = the flag under any pi
  near-threshold   rank_clean >= k / 2, k = the dictionary's Top-K k (per model: 128 TuringLLM, 32 GPT-2 small,
                   16 Pythia-70m). rank_clean = the median over the held-out anchors of the clean rank, uncapped
                   (a different "clean rank" from the fit's rank-keep, which ranks among the OTHER latents on
                   training anchors and clamps at k; protocol-v1.md §10.3.4)
"""

from __future__ import annotations

import contextlib
import io
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from eval.ablation_faithfulness import CircuitOnlyPatcher, upstream_sites
from eval.floors import collect_site_anchors, collect_site_means
from eval.wcm_eval import EVAL_BATCH, HOLDOUT_FRAC, N_SEQ, _num, circuit_alphas, contrast_split, eval_split, seed_of, \
    site_encoder
from model.hooks import multi_patch

Site = Tuple[int, str]
PIS = ("Z", "A", "C")
SIB_FRAC = 0.5                   # sibling: in the clean Top-K at the anchor on >= half the train contexts
MIN_DEN = 0.05                   # a latent's faithfulness needs (clean - empty) >= MIN_DEN x clean
AMP_MIN_TARGET = 0.5             # amplifier flag: target faithfulness at least this ...
AMP_SIBLING_RATIO = 0.8          # ... and siblings restored to at least this fraction of it


class _SiteTap:
    """Mixin: at the target site, capture every latent's pre-activation and the SAE's Top-K set at the anchors.
    The stream there passes through untouched (the target site is never in the ablation scope)."""

    def capture(self, x: torch.Tensor) -> None:
        bank = self.bank
        B = x.shape[0]
        rr = torch.arange(B, device=x.device)
        anc = self.anchors.to(x.device).clamp(0, x.shape[1] - 1)
        xa = x[rr, anc].detach()                                                  # [B, d_model]
        W, b = site_encoder(bank, self.site[0], self.site[1])
        self.pre_all = (xa @ W.to(xa.dtype).T + b.to(xa.dtype)).float()          # [B, D], the target tap's form
        with torch.no_grad():
            ta, ti = bank.encode(xa.unsqueeze(1), self.site[1], self.site[0])    # the SAE's own Top-K
        ta, ti = ta[:, 0], ti[:, 0]
        self.topk_sets = [set(int(i) for i, a in zip(ti[j].tolist(), ta[j].tolist()) if a > 0) for j in range(B)]


class CleanSiteTap(_SiteTap):
    def __init__(self, bank: Any, site: Site, anchors: torch.Tensor):
        self.bank, self.site, self.anchors = bank, site, anchors

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if (layer_idx, kind) == self.site:
            self.capture(x)
        return x


class CircuitSiteTap(CircuitOnlyPatcher, _SiteTap):
    def __init__(self, *a, anchors: torch.Tensor = None, **k):
        super().__init__(*a, **k)
        self.site, self.anchors = (self.seed_layer, self.seed_kind), anchors

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if (layer_idx, kind) == self.site:
            self.capture(x)
        return super().transform(layer_idx, kind, x)


def _run(inference: Any, make, tokens: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, List[Set[int]]]:
    """Forward ``tokens`` in chunks; returns (pre_all [N, D] on CPU, the Top-K set per row)."""
    pres, sets = [], []
    inference.disable_compile()
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            for s0 in range(0, int(tokens.shape[0]), batch_size):
                p = make(s0, min(s0 + batch_size, int(tokens.shape[0])))
                inference.forward(tokens[s0:s0 + batch_size], patcher=p, grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                pres.append(p.pre_all.cpu())
                sets.extend(p.topk_sets)
    finally:
        inference.enable_compile()
    return torch.cat(pres), sets


def latent_faith(circ: torch.Tensor, empty: torch.Tensor, clean: torch.Tensor,
                 min_den: float = MIN_DEN) -> torch.Tensor:
    """Per-latent faithfulness on the pre-activation, target form (relu c - relu e) / (relu n - relu e), averaged
    over anchors first; NaN where the denominator is below min_den x clean."""
    c, e, n = (torch.relu(t).mean(0) for t in (circ, empty, clean))
    den = n - e
    out = (c - e) / den
    out[den < min_den * n.clamp(min=1e-9)] = float("nan")
    return out


@torch.no_grad()
def specificity_rows(
    inference: Any,
    bank: Any,
    circuit: Any,
    probe: Any,
    contrast_tokens: Optional[torch.Tensor] = None,
    *,
    arm: Optional[str] = None,
    split: str = "explicit",
    batch_size: int = EVAL_BATCH,
    n_seq: int = N_SEQ,
    topk: Optional[int] = None,
    sib_frac: float = SIB_FRAC,
    min_den: float = MIN_DEN,
    holdout_frac: float = HOLDOUT_FRAC,
) -> List[Dict[str, Any]]:
    """One row per pi in (Z, A, C) for a circuit on one eval set (the run: held-out strongest). Arguments as
    wcm_eval.score_circuit; ``arm`` is a label written to each row (the run: the training arm, "A" or "B");
    ``topk`` = the sparsity-preserving fill budget, default the dictionary's own k. The split rule is applied to the
    contrast rows too (C means from the train contrast rows); on held-out strongest both split modes agree."""
    K = int(bank.k) if topk is None else int(topk)
    D = int(bank.d_sae)
    seed = seed_of(circuit)
    sf = seed.metadata["feature_id"]
    layer, kind, sl = sf.layer, sf.kind, sf.index
    key = "%d.%s.%d" % (layer, kind, sl)
    site = (layer, kind)
    alphas = circuit_alphas(circuit)
    pt, pa = probe.pos_tokens[:n_seq], probe.pos_argmax[:n_seq]
    nt = (probe.neg_tokens if contrast_tokens is None else contrast_tokens)[:n_seq]
    n_tr = eval_split(int(pt.shape[0]), probe.metadata, split, holdout_frac)
    n_ntr = contrast_split(int(nt.shape[0]), probe.metadata, holdout_frac, own=contrast_tokens is None)
    pt_tr, pa_tr, pt_ho, pa_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]
    UPS = set(upstream_sites(bank, layer, kind))
    dev = pt.device
    means = {"Z": None,
             "A": collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)[0],
             "C": collect_site_means(inference, bank, nt[:n_ntr], UPS)}
    keep = {s: set(d) for s, d in alphas.items() if d}
    kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in keep.items()}
    scales = {}
    for s, d in alphas.items():
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[kt[s]] = torch.tensor([d[int(i)] for i in kt[s].tolist()], device=dev, dtype=torch.float32)
        scales[s] = sv

    def circuit_run(keep_, pi, tokens, anchors):
        return _run(inference, lambda a, b: CircuitSiteTap(
            bank=bank, keep_indices=keep_, in_scope=UPS, seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
            pos_argmax=anchors[a:b], site_means=means[pi], respect_topk=(pi != "Z"), topk=K,
            keep_tensors={s: kt[s] for s in keep_}, keep_scales=(scales if keep_ else None), anchors=anchors[a:b]),
            tokens, batch_size)

    # siblings: co-fire with the target in the clean Top-K at the anchor on >= sib_frac of the TRAIN contexts
    _, tr_sets = _run(inference, lambda a, b: CleanSiteTap(bank, site, pa_tr[a:b]), pt_tr, batch_size)
    cnt: Dict[int, int] = defaultdict(int)
    for s_ in tr_sets:
        for j in s_:
            if j != sl:
                cnt[j] += 1
    siblings = sorted(j for j, n_ in cnt.items() if n_ >= sib_frac * len(tr_sets))

    clean_pre, clean_sets = _run(inference, lambda a, b: CleanSiteTap(bank, site, pa_ho[a:b]), pt_ho, batch_size)
    clean_active = set().union(*clean_sets) - {sl}
    control = sorted(clean_active - set(siblings))
    tau_clean = torch.tensor([min(float(clean_pre[j, i]) for i in s_) if s_ else float("inf")
                              for j, s_ in enumerate(clean_sets)])                  # clean Top-K cut per anchor

    def rank_of(pre):                                                               # 1 = highest pre-activation
        return (pre > pre[:, sl:sl + 1]).sum(1).float() + 1

    def med(t):
        return float(np.nanmedian(t.numpy())) if t.numel() and not torch.isnan(t).all() else None

    def n_ok(t):
        return int((~torch.isnan(t)).sum()) if t.numel() else 0

    def jac(A, B):
        return float(np.mean([len(a & b) / max(1, len(a | b)) for a, b in zip(A, B)]))

    rows = []
    for pi in PIS:
        empty_pre, empty_sets = circuit_run({}, pi, pt_ho, pa_ho)
        circ_pre, circ_sets = circuit_run(keep, pi, pt_ho, pa_ho)
        others = torch.ones(D, dtype=torch.bool)
        others[sl] = False
        switched = ((circ_pre > tau_clean[:, None]) & (clean_pre <= tau_clean[:, None]) & others[None]).sum(1).float()
        switched_empty = ((empty_pre > tau_clean[:, None]) & (clean_pre <= tau_clean[:, None])
                          & others[None]).sum(1).float()
        f_all = latent_faith(circ_pre, empty_pre, clean_pre, min_den)
        tf = float(f_all[sl])
        sib_f = f_all[siblings] if siblings else torch.tensor([])
        ctl_f = f_all[control] if control else torch.tensor([])
        rows.append(dict(
            arm=arm, seed=key, pi=pi, layer=layer, kind=kind, n_nodes=sum(len(d) for d in alphas.values()), K=K,
            a_clean_pre=float(torch.relu(clean_pre[:, sl]).mean()),
            target_faith_pre=None if np.isnan(tf) else tf,
            rank_clean=float(rank_of(clean_pre).median()), rank_circuit=float(rank_of(circ_pre).median()),
            rank_empty=float(rank_of(empty_pre).median()),
            in_topk_clean=float(np.mean([sl in s_ for s_ in clean_sets])),
            in_topk_circuit=float(np.mean([sl in s_ for s_ in circ_sets])),
            jaccard_circuit=jac(circ_sets, clean_sets), jaccard_empty=jac(empty_sets, clean_sets),
            n_new_circuit=float(np.mean([len(a - b) for a, b in zip(circ_sets, clean_sets)])),
            n_new_empty=float(np.mean([len(a - b) for a, b in zip(empty_sets, clean_sets)])),
            switched_on_circuit=float(switched.median()), switched_on_empty=float(switched_empty.median()),
            n_siblings=len(siblings), n_siblings_scored=n_ok(sib_f), sibling_faith_median=med(sib_f),
            n_control=len(control), n_control_scored=n_ok(ctl_f), control_faith_median=med(ctl_f),
            specificity_gap=(None if np.isnan(tf) or med(sib_f) is None else tf - med(sib_f)),
            siblings=siblings[:50]))
    return rows


# ----------------------------------------------------------------------------------------------- per-target flags
def amplifier_flag(row: Mapping[str, Any], min_target: float = AMP_MIN_TARGET,
                   sibling_ratio: float = AMP_SIBLING_RATIO) -> bool:
    """Concept-amplifier flag of one (circuit, pi) row: the target is restored (>= min_target) and its siblings
    nearly as well (>= sibling_ratio x the target). A missing value does not flag."""
    tf, sib = _num(row.get("target_faith_pre")), _num(row.get("sibling_faith_median"))
    return tf is not None and sib is not None and tf >= min_target and sib >= sibling_ratio * tf


def _median(vals: Iterable[Any]) -> Optional[float]:
    v = [x for x in (_num(x) for x in vals) if x is not None]
    return float(np.median(v)) if v else None


def near_threshold(rank_clean: Optional[float], k: int) -> bool:
    """The near-threshold stratum: the target's clean rank sits in the bottom half of the Top-K budget
    (rank_clean >= k / 2). A target without a rank is not in the stratum."""
    r = _num(rank_clean)
    return r is not None and r >= int(k) / 2


def target_specificity(rows: Sequence[Mapping[str, Any]], k: Optional[int] = None) -> Dict[str, Any]:
    """merge.py's per-target summary of one target's specificity rows (one per pi). The ``_C`` suffixes are
    merge.py's names, but each value is the median over ALL pi rows, not the C row:
      amp_any      the amplifier flag under any pi
      sib_C        median sibling_faith_median        lifted_C   median switched_on_circuit
      rank_clean   median rank_clean (one clean run, so every pi row holds the same value)
      near_threshold  rank_clean >= k / 2; k defaults to the rows' own ``K`` (the dictionary's Top-K k)"""
    rows = [r for r in rows if "error" not in r]
    if k is None:
        ks = {int(r["K"]) for r in rows if r.get("K") is not None}
        if len(ks) != 1:
            raise ValueError("pass k: the rows carry %s" % (sorted(ks) or "no K"))
        k = ks.pop()
    rank_clean = _median(r.get("rank_clean") for r in rows)
    return dict(amp_any=any(amplifier_flag(r) for r in rows) if rows else None,
                sib_C=_median(r.get("sibling_faith_median") for r in rows),
                rank_clean=rank_clean,
                lifted_C=_median(r.get("switched_on_circuit") for r in rows),
                near_threshold=near_threshold(rank_clean, k))


__all__ = ["specificity_rows", "latent_faith", "amplifier_flag", "near_threshold", "target_specificity", "PIS",
           "SIB_FRAC", "MIN_DEN"]
