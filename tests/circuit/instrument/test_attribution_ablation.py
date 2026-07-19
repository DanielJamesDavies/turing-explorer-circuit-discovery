"""Oracle tests for compute_latent_ablation_scores — the classic collapsed
path and the position-aware branch (PA-abl-local).

Oracle site (0, "attn"), D_SAE=3, B=T=1:
    acts  = [2.0, 1.0, 3.0]
    ts    = f[0,0,0] - f[0,0,2]  ->  grads [ +1, 0, -1 ]
    acts x grad per position     =  [ +2, 0, -3 ]

Classic (sum + positive filter): supports = {latent0: 2.0} only.
Position-aware (allowed set): BOTH signs are members at |score| — a
negatively-scored latent still carries stream content at its position.
"""

import pytest
import torch

from circuit.instrument.attribution_ablation import compute_latent_ablation_scores
from circuit.instrument.position_aware import PositionAwareSpec
from circuit.instrument.sae_graph import FeatureGraph
from circuit.types.feature_id import FeatureID
from circuit.types.sparse_act import SparseAct

_N_KINDS = 3
_KINDS = ["attn", "mlp", "resid"]
_N_LAYERS = 2
_D_SAE = 3
_SEED_COMP = 5  # layer 1, resid — every layer-0 site is upstream
_ACTIVE = torch.ones(_N_LAYERS * _N_KINDS, _D_SAE, dtype=torch.long)


def _single_site_oracle():
    vals = torch.tensor([[[2.0, 1.0, 3.0]]])
    leaf = vals.detach().clone().requires_grad_(True)
    graph = FeatureGraph(torch.device("cpu"))
    graph.add(0, "attn", SparseAct(act=leaf), SparseAct(act=vals.clone()),
              torch.tensor([[[0, 1, 2]]]))
    ts = leaf[0, 0, 0] - leaf[0, 0, 2]
    return graph, ts


def _two_site_oracle():
    """attn carries big attributions, mlp small — for pooled-cut semantics."""
    a_vals = torch.tensor([[[10.0, 0.0, 20.0]]])
    a_leaf = a_vals.detach().clone().requires_grad_(True)
    m_vals = torch.tensor([[[1.0, 2.0, 0.0]]])
    m_leaf = m_vals.detach().clone().requires_grad_(True)
    graph = FeatureGraph(torch.device("cpu"))
    graph.add(0, "attn", SparseAct(act=a_leaf), SparseAct(act=a_vals.clone()),
              torch.tensor([[[0, 1, 2]]]))
    graph.add(0, "mlp", SparseAct(act=m_leaf), SparseAct(act=m_vals.clone()),
              torch.tensor([[[0, 1, 2]]]))
    # grads: attn [1, 1, 1]; mlp [1, 1, 1] -> attrs attn [10, 0, 20], mlp [1, 2, 0]
    ts = a_leaf.sum() + m_leaf.sum()
    return graph, ts


def _call(graph, ts, position_aware=None, top_k_supports=10, top_k_scope="global"):
    return compute_latent_ablation_scores(
        graph=graph, target_scalar=ts, seed_comp_idx=_SEED_COMP,
        n_kinds=_N_KINDS, kinds=_KINDS, top_k_supports=top_k_supports,
        min_active_count=1, active_count=_ACTIVE, top_k_scope=top_k_scope,
        position_aware=position_aware,
    )


def _spec(**kw):
    kw.setdefault("peaks", torch.zeros(1, dtype=torch.long))
    kw.setdefault("top_n", 2)
    return PositionAwareSpec(**kw)


# ---------------------------------------------------------------- classic path

def test_classic_keeps_positive_supports_only():
    graph, ts = _single_site_oracle()
    supports = _call(graph, ts)
    assert supports == {FeatureID(0, "attn", 0): pytest.approx(2.0, abs=1e-5)}


def test_classic_unchanged_when_position_aware_none():
    """The default path must stay bit-identical — regression anchor."""
    graph, ts = _single_site_oracle()
    assert _call(graph, ts, position_aware=None) == _call(graph, ts)


# ---------------------------------------------------------- position-aware path

def test_pa_keeps_both_signs_as_members_signed():
    """Allowed-set semantics: the negatively-attributed latent (acts x grad =
    -3) is a member at its SIGNED score — zeroing it would corrupt the stream.
    The scorer preserves the sign so the caller (resolve_role_delivery) can
    label inhibitors under include / fold them under exclude."""
    graph, ts = _single_site_oracle()
    members = _call(graph, ts, position_aware=_spec())
    assert members[FeatureID(0, "attn", 0)] == pytest.approx(2.0, abs=1e-5)
    assert members[FeatureID(0, "attn", 2)] == pytest.approx(-3.0, abs=1e-5)
    assert FeatureID(0, "attn", 1) not in members  # zero attribution
    # both signs present as members (magnitude is what the allowed set needs)
    assert any(v < 0 for v in members.values())


def test_pa_bypasses_top_k_truncation():
    """The union replaces the ranking: top_k_supports=1 must not clip it."""
    graph, ts = _single_site_oracle()
    members = _call(graph, ts, position_aware=_spec(), top_k_supports=1)
    assert len(members) == 2


def test_pa_abs_pctl_pools_across_sites():
    """One admission cut per pass: at p50 of the pooled |attr| distribution
    {10, 20, 1, 2}, the mlp site's small attributions fall below a bar the
    attn site set — even though within mlp alone they would top its list."""
    graph, ts = _two_site_oracle()
    members = _call(graph, ts, position_aware=_spec(select="abs_pctl", threshold=50))
    kept = set(members)
    assert FeatureID(0, "attn", 0) in kept and FeatureID(0, "attn", 2) in kept
    assert FeatureID(0, "mlp", 0) not in kept and FeatureID(0, "mlp", 1) not in kept


def test_pa_abs_pctl_equals_abs_at_derived_cut():
    from circuit.instrument.position_aware import pooled_abs_threshold
    graph, ts = _two_site_oracle()
    pctl_members = _call(graph, ts, position_aware=_spec(select="abs_pctl", threshold=50))
    graph2, ts2 = _two_site_oracle()
    th = pooled_abs_threshold(
        [torch.tensor([[[10.0, 0.0, 20.0]]]), torch.tensor([[[1.0, 2.0, 0.0]]])], 50)
    abs_members = _call(graph2, ts2, position_aware=_spec(select="abs", threshold=th))
    assert pctl_members == abs_members


def test_pa_count_mask_still_filters():
    graph, ts = _single_site_oracle()
    dead = torch.zeros(_N_LAYERS * _N_KINDS, _D_SAE, dtype=torch.long)
    members = compute_latent_ablation_scores(
        graph=graph, target_scalar=ts, seed_comp_idx=_SEED_COMP,
        n_kinds=_N_KINDS, kinds=_KINDS, top_k_supports=10,
        min_active_count=1, active_count=dead, position_aware=_spec(),
    )
    assert members == {}
