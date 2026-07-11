"""
Unit tests for attach_direct_edges (SFC App. B direct-effect edges).

Pipeline stub: three chained sites in layer 0 (attn -> mlp -> resid, joined
by fixed matmuls), then the seed site at layer 1. SAEGraphInstrument's
transform output is numerically identical to its input (decode(f) + res +
passthrough == x), so the stub pipeline's values are mode-independent; only
gradient routing differs.

Covers:
  - edges attach only between circuit members, tagged direct_effect
  - stop-gradient property: a mediator site's SAE weights cannot influence
    a through-edge (feature paths are severed; only the passthrough carries)
  - batched (is_grads_batched) == sequential fallback
  - depth metrics exceed 1 after wiring (star -> DAG)
"""

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from analysis.circuits.gradient_size_sweep_runner import _circuit_depth_stats
from circuit.instrument.edge_attribution import attach_direct_edges
from circuit.types.feature_id import FeatureID
from store.circuits import Circuit, CircuitNode

from tests.conftest import D_MODEL, D_SAE, KINDS, MockSAEBank

B, T = 2, 4
SEED_LAYER, SEED_KIND, SEED_LATENT = 1, "attn", 5
SITE_A, SITE_M, SITE_D = (0, "attn"), (0, "mlp"), (0, "resid")


def _firing_member_indices(bank, x0, mats):
    """Pick, per site, the latent with the largest total natural activation.
    Members must actually fire (be in the encode top-k) or their deltas and
    scatter-gradients are identically zero and no edge can exist. The
    instrument's output equals its input numerically, so the natural stream
    is x0 -> x0 @ w_am -> ... directly."""
    from sae.dense import sparse_topk_to_dense

    w_am, w_md, _ = mats
    xs = {SITE_A: x0, SITE_M: x0 @ w_am, SITE_D: (x0 @ w_am) @ w_md}
    indices = {}
    for site, x in xs.items():
        ta, ti = bank.encode(x, site[1], site[0])
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32)
        # Members must fire AT THE PROBE POSITION (t=0, matching pos_argmax
        # zeros): both the cotangent (metric grads live at probe positions)
        # and the top-k scatter backward are zero elsewhere.
        indices[site] = int(dense[:, 0, :].sum(dim=0).argmax().item())
    return indices


def _shimmed_bank(seed=0):
    """MockSAEBank plus the encoder.weight / _get_bias_eff surface that
    attach_direct_edges reads for the seed tap."""
    torch.manual_seed(seed)
    bank = MockSAEBank()
    for kind in KINDS:
        for sae in bank.saes[kind]:
            sae.encoder = SimpleNamespace(weight=sae.W_enc)
            sae._get_bias_eff = (lambda s: (lambda: s.b_enc))(sae)
    return bank


def _stub_inference(x0, w_am, w_md, w_ds):
    """Chain: A -> (w_am) -> M -> (w_md) -> D -> (w_ds) -> seed site."""

    def forward_fn(tokens, patcher=None, **kwargs):
        out_a = patcher.transform(*SITE_A, x0.clone())
        out_m = patcher.transform(*SITE_M, out_a @ w_am)
        out_d = patcher.transform(*SITE_D, out_m @ w_md)
        patcher.transform(SEED_LAYER, SEED_KIND, out_d @ w_ds)

    inf = MagicMock()
    inf.forward.side_effect = forward_fn
    return inf


def _member_circuit(indices):
    circuit = Circuit(name="edge-test")
    seed = circuit.add_node(CircuitNode(metadata={
        "feature_id": FeatureID(SEED_LAYER, SEED_KIND, SEED_LATENT), "role": "seed",
    }))
    members = {}
    for (layer, kind), idx in indices.items():
        node = circuit.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(layer, kind, idx), "role": "ablation_support",
        }))
        circuit.add_edge(node.uuid, seed.uuid, weight=1.0)  # original star edges
        members[(layer, kind)] = node
    return circuit, seed, members


def _fixture(bank_seed=0, data_seed=41):
    torch.manual_seed(data_seed)
    x0 = torch.randn(B, T, D_MODEL)
    mats = tuple(torch.randn(D_MODEL, D_MODEL) * 0.3 for _ in range(3))
    bank = _shimmed_bank(seed=bank_seed)
    indices = _firing_member_indices(bank, x0, mats)
    circuit, seed, members = _member_circuit(indices)
    return bank, circuit, seed, members, indices, x0, mats


def _run_attach(bank, circuit, *, batched=True, x0=None, mats=None):
    inf = _stub_inference(x0, *mats)
    with torch.enable_grad():
        stats = attach_direct_edges(
            circuit,
            inf,
            bank,
            pos_tokens=torch.zeros(B, T, dtype=torch.long),
            pos_argmax=torch.zeros(B, dtype=torch.long),
            seed_layer=SEED_LAYER,
            seed_kind=SEED_KIND,
            seed_latent_idx=SEED_LATENT,
            top_k_edges_per_node=D_SAE,  # membership, not rank, decides in tests
            chunk_size=2,
            batched=batched,
        )
    return stats, x0, mats


def _direct_edges(circuit):
    return {
        (edge.source_uuid, edge.target_uuid): edge.metadata["weight"]
        for edge in circuit.edges
        if edge.metadata.get("kind") == "direct_effect"
    }


class TestAttachDirectEdges:
    def test_edges_only_between_members_with_star_preserved(self):
        bank, circuit, seed, members, _, x0, mats = _fixture()
        stats, _, _ = _run_attach(bank, circuit, x0=x0, mats=mats)
        edges = _direct_edges(circuit)

        member_uuids = {node.uuid for node in members.values()}
        # Site-A member has no upstream sites, so only M and D are downstream.
        assert stats["n_downstream_nodes"] == 2
        assert len(edges) >= 1
        for (src, dst) in edges:
            assert src in member_uuids
            assert dst in member_uuids
            assert dst != seed.uuid
        # The three original star edges survive untouched.
        star = [e for e in circuit.edges if e.metadata.get("kind") != "direct_effect"]
        assert len(star) == 3 and all(e.target_uuid == seed.uuid for e in star)

    def test_upstream_member_reaches_downstream_member(self):
        bank, circuit, _, members, _, x0, mats = _fixture()
        _run_attach(bank, circuit, x0=x0, mats=mats)
        edges = _direct_edges(circuit)
        key = (members[SITE_A].uuid, members[SITE_D].uuid)
        assert key in edges, "A-member should wire to D-member through the raw path"
        assert edges[key] == edges[key]  # finite (not NaN)

    def test_stop_gradient_mediator_weights_are_irrelevant(self):
        """Scaling the mediator site's SAE weights must not change the
        A->D edge: transform output is numerically x regardless, and the
        mediator's feature path is a severed leaf. If gradients leaked
        through mediator features, this invariance would break."""
        bank1, circuit1, _, members1, indices, x0, mats = _fixture()
        _run_attach(bank1, circuit1, x0=x0, mats=mats)
        w1 = _direct_edges(circuit1)[(members1[SITE_A].uuid, members1[SITE_D].uuid)]

        bank2 = _shimmed_bank(seed=0)
        mlp_sae = bank2.saes["mlp"][0]
        mlp_sae.W_enc = mlp_sae.W_enc * 7.0  # mangle the mediator's SAE
        mlp_sae.W_dec = mlp_sae.W_dec * 0.1
        circuit2, _, members2 = _member_circuit(indices)
        _run_attach(bank2, circuit2, x0=x0, mats=mats)
        w2 = _direct_edges(circuit2)[(members2[SITE_A].uuid, members2[SITE_D].uuid)]

        assert w1 == pytest.approx(w2, rel=1e-4)

    def test_batched_matches_sequential(self):
        bank1, circuit1, _, members1, indices, x0, mats = _fixture(bank_seed=3)
        _run_attach(bank1, circuit1, batched=True, x0=x0, mats=mats)

        bank2 = _shimmed_bank(seed=3)
        circuit2, _, members2 = _member_circuit(indices)
        _run_attach(bank2, circuit2, batched=False, x0=x0, mats=mats)

        remap = {members1[s].uuid: members2[s].uuid for s in members1}
        edges1 = _direct_edges(circuit1)
        edges2 = _direct_edges(circuit2)
        assert len(edges1) == len(edges2) > 0
        for (src, dst), weight in edges1.items():
            assert edges2[(remap[src], remap[dst])] == pytest.approx(weight, rel=1e-4)

    def test_depth_exceeds_one_after_wiring(self):
        bank, circuit, _, _, _, x0, mats = _fixture()
        assert _circuit_depth_stats(circuit)["node_depth_max"] == 1  # star before
        _run_attach(bank, circuit, x0=x0, mats=mats)
        stats = _circuit_depth_stats(circuit)
        assert stats["node_depth_max"] >= 2
        assert stats["n_internal_edges"] >= 1

    def test_metadata_flag_set(self):
        bank, circuit, _, _, _, x0, mats = _fixture()
        _run_attach(bank, circuit, x0=x0, mats=mats)
        assert circuit.metadata["edge_attribution"] == "direct_effect"
