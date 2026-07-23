"""
Tests for the cross-sequence recurrence prune (eval/recurrence_prune.py).

The prune drops members that fire in fewer than ``min_sequences`` of the probe
sequences. Recurrence comes from one forward pass, so we monkeypatch the
collector and check the selection logic against real Circuit structures.

Recurrence fixture: member i fires in (i + 1) sequences, so with min_sequences=3
members 0 and 1 are doomed and 2..n-1 survive.
"""

import pytest
import torch

from circuit.types.feature_id import FeatureID
from eval import recurrence_prune as rp
from store.circuits import Circuit, CircuitNode

D_SAE = 64


def _build_circuit(n_members=6):
    """Seed + n members, member i at latent index i with |score| = n - i."""
    c = Circuit(name="t")
    seed = c.add_node(CircuitNode(metadata={
        "feature_id": FeatureID(5, "mlp", 999), "role": "seed"}))
    uuids = []
    for i in range(n_members):
        node = c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(0, "mlp", i),
            "role": "ablation_support",
            "attribution_score": float(n_members - i),
        }))
        uuids.append(node.uuid)
        c.add_edge(node.uuid, seed.uuid, weight=1.0)
    return c, seed, uuids


@pytest.fixture(autouse=True)
def patch_collector(monkeypatch):
    """member i fires in exactly (i + 1) sequences."""
    counts = torch.zeros(D_SAE, dtype=torch.long)
    for i in range(D_SAE):
        counts[i] = i + 1
    monkeypatch.setattr(
        rp, "collect_sequence_recurrence",
        lambda inference, bank, tokens, sites: {site: counts for site in sites},
    )


TOKENS = torch.zeros(8, 4, dtype=torch.long)


def test_drops_members_below_threshold():
    c, seed, uuids = _build_circuit(6)
    removed = rp.prune_by_sequence_recurrence(
        None, None, c, pos_tokens=TOKENS, min_sequences=3)
    # members 0 and 1 fire in 1 and 2 sequences respectively
    assert set(removed) == {uuids[0], uuids[1]}
    assert len(c.nodes) == 5                      # 4 survivors + seed
    assert seed.uuid in c.nodes


def test_seed_is_never_removed():
    c, seed, _ = _build_circuit(6)
    rp.prune_by_sequence_recurrence(
        None, None, c, pos_tokens=TOKENS, min_sequences=D_SAE + 10)
    assert seed.uuid in c.nodes


def test_min_sequences_one_is_a_noop():
    """Every member fires in >= 1 sequence by construction, so this must not
    touch the circuit — and must not even run the forward pass."""
    c, _, _ = _build_circuit(6)
    before = len(c.nodes)
    assert rp.prune_by_sequence_recurrence(
        None, None, c, pos_tokens=TOKENS, min_sequences=1) == []
    assert len(c.nodes) == before


def test_min_keep_rescues_most_recurrent():
    """When the threshold would over-prune, the most-recurrent members are kept
    rather than emptying the circuit."""
    c, seed, uuids = _build_circuit(6)
    rp.prune_by_sequence_recurrence(
        None, None, c, pos_tokens=TOKENS, min_sequences=100, min_keep=2)
    survivors = {u for u in uuids if u in c.nodes}
    # members 4 and 5 fire in 5 and 6 sequences — the two most recurrent
    assert survivors == {uuids[4], uuids[5]}


def test_edges_to_removed_nodes_are_dropped():
    c, seed, uuids = _build_circuit(6)
    rp.prune_by_sequence_recurrence(
        None, None, c, pos_tokens=TOKENS, min_sequences=3)
    live = set(c.nodes)
    for edge in c.edges:
        assert edge.source_uuid in live and edge.target_uuid in live


def test_records_provenance_metadata():
    c, _, _ = _build_circuit(6)
    rp.prune_by_sequence_recurrence(
        None, None, c, pos_tokens=TOKENS, min_sequences=3)
    assert c.metadata["n_members_pre_recurrence_prune"] == 7   # 6 members + seed
    assert c.metadata["recurrence_prune_min_sequences"] == 3


def test_no_members_below_threshold_is_a_noop():
    c, _, _ = _build_circuit(6)
    before = len(c.nodes)
    assert rp.prune_by_sequence_recurrence(
        None, None, c, pos_tokens=TOKENS, min_sequences=1) == []
    assert len(c.nodes) == before


class TestRoleSplit:
    """Supports are judged on posctx, inhibitors on negctx. Counting an
    inhibitor over posctx would penalise it for doing exactly what it is for,
    and since the prune runs before the cf acceptance gate that can change
    whether a circuit is accepted."""

    @staticmethod
    def _mixed_circuit():
        """Two supports and two inhibitors, all at latent indices 0 and 1."""
        c = Circuit(name="t")
        c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(5, "mlp", 999), "role": "seed"}))
        uu = {}
        for idx, role in ((0, "ablation_support"), (1, "ablation_support"),
                          (2, "counterfactual_inhibitor"),
                          (3, "counterfactual_inhibitor")):
            n = c.add_node(CircuitNode(metadata={
                "feature_id": FeatureID(0, "mlp", idx), "role": role,
                "attribution_score": 1.0}))
            uu[idx] = n.uuid
        return c, uu

    @pytest.fixture
    def split_counts(self, monkeypatch):
        """posctx: latent i fires in i+1 sequences (0 -> 1, rare).
        negctx: reversed, so latent 2 is rare on posctx but common on negctx."""
        pos = torch.arange(D_SAE, dtype=torch.long) + 1
        neg = torch.flip(torch.arange(D_SAE, dtype=torch.long) + 1, dims=[0])
        seen = {}

        def fake(inference, bank, tokens, sites):
            table = pos if tokens.shape[0] == 8 else neg
            seen[int(tokens.shape[0])] = True
            return {site: table for site in sites}

        monkeypatch.setattr(rp, "collect_sequence_recurrence", fake)
        return seen

    def test_inhibitors_judged_on_negctx_not_posctx(self, split_counts):
        c, uu = self._mixed_circuit()
        pos_tok = torch.zeros(8, 4, dtype=torch.long)
        neg_tok = torch.zeros(9, 4, dtype=torch.long)
        removed = set(rp.prune_by_sequence_recurrence(
            None, None, c, pos_tokens=pos_tok, neg_tokens=neg_tok,
            min_sequences=3))
        # support at index 0 fires in 1 posctx sequence -> dropped
        assert uu[0] in removed
        # support at index 1 fires in 2 -> also below 3 -> dropped
        assert uu[1] in removed
        # inhibitors 2,3 are RARE on posctx but common on negctx -> kept
        assert uu[2] not in removed and uu[3] not in removed
        assert split_counts == {8: True, 9: True}      # both passes ran

    def test_inhibitors_exempt_when_no_negctx(self, split_counts):
        """Cannot measure them correctly, so leave them alone."""
        c, uu = self._mixed_circuit()
        removed = set(rp.prune_by_sequence_recurrence(
            None, None, c, pos_tokens=torch.zeros(8, 4, dtype=torch.long),
            neg_tokens=None, min_sequences=3))
        assert uu[0] in removed and uu[1] in removed
        assert uu[2] not in removed and uu[3] not in removed
        assert 9 not in split_counts                   # no negctx pass attempted


class TestScalesToLargeCircuits:
    """Cost is linear in members, so a small circuit cannot reveal a per-member
    regression. The original implementation indexed a CUDA tensor once per
    member; each scalar index forces a device sync (~81us measured), which cost
    ~40s on a 500k-member circuit while the two forward passes producing the
    counts cost ~2s. These guard the two properties that fixed it."""

    def test_counts_are_returned_on_cpu(self, monkeypatch):
        """A GPU count table would reintroduce a device sync per member."""
        monkeypatch.undo()

        class FakeBank:
            kinds = ["mlp"]
            d_sae = D_SAE

            def encode(self, act, kind, layer_idx):
                return (torch.ones(2, 2, 1), torch.zeros(2, 2, 1, dtype=torch.long))

        class FakeInference:
            def disable_compile(self):
                pass

            def enable_compile(self):
                pass

            def forward(self, tokens, activations_callback=None, **kw):
                activations_callback(0, (torch.zeros(2, 2, 4),))

        counts = rp.collect_sequence_recurrence(
            FakeInference(), FakeBank(), torch.zeros(2, 2, dtype=torch.long),
            {(0, "mlp")})
        for site, table in counts.items():
            assert table.device.type == "cpu", f"{site} counts must be CPU-resident"

    def test_large_circuit_scores_in_reasonable_time(self):
        """50k members must not take pathologically long. The per-member-sync
        implementation needed ~4s here; the vectorised one needs ~0.02s."""
        import time
        n = 50_000
        c = Circuit(name="big")
        c.add_node(CircuitNode(metadata={
            "feature_id": FeatureID(5, "mlp", 999), "role": "seed"}))
        for i in range(n):
            c.add_node(CircuitNode(metadata={
                "feature_id": FeatureID(0, "mlp", i % D_SAE),
                "role": "ablation_support", "attribution_score": 1.0}))
        t = time.time()
        rp.prune_by_sequence_recurrence(
            None, None, c, pos_tokens=TOKENS, min_sequences=3)
        assert time.time() - t < 3.0, "scoring loop has regressed to per-member cost"


class TestConfigSurface:
    def test_defaults_are_off_and_conservative(self):
        from config import config
        assert config.discovery.recurrence_prune is False
        assert config.discovery.recurrence_prune_min_sequences == 2
        assert config.discovery.recurrence_prune_min_keep == 1

    @pytest.mark.parametrize("field,bad", [
        ("recurrence_prune_min_sequences", 0),
        ("recurrence_prune_min_keep", 0),
    ])
    def test_validators_reject_below_one(self, field, bad):
        from config import DiscoveryConfig
        with pytest.raises(Exception):
            DiscoveryConfig(**{field: bad})


class TestRecurrenceCollector:
    """The collector counts DISTINCT sequences, not firings."""

    def test_counts_distinct_sequences(self, monkeypatch):
        monkeypatch.undo()

        class FakeBank:
            kinds = ["mlp"]
            d_sae = D_SAE

            def encode(self, act, kind, layer_idx):
                # latent 0 fires twice in sequence 0 (should count ONCE);
                # latent 1 fires once in each of sequences 0 and 1 -> 2.
                idx = torch.tensor([[[0], [0]], [[1], [1]]])
                val = torch.tensor([[[1.0], [1.0]], [[0.0], [1.0]]])
                idx = torch.cat([idx, torch.full_like(idx, 2)], dim=-1)
                val = torch.cat([val, torch.zeros_like(val)], dim=-1)
                return val, idx

        class FakeInference:
            def disable_compile(self):
                pass

            def enable_compile(self):
                pass

            def forward(self, tokens, activations_callback=None, **kw):
                activations_callback(0, (torch.zeros(2, 2, 4),))

        counts = rp.collect_sequence_recurrence(
            FakeInference(), FakeBank(), torch.zeros(2, 2, dtype=torch.long),
            {(0, "mlp")})[(0, "mlp")]
        assert int(counts[0]) == 1      # two firings, one sequence
        assert int(counts[1]) == 1      # only sequence 1 had a live value
        assert int(counts[2]) == 0      # present in indices but zero-valued
