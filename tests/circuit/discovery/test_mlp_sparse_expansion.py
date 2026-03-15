"""
Phase 7 — MlpSparseExpansion circuit discovery algorithm tests.

Tests are organised into five classes:

  TestMlpNodeExpansion  — two-hop co-activation expansion only yields MLP nodes
  TestPassthroughNodes  — active attn/resid latents are added as passthrough nodes
  TestCircuitEdges      — edge structure (seed→hop1, hop1→hop2, no passthrough edges)
  TestActivityFilter    — latents below min_active_count are excluded
  TestRejectionCases    — non-MLP seed / empty probe / impossible faithfulness threshold

Mock co-activation data
-----------------------
The seed is at (comp=1, lat=5), i.e. layer=0, kind="mlp".

Eight global-index neighbors are set for the seed; five are MLP (comp=4) and
three are non-MLP (attn / resid), so the MLP filter is explicitly exercised.

  _HOP1_GLOBALS order:
    4*32+ 7 → comp=4 (layer=1, mlp)  ← kept
    4*32+ 8 → comp=4 (layer=1, mlp)  ← kept
    0*32+ 3 → comp=0 (layer=0, attn) ← filtered
    4*32+ 9 → comp=4 (layer=1, mlp)  ← kept
    2*32+ 5 → comp=2 (layer=0, resid)← filtered
    4*32+11 → comp=4 (layer=1, mlp)  ← kept
    3*32+ 2 → comp=3 (layer=1, attn) ← filtered
    4*32+13 → comp=4 (layer=1, mlp)  ← kept

Hop-2 neighbors of (comp=4, lat=7):
    4*32+20 → comp=4 (layer=1, mlp)  ← kept
    5*32+15 → comp=5 (layer=1, resid)← filtered
    4*32+21 → comp=4 (layer=1, mlp)  ← kept
    4*32+22 → comp=4 (layer=1, mlp)  ← kept
"""
import pytest
import torch

from circuit.discovery.mlp_sparse_expansion import MlpSparseExpansion
from circuit.probe_dataset import ProbeDataset

# ---------------------------------------------------------------------------
# Local constants  (must be consistent with conftest.py shared fixtures)
# ---------------------------------------------------------------------------

N_LAYERS = 2
D_MODEL  = 16
D_SAE    = 32
K_SAE    = 4
KINDS    = ["attn", "mlp", "resid"]
N_COMP   = N_LAYERS * len(KINDS)   # 6

B_TEST   = 2
T_TEST   = 4
V_TEST   = 20

# Seed: layer=0, kind="mlp"  →  comp = 0*3 + 1 = 1
SEED_COMP = 1
SEED_LAT  = 5

# Global-index format: comp * D_SAE + lat
_HOP1_GLOBALS = [
    4 * D_SAE +  7,   # comp=4 (layer=1, mlp)  ← MLP ✓
    4 * D_SAE +  8,   # comp=4 (layer=1, mlp)  ← MLP ✓
    0 * D_SAE +  3,   # comp=0 (layer=0, attn) ← skipped
    4 * D_SAE +  9,   # comp=4 (layer=1, mlp)  ← MLP ✓
    2 * D_SAE +  5,   # comp=2 (layer=0, resid)← skipped
    4 * D_SAE + 11,   # comp=4 (layer=1, mlp)  ← MLP ✓
    3 * D_SAE +  2,   # comp=3 (layer=1, attn) ← skipped
    4 * D_SAE + 13,   # comp=4 (layer=1, mlp)  ← MLP ✓
]

# Hop-2 neighbors of (comp=4, lat=7) — mix of MLP and resid
_HOP2_GLOBALS_FOR_4_7 = [
    4 * D_SAE + 20,   # comp=4 (layer=1, mlp)  ← MLP ✓
    5 * D_SAE + 15,   # comp=5 (layer=1, resid)← skipped
    4 * D_SAE + 21,   # comp=4 (layer=1, mlp)  ← MLP ✓
    4 * D_SAE + 22,   # comp=4 (layer=1, mlp)  ← MLP ✓
]


# ---------------------------------------------------------------------------
# Lightweight mock data stores
# ---------------------------------------------------------------------------

class MockTopCoactivation:
    """Stand-in for store.top_coactivation.top_coactivation."""

    def __init__(self, n_comp: int, d_sae: int, n_neighbors: int = 8):
        self.top_indices = torch.zeros(n_comp, d_sae, n_neighbors, dtype=torch.int32)
        self.top_values  = torch.zeros(n_comp, d_sae, n_neighbors, dtype=torch.float32)

    def set_neighbors(self, comp: int, lat: int, global_indices: list):
        n = len(global_indices)
        assert n <= self.top_indices.shape[2], "more neighbors than allocated slots"
        self.top_indices[comp, lat, :n] = torch.tensor(global_indices, dtype=torch.int32)
        # Decreasing weights so ordering is deterministic for limit tests
        self.top_values[comp, lat, :n] = (
            torch.arange(n, 0, -1, dtype=torch.float32) * 0.1
        )


class MockLatentStats:
    """Stand-in for store.latent_stats.latent_stats."""

    def __init__(self, n_comp: int, d_sae: int, default_count: int = 100):
        self.active_count = torch.full(
            (n_comp, d_sae), default_count, dtype=torch.int64
        )


def _make_mock_coact() -> MockTopCoactivation:
    """Build deterministic co-activation data used by most tests."""
    coact = MockTopCoactivation(N_COMP, D_SAE, n_neighbors=8)
    coact.set_neighbors(SEED_COMP, SEED_LAT, _HOP1_GLOBALS)
    coact.set_neighbors(4, 7, _HOP2_GLOBALS_FOR_4_7)
    return coact


# ---------------------------------------------------------------------------
# MockInference
# (identical interface to tests/eval/test_evaluate.py — reproduced so that
#  this test module has no non-fixture dependency on other test files)
# ---------------------------------------------------------------------------

class MockInference:
    """
    Minimal stand-in for Inference.forward().  A fixed seeded input is reused
    on every call so that the same patcher configuration always produces the
    same logits, making evaluation-threshold tests deterministic.
    """

    def __init__(self, model, d_model: int = D_MODEL, d_vocab: int = V_TEST):
        torch.manual_seed(7)
        self.model   = model
        self.W_logit = torch.randn(d_model, d_vocab)
        torch.manual_seed(13)
        self._x = torch.randn(B_TEST, T_TEST, d_model)

    _compiled = False
    def disable_compile(self): pass
    def enable_compile(self):  pass

    def forward(
        self,
        tokens,
        num_gen:            int  = 1,
        tokenize_final:     bool = False,
        return_activations: bool = False,
        all_logits:         bool = False,
        patcher             = None,
    ):
        with torch.no_grad():
            if patcher is not None:
                with patcher(self.model):
                    output = self.model(self._x)
            else:
                output = self.model(self._x)

        if all_logits:
            logits = output @ self.W_logit
        else:
            logits = output[:, -1:, :] @ self.W_logit

        return tokens, logits, None


# ---------------------------------------------------------------------------
# Helper: build a fresh ProbeDataset with deterministic dummy tokens
# ---------------------------------------------------------------------------

def _make_probe(b: int = B_TEST) -> ProbeDataset:
    return ProbeDataset(
        pos_tokens    = torch.zeros(b, T_TEST, dtype=torch.long),
        target_tokens = torch.zeros(b, T_TEST, dtype=torch.long),
        neg_tokens    = torch.zeros(b, T_TEST, dtype=torch.long),
        pos_argmax    = torch.tensor([1, 2][:b]),
        metadata      = {},
    )


# ---------------------------------------------------------------------------
# Core fixture — wires everything together
# ---------------------------------------------------------------------------

@pytest.fixture
def setup(mock_model, mock_sae_bank, monkeypatch):
    """
    Returns (algo, probe_data, mock_coact, mock_stats).

    Patches the top_coactivation and latent_stats module-level singletons
    inside circuit.discovery.mlp_sparse_expansion with lightweight mocks.
    Sets min_faithfulness=0.0 so structural tests always get a circuit back.
    """
    mock_coact = _make_mock_coact()
    mock_stats = MockLatentStats(N_COMP, D_SAE)

    import circuit.discovery.mlp_sparse_expansion as mse_mod
    monkeypatch.setattr(mse_mod, "top_coactivation", mock_coact)
    monkeypatch.setattr(mse_mod, "latent_stats",     mock_stats)

    inference = MockInference(mock_model)
    avg_acts  = torch.zeros(N_COMP, D_SAE)

    algo = MlpSparseExpansion(
        inference        = inference,
        sae_bank         = mock_sae_bank,
        avg_acts         = avg_acts,
        probe_builder    = None,
        min_faithfulness = 0.0,    # accept all circuits for structural tests
        min_active_count = 50,
        pruning_threshold= 0.0,
        probe_batch_size = 16,
    )

    probe_data = _make_probe()
    algo.build_probe_dataset = lambda *a, **kw: probe_data

    return algo, probe_data, mock_coact, mock_stats


# ---------------------------------------------------------------------------
# TestMlpNodeExpansion
# ---------------------------------------------------------------------------

class TestMlpNodeExpansion:
    """Verifies that two-hop expansion only adds MLP-kind nodes."""

    def test_seed_node_is_mlp_with_seed_role(self, setup):
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        seed_nodes = [
            n for n in circuit.nodes.values() if n.metadata["role"] == "seed"
        ]
        assert len(seed_nodes) == 1
        assert seed_nodes[0].metadata["kind"] == "mlp"

    def test_hop1_nodes_all_mlp(self, setup):
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop1 = [n for n in circuit.nodes.values() if n.metadata["role"] == "hop1"]
        assert len(hop1) > 0
        for node in hop1:
            assert node.metadata["kind"] == "mlp", (
                f"hop-1 node has non-MLP kind: {node.metadata['kind']}"
            )

    def test_hop1_count_respects_limit(self, mock_model, mock_sae_bank, monkeypatch):
        """With coact_depth=[3] only the first 3 MLP candidates should be added."""
        mock_coact = _make_mock_coact()
        mock_stats = MockLatentStats(N_COMP, D_SAE)
        import circuit.discovery.mlp_sparse_expansion as mse_mod
        monkeypatch.setattr(mse_mod, "top_coactivation", mock_coact)
        monkeypatch.setattr(mse_mod, "latent_stats",     mock_stats)

        algo = MlpSparseExpansion(
            inference=MockInference(mock_model), sae_bank=mock_sae_bank,
            avg_acts=torch.zeros(N_COMP, D_SAE), probe_builder=None,
            coact_depth=[3],   # depth-1 only, capped at 3
            min_faithfulness=0.0, min_active_count=50,
        )
        algo.build_probe_dataset = lambda *a, **kw: _make_probe()

        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop1 = [n for n in circuit.nodes.values() if n.metadata["role"] == "hop1"]
        assert len(hop1) <= 3

    def test_hop2_nodes_all_mlp(self, setup):
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop2 = [n for n in circuit.nodes.values() if n.metadata["role"] == "hop2"]
        for node in hop2:
            assert node.metadata["kind"] == "mlp", (
                f"hop-2 node has non-MLP kind: {node.metadata['kind']}"
            )

    def test_no_duplicate_nodes(self, setup):
        """Each (layer, kind, latent_idx) triple must appear at most once."""
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        seen: set = set()
        for node in circuit.nodes.values():
            key = (
                node.metadata["layer_idx"],
                node.metadata["kind"],
                node.metadata["latent_idx"],
            )
            assert key not in seen, f"Duplicate node: {key}"
            seen.add(key)


# ---------------------------------------------------------------------------
# TestPassthroughNodes
# ---------------------------------------------------------------------------

class TestPassthroughNodes:
    """Verifies that active attn/resid latents are captured as passthrough nodes."""

    def test_passthrough_nodes_are_present(self, setup):
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        pt = [n for n in circuit.nodes.values() if n.metadata["role"] == "passthrough"]
        assert len(pt) > 0, "Expected at least one passthrough node"

    def test_passthrough_nodes_are_attn_or_resid(self, setup):
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata["role"] == "passthrough":
                assert node.metadata["kind"] in ("attn", "resid"), (
                    f"Passthrough node has unexpected kind: {node.metadata['kind']}"
                )

    def test_passthrough_set_matches_captured_latents(self, setup):
        """
        Run _capture_passthrough_nodes independently and compare the resulting
        (layer, kind, latent_idx) set against the passthrough nodes in the circuit.
        """
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None

        probe_tokens = torch.zeros(B_TEST, T_TEST, dtype=torch.long)
        passthrough_map = algo._capture_passthrough_nodes(probe_tokens)

        expected: set = set()
        for (layer, kind), latent_set in passthrough_map.items():
            for lat in latent_set:
                expected.add((layer, kind, lat))

        actual: set = {
            (n.metadata["layer_idx"], n.metadata["kind"], n.metadata["latent_idx"])
            for n in circuit.nodes.values()
            if n.metadata["role"] == "passthrough"
        }
        assert actual == expected

    def test_mlp_nodes_are_never_passthrough(self, setup):
        """MLP nodes should always carry a hop role, not 'passthrough'."""
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata["kind"] == "mlp":
                assert node.metadata["role"] != "passthrough"


# ---------------------------------------------------------------------------
# TestCircuitEdges
# ---------------------------------------------------------------------------

class TestCircuitEdges:
    """Verifies the edge structure of the produced circuit."""

    def test_seed_is_edge_source(self, setup):
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        seed_node = next(
            n for n in circuit.nodes.values() if n.metadata["role"] == "seed"
        )
        source_uuids = {e.source_uuid for e in circuit.edges}
        assert seed_node.uuid in source_uuids

    def test_hop1_nodes_are_edge_targets(self, setup):
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop1_uuids = {
            n.uuid for n in circuit.nodes.values() if n.metadata["role"] == "hop1"
        }
        target_uuids = {e.target_uuid for e in circuit.edges}
        assert hop1_uuids.issubset(target_uuids), (
            "Some hop-1 nodes are not reachable as edge targets"
        )

    def test_hop2_nodes_are_edge_targets(self, setup):
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop2_uuids = {
            n.uuid for n in circuit.nodes.values() if n.metadata["role"] == "hop2"
        }
        if not hop2_uuids:
            pytest.skip("no hop-2 nodes in this circuit")
        target_uuids = {e.target_uuid for e in circuit.edges}
        assert hop2_uuids.issubset(target_uuids), (
            "Some hop-2 nodes are not reachable as edge targets"
        )

    def test_passthrough_nodes_have_no_edges(self, setup):
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        pt_uuids = {
            n.uuid for n in circuit.nodes.values() if n.metadata["role"] == "passthrough"
        }
        if not pt_uuids:
            pytest.skip("no passthrough nodes")
        edge_uuids = (
            {e.source_uuid for e in circuit.edges} |
            {e.target_uuid for e in circuit.edges}
        )
        overlap = pt_uuids & edge_uuids
        assert not overlap, (
            f"{len(overlap)} passthrough node(s) unexpectedly participate in edges"
        )


# ---------------------------------------------------------------------------
# TestActivityFilter
# ---------------------------------------------------------------------------

class TestActivityFilter:
    """Verifies that latents below min_active_count are excluded."""

    def test_low_count_latent_excluded_from_hop1(
        self, mock_model, mock_sae_bank, monkeypatch
    ):
        """
        Set active_count for comp=4, lat=8 to 0; that MLP candidate must not
        appear as a hop-1 node even though it is in the top-coactivation list.
        """
        mock_coact = _make_mock_coact()
        mock_stats = MockLatentStats(N_COMP, D_SAE)
        mock_stats.active_count[4, 8] = 0   # below min_active_count=50

        import circuit.discovery.mlp_sparse_expansion as mse_mod
        monkeypatch.setattr(mse_mod, "top_coactivation", mock_coact)
        monkeypatch.setattr(mse_mod, "latent_stats",     mock_stats)

        algo = MlpSparseExpansion(
            inference=MockInference(mock_model), sae_bank=mock_sae_bank,
            avg_acts=torch.zeros(N_COMP, D_SAE), probe_builder=None,
            min_faithfulness=0.0, min_active_count=50,
        )
        algo.build_probe_dataset = lambda *a, **kw: _make_probe()

        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop1_lats = {
            n.metadata["latent_idx"]
            for n in circuit.nodes.values()
            if n.metadata["role"] == "hop1"
        }
        assert 8 not in hop1_lats, (
            "Latent 8 (count=0) should have been excluded by activity filter"
        )

    def test_all_hop_nodes_pass_activity_threshold(self, setup):
        """Every hop-1 and hop-2 node must have active_count >= min_active_count."""
        algo, _, _, mock_stats = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata["role"] not in ("hop1", "hop2"):
                continue
            layer    = node.metadata["layer_idx"]
            lat      = node.metadata["latent_idx"]
            kind_idx = KINDS.index(node.metadata["kind"])
            comp     = layer * len(KINDS) + kind_idx
            count    = mock_stats.active_count[comp, lat].item()
            assert count >= algo.min_active_count, (
                f"Node (layer={layer}, kind={node.metadata['kind']}, lat={lat}) "
                f"has count={count} < min_active_count={algo.min_active_count}"
            )


# ---------------------------------------------------------------------------
# TestRejectionCases
# ---------------------------------------------------------------------------

class TestRejectionCases:
    """Verifies conditions under which discover() returns None."""

    def test_attn_seed_rejected(self, setup):
        """Seed at an attn component (comp=0, layer=0) must be rejected."""
        algo, _, _, _ = setup
        assert algo.discover(0, 5) is None   # layer=0, kind="attn"

    def test_resid_seed_rejected(self, setup):
        """Seed at a resid component (comp=2, layer=0) must be rejected."""
        algo, _, _, _ = setup
        assert algo.discover(2, 3) is None   # layer=0, kind="resid"

    def test_empty_probe_dataset_rejected(self, setup):
        """An empty pos_tokens tensor triggers early rejection before expansion."""
        algo, _, _, _ = setup
        algo.build_probe_dataset = lambda *a, **kw: ProbeDataset(
            pos_tokens    = torch.zeros(0, T_TEST, dtype=torch.long),
            target_tokens = torch.zeros(0, T_TEST, dtype=torch.long),
            neg_tokens    = torch.zeros(B_TEST, T_TEST, dtype=torch.long),
            pos_argmax    = torch.tensor([], dtype=torch.long),
            metadata      = {},
        )
        assert algo.discover(SEED_COMP, SEED_LAT) is None

    def test_impossible_faithfulness_threshold_rejects(
        self, mock_model, mock_sae_bank, monkeypatch
    ):
        """With min_faithfulness=2.0 no real circuit can pass, so discover returns None."""
        mock_coact = _make_mock_coact()
        mock_stats = MockLatentStats(N_COMP, D_SAE)
        import circuit.discovery.mlp_sparse_expansion as mse_mod
        monkeypatch.setattr(mse_mod, "top_coactivation", mock_coact)
        monkeypatch.setattr(mse_mod, "latent_stats",     mock_stats)

        algo = MlpSparseExpansion(
            inference=MockInference(mock_model), sae_bank=mock_sae_bank,
            avg_acts=torch.zeros(N_COMP, D_SAE), probe_builder=None,
            min_faithfulness=2.0,   # impossible
            min_active_count=50,
        )
        algo.build_probe_dataset = lambda *a, **kw: _make_probe()
        assert algo.discover(SEED_COMP, SEED_LAT) is None

    def test_accepted_circuit_has_all_metadata_keys(self, setup):
        """A passing circuit must carry all expected metadata keys with correct values."""
        algo, _, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        required = {
            "faithfulness", "sufficiency", "completeness",
            "seed_comp", "seed_latent",
            "n_nodes", "n_edges",
            "discovery_method", "coact_depth", "n_passthrough",
        }
        assert required.issubset(circuit.metadata.keys())
        assert circuit.metadata["discovery_method"] == "mlp_sparse_expansion"
        assert circuit.metadata["seed_comp"]   == SEED_COMP
        assert circuit.metadata["seed_latent"] == SEED_LAT
        assert circuit.metadata["n_nodes"]     == len(circuit.nodes)
        assert circuit.metadata["n_edges"]     == len(circuit.edges)
