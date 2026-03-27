"""
tests/circuit/discovery/test_attn_sparse_expansion.py

Tests for the AttnTopCoactSparseExpansion circuit discovery algorithm.
Mirrors test_mlp_sparse_expansion.py exactly — only the seed component,
neighbor global indices, and kind strings are changed for "attn".

Mock co-activation data
-----------------------
KINDS = ["attn", "mlp", "resid"]  →  comp % 3:  0=attn, 1=mlp, 2=resid

Seed: comp=0 (layer=0, kind="attn"), latent=5

Hop-1 neighbors of (comp=0, lat=5):
    3*32+ 7 = 103 → comp=3 (layer=1, attn)  ← kept
    3*32+ 8 = 104 → comp=3 (layer=1, attn)  ← kept
    1*32+ 3 =  35 → comp=1 (layer=0, mlp)   ← filtered
    3*32+ 9 = 105 → comp=3 (layer=1, attn)  ← kept
    2*32+ 5 =  69 → comp=2 (layer=0, resid) ← filtered
    3*32+11 = 107 → comp=3 (layer=1, attn)  ← kept
    4*32+ 2 = 130 → comp=4 (layer=1, mlp)   ← filtered
    3*32+13 = 109 → comp=3 (layer=1, attn)  ← kept

Hop-2 neighbors of (comp=3, lat=7):
    3*32+20 = 116 → comp=3 (layer=1, attn)  ← kept
    5*32+15 = 175 → comp=5 (layer=1, resid) ← filtered
    3*32+21 = 117 → comp=3 (layer=1, attn)  ← kept
    3*32+22 = 118 → comp=3 (layer=1, attn)  ← kept
"""
import pytest
import torch

from circuit.discovery.top_coact_expansion.attn_top_coact_sparse_expansion import AttnTopCoactSparseExpansion
from circuit.probe_dataset import ProbeDataset

# ---------------------------------------------------------------------------
# Local constants
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

# Seed: layer=0, kind="attn"  →  comp = 0*3 + 0 = 0
SEED_COMP = 0
SEED_LAT  = 5

_HOP1_GLOBALS = [
    3 * D_SAE +  7,   # comp=3 (layer=1, attn)  ← attn ✓
    3 * D_SAE +  8,   # comp=3 (layer=1, attn)  ← attn ✓
    1 * D_SAE +  3,   # comp=1 (layer=0, mlp)   ← skipped
    3 * D_SAE +  9,   # comp=3 (layer=1, attn)  ← attn ✓
    2 * D_SAE +  5,   # comp=2 (layer=0, resid) ← skipped
    3 * D_SAE + 11,   # comp=3 (layer=1, attn)  ← attn ✓
    4 * D_SAE +  2,   # comp=4 (layer=1, mlp)   ← skipped
    3 * D_SAE + 13,   # comp=3 (layer=1, attn)  ← attn ✓
]

_HOP2_GLOBALS_FOR_3_7 = [
    3 * D_SAE + 20,   # comp=3 (layer=1, attn)  ← attn ✓
    5 * D_SAE + 15,   # comp=5 (layer=1, resid) ← skipped
    3 * D_SAE + 21,   # comp=3 (layer=1, attn)  ← attn ✓
    3 * D_SAE + 22,   # comp=3 (layer=1, attn)  ← attn ✓
]


# ---------------------------------------------------------------------------
# Lightweight mock data stores
# ---------------------------------------------------------------------------

class MockTopCoactivation:
    def __init__(self, n_comp: int, d_sae: int, n_neighbors: int = 8):
        self.top_indices = torch.zeros(n_comp, d_sae, n_neighbors, dtype=torch.int32)
        self.top_values  = torch.zeros(n_comp, d_sae, n_neighbors, dtype=torch.float32)

    def set_neighbors(self, comp: int, lat: int, global_indices: list):
        n = len(global_indices)
        assert n <= self.top_indices.shape[2]
        self.top_indices[comp, lat, :n] = torch.tensor(global_indices, dtype=torch.int32)
        self.top_values[comp, lat, :n] = (
            torch.arange(n, 0, -1, dtype=torch.float32) * 0.1
        )


class MockLatentStats:
    def __init__(self, n_comp: int, d_sae: int, default_count: int = 100):
        self.active_count = torch.full(
            (n_comp, d_sae), default_count, dtype=torch.int64
        )


def _make_mock_coact() -> MockTopCoactivation:
    coact = MockTopCoactivation(N_COMP, D_SAE, n_neighbors=8)
    coact.set_neighbors(SEED_COMP, SEED_LAT, _HOP1_GLOBALS)
    coact.set_neighbors(3, 7, _HOP2_GLOBALS_FOR_3_7)
    return coact


# ---------------------------------------------------------------------------
# MockInference
# ---------------------------------------------------------------------------

class MockInference:
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


def _make_probe(b: int = B_TEST) -> ProbeDataset:
    return ProbeDataset(
        pos_tokens    = torch.zeros(b, T_TEST, dtype=torch.long),
        target_tokens = torch.zeros(b, T_TEST, dtype=torch.long),
        neg_tokens    = torch.zeros(b, T_TEST, dtype=torch.long),
        pos_argmax    = torch.tensor([1, 2][:b]),
        metadata      = {},
    )


# ---------------------------------------------------------------------------
# Core fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def setup(mock_model, mock_sae_bank, monkeypatch):
    mock_coact = _make_mock_coact()
    mock_stats = MockLatentStats(N_COMP, D_SAE)

    import circuit.discovery.top_coact_expansion.attn_top_coact_sparse_expansion as mod
    monkeypatch.setattr(mod, "top_coactivation", mock_coact)
    monkeypatch.setattr(mod, "latent_stats",     mock_stats)

    algo = AttnTopCoactSparseExpansion(
        inference        = MockInference(mock_model),
        sae_bank         = mock_sae_bank,
        avg_acts         = torch.zeros(N_COMP, D_SAE),
        probe_builder    = None,
        min_faithfulness = 0.0,
        min_active_count = 50,
        pruning_threshold= 0.0,
        probe_batch_size = 16,
    )
    algo.build_probe_dataset = lambda *a, **kw: _make_probe()
    return algo, mock_coact, mock_stats


# ---------------------------------------------------------------------------
# TestAttnNodeExpansion
# ---------------------------------------------------------------------------

class TestAttnNodeExpansion:

    def test_seed_node_is_attn_with_seed_role(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        seed_nodes = [n for n in circuit.nodes.values() if n.metadata["role"] == "seed"]
        assert len(seed_nodes) == 1
        assert seed_nodes[0].metadata["kind"] == "attn"

    def test_hop1_nodes_all_attn(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop1 = [n for n in circuit.nodes.values() if n.metadata["role"] == "hop1"]
        assert len(hop1) > 0
        for node in hop1:
            assert node.metadata["kind"] == "attn", (
                f"hop-1 node has non-attn kind: {node.metadata['kind']}"
            )

    def test_hop1_count_respects_limit(self, mock_model, mock_sae_bank, monkeypatch):
        mock_coact = _make_mock_coact()
        mock_stats = MockLatentStats(N_COMP, D_SAE)
        import circuit.discovery.top_coact_expansion.attn_top_coact_sparse_expansion as mod
        monkeypatch.setattr(mod, "top_coactivation", mock_coact)
        monkeypatch.setattr(mod, "latent_stats",     mock_stats)

        algo = AttnTopCoactSparseExpansion(
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

    def test_hop2_nodes_all_attn(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop2 = [n for n in circuit.nodes.values() if n.metadata["role"] == "hop2"]
        for node in hop2:
            assert node.metadata["kind"] == "attn", (
                f"hop-2 node has non-attn kind: {node.metadata['kind']}"
            )

    def test_no_duplicate_nodes(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        seen: set = set()
        for node in circuit.nodes.values():
            key = (node.metadata["layer_idx"], node.metadata["kind"], node.metadata["latent_idx"])
            assert key not in seen, f"Duplicate node: {key}"
            seen.add(key)


# ---------------------------------------------------------------------------
# TestPassthroughNodes
# ---------------------------------------------------------------------------

class TestPassthroughNodes:

    def test_passthrough_nodes_are_present(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        pt = [n for n in circuit.nodes.values() if n.metadata["role"] == "passthrough"]
        assert len(pt) > 0

    def test_passthrough_nodes_are_mlp_or_resid(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata["role"] == "passthrough":
                assert node.metadata["kind"] in ("mlp", "resid"), (
                    f"Passthrough node has unexpected kind: {node.metadata['kind']}"
                )

    def test_passthrough_set_matches_captured_latents(self, setup):
        algo, _, _ = setup
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

    def test_attn_nodes_are_never_passthrough(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata["kind"] == "attn":
                assert node.metadata["role"] != "passthrough"


# ---------------------------------------------------------------------------
# TestCircuitEdges
# ---------------------------------------------------------------------------

class TestCircuitEdges:

    def test_seed_is_edge_source(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        seed_node = next(n for n in circuit.nodes.values() if n.metadata["role"] == "seed")
        assert seed_node.uuid in {e.source_uuid for e in circuit.edges}

    def test_hop1_nodes_are_edge_targets(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop1_uuids = {n.uuid for n in circuit.nodes.values() if n.metadata["role"] == "hop1"}
        target_uuids = {e.target_uuid for e in circuit.edges}
        assert hop1_uuids.issubset(target_uuids)

    def test_hop2_nodes_are_edge_targets(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        hop2_uuids = {n.uuid for n in circuit.nodes.values() if n.metadata["role"] == "hop2"}
        if not hop2_uuids:
            pytest.skip("no hop-2 nodes")
        assert hop2_uuids.issubset({e.target_uuid for e in circuit.edges})

    def test_passthrough_nodes_have_no_edges(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        pt_uuids = {n.uuid for n in circuit.nodes.values() if n.metadata["role"] == "passthrough"}
        if not pt_uuids:
            pytest.skip("no passthrough nodes")
        edge_uuids = (
            {e.source_uuid for e in circuit.edges} |
            {e.target_uuid for e in circuit.edges}
        )
        assert not (pt_uuids & edge_uuids)


# ---------------------------------------------------------------------------
# TestActivityFilter
# ---------------------------------------------------------------------------

class TestActivityFilter:

    def test_low_count_latent_excluded_from_hop1(
        self, mock_model, mock_sae_bank, monkeypatch
    ):
        mock_coact = _make_mock_coact()
        mock_stats = MockLatentStats(N_COMP, D_SAE)
        mock_stats.active_count[3, 8] = 0   # comp=3, lat=8 is a hop-1 attn candidate

        import circuit.discovery.top_coact_expansion.attn_top_coact_sparse_expansion as mod
        monkeypatch.setattr(mod, "top_coactivation", mock_coact)
        monkeypatch.setattr(mod, "latent_stats",     mock_stats)

        algo = AttnTopCoactSparseExpansion(
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
        assert 8 not in hop1_lats

    def test_all_hop_nodes_pass_activity_threshold(self, setup):
        algo, _, mock_stats = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata["role"] not in ("hop1", "hop2"):
                continue
            layer    = node.metadata["layer_idx"]
            lat      = node.metadata["latent_idx"]
            kind_idx = KINDS.index(node.metadata["kind"])
            comp     = layer * len(KINDS) + kind_idx
            assert mock_stats.active_count[comp, lat].item() >= algo.min_active_count


# ---------------------------------------------------------------------------
# TestRejectionCases
# ---------------------------------------------------------------------------

class TestRejectionCases:

    def test_mlp_seed_rejected(self, setup):
        algo, _, _ = setup
        assert algo.discover(1, 5) is None   # layer=0, kind="mlp"

    def test_resid_seed_rejected(self, setup):
        algo, _, _ = setup
        assert algo.discover(2, 3) is None   # layer=0, kind="resid"

    def test_empty_probe_dataset_rejected(self, setup):
        algo, _, _ = setup
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
        mock_coact = _make_mock_coact()
        mock_stats = MockLatentStats(N_COMP, D_SAE)
        import circuit.discovery.top_coact_expansion.attn_top_coact_sparse_expansion as mod
        monkeypatch.setattr(mod, "top_coactivation", mock_coact)
        monkeypatch.setattr(mod, "latent_stats",     mock_stats)

        algo = AttnTopCoactSparseExpansion(
            inference=MockInference(mock_model), sae_bank=mock_sae_bank,
            avg_acts=torch.zeros(N_COMP, D_SAE), probe_builder=None,
            min_faithfulness=2.0, min_active_count=50,
        )
        algo.build_probe_dataset = lambda *a, **kw: _make_probe()
        assert algo.discover(SEED_COMP, SEED_LAT) is None

    def test_accepted_circuit_has_all_metadata_keys(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        required = {
            "faithfulness", "sufficiency", "completeness",
            "seed_comp", "seed_latent", "n_nodes", "n_edges",
            "discovery_method", "coact_depth", "n_passthrough",
        }
        assert required.issubset(circuit.metadata.keys())
        assert circuit.metadata["discovery_method"] == "attn_top_coact_sparse_expansion"
        assert circuit.metadata["seed_comp"]   == SEED_COMP
        assert circuit.metadata["seed_latent"] == SEED_LAT
        assert circuit.metadata["n_nodes"]     == len(circuit.nodes)
        assert circuit.metadata["n_edges"]     == len(circuit.edges)
