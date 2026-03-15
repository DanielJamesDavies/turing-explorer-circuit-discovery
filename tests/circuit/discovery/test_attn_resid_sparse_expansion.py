import pytest
import torch

from circuit.discovery.attn_resid_sparse_expansion import AttnResidSparseExpansion
from circuit.probe_dataset import ProbeDataset

N_LAYERS = 2
D_MODEL = 16
D_SAE = 32
KINDS = ["attn", "mlp", "resid"]
N_COMP = N_LAYERS * len(KINDS)

B_TEST = 2
T_TEST = 4
V_TEST = 20

SEED_COMP = 0  # layer=0, kind="attn"
SEED_LAT = 5

_HOP1_GLOBALS_FOR_ATTN_SEED = [
    3 * D_SAE + 7,   # attn  keep
    5 * D_SAE + 8,   # resid keep
    1 * D_SAE + 3,   # mlp   skip
    3 * D_SAE + 9,   # attn  keep
    2 * D_SAE + 5,   # resid keep
    4 * D_SAE + 2,   # mlp   skip
    5 * D_SAE + 11,  # resid keep
    3 * D_SAE + 13,  # attn  keep
]

_HOP1_GLOBALS_FOR_RESID_SEED = [
    5 * D_SAE + 7,   # resid keep
    3 * D_SAE + 8,   # attn  keep
    1 * D_SAE + 1,   # mlp   skip
    5 * D_SAE + 9,   # resid keep
    0 * D_SAE + 3,   # attn  keep
    4 * D_SAE + 6,   # mlp   skip
    3 * D_SAE + 11,  # attn  keep
    5 * D_SAE + 13,  # resid keep
]

_HOP2_GLOBALS_FOR_3_7 = [
    5 * D_SAE + 20,  # resid keep
    4 * D_SAE + 15,  # mlp   skip
    3 * D_SAE + 21,  # attn  keep
    2 * D_SAE + 22,  # resid keep
]


class MockTopCoactivation:
    def __init__(self, n_comp: int, d_sae: int, n_neighbors: int = 8):
        self.top_indices = torch.zeros(n_comp, d_sae, n_neighbors, dtype=torch.int32)
        self.top_values = torch.zeros(n_comp, d_sae, n_neighbors, dtype=torch.float32)

    def set_neighbors(self, comp: int, lat: int, global_indices: list):
        n = len(global_indices)
        assert n <= self.top_indices.shape[2]
        self.top_indices[comp, lat, :n] = torch.tensor(global_indices, dtype=torch.int32)
        self.top_values[comp, lat, :n] = torch.arange(n, 0, -1, dtype=torch.float32) * 0.1


class MockLatentStats:
    def __init__(self, n_comp: int, d_sae: int, default_count: int = 100):
        self.active_count = torch.full((n_comp, d_sae), default_count, dtype=torch.int64)


class MockInference:
    def __init__(self, model, d_model: int = D_MODEL, d_vocab: int = V_TEST):
        torch.manual_seed(7)
        self.model = model
        self.W_logit = torch.randn(d_model, d_vocab)
        torch.manual_seed(13)
        self._x = torch.randn(B_TEST, T_TEST, d_model)

    _compiled = False

    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(
        self,
        tokens,
        num_gen: int = 1,
        tokenize_final: bool = False,
        return_activations: bool = False,
        all_logits: bool = False,
        patcher=None,
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
        pos_tokens=torch.zeros(b, T_TEST, dtype=torch.long),
        target_tokens=torch.zeros(b, T_TEST, dtype=torch.long),
        neg_tokens=torch.zeros(b, T_TEST, dtype=torch.long),
        pos_argmax=torch.tensor([1, 2][:b]),
        metadata={},
    )


def _make_mock_coact() -> MockTopCoactivation:
    coact = MockTopCoactivation(N_COMP, D_SAE, n_neighbors=8)
    coact.set_neighbors(SEED_COMP, SEED_LAT, _HOP1_GLOBALS_FOR_ATTN_SEED)
    coact.set_neighbors(2, SEED_LAT, _HOP1_GLOBALS_FOR_RESID_SEED)
    coact.set_neighbors(3, 7, _HOP2_GLOBALS_FOR_3_7)
    return coact


@pytest.fixture
def setup(mock_model, mock_sae_bank, monkeypatch):
    mock_coact = _make_mock_coact()
    mock_stats = MockLatentStats(N_COMP, D_SAE)

    import circuit.discovery.attn_resid_sparse_expansion as mod

    monkeypatch.setattr(mod, "top_coactivation", mock_coact)
    monkeypatch.setattr(mod, "latent_stats", mock_stats)

    algo = AttnResidSparseExpansion(
        inference=MockInference(mock_model),
        sae_bank=mock_sae_bank,
        avg_acts=torch.zeros(N_COMP, D_SAE),
        probe_builder=None,
        min_faithfulness=0.0,
        min_active_count=50,
        pruning_threshold=0.0,
        probe_batch_size=16,
    )
    algo.build_probe_dataset = lambda *a, **kw: _make_probe()
    return algo, mock_coact, mock_stats


class TestAttnResidNodeExpansion:
    def test_seed_attn_accepted(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None

    def test_seed_resid_accepted(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(2, SEED_LAT)  # layer=0, kind="resid"
        assert circuit is not None

    def test_seed_mlp_rejected(self, setup):
        algo, _, _ = setup
        assert algo.discover(1, SEED_LAT) is None  # layer=0, kind="mlp"

    def test_hop_nodes_are_attn_or_resid_only(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        for node in circuit.nodes.values():
            if node.metadata["role"] in ("hop1", "hop2"):
                assert node.metadata["kind"] in ("attn", "resid")


class TestPassthroughNodes:
    def test_passthrough_nodes_are_mlp_only(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        passthrough = [n for n in circuit.nodes.values() if n.metadata["role"] == "passthrough"]
        assert len(passthrough) > 0
        for node in passthrough:
            assert node.metadata["kind"] == "mlp"

    def test_passthrough_nodes_have_no_edges(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        pt_uuids = {n.uuid for n in circuit.nodes.values() if n.metadata["role"] == "passthrough"}
        if not pt_uuids:
            pytest.skip("no passthrough nodes")
        edge_uuids = {e.source_uuid for e in circuit.edges} | {e.target_uuid for e in circuit.edges}
        assert not (pt_uuids & edge_uuids)


class TestActivityFilterAndMetadata:
    def test_low_count_latent_excluded_from_hop1(self, mock_model, mock_sae_bank, monkeypatch):
        mock_coact = _make_mock_coact()
        mock_stats = MockLatentStats(N_COMP, D_SAE)
        mock_stats.active_count[5, 8] = 0  # comp=5, lat=8 candidate should be filtered

        import circuit.discovery.attn_resid_sparse_expansion as mod

        monkeypatch.setattr(mod, "top_coactivation", mock_coact)
        monkeypatch.setattr(mod, "latent_stats", mock_stats)

        algo = AttnResidSparseExpansion(
            inference=MockInference(mock_model),
            sae_bank=mock_sae_bank,
            avg_acts=torch.zeros(N_COMP, D_SAE),
            probe_builder=None,
            min_faithfulness=0.0,
            min_active_count=50,
        )
        algo.build_probe_dataset = lambda *a, **kw: _make_probe()
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None

        hop1_keys = {
            (n.metadata["layer_idx"], n.metadata["kind"], n.metadata["latent_idx"])
            for n in circuit.nodes.values()
            if n.metadata["role"] == "hop1"
        }
        assert (1, "resid", 8) not in hop1_keys

    def test_accepted_circuit_has_metadata(self, setup):
        algo, _, _ = setup
        circuit = algo.discover(SEED_COMP, SEED_LAT)
        assert circuit is not None
        required = {
            "faithfulness",
            "sufficiency",
            "completeness",
            "seed_comp",
            "seed_latent",
            "n_nodes",
            "n_edges",
            "discovery_method",
            "coact_depth",
            "n_passthrough",
        }
        assert required.issubset(circuit.metadata.keys())
        assert circuit.metadata["discovery_method"] == "attn_resid_sparse_expansion"
