import pytest
import torch
from unittest.mock import MagicMock, patch

from circuit.discovery.top_coactivation import TopCoactivationDiscovery
from circuit.feature_id import FeatureID
from store.circuits import Circuit

# ---------------------------------------------------------------------------
# Local constants
# ---------------------------------------------------------------------------

N_LAYERS = 2
D_SAE    = 32
KINDS    = ["attn", "mlp", "resid"]
N_COMP   = N_LAYERS * len(KINDS)

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockTopCoactivation:
    def __init__(self):
        self.top_indices = torch.zeros(N_COMP, D_SAE, 8, dtype=torch.int32)
        self.top_values  = torch.zeros(N_COMP, D_SAE, 8, dtype=torch.float32)

class MockLatentStats:
    def __init__(self):
        self.active_count = torch.full((N_COMP, D_SAE), 100, dtype=torch.int64)

@pytest.fixture
def discovery_setup():
    inference = MagicMock()
    sae_bank = MagicMock()
    sae_bank.kinds = KINDS
    sae_bank.d_sae = D_SAE
    
    avg_acts = torch.zeros(N_LAYERS, len(KINDS), D_SAE)
    probe_builder = MagicMock()
    
    algo = TopCoactivationDiscovery(
        inference, sae_bank, avg_acts, probe_builder,
        min_faithfulness=0.1,
        attribution_threshold=0.001,
        pruning_threshold=0.0
    )
    
    # Use mocks for the statistical stores
    algo.top_coactivation = MockTopCoactivation()
    algo.latent_stats = MockLatentStats()
    
    return algo

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_top_coactivation_discovery_calls_feature_attribution(discovery_setup):
    algo = discovery_setup
    seed_comp, seed_lat = 2, 5  # Layer 0, Resid
    
    # Mock probe dataset
    probe_data = MagicMock()
    probe_data.pos_tokens = torch.zeros(2, 4, dtype=torch.long)
    probe_data.pos_argmax = torch.zeros(2, dtype=torch.long)
    probe_data.neg_tokens = torch.zeros(2, 4, dtype=torch.long)
    probe_data.target_tokens = torch.zeros(2, 4, dtype=torch.long)
    algo.build_probe_dataset = MagicMock(return_value=probe_data)
    
    # Setup a neighbor for the seed
    # neighbor: comp 1 (layer 0, mlp), latent 10
    neighbor_global = 1 * D_SAE + 10
    algo.top_coactivation.top_indices[seed_comp, seed_lat, 0] = neighbor_global
    algo.top_coactivation.top_values[seed_comp, seed_lat, 0] = 0.5
    
    # Mock inference.forward to return (tokens, logits, activations)
    algo.inference.forward = MagicMock(return_value=(None, None, None))
    
    # Mock inference.forward to not do anything but provide a graph via instrument
    with patch("circuit.discovery.top_coactivation.SAEGraphInstrument") as mock_instrument_cls:
        mock_instrument = MagicMock()
        mock_instrument_cls.return_value = mock_instrument
        mock_instrument.graph = MagicMock()
        
        # Mock compute_feature_attribution to return a score for the neighbor
        with patch("circuit.discovery.top_coactivation.compute_feature_attribution") as mock_attr:
            neighbor_fid = FeatureID(0, "mlp", 10)
            mock_attr.return_value = {neighbor_fid: 0.1}
            
            # Mock evaluation functions
            with patch("circuit.discovery.top_coactivation.evaluate_faithfulness", return_value=0.5), \
                 patch("circuit.discovery.top_coactivation.evaluate_sufficiency", return_value=0.5), \
                 patch("circuit.discovery.top_coactivation.evaluate_completeness", return_value=0.5):
                
                circuit = algo.discover(seed_comp, seed_lat)
                
                assert circuit is not None
                assert len(circuit.nodes) == 2  # Seed + Neighbor
                
                # Check that compute_feature_attribution was called correctly
                assert mock_attr.called
                args, kwargs = mock_attr.call_args_list[0]
                assert kwargs["target_layer"] == 0
                assert kwargs["target_kind"] == "resid"
                assert kwargs["target_latent_idx"] == 5
                assert kwargs["candidate_nodes"] == [neighbor_fid]
