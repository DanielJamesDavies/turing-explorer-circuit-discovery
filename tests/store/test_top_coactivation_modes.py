import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import tempfile
from pathlib import Path

# Mock the C++ extension before importing TopCoactivation
mock_reduce = MagicMock()
sys.modules['top_coactivation_reduce'] = mock_reduce

# Mock config
from pydantic import BaseModel

class MockTopCoactivationLatentsConfig(BaseModel):
    n_latents_per_latent: int = 2
    n_candidates_per_component: int = 2
    freq_alpha: float = 2.0
    mode: str = "freq_weighted"
    pmi_clamp_min: float = -5.0
    pmi_clamp_max: float = 10.0
    dump_device: str = "cpu"
    dump_profile: bool = True
    reduce_backend: str = "single_process"
    reduce_shards: int = 1
    reduce_shard_output_dir: str | None = None
    reduce_omp_threads: int | None = None
    reduce_schedule_chunk: int = 256

class MockLatentsConfig(BaseModel):
    top_coactivation: MockTopCoactivationLatentsConfig = MockTopCoactivationLatentsConfig()

class MockDiscoveryConfig(BaseModel):
    min_faithfulness: float = 0.3
    min_active_count: int = 1
    max_neighbors: int = 32

class MockConfig(BaseModel):
    latents: MockLatentsConfig = MockLatentsConfig()
    discovery: MockDiscoveryConfig = MockDiscoveryConfig()

mock_config = MockConfig()

# Patch the config in src/store/top_coactivation.py
with patch('store.top_coactivation.config', mock_config):
    from store.top_coactivation import TopCoactivation

class TestTopCoactivationModes(unittest.TestCase):

    def setUp(self):
        # Reset mock_reduce
        mock_reduce.reduce_topk.reset_mock()
        # Mock TuringLLMConfig and SAEConfig
        with patch('store.top_coactivation.TuringLLMConfig') as mock_llm, \
             patch('store.top_coactivation.SAEConfig') as mock_sae:
            mock_llm.return_value.n_layer = 1
            mock_sae.return_value.d_sae = 4
            mock_sae.return_value.k = 4
            self.tc = TopCoactivation(device=torch.device("cpu"))
            # num_components = 1 * 3 = 3
            # d_sae = 4
            # M = min(2*4, 3*2) = 6
    
    def test_mode_property(self):
        # Directly mock the config object in the store
        with patch('store.top_coactivation.config') as mock_c:
            mock_c.latents.top_coactivation.mode = "pmi"
            self.tc._mode = None # Reset cache
            self.assertEqual(self.tc.mode, "pmi")
            
            mock_c.latents.top_coactivation.mode = "raw"
            self.tc._mode = None # Reset cache
            self.assertEqual(self.tc.mode, "raw")

    def test_update_batch_raw_mode(self):
        self.tc._mode = "raw"
        self.tc.allocate()
        self.tc.prepare_dump([10, 11])
        
        # Mock component_latents
        # batch_size=2, T=2
        # Component 0: latent 1 fires with magnitude 10.0 and 20.0 in seq 10, 0.0 and 0.0 in seq 11
        top_acts = torch.tensor([
            [10.0, 20.0], # batch_idx 0 (sid 10)
            [0.0, 0.0]    # batch_idx 1 (sid 11)
        ]).unsqueeze(0) # [1, 2, 2]
        top_indices = torch.tensor([
            [1, 1], # batch_idx 0
            [0, 0]  # batch_idx 1
        ]).unsqueeze(0) # [1, 2, 2]
        
        component_latents = {0: (top_acts[0], top_indices[0])}
        batch_ids = torch.tensor([10, 11])
        
        self.tc.update_batch(batch_ids, component_latents)
        
        # Row 0 (sid 10): mean activation for id 1 is (10+20)/2 = 15.0
        # In raw mode, no freq factors applied.
        # Check candidate_vals[0] contains 15.0
        self.assertAlmostEqual(self.tc.candidate_vals[0, 0].item(), 15.0)

    def test_update_batch_freq_weighted_mode(self):
        self.tc._mode = "freq_weighted"
        self.tc.allocate()
        self.tc.prepare_dump([10])
        
        # Set freq_factors for component 0, latent 1
        # freq_factors index = comp_idx * d_sae + latent_idx = 0*4 + 1 = 1
        self.tc.freq_factors[1] = 0.5
        
        top_acts = torch.tensor([[10.0, 20.0]]) # [1, 2]
        top_indices = torch.tensor([[1, 1]]) # [1, 2]
        component_latents = {0: (top_acts, top_indices)}
        batch_ids = torch.tensor([10])
        
        self.tc.update_batch(batch_ids, component_latents)
        
        # mean=15.0, freq_factor=0.5 -> 7.5
        self.assertAlmostEqual(self.tc.candidate_vals[0, 0].item(), 7.5)

    def test_update_batch_pmi_mode_dump(self):
        self.tc._mode = "pmi"
        self.tc.allocate()
        self.tc.prepare_dump([10])
        self.tc.total_tokens_processed = 0
        
        # T=4
        # Latent 1 fires at 3 tokens in sequence 10
        top_acts = torch.tensor([[1.0, 2.0, 0.0, 3.0]]) # [1, 4]
        top_indices = torch.tensor([[1, 1, 1, 1]]) # [1, 4]
        component_latents = {0: (top_acts, top_indices)}
        batch_ids = torch.tensor([10])
        
        self.tc.update_batch(batch_ids, component_latents)
        
        # In PMI mode, we want binary presence count: 3 tokens fired.
        self.assertAlmostEqual(self.tc.candidate_vals[0, 0].item(), 3.0)
        self.assertEqual(self.tc.total_tokens_processed, 4)

    def test_candidate_profile_matches_update_batch_raw_mode(self):
        self.tc._mode = "raw"
        self.tc.allocate()
        self.tc.prepare_dump([10, 11])

        batch_ids = torch.tensor([10, 11])
        component_latents = {
            0: (
                torch.tensor([
                    [10.0, 20.0],
                    [1.0, 3.0],
                ]),
                torch.tensor([
                    [1, 1],
                    [2, 2],
                ]),
            ),
            1: (
                torch.tensor([
                    [5.0, 5.0],
                    [6.0, 2.0],
                ]),
                torch.tensor([
                    [0, 3],
                    [1, 1],
                ]),
            ),
        }

        profile = self.tc.compute_candidate_profile(
            batch_size=2,
            component_latents=component_latents,
        )
        self.tc.update_batch(batch_ids, component_latents)

        actual_m = profile.candidate_ids.shape[1]
        self.assertTrue(torch.equal(self.tc.candidate_ids[:, :actual_m], profile.candidate_ids.cpu()))
        self.assertTrue(torch.allclose(self.tc.candidate_vals[:, :actual_m], profile.candidate_vals.cpu()))
        self.assertEqual(profile.token_count, 4)

    def test_candidate_profile_matches_update_batch_freq_weighted_mode(self):
        self.tc._mode = "freq_weighted"
        self.tc.allocate()
        self.tc.prepare_dump([10])
        self.tc.freq_factors[1] = 0.5
        self.tc.freq_factors[4] = 2.0

        batch_ids = torch.tensor([10])
        component_latents = {
            0: (
                torch.tensor([[10.0, 20.0]]),
                torch.tensor([[1, 1]]),
            ),
            1: (
                torch.tensor([[4.0, 2.0]]),
                torch.tensor([[0, 0]]),
            ),
        }

        profile = self.tc.compute_candidate_profile(
            batch_size=1,
            component_latents=component_latents,
        )
        self.tc.update_batch(batch_ids, component_latents)

        actual_m = profile.candidate_ids.shape[1]
        self.assertTrue(torch.equal(self.tc.candidate_ids[:, :actual_m], profile.candidate_ids.cpu()))
        self.assertTrue(torch.allclose(self.tc.candidate_vals[:, :actual_m], profile.candidate_vals.cpu()))
        self.assertAlmostEqual(profile.candidate_vals[0, 0].item(), 7.5)

    def test_candidate_profile_matches_update_batch_pmi_mode(self):
        self.tc._mode = "pmi"
        self.tc.allocate()
        self.tc.prepare_dump([10])
        self.tc.total_tokens_processed = 0

        batch_ids = torch.tensor([10])
        component_latents = {
            0: (
                torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
                torch.tensor([[1, 1, 1, 1]]),
            ),
            1: (
                torch.tensor([[0.0, 4.0, 5.0, 0.0]]),
                torch.tensor([[2, 2, 3, 3]]),
            ),
        }

        profile = self.tc.compute_candidate_profile(
            batch_size=1,
            component_latents=component_latents,
        )
        self.tc.update_batch(batch_ids, component_latents)

        actual_m = profile.candidate_ids.shape[1]
        self.assertTrue(torch.equal(self.tc.candidate_ids[:, :actual_m], profile.candidate_ids.cpu()))
        self.assertTrue(torch.allclose(self.tc.candidate_vals[:, :actual_m], profile.candidate_vals.cpu()))
        self.assertEqual(profile.token_count, 4)
        self.assertEqual(self.tc.total_tokens_processed, 4)

    def test_pmi_postprocess(self):
        self.tc._mode = "pmi"
        self.tc.allocate()
        self.tc.total_tokens_processed = 1000
        
        # Mock top_indices and top_values (binary counts from reduce)
        # 3 components, 4 sae, 2 neighbors
        self.tc.top_indices = torch.zeros((3, 4, 2), dtype=torch.int32)
        self.tc.top_values = torch.zeros((3, 4, 2), dtype=torch.float32)
        
        # Target: comp 0, latent 0. Candidate: comp 0, latent 1 (global_id 1)
        self.tc.top_indices[0, 0, 0] = 1
        self.tc.top_values[0, 0, 0] = 10.0 # fired 10 times in target's context
        
        # seq_offsets: target 0 has 2 sequences of len 10
        seq_offsets = torch.zeros(13, dtype=torch.int64) # 3*4+1=13
        seq_offsets[1] = 2 # Target 0 has 2 sequences
        seq_targets_global = torch.tensor([0, 0])
        seq_len = 10 # 2 * 10 = 20 tokens total for target 0 context
        
        # context_rate = 10 / 20 = 0.5
        
        # active_count (global): latent 1 fired 100 times globally.
        # total_tokens_globally is derived as active_count[0].sum() // k_sae.
        # With k_sae=4 and total_tokens=1000 we need active_count[0].sum() = 4000.
        # Distribute: latent 1 = 100 (the candidate), rest of comp 0 = 3900 (filler).
        active_count = torch.zeros((3, 4))
        active_count[0, 1] = 100.0
        active_count[0, 0] = 3900.0  # filler so sum/k_sae = 4000/4 = 1000
        # global_rate[1] = 100 / 1000 = 0.1
        
        # PMI = log(0.5 / 0.1) = log(5.0) approx 1.609
        
        with patch('store.top_coactivation.config') as mock_c:
            mock_c.latents.top_coactivation.pmi_clamp_min = -5.0
            mock_c.latents.top_coactivation.pmi_clamp_max = 10.0
            self.tc._apply_pmi_postprocess(active_count, seq_offsets, seq_targets_global, seq_len)
        
        expected_pmi = torch.tensor(0.5 / 0.1).log().item()
        self.assertAlmostEqual(self.tc.top_values[0, 0, 0].item(), expected_pmi, places=5)

    def test_save_load_mode_and_tokens(self):
        self.tc.allocate()
        self.tc.total_tokens_processed = 1234
        self.tc._mode = "pmi"
        
        path = "test_tc_save.pt"
        self.tc.save(path)
        
        # Create a new instance and load
        with patch('store.top_coactivation.TuringLLMConfig') as mock_llm, \
             patch('store.top_coactivation.SAEConfig') as mock_sae:
            mock_llm.return_value.n_layer = 1
            mock_sae.return_value.d_sae = 4
            new_tc = TopCoactivation(device=torch.device("cpu"))
            new_tc.load(path)
            
            self.assertEqual(new_tc.total_tokens_processed, 1234)
            # Note: load() doesn't set new_tc._mode, it checks it against current config
            # But the save file contains it.
            
        if os.path.exists(path):
            os.remove(path)

    def test_target_sharded_reduce_stitches_flat_ranges(self):
        self.tc._mode = "raw"
        self.tc.n_latents_per_latent = 2
        self.tc.candidate_ids = torch.zeros((2, 3), dtype=torch.int32)
        self.tc.candidate_vals = torch.zeros((2, 3), dtype=torch.float32)
        self.tc.seq_id_to_row = {1: 0, 2: 1}

        flat_total = self.tc.num_components * self.tc.d_sae

        def fake_reduce_topk(*args, **kwargs):
            start = kwargs.get("target_start", 0)
            end = kwargs.get("target_end", flat_total)
            rows = end - start
            ids = torch.arange(start, end, dtype=torch.int32).view(rows, 1).repeat(1, self.tc.n_latents_per_latent)
            vals = torch.arange(start, end, dtype=torch.float32).view(rows, 1).repeat(1, self.tc.n_latents_per_latent)
            return ids, vals

        mock_reduce.reduce_topk.side_effect = fake_reduce_topk
        with patch('store.top_coactivation.config') as mock_c:
            mock_c.latents.top_coactivation.reduce_backend = "target_sharded"
            mock_c.latents.top_coactivation.reduce_shards = 5
            mock_c.latents.top_coactivation.reduce_shard_output_dir = None
            mock_c.latents.top_coactivation.reduce_omp_threads = None
            mock_c.latents.top_coactivation.reduce_schedule_chunk = 256
            self.tc.reduce(
                seq_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
                seq_targets_global=torch.tensor([0, 1], dtype=torch.int64),
                active_count=None,
            )

        self.assertEqual(self.tc.top_indices.shape, (3, 4, 2))
        flat_ids = self.tc.top_indices.reshape(flat_total, self.tc.n_latents_per_latent)
        self.assertTrue(torch.equal(flat_ids[:, 0], torch.arange(flat_total, dtype=torch.int32)))
        self.assertEqual(mock_reduce.reduce_topk.call_count, 5)

    def test_target_sharded_reduce_can_write_and_merge_partial_files(self):
        self.tc._mode = "raw"
        self.tc.n_latents_per_latent = 2
        self.tc.candidate_ids = torch.zeros((2, 3), dtype=torch.int32)
        self.tc.candidate_vals = torch.zeros((2, 3), dtype=torch.float32)
        self.tc.seq_id_to_row = {1: 0, 2: 1}

        flat_total = self.tc.num_components * self.tc.d_sae

        def fake_reduce_topk(*args, **kwargs):
            start = kwargs.get("target_start", 0)
            end = kwargs.get("target_end", flat_total)
            rows = end - start
            ids = torch.arange(start, end, dtype=torch.int32).view(rows, 1).repeat(1, self.tc.n_latents_per_latent)
            vals = (torch.arange(start, end, dtype=torch.float32) + 0.5).view(rows, 1).repeat(1, self.tc.n_latents_per_latent)
            return ids, vals

        mock_reduce.reduce_topk.side_effect = fake_reduce_topk
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('store.top_coactivation.config') as mock_c:
                mock_c.latents.top_coactivation.reduce_backend = "target_sharded"
                mock_c.latents.top_coactivation.reduce_shards = 3
                mock_c.latents.top_coactivation.reduce_shard_output_dir = tmpdir
                mock_c.latents.top_coactivation.reduce_omp_threads = None
                mock_c.latents.top_coactivation.reduce_schedule_chunk = 256
                self.tc.reduce(
                    seq_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
                    seq_targets_global=torch.tensor([0, 1], dtype=torch.int64),
                    active_count=None,
                )

            shard_files = sorted(Path(tmpdir).glob("shard_*.pt"))
            self.assertEqual(len(shard_files), 3)
            payload = torch.load(shard_files[0], map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema"], "top_coactivation_reduce_shard_v1")
            self.assertEqual(payload["target_start"], 0)
            self.assertEqual(payload["n_latents_per_latent"], 2)

        flat_ids = self.tc.top_indices.reshape(flat_total, self.tc.n_latents_per_latent)
        flat_vals = self.tc.top_values.reshape(flat_total, self.tc.n_latents_per_latent)
        self.assertTrue(torch.equal(flat_ids[:, 0], torch.arange(flat_total, dtype=torch.int32)))
        self.assertTrue(torch.equal(flat_vals[:, 0], torch.arange(flat_total, dtype=torch.float32) + 0.5))
        self.assertEqual(mock_reduce.reduce_topk.call_count, 3)

    def test_target_sharded_reduce_cleans_current_partial_files_on_failure(self):
        self.tc._mode = "raw"
        self.tc.n_latents_per_latent = 2
        self.tc.candidate_ids = torch.zeros((2, 3), dtype=torch.int32)
        self.tc.candidate_vals = torch.zeros((2, 3), dtype=torch.float32)
        self.tc.seq_id_to_row = {1: 0, 2: 1}

        flat_total = self.tc.num_components * self.tc.d_sae

        def fake_reduce_topk(*args, **kwargs):
            start = kwargs.get("target_start", 0)
            if start > 0:
                raise RuntimeError("synthetic shard failure")
            end = kwargs.get("target_end", flat_total)
            rows = end - start
            ids = torch.arange(start, end, dtype=torch.int32).view(rows, 1).repeat(1, self.tc.n_latents_per_latent)
            vals = torch.arange(start, end, dtype=torch.float32).view(rows, 1).repeat(1, self.tc.n_latents_per_latent)
            return ids, vals

        mock_reduce.reduce_topk.side_effect = fake_reduce_topk
        with tempfile.TemporaryDirectory() as tmpdir:
            unrelated = Path(tmpdir) / "keep_me.txt"
            unrelated.write_text("not a reducer shard")
            with patch('store.top_coactivation.config') as mock_c:
                mock_c.latents.top_coactivation.reduce_backend = "target_sharded"
                mock_c.latents.top_coactivation.reduce_shards = 3
                mock_c.latents.top_coactivation.reduce_shard_output_dir = tmpdir
                mock_c.latents.top_coactivation.reduce_omp_threads = None
                mock_c.latents.top_coactivation.reduce_schedule_chunk = 256
                with self.assertRaises(RuntimeError):
                    self.tc.reduce(
                        seq_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
                        seq_targets_global=torch.tensor([0, 1], dtype=torch.int64),
                        active_count=None,
                    )

            self.assertEqual(sorted(Path(tmpdir).glob("shard_*.pt")), [])
            self.assertEqual(sorted(Path(tmpdir).glob(".shard_*.pt.tmp")), [])
            self.assertTrue(unrelated.exists())

if __name__ == '__main__':
    unittest.main()
