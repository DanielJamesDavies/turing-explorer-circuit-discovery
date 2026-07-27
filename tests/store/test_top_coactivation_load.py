"""Loading a persisted TopCoactivation store (store/top_coactivation.py).

The artifact is authoritative for its own width (n_latents_per_latent) and
mode: values are baked at build time, so a store built at top-128/pmi must
load under a config that says top-64/freq_weighted. The old loader allocated
from the current config and crashed on copy_ — hidden behind a "likely no
file yet" message — which is exactly how the 20260531 run's 1.4GB artifact
went unreadable for days. The model grid (num_components, d_sae) stays a
hard requirement.
"""
import pytest
import torch

from store.top_coactivation import TopCoactivation


def _small_store(n_per_latent=4, num_components=2, d_sae=8):
    s = TopCoactivation(device=torch.device("cpu"))
    s.num_components = num_components
    s.d_sae = d_sae
    s.n_latents_per_latent = n_per_latent
    s._allocated = False
    return s


def _saved(tmp_path, n_per_latent=4, mode="pmi", freq_factors_none=True):
    src = _small_store(n_per_latent)
    src.allocate()
    src.top_indices.copy_(torch.arange(src.top_indices.numel(),
                                       dtype=torch.int32).reshape(src.top_indices.shape))
    src.top_values.fill_(2.5)
    src.total_tokens_processed = 777
    src._mode = mode
    p = str(tmp_path / "tc.pt")
    torch.save({
        "top_indices": src.top_indices,
        "top_values": src.top_values,
        "freq_factors": None if freq_factors_none else src.freq_factors,
        "total_tokens_processed": src.total_tokens_processed,
        "mode": mode,
    }, p)
    return p, src


class TestShapeAdoption:
    def test_wider_artifact_loads_under_narrower_config(self, tmp_path):
        """The 20260531 failure: artifact top-8, config top-4."""
        p, src = _saved(tmp_path, n_per_latent=8)
        dst = _small_store(n_per_latent=4)
        dst.load(p)
        assert dst.n_latents_per_latent == 8
        assert dst.top_indices.shape == src.top_indices.shape
        assert torch.equal(dst.top_indices, src.top_indices)
        assert dst.total_tokens_processed == 777

    def test_narrower_artifact_loads_under_wider_config(self, tmp_path):
        p, src = _saved(tmp_path, n_per_latent=2)
        dst = _small_store(n_per_latent=4)
        dst.load(p)
        assert dst.n_latents_per_latent == 2
        assert torch.equal(dst.top_values, src.top_values)

    def test_exact_match_unchanged(self, tmp_path):
        p, src = _saved(tmp_path, n_per_latent=4)
        dst = _small_store(n_per_latent=4)
        dst.load(p)
        assert dst.n_latents_per_latent == 4
        assert torch.equal(dst.top_indices, src.top_indices)

    def test_wrong_model_grid_is_rejected_not_adopted(self, tmp_path, capsys):
        """A store for a different model/SAE must not be adopted."""
        p, _ = _saved(tmp_path, n_per_latent=4)
        dst = _small_store(n_per_latent=4, d_sae=16)      # different grid
        dst.load(p)
        out = capsys.readouterr().out
        assert "LOAD FAILED" in out and "grid" in out
        # store stays empty rather than half-loaded
        assert not dst._allocated or float(dst.top_values.abs().sum()) == 0.0


class TestModeAndValues:
    def test_stored_mode_is_adopted(self, tmp_path):
        p, _ = _saved(tmp_path, mode="pmi")
        dst = _small_store()
        dst._mode = "freq_weighted"
        dst.load(p)
        assert dst.mode == "pmi"

    def test_none_freq_factors_tolerated(self, tmp_path):
        """pmi artifacts store freq_factors=None; load must not crash."""
        p, _ = _saved(tmp_path, freq_factors_none=True)
        dst = _small_store()
        dst.load(p)
        assert torch.equal(dst.top_values,
                           torch.full_like(dst.top_values, 2.5))

    def test_missing_file_is_soft(self, tmp_path, capsys):
        dst = _small_store()
        dst.load(str(tmp_path / "nope.pt"))
        assert "not found" in capsys.readouterr().out
        assert not dst._allocated
