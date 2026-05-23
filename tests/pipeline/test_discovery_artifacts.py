from pathlib import Path

import pytest
import torch

from config import config
from pipeline.discovery_artifacts import load_discovery_artifacts, validate_discovery_artifacts


def _write_discovery_artifacts(run_root: Path, *, mode: str | None = None) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    mode = mode or str(config.latents.top_coactivation.mode or "freq_weighted")
    shape = (2, 3)
    ctx_shape = (2, 3, 2)
    topk_shape = (2, 3, 4)
    torch.save(
        {
            "active_count": torch.ones(shape, dtype=torch.int64),
            "seq_count": torch.ones(shape, dtype=torch.int64),
            "mean_seq": torch.ones(shape, dtype=torch.float32),
        },
        run_root / "latent_stats.pt",
    )
    for name in ("top_ctx", "mid_ctx", "neg_ctx"):
        torch.save(
            {
                "ctx_seq_idx": torch.ones(ctx_shape, dtype=torch.int32),
                "ctx_seq_val": torch.ones(ctx_shape, dtype=torch.float32),
            },
            run_root / f"{name}.pt",
        )
    torch.save(
        {
            "latent_counts": torch.ones(shape, dtype=torch.int64),
            "top_tokens": torch.ones(ctx_shape, dtype=torch.int32),
            "top_probs": torch.ones(ctx_shape, dtype=torch.float32),
        },
        run_root / "logit_ctx.pt",
    )
    torch.save(
        {
            "top_indices": torch.ones(topk_shape, dtype=torch.int32),
            "top_values": torch.ones(topk_shape, dtype=torch.float32),
            "mode": mode,
        },
        run_root / "top_coactivation.pt",
    )
    torch.save([{"comp_idx": 1, "latent_idx": 2}], run_root / "candidates.pt")


def test_validate_discovery_artifacts_accepts_synthetic_fixture(tmp_path):
    run_root = tmp_path / "run"
    _write_discovery_artifacts(run_root)

    result = validate_discovery_artifacts(
        run_root,
        candidates_path=run_root / "candidates.pt",
    )

    assert result.component_count == 2
    assert result.d_sae == 3
    assert result.candidate_count == 1
    assert result.top_coactivation_mode == str(config.latents.top_coactivation.mode or "freq_weighted")


def test_validate_discovery_artifacts_reports_missing_inputs(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()

    with pytest.raises(FileNotFoundError, match="missing discovery input artifacts"):
        validate_discovery_artifacts(run_root, candidates_path=run_root / "candidates.pt")


def test_validate_discovery_artifacts_rejects_incompatible_shapes(tmp_path):
    run_root = tmp_path / "run"
    _write_discovery_artifacts(run_root)
    torch.save(
        {
            "ctx_seq_idx": torch.ones((99, 3, 2), dtype=torch.int32),
            "ctx_seq_val": torch.ones((99, 3, 2), dtype=torch.float32),
        },
        run_root / "top_ctx.pt",
    )

    with pytest.raises(ValueError, match="leading dimensions mismatch"):
        validate_discovery_artifacts(run_root, candidates_path=run_root / "candidates.pt")


def test_validate_discovery_artifacts_rejects_mode_mismatch(tmp_path):
    run_root = tmp_path / "run"
    configured_mode = str(config.latents.top_coactivation.mode or "freq_weighted")
    mismatched_mode = "raw" if configured_mode != "raw" else "pmi"
    _write_discovery_artifacts(run_root, mode=mismatched_mode)

    with pytest.raises(ValueError, match="top_coactivation mode mismatch"):
        validate_discovery_artifacts(run_root, candidates_path=run_root / "candidates.pt")


def test_load_discovery_artifacts_uses_shared_store_loaders(monkeypatch, tmp_path):
    run_root = tmp_path / "run"
    _write_discovery_artifacts(run_root)
    seen = {}

    class FakeStore:
        def __init__(self, key):
            self.key = key

        def load(self, path):
            seen[self.key] = Path(path)

    monkeypatch.setattr("pipeline.discovery_artifacts.latent_stats", FakeStore("latent_stats"))
    monkeypatch.setattr("pipeline.discovery_artifacts.top_ctx", FakeStore("top_ctx"))
    monkeypatch.setattr("pipeline.discovery_artifacts.mid_ctx", FakeStore("mid_ctx"))
    monkeypatch.setattr("pipeline.discovery_artifacts.neg_ctx", FakeStore("neg_ctx"))
    monkeypatch.setattr("pipeline.discovery_artifacts.logit_ctx", FakeStore("logit_ctx"))
    monkeypatch.setattr("pipeline.discovery_artifacts.top_coactivation", FakeStore("top_coactivation"))

    load_discovery_artifacts(run_root, candidates_path=run_root / "candidates.pt")

    assert seen == {
        "latent_stats": run_root / "latent_stats.pt",
        "top_ctx": run_root / "top_ctx.pt",
        "mid_ctx": run_root / "mid_ctx.pt",
        "neg_ctx": run_root / "neg_ctx.pt",
        "logit_ctx": run_root / "logit_ctx.pt",
        "top_coactivation": run_root / "top_coactivation.pt",
    }
