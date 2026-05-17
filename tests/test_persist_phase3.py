import os

import numpy as np
import pandas as pd
import torch

from config import config
from pipeline import persist
from pipeline.runtime import PipelineRuntime, clear_runtime, set_runtime
from store import search_cache


def test_search_cache_disabled_and_deferred_do_not_generate(monkeypatch):
    calls = []

    def fail_generate(*_args, **_kwargs):
        calls.append("called")
        raise AssertionError("search cache should not be generated")

    monkeypatch.setattr(persist, "generate_search_cache", fail_generate)

    monkeypatch.setattr(config.persist, "search_cache_enabled", False)
    monkeypatch.setattr(config.persist, "build_search_cache_after_pipeline", True)
    assert persist.build_search_cache_if_enabled(object(), object(), object()) is False

    monkeypatch.setattr(config.persist, "search_cache_enabled", True)
    monkeypatch.setattr(config.persist, "build_search_cache_after_pipeline", False)
    assert persist.build_search_cache_if_enabled(object(), object(), object()) is False

    assert calls == []


def test_search_cache_inline_calls_generator(monkeypatch, tmp_path):
    calls = []

    def fake_generate(top_ctx_store, bank, loader, output_path):
        calls.append((top_ctx_store, bank, loader, output_path))

    monkeypatch.setattr(persist, "generate_search_cache", fake_generate)
    monkeypatch.setattr(config.persist, "search_cache_enabled", True)
    monkeypatch.setattr(config.persist, "build_search_cache_after_pipeline", True)

    top_ctx_store = object()
    bank = object()
    loader = object()
    output_path = str(tmp_path / "search_cache.parquet")

    assert persist.build_search_cache_if_enabled(
        top_ctx_store, bank, loader, output_path=output_path
    ) is True
    assert calls == [(top_ctx_store, bank, loader, output_path)]


def test_atomic_save_writes_tmp_then_replaces(monkeypatch, tmp_path):
    monkeypatch.setattr(config.persist, "atomic_saves", True)
    final_path = tmp_path / "artifact.pt"
    seen_paths = []

    def save_fn(path):
        seen_paths.append(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("ok")

    persist._save_artifact(str(final_path), save_fn)

    assert seen_paths == [f"{final_path}.tmp"]
    assert final_path.read_text(encoding="utf-8") == "ok"
    assert not os.path.exists(f"{final_path}.tmp")


def test_keep_model_loaded_for_neg_ctx_skips_offload(monkeypatch):
    model = object()
    bank = object()
    runtime = PipelineRuntime(
        fast=False,
        compile=False,
        devices=[torch.device("cpu")],
        device=torch.device("cpu"),
        cpu_device=torch.device("cpu"),
        multi_gpu=False,
        mid_ctx_warmup=0,
        model=model,
        bank=bank,
    )
    set_runtime(runtime)
    monkeypatch.setattr(config.hardware, "keep_model_loaded_for_neg_ctx", True)

    try:
        persist.offload_model_and_sae()
        assert runtime.model is model
        assert runtime.bank is bank
    finally:
        clear_runtime()


def test_search_cache_generator_writes_valid_parquet(monkeypatch, tmp_path):
    class FakeTokenizerWrapper:
        class tokenizer:
            @staticmethod
            def batch_decode(batch_tokens, skip_special_tokens=True):
                return [" ".join(str(tok) for tok in tokens) for tokens in batch_tokens]

    class FakeTopCtx:
        num_components = 1
        ctx_seq_idx = torch.tensor([[[1, 2], [2, 0]]], dtype=torch.int32)
        ctx_seq_val = torch.tensor([[[1.0, 0.5], [0.25, 0.0]]], dtype=torch.float32)

    class FakeLoader:
        shard_id_ranges = [(1, 2)]

        def load_shard_sequences(self, _shard_idx, local_indices):
            seqs = {
                0: np.asarray([11, 12], dtype=np.int64),
                1: np.asarray([21, 22], dtype=np.int64),
            }
            return {idx: seqs[idx] for idx in local_indices if idx in seqs}

    class FakeBank:
        kinds = ["attn", "mlp", "resid"]

    output_path = tmp_path / "search_cache.parquet"
    monkeypatch.setattr(search_cache, "Tokenizer", FakeTokenizerWrapper)

    search_cache.generate_search_cache(
        FakeTopCtx(),
        FakeBank(),
        FakeLoader(),
        output_path=str(output_path),
        n_sequences=2,
        component_chunk_size=1,
    )

    df = pd.read_parquet(output_path)
    assert list(df.columns) == ["component_idx", "latent_idx", "text", "layer", "kind"]
    assert len(df) == 2
    assert set(df["kind"]) == {"attn"}
    assert output_path.exists()
