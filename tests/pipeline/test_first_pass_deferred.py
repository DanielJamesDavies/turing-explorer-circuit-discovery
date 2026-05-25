from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch

# Importing pipeline.first_pass pulls pipeline.runtime, which imports
# pipeline.distributed.devices. Avoid executing pipeline.distributed.__init__ in
# this focused unit test because that package re-exports controller modules and
# can trigger a runtime/second_pass import cycle during collection.
distributed_pkg = types.ModuleType("pipeline.distributed")
distributed_pkg.__path__ = [str(Path(__file__).parents[2] / "src" / "pipeline" / "distributed")]
sys.modules.setdefault("pipeline.distributed", distributed_pkg)

from pipeline import first_pass
from config import config


class FakeLoader:
    shard_id_ranges = [(1, 2)]

    def __len__(self) -> int:
        return 1

    def get_batches(self):
        yield torch.tensor([1], dtype=torch.int64), torch.tensor([[1, 2]], dtype=torch.int64)

    def get_batches_for_shards(self, _shard_ids: Sequence[int]):
        return self.get_batches()

    def num_batches_for_shards(self, _shard_ids: Sequence[int]) -> int:
        return 1


class FakeSeqRepr:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def update(self, _sequence_ids, _activations) -> None:
        self.events.append("seq_repr")

    def print_stats(self) -> None:
        pass


class FakeModel:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def forward(
        self,
        tokens,
        *,
        num_gen: int,
        tokenize_final: bool,
        activations_callback,
        return_activations: bool,
    ):
        del tokens, num_gen, tokenize_final, return_activations
        for layer_idx in range(2):
            self.events.append(f"callback:{layer_idx}")
            acts = tuple(torch.full((1, 2, 4), layer_idx + kind_idx, dtype=torch.float32) for kind_idx in range(3))
            activations_callback(layer_idx, acts)
        self.events.append("forward_done")
        return None, torch.zeros((1, 2, 8), dtype=torch.float32), None


def _make_runtime(events: list[str]):
    return SimpleNamespace(
        bank=SimpleNamespace(n_layer=2, kinds=["attn", "mlp", "resid"]),
        seq_repr=FakeSeqRepr(events),
        loader=FakeLoader(),
        model=FakeModel(events),
        device=torch.device("cpu"),
        multi_gpu=False,
        mid_ctx_warmup=0,
    )


def _run_mode(monkeypatch, mode: str) -> list[str]:
    events: list[str] = []
    runtime = _make_runtime(events)
    monkeypatch.setattr(config.first_pass, "sae_encode_mode", mode)
    monkeypatch.setattr(config.latents.seq_latent_index, "enabled", False)
    monkeypatch.setattr(first_pass, "get_runtime", lambda: runtime)

    def fake_encode_layer_components(bank, layer_idx, activations, *, primary_device, multi_gpu):
        del bank, activations, primary_device, multi_gpu
        events.append(f"encode:{layer_idx}")
        latents = (
            torch.ones((1, 2, 2), dtype=torch.float32),
            torch.full((1, 2, 2), layer_idx, dtype=torch.int64),
        )
        return {layer_idx: latents}

    def fake_update_stores(mid_ctx_warmup, current_batch_last_latents, comp_idx, sequence_ids, latents):
        del mid_ctx_warmup, sequence_ids
        events.append(f"update:{comp_idx}")
        current_batch_last_latents[comp_idx] = latents[1][:, -1, :].detach()

    class FakeLogitCtx:
        def update(self, current_batch_last_latents, probs) -> None:
            del probs
            events.append(f"logit_ctx:{sorted(current_batch_last_latents)}")

    monkeypatch.setattr(first_pass, "encode_layer_components", fake_encode_layer_components)
    monkeypatch.setattr(first_pass, "_update_stores", fake_update_stores)
    monkeypatch.setattr(first_pass, "logit_ctx", FakeLogitCtx())

    first_pass.run_first_pass()
    return events


def test_deferred_sae_encode_waits_until_after_forward(monkeypatch) -> None:
    events = _run_mode(monkeypatch, "deferred")

    assert events.index("forward_done") < events.index("encode:0")
    assert events.index("encode:0") < events.index("update:0")
    assert events.index("encode:1") < events.index("update:1")
    assert events[-1] == "logit_ctx:[0, 1]"


def test_streaming_and_deferred_update_components_in_same_order(monkeypatch) -> None:
    streaming_events = _run_mode(monkeypatch, "streaming")
    deferred_events = _run_mode(monkeypatch, "deferred")

    streaming_updates = [event for event in streaming_events if event.startswith("update:")]
    deferred_updates = [event for event in deferred_events if event.startswith("update:")]
    assert streaming_updates == deferred_updates == ["update:0", "update:1"]
