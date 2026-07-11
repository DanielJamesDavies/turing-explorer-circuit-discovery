from __future__ import annotations

import pytest
import torch

from store.context import build_global_sequence_ids_tensor
from utils.neg_context_selector import NegContextSelector


class FakeLoader:
    def __init__(self, batch_size: int = 2, events: list[str] | None = None):
        self.batch_size = batch_size
        self.events = events
        self.loaded_calls: list[list[int]] = []

    def get_batches_by_ids(self, sequence_ids, max_length=64):
        self.loaded_calls.append(list(sequence_ids))
        if self.events is not None:
            self.events.append(f"load:{list(sequence_ids)}")
        for start in range(0, len(sequence_ids), self.batch_size):
            ids = sequence_ids[start : start + self.batch_size]
            tokens = torch.zeros((len(ids), max_length), dtype=torch.long)
            tokens[:, 0] = torch.tensor(ids, dtype=torch.long)
            yield torch.tensor(ids, dtype=torch.int32), tokens


class CachedFakeLoader(FakeLoader):
    def __init__(self, batch_size: int = 2, events: list[str] | None = None, cached_ids: list[int] | None = None):
        super().__init__(batch_size=batch_size, events=events)
        self.cached_ids = list(cached_ids or [])
        self.preload_calls: list[list[int]] = []
        self.cache_metadata = {}

    def preload_sequence_tokens(self, sequence_ids, max_length=64, dtype=torch.int32, max_bytes=None):
        del max_bytes
        self.preload_calls.append(list(sequence_ids))
        if not self.cached_ids:
            self.cached_ids = list(sequence_ids)
        self.cache_metadata = {
            "requested_count": len(sequence_ids),
            "loaded_count": len(self.cached_ids),
            "max_length": max_length,
            "dtype": str(dtype).replace("torch.", ""),
            "bytes": len(self.cached_ids) * max_length * torch.empty((), dtype=dtype).element_size(),
        }
        return dict(self.cache_metadata)

    def has_token_cache(self, max_length=64):
        del max_length
        return bool(self.cached_ids)

    def get_cached_tokens_by_ids(self, sequence_ids, max_length=64, device=None):
        hit_ids = [int(seq_id) for seq_id in sequence_ids if int(seq_id) in set(self.cached_ids)]
        miss_ids = [int(seq_id) for seq_id in sequence_ids if int(seq_id) not in set(self.cached_ids)]
        tokens = torch.zeros((len(hit_ids), max_length), dtype=torch.long, device=device or torch.device("cpu"))
        if hit_ids:
            tokens[:, 0] = torch.tensor(hit_ids, dtype=torch.long, device=tokens.device)
        return hit_ids, tokens, miss_ids


class FakeInference:
    def disable_compile(self):
        pass

    def enable_compile(self):
        pass

    def forward(self, tokens, activations_callback=None, **_kwargs):
        if activations_callback is not None:
            resid = tokens.float().unsqueeze(-1).expand(*tokens.shape, 3)
            activations_callback(0, (resid,))


class FakeBank:
    kinds = ["resid"]
    d_sae = 3
    n_layer = 1
    device = torch.device("cpu")

    def __init__(self, vectors: dict[int, list[float]]):
        self.vectors = vectors

    def encode(self, act, kind, layer):
        del kind, layer
        if act.ndim == 3:
            batch, seq_len, _repr_dim = act.shape
        else:
            batch, seq_len = act.shape
        top_acts = torch.zeros((batch, seq_len, self.d_sae), dtype=torch.float32)
        top_indices = torch.arange(self.d_sae, dtype=torch.long).view(1, 1, -1).expand(batch, seq_len, -1).clone()
        for row in range(batch):
            seq_id = int(act[row, 0, 0].item() if act.ndim == 3 else act[row, 0].item())
            top_acts[row, 0, :] = torch.tensor(self.vectors.get(seq_id, [0.0, 0.0, 0.0]))
        return top_acts, top_indices


class FakeNegCtx:
    _allocated = True

    def __init__(self):
        self.ctx_seq_idx = torch.tensor(
            [
                [
                    [1, 2, 0, 2],
                    [3, 4, 0, 0],
                    [1, 2, 0, 0],
                ]
            ],
            dtype=torch.int32,
        )
        self._global_seq_ids_cache = None

    def cached_global_sequence_ids(self):
        return self._global_seq_ids_cache

    def set_global_sequence_ids_cache(self, sequence_ids):
        self._global_seq_ids_cache = torch.unique(
            sequence_ids.detach().cpu().to(torch.int64)[sequence_ids > 0],
            sorted=True,
        )
        return self._global_seq_ids_cache


class FakeCtx:
    _allocated = True

    def __init__(self, rows: dict[tuple[int, int], list[int]] | None = None):
        self.ctx_seq_idx = torch.zeros((1, 3, 4), dtype=torch.int32)
        self.ctx_seq_val = torch.zeros((1, 3, 4), dtype=torch.float32)
        rows = rows or {}
        for (comp_idx, latent_idx), ids in rows.items():
            self.ctx_seq_idx[comp_idx, latent_idx, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
            self.ctx_seq_val[comp_idx, latent_idx, : len(ids)] = 1.0


class FakeSeqRepr:
    def __init__(self, vectors: dict[int, list[float]], events: list[str] | None = None):
        self.vectors = vectors
        self.repr_dim = 3
        self.events = events
        self.requested_ids: list[list[int]] = []

    def get_repr(self, seq_ids):
        ids = [int(seq_id) for seq_id in seq_ids.detach().cpu().tolist()]
        self.requested_ids.append(ids)
        if self.events is not None:
            self.events.append(f"repr:{ids}")
        return torch.tensor([self.vectors.get(seq_id, [0.0, 0.0, 0.0]) for seq_id in ids], dtype=torch.float32)


def _selector(
    vectors: dict[int, list[float]],
    *,
    top_rows: dict[tuple[int, int], list[int]] | None = None,
    mid_rows: dict[tuple[int, int], list[int]] | None = None,
    events: list[str] | None = None,
    neg_ctx: FakeNegCtx | None = None,
    loader: FakeLoader | None = None,
) -> NegContextSelector:
    return NegContextSelector(
        FakeInference(),
        FakeBank(vectors),
        loader or FakeLoader(batch_size=1, events=events),
        neg_ctx or FakeNegCtx(),
        FakeSeqRepr(vectors, events=events),
        FakeCtx(top_rows),
        FakeCtx(mid_rows),
    )


def test_random_samples_real_global_negctx_ids_excludes_positives_and_filters_active(monkeypatch):
    selector = _selector(
        {
            1: [0.0, 0.0, 0.0],
            2: [0.0, 0.0, 1.0],
            3: [0.0, 0.0, 0.0],
            4: [0.0, 0.0, 0.0],
        },
        top_rows={(0, 2): [4]},
        mid_rows={(0, 2): [3]},
    )
    monkeypatch.setattr(selector, "_deterministic_shuffle", lambda ids, *_args: list(ids))

    selection = selector.select(
        0,
        2,
        "random",
        max_sequences=2,
        batch_size=1,
        selection_seed=0,
    )

    assert selection is not None
    assert selection.sequence_ids == [1]
    assert 3 not in selection.sequence_ids
    assert 4 not in selection.sequence_ids
    assert selection.tokens[:, 0].tolist() == selection.sequence_ids
    assert selection.metadata["filtered_seed_active"] == 1


def test_global_ids_are_deduped_and_ignore_sentinel_zero():
    selector = _selector({})

    assert selector.global_negctx_ids() == [1, 2, 3, 4]


def test_global_ids_are_cached_across_selectors_sharing_negctx(monkeypatch):
    neg_ctx = FakeNegCtx()
    first = _selector({}, neg_ctx=neg_ctx)
    second = _selector({}, neg_ctx=neg_ctx)
    calls = 0

    original = first._build_context_global_ids

    def tracked_build():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(first, "_build_context_global_ids", tracked_build)
    monkeypatch.setattr(second, "_build_context_global_ids", lambda: pytest.fail("cache was not reused"))

    assert first.global_negctx_ids() == [1, 2, 3, 4]
    assert second.global_negctx_ids() == [1, 2, 3, 4]
    assert calls == 1


def test_tensor_global_id_builder_is_sorted_unique_and_positive():
    ids = build_global_sequence_ids_tensor(torch.tensor([[4, 0, 2], [4, 1, 2]], dtype=torch.int32))

    assert ids.tolist() == [1, 2, 4]


def test_close_filters_seed_activating_candidates_and_tops_up(monkeypatch):
    vectors = {
        99: [1.0, 0.0, 1.0],  # posctx reference, seed latent 2 active
        1: [1.0, 0.0, 1.0],   # filtered: seed latent activates
        2: [0.9, 0.1, 0.0],
        3: [0.7, 0.3, 0.0],
        4: [0.1, 0.9, 0.0],
    }
    selector = _selector(vectors, top_rows={(0, 2): [99]})
    monkeypatch.setattr(selector, "_deterministic_shuffle", lambda ids, *_args: list(ids))

    selection = selector.select(
        0,
        2,
        "close",
        max_sequences=2,
        batch_size=1,
        selection_seed=0,
    )

    assert selection is not None
    assert selection.sequence_ids == [2, 3]
    assert selection.metadata["filtered_seed_active"] == 1
    assert selection.metadata["candidate_ids_scanned"] >= 3
    assert selection.metadata["ranking_source"] == "seq_repr"
    assert selection.metadata["ranking_method"] == "chunked_positive_set_topk"
    assert selection.metadata["reference_strategy"] == "positive_set_max"
    assert selection.metadata["ranking_device"] == "cpu"
    assert selection.metadata["reference_source"] == "seq_repr_top_ctx"


def test_distant_reuses_filter_and_reverses_similarity_ranking(monkeypatch):
    vectors = {
        99: [1.0, 0.0, 0.0],
        1: [0.0, 1.0, 1.0],   # filtered
        2: [0.9, 0.1, 0.0],
        3: [0.1, 0.9, 0.0],
        4: [0.8, 0.2, 0.0],
    }
    selector = _selector(vectors, top_rows={(0, 2): [99]})
    monkeypatch.setattr(selector, "_deterministic_shuffle", lambda ids, *_args: list(ids))

    selection = selector.select(
        0,
        2,
        "distant",
        max_sequences=2,
        batch_size=2,
        exact=True,
        selection_seed=0,
    )

    assert selection is not None
    assert selection.sequence_ids == [3, 4]
    assert selection.metadata["filtered_seed_active"] == 1
    assert selection.metadata["score_name"] == "one_minus_max_cosine_sim"


def test_close_uses_positive_set_not_centroid_for_multimodal_topctx():
    selector = _selector(
        {
            90: [1.0, 0.0, 0.0],
            91: [0.0, 1.0, 0.0],
            1: [0.7, 0.7, 0.0],  # closest to the centroid, not to either positive mode
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [-1.0, -1.0, 0.0],
        },
        top_rows={(0, 2): [90, 91]},
    )

    selection = selector.select(
        0,
        2,
        "close",
        max_sequences=2,
        batch_size=2,
        exact=True,
        selection_seed=0,
    )

    assert selection is not None
    assert set(selection.sequence_ids) == {2, 3}
    assert selection.metadata["reference_count"] == 2
    assert selection.metadata["reference_strategy"] == "positive_set_max"


def test_distant_uses_far_from_every_positive_mode():
    selector = _selector(
        {
            90: [1.0, 0.0, 0.0],
            91: [0.0, 1.0, 0.0],
            1: [0.7, 0.7, 0.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [-1.0, -1.0, 0.0],
        },
        top_rows={(0, 2): [90, 91]},
    )

    selection = selector.select(
        0,
        2,
        "distant",
        max_sequences=2,
        batch_size=2,
        exact=True,
        selection_seed=0,
    )

    assert selection is not None
    assert set(selection.sequence_ids) == {1, 4}
    assert 2 not in selection.sequence_ids
    assert 3 not in selection.sequence_ids


def test_close_ranks_before_loading_candidate_tokens():
    events: list[str] = []
    selector = _selector(
        {
            99: [1.0, 0.0, 0.0],
            1: [0.2, 0.8, 0.0],
            2: [0.9, 0.1, 0.0],
            3: [0.1, 0.9, 0.0],
            4: [0.8, 0.2, 0.0],
        },
        top_rows={(0, 2): [99]},
        events=events,
    )

    selection = selector.select(
        0,
        2,
        "close",
        max_sequences=2,
        batch_size=1,
        candidate_pool_size=2,
        selection_seed=0,
    )

    assert selection is not None
    assert selection.sequence_ids == [2, 4]
    assert events[:2] == ["repr:[99]", "repr:[1, 2, 3, 4]"]
    assert events[2:] == ["load:[2]", "load:[4]"]


def test_load_window_preserves_ranked_order_while_filtering(monkeypatch):
    selector = _selector(
        {
            1: [0.0, 0.0, 1.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [-1.0, -1.0, 0.0],
        }
    )
    monkeypatch.setattr(
        selector,
        "_ranked_candidate_ids",
        lambda *_args, **_kwargs: ([1, 2, 3, 4], {"ranking_source": "test"}),
    )

    selection = selector.select(
        0,
        2,
        "close",
        max_sequences=2,
        batch_size=1,
        filter_batch_size=4,
        load_window_size=4,
        selection_seed=0,
    )

    assert selection is not None
    assert selection.sequence_ids == [2, 3]
    assert selection.metadata["load_window_size"] == 4
    assert selection.metadata["filter_batch_size"] == 4
    assert selector.loader.loaded_calls == [[1, 2, 3, 4]]


def test_selector_prefers_cached_tokens_and_reports_cache_metadata(monkeypatch):
    cached_loader = CachedFakeLoader(batch_size=1, cached_ids=[1, 2, 3, 4])
    selector = _selector(
        {
            1: [0.0, 0.0, 1.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [-1.0, -1.0, 0.0],
        },
        loader=cached_loader,
    )
    monkeypatch.setattr(
        selector,
        "_ranked_candidate_ids",
        lambda *_args, **_kwargs: ([1, 2, 3, 4], {"ranking_source": "test"}),
    )

    selection = selector.select(
        0,
        2,
        "close",
        max_sequences=2,
        batch_size=1,
        filter_batch_size=4,
        load_window_size=4,
        selection_seed=0,
    )

    assert selection is not None
    assert selection.sequence_ids == [2, 3]
    assert cached_loader.loaded_calls == []
    assert selection.metadata["token_cache_enabled"] is True
    assert selection.metadata["token_cache_preloaded"] is True
    assert selection.metadata["token_cache_hit_count"] == 4
    assert selection.metadata["token_cache_miss_count"] == 0


def test_selector_restores_ranked_order_with_cache_misses(monkeypatch):
    cached_loader = CachedFakeLoader(batch_size=1, cached_ids=[2, 4])
    selector = _selector(
        {
            1: [0.0, 0.0, 1.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [-1.0, -1.0, 0.0],
        },
        loader=cached_loader,
    )
    monkeypatch.setattr(
        selector,
        "_ranked_candidate_ids",
        lambda *_args, **_kwargs: ([1, 2, 3, 4], {"ranking_source": "test"}),
    )

    selection = selector.select(
        0,
        2,
        "close",
        max_sequences=2,
        batch_size=1,
        filter_batch_size=4,
        load_window_size=4,
        selection_seed=0,
    )

    assert selection is not None
    assert selection.sequence_ids == [2, 3]
    assert cached_loader.loaded_calls == [[1, 3]]
    assert selection.metadata["token_cache_hit_count"] == 2
    assert selection.metadata["token_cache_miss_count"] == 2


def test_topctx_reference_forwards_missing_cached_representations():
    events: list[str] = []
    selector = _selector(
        {},
        top_rows={(0, 2): [100]},
        events=events,
    )

    reference, metadata = selector.topctx_reference_repr(0, 2)

    assert reference.shape == (3,)
    assert torch.allclose(reference, torch.tensor([100.0 / 64.0, 100.0 / 64.0, 100.0 / 64.0]))
    assert metadata["reference_source"] == "seq_repr_top_ctx+forward_top_ctx"
    assert metadata["reference_forwarded_count"] == 1
    assert events == ["repr:[100]", "load:[100]"]


def test_topctx_reference_reprs_are_cached_per_seed():
    events: list[str] = []
    selector = _selector(
        {
            99: [1.0, 0.0, 0.0],
        },
        top_rows={(0, 2): [99]},
        events=events,
    )

    first_reps, first_metadata = selector.topctx_reference_reprs(0, 2)
    second_reps, second_metadata = selector.topctx_reference_reprs(0, 2)

    assert torch.equal(first_reps, second_reps)
    assert first_metadata["reference_cache_hit"] is False
    assert second_metadata["reference_cache_hit"] is True
    assert events == ["repr:[99]"]


def test_seed_activation_cache_avoids_repeated_forwards(monkeypatch):
    selector = _selector(
        {
            1: [0.0, 0.0, 1.0],
            2: [1.0, 0.0, 0.0],
        }
    )
    loaded_ids, tokens = selector.load_tokens([1, 2], max_length=64)
    calls = 0
    original = selector.collect_seed_max_activations

    def tracked_collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(selector, "collect_seed_max_activations", tracked_collect)

    first, first_hits, first_misses = selector._cached_seed_max_activations(
        tokens,
        loaded_ids,
        0,
        2,
        batch_size=2,
    )
    second, second_hits, second_misses = selector._cached_seed_max_activations(
        tokens,
        loaded_ids,
        0,
        2,
        batch_size=2,
    )

    assert torch.equal(first, second)
    assert (first_hits, first_misses) == (0, 2)
    assert (second_hits, second_misses) == (2, 0)
    assert calls == 1


def test_positive_set_similarity_cpu_and_cuda_match_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    selector = _selector({})
    reps = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.7, 0.0],
        ],
        dtype=torch.float32,
    )
    references = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    cpu_refs = selector._normalized_reference_reps(references, torch.device("cpu"))
    cuda_refs = selector._normalized_reference_reps(references, torch.device("cuda:0"))
    cpu_scores = selector._positive_set_max_similarity(reps, cpu_refs, torch.device("cpu"))
    cuda_scores = selector._positive_set_max_similarity(reps, cuda_refs, torch.device("cuda:0"))

    assert torch.allclose(cpu_scores, cuda_scores, atol=1e-6)
