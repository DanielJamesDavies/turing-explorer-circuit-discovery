from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

import torch

from config import config
from pipeline.component_index import split_component_idx
from sae.dense import target_latent_activations
from store.context import build_global_sequence_ids_tensor


@dataclass
class NegContextSelection:
    tokens: torch.Tensor
    sequence_ids: list[int]
    mode: str
    metadata: dict[str, Any]


class NegContextSelector:
    """
    Select mode-aware negative context tokens from the saved global neg_ctx set.

    This utility is intentionally dependency-explicit: callers provide the model
    runtime, SAE bank, loader, and neg_ctx artifact rather than relying on global
    singletons. The selector can therefore be reused by discovery, analysis, and
    replay code while preserving one definition of random/close/distant negctx.
    """

    def __init__(
        self,
        inference: Any,
        bank: Any,
        loader: Any,
        neg_ctx: Any,
        seq_repr: Any,
        top_ctx: Any,
        mid_ctx: Any,
    ):
        self.inference = inference
        self.bank = bank
        self.loader = loader
        self.neg_ctx = neg_ctx
        self.seq_repr = seq_repr
        self.top_ctx = top_ctx
        self.mid_ctx = mid_ctx
        self._global_negctx_ids_cache: list[int] | None = None
        self._topctx_reference_reprs_cache: dict[tuple[int, int], tuple[torch.Tensor, dict[str, Any]]] = {}
        self._seed_activation_cache: dict[tuple[int, int, int, bool], float] = {}
        self._last_token_load_metadata: dict[str, Any] = {}

    def select(
        self,
        comp_idx: int,
        latent_idx: int,
        mode: str,
        max_sequences: int,
        batch_size: int,
        *,
        candidate_pool_size: Optional[int] = None,
        exact: bool = False,
        non_activation_threshold: float = 0.0,
        preact_filter: bool = False,
        preact_select: str = "cleanest",
        preact_max_frac: float = 0.0,
        posctx_reference: Optional[float] = None,
        selection_seed: int = 0,
        filter_batch_size: Optional[int] = None,
        load_window_size: Optional[int] = None,
        logger: Any | None = None,
    ) -> NegContextSelection | None:
        max_sequences = int(max_sequences)
        if max_sequences <= 0:
            self._reject(logger, f"neg_mode={mode}: max_sequences <= 0")
            return None

        # Under preact_filter the raw threshold is no longer "did the seed
        # appear in top-k" (which censors near-misses to exactly 0) but "how
        # close did the seed come to its typical firing level". The bar is a
        # FRACTION of the seed's posctx reference, so it is scale-free across
        # seeds whose natural activations differ by orders of magnitude.
        if preact_filter:
            if posctx_reference is None:
                raise ValueError(
                    "preact_filter=True needs posctx_reference (the seed's "
                    "typical posctx activation) to scale the threshold against."
                    " Refusing to fall back to an absolute bar: it would mean "
                    "something different for every seed.")
            non_activation_threshold = max(
                float(non_activation_threshold),
                float(preact_max_frac) * float(posctx_reference))
            self._last_preact_threshold = float(non_activation_threshold)

        if mode == "random":
            candidate_ids, ranking_metadata = self._random_candidate_ids(
                comp_idx,
                latent_idx,
                max_sequences=max_sequences,
                candidate_pool_size=candidate_pool_size,
                selection_seed=selection_seed,
            )
            return self._select_from_ordered_ids(
                comp_idx,
                latent_idx,
                mode,
                candidate_ids,
                max_sequences,
                batch_size,
                non_activation_threshold=float(non_activation_threshold),
                preact_filter=bool(preact_filter),
                preact_select=str(preact_select),
                selection_seed=selection_seed,
                filter_batch_size=filter_batch_size,
                load_window_size=load_window_size,
                ranking_metadata=ranking_metadata,
                logger=logger,
            )
        if mode in ("close", "distant"):
            candidate_ids, ranking_metadata = self._ranked_candidate_ids(
                comp_idx,
                latent_idx,
                mode,
                candidate_pool_size=None if exact else candidate_pool_size,
                max_sequences=max_sequences,
                exact=exact,
                selection_seed=selection_seed,
            )
            return self._select_from_ordered_ids(
                comp_idx,
                latent_idx,
                mode,
                candidate_ids,
                max_sequences,
                batch_size,
                non_activation_threshold=float(non_activation_threshold),
                preact_filter=bool(preact_filter),
                preact_select=str(preact_select),
                filter_batch_size=filter_batch_size,
                load_window_size=load_window_size,
                ranking_metadata={**ranking_metadata, "exact": bool(exact)},
                logger=logger,
            )
        if mode == "fused":
            return self._select_fused(
                comp_idx,
                latent_idx,
                max_sequences,
                batch_size,
                candidate_pool_size=candidate_pool_size,
                exact=exact,
                non_activation_threshold=float(non_activation_threshold),
                preact_filter=bool(preact_filter),
                preact_select=str(preact_select),
                selection_seed=selection_seed,
                filter_batch_size=filter_batch_size,
                load_window_size=load_window_size,
                logger=logger,
            )
        raise ValueError(f"Unknown neg_mode: {mode!r}")

    def _select_fused(
        self,
        comp_idx: int,
        latent_idx: int,
        max_sequences: int,
        batch_size: int,
        *,
        candidate_pool_size: Optional[int],
        exact: bool,
        non_activation_threshold: float,
        preact_filter: bool = False,
        preact_select: str = "cleanest",
        preact_max_frac: float = 0.0,
        posctx_reference: Optional[float] = None,
        selection_seed: int,
        filter_batch_size: Optional[int],
        load_window_size: Optional[int],
        logger: Any | None,
    ) -> NegContextSelection | None:
        """One contrast set drawing quotas from all three sub-modes.

        Quotas: close gets n//3 + remainder (the sharpest boundary signal),
        distant and random n//3 each. Sub-selections run through the normal
        per-mode paths (same ranking, same non-activation filter), then merge
        in close -> distant -> random order with sequence-id dedup (first
        occurrence wins, so a sequence that is both a hard negative and a
        random draw counts against the close quota). Dedup losses and
        sub-mode shortfalls are NOT topped up — the fused set may come in
        under max_sequences; per-mode counts land in metadata. The distant
        quota uses the generic candidate pool rather than distant_pool_size
        (callers resolve that knob for pure-"distant" only); the pool is
        clamped >= quota by _candidate_pool_limit either way.
        """

        quota = max_sequences // 3
        quotas = {
            "close": quota + (max_sequences - 3 * quota),
            "distant": quota,
            "random": quota,
        }
        merged_ids: list[int] = []
        merged_tokens: list[torch.Tensor] = []
        seen: set[int] = set()
        per_mode_counts: dict[str, int] = {}
        sub_metadata: dict[str, Any] = {}
        for sub_mode in ("close", "distant", "random"):
            sub_quota = quotas[sub_mode]
            if sub_quota <= 0:
                per_mode_counts[sub_mode] = 0
                continue
            selection = self.select(
                comp_idx,
                latent_idx,
                sub_mode,
                sub_quota,
                batch_size,
                candidate_pool_size=candidate_pool_size,
                exact=exact,
                # already resolved to an absolute bar above; passing the
                # reference through keeps the recursion idempotent.
                non_activation_threshold=non_activation_threshold,
                preact_filter=preact_filter,
                preact_select=preact_select,
                preact_max_frac=preact_max_frac,
                posctx_reference=posctx_reference,
                selection_seed=selection_seed,
                filter_batch_size=filter_batch_size,
                load_window_size=load_window_size,
                logger=logger,
            )
            if selection is None:
                per_mode_counts[sub_mode] = 0
                sub_metadata[sub_mode] = {"selected_count": 0, "rejected": True}
                continue
            kept = 0
            for row, seq_id in enumerate(selection.sequence_ids):
                seq_id_int = int(seq_id)
                if seq_id_int in seen:
                    continue
                seen.add(seq_id_int)
                merged_ids.append(seq_id_int)
                merged_tokens.append(selection.tokens[row])
                kept += 1
            per_mode_counts[sub_mode] = kept
            sub_metadata[sub_mode] = {
                "selected_count": int(selection.tokens.shape[0]),
                "kept_after_dedup": kept,
            }
        if not merged_ids:
            self._reject(logger, "neg_mode=fused: every sub-mode came back empty")
            return None
        tokens = torch.stack(merged_tokens, dim=0).to(self.bank.device)
        self._note(
            logger,
            f"neg_mode=fused: selected {tokens.shape[0]} sequences "
            f"(close={per_mode_counts.get('close', 0)} "
            f"distant={per_mode_counts.get('distant', 0)} "
            f"random={per_mode_counts.get('random', 0)})",
        )
        return NegContextSelection(
            tokens=tokens,
            sequence_ids=merged_ids,
            mode="fused",
            metadata={
                "fused_quotas": quotas,
                "fused_counts": per_mode_counts,
                "selected_count": int(tokens.shape[0]),
                "submodes": sub_metadata,
            },
        )

    def global_negctx_ids(self) -> list[int]:
        if self._global_negctx_ids_cache is not None:
            return self._global_negctx_ids_cache
        if not getattr(self.neg_ctx, "_allocated", False):
            self._global_negctx_ids_cache = []
            return []
        cached = self._cached_context_global_ids()
        if cached is None:
            cached = self._build_context_global_ids()
            setter = getattr(self.neg_ctx, "set_global_sequence_ids_cache", None)
            if callable(setter):
                cached = cast(torch.Tensor, setter(cached))
        self._global_negctx_ids_cache = [int(seq_id) for seq_id in cached.detach().cpu().tolist()]
        return self._global_negctx_ids_cache

    def _cached_context_global_ids(self) -> torch.Tensor | None:
        getter = getattr(self.neg_ctx, "cached_global_sequence_ids", None)
        if not callable(getter):
            return None
        cached = getter()
        if cached is None:
            return None
        if not isinstance(cached, torch.Tensor):
            cached = torch.as_tensor(cached, dtype=torch.int64)
        return cast(torch.Tensor, cached).detach().cpu().to(torch.int64).reshape(-1)

    def _build_context_global_ids(self) -> torch.Tensor:
        return build_global_sequence_ids_tensor(self.neg_ctx.ctx_seq_idx)

    def seed_row_negctx_ids(self, comp_idx: int, latent_idx: int) -> list[int]:
        if not getattr(self.neg_ctx, "_allocated", False):
            return []
        raw = self.neg_ctx.ctx_seq_idx[comp_idx, latent_idx].detach().cpu().to(torch.int64)
        ids: list[int] = []
        seen: set[int] = set()
        for seq_id in raw.tolist():
            seq_id_int = int(seq_id)
            if seq_id_int > 0 and seq_id_int not in seen:
                ids.append(seq_id_int)
                seen.add(seq_id_int)
        return ids

    def positive_context_ids(self, comp_idx: int, latent_idx: int) -> tuple[list[int], list[int]]:
        top_ids = self._ctx_row_ids(self.top_ctx, comp_idx, latent_idx)
        mid_ids = self._ctx_row_ids(self.mid_ctx, comp_idx, latent_idx)
        return top_ids, mid_ids

    def excluded_positive_ids(self, comp_idx: int, latent_idx: int) -> set[int]:
        top_ids, mid_ids = self.positive_context_ids(comp_idx, latent_idx)
        return set(top_ids) | set(mid_ids)

    def _ctx_row_ids(self, ctx: Any, comp_idx: int, latent_idx: int) -> list[int]:
        if ctx is None or not getattr(ctx, "_allocated", False):
            return []
        raw = ctx.ctx_seq_idx[comp_idx, latent_idx].detach().cpu().to(torch.int64)
        ids: list[int] = []
        seen: set[int] = set()
        values = getattr(ctx, "ctx_seq_val", None)
        if values is not None:
            vals = values[comp_idx, latent_idx].detach().cpu().float()
        else:
            vals = None
        for index, seq_id in enumerate(raw.tolist()):
            seq_id_int = int(seq_id)
            if seq_id_int <= 0 or seq_id_int in seen:
                continue
            if vals is not None and float(vals[index].item()) <= 0:
                continue
            ids.append(seq_id_int)
            seen.add(seq_id_int)
        return ids

    def load_tokens(self, sequence_ids: list[int], max_length: int = 64) -> tuple[list[int], torch.Tensor]:
        if not sequence_ids:
            self._last_token_load_metadata = {
                "token_cache_enabled": False,
                "token_cache_hit_count": 0,
                "token_cache_miss_count": 0,
                "token_cache_preloaded": False,
            }
            return [], torch.zeros((0, max_length), dtype=torch.long, device=self.bank.device)

        self._ensure_negctx_token_cache(max_length=max_length)
        cached_getter = getattr(self.loader, "get_cached_tokens_by_ids", None)
        if callable(cached_getter):
            cached_ids, cached_tokens, miss_ids = cached_getter(
                sequence_ids,
                max_length=max_length,
                device=self.bank.device,
            )
            if cached_ids and not miss_ids:
                self._last_token_load_metadata = {
                    "token_cache_enabled": True,
                    "token_cache_hit_count": len(cached_ids),
                    "token_cache_miss_count": 0,
                    "token_cache_preloaded": bool(self._token_cache_is_available(max_length=max_length)),
                }
                return cached_ids, cached_tokens
            if cached_ids:
                disk_ids, disk_tokens = self._load_tokens_from_disk(miss_ids, max_length=max_length)
                id_to_token = {
                    int(seq_id): cached_tokens[row].detach()
                    for row, seq_id in enumerate(cached_ids)
                }
                for row, seq_id in enumerate(disk_ids):
                    id_to_token[int(seq_id)] = disk_tokens[row].detach()
                ordered_ids = [int(seq_id) for seq_id in sequence_ids if int(seq_id) in id_to_token]
                if ordered_ids:
                    ordered_tokens = torch.stack([id_to_token[seq_id] for seq_id in ordered_ids], dim=0).to(self.bank.device)
                else:
                    ordered_tokens = torch.zeros((0, max_length), dtype=torch.long, device=self.bank.device)
                self._last_token_load_metadata = {
                    "token_cache_enabled": True,
                    "token_cache_hit_count": len(cached_ids),
                    "token_cache_miss_count": len(miss_ids),
                    "token_cache_preloaded": bool(self._token_cache_is_available(max_length=max_length)),
                }
                return ordered_ids, ordered_tokens

        loaded_ids, tokens = self._load_tokens_from_disk(sequence_ids, max_length=max_length)
        self._last_token_load_metadata = {
            "token_cache_enabled": bool(callable(cached_getter)),
            "token_cache_hit_count": 0,
            "token_cache_miss_count": len(sequence_ids),
            "token_cache_preloaded": bool(self._token_cache_is_available(max_length=max_length)),
        }
        return loaded_ids, tokens

    def _load_tokens_from_disk(self, sequence_ids: list[int], max_length: int = 64) -> tuple[list[int], torch.Tensor]:
        loaded_ids: list[int] = []
        batches: list[torch.Tensor] = []
        batch_iter = self._token_batch_iter(sequence_ids, max_length=max_length)
        for batch_ids, batch_tokens in batch_iter:
            if not isinstance(batch_tokens, torch.Tensor):
                continue
            loaded_ids.extend(int(seq_id) for seq_id in batch_ids.detach().cpu().tolist())
            batches.append(batch_tokens.to(self.bank.device))

        if not batches:
            return [], torch.zeros((0, max_length), dtype=torch.long, device=self.bank.device)
        return loaded_ids, torch.cat(batches, dim=0)

    def _ensure_negctx_token_cache(self, max_length: int = 64) -> None:
        cfg = config.discovery.neg_context_selection
        if not bool(getattr(cfg, "preload_negctx_tokens", False)):
            return
        if self._token_cache_is_available(max_length=max_length):
            return
        preload = getattr(self.loader, "preload_sequence_tokens", None)
        if not callable(preload):
            return
        global_ids = self.global_negctx_ids()
        if not global_ids:
            return
        dtype = self._token_cache_dtype(str(getattr(cfg, "token_cache_dtype", "int32")))
        max_bytes = int(float(getattr(cfg, "token_cache_max_gb", 10.0)) * (1024 ** 3))
        preload(global_ids, max_length=max_length, dtype=dtype, max_bytes=max_bytes)

    def _token_cache_is_available(self, max_length: int = 64) -> bool:
        has_cache = getattr(self.loader, "has_token_cache", None)
        return bool(callable(has_cache) and has_cache(max_length=max_length))

    def _token_cache_dtype(self, dtype_name: str) -> torch.dtype:
        if dtype_name == "int64":
            return torch.int64
        return torch.int32

    def _token_batch_iter(self, sequence_ids: list[int], max_length: int = 64):
        grouped_loader = getattr(self.loader, "get_batches_by_ids_grouped", None)
        if callable(grouped_loader):
            return grouped_loader(sequence_ids, max_length=max_length, restore_order=True)
        return self.loader.get_batches_by_ids(sequence_ids, max_length=max_length)

    def topctx_reference_repr(self, comp_idx: int, latent_idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        reps, metadata = self.topctx_reference_reprs(comp_idx, latent_idx)
        if reps.shape[0] == 0:
            return torch.zeros(int(self.seq_repr.repr_dim), dtype=torch.float32), metadata
        centroid_metadata = dict(metadata)
        centroid_metadata["reference_strategy"] = "centroid"
        return reps.float().mean(dim=0), centroid_metadata

    def topctx_reference_reprs(self, comp_idx: int, latent_idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        cache_key = (int(comp_idx), int(latent_idx))
        cached = self._topctx_reference_reprs_cache.get(cache_key)
        if cached is not None:
            reps, metadata = cached
            cached_metadata = dict(metadata)
            cached_metadata["reference_cache_hit"] = True
            return reps.clone(), cached_metadata

        top_ids, _mid_ids = self.positive_context_ids(comp_idx, latent_idx)
        if not top_ids:
            result = torch.zeros((0, int(self.seq_repr.repr_dim)), dtype=torch.float32)
            metadata = {
                "reference_source": "seq_repr_top_ctx",
                "reference_strategy": "positive_set_max",
                "reference_count": 0,
                "reference_forwarded_count": 0,
                "reference_cache_hit": False,
            }
            self._topctx_reference_reprs_cache[cache_key] = (result, dict(metadata))
            return result.clone(), metadata
        reps = self._get_repr_for_ids(top_ids)
        valid = reps.abs().sum(dim=1) > 0
        missing_ids = [top_ids[i] for i in (~valid).nonzero(as_tuple=True)[0].tolist()]
        if bool(valid.any().item()):
            reps = reps[valid]
            used_ids = [top_ids[i] for i in valid.nonzero(as_tuple=True)[0].tolist()]
        else:
            used_ids = []
            reps = torch.zeros((0, int(self.seq_repr.repr_dim)), dtype=torch.float32)
        if missing_ids:
            forward_reps = self._forward_topctx_reprs(missing_ids)
            if forward_reps.shape[0] > 0:
                reps = torch.cat([reps, forward_reps], dim=0)
                used_ids.extend(missing_ids[: forward_reps.shape[0]])
        if reps.shape[0] == 0:
            result = torch.zeros((0, int(self.seq_repr.repr_dim)), dtype=torch.float32)
            metadata = {
                "reference_source": "seq_repr_top_ctx",
                "reference_strategy": "positive_set_max",
                "reference_count": 0,
                "reference_forwarded_count": 0,
                "reference_cache_hit": False,
            }
            self._topctx_reference_reprs_cache[cache_key] = (result, dict(metadata))
            return result.clone(), metadata
        reference_source = "seq_repr_top_ctx"
        if missing_ids:
            reference_source = "seq_repr_top_ctx+forward_top_ctx"
        result = reps.float()
        metadata = {
            "reference_source": reference_source,
            "reference_strategy": "positive_set_max",
            "reference_count": len(used_ids),
            "reference_forwarded_count": max(0, len(used_ids) - int(valid.sum().item())),
            "reference_cache_hit": False,
        }
        self._topctx_reference_reprs_cache[cache_key] = (result.detach().cpu(), dict(metadata))
        return result, metadata

    def _forward_topctx_reprs(self, sequence_ids: list[int]) -> torch.Tensor:
        loaded_ids, tokens = self.load_tokens(sequence_ids, max_length=64)
        if tokens.shape[0] == 0 or not loaded_ids:
            return torch.zeros((0, int(self.seq_repr.repr_dim)), dtype=torch.float32)
        id_to_row = {seq_id: row for row, seq_id in enumerate(loaded_ids)}
        pooled_by_id: dict[int, torch.Tensor] = {}
        last_layer_idx = max(0, int(self.bank.n_layer) - 1)
        resid_kind_idx = self._resid_kind_idx()
        repr_mode = str(getattr(self.seq_repr, "repr_mode", "mean_pool"))

        self._disable_compile()
        try:
            for batch_start in range(0, tokens.shape[0], max(1, int(getattr(self.loader, "batch_size", 1)))):
                batch_tokens = tokens[batch_start : batch_start + max(1, int(getattr(self.loader, "batch_size", 1)))]
                batch_ids = loaded_ids[batch_start : batch_start + int(batch_tokens.shape[0])]
                captured: list[torch.Tensor] = []

                def capture_hook(layer_idx: int, activations: tuple) -> None:
                    if layer_idx != last_layer_idx:
                        return
                    resid = activations[resid_kind_idx]
                    if repr_mode == "last_token":
                        pooled = resid[:, -1, :].float().detach().cpu()
                    else:
                        pooled = resid.float().mean(dim=1).detach().cpu()
                    captured.append(pooled)

                with torch.no_grad():
                    self.inference.forward(
                        batch_tokens.to(self.bank.device),
                        activations_callback=capture_hook,
                        return_activations=False,
                        tokenize_final=False,
                    )
                if captured:
                    for row, seq_id in enumerate(batch_ids[: captured[0].shape[0]]):
                        pooled_by_id[int(seq_id)] = captured[0][row]
        finally:
            self._enable_compile()

        ordered = [pooled_by_id[seq_id] for seq_id in sequence_ids if seq_id in pooled_by_id and seq_id in id_to_row]
        if not ordered:
            return torch.zeros((0, int(self.seq_repr.repr_dim)), dtype=torch.float32)
        return torch.stack(ordered, dim=0).float()

    def _resid_kind_idx(self) -> int:
        try:
            return list(self.bank.kinds).index("resid")
        except ValueError:
            return len(self.bank.kinds) - 1

    def collect_seed_max_activations(
        self,
        tokens: torch.Tensor,
        comp_idx: int,
        latent_idx: int,
        batch_size: int,
        preact: bool = False,
    ) -> torch.Tensor:
        """
        Run no-grad forwards and return max seed activation per sequence.

        ``preact=False`` (default) reads the seed POST-TOP-K, via
        target_latent_activations — which returns exactly 0 whenever the seed
        misses the top-k. A sequence where the seed very nearly fired is
        therefore indistinguishable from one where it is genuinely silent, and
        both pass a non-activation filter as "clean" negatives. Since "close"
        negatives are the ones most likely to nearly fire, this
        preferentially contaminates exactly the mode that should be strongest.

        ``preact=True`` reads relu(x @ w_seed + b_seed) instead — the value the
        seed WOULD have without top-k censoring (the SAE computes
        relu(linear(...)) then takes top-k, so this is the uncensored
        activation, not a different quantity).
        """
        n_kinds = len(self.bank.kinds)
        seed_layer, seed_kind_idx = split_component_idx(comp_idx, n_kinds)
        seed_kind = self.bank.kinds[seed_kind_idx]
        seed_acts_list: list[torch.Tensor] = []
        if preact:
            sae = self.bank.saes[seed_kind][seed_layer]
            w_seed = sae.encoder.weight[int(latent_idx)].detach()
            b_seed = sae._get_bias_eff()[int(latent_idx)].detach()

        self._disable_compile()
        try:
            for batch_start in range(0, tokens.shape[0], max(1, int(batch_size))):
                batch = tokens[batch_start : batch_start + max(1, int(batch_size))].to(self.bank.device)
                batch_seed_acts: list[torch.Tensor] = []

                def capture_hook(layer_idx: int, activations: tuple) -> None:
                    if layer_idx != seed_layer:
                        return
                    act = activations[seed_kind_idx]
                    if preact:
                        w = w_seed.to(device=act.device, dtype=act.dtype)
                        b = b_seed.to(device=act.device, dtype=act.dtype)
                        seed_vals = torch.relu(act @ w + b)
                    else:
                        top_acts, top_indices = self.bank.encode(act, seed_kind, seed_layer)
                        seed_vals = target_latent_activations(top_acts, top_indices, latent_idx)
                    batch_seed_acts.append(seed_vals.max(dim=-1).values.float().cpu())

                with torch.no_grad():
                    self.inference.forward(
                        batch,
                        activations_callback=capture_hook,
                        return_activations=False,
                        tokenize_final=False,
                    )

                if batch_seed_acts:
                    seed_acts_list.append(batch_seed_acts[0])
        finally:
            self._enable_compile()

        if not seed_acts_list:
            return torch.zeros(0, dtype=torch.float32)
        return torch.cat(seed_acts_list, dim=0)

    def posctx_reference(
        self,
        pos_tokens: torch.Tensor,
        comp_idx: int,
        latent_idx: int,
        batch_size: int,
        stat: str = "median",
    ) -> Optional[float]:
        """The seed's typical PRE-TOP-K activation on its positive contexts —
        the scale preact_filter's threshold is a fraction of.

        Measured pre-top-k on purpose: the bar has to be comparable with the
        candidates' pre-top-k values, and a posctx reference read post-top-k
        would be the censored quantity we are trying to get away from.

        One definition, on the selector, so callers cannot drift apart on how
        the reference is computed.
        """
        if pos_tokens is None or int(pos_tokens.shape[0]) == 0:
            return None
        vals = self.collect_seed_max_activations(
            pos_tokens, comp_idx, latent_idx,
            batch_size=max(1, int(batch_size)), preact=True)
        if int(vals.numel()) == 0:
            return None
        return float(vals.mean() if stat == "mean" else vals.median())

    def _cached_seed_max_activations(
        self,
        tokens: torch.Tensor,
        sequence_ids: list[int],
        comp_idx: int,
        latent_idx: int,
        batch_size: int,
        preact: bool = False,
    ) -> tuple[torch.Tensor, int, int]:
        usable = min(int(tokens.shape[0]), len(sequence_ids))
        if usable <= 0:
            return torch.zeros(0, dtype=torch.float32), 0, 0

        tokens = tokens[:usable]
        sequence_ids = sequence_ids[:usable]
        cached_values: list[float | None] = []
        miss_indices: list[int] = []
        hits = 0
        for index, seq_id in enumerate(sequence_ids):
            # preact is PART OF THE KEY: post-top-k and pre-top-k are different
            # measurements of the same (seed, sequence), so sharing a cache
            # slot would silently serve censored values to a pre-act run.
            cache_key = (int(comp_idx), int(latent_idx), int(seq_id), bool(preact))
            cached = self._seed_activation_cache.get(cache_key)
            if cached is None:
                cached_values.append(None)
                miss_indices.append(index)
            else:
                cached_values.append(float(cached))
                hits += 1

        if miss_indices:
            miss_index_tensor = torch.tensor(miss_indices, dtype=torch.long, device=tokens.device)
            miss_tokens = tokens.index_select(0, miss_index_tensor)
            miss_acts = self.collect_seed_max_activations(
                miss_tokens,
                comp_idx,
                latent_idx,
                batch_size=max(1, int(batch_size)),
                preact=preact,
            )
            for offset, index in enumerate(miss_indices[: int(miss_acts.shape[0])]):
                value = float(miss_acts[offset].item())
                cached_values[index] = value
                self._seed_activation_cache[
                    (int(comp_idx), int(latent_idx), int(sequence_ids[index]), bool(preact))
                ] = value

        resolved = [0.0 if value is None else float(value) for value in cached_values]
        return torch.tensor(resolved, dtype=torch.float32), hits, len(miss_indices)

    def _random_candidate_ids(
        self,
        comp_idx: int,
        latent_idx: int,
        *,
        max_sequences: int,
        candidate_pool_size: Optional[int],
        selection_seed: int,
    ) -> tuple[list[int], dict[str, Any]]:
        excluded = self.excluded_positive_ids(comp_idx, latent_idx)
        global_ids = self.global_negctx_ids()
        ids = [seq_id for seq_id in global_ids if seq_id not in excluded]
        pool_limit = self._candidate_pool_limit(candidate_pool_size, max_sequences=max_sequences, exact=False)
        shuffled = self._deterministic_sample_order(
            ids,
            pool_limit,
            comp_idx,
            latent_idx,
            "random",
            selection_seed,
        )
        return shuffled, {
            "candidate_ids_total": len(global_ids),
            "excluded_positive_count": len(excluded),
            "candidate_ids_ranked": len(shuffled),
            "candidate_pool_size": pool_limit,
            "ranking_source": "random_shuffle",
            "exact": False,
        }

    def _select_from_ordered_ids(
        self,
        comp_idx: int,
        latent_idx: int,
        mode: str,
        candidate_ids: list[int],
        max_sequences: int,
        batch_size: int,
        *,
        non_activation_threshold: float,
        preact_filter: bool = False,
        preact_select: str = "cleanest",
        selection_seed: int = 0,
        filter_batch_size: Optional[int] = None,
        load_window_size: Optional[int] = None,
        ranking_metadata: Optional[dict[str, Any]] = None,
        logger: Any | None,
    ) -> NegContextSelection | None:
        del selection_seed
        ranking_metadata = ranking_metadata or {}
        if not candidate_ids:
            self._reject(logger, f"neg_mode={mode}: no global negctx candidate sequences available")
            return None

        candidate_ids_scanned = 0
        filtered_active = 0
        valid_ids: list[int] = []
        valid_tokens: list[torch.Tensor] = []
        preact_rank = bool(preact_filter) and preact_select == "cleanest"
        ranked_scores: list[float] = []
        ranked_ids: list[int] = []
        ranked_tokens: list[torch.Tensor] = []
        rank_stats: dict[str, Any] = {}
        rank_pool_mult = 4
        activation_cache_hits = 0
        activation_cache_misses = 0
        token_cache_hit_count = 0
        token_cache_miss_count = 0
        token_cache_enabled = False
        token_cache_preloaded = False
        filter_batch_size = max(1, int(filter_batch_size or batch_size))
        configured_load_window_size = max(1, int(load_window_size or batch_size))
        load_window_size = min(
            configured_load_window_size,
            max(max_sequences * 4, max(1, int(batch_size))),
        )

        for start in range(0, len(candidate_ids), load_window_size):
            window_ids = candidate_ids[start : start + load_window_size]
            loaded_ids, tokens = self.load_tokens(window_ids, max_length=64)
            load_metadata = self._last_token_load_metadata
            token_cache_hit_count += int(load_metadata.get("token_cache_hit_count", 0))
            token_cache_miss_count += int(load_metadata.get("token_cache_miss_count", 0))
            token_cache_enabled = token_cache_enabled or bool(load_metadata.get("token_cache_enabled", False))
            token_cache_preloaded = token_cache_preloaded or bool(load_metadata.get("token_cache_preloaded", False))
            if tokens.shape[0] == 0:
                candidate_ids_scanned += len(window_ids)
                continue

            seed_acts, cache_hits, cache_misses = self._cached_seed_max_activations(
                tokens,
                loaded_ids,
                comp_idx,
                latent_idx,
                batch_size=filter_batch_size,
                preact=preact_filter,
            )
            activation_cache_hits += cache_hits
            activation_cache_misses += cache_misses
            usable = min(tokens.shape[0], seed_acts.shape[0], len(loaded_ids))
            if usable == 0:
                candidate_ids_scanned += len(window_ids)
                continue

            seed_acts = seed_acts[:usable]
            tokens = tokens[:usable]
            loaded_ids = loaded_ids[:usable]
            candidate_ids_scanned += len(window_ids)

            if preact_rank:
                # RANK MODE: keep every scanned candidate with its pre-top-k
                # score and choose the CLEANEST at the end.
                #
                # An absolute bar cannot work across depth: contamination runs
                # ~3% of the posctx reference at L2 and ~28% at L10, so one
                # fraction is either inert (0.25 rejected NOTHING at L2/L5/L8,
                # measured) or rejects everything (0.10 would reject 100% of
                # close candidates at L10). Ranking adapts automatically and
                # cannot be infeasible � at the price that it always returns
                # something, so the selected set's own contamination is
                # reported in metadata rather than assumed away.
                ranked_scores.extend(float(v) for v in seed_acts.tolist())
                ranked_ids.extend(int(i) for i in loaded_ids)
                ranked_tokens.append(tokens.detach().cpu())
                if len(ranked_ids) >= max(max_sequences * rank_pool_mult,
                                          max_sequences):
                    break
                continue

            non_activating = seed_acts <= non_activation_threshold
            filtered_active += int((~non_activating).sum().item())

            if bool(non_activating.any().item()):
                valid_ids.extend([loaded_ids[i] for i in non_activating.nonzero(as_tuple=True)[0].tolist()])
                valid_tokens.append(tokens[non_activating].detach().cpu())

            if len(valid_ids) >= max_sequences:
                break

        if preact_rank and ranked_ids:
            order = sorted(range(len(ranked_ids)), key=lambda i: ranked_scores[i])
            keep = order[:max_sequences]
            all_tokens = torch.cat(ranked_tokens, dim=0)
            valid_ids = [ranked_ids[i] for i in keep]
            valid_tokens = [all_tokens[torch.tensor(keep, dtype=torch.long)]]
            kept_scores = [ranked_scores[i] for i in keep]
            filtered_active = len(ranked_ids) - len(keep)
            rank_stats = {
                "preact_rank_pool": len(ranked_ids),
                "preact_kept_max": round(max(kept_scores), 6) if kept_scores else None,
                "preact_kept_median": (round(sorted(kept_scores)[len(kept_scores) // 2], 6)
                                       if kept_scores else None),
                "preact_pool_median": round(sorted(ranked_scores)[len(ranked_scores) // 2], 6),
                "preact_pool_max": round(max(ranked_scores), 6),
            }

        if not valid_ids:
            self._reject(logger, f"neg_mode={mode}: no non-activating negctx candidates found")
            return None

        all_tokens = torch.cat(valid_tokens, dim=0)
        k = min(max_sequences, all_tokens.shape[0])
        selected_tokens = all_tokens[:k].to(self.bank.device)
        selected_ids = valid_ids[:k]
        metadata = {
            "candidate_ids_total": ranking_metadata.get("candidate_ids_total", len(candidate_ids)),
            "candidate_ids_scanned": candidate_ids_scanned,
            "preact_filter": bool(preact_filter),
            "preact_select": (preact_select if preact_filter else None),
            **rank_stats,
            "filtered_seed_active": filtered_active,
            "non_activating_count": len(valid_ids),
            "selected_count": int(selected_tokens.shape[0]),
            "non_activation_threshold": float(non_activation_threshold),
            "filter_batch_size": int(filter_batch_size),
            "load_window_size": int(load_window_size),
            "configured_load_window_size": int(configured_load_window_size),
            "activation_cache_hits": int(activation_cache_hits),
            "activation_cache_misses": int(activation_cache_misses),
            "token_cache_enabled": bool(token_cache_enabled),
            "token_cache_preloaded": bool(token_cache_preloaded),
            "token_cache_hit_count": int(token_cache_hit_count),
            "token_cache_miss_count": int(token_cache_miss_count),
            **ranking_metadata,
        }
        self._note(
            logger,
            f"neg_mode={mode}: selected {selected_tokens.shape[0]} sequences | "
            f"scanned={candidate_ids_scanned} non_activating={len(valid_ids)}",
        )
        return NegContextSelection(tokens=selected_tokens, sequence_ids=selected_ids, mode=mode, metadata=metadata)

    def _ranked_candidate_ids(
        self,
        comp_idx: int,
        latent_idx: int,
        mode: str,
        *,
        candidate_pool_size: Optional[int],
        max_sequences: int,
        exact: bool,
        selection_seed: int,
    ) -> tuple[list[int], dict[str, Any]]:
        del selection_seed
        global_ids = self.global_negctx_ids()
        excluded = self.excluded_positive_ids(comp_idx, latent_idx)
        ids = [seq_id for seq_id in global_ids if seq_id not in excluded]
        if not ids:
            return [], {
                "candidate_ids_total": len(global_ids),
                "excluded_positive_count": len(excluded),
                "ranking_source": "seq_repr",
            }

        reference_reps, ref_metadata = self.topctx_reference_reprs(comp_idx, latent_idx)
        pool_limit = self._candidate_pool_limit(candidate_pool_size, max_sequences=max_sequences, exact=exact)
        ranked_ids, ranked_sims, scored_count = self._chunked_topk_ranked_ids(
            ids,
            reference_reps,
            mode,
            pool_limit=pool_limit,
        )
        if not ranked_ids:
            return [], {
                "candidate_ids_total": len(global_ids),
                "excluded_positive_count": len(excluded),
                "candidate_ids_scored": scored_count,
                "ranking_source": "seq_repr",
                **ref_metadata,
            }
        score_values = ranked_sims if mode == "close" else 1.0 - ranked_sims
        return ranked_ids, {
            "candidate_ids_total": len(global_ids),
            "excluded_positive_count": len(excluded),
            "candidate_ids_ranked": len(ranked_ids),
            "candidate_ids_scored": scored_count,
            "candidate_pool_size": pool_limit,
            "ranking_source": "seq_repr",
            "ranking_method": self._positive_set_ranking_method_name(),
            "ranking_device": str(self._ranking_device()),
            "score_name": "max_cosine_sim" if mode == "close" else "one_minus_max_cosine_sim",
            "score_min": float(score_values.min().item()),
            "score_max": float(score_values.max().item()),
            **ref_metadata,
        }

    def _candidate_pool_limit(
        self,
        candidate_pool_size: Optional[int],
        *,
        max_sequences: int,
        exact: bool,
    ) -> Optional[int]:
        if exact:
            return None
        if candidate_pool_size is not None:
            return max(int(candidate_pool_size), int(max_sequences))
        return max(int(max_sequences) * 128, 2048)

    def _chunked_topk_ranked_ids(
        self,
        ids: list[int],
        reference_reps: torch.Tensor,
        mode: str,
        *,
        pool_limit: Optional[int],
        chunk_size: int = 8192,
    ) -> tuple[list[int], torch.Tensor, int]:
        if not ids:
            return [], torch.zeros(0, dtype=torch.float32), 0

        target_k = len(ids) if pool_limit is None else min(int(pool_limit), len(ids))
        if target_k <= 0:
            return [], torch.zeros(0, dtype=torch.float32), 0

        score_chunks: list[torch.Tensor] = []
        sim_chunks: list[torch.Tensor] = []
        id_chunks: list[torch.Tensor] = []
        scored_count = 0
        largest = mode == "close"
        ranking_device = self._ranking_device()
        reference_norm = self._normalized_reference_reps(reference_reps, ranking_device)

        for start in range(0, len(ids), max(1, int(chunk_size))):
            chunk_ids = ids[start : start + max(1, int(chunk_size))]
            reps = self._get_repr_for_ids(chunk_ids)
            valid = reps.abs().sum(dim=1) > 0
            if not bool(valid.any().item()):
                continue

            valid_positions = valid.nonzero(as_tuple=True)[0]
            valid_ids = torch.tensor(chunk_ids, dtype=torch.long)[valid_positions]
            reps = reps[valid]
            sims = self._positive_set_max_similarity(reps, reference_norm, ranking_device)
            scored_count += int(sims.numel())
            scores = sims if largest else -sims

            local_k = min(target_k, int(scores.numel()))
            top_scores, top_idx = torch.topk(scores, k=local_k, largest=True, sorted=True)
            score_chunks.append(top_scores.cpu())
            sim_chunks.append(sims[top_idx].cpu())
            id_chunks.append(valid_ids[top_idx].cpu())

        if not score_chunks:
            return [], torch.zeros(0, dtype=torch.float32), scored_count

        scores_all = torch.cat(score_chunks, dim=0)
        sims_all = torch.cat(sim_chunks, dim=0)
        ids_all = torch.cat(id_chunks, dim=0)
        final_k = min(target_k, int(scores_all.numel()))
        _final_scores, final_idx = torch.topk(scores_all, k=final_k, largest=True, sorted=True)
        ranked_ids = [int(seq_id) for seq_id in ids_all[final_idx].tolist()]
        ranked_sims = sims_all[final_idx].float()
        return ranked_ids, ranked_sims, scored_count

    def _get_repr_for_ids(self, ids: list[int]) -> torch.Tensor:
        if not ids:
            return torch.zeros((0, int(self.seq_repr.repr_dim)), dtype=torch.float32)
        seq_ids = torch.tensor(ids, dtype=torch.long)
        return self.seq_repr.get_repr(seq_ids).float()

    def _ranking_device(self) -> torch.device:
        device = getattr(self.bank, "device", torch.device("cpu"))
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    def _positive_set_ranking_method_name(self) -> str:
        device = self._ranking_device()
        if device.type == "cuda":
            return "chunked_gpu_positive_set_topk"
        return "chunked_positive_set_topk"

    def _normalized_reference_reps(self, reference_reps: torch.Tensor, device: torch.device) -> torch.Tensor:
        if reference_reps.shape[0] == 0:
            reference_reps = torch.zeros((1, int(self.seq_repr.repr_dim)), dtype=torch.float32)
        return self._normalize_rows(reference_reps.to(device=device, dtype=torch.float32))

    def _positive_set_max_similarity(
        self,
        reps: torch.Tensor,
        normalized_reference_reps: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        normalized_reps = self._normalize_rows(reps.to(device=device, dtype=torch.float32))
        similarities = normalized_reps @ normalized_reference_reps.T
        return similarities.max(dim=1).values.detach().cpu().float()

    def _normalize_rows(self, vectors: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return vectors.float() / (vectors.float().norm(dim=-1, keepdim=True) + eps)

    def _cosine_similarity(self, vectors: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        ref = reference.to(vectors.device, dtype=torch.float32)
        ref_norm = ref / (ref.norm() + eps)
        vec_norm = vectors.float() / (vectors.float().norm(dim=-1, keepdim=True) + eps)
        return vec_norm @ ref_norm

    def _deterministic_shuffle(
        self,
        ids: list[int],
        comp_idx: int,
        latent_idx: int,
        mode: str,
        selection_seed: int,
    ) -> list[int]:
        if len(ids) <= 1:
            return list(ids)
        mode_offset = sum((i + 1) * ord(char) for i, char in enumerate(mode))
        seed = int(selection_seed) + int(comp_idx) * 1_000_003 + int(latent_idx) * 97 + mode_offset
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(len(ids), generator=generator).tolist()
        return [ids[int(i)] for i in order]

    def _deterministic_sample_order(
        self,
        ids: list[int],
        limit: Optional[int],
        comp_idx: int,
        latent_idx: int,
        mode: str,
        selection_seed: int,
    ) -> list[int]:
        if limit is None or len(ids) <= int(limit):
            return self._deterministic_shuffle(ids, comp_idx, latent_idx, mode, selection_seed)
        mode_offset = sum((i + 1) * ord(char) for i, char in enumerate(mode))
        seed = int(selection_seed) + int(comp_idx) * 1_000_003 + int(latent_idx) * 97 + mode_offset
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(len(ids), generator=generator)[: int(limit)].tolist()
        return [ids[int(i)] for i in order]

    def _disable_compile(self) -> None:
        if hasattr(self.inference, "disable_compile"):
            self.inference.disable_compile()

    def _enable_compile(self) -> None:
        if hasattr(self.inference, "enable_compile"):
            self.inference.enable_compile()

    def _note(self, logger: Any | None, message: str) -> None:
        if logger is not None and hasattr(logger, "note"):
            logger.note(message)

    def _reject(self, logger: Any | None, message: str) -> None:
        if logger is not None and hasattr(logger, "reject"):
            logger.reject(message)


__all__ = ["NegContextSelection", "NegContextSelector"]
