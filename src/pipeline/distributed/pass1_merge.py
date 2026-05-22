"""Merge helpers for distributed pass-1 partial artifacts."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import tracemalloc
from typing import Callable, Dict, Sequence

import torch

from .interfaces import build_output_paths
from .layout import build_run_layout
from .manifest import DistributedRunManifest, ManifestStatus, save_manifest
from .pass1_partials import (
    Pass1PartialMetadata,
    load_pass1_partial,
    validate_pass1_partial,
)
from .seq_repr_mapping import validate_seq_repr_mapping


LatentStatsPartial = tuple[Pass1PartialMetadata, Dict[str, object]]
TopCtxPartial = tuple[Pass1PartialMetadata, Dict[str, object]]
MidCtxCandidatesPartial = tuple[Pass1PartialMetadata, Dict[str, object]]
SeqReprPartial = tuple[Pass1PartialMetadata, Dict[str, object]]
LogitCtxPartial = tuple[Pass1PartialMetadata, Dict[str, object]]

MID_CTX_CANDIDATE_POOL_DEFAULTS = {
    "enabled": True,
    "band_margin_sigma": 1.0,
    "on_truncation": "replay_fallback",
}

PASS1_PARTIAL_FILENAMES = {
    "latent_stats": "latent_stats.partial.pt",
    "top_ctx": "top_ctx.partial.pt",
    "mid_ctx_candidates": "mid_ctx_candidates.partial.pt",
    "seq_repr": "seq_repr.partial.pt",
    "logit_ctx": "logit_ctx.partial.pt",
}


def load_and_merge_latent_stats_partials(
    partial_paths: Sequence[str | Path],
    *,
    expected_config_hash: str | None = None,
) -> Dict[str, object]:
    """Load latent-stats partial files and merge them into one canonical payload."""

    partials = [
        load_pass1_partial(
            path,
            expected_artifact_name="latent_stats",
            expected_config_hash=expected_config_hash,
        )
        for path in partial_paths
    ]
    return merge_latent_stats_partials(partials)


def merge_latent_stats_partials(
    partials: Sequence[LatentStatsPartial],
) -> Dict[str, object]:
    """Merge latent-stats partial payloads with parallel Welford semantics."""

    if not partials:
        raise ValueError("at least one latent_stats partial is required")
    _validate_latent_stats_partial_set(partials)

    first_payload = partials[0][1]
    active_count = torch.zeros_like(first_payload["active_count"])
    mean = torch.zeros_like(first_payload["mean"])
    mean_abs = torch.zeros_like(first_payload["mean_abs"])
    m2 = torch.zeros_like(first_payload["m2"])
    m2_abs = torch.zeros_like(first_payload["m2_abs"])
    seq_count = torch.zeros_like(first_payload["seq_count"])
    mean_seq = torch.zeros_like(first_payload["mean_seq"])
    m2_seq = torch.zeros_like(first_payload["m2_seq"])
    component_steps: dict[int, int] = defaultdict(int)

    for _metadata, payload in partials:
        previous_active_count = active_count
        active_count, mean, m2 = _merge_welford_state(
            previous_active_count,
            mean,
            m2,
            payload["active_count"],
            payload["mean"],
            payload["m2"],
        )
        _merged_abs_count, mean_abs, m2_abs = _merge_welford_state(
            previous_active_count,
            mean_abs,
            m2_abs,
            payload["active_count"],
            payload["mean_abs"],
            payload["m2_abs"],
        )
        seq_count, mean_seq, m2_seq = _merge_welford_state(
            seq_count,
            mean_seq,
            m2_seq,
            payload["seq_count"],
            payload["mean_seq"],
            payload["m2_seq"],
        )
        for comp_idx, count in payload["component_steps"].items():
            component_steps[int(comp_idx)] += int(count)

    merged = {
        "active_count": active_count,
        "mean": mean,
        "mean_abs": mean_abs,
        "m2": m2,
        "m2_abs": m2_abs,
        "seq_count": seq_count,
        "mean_seq": mean_seq,
        "m2_seq": m2_seq,
        "component_steps": dict(component_steps),
    }
    _validate_merged_latent_stats(merged, partials)
    return merged


def load_and_merge_top_ctx_partials(
    partial_paths: Sequence[str | Path],
    *,
    expected_config_hash: str | None = None,
) -> Dict[str, object]:
    """Load top-context partial files and merge them into one canonical payload."""

    partials = [
        load_pass1_partial(
            path,
            expected_artifact_name="top_ctx",
            expected_config_hash=expected_config_hash,
        )
        for path in partial_paths
    ]
    return merge_top_ctx_partials(partials)


def merge_top_ctx_partials(
    partials: Sequence[TopCtxPartial],
) -> Dict[str, object]:
    """Merge top-context partial payloads by global top-K per component/latent."""

    if not partials:
        raise ValueError("at least one top_ctx partial is required")
    _validate_top_ctx_partial_set(partials)

    first_payload = partials[0][1]
    candidate_indices = torch.cat(
        [payload["ctx_seq_idx"].to(torch.int64) for _metadata, payload in partials],
        dim=2,
    )
    candidate_values = torch.cat(
        [payload["ctx_seq_val"].to(torch.float32) for _metadata, payload in partials],
        dim=2,
    )
    candidate_values = torch.where(
        (candidate_indices > 0) & (candidate_values > 0),
        candidate_values,
        torch.zeros_like(candidate_values),
    )
    candidate_indices = torch.where(
        candidate_values > 0,
        candidate_indices,
        torch.zeros_like(candidate_indices),
    )

    output_k = int(first_payload["ctx_seq_idx"].shape[2])
    order_by_sequence = torch.argsort(candidate_indices, dim=2, stable=True)
    values_by_sequence = candidate_values.gather(2, order_by_sequence)
    indices_by_sequence = candidate_indices.gather(2, order_by_sequence)
    order_by_value = torch.argsort(values_by_sequence, dim=2, descending=True, stable=True)
    top_positions = order_by_value[:, :, :output_k]
    merged_values = values_by_sequence.gather(2, top_positions)
    merged_indices = indices_by_sequence.gather(2, top_positions)

    valid = merged_values > 0
    merged_values = merged_values.masked_fill(~valid, 0.0)
    merged_indices = merged_indices.masked_fill(~valid, 0)
    merged = {
        "ctx_seq_idx": merged_indices.to(first_payload["ctx_seq_idx"].dtype),
        "ctx_seq_val": merged_values.to(first_payload["ctx_seq_val"].dtype),
        "ctx_type": "top",
    }
    _validate_merged_top_ctx(merged, partials)
    return merged


def load_and_merge_mid_ctx_candidate_partials(
    partial_paths: Sequence[str | Path],
    *,
    latent_stats_payload: Dict[str, object],
    expected_config_hash: str | None = None,
    num_ctx_sequences: int | None = None,
    band_low_sigma: float = 0.5,
    band_high_sigma: float = 1.5,
    on_truncation: str = "replay_fallback",
    replay_fallback_fn: Callable[..., Dict[str, object]] | None = None,
) -> Dict[str, object]:
    """Load mid-context candidate partial files and merge into canonical mid_ctx."""

    partials = [
        load_pass1_partial(
            path,
            expected_artifact_name="mid_ctx_candidates",
            expected_config_hash=expected_config_hash,
        )
        for path in partial_paths
    ]
    return merge_mid_ctx_candidate_partials(
        partials,
        latent_stats_payload=latent_stats_payload,
        num_ctx_sequences=num_ctx_sequences,
        band_low_sigma=band_low_sigma,
        band_high_sigma=band_high_sigma,
        on_truncation=on_truncation,
        replay_fallback_fn=replay_fallback_fn,
    )


def merge_mid_ctx_candidate_partials(
    partials: Sequence[MidCtxCandidatesPartial],
    *,
    latent_stats_payload: Dict[str, object],
    num_ctx_sequences: int | None = None,
    band_low_sigma: float = 0.5,
    band_high_sigma: float = 1.5,
    on_truncation: str = "replay_fallback",
    replay_fallback_fn: Callable[..., Dict[str, object]] | None = None,
) -> Dict[str, object]:
    """Filter candidate pools with global stats and select deterministic priorities."""

    if not partials:
        raise ValueError("at least one mid_ctx candidate partial is required")
    if on_truncation not in {"fail", "replay_fallback", "allow_bounded_approx"}:
        raise ValueError("unsupported mid_ctx candidate truncation policy")
    _validate_mid_ctx_candidate_partial_set(partials)
    _validate_latent_stats_for_mid_ctx(latent_stats_payload, partials[0][0])

    metadata = partials[0][0]
    first_payload = partials[0][1]
    output_k = int(
        num_ctx_sequences
        or first_payload.get("candidate_pool_settings", {}).get("final_num_ctx_sequences", 0)
        or first_payload.get("candidate_pool_settings", {}).get("num_ctx_sequences", 0)
        or first_payload["ctx_seq_idx"].shape[2]
    )
    if output_k < 1:
        raise ValueError("num_ctx_sequences must be >= 1")

    component_count = metadata.component_count
    d_sae = metadata.d_sae
    ctx_seq_idx = torch.zeros((component_count, d_sae, output_k), dtype=torch.int32)
    ctx_seq_val = torch.zeros((component_count, d_sae, output_k), dtype=torch.float32)
    reservoir_fill = torch.zeros((component_count, d_sae), dtype=torch.int32)
    reservoir_n = torch.zeros((component_count, d_sae), dtype=torch.int64)
    candidate_count = torch.zeros((component_count, d_sae), dtype=torch.int64)
    truncation_counters = torch.zeros((component_count, d_sae), dtype=torch.int64)

    candidates = _concatenate_mid_ctx_candidates(partials)
    if candidates["component_ids"].numel() > 0:
        component_ids = candidates["component_ids"].to(torch.long)
        latent_ids = candidates["latent_ids"].to(torch.long)
        sequence_ids = candidates["sequence_ids"].to(torch.int32)
        activation_values = candidates["activation_values"].to(torch.float32)
        priorities = candidates["priorities"].to(torch.float32)
        flat_ids = component_ids * d_sae + latent_ids
        candidate_count.view(-1).scatter_add_(
            0,
            flat_ids,
            torch.ones_like(flat_ids, dtype=torch.int64),
        )

        mean_seq = latent_stats_payload["mean_seq"].to(torch.float32)
        std_seq = _std_seq_from_latent_stats(latent_stats_payload)
        low = mean_seq[component_ids, latent_ids] + band_low_sigma * std_seq[component_ids, latent_ids]
        high = mean_seq[component_ids, latent_ids] + band_high_sigma * std_seq[component_ids, latent_ids]
        valid = (activation_values > low) & (activation_values < high)
        if bool(valid.any()):
            valid_components = component_ids[valid]
            valid_latents = latent_ids[valid]
            valid_sequences = sequence_ids[valid]
            valid_values = activation_values[valid]
            valid_priorities = priorities[valid]
            valid_flat_ids = flat_ids[valid]
            reservoir_n.view(-1).scatter_add_(
                0,
                valid_flat_ids,
                torch.ones_like(valid_flat_ids, dtype=torch.int64),
            )
            _select_mid_ctx_candidates(
                ctx_seq_idx,
                ctx_seq_val,
                reservoir_fill,
                valid_components,
                valid_latents,
                valid_sequences,
                valid_values,
                valid_priorities,
                output_k,
            )

    for _metadata, payload in partials:
        truncation_counters += payload["truncation_counters"].to(torch.int64)
    requires_replay_fallback = bool((truncation_counters > 0).any())
    if requires_replay_fallback and on_truncation == "fail":
        raise ValueError("mid_ctx candidate pool truncation detected")
    if requires_replay_fallback and on_truncation == "replay_fallback" and replay_fallback_fn:
        replayed = replay_fallback_fn(
            partials=partials,
            latent_stats_payload=latent_stats_payload,
            num_ctx_sequences=output_k,
            band_low_sigma=band_low_sigma,
            band_high_sigma=band_high_sigma,
        )
        replayed.setdefault("merge_report", {})
        replayed["merge_report"].update(
            {
                "mode": "stats_aware_replay_fallback",
                "replay_fallback_executed": True,
                "truncation_counters": truncation_counters,
                "num_ctx_sequences": int(output_k),
            }
        )
        _validate_merged_mid_ctx(replayed, partials)
        return replayed

    merge_report = {
        "candidate_count": candidate_count,
        "valid_count": reservoir_n,
        "selected_count": reservoir_fill.to(torch.int64),
        "truncation_counters": truncation_counters,
        "requires_replay_fallback": requires_replay_fallback
        and on_truncation == "replay_fallback",
        "replay_fallback_executed": False,
        "bounded_approximation": requires_replay_fallback
        and on_truncation == "allow_bounded_approx",
        "candidate_pool_cleanup_eligible": not requires_replay_fallback,
        "fill_rate": reservoir_fill.float() / float(output_k),
        "band_low_sigma": float(band_low_sigma),
        "band_high_sigma": float(band_high_sigma),
        "num_ctx_sequences": int(output_k),
        "priority_mode": "deterministic_priority_reservoir",
        "candidate_pool_defaults": {
            **MID_CTX_CANDIDATE_POOL_DEFAULTS,
            "max_candidates_per_latent": max(256, 4 * output_k),
        },
        "candidate_pool_settings": dict(first_payload.get("candidate_pool_settings", {})),
        "priority_hash_version": first_payload.get("candidate_pool_settings", {}).get(
            "priority_hash_version"
        ),
    }
    merged = {
        "ctx_seq_idx": ctx_seq_idx,
        "ctx_seq_val": ctx_seq_val,
        "ctx_type": "mid",
        "mode": "distributed_priority_reservoir",
        "band_low_sigma": float(band_low_sigma),
        "band_high_sigma": float(band_high_sigma),
        "num_ctx_sequences": int(output_k),
        "reservoir_fill": reservoir_fill,
        "reservoir_n": reservoir_n,
        "merge_report": merge_report,
    }
    _validate_merged_mid_ctx(merged, partials)
    return merged


def load_and_merge_seq_repr_partials(
    partial_paths: Sequence[str | Path],
    *,
    expected_config_hash: str | None = None,
    seq_repr_mapping: dict[str, object] | None = None,
) -> Dict[str, object]:
    """Load seq-repr partial files and merge them by global sequence ID."""

    partials = [
        load_pass1_partial(
            path,
            expected_artifact_name="seq_repr",
            expected_config_hash=expected_config_hash,
        )
        for path in partial_paths
    ]
    return merge_seq_repr_partials(partials, seq_repr_mapping=seq_repr_mapping)


def merge_seq_repr_partials(
    partials: Sequence[SeqReprPartial],
    *,
    seq_repr_mapping: dict[str, object] | None = None,
) -> Dict[str, object]:
    """Merge capped or uncapped seq_repr partials into one global store payload."""

    if not partials:
        raise ValueError("at least one seq_repr partial is required")
    _validate_seq_repr_partial_set(partials)
    target_mapping = seq_repr_mapping or _mapping_from_seq_repr_payload(partials[0][1])
    validate_seq_repr_mapping(target_mapping)

    n_seqs = int(target_mapping["n_seqs"])
    n_stored = int(target_mapping["n_stored"])
    slot_to_id = target_mapping["slot_to_id"]
    id_to_slot = target_mapping["id_to_slot"]
    is_capped = bool(target_mapping["is_capped"])
    repr_dim = int(partials[0][1]["repr_dim"])
    repr_mode = str(partials[0][1]["repr_mode"])
    repr_buf = torch.zeros((n_stored + 1, repr_dim), dtype=partials[0][1]["repr_buf"].dtype)
    written_slots = torch.zeros(n_stored + 1, dtype=torch.bool)

    for metadata, payload in partials:
        source_buf = payload["repr_buf"]
        source_id_to_slot = payload.get("id_to_slot")
        if source_id_to_slot is not None:
            source_id_to_slot = source_id_to_slot.to(torch.int64)
        for sequence_id in range(metadata.sequence_id_min or 1, (metadata.sequence_id_max or 0) + 1):
            if sequence_id < 1 or sequence_id > n_seqs:
                raise ValueError("seq_repr sequence ID out of global range")
            target_slot = int(id_to_slot[sequence_id].item())
            if target_slot == 0:
                continue
            source_slot = (
                int(source_id_to_slot[sequence_id].item())
                if source_id_to_slot is not None
                else sequence_id
            )
            if source_slot == 0:
                continue
            if source_slot >= source_buf.shape[0]:
                raise ValueError("seq_repr source slot out of range")
            if written_slots[target_slot]:
                raise ValueError(f"seq_repr slot written more than once: {target_slot}")
            row = source_buf[source_slot]
            if not torch.isfinite(row.float()).all():
                raise ValueError("seq_repr row contains non-finite values")
            repr_buf[target_slot] = row.to(repr_buf.dtype)
            written_slots[target_slot] = True

    merged = {
        "repr_buf": repr_buf,
        "repr_mode": repr_mode,
        "repr_dim": repr_dim,
        "n_seqs": n_seqs,
        "n_stored": n_stored,
        "is_capped": is_capped,
        "merge_report": {
            "selected_slots": int(n_stored),
            "written_slots": int(written_slots[1:].sum().item()),
            "missing_slots": int((~written_slots[1:]).sum().item()),
            "sampling_seed": target_mapping.get("sampling_seed"),
            "derived_seed": target_mapping.get("derived_seed"),
            "dataset_fingerprint": target_mapping.get("dataset_fingerprint"),
        },
    }
    if is_capped:
        merged["slot_to_id"] = slot_to_id.to(torch.int64)
        merged["id_to_slot"] = id_to_slot.to(torch.int32)
    _validate_merged_seq_repr(merged)
    return merged


def load_and_merge_logit_ctx_partials(
    partial_paths: Sequence[str | Path],
    *,
    expected_config_hash: str | None = None,
    vocab_size: int | None = None,
) -> Dict[str, object]:
    """Load logit-context partial files and merge event top-K rows."""

    partials = [
        load_pass1_partial(
            path,
            expected_artifact_name="logit_ctx",
            expected_config_hash=expected_config_hash,
        )
        for path in partial_paths
    ]
    return merge_logit_ctx_partials(partials, vocab_size=vocab_size)


def merge_logit_ctx_partials(
    partials: Sequence[LogitCtxPartial],
    *,
    vocab_size: int | None = None,
) -> Dict[str, object]:
    """Merge logit-context partials with exact event top-K semantics."""

    if not partials:
        raise ValueError("at least one logit_ctx partial is required")
    _validate_logit_ctx_partial_set(partials, vocab_size=vocab_size)

    first_payload = partials[0][1]
    latent_counts = sum(
        (payload["latent_counts"].to(torch.int64) for _metadata, payload in partials),
        start=torch.zeros_like(first_payload["latent_counts"], dtype=torch.int64),
    )
    output_k = int(first_payload["top_tokens"].shape[2])
    top_tokens = torch.zeros_like(first_payload["top_tokens"], dtype=torch.int32)
    top_probs = torch.zeros_like(first_payload["top_probs"], dtype=torch.float32)

    candidate_tokens = torch.cat(
        [payload["top_tokens"].to(torch.int64) for _metadata, payload in partials],
        dim=2,
    )
    candidate_probs = torch.cat(
        [payload["top_probs"].to(torch.float32) for _metadata, payload in partials],
        dim=2,
    )
    candidate_workers = torch.cat(
        [
            torch.full_like(payload["top_tokens"].to(torch.int64), metadata.worker_id)
            for metadata, payload in partials
        ],
        dim=2,
    )
    candidate_rows = torch.cat(
        [
            torch.arange(payload["top_tokens"].shape[2], dtype=torch.int64)
            .view(1, 1, -1)
            .expand_as(payload["top_tokens"])
            for _metadata, payload in partials
        ],
        dim=2,
    )

    valid = candidate_probs > 0
    if vocab_size is not None:
        valid &= candidate_tokens < int(vocab_size)
    candidate_probs = candidate_probs.masked_fill(~valid, 0.0)
    candidate_tokens = candidate_tokens.masked_fill(~valid, 0)

    _select_logit_ctx_events(
        top_tokens,
        top_probs,
        candidate_tokens,
        candidate_probs,
        candidate_workers,
        candidate_rows,
        output_k,
    )
    merged = {
        "latent_counts": latent_counts,
        "top_tokens": top_tokens,
        "top_probs": top_probs,
        "merge_report": {
            "num_partials": len(partials),
            "event_top_k": output_k,
            "tie_breaking": "probability_desc_token_asc_worker_asc_candidate_row_asc",
        },
    }
    _validate_merged_logit_ctx(merged, partials, vocab_size=vocab_size)
    return merged


def merge_seq_latent_index_shards(
    worker_index_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    expected_shard_ids: Sequence[int],
    enabled: bool = True,
    shard_id_ranges: Dict[int, tuple[int, int]] | None = None,
) -> Dict[str, object]:
    """Copy worker seq_latent_index shard files into one canonical directory."""

    if not enabled:
        return {
            "enabled": False,
            "copied_shards": [],
            "duplicate_identical_shards": [],
            "output_dir": str(output_dir),
        }
    expected = sorted({int(shard_id) for shard_id in expected_shard_ids})
    if len(expected) != len(list(expected_shard_ids)):
        raise ValueError("expected seq_latent_index shard IDs must be unique")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    copied: list[int] = []
    duplicate_identical: list[int] = []
    seen_sources: dict[int, Path] = {}

    for worker_dir in worker_index_dirs:
        worker_path = Path(worker_dir)
        if not worker_path.exists():
            continue
        if not worker_path.is_dir():
            raise ValueError(f"seq_latent_index worker path is not a directory: {worker_path}")
        for source_path in sorted(worker_path.glob("shard_*.pt")):
            shard_id = _parse_seq_latent_index_shard_id(source_path)
            if shard_id not in expected:
                raise ValueError(f"unexpected seq_latent_index shard output: {shard_id}")
            _validate_seq_latent_index_shard_file(
                source_path,
                shard_id=shard_id,
                shard_id_ranges=shard_id_ranges,
            )
            destination = output_path / source_path.name
            if shard_id in seen_sources or destination.exists():
                if not destination.exists():
                    _copy_file_atomic(seen_sources[shard_id], destination)
                if not _seq_latent_index_files_equivalent(destination, source_path):
                    raise ValueError(
                        f"duplicate seq_latent_index shard differs: shard_{shard_id}.pt"
                    )
                duplicate_identical.append(shard_id)
                continue
            _copy_file_atomic(source_path, destination)
            seen_sources[shard_id] = source_path
            copied.append(shard_id)

    missing = [shard_id for shard_id in expected if not (output_path / f"shard_{shard_id}.pt").exists()]
    if missing:
        raise ValueError(f"missing seq_latent_index shard outputs: {missing}")

    return {
        "enabled": True,
        "copied_shards": copied,
        "duplicate_identical_shards": duplicate_identical,
        "expected_shards": expected,
        "output_dir": str(output_path),
    }


def merge_pass1_worker_outputs(
    manifest: DistributedRunManifest,
    *,
    seq_latent_index_enabled: bool = True,
    vocab_size: int | None = None,
    mid_ctx_num_ctx_sequences: int | None = None,
    mid_ctx_band_low_sigma: float = 0.5,
    mid_ctx_band_high_sigma: float = 1.5,
    mid_ctx_on_truncation: str = "replay_fallback",
) -> Dict[str, object]:
    """Merge all worker pass-1 outputs and write canonical global artifacts."""

    start_time = time.perf_counter()
    tracemalloc.start()
    layout = build_run_layout(manifest)
    output_paths = build_output_paths(layout.run_root)
    output_paths.run_root.mkdir(parents=True, exist_ok=True)
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    partial_paths = _pass1_partial_paths(manifest)

    latent_stats_payload = load_and_merge_latent_stats_partials(
        partial_paths["latent_stats"],
        expected_config_hash=manifest.normalized_config_hash,
    )
    top_ctx_payload = load_and_merge_top_ctx_partials(
        partial_paths["top_ctx"],
        expected_config_hash=manifest.normalized_config_hash,
    )
    mid_ctx_payload = load_and_merge_mid_ctx_candidate_partials(
        partial_paths["mid_ctx_candidates"],
        latent_stats_payload=latent_stats_payload,
        expected_config_hash=manifest.normalized_config_hash,
        num_ctx_sequences=mid_ctx_num_ctx_sequences,
        band_low_sigma=mid_ctx_band_low_sigma,
        band_high_sigma=mid_ctx_band_high_sigma,
        on_truncation=mid_ctx_on_truncation,
    )
    seq_repr_payload = load_and_merge_seq_repr_partials(
        partial_paths["seq_repr"],
        expected_config_hash=manifest.normalized_config_hash,
    )
    logit_ctx_payload = load_and_merge_logit_ctx_partials(
        partial_paths["logit_ctx"],
        expected_config_hash=manifest.normalized_config_hash,
        vocab_size=vocab_size,
    )

    artifacts = {
        "latent_stats": (output_paths.latent_stats, _with_canonical_metadata(latent_stats_payload, manifest, "latent_stats")),
        "top_ctx": (output_paths.top_ctx, _with_canonical_metadata(top_ctx_payload, manifest, "top_ctx")),
        "mid_ctx": (output_paths.mid_ctx, _with_canonical_metadata(mid_ctx_payload, manifest, "mid_ctx")),
        "seq_repr": (output_paths.seq_repr, _with_canonical_metadata(seq_repr_payload, manifest, "seq_repr")),
        "logit_ctx": (output_paths.logit_ctx, _with_canonical_metadata(logit_ctx_payload, manifest, "logit_ctx")),
    }
    for _name, (path, payload) in artifacts.items():
        _atomic_torch_save(payload, path)

    seq_latent_index_report = merge_seq_latent_index_shards(
        [worker.pass1_dir / "seq_latent_index" for worker in layout.workers.values()],
        output_paths.seq_latent_index_dir,
        expected_shard_ids=[record.shard_index for record in manifest.shard_table],
        enabled=seq_latent_index_enabled,
        shard_id_ranges={
            record.shard_index: (record.global_start_id, record.global_end_id)
            for record in manifest.shard_table
        },
    )

    _validate_written_artifacts(artifacts)
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = build_pass1_sanity_report(
        manifest,
        {
            "latent_stats": latent_stats_payload,
            "top_ctx": top_ctx_payload,
            "mid_ctx": mid_ctx_payload,
            "seq_repr": seq_repr_payload,
            "logit_ctx": logit_ctx_payload,
        },
        artifact_paths={name: str(path) for name, (path, _payload) in artifacts.items()},
        seq_latent_index_report=seq_latent_index_report,
        elapsed_s=time.perf_counter() - start_time,
        peak_cpu_memory_bytes=max(int(current_memory), int(peak_memory)),
    )
    report_path = layout.reports_dir / "pass1_sanity_report.json"
    _atomic_write_json(report_path, report)

    completed_manifest = manifest.model_copy(update={"status": ManifestStatus.COMPLETED})
    save_manifest(completed_manifest, manifest.manifest_path)

    return {
        "artifacts": {name: str(path) for name, (path, _payload) in artifacts.items()},
        "seq_latent_index": seq_latent_index_report,
        "sanity_report": str(report_path),
        "manifest_path": manifest.manifest_path,
        "status": completed_manifest.status.value,
        "elapsed_s": report["timing"]["elapsed_s"],
        "peak_cpu_memory_bytes": report["timing"]["peak_cpu_memory_bytes"],
    }


def build_pass1_sanity_report(
    manifest: DistributedRunManifest,
    payloads: Dict[str, Dict[str, object]],
    *,
    artifact_paths: Dict[str, str],
    seq_latent_index_report: Dict[str, object],
    elapsed_s: float,
    peak_cpu_memory_bytes: int,
) -> Dict[str, object]:
    """Build a JSON-serializable sanity report for merged pass-1 artifacts."""

    return {
        "run_id": manifest.run_id,
        "config_hash": manifest.normalized_config_hash,
        "status": "completed",
        "artifacts": {
            name: {
                "path": artifact_paths[name],
                "tensors": _tensor_summary(payload),
            }
            for name, payload in payloads.items()
        },
        "sequence_id_range": _sequence_id_range(payloads),
        "context_fill_rates": {
            "top_ctx": _context_fill_rate(payloads["top_ctx"]),
            "mid_ctx": _context_fill_rate(payloads["mid_ctx"]),
        },
        "seq_repr_fill": _seq_repr_fill(payloads["seq_repr"]),
        "logit_ctx_counts": _logit_ctx_count_summary(payloads["logit_ctx"]),
        "seq_latent_index": seq_latent_index_report,
        "timing": {
            "elapsed_s": float(elapsed_s),
            "peak_cpu_memory_bytes": int(peak_cpu_memory_bytes),
        },
    }


def _with_canonical_metadata(
    payload: Dict[str, object],
    manifest: DistributedRunManifest,
    artifact_name: str,
) -> Dict[str, object]:
    enriched = dict(payload)
    enriched["metadata"] = {
        "schema_version": 1,
        "artifact_name": artifact_name,
        "run_id": manifest.run_id,
        "config_hash": manifest.normalized_config_hash,
        "manifest_path": manifest.manifest_path,
        "source": "distributed_pass1_merge",
    }
    enriched["config_hash"] = manifest.normalized_config_hash
    return enriched


def _merge_welford_state(
    count_a: torch.Tensor,
    mean_a: torch.Tensor,
    m2_a: torch.Tensor,
    count_b: torch.Tensor,
    mean_b: torch.Tensor,
    m2_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count_a_f = count_a.to(torch.float64)
    count_b_f = count_b.to(torch.float64)
    mean_a_f = mean_a.to(torch.float64)
    mean_b_f = mean_b.to(torch.float64)
    m2_a_f = m2_a.to(torch.float64)
    m2_b_f = m2_b.to(torch.float64)

    count_total_f = count_a_f + count_b_f
    safe_count_total = count_total_f.clamp(min=1)
    delta = mean_b_f - mean_a_f
    merged_mean = mean_a_f + delta * (count_b_f / safe_count_total)
    merged_m2 = m2_a_f + m2_b_f + delta.square() * (
        count_a_f * count_b_f / safe_count_total
    )
    merged_mean = torch.where(count_total_f > 0, merged_mean, torch.zeros_like(merged_mean))
    merged_m2 = torch.where(count_total_f > 0, merged_m2, torch.zeros_like(merged_m2))
    return (
        count_total_f.to(torch.int64),
        merged_mean.to(mean_a.dtype),
        merged_m2.to(m2_a.dtype),
    )


def _validate_latent_stats_partial_set(
    partials: Sequence[LatentStatsPartial],
) -> None:
    seen_workers: set[int] = set()
    first_metadata = partials[0][0]
    for metadata, payload in partials:
        if metadata.artifact_name != "latent_stats":
            raise ValueError("all partials must be latent_stats artifacts")
        if metadata.worker_id in seen_workers:
            raise ValueError(f"duplicate latent_stats partial for worker {metadata.worker_id}")
        seen_workers.add(metadata.worker_id)
        if metadata.run_id != first_metadata.run_id:
            raise ValueError("latent_stats partial run_id mismatch")
        if metadata.config_hash != first_metadata.config_hash:
            raise ValueError("latent_stats partial config hash mismatch")
        if metadata.component_count != first_metadata.component_count:
            raise ValueError("latent_stats partial component count mismatch")
        if metadata.d_sae != first_metadata.d_sae:
            raise ValueError("latent_stats partial d_sae mismatch")
        validate_pass1_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_artifact_name="latent_stats",
            expected_config_hash=first_metadata.config_hash,
        )


def _validate_top_ctx_partial_set(
    partials: Sequence[TopCtxPartial],
) -> None:
    seen_workers: set[int] = set()
    first_metadata = partials[0][0]
    first_shape = partials[0][1]["ctx_seq_idx"].shape
    for metadata, payload in partials:
        if metadata.artifact_name != "top_ctx":
            raise ValueError("all partials must be top_ctx artifacts")
        if metadata.worker_id in seen_workers:
            raise ValueError(f"duplicate top_ctx partial for worker {metadata.worker_id}")
        seen_workers.add(metadata.worker_id)
        if metadata.run_id != first_metadata.run_id:
            raise ValueError("top_ctx partial run_id mismatch")
        if metadata.config_hash != first_metadata.config_hash:
            raise ValueError("top_ctx partial config hash mismatch")
        if metadata.component_count != first_metadata.component_count:
            raise ValueError("top_ctx partial component count mismatch")
        if metadata.d_sae != first_metadata.d_sae:
            raise ValueError("top_ctx partial d_sae mismatch")
        if payload["ctx_seq_idx"].shape != first_shape:
            raise ValueError("top_ctx partial shape mismatch")
        validate_pass1_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_artifact_name="top_ctx",
            expected_config_hash=first_metadata.config_hash,
        )
        _validate_top_ctx_sequence_range(metadata, payload)


def _validate_mid_ctx_candidate_partial_set(
    partials: Sequence[MidCtxCandidatesPartial],
) -> None:
    seen_workers: set[int] = set()
    first_metadata = partials[0][0]
    for metadata, payload in partials:
        if metadata.artifact_name != "mid_ctx_candidates":
            raise ValueError("all partials must be mid_ctx_candidates artifacts")
        if metadata.worker_id in seen_workers:
            raise ValueError(
                f"duplicate mid_ctx_candidates partial for worker {metadata.worker_id}"
            )
        seen_workers.add(metadata.worker_id)
        if metadata.run_id != first_metadata.run_id:
            raise ValueError("mid_ctx_candidates partial run_id mismatch")
        if metadata.config_hash != first_metadata.config_hash:
            raise ValueError("mid_ctx_candidates partial config hash mismatch")
        if metadata.component_count != first_metadata.component_count:
            raise ValueError("mid_ctx_candidates partial component count mismatch")
        if metadata.d_sae != first_metadata.d_sae:
            raise ValueError("mid_ctx_candidates partial d_sae mismatch")
        validate_pass1_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_artifact_name="mid_ctx_candidates",
            expected_config_hash=first_metadata.config_hash,
        )


def _validate_seq_repr_partial_set(
    partials: Sequence[SeqReprPartial],
) -> None:
    seen_workers: set[int] = set()
    first_metadata = partials[0][0]
    first_payload = partials[0][1]
    for metadata, payload in partials:
        if metadata.artifact_name != "seq_repr":
            raise ValueError("all partials must be seq_repr artifacts")
        if metadata.worker_id in seen_workers:
            raise ValueError(f"duplicate seq_repr partial for worker {metadata.worker_id}")
        seen_workers.add(metadata.worker_id)
        if metadata.run_id != first_metadata.run_id:
            raise ValueError("seq_repr partial run_id mismatch")
        if metadata.config_hash != first_metadata.config_hash:
            raise ValueError("seq_repr partial config hash mismatch")
        for key in ["repr_mode", "repr_dim", "n_seqs"]:
            if payload[key] != first_payload[key]:
                raise ValueError(f"seq_repr partial {key} mismatch")
        validate_pass1_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_artifact_name="seq_repr",
            expected_config_hash=first_metadata.config_hash,
        )
        _validate_seq_repr_mapping_compatibility(first_payload, payload)


def _validate_logit_ctx_partial_set(
    partials: Sequence[LogitCtxPartial],
    *,
    vocab_size: int | None = None,
) -> None:
    seen_workers: set[int] = set()
    first_metadata = partials[0][0]
    first_shape = partials[0][1]["top_tokens"].shape
    for metadata, payload in partials:
        if metadata.artifact_name != "logit_ctx":
            raise ValueError("all partials must be logit_ctx artifacts")
        if metadata.worker_id in seen_workers:
            raise ValueError(f"duplicate logit_ctx partial for worker {metadata.worker_id}")
        seen_workers.add(metadata.worker_id)
        if metadata.run_id != first_metadata.run_id:
            raise ValueError("logit_ctx partial run_id mismatch")
        if metadata.config_hash != first_metadata.config_hash:
            raise ValueError("logit_ctx partial config hash mismatch")
        if metadata.component_count != first_metadata.component_count:
            raise ValueError("logit_ctx partial component count mismatch")
        if metadata.d_sae != first_metadata.d_sae:
            raise ValueError("logit_ctx partial d_sae mismatch")
        if payload["top_tokens"].shape != first_shape:
            raise ValueError("logit_ctx partial top tensor shape mismatch")
        validate_pass1_partial(
            {"metadata": metadata.model_dump(mode="json"), "payload": payload},
            expected_artifact_name="logit_ctx",
            expected_config_hash=first_metadata.config_hash,
        )
        _validate_logit_ctx_token_range(payload, vocab_size=vocab_size)


def _validate_merged_latent_stats(
    merged: Dict[str, object],
    partials: Sequence[LatentStatsPartial],
) -> None:
    for count_name in ["active_count", "seq_count"]:
        expected = sum(
            (payload[count_name] for _metadata, payload in partials),
            start=torch.zeros_like(partials[0][1][count_name]),
        )
        if not torch.equal(merged[count_name], expected):
            raise ValueError(f"merged {count_name} does not equal sum of partial counts")

    for tensor_name in ["mean", "mean_abs", "m2", "m2_abs", "mean_seq", "m2_seq"]:
        tensor = merged[tensor_name]
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"merged {tensor_name} contains non-finite values")
    for tensor_name in ["m2", "m2_abs", "m2_seq"]:
        tensor = merged[tensor_name]
        if bool((tensor < 0).any()):
            raise ValueError(f"merged {tensor_name} contains negative variance state")


def _validate_top_ctx_sequence_range(
    metadata: Pass1PartialMetadata,
    payload: Dict[str, object],
) -> None:
    sequence_ids = payload["ctx_seq_idx"]
    valid = sequence_ids != 0
    if not bool(valid.any()):
        return
    if metadata.sequence_id_min is not None and int(sequence_ids[valid].min()) < metadata.sequence_id_min:
        raise ValueError("top_ctx sequence IDs below worker range")
    if metadata.sequence_id_max is not None and int(sequence_ids[valid].max()) > metadata.sequence_id_max:
        raise ValueError("top_ctx sequence IDs above worker range")


def _validate_merged_top_ctx(
    merged: Dict[str, object],
    partials: Sequence[TopCtxPartial],
) -> None:
    metadata = partials[0][0]
    validate_pass1_partial(
        {"metadata": metadata.model_copy(update={"worker_id": 0}).model_dump(mode="json"), "payload": merged},
        expected_artifact_name="top_ctx",
        expected_config_hash=metadata.config_hash,
    )
    values = merged["ctx_seq_val"].float()
    indices = merged["ctx_seq_idx"]
    if bool((values < 0).any()):
        raise ValueError("merged top_ctx values must be non-negative")
    global_min = min(
        item.sequence_id_min for item, _payload in partials if item.sequence_id_min is not None
    )
    global_max = max(
        item.sequence_id_max for item, _payload in partials if item.sequence_id_max is not None
    )
    valid = indices != 0
    if bool(valid.any()):
        if int(indices[valid].min()) < global_min:
            raise ValueError("merged top_ctx sequence IDs below global range")
        if int(indices[valid].max()) > global_max:
            raise ValueError("merged top_ctx sequence IDs above global range")
    invalid = ~valid
    if bool((values[invalid] != 0).any()):
        raise ValueError("merged top_ctx invalid sentinel values must be zero")


def _validate_latent_stats_for_mid_ctx(
    latent_stats_payload: Dict[str, object],
    metadata: Pass1PartialMetadata,
) -> None:
    shape = (metadata.component_count, metadata.d_sae)
    for name in ["seq_count", "mean_seq", "m2_seq"]:
        tensor = latent_stats_payload.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"latent_stats payload missing valid {name}")
        if name != "seq_count" and not torch.isfinite(tensor.float()).all():
            raise ValueError(f"latent_stats {name} contains non-finite values")


def _std_seq_from_latent_stats(latent_stats_payload: Dict[str, object]) -> torch.Tensor:
    seq_count = latent_stats_payload["seq_count"].to(torch.float32)
    m2_seq = latent_stats_payload["m2_seq"].to(torch.float32)
    return (m2_seq / (seq_count - 1).clamp(min=1)).clamp(min=0).sqrt()


def _concatenate_mid_ctx_candidates(
    partials: Sequence[MidCtxCandidatesPartial],
) -> Dict[str, torch.Tensor]:
    names = [
        "component_ids",
        "latent_ids",
        "sequence_ids",
        "activation_values",
        "priorities",
    ]
    return {
        name: torch.cat([payload[name] for _metadata, payload in partials], dim=0)
        for name in names
    }


def _select_mid_ctx_candidates(
    ctx_seq_idx: torch.Tensor,
    ctx_seq_val: torch.Tensor,
    reservoir_fill: torch.Tensor,
    component_ids: torch.Tensor,
    latent_ids: torch.Tensor,
    sequence_ids: torch.Tensor,
    activation_values: torch.Tensor,
    priorities: torch.Tensor,
    output_k: int,
) -> None:
    groups = torch.stack([component_ids, latent_ids], dim=1).unique(dim=0)
    for component_id, latent_id in groups.tolist():
        mask = (component_ids == component_id) & (latent_ids == latent_id)
        group_priorities = priorities[mask]
        group_sequences = sequence_ids[mask].to(torch.long)
        group_values = activation_values[mask]
        order_by_sequence = torch.argsort(group_sequences, stable=True)
        priority_sorted = group_priorities[order_by_sequence]
        sequences_sorted = group_sequences[order_by_sequence]
        values_sorted = group_values[order_by_sequence]
        order_by_priority = torch.argsort(priority_sorted, stable=True)
        selected = order_by_priority[:output_k]
        selected_count = int(selected.numel())
        if selected_count == 0:
            continue
        ctx_seq_idx[component_id, latent_id, :selected_count] = sequences_sorted[
            selected
        ].to(torch.int32)
        ctx_seq_val[component_id, latent_id, :selected_count] = values_sorted[
            selected
        ].to(torch.float32)
        reservoir_fill[component_id, latent_id] = selected_count


def _validate_merged_mid_ctx(
    merged: Dict[str, object],
    partials: Sequence[MidCtxCandidatesPartial],
) -> None:
    metadata = partials[0][0]
    idx = merged["ctx_seq_idx"]
    vals = merged["ctx_seq_val"]
    if idx.dtype != torch.int32:
        raise ValueError("merged mid_ctx sequence IDs must be int32")
    if idx.shape != vals.shape:
        raise ValueError("merged mid_ctx tensors must have matching shapes")
    if idx.ndim != 3 or idx.shape[0] != metadata.component_count or idx.shape[1] != metadata.d_sae:
        raise ValueError("merged mid_ctx tensors have invalid shape")
    if not torch.isfinite(vals.float()).all():
        raise ValueError("merged mid_ctx values contain non-finite values")
    if bool((vals < 0).any()):
        raise ValueError("merged mid_ctx values must be non-negative")
    invalid = idx == 0
    if bool((vals[invalid] != 0).any()):
        raise ValueError("merged mid_ctx invalid sentinel values must be zero")


def _mapping_from_seq_repr_payload(payload: Dict[str, object]) -> dict[str, object]:
    n_seqs = int(payload["n_seqs"])
    n_stored = int(payload["n_stored"])
    if bool(payload["is_capped"]):
        return {
            "slot_to_id": payload["slot_to_id"],
            "id_to_slot": payload["id_to_slot"],
            "n_seqs": n_seqs,
            "n_stored": n_stored,
            "is_capped": True,
        }
    return {
        "slot_to_id": torch.arange(n_seqs + 1, dtype=torch.int64),
        "id_to_slot": torch.arange(n_seqs + 1, dtype=torch.int32),
        "n_seqs": n_seqs,
        "n_stored": n_stored,
        "is_capped": False,
    }


def _validate_seq_repr_mapping_compatibility(
    first_payload: Dict[str, object],
    payload: Dict[str, object],
) -> None:
    if bool(first_payload["is_capped"]) != bool(payload["is_capped"]):
        raise ValueError("seq_repr partial cap mode mismatch")
    if bool(first_payload["is_capped"]):
        if not torch.equal(first_payload["slot_to_id"], payload["slot_to_id"]):
            raise ValueError("seq_repr partial slot_to_id mismatch")
        if not torch.equal(first_payload["id_to_slot"], payload["id_to_slot"]):
            raise ValueError("seq_repr partial id_to_slot mismatch")


def _validate_merged_seq_repr(merged: Dict[str, object]) -> None:
    repr_buf = merged["repr_buf"]
    if not isinstance(repr_buf, torch.Tensor) or repr_buf.ndim != 2:
        raise ValueError("merged seq_repr repr_buf must be 2D")
    if not torch.isfinite(repr_buf.float()).all():
        raise ValueError("merged seq_repr repr_buf contains non-finite values")
    expected_shape = (int(merged["n_stored"]) + 1, int(merged["repr_dim"]))
    if tuple(repr_buf.shape) != expected_shape:
        raise ValueError("merged seq_repr repr_buf shape mismatch")
    if bool(merged["is_capped"]):
        validate_seq_repr_mapping(
            {
                "slot_to_id": merged["slot_to_id"],
                "id_to_slot": merged["id_to_slot"],
                "n_seqs": merged["n_seqs"],
                "n_stored": merged["n_stored"],
                "is_capped": True,
            }
        )


def _select_logit_ctx_events(
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    candidate_tokens: torch.Tensor,
    candidate_probs: torch.Tensor,
    candidate_workers: torch.Tensor,
    candidate_rows: torch.Tensor,
    output_k: int,
) -> None:
    component_count, d_sae, _candidate_count = candidate_tokens.shape
    for component_id in range(component_count):
        for latent_id in range(d_sae):
            probs = candidate_probs[component_id, latent_id]
            valid = probs > 0
            if not bool(valid.any()):
                continue
            tokens = candidate_tokens[component_id, latent_id][valid]
            workers = candidate_workers[component_id, latent_id][valid]
            rows = candidate_rows[component_id, latent_id][valid]
            probs = probs[valid]

            order = torch.argsort(rows, stable=True)
            tokens = tokens[order]
            workers = workers[order]
            rows = rows[order]
            probs = probs[order]

            order = torch.argsort(workers, stable=True)
            tokens = tokens[order]
            workers = workers[order]
            rows = rows[order]
            probs = probs[order]

            order = torch.argsort(tokens, stable=True)
            tokens = tokens[order]
            probs = probs[order]

            order = torch.argsort(probs, descending=True, stable=True)
            selected = order[:output_k]
            selected_count = int(selected.numel())
            top_tokens[component_id, latent_id, :selected_count] = tokens[selected].to(
                torch.int32
            )
            top_probs[component_id, latent_id, :selected_count] = probs[selected].to(
                torch.float32
            )


def _validate_logit_ctx_token_range(
    payload: Dict[str, object],
    *,
    vocab_size: int | None = None,
) -> None:
    tokens = payload["top_tokens"]
    if tokens.numel() > 0 and int(tokens.min()) < 0:
        raise ValueError("logit_ctx token IDs must be non-negative")
    if vocab_size is not None and tokens.numel() > 0 and int(tokens.max()) >= int(vocab_size):
        raise ValueError("logit_ctx token IDs above vocabulary range")


def _validate_merged_logit_ctx(
    merged: Dict[str, object],
    partials: Sequence[LogitCtxPartial],
    *,
    vocab_size: int | None = None,
) -> None:
    expected_counts = sum(
        (payload["latent_counts"].to(torch.int64) for _metadata, payload in partials),
        start=torch.zeros_like(partials[0][1]["latent_counts"], dtype=torch.int64),
    )
    if not torch.equal(merged["latent_counts"], expected_counts):
        raise ValueError("merged logit_ctx latent_counts does not equal sum of partial counts")
    top_tokens = merged["top_tokens"]
    top_probs = merged["top_probs"]
    if top_tokens.dtype != torch.int32:
        raise ValueError("merged logit_ctx top_tokens must be int32")
    if top_probs.shape != top_tokens.shape or top_probs.ndim != 3:
        raise ValueError("merged logit_ctx top tensors have invalid shape")
    if not torch.isfinite(top_probs.float()).all():
        raise ValueError("merged logit_ctx top_probs contains non-finite values")
    if bool((top_probs < 0).any()):
        raise ValueError("merged logit_ctx top_probs must be non-negative")
    _validate_logit_ctx_token_range(merged, vocab_size=vocab_size)
    invalid = top_probs == 0
    if bool((top_tokens[invalid] != 0).any()):
        raise ValueError("merged logit_ctx invalid sentinel tokens must be zero")


def _parse_seq_latent_index_shard_id(path: Path) -> int:
    stem = path.stem
    prefix = "shard_"
    if not stem.startswith(prefix):
        raise ValueError(f"invalid seq_latent_index shard filename: {path.name}")
    try:
        return int(stem[len(prefix) :])
    except ValueError as exc:
        raise ValueError(f"invalid seq_latent_index shard filename: {path.name}") from exc


def _validate_seq_latent_index_shard_file(
    path: Path,
    *,
    shard_id: int,
    shard_id_ranges: Dict[int, tuple[int, int]] | None = None,
) -> Dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"seq_latent_index shard_{shard_id}.pt must contain a dict")
    for component_id, tensor in payload.items():
        if not isinstance(component_id, int):
            raise ValueError("seq_latent_index component keys must be integers")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("seq_latent_index component values must be tensors")
        if tensor.dtype != torch.int32 or tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError("seq_latent_index tensors must have shape [N, 2] and dtype int32")
        if tensor.numel() == 0:
            continue
        sequence_ids = tensor[:, 0]
        latent_ids = tensor[:, 1]
        if int(sequence_ids.min()) < 1:
            raise ValueError("seq_latent_index sequence IDs must be positive")
        if int(latent_ids.min()) < 0:
            raise ValueError("seq_latent_index latent IDs must be non-negative")
        if shard_id_ranges is not None:
            if shard_id not in shard_id_ranges:
                raise ValueError(f"missing expected sequence range for shard {shard_id}")
            start_id, end_id = shard_id_ranges[shard_id]
            if int(sequence_ids.min()) < start_id or int(sequence_ids.max()) > end_id:
                raise ValueError("seq_latent_index sequence IDs outside shard range")
    return payload


def _seq_latent_index_files_equivalent(first: Path, second: Path) -> bool:
    if first.read_bytes() == second.read_bytes():
        return True
    first_payload = _validate_seq_latent_index_shard_file(first, shard_id=_parse_seq_latent_index_shard_id(first))
    second_payload = _validate_seq_latent_index_shard_file(second, shard_id=_parse_seq_latent_index_shard_id(second))
    if set(first_payload) != set(second_payload):
        return False
    return all(torch.equal(first_payload[key], second_payload[key]) for key in first_payload)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f"{destination.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    shutil.copyfile(source, tmp_path)
    tmp_path.replace(destination)


def _pass1_partial_paths(manifest: DistributedRunManifest) -> Dict[str, list[Path]]:
    layout = build_run_layout(manifest)
    partial_paths: dict[str, list[Path]] = {
        name: [] for name in PASS1_PARTIAL_FILENAMES
    }
    for worker_id in range(manifest.worker_count):
        worker_dir = layout.workers[worker_id].pass1_dir
        for artifact_name, filename in PASS1_PARTIAL_FILENAMES.items():
            path = worker_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"missing pass1 partial: {path}")
            partial_paths[artifact_name].append(path)
    return partial_paths


def _atomic_torch_save(payload: Dict[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)


def _atomic_write_json(path: str | Path, payload: Dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def _validate_written_artifacts(
    artifacts: Dict[str, tuple[Path, Dict[str, object]]],
) -> None:
    for artifact_name, (path, expected_payload) in artifacts.items():
        if not path.exists():
            raise FileNotFoundError(f"merged artifact was not written: {path}")
        loaded = torch.load(path, map_location="cpu")
        if not isinstance(loaded, dict):
            raise ValueError(f"merged {artifact_name} artifact must contain a dict")
        for key, expected_value in expected_payload.items():
            if isinstance(expected_value, torch.Tensor):
                loaded_value = loaded.get(key)
                if not isinstance(loaded_value, torch.Tensor) or not torch.equal(
                    loaded_value.cpu(), expected_value.cpu()
                ):
                    raise ValueError(f"merged {artifact_name}.{key} failed validation")


def _tensor_summary(payload: Dict[str, object]) -> Dict[str, object]:
    summary: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            finite = bool(torch.isfinite(tensor.float()).all()) if tensor.numel() else True
            item: dict[str, object] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "finite": finite,
            }
            if tensor.numel() and not tensor.is_floating_point():
                item["min"] = int(tensor.min().item())
                item["max"] = int(tensor.max().item())
            summary[key] = item
    return summary


def _sequence_id_range(payloads: Dict[str, Dict[str, object]]) -> Dict[str, int | None]:
    sequence_tensors: list[torch.Tensor] = []
    for payload in [payloads["top_ctx"], payloads["mid_ctx"]]:
        idx = payload.get("ctx_seq_idx")
        if isinstance(idx, torch.Tensor):
            valid = idx[idx > 0]
            if valid.numel():
                sequence_tensors.append(valid.to(torch.int64))
    seq_repr = payloads["seq_repr"]
    if bool(seq_repr.get("is_capped")) and isinstance(seq_repr.get("slot_to_id"), torch.Tensor):
        valid = seq_repr["slot_to_id"][1:].to(torch.int64)
        if valid.numel():
            sequence_tensors.append(valid)
    if not sequence_tensors:
        return {"min": None, "max": None}
    all_ids = torch.cat(sequence_tensors)
    return {"min": int(all_ids.min().item()), "max": int(all_ids.max().item())}


def _context_fill_rate(payload: Dict[str, object]) -> float:
    idx = payload.get("ctx_seq_idx")
    if not isinstance(idx, torch.Tensor) or idx.numel() == 0:
        return 0.0
    return float((idx > 0).sum().item() / idx.numel())


def _seq_repr_fill(payload: Dict[str, object]) -> Dict[str, object]:
    repr_buf = payload["repr_buf"]
    filled = int((repr_buf[1:].float().abs().sum(dim=1) > 0).sum().item())
    n_stored = int(payload["n_stored"])
    return {
        "filled": filled,
        "n_stored": n_stored,
        "fill_rate": float(filled / n_stored) if n_stored else 0.0,
        "is_capped": bool(payload["is_capped"]),
    }


def _logit_ctx_count_summary(payload: Dict[str, object]) -> Dict[str, int]:
    latent_counts = payload["latent_counts"]
    return {
        "total": int(latent_counts.sum().item()),
        "nonzero_latents": int((latent_counts > 0).sum().item()),
        "max": int(latent_counts.max().item()) if latent_counts.numel() else 0,
    }
