"""Latent-stats merge helpers for distributed pass 1."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import torch

from ..pass1_partials import load_pass1_partial, validate_pass1_partial
from .contracts import LatentStatsPartial


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
    for tensor_name in ["m2", "m2_abs", "m2_seq"]:
        merged[tensor_name] = _clamp_small_negative_variance_state(
            tensor_name,
            merged[tensor_name],
        )
    _validate_merged_latent_stats(merged, partials)
    return merged


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
    m2_a_f = _clamp_small_negative_variance_state("m2_a", m2_a).to(torch.float64)
    m2_b_f = _clamp_small_negative_variance_state("m2_b", m2_b).to(torch.float64)

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


def _clamp_small_negative_variance_state(
    tensor_name: str,
    tensor: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-6,
) -> torch.Tensor:
    """Clamp tiny float32 Welford variance noise while rejecting real negatives."""

    negative = tensor < 0
    if not bool(negative.any()):
        return tensor
    min_value = float(tensor[negative].min())
    tolerance = max(atol, float(tensor.detach().abs().max()) * rtol)
    if min_value < -tolerance:
        raise ValueError(
            f"merged {tensor_name} contains negative variance state "
            f"(min={min_value:.6g}, tolerance={tolerance:.6g})"
        )
    return tensor.clamp_min(0)


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


__all__ = [
    "load_and_merge_latent_stats_partials",
    "merge_latent_stats_partials",
]
