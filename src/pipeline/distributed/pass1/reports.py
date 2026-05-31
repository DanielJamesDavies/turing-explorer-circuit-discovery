"""Report helpers for distributed pass-1 merge."""

from __future__ import annotations

from typing import Dict

import torch

from ..manifest import DistributedRunManifest


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
        "mid_ctx_merge": _mid_ctx_merge_summary(payloads["mid_ctx"]),
        "seq_repr_fill": _seq_repr_fill(payloads["seq_repr"]),
        "logit_ctx_counts": _logit_ctx_count_summary(payloads["logit_ctx"]),
        "seq_latent_index": seq_latent_index_report,
        "timing": {
            "elapsed_s": float(elapsed_s),
            "peak_cpu_memory_bytes": int(peak_cpu_memory_bytes),
        },
    }


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


def _mid_ctx_merge_summary(payload: Dict[str, object]) -> Dict[str, object]:
    merge_report = payload.get("merge_report")
    if not isinstance(merge_report, dict):
        return {"merge_mode": payload.get("mode")}
    reservoir_n = payload.get("reservoir_n")
    reservoir_fill = payload.get("reservoir_fill")
    empty_worker_rows = merge_report.get("empty_worker_rows")
    return {
        "merge_mode": merge_report.get("merge_mode", merge_report.get("mode")),
        "priority_mode": merge_report.get("priority_mode"),
        "priority_hash_version": merge_report.get("priority_hash_version"),
        "num_ctx_sequences": merge_report.get("num_ctx_sequences"),
        "total_reservoir_n": int(reservoir_n.sum().item())
        if isinstance(reservoir_n, torch.Tensor)
        else None,
        "nonzero_reservoir_rows": int((reservoir_n > 0).sum().item())
        if isinstance(reservoir_n, torch.Tensor)
        else None,
        "selected_count": int(reservoir_fill.sum().item())
        if isinstance(reservoir_fill, torch.Tensor)
        else None,
        "any_worker_reservoir_empty": bool(merge_report.get("any_worker_reservoir_empty", False)),
        "empty_worker_row_count": int((empty_worker_rows > 0).sum().item())
        if isinstance(empty_worker_rows, torch.Tensor)
        else None,
    }


def _logit_ctx_count_summary(payload: Dict[str, object]) -> Dict[str, int]:
    latent_counts = payload["latent_counts"]
    return {
        "total": int(latent_counts.sum().item()),
        "nonzero_latents": int((latent_counts > 0).sum().item()),
        "max": int(latent_counts.max().item()) if latent_counts.numel() else 0,
    }


__all__ = ["build_pass1_sanity_report"]
