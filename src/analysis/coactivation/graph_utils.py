"""Shared helpers for latent-level coactivation graph analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from analysis.io import resolve_run_root


@dataclass(frozen=True)
class HighPmiEdges:
    """Directed high-PMI edges between flattened latent IDs."""

    source: torch.Tensor
    dest: torch.Tensor
    score: torch.Tensor
    num_latents: int
    threshold: float

    @property
    def count(self) -> int:
        return int(self.source.numel())


@dataclass(frozen=True)
class TopContextArtifact:
    """Validated tensors from `top_ctx.pt`."""

    path: Path
    ctx_seq_idx: torch.Tensor
    ctx_seq_val: torch.Tensor | None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self.ctx_seq_idx.shape)


def build_high_pmi_edges(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    threshold: float = 2.0,
) -> HighPmiEdges:
    """Build compact directed edge tensors for all stored PMI scores above threshold."""

    if top_values.ndim != 3 or top_indices.ndim != 3:
        raise ValueError("top_values and top_indices must have shape [components, d_sae, top_k]")
    if tuple(top_values.shape) != tuple(top_indices.shape):
        raise ValueError("top_values and top_indices shapes must match")

    values = top_values.detach().cpu().to(torch.float32)
    indices = top_indices.detach().cpu().to(torch.int64)
    num_components, d_sae, top_k = (int(dim) for dim in values.shape)
    num_latents = num_components * d_sae
    sources: list[torch.Tensor] = []
    dests: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []

    for component in range(num_components):
        component_values = values[component].reshape(d_sae, top_k)
        component_indices = indices[component].reshape(d_sae, top_k).clamp(min=0, max=num_latents - 1)
        mask = component_values > float(threshold)
        if not bool(mask.any()):
            continue
        rows, cols = mask.nonzero(as_tuple=True)
        sources.append(rows.to(torch.int64) + component * d_sae)
        dests.append(component_indices[rows, cols])
        scores.append(component_values[rows, cols])

    if not sources:
        empty_i64 = torch.empty(0, dtype=torch.int64)
        empty_f32 = torch.empty(0, dtype=torch.float32)
        return HighPmiEdges(empty_i64, empty_i64, empty_f32, num_latents, float(threshold))

    return HighPmiEdges(
        source=torch.cat(sources),
        dest=torch.cat(dests),
        score=torch.cat(scores),
        num_latents=num_latents,
        threshold=float(threshold),
    )


def edge_codes(source: torch.Tensor, dest: torch.Tensor, num_latents: int) -> torch.Tensor:
    """Encode directed edge endpoints as a single int64 code."""

    return source.to(torch.int64) * int(num_latents) + dest.to(torch.int64)


def high_pmi_in_degree(edges: HighPmiEdges) -> torch.Tensor:
    """Count high-PMI incoming edges per coacting latent."""

    return torch.bincount(edges.dest, minlength=edges.num_latents).to(torch.int64)


def top_edges_by_score(
    source: torch.Tensor,
    dest: torch.Tensor,
    score: torch.Tensor,
    *,
    d_sae: int,
    limit: int,
    extra_columns: Mapping[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Return top edge rows as JSON/CSV-friendly dictionaries."""

    if score.numel() == 0:
        return []
    top_count = min(int(limit), int(score.numel()))
    top_scores, top_positions = torch.topk(score, k=top_count)
    rows = []
    for rank, (edge_score, pos) in enumerate(zip(top_scores.tolist(), top_positions.tolist()), start=1):
        src = int(source[pos].item())
        dst = int(dest[pos].item())
        row: dict[str, Any] = {
            "rank": rank,
            "source_global_id": src,
            "source_component": src // int(d_sae),
            "source_latent": src % int(d_sae),
            "dest_global_id": dst,
            "dest_component": dst // int(d_sae),
            "dest_latent": dst % int(d_sae),
            "score": float(edge_score),
        }
        if extra_columns:
            for key, values in extra_columns.items():
                value = values[pos]
                row[key] = float(value.item()) if value.is_floating_point() else int(value.item())
        rows.append(row)
    return rows


def load_top_context(run_root: str | Path) -> TopContextArtifact:
    """Load and validate `top_ctx.pt` from a run root."""

    root = resolve_run_root(run_root)
    artifact_path = root / "top_ctx.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"top_ctx.pt not found under run root: {artifact_path}")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"top_ctx artifact must be a mapping, got {type(payload).__name__}")
    if "ctx_seq_idx" not in payload:
        raise KeyError(f"{artifact_path} missing required field: ctx_seq_idx")
    ctx_seq_idx = payload["ctx_seq_idx"]
    if not isinstance(ctx_seq_idx, torch.Tensor) or ctx_seq_idx.ndim != 3:
        raise ValueError("ctx_seq_idx must be a tensor with shape [components, d_sae, top_ctx_k]")
    ctx_seq_val = payload.get("ctx_seq_val")
    if ctx_seq_val is not None and not isinstance(ctx_seq_val, torch.Tensor):
        ctx_seq_val = None
    return TopContextArtifact(
        path=artifact_path,
        ctx_seq_idx=ctx_seq_idx.to(torch.int64),
        ctx_seq_val=ctx_seq_val,
    )


def deterministic_edge_sample(edge_count: int, max_samples: int) -> torch.Tensor:
    """Evenly sample edge row positions."""

    if edge_count <= 0:
        return torch.empty(0, dtype=torch.int64)
    sample_count = min(int(edge_count), int(max_samples))
    return torch.linspace(0, edge_count - 1, steps=sample_count, dtype=torch.float64).round().to(torch.int64).unique()

