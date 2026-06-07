"""Load and validate top-coactivation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from analysis.io import resolve_run_root


@dataclass(frozen=True)
class TopCoactivationArtifact:
    """Validated tensors from a canonical `top_coactivation.pt` artifact."""

    path: Path
    mode: str
    top_indices: torch.Tensor
    top_values: torch.Tensor
    total_tokens_processed: int | None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self.top_values.shape)

    @property
    def num_components(self) -> int:
        return self.shape[0]

    @property
    def d_sae(self) -> int:
        return self.shape[1]

    @property
    def top_k(self) -> int:
        return self.shape[2]

    @property
    def num_targets(self) -> int:
        return self.num_components * self.d_sae


def load_top_coactivation(run_root: str | Path) -> TopCoactivationArtifact:
    """Load the canonical top coactivation artifact for a run."""

    root = resolve_run_root(run_root)
    artifact_path = root / "top_coactivation.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"top_coactivation.pt not found under run root: {artifact_path}")

    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"top_coactivation artifact must be a mapping, got {type(payload).__name__}")

    _require_keys(payload, artifact_path)
    top_indices = payload["top_indices"]
    top_values = payload["top_values"]
    if not isinstance(top_indices, torch.Tensor):
        raise TypeError("top_indices must be a torch.Tensor")
    if not isinstance(top_values, torch.Tensor):
        raise TypeError("top_values must be a torch.Tensor")
    if top_indices.ndim != 3 or top_values.ndim != 3:
        raise ValueError("top_indices and top_values must both have shape [components, d_sae, top_k]")
    if tuple(top_indices.shape) != tuple(top_values.shape):
        raise ValueError("top_indices and top_values shapes must match")
    if not torch.isfinite(top_values.float()).all():
        raise ValueError("top_values must be finite")

    total_tokens = payload.get("total_tokens_processed")
    if total_tokens is not None:
        total_tokens = int(total_tokens)

    return TopCoactivationArtifact(
        path=artifact_path,
        mode=str(payload["mode"]),
        top_indices=top_indices.to(torch.int64),
        top_values=top_values.to(torch.float32),
        total_tokens_processed=total_tokens,
    )


def _require_keys(payload: Mapping[str, Any], artifact_path: Path) -> None:
    missing = {"mode", "top_indices", "top_values"} - set(payload)
    if missing:
        keys = ", ".join(sorted(missing))
        raise KeyError(f"{artifact_path} missing required field(s): {keys}")

