"""Shared loading and validation for discovery-stage global artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Optional

import torch

from config import config
from store.context import mid_ctx, neg_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.top_coactivation import top_coactivation


DISCOVERY_REQUIRED_ARTIFACTS = (
    "latent_stats",
    "top_ctx",
    "mid_ctx",
    "neg_ctx",
    "logit_ctx",
    "top_coactivation",
)


@dataclass(frozen=True)
class DiscoveryArtifactValidation:
    run_root: Path
    paths: Dict[str, Path]
    component_count: int
    d_sae: int
    top_coactivation_mode: Optional[str]
    candidate_count: Optional[int] = None


def discovery_artifact_paths(
    output_root: str | Path = "outputs",
    *,
    candidates_path: str | Path | None = None,
) -> Dict[str, Path]:
    """Resolve the canonical discovery input artifacts for a run root."""

    root = Path(output_root)
    paths = {
        "latent_stats": root / "latent_stats.pt",
        "top_ctx": root / "top_ctx.pt",
        "mid_ctx": root / "mid_ctx.pt",
        "neg_ctx": root / "neg_ctx.pt",
        "logit_ctx": root / "logit_ctx.pt",
        "top_coactivation": root / "top_coactivation.pt",
    }
    if candidates_path is not None:
        paths["candidates"] = Path(candidates_path)
    return paths


def validate_discovery_artifacts(
    output_root: str | Path = "outputs",
    *,
    candidates_path: str | Path | None = None,
) -> DiscoveryArtifactValidation:
    """Validate discovery inputs before model/SAE initialization."""

    root = Path(output_root)
    paths = discovery_artifact_paths(root, candidates_path=candidates_path)
    _reject_missing(paths)
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }

    latent_shape = _require_shape(payloads["latent_stats"], "latent_stats", "active_count", ndim=2)
    component_count, d_sae = latent_shape
    _require_shape(payloads["latent_stats"], "latent_stats", "seq_count", ndim=2, expected=latent_shape)
    _require_shape(payloads["latent_stats"], "latent_stats", "mean_seq", ndim=2, expected=latent_shape)

    for name in ("top_ctx", "mid_ctx", "neg_ctx"):
        _require_shape(payloads[name], name, "ctx_seq_idx", ndim=3, prefix=latent_shape)
        _require_shape(payloads[name], name, "ctx_seq_val", ndim=3, prefix=latent_shape)

    _require_shape(payloads["logit_ctx"], "logit_ctx", "latent_counts", ndim=2, expected=latent_shape)
    _require_shape(payloads["logit_ctx"], "logit_ctx", "top_tokens", ndim=3, prefix=latent_shape)
    _require_shape(payloads["logit_ctx"], "logit_ctx", "top_probs", ndim=3, prefix=latent_shape)

    _require_shape(payloads["top_coactivation"], "top_coactivation", "top_indices", ndim=3, prefix=latent_shape)
    _require_shape(payloads["top_coactivation"], "top_coactivation", "top_values", ndim=3, prefix=latent_shape)
    stored_mode = payloads["top_coactivation"].get("mode")
    configured_mode = str(config.latents.top_coactivation.mode or "freq_weighted")
    if stored_mode is not None and stored_mode != configured_mode:
        raise ValueError(
            f"top_coactivation mode mismatch: artifact={stored_mode!r}, config={configured_mode!r}"
        )

    candidate_count: Optional[int] = None
    if "candidates" in payloads:
        candidates = payloads["candidates"]
        if not isinstance(candidates, list):
            raise ValueError("candidates artifact must contain a list")
        candidate_count = len(candidates)
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError(f"candidate {index} must be a dict")
            if "comp_idx" not in candidate or "latent_idx" not in candidate:
                raise ValueError(f"candidate {index} missing comp_idx/latent_idx")

    return DiscoveryArtifactValidation(
        run_root=root,
        paths=paths,
        component_count=component_count,
        d_sae=d_sae,
        top_coactivation_mode=stored_mode,
        candidate_count=candidate_count,
    )


def load_discovery_artifacts(
    output_root: str | Path = "outputs",
    *,
    candidates_path: str | Path | None = None,
) -> DiscoveryArtifactValidation:
    """Validate and load all global stores needed by discovery."""

    validation = validate_discovery_artifacts(output_root, candidates_path=candidates_path)
    paths = validation.paths
    latent_stats.load(str(paths["latent_stats"]))
    top_ctx.load(str(paths["top_ctx"]))
    mid_ctx.load(str(paths["mid_ctx"]))
    neg_ctx.load(str(paths["neg_ctx"]))
    logit_ctx.load(str(paths["logit_ctx"]))
    top_coactivation.load(str(paths["top_coactivation"]))
    return validation


def hash_discovery_artifacts(
    output_root: str | Path = "outputs",
    *,
    candidates_path: str | Path | None = None,
) -> Dict[str, str]:
    """Return SHA-256 hashes for discovery input artifacts."""

    paths = discovery_artifact_paths(output_root, candidates_path=candidates_path)
    _reject_missing(paths)
    return {name: _sha256_file(path) for name, path in paths.items()}


def _reject_missing(paths: Dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing discovery input artifacts: {sorted(missing)}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_shape(
    payload: object,
    artifact_name: str,
    tensor_name: str,
    *,
    ndim: int,
    expected: tuple[int, ...] | None = None,
    prefix: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} artifact must contain a dict payload")
    if tensor_name not in payload:
        raise ValueError(f"{artifact_name} missing required tensor {tensor_name}")
    tensor = payload[tensor_name]
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{artifact_name}.{tensor_name} must be a tensor")
    shape = tuple(int(dim) for dim in tensor.shape)
    if len(shape) != ndim:
        raise ValueError(f"{artifact_name}.{tensor_name} must be {ndim}D")
    if expected is not None and shape != expected:
        raise ValueError(f"{artifact_name}.{tensor_name} shape mismatch")
    if prefix is not None and shape[: len(prefix)] != prefix:
        raise ValueError(f"{artifact_name}.{tensor_name} leading dimensions mismatch")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"{artifact_name}.{tensor_name} must be finite")
    return shape
