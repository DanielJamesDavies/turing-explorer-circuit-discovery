"""Shared loading and validation for discovery-stage global artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Optional, cast

import torch

from config import config
from store.context import mid_ctx, neg_ctx, top_ctx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store import seq_repr as seq_repr_store
from store.top_coactivation import top_coactivation


DISCOVERY_REQUIRED_ARTIFACTS = (
    "latent_stats",
    "top_ctx",
    "mid_ctx",
    "neg_ctx",
    "seq_repr",
    "logit_ctx",
    "top_coactivation",
)

GLOBAL_NEGCTX_IDS_FILENAME = "global_negctx_ids.pt"


@dataclass(frozen=True)
class DiscoveryArtifactValidation:
    run_root: Path
    paths: Dict[str, Path]
    component_count: int
    d_sae: int
    top_coactivation_mode: Optional[str]
    candidate_count: Optional[int] = None


@dataclass
class LoadedDiscoverySeqRepr:
    repr_buf: torch.Tensor
    repr_mode: str
    repr_dim: int
    n_seqs: int
    n_stored: int
    is_capped: bool
    slot_to_id: Optional[torch.Tensor] = None
    id_to_slot: Optional[torch.Tensor] = None

    def get_repr(self, seq_ids: torch.Tensor) -> torch.Tensor:
        ids = seq_ids.long().cpu()
        if self.is_capped:
            if self.id_to_slot is None:
                raise ValueError("capped seq_repr missing id_to_slot")
            slots = self.id_to_slot[ids].long()
            return self.repr_buf[slots].float()
        return self.repr_buf[ids].float()


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
        "seq_repr": root / "seq_repr.pt",
        "logit_ctx": root / "logit_ctx.pt",
        "top_coactivation": root / "top_coactivation.pt",
    }
    if candidates_path is not None:
        paths["candidates"] = Path(candidates_path)
    return paths


def optional_discovery_artifact_paths(output_root: str | Path = "outputs") -> Dict[str, Path]:
    root = Path(output_root)
    return {
        "global_negctx_ids": root / GLOBAL_NEGCTX_IDS_FILENAME,
    }


def validate_discovery_artifacts(
    output_root: str | Path = "outputs",
    *,
    candidates_path: str | Path | None = None,
) -> DiscoveryArtifactValidation:
    """Validate discovery inputs before model/SAE initialization."""

    root = Path(output_root)
    paths = discovery_artifact_paths(root, candidates_path=candidates_path)
    _reject_missing(paths)
    optional_paths = optional_discovery_artifact_paths(root)
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

    seq_repr_n_seqs = _validate_seq_repr_payload(payloads["seq_repr"])
    max_ctx_seq_id = max(
        _max_positive_id(payloads["top_ctx"]["ctx_seq_idx"]),
        _max_positive_id(payloads["mid_ctx"]["ctx_seq_idx"]),
        _max_positive_id(payloads["neg_ctx"]["ctx_seq_idx"]),
    )
    if max_ctx_seq_id > seq_repr_n_seqs:
        raise ValueError(
            f"discovery context sequence id {max_ctx_seq_id} exceeds seq_repr n_seqs {seq_repr_n_seqs}"
        )

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

    global_negctx_ids_path = optional_paths["global_negctx_ids"]
    if global_negctx_ids_path.exists():
        global_negctx_ids = _validate_global_negctx_ids_payload(
            torch.load(global_negctx_ids_path, map_location="cpu", weights_only=False)
        )
        if global_negctx_ids.numel() > 0 and int(global_negctx_ids.max().item()) > seq_repr_n_seqs:
            raise ValueError(
                f"global_negctx_ids sequence id {int(global_negctx_ids.max().item())} "
                f"exceeds seq_repr n_seqs {seq_repr_n_seqs}"
            )

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
    global_negctx_ids_path = optional_discovery_artifact_paths(validation.run_root)["global_negctx_ids"]
    if global_negctx_ids_path.exists():
        setter = getattr(neg_ctx, "set_global_sequence_ids_cache", None)
        if callable(setter):
            setter(
                _global_negctx_ids_from_payload(
                    torch.load(global_negctx_ids_path, map_location="cpu", weights_only=False)
                )
            )
    seq_repr_store.seq_repr = _seq_repr_from_payload(torch.load(paths["seq_repr"], map_location="cpu", weights_only=False))
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
        details = [f"{name} ({paths[name].name})" for name in sorted(missing)]
        raise FileNotFoundError(f"missing discovery input artifacts: {details}")


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


def _validate_seq_repr_payload(payload: object) -> int:
    if not isinstance(payload, dict):
        raise ValueError("seq_repr artifact must contain a dict payload")
    repr_buf = payload.get("repr_buf")
    if not isinstance(repr_buf, torch.Tensor):
        raise ValueError("seq_repr missing required tensor repr_buf")
    repr_buf = cast(torch.Tensor, repr_buf)
    if repr_buf.ndim != 2:
        raise ValueError("seq_repr.repr_buf must be 2D")
    if not torch.isfinite(repr_buf.float()).all():
        raise ValueError("seq_repr.repr_buf must be finite")
    n_seqs = int(payload.get("n_seqs", 0))
    n_stored = int(payload.get("n_stored", n_seqs))
    repr_dim = int(payload.get("repr_dim", repr_buf.shape[1]))
    if n_seqs < 1:
        raise ValueError("seq_repr n_seqs must be positive")
    if n_stored < 1 or n_stored > n_seqs:
        raise ValueError("seq_repr n_stored must be in [1, n_seqs]")
    if tuple(repr_buf.shape) != (n_stored + 1, repr_dim):
        raise ValueError("seq_repr repr_buf shape does not match n_stored/repr_dim")
    if bool(payload.get("is_capped", False)):
        slot_to_id = payload.get("slot_to_id")
        id_to_slot = payload.get("id_to_slot")
        if not isinstance(slot_to_id, torch.Tensor) or not isinstance(id_to_slot, torch.Tensor):
            raise ValueError("capped seq_repr requires slot_to_id and id_to_slot")
        slot_to_id = cast(torch.Tensor, slot_to_id)
        id_to_slot = cast(torch.Tensor, id_to_slot)
        if tuple(slot_to_id.shape) != (n_stored + 1,):
            raise ValueError("seq_repr slot_to_id shape mismatch")
        if tuple(id_to_slot.shape) != (n_seqs + 1,):
            raise ValueError("seq_repr id_to_slot shape mismatch")
    return n_seqs


def _validate_global_negctx_ids_payload(payload: object) -> torch.Tensor:
    ids = torch.unique(_global_negctx_ids_from_payload(payload), sorted=True)
    if ids.ndim != 1:
        raise ValueError("global_negctx_ids must be 1D")
    if ids.numel() > 0 and int(ids.min().item()) <= 0:
        raise ValueError("global_negctx_ids must contain positive sequence IDs")
    return ids


def _global_negctx_ids_from_payload(payload: object) -> torch.Tensor:
    if isinstance(payload, dict):
        payload = payload.get("global_negctx_ids")
    if not isinstance(payload, torch.Tensor):
        raise ValueError("global_negctx_ids artifact must contain a tensor")
    tensor = cast(torch.Tensor, payload)
    return tensor.detach().cpu().to(torch.int64).reshape(-1)


def _seq_repr_from_payload(payload: object) -> LoadedDiscoverySeqRepr:
    _validate_seq_repr_payload(payload)
    assert isinstance(payload, dict)
    repr_buf = payload["repr_buf"]
    assert isinstance(repr_buf, torch.Tensor)
    repr_buf = cast(torch.Tensor, repr_buf)
    is_capped = bool(payload.get("is_capped", False))
    slot_to_id = payload.get("slot_to_id")
    id_to_slot = payload.get("id_to_slot")
    slot_to_id_tensor = cast(torch.Tensor, slot_to_id) if isinstance(slot_to_id, torch.Tensor) else None
    id_to_slot_tensor = cast(torch.Tensor, id_to_slot) if isinstance(id_to_slot, torch.Tensor) else None
    return LoadedDiscoverySeqRepr(
        repr_buf=repr_buf.cpu(),
        repr_mode=str(payload.get("repr_mode", "mean_pool")),
        repr_dim=int(payload.get("repr_dim", repr_buf.shape[1])),
        n_seqs=int(payload["n_seqs"]),
        n_stored=int(payload.get("n_stored", int(payload["n_seqs"]))),
        is_capped=is_capped,
        slot_to_id=slot_to_id_tensor.cpu().to(torch.int64) if slot_to_id_tensor is not None else None,
        id_to_slot=id_to_slot_tensor.cpu().to(torch.int32) if id_to_slot_tensor is not None else None,
    )


def _max_positive_id(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    positive = tensor[tensor > 0]
    if positive.numel() == 0:
        return 0
    return int(positive.max().item())
