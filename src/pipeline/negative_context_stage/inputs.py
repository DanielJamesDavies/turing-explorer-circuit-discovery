"""Input contracts and artifact loading for the negative-context stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

import torch

from pipeline.distributed.interfaces import PipelineOutputPaths, build_output_paths
from store.context import neg_ctx
from store.neg_context import NegCtxStats


class SeqReprLike(Protocol):
    repr_buf: torch.Tensor
    repr_mode: str
    repr_dim: int
    n_seqs: int
    n_stored: int
    is_capped: bool
    slot_to_id: Optional[torch.Tensor]
    id_to_slot: Optional[torch.Tensor]


@dataclass
class LoadedContext:
    ctx_type: str
    ctx_seq_idx: torch.Tensor
    ctx_seq_val: torch.Tensor
    num_components: int
    d_sae: int
    num_ctx_sequences: int
    mode: Optional[str] = None
    reservoir_fill: Optional[torch.Tensor] = None
    reservoir_n: Optional[torch.Tensor] = None

    def save(self, path: str | Path) -> None:
        checkpoint: Dict[str, object] = {
            "ctx_seq_idx": self.ctx_seq_idx,
            "ctx_seq_val": self.ctx_seq_val,
            "ctx_type": self.ctx_type,
        }
        if self.ctx_type == "mid":
            if self.mode is not None:
                checkpoint["mode"] = self.mode
            checkpoint["num_ctx_sequences"] = self.num_ctx_sequences
            if self.reservoir_fill is not None:
                checkpoint["reservoir_fill"] = self.reservoir_fill
            if self.reservoir_n is not None:
                checkpoint["reservoir_n"] = self.reservoir_n
        torch.save(checkpoint, path)


@dataclass
class LoadedSeqRepr:
    repr_buf: torch.Tensor
    repr_mode: str
    repr_dim: int
    n_seqs: int
    n_stored: int
    is_capped: bool
    slot_to_id: Optional[torch.Tensor] = None
    id_to_slot: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class NegativeContextInputs:
    top_ctx: LoadedContext
    mid_ctx: LoadedContext
    seq_repr: SeqReprLike
    paths: PipelineOutputPaths


BuildNegCtxFn = Callable[
    [SeqReprLike, LoadedContext, LoadedContext, LoadedContext],
    NegCtxStats,
]


def load_negative_context_inputs(
    output_root: str | Path = "outputs",
    *,
    expected_config_hash: Optional[str] = None,
) -> NegativeContextInputs:
    """Load and validate merged pass-1 artifacts for a standalone neg_ctx stage."""

    paths = build_output_paths(output_root)
    _require_artifacts(
        {
            "top_ctx": paths.top_ctx,
            "mid_ctx": paths.mid_ctx,
            "seq_repr": paths.seq_repr,
        }
    )
    top_payload = _load_torch_payload(paths.top_ctx, expected_config_hash=expected_config_hash)
    mid_payload = _load_torch_payload(paths.mid_ctx, expected_config_hash=expected_config_hash)
    seq_payload = _load_torch_payload(paths.seq_repr, expected_config_hash=expected_config_hash)

    loaded_top_ctx = _context_from_payload(top_payload, expected_ctx_type="top")
    loaded_mid_ctx = _context_from_payload(mid_payload, expected_ctx_type="mid")
    loaded_seq_repr = _seq_repr_from_payload(seq_payload)
    _validate_negative_context_inputs(loaded_top_ctx, loaded_mid_ctx, loaded_seq_repr)
    return NegativeContextInputs(
        top_ctx=loaded_top_ctx,
        mid_ctx=loaded_mid_ctx,
        seq_repr=loaded_seq_repr,
        paths=paths,
    )


def _require_artifacts(paths: Dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        details = ", ".join(f"{name} ({paths[name]})" for name in missing)
        raise FileNotFoundError(f"missing required pass-1 artifact(s): {details}")


def _load_torch_payload(
    path: Path,
    *,
    expected_config_hash: Optional[str],
) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact payload must be a dict: {path}")
    _validate_config_hash_if_present(payload, path, expected_config_hash)
    return payload


def _validate_config_hash_if_present(
    payload: Dict[str, object],
    path: Path,
    expected_config_hash: Optional[str],
) -> None:
    if expected_config_hash is None:
        return
    metadata = payload.get("metadata")
    observed = payload.get("config_hash")
    if observed is None and isinstance(metadata, dict):
        observed = metadata.get("config_hash")
    if observed is None:
        raise ValueError(f"artifact config hash missing for {path}")
    if observed is not None and str(observed) != expected_config_hash:
        raise ValueError(f"artifact config hash mismatch for {path}")


def _context_from_payload(
    payload: Dict[str, object],
    *,
    expected_ctx_type: str,
) -> LoadedContext:
    ctx_seq_idx = payload.get("ctx_seq_idx")
    ctx_seq_val = payload.get("ctx_seq_val")
    if not isinstance(ctx_seq_idx, torch.Tensor) or not isinstance(ctx_seq_val, torch.Tensor):
        raise ValueError(f"{expected_ctx_type}_ctx artifact must contain context tensors")
    if ctx_seq_idx.ndim != 3 or ctx_seq_val.ndim != 3:
        raise ValueError(f"{expected_ctx_type}_ctx tensors must be rank-3")
    if ctx_seq_idx.shape != ctx_seq_val.shape:
        raise ValueError(f"{expected_ctx_type}_ctx tensor shape mismatch")
    if payload.get("ctx_type") != expected_ctx_type:
        raise ValueError(f"expected ctx_type={expected_ctx_type!r}")
    if not torch.is_floating_point(ctx_seq_val):
        raise ValueError(f"{expected_ctx_type}_ctx values must be floating point")
    if not torch.isfinite(ctx_seq_val.float()).all():
        raise ValueError(f"{expected_ctx_type}_ctx values contain non-finite entries")
    if (ctx_seq_idx < 0).any():
        raise ValueError(f"{expected_ctx_type}_ctx sequence IDs must be non-negative")
    return LoadedContext(
        ctx_type=expected_ctx_type,
        ctx_seq_idx=ctx_seq_idx.to(torch.int32).cpu(),
        ctx_seq_val=ctx_seq_val.cpu(),
        num_components=int(ctx_seq_idx.shape[0]),
        d_sae=int(ctx_seq_idx.shape[1]),
        num_ctx_sequences=int(ctx_seq_idx.shape[2]),
        mode=str(payload["mode"]) if "mode" in payload else None,
        reservoir_fill=payload.get("reservoir_fill")
        if isinstance(payload.get("reservoir_fill"), torch.Tensor)
        else None,
        reservoir_n=payload.get("reservoir_n")
        if isinstance(payload.get("reservoir_n"), torch.Tensor)
        else None,
    )


def _seq_repr_from_payload(payload: Dict[str, object]) -> LoadedSeqRepr:
    repr_buf = payload.get("repr_buf")
    if not isinstance(repr_buf, torch.Tensor):
        raise ValueError("seq_repr artifact must contain repr_buf")
    if repr_buf.ndim != 2:
        raise ValueError("seq_repr repr_buf must be rank-2")
    if not torch.isfinite(repr_buf.float()).all():
        raise ValueError("seq_repr repr_buf contains non-finite entries")
    n_seqs = int(payload.get("n_seqs", 0))
    n_stored = int(payload.get("n_stored", n_seqs))
    repr_dim = int(payload.get("repr_dim", repr_buf.shape[1]))
    is_capped = bool(payload.get("is_capped", False))
    if n_seqs < 1:
        raise ValueError("seq_repr n_seqs must be positive")
    if n_stored < 1 or n_stored > n_seqs:
        raise ValueError("seq_repr n_stored must be in [1, n_seqs]")
    if repr_buf.shape != (n_stored + 1, repr_dim):
        raise ValueError("seq_repr repr_buf shape does not match n_stored/repr_dim")

    slot_to_id = payload.get("slot_to_id")
    id_to_slot = payload.get("id_to_slot")
    if is_capped:
        if not isinstance(slot_to_id, torch.Tensor) or not isinstance(id_to_slot, torch.Tensor):
            raise ValueError("capped seq_repr requires slot_to_id and id_to_slot")
        _validate_seq_repr_cap_mapping(slot_to_id, id_to_slot, n_seqs, n_stored)
        loaded_slot_to_id = slot_to_id.to(torch.int64).cpu()
        loaded_id_to_slot = id_to_slot.to(torch.int32).cpu()
    else:
        loaded_slot_to_id = None
        loaded_id_to_slot = None

    return LoadedSeqRepr(
        repr_buf=repr_buf.cpu(),
        repr_mode=str(payload.get("repr_mode", "mean_pool")),
        repr_dim=repr_dim,
        n_seqs=n_seqs,
        n_stored=n_stored,
        is_capped=is_capped,
        slot_to_id=loaded_slot_to_id,
        id_to_slot=loaded_id_to_slot,
    )


def _validate_seq_repr_cap_mapping(
    slot_to_id: torch.Tensor,
    id_to_slot: torch.Tensor,
    n_seqs: int,
    n_stored: int,
) -> None:
    if slot_to_id.shape != (n_stored + 1,):
        raise ValueError("seq_repr slot_to_id shape mismatch")
    if id_to_slot.shape != (n_seqs + 1,):
        raise ValueError("seq_repr id_to_slot shape mismatch")
    slot_to_id_i64 = slot_to_id.to(torch.int64)
    id_to_slot_i64 = id_to_slot.to(torch.int64)
    if int(slot_to_id_i64[0].item()) != 0 or int(id_to_slot_i64[0].item()) != 0:
        raise ValueError("seq_repr cap mappings must keep sentinel zero")
    selected = slot_to_id_i64[1:]
    if ((selected < 1) | (selected > n_seqs)).any():
        raise ValueError("seq_repr slot_to_id contains out-of-range sequence IDs")
    expected_slots = torch.arange(1, n_stored + 1, dtype=torch.int64)
    if not torch.equal(id_to_slot_i64[selected], expected_slots):
        raise ValueError("seq_repr cap mappings are inconsistent")


def _validate_negative_context_inputs(
    loaded_top_ctx: LoadedContext,
    loaded_mid_ctx: LoadedContext,
    loaded_seq_repr: SeqReprLike,
) -> None:
    if loaded_top_ctx.num_components != loaded_mid_ctx.num_components:
        raise ValueError("top_ctx and mid_ctx component counts differ")
    if loaded_top_ctx.d_sae != loaded_mid_ctx.d_sae:
        raise ValueError("top_ctx and mid_ctx SAE widths differ")
    if loaded_top_ctx.num_ctx_sequences < 1 or loaded_mid_ctx.num_ctx_sequences < 1:
        raise ValueError("context artifacts must have at least one context slot")
    max_sequence_id = max(
        int(loaded_top_ctx.ctx_seq_idx.max().item()),
        int(loaded_mid_ctx.ctx_seq_idx.max().item()),
    )
    if max_sequence_id > loaded_seq_repr.n_seqs:
        raise ValueError("context sequence ID exceeds seq_repr n_seqs")


def _empty_neg_context_like(loaded_top_ctx: LoadedContext) -> LoadedContext:
    n_sequences = int(neg_ctx.num_ctx_sequences)
    return LoadedContext(
        ctx_type="neg",
        ctx_seq_idx=torch.zeros(
            (loaded_top_ctx.num_components, loaded_top_ctx.d_sae, n_sequences),
            dtype=torch.int32,
        ),
        ctx_seq_val=torch.zeros(
            (loaded_top_ctx.num_components, loaded_top_ctx.d_sae, n_sequences),
            dtype=torch.float32,
        ),
        num_components=loaded_top_ctx.num_components,
        d_sae=loaded_top_ctx.d_sae,
        num_ctx_sequences=n_sequences,
    )


__all__ = [
    "BuildNegCtxFn",
    "LoadedContext",
    "LoadedSeqRepr",
    "NegativeContextInputs",
    "SeqReprLike",
    "load_negative_context_inputs",
]
