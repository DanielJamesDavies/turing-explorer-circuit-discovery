"""Config loading and CLI default helpers for distributed controller planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def load_and_hash_config(config_path: str | Path) -> tuple[Dict[str, object], str]:
    """Strictly load config data and return normalized data plus SHA-256 hash."""

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _validate_config_strict(raw)
    normalized = _normalize_for_hash(raw)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return normalized, hashlib.sha256(encoded).hexdigest()


def _validate_config_strict(data: Dict[str, object]) -> None:
    from config import RootConfig

    RootConfig.model_validate(data)


def _normalize_for_hash(value):
    if isinstance(value, dict):
        return {key: _normalize_for_hash(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _root_config_dump(normalized_config: Dict[str, object]) -> Dict[str, object]:
    from config import RootConfig

    return RootConfig.model_validate(normalized_config).model_dump()


def _resolve_config_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _candidate_dump_m_from_config(normalized_config: Dict[str, object]) -> int:
    latents = normalized_config.get("latents", {})
    top_coactivation = (
        latents.get("top_coactivation", {})
        if isinstance(latents, dict)
        else {}
    )
    n_latents_per_latent = int(
        top_coactivation.get("n_latents_per_latent", 64)
        if isinstance(top_coactivation, dict)
        else 64
    )
    n_candidates_per_component = int(
        top_coactivation.get("n_candidates_per_component", 16)
        if isinstance(top_coactivation, dict)
        else 16
    )
    candidate_oversample_factor = int(
        top_coactivation.get("candidate_oversample_factor", 4)
        if isinstance(top_coactivation, dict)
        else 4
    )
    # The model config currently defaults to 12 layers with three SAE components per layer.
    default_num_components = 36
    return min(
        n_latents_per_latent * candidate_oversample_factor,
        default_num_components * n_candidates_per_component,
    )


def _distributed_cli_defaults(config_path: str | Path) -> Dict[str, object]:
    normalized_config, _config_hash = load_and_hash_config(config_path)
    return dict(_root_config_dump(normalized_config)["distributed"])


def _parse_physical_ids(
    raw_devices: Optional[str],
    config_devices: object,
) -> Optional[List[int]]:
    devices = (
        [part.strip() for part in raw_devices.split(",") if part.strip()]
        if raw_devices is not None
        else list(config_devices) if isinstance(config_devices, list) else []
    )
    if not devices:
        return None
    parsed: List[int] = []
    for device in devices:
        if isinstance(device, int):
            parsed.append(device)
            continue
        text = str(device)
        if text.startswith("cuda:"):
            text = text.split(":", 1)[1]
        parsed.append(int(text))
    return parsed


__all__ = [
    "load_and_hash_config",
]
