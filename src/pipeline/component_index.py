from typing import Sequence, Tuple


def component_idx(layer_idx: int, kind_idx: int, n_kinds: int) -> int:
    """Map (layer, kind) to a flat component index."""
    return layer_idx * n_kinds + kind_idx


def split_component_idx(component_idx_value: int, n_kinds: int) -> Tuple[int, int]:
    """Inverse of component_idx -> (layer_idx, kind_idx)."""
    return divmod(component_idx_value, n_kinds)


def layer_component_bounds(layer_idx: int, n_kinds: int) -> Tuple[int, int]:
    """Return inclusive/exclusive component index bounds for one layer."""
    start = component_idx(layer_idx, 0, n_kinds)
    return start, start + n_kinds


def kind_name_for_component(component_idx_value: int, kinds: Sequence[str]) -> str:
    """Return kind name for a flat component index."""
    _, kind_idx = split_component_idx(component_idx_value, len(kinds))
    return kinds[kind_idx]
