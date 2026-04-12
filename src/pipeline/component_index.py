from typing import Sequence, Tuple, List, Optional


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


def get_predecessor_components(
    comp_idx: int, n_kinds: int, kinds: Sequence[str]
) -> List[int]:
    """
    Returns the list of component indices that causally precede the given component
    based on the transformer's residual arithmetic.

    Predecessor Logic (assuming kinds = ["attn", "mlp", "resid"]):
    - attn  at layer L: resid at layer L-1
    - mlp   at layer L: resid at layer L-1, attn at layer L
    - resid at layer L: resid at layer L-1, attn at layer L, mlp at layer L

    Args:
        comp_idx: Flat index of the target component.
        n_kinds:  Number of kinds (usually 3).
        kinds:    Sequence of kind names (usually ["attn", "mlp", "resid"]).

    Returns:
        List of predecessor component indices.
    """
    layer, kind_idx = split_component_idx(comp_idx, n_kinds)
    kind = kinds[kind_idx]
    predecessors = []

    # Map kind names to their indices for easier lookup
    kind_to_idx = {name: i for i, name in enumerate(kinds)}
    resid_idx = kind_to_idx.get("resid")
    attn_idx = kind_to_idx.get("attn")
    mlp_idx = kind_to_idx.get("mlp")

    # All components depend on the residual stream from the previous layer
    if layer > 0 and resid_idx is not None:
        predecessors.append(component_idx(layer - 1, resid_idx, n_kinds))

    # MLP and resid also depend on components within the same layer
    if kind == "mlp":
        if attn_idx is not None:
            predecessors.append(component_idx(layer, attn_idx, n_kinds))
    elif kind == "resid":
        if attn_idx is not None:
            predecessors.append(component_idx(layer, attn_idx, n_kinds))
        if mlp_idx is not None:
            predecessors.append(component_idx(layer, mlp_idx, n_kinds))

    return predecessors


def get_all_upstream_components(
    comp_idx: int,
    n_kinds: int,
    kinds: Sequence[str],
    min_layer: int = 0,
    include_same_layer: bool = True,
) -> List[int]:
    """
    Returns every component index that lies upstream of the given component.

    Unlike ``get_predecessor_components``, which follows the transformer's causal
    wiring (only the directly-adjacent layer), this function returns *all*
    (layer', kind') pairs for which layer' < layer, plus — when
    ``include_same_layer`` is True — the within-layer causal predecessors
    (e.g. attn before mlp at the same layer).

    Args:
        comp_idx:           Flat component index of the target.
        n_kinds:            Number of SAE kinds per layer (typically 3).
        kinds:              Sequence of kind names (e.g. ["attn", "mlp", "resid"]).
        min_layer:          Inclusive lower bound on predecessor layers.
                            Useful for limiting scope (e.g. ``max_layers_back``).
        include_same_layer: If True, also include within-layer causal predecessors
                            via ``get_predecessor_components`` (e.g. attn → mlp).

    Returns:
        Deduplicated list of predecessor component indices, ordered by layer
        (ascending) then kind index.
    """
    layer, _ = split_component_idx(comp_idx, n_kinds)

    upstream: List[int] = []

    # All (layer', kind') pairs for strictly preceding layers.
    for layer_p in range(min_layer, layer):
        for kind_idx_p in range(n_kinds):
            upstream.append(component_idx(layer_p, kind_idx_p, n_kinds))

    # Within-layer causal predecessors (e.g. attn@L before mlp@L).
    if include_same_layer:
        same_layer = get_predecessor_components(comp_idx, n_kinds, kinds)
        for c in same_layer:
            layer_c, _ = split_component_idx(c, n_kinds)
            if layer_c == layer and c not in upstream:
                upstream.append(c)

    return upstream
