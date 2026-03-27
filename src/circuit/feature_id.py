from dataclasses import dataclass
from typing import Tuple, List, Sequence, Optional


@dataclass(frozen=True)
class FeatureID:
    """
    Unified identifier for an SAE feature (latent).
    
    Attributes:
        layer: The layer index (0-indexed).
        kind:  The component kind ("attn", "mlp", "resid").
        index: The latent index within the SAE (0 to d_sae-1).
    """
    layer: int
    kind: str
    index: int

    def __repr__(self) -> str:
        return f"L{self.layer}.{self.kind}.f{self.index}"

    @property
    def key(self) -> Tuple[int, str, int]:
        """Returns the canonical tuple representation (layer, kind, index)."""
        return (self.layer, self.kind, self.index)

    @classmethod
    def from_global_id(cls, global_id: int, n_kinds: int, d_sae: int, kinds: Sequence[str]) -> "FeatureID":
        """
        Creates a FeatureID from a flat 'global' ID (comp_idx * d_sae + latent_idx).
        
        Args:
            global_id: The flat integer ID.
            n_kinds:   Number of SAE kinds per layer (typically 3).
            d_sae:     The dimension of the SAE (typically 40960).
            kinds:     The list of kind names in order (typically ["attn", "mlp", "resid"]).
        """
        comp_idx, latent_idx = divmod(global_id, d_sae)
        layer, kind_idx = divmod(comp_idx, n_kinds)
        return cls(layer=layer, kind=kinds[kind_idx], index=latent_idx)

    def to_global_id(self, n_kinds: int, d_sae: int, kinds: Sequence[str]) -> int:
        """
        Converts the FeatureID back to a flat 'global' ID.
        """
        kind_idx = kinds.index(self.kind)
        comp_idx = self.layer * n_kinds + kind_idx
        return comp_idx * d_sae + self.index

    @classmethod
    def from_component_id(cls, comp_idx: int, latent_idx: int, n_kinds: int, kinds: Sequence[str]) -> "FeatureID":
        """
        Creates a FeatureID from a component index and a latent index.
        """
        layer, kind_idx = divmod(comp_idx, n_kinds)
        return cls(layer=layer, kind=kinds[kind_idx], index=latent_idx)

    def to_component_id(self, n_kinds: int, kinds: Sequence[str]) -> Tuple[int, int]:
        """
        Returns (comp_idx, latent_idx).
        """
        kind_idx = kinds.index(self.kind)
        comp_idx = self.layer * n_kinds + kind_idx
        return comp_idx, self.index
