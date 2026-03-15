from __future__ import annotations
import uuid
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import torch

@dataclass
class CircuitNode:
    """
    Represents a single node in a discovered circuit (e.g. a latent or attention head).
    """
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Common metadata accessors
    @property
    def weight(self) -> Optional[float]:
        return self.metadata.get("weight")
    
    @property
    def source(self) -> Optional[str]:
        return self.metadata.get("source")

@dataclass
class CircuitEdge:
    """
    Represents a causal or attribution edge between two nodes.
    """
    source_uuid: str
    target_uuid: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def weight(self) -> Optional[float]:
        return self.metadata.get("weight")

@dataclass
class Circuit:
    """
    A collection of nodes and edges representing a single discovered mechanism.
    """
    name: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    nodes: Dict[str, CircuitNode] = field(default_factory=dict)
    edges: List[CircuitEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: CircuitNode) -> CircuitNode:
        """Adds a node to the circuit and returns it."""
        self.nodes[node.uuid] = node
        return node

    def add_edge(self, source_uuid: str, target_uuid: str, **metadata) -> CircuitEdge:
        """
        Adds an edge between two existing nodes.
        Note: We don't strictly enforce node existence here to allow for
        out-of-order construction if needed.
        """
        edge = CircuitEdge(source_uuid=source_uuid, target_uuid=target_uuid, metadata=metadata)
        self.edges.append(edge)
        return edge

class CircuitStore:
    """
    A central store for managing multiple discovered circuits.
    """
    def __init__(self):
        self.circuits: Dict[str, Circuit] = {}

    def add_circuit(self, circuit: Circuit) -> Circuit:
        """Adds a circuit to the store and returns it."""
        self.circuits[circuit.uuid] = circuit
        return circuit

    def get_circuit(self, circuit_uuid: str) -> Optional[Circuit]:
        """Retrieves a circuit by its UUID."""
        return self.circuits.get(circuit_uuid)

    def save(self, path: str):
        """
        Persists the entire store using torch.save.
        This handles any tensors stored in metadata efficiently.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.circuits, path)

    def load(self, path: str):
        """Loads circuits from a saved file."""
        if os.path.exists(path):
            # PyTorch 2.6+ defaults to weights_only=True, which blocks custom classes.
            # We allowlist our classes or set weights_only=False.
            try:
                self.circuits = torch.load(path, weights_only=False)
            except TypeError:
                # Fallback for older torch versions that don't have weights_only
                self.circuits = torch.load(path)
        else:
            print(f"[circuit_store] Warning: save file not found at {path}")

# Singleton instance for project-wide use
circuit_store = CircuitStore()
