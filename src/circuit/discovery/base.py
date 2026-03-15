import torch
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from store.circuits import Circuit
from circuit.probe_dataset import ProbeDatasetBuilder, ProbeDataset
from sae.bank import SAEBank

class DiscoveryMethod(ABC):
    """
    Abstract base class for all circuit discovery algorithms.
    Provides a common interface and access to necessary model/data resources.
    """
    def __init__(
        self, 
        inference: Any, 
        sae_bank: SAEBank, 
        avg_acts: torch.Tensor,
        probe_builder: ProbeDatasetBuilder
    ):
        """
        Args:
            inference: The Inference instance for running the LLM.
            sae_bank: The SAEBank containing the encoders and decoders.
            avg_acts: Tensor of average activations [n_layers, d_sae].
            probe_builder: Utility to build positive/negative datasets for seeds.
        """
        self.inference = inference
        self.sae_bank = sae_bank
        self.avg_acts = avg_acts
        self.probe_builder = probe_builder

    @abstractmethod
    def discover(self, seed_comp_idx: int, seed_latent_idx: int) -> Optional[Circuit]:
        """
        Executes the discovery algorithm starting from a seed latent.
        
        Args:
            seed_comp_idx: The component index of the seed.
            seed_latent_idx: The latent index of the seed.
            
        Returns:
            A Circuit object if a faithful mechanism was found, else None.
        """
        pass

    def build_probe_dataset(
        self, 
        comp_idx: int, 
        latent_idx: int,
        n_pos: int = 64,
        n_neg: int = 64
    ) -> ProbeDataset:
        """Helper to build a probe dataset for a latent using the injected builder."""
        from store.context import top_ctx, mid_ctx, neg_ctx
        return self.probe_builder.build_for_latent(
            comp_idx, latent_idx, top_ctx, mid_ctx, neg_ctx, 
            n_pos=n_pos, n_neg=n_neg
        )
