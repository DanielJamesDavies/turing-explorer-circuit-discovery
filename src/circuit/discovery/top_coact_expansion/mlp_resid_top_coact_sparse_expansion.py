import torch
from typing import Optional, Any, List

from .top_coact_sparse_expansion import TopCoactSparseExpansion
from config import config
from store.top_coactivation import top_coactivation
from store.latent_stats import latent_stats


class MlpResidTopCoactSparseExpansion(TopCoactSparseExpansion):
    """
    MLP+resid-targeted variable-depth top-coactivation sparse expansion with attn passthrough.
    """

    def __init__(
        self,
        inference: Any,
        sae_bank: Any,
        avg_acts: torch.Tensor,
        probe_builder: Any,
        coact_depth: Optional[List[int]] = None,
        min_faithfulness: Optional[float] = None,
        min_active_count: Optional[int] = None,
        pruning_threshold: Optional[float] = None,
        probe_batch_size: Optional[int] = None,
    ):
        super().__init__(
            inference=inference,
            sae_bank=sae_bank,
            avg_acts=avg_acts,
            probe_builder=probe_builder,
            target_kinds=("mlp", "resid"),
            passthrough_kinds=("attn",),
            method_name="mlp_resid_top_coact_sparse_expansion",
            config_node=config.discovery.mlp_resid_top_coact_sparse_expansion,
            coact_depth=coact_depth,
            min_faithfulness=min_faithfulness,
            min_active_count=min_active_count,
            pruning_threshold=pruning_threshold,
            probe_batch_size=probe_batch_size,
        )
        self.top_coactivation = top_coactivation
        self.latent_stats = latent_stats
