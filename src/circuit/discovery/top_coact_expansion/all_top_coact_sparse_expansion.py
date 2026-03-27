import torch
from typing import Optional, Any, List

from .top_coact_sparse_expansion import TopCoactSparseExpansion
from config import config
from store.top_coactivation import top_coactivation
from store.latent_stats import latent_stats


class AllTopCoactSparseExpansion(TopCoactSparseExpansion):
    """
    All-kinds variable-depth top-coactivation sparse expansion with no passthrough stage.
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
            target_kinds=("attn", "mlp", "resid"),
            passthrough_kinds=(),
            method_name="all_top_coact_sparse_expansion",
            config_node=config.discovery.all_top_coact_sparse_expansion,
            coact_depth=coact_depth,
            min_faithfulness=min_faithfulness,
            min_active_count=min_active_count,
            pruning_threshold=pruning_threshold,
            probe_batch_size=probe_batch_size,
        )
        self.top_coactivation = top_coactivation
        self.latent_stats = latent_stats
