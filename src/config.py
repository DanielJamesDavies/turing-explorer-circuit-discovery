# A file to load the config from the config.yaml file with strict validation

import yaml
import os
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Any, Union

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == "src":
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

def _resolve_path(val: Any) -> Any:
    """Helper to resolve relative paths in the config relative to PROJECT_ROOT."""
    if isinstance(val, str) and (val.startswith("./") or "/" in val or "\\" in val) and not os.path.isabs(val):
        # Check if it looks like a path and exists relative to PROJECT_ROOT
        full_path = os.path.normpath(os.path.join(PROJECT_ROOT, val))
        return full_path
    return val

class WeightsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', protected_namespaces=())
    model_path: str
    sae_path: str

    @field_validator("model_path", "sae_path", mode="after")
    @classmethod
    def resolve_paths(cls, v: str) -> str:
        return _resolve_path(v)

class DataConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset_path: str
    n_shards: int = 256
    batch_size: int = 512

    @field_validator("dataset_path", mode="after")
    @classmethod
    def resolve_paths(cls, v: str) -> str:
        return _resolve_path(v)

class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    multi_gpu: bool = False
    memory: str = "efficient"
    compile: bool = True
    parallel_kinds: bool = False
    ann_device: str = "auto"

class TopCtxConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_sequences: int = 64

class MidCtxConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_sequences: int = 64
    band_low_sigma: float = 0.5
    band_high_sigma: float = 1.5
    warmup_batches: int = 100

class NegCtxConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_sequences: int = 64
    n_neighbors: int = 512
    min_pos_ctx: int = 8
    repr_mode: str = "mean_pool"
    max_repr_seqs: Optional[int] = 200000

class LogitCtxConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_tokens_per_latent: int = 32
    topk_output_tokens: int = 32

class TopCoactivationLatentsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_latents_per_latent: int = 64
    n_candidates_per_component: int = 16
    freq_alpha: float = 2.0
    mode: str = "freq_weighted"  # "freq_weighted" | "raw" | "pmi"
    pmi_clamp_min: float = -5.0
    pmi_clamp_max: float = 10.0

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = ["freq_weighted", "raw", "pmi"]
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v}")
        return v

class LatentsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    top_ctx: TopCtxConfig = Field(default_factory=TopCtxConfig)
    mid_ctx: MidCtxConfig = Field(default_factory=MidCtxConfig)
    neg_ctx: NegCtxConfig = Field(default_factory=NegCtxConfig)
    logit_ctx: LogitCtxConfig = Field(default_factory=LogitCtxConfig)
    top_coactivation: TopCoactivationLatentsConfig = Field(default_factory=TopCoactivationLatentsConfig)

class CoactivationStatisticalConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    coactivation_threshold: float = 0.1
    pmi_coactivation_threshold: float = 1.0
    max_neighbors: int = 32
    pruning_threshold: float = 0.0

class LogitAttributionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    logit_threshold: float = 0.001
    edge_threshold: float = 0.00001
    max_neighbors: int = 32
    pruning_threshold: float = 0.0

class SFCAttributionPatchingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    node_threshold: float = 0.1
    edge_threshold: float = 0.01
    patch_mode: str = "mean_neg"
    max_neg: int = 8
    pruning_threshold: float = 0.0
    ig_steps: int = 10

class NeighborhoodExpansionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_expand: int = 16
    m_neighbors: int = 16
    pruning_threshold: float = 0.0

class SparseExpansionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    coact_depth: List[int] = Field(default_factory=lambda: [32, 16])
    pruning_threshold: float = 0.0

class HardNegativeCoactSparseExpansionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    coact_depth: List[int] = Field(default_factory=lambda: [32, 16])
    neg_candidate_limit: int = 32
    attribution_threshold: float = 0.01
    pruning_threshold: float = 0.0

class DifferentialActivationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_activator_candidates: int = 64
    n_inhibitor_candidates: int = 32
    attribution_threshold: float = 0.01
    pruning_threshold: float = 0.0

class GradientUpstreamConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    depth: int = 3                    # number of backward hops from the seed
    top_k_per_hop: int = 8            # top-K upstream latents to select per node per hop
    attribution_threshold: float = 0.01  # min |score| to include a latent
    min_active_count: int = 1         # skip latents below this global firing count
    max_ctx_sequences: int = 4        # total ctx sequences to use per node across all microbatches
    hop_batch_size: int = 4           # sequences per microbatch in _run_hop (gradient accumulation)
    absent_inhibitor_top_k: int = 4   # top-K absent inhibitors to find per hop (0 = disabled)
    absent_inhibitor_threshold: float = 0.01  # min |raw gradient| to flag an absent inhibitor
    pruning_threshold: float = 0.0    # faithfulness drop threshold for minimality pruning
    min_faithfulness: float = 0.2     # minimum faithfulness score for the circuit to be accepted

class LayerwiseGradientUpstreamConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    top_k_per_node: int = 8               # top-K upstream latents per node per pass
    attribution_threshold: float = 0.01   # min |score| to include a latent
    min_active_count: int = 1             # skip latents below this global firing count
    max_ctx_sequences: int = 4            # ctx sequences per node (across microbatches)
    hop_batch_size: int = 4               # sequences per microbatch in _run_node
    absent_inhibitor_top_k: int = 4       # top-K absent inhibitors per node (0 = disabled)
    absent_inhibitor_threshold: float = 0.01
    max_layers_back: int = 0              # 0 = go back to layer 0; >0 = limit depth
    include_same_layer: bool = True       # also include within-layer causal predecessors
    pruning_threshold: float = 0.0        # faithfulness drop threshold for minimality pruning
    min_faithfulness: float = 0.2         # minimum faithfulness score for circuit acceptance
    profile_first_node: bool = False      # run torch.profiler on the first node and exit

class TopCoactAttrDiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    attribution_threshold: float = 0.01
    max_neighbors: int = 32
    max_hops: int = 2
    pruning_threshold: float = 0.01

class CounterfactualGradientConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    top_k_activators: int = 8
    top_k_inhibitors: int = 8
    activator_threshold: float = 0.01
    inhibitor_threshold: float = 0.01
    min_active_count: int = 1
    max_neg_sequences: int = 4
    pruning_threshold: float = 0.0
    min_faithfulness: float = 0.2

class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_seeds: int = 128
    probe_batch_size: int = 16
    neg_ctx_eval_max: int = 16
    min_faithfulness: float = 0.2
    min_active_count: int = 1
    max_neighbors: int = 32
    methods: List[str] = Field(default_factory=lambda: ["coactivation_statistical", "logit_attribution"])
    
    coactivation_statistical: CoactivationStatisticalConfig = Field(default_factory=CoactivationStatisticalConfig)
    logit_attribution: LogitAttributionConfig = Field(default_factory=LogitAttributionConfig)
    sfc_attribution_patching: SFCAttributionPatchingConfig = Field(default_factory=SFCAttributionPatchingConfig)
    neighborhood_expansion: NeighborhoodExpansionConfig = Field(default_factory=NeighborhoodExpansionConfig)
    
    attn_top_coact_sparse_expansion: SparseExpansionConfig = Field(default_factory=SparseExpansionConfig)
    mlp_top_coact_sparse_expansion: SparseExpansionConfig = Field(default_factory=SparseExpansionConfig)
    resid_top_coact_sparse_expansion: SparseExpansionConfig = Field(default_factory=SparseExpansionConfig)
    attn_mlp_top_coact_sparse_expansion: SparseExpansionConfig = Field(default_factory=SparseExpansionConfig)
    attn_resid_top_coact_sparse_expansion: SparseExpansionConfig = Field(default_factory=SparseExpansionConfig)
    mlp_resid_top_coact_sparse_expansion: SparseExpansionConfig = Field(default_factory=SparseExpansionConfig)
    all_top_coact_sparse_expansion: SparseExpansionConfig = Field(default_factory=SparseExpansionConfig)
    hard_negative_coact_sparse_expansion: HardNegativeCoactSparseExpansionConfig = Field(default_factory=HardNegativeCoactSparseExpansionConfig)
    differential_activation: DifferentialActivationConfig = Field(default_factory=DifferentialActivationConfig)
    gradient_upstream: GradientUpstreamConfig = Field(default_factory=GradientUpstreamConfig)
    layerwise_gradient_upstream: LayerwiseGradientUpstreamConfig = Field(default_factory=LayerwiseGradientUpstreamConfig)
    counterfactual_gradient: CounterfactualGradientConfig = Field(default_factory=CounterfactualGradientConfig)
    
    top_coact_attr: TopCoactAttrDiscoveryConfig = Field(default_factory=TopCoactAttrDiscoveryConfig)

class PersistConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    save_workers: int = 1
    search_cache_enabled: bool = True
    search_cache_n_sequences: int = 8
    search_cache_component_chunk: int = 4

class RootConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    weights: WeightsConfig
    data: DataConfig = Field(default_factory=DataConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    latents: LatentsConfig = Field(default_factory=LatentsConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    persist: PersistConfig = Field(default_factory=PersistConfig)

def load_config() -> RootConfig:
    data = {}
    config_found = False
    for path in [
        os.path.join(PROJECT_ROOT, "config.yaml"),
        os.path.join(os.getcwd(), "config.yaml"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            config_found = True
            break

    if not config_found:
        print(f"[config] Warning: config.yaml not found in PROJECT_ROOT ({PROJECT_ROOT}) or CWD ({os.getcwd()})")
        # We allow it to continue if it can validate with defaults, but RootConfig requires 'weights'
    
    try:
        return RootConfig.model_validate(data)
    except Exception as e:
        print(f"[config] Error: Configuration validation failed!")
        print(e)
        raise

config = load_config()
