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
    keep_model_loaded_for_neg_ctx: bool = False

class TopCtxConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_sequences: int = 64

class MidCtxConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_sequences: int = 64
    mode: str = "reservoir_cpu"
    band_low_sigma: float = 0.5
    band_high_sigma: float = 1.5
    warmup_batches: int = 100

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = ["reservoir_cpu", "gpu_topk_mid"]
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v!r}")
        return v

class NegCtxConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_sequences: int = 64
    n_neighbors: int = 512
    min_pos_ctx: int = 8
    repr_mode: str = "mean_pool"
    max_repr_seqs: Optional[int] = 200000
    backend: str = "single_gpu_exact"
    devices: List[Union[int, str]] = Field(default_factory=list)
    memory_guardrail_fraction: float = 0.90
    fail_on_memory_guardrail: bool = True

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = ["single_gpu_exact", "multi_gpu_exact", "multi_gpu_index_sharded_exact"]
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}, got {v!r}")
        return v

    @field_validator("memory_guardrail_fraction")
    @classmethod
    def validate_memory_guardrail_fraction(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("memory_guardrail_fraction must be in (0, 1]")
        return v

class LogitCtxConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_tokens_per_latent: int = 32
    topk_output_tokens: int = 32

class SeqLatentIndexConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool = False
    top_k_per_component: int = 12

class DistributedMidCtxCandidatePoolConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    enabled: bool = True
    band_margin_sigma: float = 1.0
    max_candidates_per_latent: Optional[int] = None
    on_truncation: str = "replay_fallback"

    @field_validator("band_margin_sigma")
    @classmethod
    def validate_band_margin_sigma(cls, v: float) -> float:
        if v < 0:
            raise ValueError("band_margin_sigma must be >= 0")
        return v

    @field_validator("max_candidates_per_latent")
    @classmethod
    def validate_max_candidates_per_latent(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("max_candidates_per_latent must be null or >= 1")
        return v

    @field_validator("on_truncation")
    @classmethod
    def validate_on_truncation(cls, v: str) -> str:
        allowed = ["fail", "replay_fallback", "allow_bounded_approx"]
        if v not in allowed:
            raise ValueError(f"on_truncation must be one of {allowed}, got {v!r}")
        return v

class DistributedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sampling_seed: int = 0
    mid_ctx_candidate_pool: DistributedMidCtxCandidatePoolConfig = Field(
        default_factory=DistributedMidCtxCandidatePoolConfig
    )

    @field_validator("sampling_seed")
    @classmethod
    def validate_sampling_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("sampling_seed must be >= 0")
        return v

class TopCoactivationLatentsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_latents_per_latent: int = 64
    n_candidates_per_component: int = 16
    freq_alpha: float = 2.0
    mode: str = "freq_weighted"  # "freq_weighted" | "raw" | "pmi"
    pmi_clamp_min: float = -5.0
    pmi_clamp_max: float = 10.0
    dump_device: str = "cpu"
    dump_profile: bool = True
    reduce_backend: str = "single_process"
    reduce_shards: int = 1
    reduce_shard_output_dir: Optional[str] = None
    reduce_omp_threads: Optional[int] = None
    reduce_schedule_chunk: int = 256

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = ["freq_weighted", "raw", "pmi"]
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v}")
        return v

    @field_validator("dump_device")
    @classmethod
    def validate_dump_device(cls, v: str) -> str:
        allowed = ["cpu", "gpu"]
        if v not in allowed:
            raise ValueError(f"dump_device must be one of {allowed}, got {v!r}")
        return v

    @field_validator("reduce_backend")
    @classmethod
    def validate_reduce_backend(cls, v: str) -> str:
        allowed = ["single_process", "target_sharded"]
        if v not in allowed:
            raise ValueError(f"reduce_backend must be one of {allowed}, got {v!r}")
        return v

    @field_validator("reduce_shards")
    @classmethod
    def validate_reduce_shards(cls, v: int) -> int:
        if v < 1:
            raise ValueError("reduce_shards must be >= 1")
        return v

    @field_validator("reduce_omp_threads")
    @classmethod
    def validate_reduce_omp_threads(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("reduce_omp_threads must be null or >= 1")
        return v

    @field_validator("reduce_schedule_chunk")
    @classmethod
    def validate_reduce_schedule_chunk(cls, v: int) -> int:
        if v < 1:
            raise ValueError("reduce_schedule_chunk must be >= 1")
        return v

class LatentsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    top_ctx: TopCtxConfig = Field(default_factory=TopCtxConfig)
    mid_ctx: MidCtxConfig = Field(default_factory=MidCtxConfig)
    neg_ctx: NegCtxConfig = Field(default_factory=NegCtxConfig)
    logit_ctx: LogitCtxConfig = Field(default_factory=LogitCtxConfig)
    top_coactivation: TopCoactivationLatentsConfig = Field(default_factory=TopCoactivationLatentsConfig)
    seq_latent_index: SeqLatentIndexConfig = Field(default_factory=SeqLatentIndexConfig)

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
    neg_mode: str = "close"        # "close" | "random" | "distant"
    distant_pool_size: int = 512   # sequences to sample and rank for "distant" mode
    top_k_activators: int = 8
    top_k_inhibitors: int = 8
    top_k_scope: str = "global"   # "global" | "layer_kind"
    activator_threshold: float = 0.01
    inhibitor_threshold: float = 0.01
    min_active_count: int = 1
    max_neg_sequences: int = 4
    neg_batch_size: int = 4           # sequences per grad-enabled contrast pass (lower = less VRAM)
    pruning_threshold: float = 0.0
    min_faithfulness: float = 0.2
    node_presence_eval: bool = True

    @field_validator("neg_mode")
    @classmethod
    def validate_neg_mode(cls, v: str) -> str:
        allowed = ["close", "random", "distant"]
        if v not in allowed:
            raise ValueError(f"neg_mode must be one of {allowed}, got {v!r}")
        return v

class CircuitTracerBaselineConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    probe_batch_size: int = 1        # sequences per forward pass (keep low for 16 GB VRAM)
    max_sequences: int = 8           # pos_ctx sequences used to build the matrix
    target_chunk_size: int = 8       # deprecated — no longer used; retained for YAML compat
    logit_top_k: int = 5             # maximum number of logit target nodes
    desired_logit_prob: float = 0.95 # cumulative softmax probability to cover when selecting logit targets
    influence_max_iter: int = 1000   # power-iteration safety cap; raises RuntimeError on non-convergence (matches original circuit-tracer)
    node_threshold: float = 0.8      # fraction of total node-influence to retain (0–1); scale-invariant
    edge_threshold: float = 0.98     # fraction of total edge-influence to retain (0–1); scale-invariant
    min_faithfulness: float = 0.2    # upstream-faithfulness acceptance gate
    pruning_threshold: float = 0.0   # minimality pruning drop threshold (0 = disabled)
    min_active_count: int = 1        # skip latents below this global lifetime firing count
    max_feature_nodes: int = 2048    # cap on feature nodes; top-N by logit-influence ranking (or peak activation when logit_top_k=0)
    stop_error_grad: bool = False    # if True, gradient flows only through SAE features (not error term)
    include_error_nodes: bool = True # if True, add one error sentinel node per (layer, kind) for reconstruction-error attribution
    online_ranking_interval: int = 4   # re-rank remaining features every N cycles (matches original update_interval=4); 0 = one-shot ordering
    feature_batch_size: int = 32       # features processed per cycle (matches original batch_size=32); 4×32=128 features per ranking update
    include_token_nodes: bool = False # if True, add T token sentinel nodes (one per input position) enabling per-position embedding attribution

class ClusterContrastConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_clusters: int = 3000
    top_clusters: int = 8
    kmeans_iters: int = 40
    kmeans_seed: int = 42
    num_pos_seqs: int = 64              # positive sequences per cluster (top-N in-cluster, near centroid)
    num_neg_seqs: int = 64              # negative sequences per cluster (top-N out-of-cluster, near centroid)
    batch_size: int = 8                 # sequences per forward pass
    eval_position: str = "last"         # "last" | "all"
    top_k_activators: int = 12
    top_k_inhibitors: int = 12
    top_k_scope: str = "layer_kind"     # "global" | "layer_kind"
    activator_threshold: float = 0.01
    inhibitor_threshold: float = 0.01
    min_active_count: int = 1
    pruning_threshold: float = 0.0
    run_eval: bool = True

    @field_validator("eval_position")
    @classmethod
    def validate_eval_position(cls, v: str) -> str:
        allowed = ["last", "all"]
        if v not in allowed:
            raise ValueError(f"eval_position must be one of {allowed}, got {v!r}")
        return v

    @field_validator("top_k_scope")
    @classmethod
    def validate_top_k_scope(cls, v: str) -> str:
        allowed = ["global", "layer_kind"]
        if v not in allowed:
            raise ValueError(f"top_k_scope must be one of {allowed}, got {v!r}")
        return v

class SeedFilterConfig(BaseModel):
    """Constrains which seeds CandidateSelector returns by layer and/or kind.

    Both fields default to empty lists, which means no filtering (all layers
    and all kinds are allowed). Any non-empty list acts as an allowlist.

    Example — only MLP and residual seeds from the first four layers:
        layers: [0, 1, 2, 3]
        kinds:  ["mlp", "resid"]
    """
    model_config = ConfigDict(extra='forbid')
    layers: List[int] = Field(default_factory=list)
    kinds: List[str] = Field(default_factory=list)

    @field_validator("kinds")
    @classmethod
    def validate_kinds(cls, v: List[str]) -> List[str]:
        allowed = {"attn", "mlp", "resid"}
        for k in v:
            if k not in allowed:
                raise ValueError(
                    f"seed_filter.kinds entries must be one of {sorted(allowed)}, got {k!r}"
                )
        return v


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_seeds: int = 128
    probe_batch_size: int = 16
    neg_ctx_eval_max: int = 16
    min_faithfulness: float = 0.2
    min_active_count: int = 1
    max_neighbors: int = 32
    focal_monosemantic_min_partners: int = 4  # min observed coact partners for focal_monosemantic criterion
    methods: List[str] = Field(default_factory=lambda: ["coactivation_statistical", "logit_attribution"])
    # Which scoring criteria to use when shortlisting seed latents.
    # Available: "logit_impact", "connectivity", "surprise", "context_coherence",
    #            "activation_variance", "logit_specificity", "coactivation_diversity",
    #            "last_token_activity", "top_ctx_saturation",
    #            "pos_neg_contrast", "cross_layer_reach", "cross_component_breadth",
    #            "burstiness", "mid_ctx_richness", "activation_skew", "logit_diversity",
    #            "pagerank_centrality", "activation_entropy", "coactivation_uniqueness",
    #            "stratified_random", "circuit_yield"
    seed_criteria: List[str] = Field(default_factory=lambda: [
        "logit_impact", "connectivity", "surprise", "context_coherence",
        "activation_variance", "logit_specificity", "coactivation_diversity",
        "last_token_activity", "top_ctx_saturation",
        "pos_neg_contrast", "cross_layer_reach", "cross_component_breadth",
        "burstiness", "mid_ctx_richness", "activation_skew", "logit_diversity",
        "pagerank_centrality", "activation_entropy", "coactivation_uniqueness",
        "stratified_random", "circuit_yield",
    ])
    seed_filter: SeedFilterConfig = Field(default_factory=SeedFilterConfig)

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
    circuit_tracer_baseline: CircuitTracerBaselineConfig = Field(default_factory=CircuitTracerBaselineConfig)
    cluster_contrast: ClusterContrastConfig = Field(default_factory=ClusterContrastConfig)

    top_coact_attr: TopCoactAttrDiscoveryConfig = Field(default_factory=TopCoactAttrDiscoveryConfig)

class PersistConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    save_workers: int = 1
    search_cache_enabled: bool = True
    build_search_cache_after_pipeline: bool = True
    atomic_saves: bool = True
    search_cache_n_sequences: int = 8
    search_cache_component_chunk: int = 4

class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    methods: List[str] = Field(default_factory=lambda: ["coactivation_overlap"])

class RootConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    weights: WeightsConfig
    data: DataConfig = Field(default_factory=DataConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    latents: LatentsConfig = Field(default_factory=LatentsConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    persist: PersistConfig = Field(default_factory=PersistConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

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
