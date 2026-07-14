# A file to load the config from the config.yaml file with strict validation

import yaml
import os
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Dict, List, Optional, Any, Union

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

class SaeRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    encode_backend: str = "standard"
    topk_backend: str = "triton"
    fused_exact_topk_use_native: bool = False

    @field_validator("encode_backend")
    @classmethod
    def validate_encode_backend(cls, v: str) -> str:
        allowed = ["standard", "fused_exact_topk"]
        if v not in allowed:
            raise ValueError(f"encode_backend must be one of {allowed}, got {v!r}")
        return v

    @field_validator("topk_backend")
    @classmethod
    def validate_topk_backend(cls, v: str) -> str:
        allowed = ["triton", "pytorch"]
        if v not in allowed:
            raise ValueError(f"topk_backend must be one of {allowed}, got {v!r}")
        return v

class FirstPassConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sae_encode_mode: str = "streaming"

    @field_validator("sae_encode_mode")
    @classmethod
    def validate_sae_encode_mode(cls, v: str) -> str:
        allowed = ["streaming", "deferred"]
        if v not in allowed:
            raise ValueError(f"sae_encode_mode must be one of {allowed}, got {v!r}")
        return v

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
        allowed = ["reservoir_cpu"]
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

class DistributedMidCtxMergeConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mode: str = "weighted_reservoir"
    sampling_seed: Optional[int] = None

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = ["weighted_reservoir", "candidate_pool"]
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("sampling_seed")
    @classmethod
    def validate_sampling_seed(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("mid_ctx_merge.sampling_seed must be null or >= 0")
        return v

class DistributedSchemaVersionsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    manifest: int = 1
    partial_artifacts: int = 1
    metrics_jsonl: int = 1
    sanity_reports: int = 1
    run_summaries: int = 1

    @field_validator("*")
    @classmethod
    def validate_schema_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError("distributed schema versions must be 1 for the current contracts")
        return v

class DistributedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mode: str = "single_process"
    run_id: Optional[str] = None
    output_base: str = "outputs"
    worker_count: int = 1
    devices: List[Union[int, str]] = Field(default_factory=list)
    launch_strategy: str = "manual_commands"
    resume_policy: str = "fresh"
    cleanup_policy: str = "keep_all"
    parts: List[str] = Field(default_factory=list)
    strict_equivalence: bool = True
    experimental_acknowledgement: bool = False
    experimental_exact_baseline_root: Optional[str] = None
    experimental_quality_toggles: Dict[str, Union[bool, int, float, str]] = Field(default_factory=dict)
    observability_sample_interval_s: float = 30.0
    schema_versions: DistributedSchemaVersionsConfig = Field(
        default_factory=DistributedSchemaVersionsConfig
    )
    sampling_seed: int = 0
    mid_ctx_candidate_pool: DistributedMidCtxCandidatePoolConfig = Field(
        default_factory=DistributedMidCtxCandidatePoolConfig
    )
    mid_ctx_merge: DistributedMidCtxMergeConfig = Field(
        default_factory=DistributedMidCtxMergeConfig
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = [
            "single_process",
            "distributed_simple_exact",
            "distributed_mapreduce_exact",
            "distributed_experimental_fast",
        ]
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        import re

        if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-fA-F]{8}", v):
            raise ValueError("run_id must match YYYYMMDD-HHMMSS-<config_hash_8>")
        return v

    @field_validator("output_base")
    @classmethod
    def validate_output_base(cls, v: str) -> str:
        if not v:
            raise ValueError("output_base must be non-empty")
        return v

    @field_validator("worker_count")
    @classmethod
    def validate_worker_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError("worker_count must be >= 1")
        return v

    @field_validator("launch_strategy")
    @classmethod
    def validate_launch_strategy(cls, v: str) -> str:
        allowed = ["manual_commands", "subprocess", "external_scheduler"]
        if v not in allowed:
            raise ValueError(f"launch_strategy must be one of {allowed}, got {v!r}")
        return v

    @field_validator("resume_policy")
    @classmethod
    def validate_resume_policy(cls, v: str) -> str:
        allowed = ["fresh", "resume", "auto"]
        if v not in allowed:
            raise ValueError(f"resume_policy must be one of {allowed}, got {v!r}")
        return v

    @field_validator("cleanup_policy")
    @classmethod
    def validate_cleanup_policy(cls, v: str) -> str:
        allowed = [
            "keep_all",
            "delete_large_partials_on_success",
            "delete_all_partials_on_success",
            "manual_cleanup_only",
        ]
        if v not in allowed:
            raise ValueError(f"cleanup_policy must be one of {allowed}, got {v!r}")
        return v

    @field_validator("sampling_seed")
    @classmethod
    def validate_sampling_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("sampling_seed must be >= 0")
        return v

    @field_validator("observability_sample_interval_s")
    @classmethod
    def validate_observability_sample_interval(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("observability_sample_interval_s must be > 0")
        return v

    @model_validator(mode="after")
    def validate_mode_policy(self) -> "DistributedConfig":
        if self.mode == "single_process":
            if self.worker_count != 1:
                raise ValueError("single_process mode must use worker_count=1")
            if self.devices:
                raise ValueError("single_process mode must not declare distributed devices")
            if self.launch_strategy != "manual_commands":
                raise ValueError("single_process mode must use launch_strategy='manual_commands'")
        else:
            if self.mode != "distributed_experimental_fast" and self.experimental_acknowledgement:
                raise ValueError("experimental_acknowledgement is only valid for distributed_experimental_fast")

        if self.mode == "distributed_experimental_fast" and not self.experimental_acknowledgement:
            raise ValueError("distributed_experimental_fast requires experimental_acknowledgement=true")
        if self.mode == "distributed_experimental_fast":
            if not self.experimental_exact_baseline_root:
                raise ValueError("distributed_experimental_fast requires experimental_exact_baseline_root")
            if not self.experimental_quality_toggles:
                raise ValueError("distributed_experimental_fast requires experimental_quality_toggles")
            output_marker = self.output_base.replace("\\", "/").lower()
            if "experimental" not in output_marker and "fast" not in output_marker:
                raise ValueError(
                    "distributed_experimental_fast output_base must be clearly marked experimental/fast"
                )

        if (
            self.mode in {"distributed_simple_exact", "distributed_mapreduce_exact"}
            and self.mid_ctx_merge.mode == "candidate_pool"
            and self.mid_ctx_candidate_pool.on_truncation == "allow_bounded_approx"
        ):
            raise ValueError("exact distributed modes require exact mid_ctx truncation handling")

        if self.devices and len(self.devices) != len(set(str(device) for device in self.devices)):
            raise ValueError("distributed devices must be unique")

        if self.devices and len(self.devices) != self.worker_count:
            raise ValueError("distributed devices must match worker_count when explicitly provided")

        return self

class TopCoactivationLatentsConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_latents_per_latent: int = 64
    n_candidates_per_component: int = 16
    candidate_oversample_factor: int = 4
    freq_alpha: float = 2.0
    mode: str = "freq_weighted"  # "freq_weighted" | "raw" | "pmi"
    pmi_clamp_min: float = -5.0
    pmi_clamp_max: float = 10.0
    dump_device: str = "cpu"
    dump_profile: bool = True
    dump_memory_guardrail_bytes: Optional[int] = None
    fail_on_dump_memory_guardrail: bool = True
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

    @field_validator("n_latents_per_latent", "n_candidates_per_component", "candidate_oversample_factor")
    @classmethod
    def validate_positive_top_coactivation_counts(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top coactivation counts must be >= 1")
        return v

    @field_validator("dump_device")
    @classmethod
    def validate_dump_device(cls, v: str) -> str:
        allowed = ["cpu", "gpu"]
        if v not in allowed:
            raise ValueError(f"dump_device must be one of {allowed}, got {v!r}")
        return v

    @field_validator("dump_memory_guardrail_bytes")
    @classmethod
    def validate_dump_memory_guardrail_bytes(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("dump_memory_guardrail_bytes must be null or >= 1")
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

class RestorationConfig(BaseModel):
    """Parameters for the iterative greedy restoration loop, shared by
    attribution_mode="restoration" (single-point gradient per round at the
    current restored state) and "ig_restoration" (per-round integrated
    gradients along the floor->natural path, restored latents connected).
    Only read when one of those modes is active."""

    model_config = ConfigDict(extra='forbid')
    rounds: int = 8
    per_round_k: int = 64             # global (all-site) budget per round
    certificate_tol: float = 0.05     # relative: stop when |gap| <= tol * target
    ig_steps: int = 4                 # alpha samples per round for "ig_restoration"
                                      # (grid {i/N}; alpha=0 always sampled: it
                                      # supplies the loop metric/certificate)
    # One IG re-scoring pass over the final selected set (restoration and
    # ig_restoration): membership unchanged, only ranking scores replaced so
    # truncation reflects the complete circuit rather than the round-by-round
    # state each node was scored in. Uses the method-level ig_steps.
    final_ig_polish: bool = False


class CounterfactualGradientConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    neg_mode: str = "close"        # "close" | "random" | "distant"
    # Attribution mode: "local" = single-point gradient at the live contrast
    # input (original behaviour); "ig_baseline" = integrated gradients along
    # the path from the mean-ablated floor to the natural posctx state
    # (recipe from Sparse Feature Circuits, Marks et al. 2025: IE_ig against
    # the patch baseline), so selection linearises the circuit-only
    # counterfactual that ablation faithfulness evaluates; "restoration" =
    # iterative re-linearisation along the greedy restoration trajectory
    # (state-dependent schedule; see RestorationConfig); "ig_restoration" =
    # the restoration loop with per-round integrated-gradients scoring
    # (state-dependent schedule + path-integrated credit).
    attribution_mode: str = "local"   # "local" | "ig_baseline" | "restoration" | "ig_restoration"
    restoration: RestorationConfig = Field(default_factory=RestorationConfig)
    # Method-level: may counterfactual-gradient circuits carry
    # inhibitor-role members? "include" is the method's classic signed-roles
    # identity; "exclude" builds activator-only circuits (the ablation study
    # of the inhibitors' contribution). Honoured by all three modes.
    negative_roles: str = "include"   # "include" | "exclude"
    ig_steps: int = 10                # interpolation steps for ig_baseline
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

    @field_validator("attribution_mode")
    @classmethod
    def validate_attribution_mode(cls, v: str) -> str:
        allowed = ["local", "ig_baseline", "restoration", "ig_restoration"]
        if v not in allowed:
            raise ValueError(f"attribution_mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("negative_roles")
    @classmethod
    def validate_negative_roles(cls, v: str) -> str:
        allowed = ["include", "exclude"]
        if v not in allowed:
            raise ValueError(f"negative_roles must be one of {allowed}, got {v!r}")
        return v

class AblationGradientConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    neg_mode: str = "close"        # "close" | "random" | "distant"
    # See CounterfactualGradientConfig.attribution_mode.
    attribution_mode: str = "local"   # "local" | "ig_baseline" | "restoration" | "ig_restoration"
    restoration: RestorationConfig = Field(default_factory=RestorationConfig)
    # Method-level: may ablation-gradient circuits carry inhibitor-role
    # members (negative-scored latents) alongside supports? Honoured by the
    # ig_baseline and restoration modes; local mode's attribution util
    # selects supports only and logs a note when "include" is requested.
    negative_roles: str = "exclude"   # "include" | "exclude"
    top_k_inhibitors: int = 12        # per selection scope, when included

    @field_validator("negative_roles")
    @classmethod
    def validate_negative_roles(cls, v: str) -> str:
        allowed = ["include", "exclude"]
        if v not in allowed:
            raise ValueError(f"negative_roles must be one of {allowed}, got {v!r}")
        return v
    ig_steps: int = 10                # interpolation steps for ig_baseline
    distant_pool_size: int = 512   # sequences to sample and rank for "distant" mode
    top_k_supports: int = 12
    top_k_scope: str = "layer_kind"   # "global" | "layer_kind"
    support_threshold: float = 0.01
    min_active_count: int = 1
    max_neg_sequences: int = 16
    pruning_threshold: float = 0.0
    min_suppression_score: float = 0.2

    @field_validator("neg_mode")
    @classmethod
    def validate_neg_mode(cls, v: str) -> str:
        allowed = ["close", "random", "distant"]
        if v not in allowed:
            raise ValueError(f"neg_mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("top_k_scope")
    @classmethod
    def validate_top_k_scope(cls, v: str) -> str:
        allowed = ["global", "layer_kind"]
        if v not in allowed:
            raise ValueError(f"top_k_scope must be one of {allowed}, got {v!r}")
        return v

    @field_validator("attribution_mode")
    @classmethod
    def validate_attribution_mode(cls, v: str) -> str:
        allowed = ["local", "ig_baseline", "restoration", "ig_restoration"]
        if v not in allowed:
            raise ValueError(f"attribution_mode must be one of {allowed}, got {v!r}")
        return v

class HybridGradientConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    run_counterfactual: bool = True
    run_ablation: bool = True
    min_counterfactual_faithfulness: float = 0.2
    min_suppression_score: float = 0.2
    acceptance_mode: str = "either"  # "cf" | "suppression" | "both" | "either"
    pruning_enabled: bool = False
    pruning_method: str = "leave_one_out"  # "leave_one_out" | "sfc_threshold"
    pruning_threshold: float = 0.0
    pruning_objective: str = "both"  # "cf" | "suppression" | "both"
    sfc_node_threshold: float = 0.01
    sfc_edge_threshold: float = 0.01
    sfc_score_mode: str = "abs"  # "abs"

    @field_validator("acceptance_mode")
    @classmethod
    def validate_acceptance_mode(cls, v: str) -> str:
        allowed = ["cf", "suppression", "both", "either"]
        if v not in allowed:
            raise ValueError(f"acceptance_mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("pruning_objective")
    @classmethod
    def validate_pruning_objective(cls, v: str) -> str:
        allowed = ["cf", "suppression", "both"]
        if v not in allowed:
            raise ValueError(f"pruning_objective must be one of {allowed}, got {v!r}")
        return v

    @field_validator("pruning_method")
    @classmethod
    def validate_pruning_method(cls, v: str) -> str:
        allowed = ["leave_one_out", "sfc_threshold"]
        if v not in allowed:
            raise ValueError(f"pruning_method must be one of {allowed}, got {v!r}")
        return v

    @field_validator("sfc_score_mode")
    @classmethod
    def validate_sfc_score_mode(cls, v: str) -> str:
        allowed = ["abs"]
        if v not in allowed:
            raise ValueError(f"sfc_score_mode must be one of {allowed}, got {v!r}")
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

class NegContextSelectionConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidate_pool_size: Optional[int] = None
    exact_negctx_ranking: bool = False
    non_activation_threshold: float = 0.0
    selection_seed: int = 17
    filter_batch_size: int = 128
    load_window_size: int = 1024
    preload_negctx_tokens: bool = True
    token_cache_max_gb: float = 10.0
    token_cache_dtype: str = "int32"

    @field_validator("candidate_pool_size")
    @classmethod
    def validate_candidate_pool_size(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("candidate_pool_size must be null or > 0")
        return v

    @field_validator("filter_batch_size", "load_window_size")
    @classmethod
    def validate_positive_selection_sizes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("neg-context selection sizes must be > 0")
        return v

    @field_validator("token_cache_max_gb")
    @classmethod
    def validate_token_cache_max_gb(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("token_cache_max_gb must be > 0")
        return v

    @field_validator("token_cache_dtype")
    @classmethod
    def validate_token_cache_dtype(cls, v: str) -> str:
        allowed = {"int32", "int64"}
        if v not in allowed:
            raise ValueError(f"token_cache_dtype must be one of {sorted(allowed)}")
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
    # Shared floor knob for every consumer of mean-ablation floors
    # (ig_baseline, restoration, ablation-faithfulness evals):
    # "posctx" = per-seed means over the seed's positive probe batch
    # (SFC's distribution-matched baseline); "global" = seed-independent
    # means over a sample of random corpus sequences (colder floor, no
    # evaluation-distribution leakage), cached once per process.
    # "diverse" = farthest-point sample over stored sequence representations
    # (coverage-weighted: a different floor semantics from "global"'s
    # density-weighted corpus expectation, not a variance-reduced estimate).
    floor_source: str = "posctx"     # "posctx" | "global" | "diverse"
    # Position-aware allowed-set selection (orthogonal to attribution_mode).
    # When true, discovery keeps the token-position axis of the gradient
    # attribution and selects the allowed set as the union, over each seed's
    # causal prefix, of every position's top-N latents by |attribution| —
    # instead of collapsing positions to one fixed top-k set. Circuit stays a
    # flat membership set; only the selection rule changes. Fixes the
    # position-aggregation ("star") sufficiency gap for deep seeds.
    position_aware: bool = False
    position_aware_top_n: int = 64   # per-position latents unioned into the set
    # Per-position selection rule for the allowed set (only used when
    # position_aware is true). "top_n" keeps a fixed count per position
    # (position_aware_top_n). The threshold rules keep a *variable* count by
    # |attribution|, to shrink the union ("menu") when the per-position
    # attribution is peaked:
    #   "abs"      -> keep latents with |attr| >= position_aware_threshold
    #                 (global absolute cut; threshold in raw attribution units).
    #   "relative" -> keep |attr| >= threshold * max_latent|attr| at that
    #                 position (scale-free fraction, threshold in [0, 1]).
    #   "mass"     -> keep the smallest set covering `threshold` of that
    #                 position's total |attr| mass (cumulative, threshold in (0, 1]).
    position_aware_select: str = "top_n"   # "top_n" | "abs" | "relative" | "mass"
    position_aware_threshold: float = 0.0  # meaning depends on position_aware_select
    # Soft position weighting: scale each (position, site) block's attributions by
    # its normalised read-strength so latents at weakly-read positions score lower
    # (a gradient proxy for attention routing). Down-weight, not a hard drop; pairs
    # with position_aware_select="abs" and the magnitude prune.
    position_aware_position_weight: bool = False
    # Membership scope: "aggregate" unions over the probe batch (one circuit for
    # the seed); "per_instance" builds it from a single sequence (the per-input
    # "meal"-sized circuit).
    position_aware_scope: str = "aggregate"   # "aggregate" | "per_instance"
    # Global magnitude prune (post-assembly, method-agnostic — cf / abl / hybrid).
    # Ranks every non-seed member by |attribution_score| and keeps the smallest
    # top-K prefix whose circuit-only (free-phi) sufficiency still meets a target,
    # found by bisection over K (~log2(N) forward passes) — scalable to the large
    # position-aware allowed sets where leave-one-out minimality is intractable.
    magnitude_prune: bool = False
    # Keep-target: if magnitude_prune_target > 0 it is an absolute phi floor;
    # otherwise the floor is (full-circuit phi - magnitude_prune_tolerance).
    magnitude_prune_tolerance: float = 0.05
    magnitude_prune_target: float = 0.0
    magnitude_prune_min_keep: int = 1      # never prune below this many members
    # Which sufficiency the prune preserves — the drivers/closure decomposition:
    # "free" (default) keeps a self-contained CLOSED circuit (free-phi); "pinned"
    # keeps the causal DRIVERS only (pinned-phi, kept latents clamped to clean
    # position-specific values) — compact when the closure tax is high.
    magnitude_prune_objective: str = "free"   # "free" | "pinned"

    @field_validator("floor_source")
    @classmethod
    def validate_floor_source(cls, v: str) -> str:
        allowed = ["posctx", "global", "diverse"]
        if v not in allowed:
            raise ValueError(f"floor_source must be one of {allowed}, got {v!r}")
        return v

    @field_validator("position_aware_top_n")
    @classmethod
    def validate_position_aware_top_n(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"position_aware_top_n must be >= 1, got {v}")
        return v

    @field_validator("position_aware_select")
    @classmethod
    def validate_position_aware_select(cls, v: str) -> str:
        allowed = ["top_n", "abs", "relative", "mass"]
        if v not in allowed:
            raise ValueError(f"position_aware_select must be one of {allowed}, got {v!r}")
        return v

    @field_validator("position_aware_scope")
    @classmethod
    def validate_position_aware_scope(cls, v: str) -> str:
        allowed = ["aggregate", "per_instance"]
        if v not in allowed:
            raise ValueError(f"position_aware_scope must be one of {allowed}, got {v!r}")
        return v

    @field_validator("position_aware_threshold")
    @classmethod
    def validate_position_aware_threshold(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"position_aware_threshold must be >= 0, got {v}")
        return v

    @field_validator("magnitude_prune_tolerance", "magnitude_prune_target")
    @classmethod
    def validate_magnitude_prune_phi(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"magnitude_prune tolerance/target must be in [0, 1], got {v}")
        return v

    @field_validator("magnitude_prune_min_keep")
    @classmethod
    def validate_magnitude_prune_min_keep(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"magnitude_prune_min_keep must be >= 1, got {v}")
        return v

    @field_validator("magnitude_prune_objective")
    @classmethod
    def validate_magnitude_prune_objective(cls, v: str) -> str:
        if v not in ("free", "pinned"):
            raise ValueError(f"magnitude_prune_objective must be 'free' or 'pinned', got {v!r}")
        return v
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
    neg_context_selection: NegContextSelectionConfig = Field(default_factory=NegContextSelectionConfig)

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
    ablation_gradient: AblationGradientConfig = Field(default_factory=AblationGradientConfig)
    hybrid_gradient: HybridGradientConfig = Field(default_factory=HybridGradientConfig)
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
    sae: SaeRuntimeConfig = Field(default_factory=SaeRuntimeConfig)
    first_pass: FirstPassConfig = Field(default_factory=FirstPassConfig)
    latents: LatentsConfig = Field(default_factory=LatentsConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    persist: PersistConfig = Field(default_factory=PersistConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @model_validator(mode="before")
    @classmethod
    def apply_distributed_mode_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        distributed = data.get("distributed") or {}
        if not isinstance(distributed, dict):
            return data
        if distributed.get("mode", "single_process") == "single_process":
            return data

        updated = dict(data)
        persist = dict(updated.get("persist") or {})
        persist.setdefault("build_search_cache_after_pipeline", False)
        updated["persist"] = persist
        return updated

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
