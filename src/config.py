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
    # Scalar m that IG attributes. "logit" (default) preserves existing
    # behaviour; "logprob" log-softmaxes first. SFC's own metric is a log
    # PROBABILITY difference — a raw logit keeps an input-independent baseline
    # that mean-ablation cannot remove, collapsing the faithfulness denominator
    # (measured m(full)-m(empty)=1.81, leaving every ratio noise-dominated).
    metric_mode: str = "logit"        # "logit" | "logprob"
    # How per-token effects become one score per node.
    #
    # "npa" (default) is SFC's OWN rule for non-templatic data, verified against
    #       both the paper (App. C, "Aggregating across token positions and
    #       examples") and their reference implementation (feature-circuits
    #       circuit.py:69-70, `.sum(dim=1)` over position then `.mean(dim=0)`
    #       over examples): sum across token positions, then example-wise mean.
    #       One node per (site, latent).
    # "pa_union" is OURS, and is labelled as such — an extension on top of their
    #       attribution, NOT part of SFC: per-position selection, then union over
    #       the causal prefix ("allow, don't force"). Routes through the same
    #       `select_position_aware` backend our own methods use, so an
    #       SFC-vs-ours comparison isolates the ATTRIBUTION, not the reduction.
    #
    # SFC's OTHER rule — templatic position-INDEXED nodes — is deliberately not
    # implemented: it needs position-varying keep-sets in CircuitOnlyPatcher, and
    # it presupposes the templatic data our corpus does not have.
    #
    # A third value, "pa_peak" (score each latent at its strongest position), was
    # removed 2026-07-22. It was OUR invention, not SFC's, and shipping it under
    # a mode list headed "SFC" risked misrepresenting their algorithm. It was
    # also measured degenerate with npa (Jaccard 0.999).
    position_mode: str = "npa"        # "npa" | "pa_union"
    # Selection rule for position_mode="pa_union" only (ignored otherwise).
    # Mirrors discovery.position_aware_select; "abs_pctl" @ 90 is the default
    # validated for our own methods, so the two stacks stay comparable.
    pa_select: str = "abs_pctl"       # "top_n" | "abs" | "abs_pctl" | "relative" | "mass"
    pa_top_n: int = 64                # for select="top_n"
    pa_threshold: float = 90.0        # abs: raw cut; abs_pctl: PERCENTILE (0-100)

    @field_validator("metric_mode")
    @classmethod
    def validate_sfc_metric_mode(cls, v: str) -> str:
        allowed = ["logit", "logprob"]
        if v not in allowed:
            raise ValueError(f"metric_mode must be one of {allowed}, got {v}")
        return v

    @field_validator("position_mode")
    @classmethod
    def validate_sfc_position_mode(cls, v: str) -> str:
        allowed = ["npa", "pa_union"]
        if v not in allowed:
            raise ValueError(f"position_mode must be one of {allowed}, got {v}")
        return v

    @field_validator("pa_select")
    @classmethod
    def validate_sfc_pa_select(cls, v: str) -> str:
        allowed = ["top_n", "abs", "abs_pctl", "relative", "mass"]
        if v not in allowed:
            raise ValueError(f"pa_select must be one of {allowed}, got {v}")
        return v

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
    # Round admission rule (crosses with discovery.position_aware -> 4 cells):
    #   "top_k"    -> classic: global top-per_round_k by |score| (PA on: the
    #                 per-position top-per_round_k union — budget grows with
    #                 live positions, prefer abs_pctl when PA is on).
    #   "abs_pctl" -> admit every latent whose round |score| clears the
    #                 round_abs_pctl percentile of the round's pooled nonzero
    #                 |score| across sites; variable count per round, and the
    #                 cut RE-RESOLVES each round at the moving linearisation
    #                 state (adaptive admission). PA on: per-position pooled-
    #                 percentile union — greedy positional coverage.
    round_select: str = "top_k"       # "top_k" | "abs_pctl"
    round_abs_pctl: float = 95.0      # percentile (0-100) for round_select="abs_pctl"
    certificate_tol: float = 0.05     # relative: stop when |gap| <= tol * target
    ig_steps: int = 4                 # alpha samples per round for "ig_restoration"
                                      # (grid {i/N}; alpha=0 always sampled: it
                                      # supplies the loop metric/certificate)
    # One IG re-scoring pass over the final selected set (restoration and
    # ig_restoration): membership unchanged, only ranking scores replaced so
    # truncation reflects the complete circuit rather than the round-by-round
    # state each node was scored in. Uses the method-level ig_steps.
    final_ig_polish: bool = False
    # Sequences per round grad pass. None = discovery.probe_batch_size. The
    # global probe batch was sized for the CONTRASTIVE memory law (dense
    # anchors+deltas at every site); restoration's instrument is far lighter
    # (profiled 2026-07-18: 10.6G peak at batch 4 with 5G headroom), so its
    # rounds can run wider and pay proportionally fewer passes.
    grad_batch_size: Optional[int] = None

    @field_validator("grad_batch_size")
    @classmethod
    def validate_grad_batch_size(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError(f"grad_batch_size must be None or >= 1, got {v}")
        return v

    @field_validator("round_select")
    @classmethod
    def validate_round_select(cls, v: str) -> str:
        allowed = ["top_k", "abs_pctl"]
        if v not in allowed:
            raise ValueError(f"round_select must be one of {allowed}, got {v!r}")
        return v

    @field_validator("round_abs_pctl")
    @classmethod
    def validate_round_abs_pctl(cls, v: float) -> float:
        if not (0 < v < 100):
            raise ValueError(f"round_abs_pctl must be a percentile in (0, 100), got {v}")
        return v


class LearnedMaskConfig(BaseModel):
    """The learned continuous-mask modes (abl-mask, cf-mask_contrast,
    cf-mask_negctx). Shared by all three objectives; beta is read only by
    mask_contrast. Losses target the seed PRE-activation and every target is
    a natural level ("reproduce, don't maximise"). Motivation: gradient signs
    are state-dependent to near-independence (corr 0.05 at L8, 2026-07-24),
    so membership is optimised against the loss rather than read off any
    single gradient."""
    model_config = ConfigDict(extra='forbid')
    # Defaults are the CALIBRATED configuration (L2/L8/L10 sweep,
    # 2026-07-25): AdamW at wd 0.05 puts held-out free0 nearest 1.0 across
    # seeds. See the weight-decay note below — wd is only meaningful jointly
    # with steps and lr.
    steps: int = 400
    lr: float = 0.05
    # Per-latent price (the penalty is a SUM over latents, not a mean —
    # mean-normalising put the per-latent gradient under Adam's eps and
    # nothing pruned). 1e-4 means "a latent must reduce the squared preact
    # error by ~1e-4 to pay for itself".
    l1_lambda: float = 0.0001
    beta: float = 1.0                # mask_contrast: weight of negctx silence
    # mask_inject only. delta is priced on its OWN scale: it carries
    # activation magnitudes while the mask's l1_lambda prices unitless edits,
    # and sharing one price let a diffuse sub-threshold delta blanket reach
    # the target with zero selected latents (v1 degeneracy, L8 2026-07-24).
    # None => fall back to l1_lambda (v1 behaviour, kept reproducible).
    inject_lambda: Optional[float] = 0.01
    # Exclude the N sites nearest the seed from injection: additive steering
    # at the adjacent resid site is trivially expressive.
    inject_exclude_sites: int = 0
    keep_threshold: float = 0.5      # m above (pos/contrast) / edit above (negctx)
    holdout_frac: float = 0.25       # probe split: train on the rest, report both
    theta_init: float = 4.0          # sigmoid(4) ~= 0.982: start at ~natural
    log_every: int = 50
    # Depth-adaptive VRAM guard, same law as ig_negctx's (peak ~= base +
    # sites x per-site tensors; one backward holds ALL upstream sites at
    # once, and under WDDM the margin silently spills to system RAM at PCIe
    # speed instead of OOMing — measured on the L10 sweep, 32 sites at
    # batch 4). Above deep_site_threshold sites the engine shrinks the
    # MICRO-batch to deep_batch_size and keeps the effective batch via
    # gradient accumulation, so deep and shallow seeds share one
    # optimisation regime — only peak VRAM (and a little wall-clock) differ.
    deep_site_threshold: int = 21    # switch from L7-mlp upward, as ig_negctx
    deep_batch_size: int = 2         # micro-batch under the guard
    # "adamw" adds decoupled decay pulling theta toward 0 == m toward 0.5
    # (the keep-threshold boundary, NOT sparsity): a confidence regulariser;
    # the L1 term remains the only sparsifier.
    optimizer: str = "adamw"         # "adam" | "adamw"
    # DECAY IS SCHEDULE-COUPLED. Decay shrinks theta by (1 - lr*wd) per step,
    # so the total shrinkage is exp(-steps*lr*wd) — only the PRODUCT matters.
    # The calibrated point is steps*lr*wd ~= 1.0 (400 * 0.05 * 0.05), which
    # holds kept-member m near 0.75; that value, not wd itself, is what the
    # seeds agree on (L2 0.75 / L8 0.76 / L10 0.70 at their optima).
    # CHANGING steps OR lr REQUIRES RESCALING wd to keep the product ~1.0,
    # otherwise the calibration breaks silently. The engine logs the product.
    weight_decay: float = 0.05       # adamw only
    # "stream" matches the eval patcher's dtype (ordinary mixed precision:
    # fp32 params, stream-dtype activations). "fp32" forces float32 dense
    # codes, which roughly doubles the optimisation footprint.
    code_dtype: str = "stream"       # "stream" | "fp32"
    # Learning-rate schedule. Membership is a threshold crossing, so decaying
    # lr freezes it progressively instead of letting the final step decide.
    # NOTE both budgets scale with sum(lr): a decaying schedule roughly HALVES
    # sum(lr) for the same peak, so a decayed run needs peak lr ~2x the
    # calibrated constant lr to keep lambda and wd meaningful. The engine logs
    # sum(lr) and both budgets.
    # MEASURED 2026-07-29 (L2/L8/L10, matched sum(lr)): decay is the WRONG
    # direction — cosine/linear gave 11-28% BIGGER circuits than constant with
    # no consistent quality gain, so "constant" stays the default. Pruning is a
    # slow threshold crossing, so late-training lr is what shrinks a circuit;
    # the "_up" warmup variants exist to push that lever the other way.
    lr_schedule: str = "constant"    # constant | cosine | linear | cosine_up | linear_up
    lr_min_frac: float = 0.05        # floor as a fraction of peak lr
    warmup_frac: float = 0.0         # ramp floor->peak over this fraction, then decay
    # What a fully masked (m=0) latent becomes. "zero" is the historical
    # behaviour and keeps the mask's training counterfactual identical to
    # free0's — which means the mask is always scored on home turf. A mean
    # floor makes m=0 reproduce the state freeM/freeN measure against, so mask
    # and mean-floor methods can be compared on a metric neither owns.
    # Prefer "negctx" over "posctx": the posctx fill reaches 23%/30% of a_pos
    # at L8/L9 (it credits itself at depth), negctx measures 0.0000 everywhere.
    # Separate from discovery.floor_source ON PURPOSE — that knob is shared
    # with the ig hops and would silently move the other arms in a run.
    mask_floor_source: str = "zero"   # zero | posctx | negctx | dual
    # "dual" scores the mask under BOTH the zero and negctx floors every
    # step, each term normalised by its own closed-mask loss (they differ
    # by a large factor, in either direction � a negctx floor can install
    # active suppression rather than mere silence). Measured
    # motivation: a negctx-only floor learns the DELTA from the negative
    # baseline and its free0 is exactly 0.0 at L5/L8 (members alone cannot
    # reach top-k); a zero-only floor is sufficient but never asked what
    # distinguishes firing from a near-identical silent context.
    dual_floor_weight: float = 1.0    # gamma on the negctx term

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"steps must be >= 1, got {v}")
        return v

    @field_validator("keep_threshold")
    @classmethod
    def validate_keep_threshold(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"keep_threshold must be in (0, 1), got {v}")
        return v

    @field_validator("holdout_frac")
    @classmethod
    def validate_holdout_frac(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"holdout_frac must be in [0, 1), got {v}")
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        if v not in ("adam", "adamw"):
            raise ValueError(f"optimizer must be 'adam' or 'adamw', got {v!r}")
        return v

    @field_validator("mask_floor_source")
    @classmethod
    def validate_mask_floor_source(cls, v: str) -> str:
        if v not in ("zero", "posctx", "negctx", "dual"):
            raise ValueError("mask_floor_source must be 'zero', 'posctx', "
                             f"'negctx' or 'dual', got {v!r}")
        return v

    @field_validator("code_dtype")
    @classmethod
    def validate_code_dtype(cls, v: str) -> str:
        if v not in ("stream", "fp32"):
            raise ValueError(f"code_dtype must be 'stream' or 'fp32', got {v!r}")
        return v

    @field_validator("lr_schedule")
    @classmethod
    def validate_lr_schedule(cls, v: str) -> str:
        allowed = ("constant", "cosine", "linear", "cosine_up", "linear_up")
        if v not in allowed:
            raise ValueError(f"lr_schedule must be one of {allowed}, got {v!r}")
        return v

    @field_validator("lr_min_frac", "warmup_frac")
    @classmethod
    def validate_lr_min_frac(cls, v: float, info) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"{info.field_name} must be in [0, 1), got {v}")
        return v

    @field_validator("beta", "l1_lambda", "weight_decay")
    @classmethod
    def validate_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"must be >= 0, got {v}")
        return v

    @field_validator("inject_lambda")
    @classmethod
    def validate_inject_lambda(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"inject_lambda must be >= 0 or None, got {v}")
        return v

    @field_validator("inject_exclude_sites")
    @classmethod
    def validate_inject_exclude_sites(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"inject_exclude_sites must be >= 0, got {v}")
        return v

    @field_validator("lr")
    @classmethod
    def validate_lr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"lr must be > 0, got {v}")
        return v


class CounterfactualGradientConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    # "fused" = one contrast set drawing quotas from all three sub-modes
    # (close first — the sharpest boundary signal — then distant, then
    # random), deduplicated by sequence id. A richer baseline than any
    # single mode: close-only can miss latents that are absent across the
    # BROADER distribution, which the distant/random quotas supply.
    neg_mode: str = "close"        # "close" | "random" | "distant" | "fused"
    # Attribution mode: "local" = single-point gradient at the live contrast
    # input (original behaviour); "ig_mean" = integrated gradients along
    # the path from the mean-ablated floor to the natural posctx state
    # (recipe from Sparse Feature Circuits, Marks et al. 2025: IE_ig against
    # the patch baseline), so selection linearises the circuit-only
    # counterfactual that ablation faithfulness evaluates; "restoration" =
    # iterative re-linearisation along the greedy restoration trajectory
    # (state-dependent schedule; see RestorationConfig); "ig_restoration" =
    # the restoration loop with per-round integrated-gradients scoring
    # (state-dependent schedule + path-integrated credit); "ig_negctx" =
    # integrated gradients along the LATENT-SPACE path from the negctx state
    # to the posctx target (the same values the counterfactual-faithfulness
    # eval injects), run on negctx tokens — the exact estimator of the
    # single-point "local" contrast gradient, with an IG completeness
    # certificate (sum of attributions ~= the seed's actual change under the
    # eval's intervention). cf-only: it needs a contrast input, so it has no
    # ablation-gradient counterpart. See
    # dev-notes/contrastive-ig-for-position-aware-cf.md. "restoration_negctx"
    # = the restoration loop transplanted onto ig_negctx's trajectory: run on
    # negctx tokens, restored latents PINNED to their posctx targets (the cf
    # eval's injection semantics — the loop's final state IS the eval's
    # intervened state), unrestored latents live-connected on the modified
    # negctx stream; each round re-linearises grad x (target - live value) at
    # the current injected state, so the certificate closing means the
    # selected set makes the seed fire on negctx under injection
    # (cf-faithfulness ~= 1 by construction). cf-only, like ig_negctx.
    attribution_mode: str = "local"   # "local" | "ig_mean" | "restoration" | "ig_restoration" | "ig_negctx" | "restoration_negctx"
    restoration: RestorationConfig = Field(default_factory=RestorationConfig)
    # Method-level: may counterfactual-gradient circuits carry
    # inhibitor-role members? "include" is the method's classic signed-roles
    # identity; "exclude" builds activator-only circuits (the ablation study
    # of the inhibitors' contribution). Honoured by all three modes.
    negative_roles: str = "include"   # "include" | "exclude"
    # Which quantity ranks ABSENT ACTIVATORS (inhibitors are unaffected —
    # acts x grad is already an effect, not a sensitivity).
    #
    # "gradient"          — the raw dL/df_g: the seed's per-unit sensitivity to
    #                       the latent. The method's original behaviour.
    # "gradient_x_posctx" — grad x the latent's posctx target value: the
    #                       first-order effect of the intervention the
    #                       evaluation actually performs, since counterfactual
    #                       faithfulness INJECTS each activator at its posctx
    #                       value. Ranking therefore matches what the metric
    #                       rewards, and latents with no posctx value score 0
    #                       rather than ranking on a sensitivity they can never
    #                       cash in. That restores k-sparsity to the signal,
    #                       which is what bounds the position-aware union
    #                       (the raw gradient is dense over the whole
    #                       dictionary; see dev-notes/contrastive-ig-for-
    #                       position-aware-cf.md).
    #
    # Applies to the "local" attribution mode (the contrast hop), position-aware
    # or not. ig_mean/restoration already attribute with grad x delta by
    # construction, so the knob does not reach them.
    activator_signal: str = "gradient"   # "gradient" | "gradient_x_posctx"
    # Metric integrated along ig_negctx's negctx -> posctx-target path:
    # "drive" = the seed's pre-activation itself. Attributions sum to the
    #           seed's ACTUAL rise under the eval's intervention; the metric is
    #           linear in the latents, so the left-Riemann completeness
    #           certificate is well-conditioned. Overshooting latents keep
    #           positive credit.
    # "gap"   = -(pre-activation - target_act_pos)^2, the same objective the
    #           "local" contrast hop optimises. Attributions sum to how much
    #           the GAP closed; latents that push the seed past the posctx
    #           target get negative credit (reclassified as inhibitors). The
    #           quadratic metric makes the certificate O(1/ig_steps) even on a
    #           linear model — expect looser certificates.
    ig_negctx_objective: str = "drive"   # "drive" | "gap"
    # How restoration_negctx handles MEMBERSHIP (which latents) and INJECTION
    # (what value restored latents are pinned to for the certificate) — two
    # orthogonal axes, three useful combinations:
    # "posctx"      — membership: top-|posctx-injection effect|, both signs;
    #                 injection: ALL restored -> posctx. The original behaviour.
    #                 Good free0 (both-sign membership) but cf-INCONSISTENT: the
    #                 eval suppresses inhibitors to 0 while this pins them to
    #                 posctx, so the certificate models a state the eval never
    #                 scores (measured: free0 0.88 / deep 0.71, cf ~0.80).
    # "directional" — membership: helping moves ONLY (drop latents that neither
    #                 raising-to-posctx nor removing-to-0 helps); injection:
    #                 activators->posctx, inhibitors->0. cf-consistent, but the
    #                 helping filter strips both-sign members free0 needs
    #                 (measured: free0 collapses to 0.66 / deep 0.49).
    # "both_sign"   — membership: top-|posctx-injection effect|, both signs (as
    #                 "posctx", to keep free0's both-sign closure); injection:
    #                 activators->posctx, inhibitors->0 (as "directional", so
    #                 the certificate matches the eval's Score-1 intervention).
    #                 The reconciliation: separates "which latents are members"
    #                 (free0) from "what value they inject at" (cf).
    restoration_negctx_mode: str = "posctx"   # "posctx" | "directional" | "both_sign"
    # Depth-adaptive neg microbatch for ig_negctx. Its per-site residency
    # is the largest of any arm (leaf + grad + fp32 delta + fp32 per-position
    # accumulator ~= 252MB/site at B=8), and one backward holds ALL upstream
    # sites at once — measured peak ~= 7G + sites x 252MB, which crosses a
    # 16GB card at ~29 sites (L10+; the paging tax tripled those chunks'
    # wall-clock). When the seed's upstream site count exceeds
    # ig_negctx_deep_site_threshold, the hop drops its microbatch to
    # ig_negctx_deep_neg_batch (halving every per-site tensor); shallower
    # seeds keep neg_batch_size so they don't pay the extra chunk overhead.
    # 21 = switch from L7-mlp upward (site count 3*layer + earlier kinds),
    # leaving ~2GB more margin than the measured crossing point.
    ig_negctx_deep_site_threshold: int = 21
    ig_negctx_deep_neg_batch: int = 4
    ig_steps: int = 10                # interpolation steps for ig_mean and ig_negctx
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
        allowed = ["close", "random", "distant", "fused"]
        if v not in allowed:
            raise ValueError(f"neg_mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("attribution_mode")
    @classmethod
    def validate_attribution_mode(cls, v: str) -> str:
        # "ig_negctx" and "restoration_negctx" are cf-only: the modes the two
        # gradient methods do NOT share. Both anchor on a negctx contrast
        # input, which ablation gradient does not have.
        # ("activation_gradient" was removed: it is a top-level method now
        # (ActivationGradientDiscovery), not a mode — it runs on posctx and
        # never answered counterfactual gradient's negctx question.)
        # "mask_contrast"/"mask_negctx" are the cf-hosted learned-mask modes
        # (see LearnedMaskConfig): contrast optimises reconstruction on posctx
        # PLUS silence on negctx; mask_negctx is the pure gate-opening search
        # on negctx (the _negctx suffix marks the counterfactual-distribution
        # worker, as with ig_negctx/restoration_negctx).
        # "mask_inject" is the full learned heir of the original cf
        # question: value' = m*value + delta on negctx, so it learns BOTH
        # C1 roles — delta-selected absent activators and edit-selected
        # present inhibitors.
        allowed = ["local", "ig_mean", "restoration", "ig_restoration",
                   "ig_negctx", "restoration_negctx",
                   "mask_contrast", "mask_negctx", "mask_inject"]
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

    @field_validator("ig_negctx_objective")
    @classmethod
    def validate_ig_negctx_objective(cls, v: str) -> str:
        allowed = ["drive", "gap"]
        if v not in allowed:
            raise ValueError(f"ig_negctx_objective must be one of {allowed}, got {v!r}")
        return v

    @field_validator("restoration_negctx_mode")
    @classmethod
    def validate_restoration_negctx_mode(cls, v: str) -> str:
        allowed = ["posctx", "directional", "both_sign"]
        if v not in allowed:
            raise ValueError(f"restoration_negctx_mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("ig_negctx_deep_site_threshold", "ig_negctx_deep_neg_batch")
    @classmethod
    def validate_ig_negctx_deep(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"ig_negctx deep-site settings must be >= 1, got {v}")
        return v

    @field_validator("activator_signal")
    @classmethod
    def validate_activator_signal(cls, v: str) -> str:
        allowed = ["gradient", "gradient_x_posctx"]
        if v not in allowed:
            raise ValueError(f"activator_signal must be one of {allowed}, got {v!r}")
        return v

class AblationGradientConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    neg_mode: str = "close"        # "close" | "random" | "distant"
    # See CounterfactualGradientConfig.attribution_mode.
    attribution_mode: str = "local"   # "local" | "ig_mean" | "restoration" | "ig_restoration"
    restoration: RestorationConfig = Field(default_factory=RestorationConfig)
    # Method-level: may ablation-gradient circuits carry inhibitor-role
    # members (negative-scored latents) alongside supports? Honoured by the
    # ig_mean and restoration modes; local mode's attribution util
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
    ig_steps: int = 10                # interpolation steps for ig_mean
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
        allowed = ["close", "random", "distant", "fused"]
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
        # "activation_gradient" removed: promoted to a top-level method
        # (ActivationGradientDiscovery). "ig_negctx" is cf-only.
        # "mask" = the posctx learned-mask mode (abl-mask): sparsest soft
        # membership reproducing natural firing (see LearnedMaskConfig). The
        # negctx-aware mask modes are cf-only, like ig_negctx.
        allowed = ["local", "ig_mean", "restoration", "ig_restoration", "mask"]
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
    # A negative is normally rejected on its POST-TOP-K seed value, which is
    # exactly 0 whenever the seed misses top-k — so a sequence where the seed
    # very nearly fired looks identical to one where it is silent. That
    # contaminates "close" negatives most, since they are the likeliest to
    # nearly fire, and is a candidate explanation for random beating close.
    # preact_filter measures relu(x @ w_seed + b_seed) instead (the uncensored
    # value) and rejects any candidate above preact_max_frac of the seed's
    # posctx reference — i.e. keep only sequences that are genuinely NOT
    # driving the seed, not merely ones where top-k hid that they were.
    preact_filter: bool = False
    # "cleanest" RANKS a bounded candidate pool by pre-top-k value and keeps
    # the quietest max_sequences � adaptive by construction, and the reason
    # an absolute bar was abandoned: contamination runs ~3% of the posctx
    # reference at L2 and ~28% at L10, so preact_max_frac=0.25 rejected
    # NOTHING at L2/L5/L8 while 0.10 would have rejected 100% of close
    # candidates at L10. "threshold" keeps the old absolute-bar behaviour.
    preact_select: str = "cleanest"       # cleanest | threshold
    preact_max_frac: float = 0.1          # threshold mode only
    preact_reference_stat: str = "median"  # median | mean
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

    @field_validator("preact_max_frac")
    @classmethod
    def validate_preact_max_frac(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"preact_max_frac must be in [0, 1), got {v}")
        return v

    @field_validator("preact_select")
    @classmethod
    def validate_preact_select(cls, v: str) -> str:
        if v not in ("cleanest", "threshold"):
            raise ValueError(
                f"preact_select must be 'cleanest' or 'threshold', got {v!r}")
        return v

    @field_validator("preact_reference_stat")
    @classmethod
    def validate_preact_reference_stat(cls, v: str) -> str:
        if v not in ("median", "mean"):
            raise ValueError(
                f"preact_reference_stat must be 'median' or 'mean', got {v!r}")
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
    # Sequence COUNT vs batch SIZE are separate ideas (the neg side has had
    # this split all along: max_neg_sequences vs neg_batch_size; the pos side
    # historically conflated them into probe_batch_size):
    #   *_sequence_count — how many probe-dataset positive sequences INFORM the
    #     result (statistical width; the probe dataset holds up to 64).
    #   *batch_size — how many sequences go through ONE forward pass (VRAM).
    # Counts above a batch size are processed in microbatches and merged, so
    # raising a count costs time, not memory.
    probe_sequence_count: int = 16   # pos sequences used by discovery attribution
    probe_batch_size: int = 16       # sequences per grad-enabled forward (VRAM-bound)
    eval_sequence_count: int = 16    # pos sequences used by the faithfulness evals
    eval_batch_size: int = 16        # sequences per no-grad eval forward
    # Shared floor knob for every consumer of mean-ablation floors
    # (ig_mean, restoration, ablation-faithfulness evals):
    # "posctx" = per-seed means over the seed's positive probe batch
    # (SFC's distribution-matched baseline); "global" = seed-independent
    # means over a sample of random corpus sequences (colder floor, no
    # evaluation-distribution leakage), cached once per process.
    # "diverse" = farthest-point sample over stored sequence representations
    # (coverage-weighted: a different floor semantics from "global"'s
    # density-weighted corpus expectation, not a variance-reduced estimate).
    # "negctx" = per-seed means over the seed's retrieved NEGATIVE contexts,
    # i.e. sequences chosen because the seed is silent. Removes seed-specific
    # content while keeping generic stream content, where "posctx" removes
    # nothing (it leaks 23-30% of a_pos into deep seeds before any circuit
    # exists) and "zero" removes everything. Requires the caller to thread
    # neg_tokens; see the ordering argument in eval/floors.py.
    floor_source: str = "posctx"     # "posctx" | "negctx" | "global" | "diverse"
    # Which negatives define floor_source="negctx". "store" (default) reuses
    # the probe dataset's negatives, i.e. the per-latent neg_ctx KNN store —
    # the nearest non-activating sequences, so close/hard negatives at no extra
    # cost. "close"/"random"/"distant" re-retrieve through the shared
    # negative-context selector, letting the FLOOR's negative hardness be varied
    # independently of a method's own neg_mode (which governs ig_negctx and
    # phi_cf, never the floor). Inert unless floor_source == "negctx".
    floor_negctx_mode: str = "store"  # "store" | "close" | "random" | "distant"
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
    #   "abs_pctl" -> "abs" whose cut is computed IN-RUN as the threshold-th
    #                 PERCENTILE (0-100) of the pass's pooled nonzero |attr|
    #                 across all sites — self-calibrating, so e.g. 90 means the
    #                 same admission bar on every seed even though raw
    #                 attribution scales differ by orders of magnitude. At full
    #                 sampling width (64 pos / 64 neg), p90 was ~size-neutral
    #                 vs top_n=96 while RAISING free closure by 0.06-0.21 (a
    #                 fixed per-position count clips heavy positions); p95
    #                 halves size for ~0.05 free0; p99 collapses (collective
    #                 closure). See the thresh64 experiment (2026-07-16).
    #   "relative" -> keep |attr| >= threshold * max_latent|attr| at that
    #                 position (scale-free fraction, threshold in [0, 1]).
    #   "mass"     -> keep the smallest set covering `threshold` of that
    #                 position's total |attr| mass (cumulative, threshold in (0, 1]).
    position_aware_select: str = "top_n"   # "top_n" | "abs" | "abs_pctl" | "relative" | "mass"
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
    # Cross-sequence recurrence prune (post-assembly, runs BEFORE the magnitude
    # prune). Drops members that fire in fewer than `min_sequences` of the probe
    # sequences — the PA union's one-sequence tail, which is context-specific
    # rather than mechanism. Role-split: supports/activators are judged on
    # posctx, inhibitors on negctx (a member that suppresses the seed is only
    # observable where the seed is absent); inhibitors are exempt when no negctx
    # is available rather than being dropped unseen.
    recurrence_prune: bool = False
    recurrence_prune_min_sequences: int = 2   # rec2 is the validated deep default
    recurrence_prune_min_keep: int = 1        # never prune below this many members

    @field_validator("recurrence_prune_min_sequences")
    @classmethod
    def validate_recurrence_prune_min_sequences(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"recurrence_prune_min_sequences must be >= 1, got {v}")
        return v

    @field_validator("recurrence_prune_min_keep")
    @classmethod
    def validate_recurrence_prune_min_keep(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"recurrence_prune_min_keep must be >= 1, got {v}")
        return v

    @field_validator("floor_source")
    @classmethod
    def validate_floor_source(cls, v: str) -> str:
        # "zero": every floor consumer attributes/restores against the
        # ZERO-ablation counterfactual (the one free0 evaluates) instead of a
        # mean. Makes ig_mean the free0-coherent "integrated activation
        # gradient" (0 -> natural) and restoration a greedy free0 climber.
        # "negctx": the on-manifold counterpart of "zero" — it reaches the same
        # clean denominator (measured a_empty == 0.0) from a real non-firing
        # state, and makes ig_mean integrate along exactly the contrast the
        # eval floor scores against.
        allowed = ["posctx", "negctx", "global", "diverse", "zero"]
        if v not in allowed:
            raise ValueError(f"floor_source must be one of {allowed}, got {v!r}")
        return v

    @field_validator("floor_negctx_mode")
    @classmethod
    def validate_floor_negctx_mode(cls, v: str) -> str:
        allowed = ["store", "close", "random", "distant"]
        if v not in allowed:
            raise ValueError(f"floor_negctx_mode must be one of {allowed}, got {v!r}")
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
        allowed = ["top_n", "abs", "abs_pctl", "relative", "mass"]
        if v not in allowed:
            raise ValueError(f"position_aware_select must be one of {allowed}, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_abs_pctl_threshold(self) -> "DiscoveryConfig":
        if self.position_aware_select == "abs_pctl" and not (0 < self.position_aware_threshold < 100):
            raise ValueError(
                "position_aware_select='abs_pctl' needs position_aware_threshold as a "
                f"percentile in (0, 100), got {self.position_aware_threshold}"
            )
        return self

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
    learned_mask: LearnedMaskConfig = Field(default_factory=LearnedMaskConfig)
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
