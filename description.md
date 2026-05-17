# Repository Description

## Overview

This repository is a Python-first transformer interpretability project centered on TuringLLM and a bank of sparse autoencoders (SAEs). Its main job is to discover, evaluate, and analyze minimal latent circuits: small sub-networks of SAE features that reproduce a target behavior or concept.

At a high level, the codebase runs a multi-pass pipeline: it loads model and SAE weights, scans dataset shards to collect latent statistics and context examples, builds negative contexts and co-activation graphs, selects promising seed latents, applies multiple circuit-discovery algorithms, then scores the resulting circuits with faithfulness-style evaluation metrics and post-hoc analyses.

## Major Areas

- `src/model`: TuringLLM model code, inference helpers, hooks, and tokenization.
- `src/sae`: SAE bank management, accelerated Top-K sparsification, and fused kernels/helpers.
- `src/pipeline`: The staged discovery pipeline, runtime setup, persistence, and orchestration.
- `src/store`: Persistent and in-memory stores for statistics, contexts, co-activations, and circuits.
- `src/circuit`: Circuit types, instrumentation, discovery methods, probe datasets, feature selection, and post-analysis.
- `src/eval`: Faithfulness, sufficiency, completeness, minimality, and counterfactual evaluation.
- `src/display` and `src/observability`: Terminal presentation, logging, timing, and progress tracking.
- `src/native`: Build/test glue for native CUDA/C++ acceleration kernels such as latent stats, reservoir sampling, co-activation reduction, and fused linear-ReLU.

## Entry Points

- `src/main.py`: Full pipeline entry point.
- `src/discover_circuits.py`: Discovery-only CLI over existing pipeline outputs.
- `src/display_latents.py`: Interactive latent inspection CLI.
- `src/search_latents.py`: Keyword search and optional patch-clamp flow over stored latent contexts.
- `src/ablation_sensitivity.py`: Hyperparameter sweep / ablation utility for discovery settings.

## Scope Of The Reference

This document inventories the production Python callable surface of the repository: `src/**/*.py`. Native `.cu` and `.cpp` kernels are described at subsystem level rather than enumerated function-by-function here.

Method and class descriptions prefer source docstrings when the code provides them; otherwise they fall back to concise name-based summaries so the inventory stays complete.

## Inventory Counts

- Python files covered: `109`
- Top-level functions covered: `150`
- Classes covered: `111`
- Class methods covered: `364`

## Python Module And Method Reference

### `src`

#### `src/ablation_sensitivity.py`

This module focuses on ablation sensitivity.

Top-level functions:
- `_try_load(ctx, path, name)` - Load a context store if the file exists; print a notice otherwise.
- `parse_latent_input(raw)` - Parse '<layer> <kind> <latent>' from a string.
- `main()` - Handles main.

Classes and methods:
- `AblationSensitivityTool` - Represents AblationSensitivityTool.
  - `__init__(self, device)` - Initializes the instance.
  - `discover_candidates(self, layer_idx, kind, latent_idx)` - Candidate Discovery using Attribution, Co-activation, and Frequency Analysis.
  - `discover_frequent_latents(self, tokens)` - Find latents that fire most frequently during the seed's sequences.
  - `_capture_acts(self, tokens, pos_argmax, fids, patcher?)` - Capture activations for a list of FIDs at specific positions.
  - `run_sensitivity_sweep(self, layer_idx, kind, latent_idx, candidates)` - Measuring Causal Sensitivity across the neighborhood.

#### `src/config.py`

This module focuses on config.

Top-level functions:
- `_resolve_path(val)` - Helper to resolve relative paths in the config relative to PROJECT_ROOT.
- `load_config()` - Loads config.

Classes and methods:
- `WeightsConfig` - Represents WeightsConfig.
  - `resolve_paths(cls, v)` - Handles resolve paths.
- `DataConfig` - Represents DataConfig.
  - `resolve_paths(cls, v)` - Handles resolve paths.
- `HardwareConfig` - Represents HardwareConfig.
- `TopCtxConfig` - Represents TopCtxConfig.
- `MidCtxConfig` - Represents MidCtxConfig.
- `NegCtxConfig` - Represents NegCtxConfig.
- `LogitCtxConfig` - Represents LogitCtxConfig.
- `SeqLatentIndexConfig` - Represents SeqLatentIndexConfig.
- `TopCoactivationLatentsConfig` - Represents TopCoactivationLatentsConfig.
  - `validate_mode(cls, v)` - Validates mode.
- `LatentsConfig` - Represents LatentsConfig.
- `CoactivationStatisticalConfig` - Represents CoactivationStatisticalConfig.
- `LogitAttributionConfig` - Represents LogitAttributionConfig.
- `SFCAttributionPatchingConfig` - Represents SFCAttributionPatchingConfig.
- `NeighborhoodExpansionConfig` - Represents NeighborhoodExpansionConfig.
- `SparseExpansionConfig` - Represents SparseExpansionConfig.
- `HardNegativeCoactSparseExpansionConfig` - Represents HardNegativeCoactSparseExpansionConfig.
- `DifferentialActivationConfig` - Represents DifferentialActivationConfig.
- `GradientUpstreamConfig` - Represents GradientUpstreamConfig.
- `LayerwiseGradientUpstreamConfig` - Represents LayerwiseGradientUpstreamConfig.
- `TopCoactAttrDiscoveryConfig` - Represents TopCoactAttrDiscoveryConfig.
- `CounterfactualGradientConfig` - Represents CounterfactualGradientConfig.
  - `validate_neg_mode(cls, v)` - Validates neg mode.
- `CircuitTracerBaselineConfig` - Represents CircuitTracerBaselineConfig.
- `ClusterContrastConfig` - Represents ClusterContrastConfig.
  - `validate_eval_position(cls, v)` - Validates eval position.
  - `validate_top_k_scope(cls, v)` - Validates top k scope.
- `SeedFilterConfig` - Constrains which seeds CandidateSelector returns by layer and/or kind.
  - `validate_kinds(cls, v)` - Validates kinds.
- `DiscoveryConfig` - Represents DiscoveryConfig.
- `PersistConfig` - Represents PersistConfig.
- `AnalysisConfig` - Represents AnalysisConfig.
- `RootConfig` - Represents RootConfig.

#### `src/discover_circuits.py`

This module focuses on discover circuits.

Top-level functions:
- `discover_circuits(candidates_path?, reselect?, n_seeds?)` - Discovers circuits.

#### `src/display_latents.py`

Terminal display helpers for circuit summaries, latent views, and candidate inspection. This module focuses on display latents.

Top-level functions:
- `_try_load(ctx, path, name)` - Load a context store if the file exists; print a notice otherwise.
- `parse_latent_input(raw)` - Parse '<layer> <kind> <latent>' from a string.
- `analyze(model, bank, loader, layer_idx, kind, latent_idx, n_sequences)` - Analyzes the requested value.
- `main()` - Handles main.

#### `src/hardware.py`

This module focuses on hardware.

Top-level functions:
- `detect_devices()` - Auto-detects available CUDA devices based on config.hardware.multi_gpu.
- `get_primary_device()` - Returns primary device.
- `is_fast_memory()` - Checks whether the object is fast memory.
- `should_compile()` - Handles should compile.
- `is_multi_gpu()` - Checks whether the object is multi gpu.

#### `src/main.py`

This module focuses on main.

Top-level functions:
- `main()` - Handles main.

#### `src/search_latents.py`

This module focuses on search latents.

Top-level functions:
- `get_latent_avg_activations(model, bank, loader, target_latents)` - Calculates the average of the maximum activation values for each target latent across its top sequences.
- `run_search(args, model, bank, loader, top_ctx, df, query_str, device)` - Runs search.
- `main()` - Handles main.

### `src/circuit`

#### `src/circuit/discovery_window.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on discovery window.

Top-level functions:
- `_build_methods(inference, bank, avg_acts, probe_builder)` - Instantiates all discovery methods listed in config.discovery.methods.
- `run_discovery_window(inference, bank, loader, candidates_path?)` - Entry point to run a discovery window from saved candidates.

Classes and methods:
- `DiscoveryWindow` - Orchestrates circuit discovery for a list of seed candidates.
  - `__init__(self, inference, bank, loader, output_dir?)` - Initializes the instance.
  - `run(self, candidates, save_interval?)` - Runs all discovery methods for each seed candidate.
  - `_run_node_presence_eval(self, circuit)` - Runs the posctx node-presence evaluation for CounterfactualGradientDiscovery circuits and merges results into circuit.metadata.
  - `_consolidate_evals(self, circuit)` - Moves top-level eval scores into the nested ``circuit.metadata["evals"]`` dict so all evaluation results live in one place in the saved JSON.
  - `_print_eval_stats_table(self)` - Prints mean and variance for every eval field across all discovered circuits.
  - `_print_summary_table(self)` - Prints a Rich-formatted table of all discovered circuits sorted by faithfulness.
  - `_print_eval_stats_table(self)` - Prints Rich tables of aggregate stats for evals, post-analysis, and seed criteria.
  - `_build_stats_table(title, metrics, circuits)` - Builds and returns a Rich Table of per-metric aggregate statistics.
  - `save_store(self)` - Persists the circuit store to disk.
  - `_save_summary(self)` - Saves a JSON summary of all discovered circuits.
  - `_flatten(d, prefix?)` - Recursively flattens a nested dict using dot-separated keys.
  - `_save_summary_xlsx(self)` - Saves a flat Excel summary of all discovered circuits (one row per circuit).
  - `_write_correlation_sheet(writer, df)` - Writes a correlation matrix of all numeric columns to a second sheet.

#### `src/circuit/feature_selection.py`

Circuit-specific orchestration, feature selection, and probe dataset preparation. This module focuses on feature selection.

Top-level functions:
- `_seed_passes_filter(candidate, filter_layers, filter_kinds)` - Return True if the candidate satisfies the active layer/kind allowlists.

Classes and methods:
- `CandidateSelector` - Identifies 'Seed Latents' for global circuit discovery.
  - `__init__(self, n_seeds?, device?)` - Initializes the instance.
  - `select_candidates(self)` - Runs all enabled scoring criteria and returns a merged, deduplicated list of candidate seed latents sorted by combined score.
  - `_top_k(self, score_tensor, reason)` - Return top-n_seeds (comp_idx, latent_idx) entries from a [C, D] score tensor.
  - `_top_k_indices(self, score_tensor, k, reason)` - Handles top k indices.
  - `get_summary_stats(self, candidates)` - Prints a breakdown of selected candidates by layer and kind.

#### `src/circuit/probe_dataset.py`

Circuit-specific orchestration, feature selection, and probe dataset preparation. This module focuses on probe dataset.

Classes and methods:
- `ProbeDataset` - A high-contrast dataset for evaluating a circuit around a seed latent.
- `ProbeDatasetBuilder` - Represents ProbeDatasetBuilder.
  - `__init__(self, inference, bank, loader)` - Args: inference: The Inference instance to run the model.
  - `build_for_latent(self, comp_idx, latent_idx, top_ctx, mid_ctx, neg_ctx, n_pos?, n_neg?)` - Builds a ProbeDataset for a specific latent by gathering tokens and finding argmax.
  - `build_pos_tokens(self, comp_idx, latent_idx, top_ctx, mid_ctx, n_pos?)` - Returns the positive token tensor ``[N, 64]`` for a seed latent without running any model forward pass (pure data lookup from top_ctx / mid_ctx).
  - `_load_all_ids(self, ids, max_length?)` - Helper to load a list of sequence IDs into a single tensor.
  - `_calculate_argmax(self, comp_idx, latent_idx, tokens)` - Runs the model and SAE encoder to find the token position of peak activation.

### `src/circuit/analysis`

#### `src/circuit/analysis/__init__.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on   init  .

This module does not define Python functions or classes directly.

#### `src/circuit/analysis/base.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on base.

Classes and methods:
- `AnalysisContext` - Shared resources passed to every post-circuit analysis.
- `CircuitAnalysis` - Abstract base class for post-circuit analysis methods.
  - `analyse(self, circuit, context)` - Analyse a discovered circuit and return a dict of metadata key/value pairs to merge into ``circuit.metadata``.

#### `src/circuit/analysis/coactivation_overlap.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on coactivation overlap.

Classes and methods:
- `CoactivationOverlapAnalysis` - Computes what percentage of a circuit's SAE feature nodes also appear in the seed latent's stored top-K co-activating latents.
  - `analyse(self, circuit, context)` - Handles analyse.
  - `_run(self, circuit, context)` - Runs the requested value.

#### `src/circuit/analysis/edge_weight_gini.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on edge weight gini.

Classes and methods:
- `EdgeWeightGiniAnalysis` - Computes the Gini coefficient of the circuit's absolute edge weights.
  - `analyse(self, circuit, context)` - Handles analyse.
  - `_run(self, circuit, context)` - Runs the requested value.

#### `src/circuit/analysis/internode_coact_density.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on internode coact density.

Classes and methods:
- `InternodeCoactDensityAnalysis` - Computes what percentage of unordered node pairs in the circuit are *mutually* co-activating: A's global ID appears in B's top-K coact list AND B's global ID appears in A's top-K coact list.
  - `analyse(self, circuit, context)` - Handles analyse.
  - `_run(self, circuit, context)` - Runs the requested value.

#### `src/circuit/analysis/layer_distribution.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on layer distribution.

Classes and methods:
- `LayerDistributionAnalysis` - Computes the distribution of circuit feature nodes across transformer layers.
  - `analyse(self, circuit, context)` - Handles analyse.
  - `_run(self, circuit, context)` - Runs the requested value.

#### `src/circuit/analysis/node_activity.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on node activity.

Classes and methods:
- `NodeActivityAnalysis` - Computes the distribution of lifetime firing counts for circuit feature nodes, using ``latent_stats.active_count``.
  - `analyse(self, circuit, context)` - Handles analyse.
  - `_run(self, circuit, context)` - Runs the requested value.

#### `src/circuit/analysis/node_rarity.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on node rarity.

Classes and methods:
- `NodeRarityAnalysis` - Computes the fraction of circuit feature nodes whose lifetime firing count falls at or below the global 10th percentile of ``latent_stats.active_count``.
  - `analyse(self, circuit, context)` - Handles analyse.
  - `_run(self, circuit, context)` - Runs the requested value.

#### `src/circuit/analysis/runner.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on runner.

Top-level functions:
- `build_analyses(context)` - Instantiates all post-circuit analysis methods listed in ``config.analysis.methods``.
- `run_post_circuit_analyses(circuit, context, analyses)` - Runs all analyses in order and merges their results into ``circuit.metadata["post_analysis"]``.

#### `src/circuit/analysis/top_token_consistency.py`

Post-discovery analysis passes that compute descriptive metrics for accepted circuits. This module focuses on top token consistency.

Classes and methods:
- `TopTokenConsistencyAnalysis` - Computes what percentage of circuit feature nodes share the seed latent's top-1 predicted output token, as stored in ``logit_ctx``.
  - `analyse(self, circuit, context)` - Handles analyse.
  - `_run(self, circuit, context)` - Runs the requested value.

### `src/circuit/discovery`

#### `src/circuit/discovery/__init__.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on   init  .

This module does not define Python functions or classes directly.

#### `src/circuit/discovery/base.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on base.

Classes and methods:
- `DiscoveryMethod` - Abstract base class for all circuit discovery algorithms.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder)` - Args: inference: The Inference instance for running the LLM.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Executes the discovery algorithm starting from a seed latent.
  - `build_probe_dataset(self, comp_idx, latent_idx, n_pos?, n_neg?)` - Helper to build a probe dataset for a latent using the injected builder.

#### `src/circuit/discovery/circuit_tracer_baseline.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on circuit tracer baseline.

Classes and methods:
- `CircuitTracerBaseline` - SAE-adapted Circuit Tracing baseline, aligned with Anthropic's Attribution Graphs method.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, probe_batch_size?, max_sequences?, target_chunk_size?, logit_top_k?, desired_logit_prob?, influence_max_iter?, node_threshold?, edge_threshold?, min_faithfulness?, pruning_threshold?, min_active_count?, max_feature_nodes?, stop_error_grad?, include_error_nodes?, online_ranking_interval?, feature_batch_size?, include_token_nodes?)` - Initializes the instance.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Discovers the requested value.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.

#### `src/circuit/discovery/cluster_contrast.py`

Cluster Contrast Discovery: seed-free circuit discovery via embedding-space clustering.

Top-level functions:
- `collect_neg_seq_ids(neg_ctx)` - Flatten the entire neg_ctx store and return deduplicated, non-zero sequence IDs.
- `kmeans_cosine(norm, k, n_iters?, chunk?, seed?, device?)` - Cosine k-means on pre-normalised embeddings.

Classes and methods:
- `ClusterResult` - Output of a single k-means run over neg_ctx sequence embeddings.
- `ClusterContrastDiscovery` - Seed-free circuit discovery using embedding-space cluster contrast.
  - `__init__(self, inference, sae_bank, loader)` - Initializes the instance.
  - `_load_tokens(self, seq_ids, max_length?)` - Load and pad a list of sequence IDs into a [N, max_length] long tensor.
  - `_compute_target_logits(self, center_tokens)` - Runs no-grad forward passes (in batches) on center sequences and returns the mean logit vector over all sequences.
  - `_run_cf_hop(self, cf_tokens, target_logits, logger)` - Grad-enabled forward on counterfactual sequences → KL loss → backward.
  - `discover_cluster(self, cluster_id, center_seq_ids, cf_seq_ids, cluster_size)` - Runs discovery for a single cluster.
  - `_discover_cluster(self, cluster_id, center_seq_ids, cf_seq_ids, cluster_size, logger)` - Discovers cluster.

#### `src/circuit/discovery/coactivation_statistical.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on coactivation statistical.

Classes and methods:
- `CoactivationStatistical` - Baseline circuit discovery using raw co-activation statistics.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, min_faithfulness?, coactivation_threshold?, max_neighbors?, min_active_count?, pruning_threshold?)` - Initializes the instance.
  - `_get_default_threshold(self, cfg)` - Returns the appropriate threshold based on the co-activation mode.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Expands seed to co-activation neighbors above threshold, then evaluates faithfulness.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.

#### `src/circuit/discovery/counterfactual_gradient.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on counterfactual gradient.

Classes and methods:
- `SeedProjectionInstrument` - SAEGraphInstrument subclass that captures the seed latent's encoder pre-activation during the forward pass.
  - `__init__(self, bank, seed_layer, seed_kind, w_seed, b_seed)` - Initializes the instance.
  - `transform(self, layer_idx, kind, x)` - Handles transform.
- `CounterfactualGradientDiscovery` - Discovers circuit nodes by running gradient attribution on contrast sequences — inputs where the seed latent is inactive.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, top_k_activators?, top_k_inhibitors?, top_k_scope?, activator_threshold?, inhibitor_threshold?, min_active_count?, max_neg_sequences?, pruning_threshold?, min_faithfulness?)` - Initializes the instance.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Discovers the requested value.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.
  - `_get_neg_tokens(self, probe_data, seed_comp_idx, seed_latent_idx, pos_tokens_eval, pos_argmax_eval, logger)` - Returns the contrast token batch ``[N, 64]`` for the gradient attribution pass, according to ``self.neg_mode``: - ``"close"`` — hard negatives from neg_ctx (up to ``max_neg_sequences``).
  - `_get_posctx_sae_mean(self, seed_comp_idx, seed_latent_idx, pos_tokens_eval, pos_argmax_eval)` - Runs a no-grad forward on pos_tokens_eval and returns a ``[d_sae]`` float tensor representing the mean SAE activation at ``(seed_layer, seed_kind)`` evaluated at each sequence's ``pos_argmax`` position, averaged over the batch.
  - `_get_distant_tokens(self, seed_comp_idx, seed_latent_idx, pos_tokens_eval, pos_argmax_eval, logger)` - Implements ``neg_mode="distant"``: samples ``distant_pool_size`` sequences from the full corpus, filters to those where the seed never activates, then returns the ``max_neg_sequences`` most distant from posctx in SAE latent space at ``(seed_layer, seed_kind)``.
  - `_get_posctx_activation(self, seed_comp_idx, seed_latent_idx, pos_tokens, pos_argmax)` - Runs a no-grad forward on pos_tokens and returns the seed latent's mean SAE activation at the pos_argmax positions — used as target_act_pos for the MSE loss.
  - `_run_contrast_hop(self, seed_comp_idx, seed_latent_idx, neg_tokens, target_act_pos, logger)` - Runs grad-enabled forward passes on the contrast sequences using SeedProjectionInstrument, then calls compute_latent_counterfactual_scores to extract absent activators and present inhibitors.

#### `src/circuit/discovery/differential_activation.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on differential activation.

Classes and methods:
- `DifferentialActivation` - Discovers circuits by finding latents that are differentially active between positive contexts (seed fires) and hard-negative contexts (seed expected but absent).
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, n_activator_candidates?, n_inhibitor_candidates?, attribution_threshold?, pruning_threshold?, min_faithfulness?, probe_batch_size?)` - Initializes the instance.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Discovers the requested value.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.
  - `_collect_activations(self, tokens)` - No-grad forward pass collecting total activation per latent.

#### `src/circuit/discovery/gradient_upstream.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on gradient upstream.

Classes and methods:
- `GradientUpstreamDiscovery` - Discovers circuits by propagating gradient attribution backwards through the model.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, depth?, top_k_per_hop?, attribution_threshold?, min_active_count?, max_ctx_sequences?, hop_batch_size?, absent_inhibitor_top_k?, absent_inhibitor_threshold?, pruning_threshold?, min_faithfulness?)` - Initializes the instance.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Backwards gradient BFS with context switching.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.
  - `_run_hop(self, comp_idx, latent_idx, tokens, logger)` - Runs grad-enabled forward passes in microbatches and accumulates upstream scores.

#### `src/circuit/discovery/layerwise_gradient_upstream.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on layerwise gradient upstream.

Classes and methods:
- `LayerwiseGradientUpstreamDiscovery` - Discovers circuits by sweeping layer-by-layer backwards through the model.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, top_k_per_node?, attribution_threshold?, min_active_count?, max_ctx_sequences?, hop_batch_size?, absent_inhibitor_top_k?, absent_inhibitor_threshold?, max_layers_back?, include_same_layer?, pruning_threshold?, min_faithfulness?, profile_first_node?)` - Initializes the instance.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Layer-by-layer gradient sweep with full upstream attribution.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.
  - `_run_node_profiled(self, comp_idx, latent_idx, tokens, all_upstream_comps, logger, fid)` - Identical to _run_node but wrapped in torch.profiler for one-shot diagnostics.
  - `_run_node(self, comp_idx, latent_idx, tokens, all_upstream_comps, logger)` - Runs grad-enabled forward passes in microbatches and accumulates upstream scores.

#### `src/circuit/discovery/logit_attribution.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on logit attribution.

Classes and methods:
- `LogitAttribution` - Two-pass gradient-based circuit discovery.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, min_faithfulness?, logit_threshold?, edge_threshold?, max_neighbors?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Two-pass logit attribution discovery.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.

#### `src/circuit/discovery/neighborhood_expansion.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on neighborhood expansion.

Classes and methods:
- `NeighborhoodExpansion` - Structural circuit discovery via two-hop co-activation neighborhood expansion.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, min_faithfulness?, n_expand?, m_neighbors?, min_active_count?, pruning_threshold?)` - Initializes the instance.
  - `_causal_before(self, a, b)` - Return True if component A is causally upstream of component B.
  - `_add_node_if_new(self, circuit, node_id_map, fid, role)` - Add a node for FeatureID if it passes the activity filter and hasn't been added yet.
  - `_add_edge(self, circuit, node_id_map, fid_a, fid_b, weight)` - Add a directed edge between two already-registered nodes, respecting causal order (earlier → later).
  - `_expand_neighbors(self, fid, limit, exclude)` - Yield (neighbor_fid, weight) for the top ``limit`` co-activation neighbors of fid that pass the activity filter and are not in ``exclude``.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Build a two-hop co-activation neighbourhood circuit from the seed feature.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.

#### `src/circuit/discovery/sfc_attribution_patching.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on sfc attribution patching.

Top-level functions:
- `_vram_audit(label)` - Prints allocated VRAM and the 10 largest live CUDA tensors.

Classes and methods:
- `TopKState` - Memory-efficient sparse SAE state.
  - `device(self)` - Handles device.
  - `zeros_like(self)` - Returns a zeroed state with the same index structure.
  - `to_sparse_act(self, d_sae)` - Expands sparse [B, T, k] representation to dense [B, T, d_sae] SparseAct.
- `SingleSubmodPatcher` - Patches only a single (layer, kind) submodule with interpolated SAE features for per-submodule Integrated Gradients, matching Marks et al.
  - `__init__(self, bank, target_lk, f_act, f_res)` - Initializes the instance.
  - `__call__(self, model)` - Invokes the instance as a callable.
- `SFCAttributionPatching` - Sparse Feature Circuits (Marks et al.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, node_threshold?, edge_threshold?, patch_mode?, max_neg?, pruning_threshold?, probe_batch_size?, min_faithfulness?, ig_steps?)` - Initializes the instance.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Discovers the requested value.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.
  - `_add_jvp_edges(self, instrument, grads, deltas, circuit, node_id_map, resid_node_id_map, active_latents, upstream, downstream, stop_grads)` - Computes JVP edge attributions for a single (upstream → downstream) pair.
  - `_get_all_states(self, tokens)` - Returns all states.
  - `_pe_ig(self, clean_states, patch_states, tokens, argmax, targets)` - Per-submodule Integrated Gradients, matching Marks et al.

#### `src/circuit/discovery/top_coact_attr.py`

Circuit discovery methods, shared abstractions, and seed-level discovery orchestration helpers. This module focuses on top coact attr.

Classes and methods:
- `TopCoactAttrDiscovery` - Discovers circuits by expanding the neighborhood of a seed latent using statistical co-activation data, followed by multi-hop causal attribution.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, min_faithfulness?, max_neighbors?, max_hops?, min_active_count?, attribution_threshold?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Executes the multi-hop causal attribution discovery episode.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.

### `src/circuit/discovery/top_coact_expansion`

#### `src/circuit/discovery/top_coact_expansion/__init__.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on   init  .

This module does not define Python functions or classes directly.

#### `src/circuit/discovery/top_coact_expansion/all_top_coact_sparse_expansion.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on all top coact sparse expansion.

Classes and methods:
- `AllTopCoactSparseExpansion` - All-kinds variable-depth top-coactivation sparse expansion with no passthrough stage.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, coact_depth?, min_faithfulness?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.

#### `src/circuit/discovery/top_coact_expansion/attn_mlp_top_coact_sparse_expansion.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on attn mlp top coact sparse expansion.

Classes and methods:
- `AttnMlpTopCoactSparseExpansion` - Attn+MLP-targeted variable-depth top-coactivation sparse expansion with resid passthrough.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, coact_depth?, min_faithfulness?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.

#### `src/circuit/discovery/top_coact_expansion/attn_resid_top_coact_sparse_expansion.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on attn resid top coact sparse expansion.

Classes and methods:
- `AttnResidTopCoactSparseExpansion` - Attn+resid-targeted variable-depth top-coactivation sparse expansion with mlp passthrough.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, coact_depth?, min_faithfulness?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.

#### `src/circuit/discovery/top_coact_expansion/attn_top_coact_sparse_expansion.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on attn top coact sparse expansion.

Classes and methods:
- `AttnTopCoactSparseExpansion` - Attn-targeted variable-depth top-coactivation sparse expansion with full MLP/resid passthrough.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, coact_depth?, min_faithfulness?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.

#### `src/circuit/discovery/top_coact_expansion/hard_negative_coact_sparse_expansion.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on hard negative coact sparse expansion.

Classes and methods:
- `HardNegativeCoactSparseExpansion` - Expansion discovery method that identifies inhibitors by finding latents unusually active in hard-negative contexts of the seed, validated by attribution.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, coact_depth?, neg_candidate_limit?, attribution_threshold?, min_faithfulness?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.
  - `_collect_neg_activations(self, neg_tokens)` - Runs a no-grad forward pass on neg_tokens to find active latents.

#### `src/circuit/discovery/top_coact_expansion/mlp_resid_top_coact_sparse_expansion.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on mlp resid top coact sparse expansion.

Classes and methods:
- `MlpResidTopCoactSparseExpansion` - MLP+resid-targeted variable-depth top-coactivation sparse expansion with attn passthrough.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, coact_depth?, min_faithfulness?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.

#### `src/circuit/discovery/top_coact_expansion/mlp_top_coact_sparse_expansion.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on mlp top coact sparse expansion.

Classes and methods:
- `MlpTopCoactSparseExpansion` - MLP-targeted variable-depth top-coactivation sparse expansion with full attn/resid passthrough.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, coact_depth?, min_faithfulness?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.

#### `src/circuit/discovery/top_coact_expansion/resid_top_coact_sparse_expansion.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on resid top coact sparse expansion.

Classes and methods:
- `ResidTopCoactSparseExpansion` - Resid-targeted variable-depth top-coactivation sparse expansion with full attn/MLP passthrough.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, coact_depth?, min_faithfulness?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.

#### `src/circuit/discovery/top_coact_expansion/top_coact_sparse_expansion.py`

Sparse expansion discovery methods that grow circuits through co-activation neighborhoods. This module focuses on top coact sparse expansion.

Classes and methods:
- `TopCoactSparseExpansion` - Base class for targeted variable-depth top-coactivation sparse expansion algorithms.
  - `__init__(self, inference, sae_bank, avg_acts, probe_builder, target_kinds, passthrough_kinds, method_name, config_node, coact_depth?, min_faithfulness?, min_active_count?, pruning_threshold?, probe_batch_size?)` - Initializes the instance.
  - `_expand_neighbors(self, fid, limit, exclude)` - Yield (neighbor_fid, weight) for the top ``limit`` co-activation neighbors of fid.
  - `_capture_passthrough_nodes(self, probe_tokens)` - Run a no-grad forward pass and collect FeatureIDs for components in self.passthrough_kinds.
  - `discover(self, seed_comp_idx, seed_latent_idx)` - Entry point for discovery.
  - `_discover(self, seed_comp_idx, seed_latent_idx, logger)` - Discovers the requested value.

### `src/circuit/instrument`

#### `src/circuit/instrument/__init__.py`

Instrumentation, attribution, patching, and intervention utilities over SAE-augmented model graphs. This module focuses on   init  .

This module does not define Python functions or classes directly.

#### `src/circuit/instrument/attribution.py`

Instrumentation, attribution, patching, and intervention utilities over SAE-augmented model graphs. This module focuses on attribution.

Top-level functions:
- `compute_logit_attribution(graph, logits, pos_argmax, target_tokens)` - Pass 1 — Logit-based attribution.
- `compute_feature_attribution(graph, target_layer, target_kind, target_latent_idx, pos_argmax, candidate_nodes?)` - Pass 2 — Feature-to-feature attribution.
- `compute_feature_gradient(graph, target_layer, target_kind, target_latent_idx, pos_argmax, candidate_nodes)` - Returns the raw gradient d(TargetAct)/d(CandidateAct) rather than Act * Grad.
- `compute_latent_counterfactual_scores(graph, target_scalar, seed_layer, n_kinds, kinds, top_k_activators, top_k_inhibitors, min_active_count, active_count, top_k_scope?)` - Single-backward-pass counterfactual scoring for negctx sequences.
- `collect_active_feature_nodes(graph, kinds, n_kinds, min_active_count?, active_count?, max_feature_nodes?)` - Returns a deduplicated list of FeatureIDs for every latent that fired in at least one (layer, kind) of the forward pass, ranked by peak activation magnitude and optionally capped at max_feature_nodes.
- `_find_logit_targets(inference, tokens, logit_top_k, desired_logit_prob?)` - Selects logit target tokens by cumulative softmax probability, matching Anthropic's two-stage logit-target selection from attribute_transformerlens.py.
- `_upstream_anchors_for_target(graph, tgt_layer, tgt_kind, kinds, n_kinds)` - Collects leaf anchor tensors for all (layer, kind) pairs causally upstream of (tgt_layer, tgt_kind), respecting the per-layer ordering defined by get_all_upstream_components (attn → mlp → resid within a layer, plus all strictly preceding layers).
- `_all_feature_anchors_with_meta(graph)` - Collects ALL leaf anchor tensors from the graph.
- `_score_grads_into_adj(grads, anchor_meta, graph, feature_nodes, node_to_idx, tgt_idx, adj)` - Extracts direct-effect scores for each source node and accumulates them into adj[(src_idx, tgt_idx)].
- `_score_token_grads_into_adj(grad_emb, emb_anchor, token_nodes, node_to_idx, tgt_idx, adj)` - Scores each input-token position's direct contribution to `tgt_idx` and accumulates results into `adj`.
- `_compute_partial_one_hop_influence(partial_adj, n_feature_nodes, n_error_nodes, n_logit_nodes, logit_probs)` - Computes a sparse one-hop logit-influence score for each feature node.
- `compute_direct_effects_matrix(tokens, inference, bank, logit_top_k, probe_batch_size, kinds, n_kinds, min_active_count?, active_count?, max_feature_nodes?, stop_error_grad?, target_chunk_size?, desired_logit_prob?, include_error_nodes?, online_ranking_interval?, feature_batch_size?, include_token_nodes?)` - Builds a sparse prompt-local direct-effects adjacency matrix over active SAE latents (attn/mlp/resid across all layers), plus logit sentinel nodes.
- `compute_latent_upstream_scores(graph, target_layer, target_kind, target_latent_idx, pos_argmax, predecessor_comp_indices, n_kinds, kinds, top_k, min_active_count, active_count, absent_inhibitor_top_k?, absent_inhibitor_threshold?)` - Vectorised predecessor scoring for node discovery.

Classes and methods:
- `UpstreamScores` - Scores produced by a single backward pass in compute_latent_upstream_scores.

#### `src/circuit/instrument/ct_influence.py`

ct_influence.py — Influence propagation and graph pruning for the circuit-tracer baseline.

Top-level functions:
- `compute_ct_influence(adj, all_nodes, logit_top_k, max_iter?, logit_probabilities?)` - Propagates influence backward from logit root nodes through the direct-effects adjacency matrix using a truncated Neumann series: influence = logit_weights @ (A_norm + A_norm² + A_norm³ + …) This is the SAE analogue of the influence computation in Anthropic's Attribution Graphs paper (transformer-circuits.pub/2025/attribution-graphs/methods.html §3.3).
- `_find_threshold(scores, fraction)` - Returns the minimum score cutpoint such that nodes/edges with score >= cutpoint collectively account for at least `fraction` of the total score mass.
- `_compute_partial_neumann_influence(adj, n_nodes, logit_weights, max_iter?)` - Computes a full Neumann-series influence vector from a partial (growing) adj dict.
- `_compute_edge_influence(A_pruned, logit_weights, max_iter?)` - Scores each directed edge by influence flowing through the PRUNED subgraph.
- `prune_ct_graph(adj, all_nodes, influence, node_threshold, edge_threshold, logit_top_k, logit_probabilities?, max_iter?)` - Prunes the direct-effects graph using scale-invariant fraction-based thresholds, matching Anthropic's circuit-tracer prune_graph algorithm.

#### `src/circuit/instrument/neg_ctx_baseline.py`

Instrumentation, attribution, patching, and intervention utilities over SAE-augmented model graphs. This module focuses on neg ctx baseline.

Top-level functions:
- `compute_neg_ctx_means(inference, sae_bank, neg_tokens, max_neg?)` - Compute per-latent mean activations over negative-context sequences.

#### `src/circuit/instrument/patcher.py`

Instrumentation, attribution, patching, and intervention utilities over SAE-augmented model graphs. This module focuses on patcher.

Classes and methods:
- `CircuitPatcher` - A patcher that intervenes on activations using a circuit definition.
  - `__init__(self, bank, circuit, avg_acts, inverse?, pos_argmax?, patch_kinds?, full_circuit?, max_layer?, circuit_layers?)` - Args: bank: The SAEBank containing the models.
  - `__call__(self, model)` - Invokes the instance as a callable.
  - `transform(self, layer_idx, kind, x)` - Handles transform.

#### `src/circuit/instrument/sae_graph.py`

Instrumentation, attribution, patching, and intervention utilities over SAE-augmented model graphs. This module focuses on sae graph.

Classes and methods:
- `FeatureGraph` - Stores grad-anchors and graph-connected activations as SparseAct objects.
  - `__init__(self, device)` - Initializes the instance.
  - `add(self, layer_idx, kind, state_grad, state_connected, top_indices)` - Handles add.
  - `get_latents(self, layer_idx, kind, step?)` - Returns (state_grad, state_connected, top_indices) for the given layer/kind.
  - `get_latents_by_id(self, feature_id, step?)` - Returns (state_grad, state_connected, top_indices) for the given FeatureID.
  - `all_anchors(self)` - Returns all leaf anchor tensors (act and res) that require grad.
  - `zero_grad(self)` - Zeros accumulated gradients on all leaf anchors.
- `SAEGraphInstrument` - Instruments the forward pass to capture SAE features and residual error term with gradients enabled, matching the Sparse Feature Circuits (Marks et al.
  - `__init__(self, bank, stop_error_grad?)` - Initializes the instance.
  - `__call__(self, model)` - Hook entry point for Inference.forward(patcher=instrument).
  - `transform(self, layer_idx, kind, x)` - Handles transform.
- `SAEGraphInstrumentWithEmbedding` - Thin subclass of SAEGraphInstrument that captures the input embedding (residual stream at layer 0, before the first SAE hook) as a detached leaf tensor, enabling gradient-based token attribution.
  - `__init__(self, bank, stop_error_grad?, first_layer?, first_kind?)` - Initializes the instance.
  - `transform(self, layer_idx, kind, x)` - Handles transform.

### `src/circuit/types`

#### `src/circuit/types/__init__.py`

Core circuit data types and low-level tensor helpers. This module focuses on   init  .

This module does not define Python functions or classes directly.

#### `src/circuit/types/feature_id.py`

Core circuit data types and low-level tensor helpers. This module focuses on feature id.

Classes and methods:
- `FeatureID` - Unified identifier for an SAE feature (latent).
  - `__repr__(self)` - Returns a debug representation of the instance.
  - `key(self)` - Returns the canonical tuple representation (layer, kind, index).
  - `from_global_id(cls, global_id, n_kinds, d_sae, kinds)` - Creates a FeatureID from a flat 'global' ID (comp_idx * d_sae + latent_idx).
  - `to_global_id(self, n_kinds, d_sae, kinds)` - Converts the FeatureID back to a flat 'global' ID.
  - `from_component_id(cls, comp_idx, latent_idx, n_kinds, kinds)` - Creates a FeatureID from a component index and a latent index.
  - `to_component_id(self, n_kinds, kinds)` - Returns (comp_idx, latent_idx).

#### `src/circuit/types/fused_ops.py`

Core circuit data types and low-level tensor helpers. This module focuses on fused ops.

Top-level functions:
- `fused_sparse_matmul(act1, act2, res1, res2, resc1, resc2)` - Fused kernel for SparseAct matrix multiplication.
- `fused_sparse_add(act1, act2, res1, res2, resc1, resc2)` - Fused kernel for SparseAct addition.

#### `src/circuit/types/sparse_act.py`

Core circuit data types and low-level tensor helpers. This module focuses on sparse act.

Classes and methods:
- `SparseAct` - A SparseAct represents a vector in the sparse feature basis provided by an SAE, jointly with the SAE error term (residual).
  - `__init__(self, act?, res?, resc?)` - Initializes the instance.
  - `_map(self, f, aux?)` - Handles map.
  - `__mul__(self, other)` - Handles mul.
  - `__rmul__(self, other)` - Handles rmul.
  - `__matmul__(self, other)` - Dot product between two SparseActs.
  - `__add__(self, other)` - Handles add.
  - `__radd__(self, other)` - Handles radd.
  - `__sub__(self, other)` - Handles sub.
  - `__truediv__(self, other)` - Handles truediv.
  - `__neg__(self)` - Handles neg.
  - `__invert__(self)` - Handles invert.
  - `__getitem__(self, index)` - Returns an item by key or index.
  - `sum(self, dim?)` - Handles sum.
  - `mean(self, dim?)` - Handles mean.
  - `grad(self)` - Handles grad.
  - `clone(self)` - Handles clone.
  - `detach(self)` - Handles detach.
  - `to_tensor(self)` - Concatenates features and (contracted) residual into one dense tensor.
  - `to(self, device)` - Handles to.
  - `nonzero(self)` - Handles nonzero.
  - `squeeze(self, dim)` - Handles squeeze.
  - `expand_as(self, other)` - Handles expand as.
  - `zeros_like(self)` - Handles zeros like.
  - `ones_like(self)` - Handles ones like.
  - `abs(self)` - Handles abs.
  - `device(self)` - Handles device.
  - `shape(self)` - Handles shape.
  - `is_leaf(self)` - Checks whether the object is leaf.
  - `requires_grad(self)` - Handles requires grad.
  - `grad_fn(self)` - Handles grad fn.
  - `__repr__(self)` - Returns a debug representation of the instance.

### `src/data`

#### `src/data/loader.py`

This module focuses on loader.

Classes and methods:
- `DataLoader` - DataLoader for SAE training.
  - `__init__(self, device, skip_first_token?, pin_memory?)` - Initializes the instance.
  - `_get_shard_files(self)` - Lists and sorts all .npy shard files in the data directory.
  - `_load_sequence_counts(self)` - Load or build per-shard sequence indices and store global ID ranges.
  - `_get_index_path(self, shard_index)` - Returns the path to the cached index file for a given shard.
  - `_build_shard_index(self, shard_index)` - Scans a shard once to record the (start, end) byte positions of every valid cleaned sequence, then saves the result as a .idx.npy cache file.
  - `_load_or_build_index(self, shard_index)` - Returns the cached index, rebuilding it if missing or stale.
  - `__len__(self)` - Returns the total number of batches across all (limited) shards.
  - `_load_shard(self, shard_index)` - Loads a specific shard by index, splits it into sequences using -1 as a separator.
  - `load_shard(self, shard_index)` - Public API for loading a shard (delegates to _load_shard).
  - `load_shard_sequences(self, shard_index, local_indices)` - Loads specific sequences from a shard by their local (within-shard) indices using the pre-built position index.
  - `get_sequence(self, sequence_id)` - Retrieves a specific sequence (row) by its global ID.
  - `get_batches(self, pad_to_max?, max_length?, device?)` - Yields batches of sequences across all shards.
  - `get_batches_by_ids(self, sequence_ids, pad_to_max?, max_length?, device?)` - Optimized batching for a specific set of IDs.
  - `_batch_to_tensor(self, batch, pad_to_max, max_length, device)` - Convert a list of sequences to a padded tensor or list of tensors.

### `src/debug`

#### `src/debug/circuit_evals.py`

Debug script: evaluates the SAE reconstruction patching framework under a "full circuit" baseline — every latent is treated as a circuit member, making the patcher a mathematical identity (recon + error = x).

Top-level functions:
- `run_full_circuit_eval(inference, bank, avg_acts, tokens, target_tokens, pos_argmax, label?)` - Runs faithfulness, sufficiency, and completeness for a full-circuit patcher.
- `main()` - Handles main.

#### `src/debug/faithfulness.py`

Debug script: runs ONE candidate through SFCAttributionPatching and prints verbose diagnostics at every stage, including the raw logit tensors and MSE values from the three evaluation passes.

Top-level functions:
- `_stats(t, label)` - Handles stats.
- `_logit_diff(orig, other, label, batch_idx, pos_argmax)` - Print per-sequence logit diff at the probe positions.
- `debug_baseline_mse(inference, sae_bank, avg_acts, tokens, pos_argmax, label?)` - Runs only the original and total-ablation baseline passes and prints the faithfulness DENOMINATOR: MSE(baseline, original) at probe positions.
- `debug_eval(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax)` - Replicates evaluate_faithfulness but with verbose intermediate printing.
- `main()` - Handles main.

#### `src/debug/mlp_sparse_expansion_step.py`

Debug script for one-seed MLP sparse expansion.

Top-level functions:
- `_decode_global_latent(global_idx, d_sae, n_kinds, kinds)` - Decodes global latent.
- `_print_top_coactivation(comp_idx, latent_idx, bank)` - Print full stored top-coactivation list for the seed, decoded like display.py.
- `_filter_neighbors_mlp_only(parent_comp, parent_lat, limit, in_circuit, bank, min_active_count)` - Mirror MlpSparseExpansion._expand_mlp_neighbors logic with reason accounting.
- `_capture_passthrough_nodes(inference, bank, probe_tokens)` - Mirror MlpSparseExpansion._capture_passthrough_nodes.
- `main()` - Handles main.

#### `src/debug/neg_ctx_cluster_display.py`

Cluster all neg_ctx sequences by embedding space and display representative sequences from the largest clusters.

Top-level functions:
- `_kmeans_cosine(norm, k, n_iters, chunk?, seed?)` - Cosine k-means on pre-normalised embeddings.
- `_fmt_sequence(text, max_chars?)` - Wrap and truncate decoded sequence text for display.
- `main()` - Handles main.

#### `src/debug/profile_pipeline.py`

Full pipeline profiler.

Top-level functions:
- `_make_callback(bank)` - Returns an SAE callback that annotates each section for the profiler.
- `_run_batch(model, batch_tokens, batch_ids, callback)` - Runs batch.
- `main()` - Handles main.

#### `src/debug/test_gradient_upstream.py`

Ad hoc debugging, profiling, and investigative scripts for individual pipeline stages. This module focuses on test gradient upstream.

Top-level functions:
- `main()` - Handles main.

#### `src/debug/test_inference.py`

Standalone inference test for TuringLLM.

Top-level functions:
- `load_first_sequences(n?, context_len?)` - Reads the first shard, splits on -1 separators, and returns the first n valid sequences (each truncated to context_len tokens).
- `pad_sequences(sequences, pad_token)` - Right-pads sequences to the same length and returns a LongTensor.
- `main()` - Handles main.

### `src/display`

#### `src/display/display.py`

Terminal display helpers for circuit summaries, latent views, and candidate inspection. This module focuses on display.

Top-level functions:
- `_build_mid_neg_seqs(component_idx, latent_idx, n_sequences)` - Builds mid neg seqs.

Classes and methods:
- `Display` - Represents Display.
  - `__init__(self)` - Initializes the instance.
  - `_resolve_token_parts(self, tokens)` - Decode a token list into display-ready text parts that preserve spacing.
  - `_compute_intensities(self, values)` - Z-score normalize activation values to [0, 1] intensities (-1 = no activation).
  - `_intensity_to_style(intensity, scheme?)` - Map a [0, 1] intensity to a rich style string.
  - `build_sequence_text(self, tokens, values?, scheme?)` - Build a rich Text object with optional heatmap coloring.
  - `print_sequence(self, tokens, values?, title?, border_style?, scheme?)` - Print a token sequence inside a rounded panel with optional heatmap.
  - `analyze_and_print_top_latents(self, top_ctx, model, bank, loader, n_latents?, n_sequences?)` - Identify top latents (normalized per component) and display their analysis.
  - `analyze_and_print_specific_latent(self, top_ctx, model, bank, loader, layer_idx, kind, latent_idx, n_sequences?)` - Analyze and display a specific latent by layer, kind, and index.
  - `analyze_and_print_latents(self, model, bank, loader, latents_info)` - Re-run the model and display per-token activations for each latent.

#### `src/display/get_top_candidates.py`

Terminal display helpers for circuit summaries, latent views, and candidate inspection. This module focuses on get top candidates.

Top-level functions:
- `print_top_candidates(n_seeds?)` - Prints top candidates.

#### `src/display/list_circuits.py`

Terminal display helpers for circuit summaries, latent views, and candidate inspection. This module focuses on list circuits.

Top-level functions:
- `_flatten_evals(evals)` - Flatten an evals dict into a single-level {label: value} mapping.
- `_print_eval_stats(circuits)` - Compute and print mean + variance for every eval field across all circuits.
- `list_discovered_circuits(summary_path?)` - Lists discovered circuits.

### `src/eval`

#### `src/eval/cluster_faithfulness.py`

Log-prob faithfulness evaluation for ClusterContrastDiscovery circuits.

Top-level functions:
- `_last_token_logits(logits)` - Return [B, vocab] last-token logits regardless of input shape.
- `_mean_log_prob(logits, token_ids)` - Mean log-probability of token_ids[i] under logits[i].
- `_zero_avg_acts(bank)` - Zero avg_acts for CircuitPatcher (zero-ablation baseline, same as SFC patch=None).
- `evaluate_cluster_faithfulness(inference, bank, circuit, pos_tokens, neg_tokens, eval_position?, batch_size?)` - Evaluate a ClusterContrastDiscovery circuit using the SFC faithfulness formula.

Classes and methods:
- `_ClusterInjectionPatcher` - Injects cluster_activator latents to their mean positive-sequence values and suppresses cluster_inhibitor latents to zero, preserving the SAE error term.
  - `__init__(self, bank, activator_targets, inhibitor_indices)` - Initializes the instance.
  - `__call__(self, model)` - Invokes the instance as a callable.
  - `transform(self, layer_idx, kind, x)` - Handles transform.

#### `src/eval/completeness.py`

Evaluation metrics for faithfulness, sufficiency, completeness, minimality, and counterfactual behavior. This module focuses on completeness.

Top-level functions:
- `evaluate_completeness(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax?, max_layer?, circuit_layers?)` - Measures if the circuit is complete by checking the performance of its complement.

#### `src/eval/counterfactual_faithfulness.py`

Counterfactual faithfulness evaluation for CounterfactualGradientDiscovery.

Top-level functions:
- `evaluate_counterfactual_faithfulness(inference, sae_bank, avg_acts, circuit, neg_tokens, pos_tokens, seed_layer, seed_kind, seed_latent_idx, pos_argmax?, circuit_layers?)` - Measure how well the discovered activators and inhibitors causally explain the seed's firing behaviour in both directions.

Classes and methods:
- `CounterfactualInterventionPatcher` - A forward-pass hook that injects activator activations and suppresses inhibitor activations on negctx sequences, then captures the seed latent's response at the probe position.
  - `__init__(self, bank, activator_targets, inhibitor_indices, seed_layer, seed_kind, seed_latent_idx, pos_argmax?, circuit_layers?)` - Initializes the instance.
  - `__call__(self, model)` - Invokes the instance as a callable.
  - `transform(self, layer_idx, kind, x)` - Handles transform.

#### `src/eval/faithfulness.py`

Evaluation metrics for faithfulness, sufficiency, completeness, minimality, and counterfactual behavior. This module focuses on faithfulness.

Top-level functions:
- `_calculate_faithfulness_score(original_logits, intervened_logits, baseline_logits, pos_argmax?)` - Calculates the normalized faithfulness score from logit tensors.
- `evaluate_faithfulness(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax?, max_layer?, circuit_layers?)` - Calculates the faithfulness of a circuit on a specific sequence.
- `evaluate_kind_local_faithfulness(inference, sae_bank, avg_acts, circuit, tokens, target_kinds, pos_argmax?, max_layer?)` - Calculates faithfulness restricted to selected SAE kinds.

#### `src/eval/minimality.py`

Evaluation metrics for faithfulness, sufficiency, completeness, minimality, and counterfactual behavior. This module focuses on minimality.

Top-level functions:
- `evaluate_minimality(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax?, max_layer?, circuit_layers?)` - Checks for "dead weight" in a circuit.
- `prune_non_minimal_nodes(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax?, threshold?, max_layer?, circuit_layers?)` - Identifies and removes nodes that contribute less than a threshold to faithfulness.
- `prune_non_minimal_nodes_cf(inference, sae_bank, avg_acts, circuit, neg_tokens, pos_tokens, seed_layer, seed_kind, seed_latent_idx, pos_argmax?, threshold?, circuit_layers?, max_candidates_per_iter?, max_iterations?)` - Counterfactual-faithfulness variant of iterative minimality pruning.

#### `src/eval/node_presence.py`

Posctx node presence and circuit sufficiency evaluation.

Top-level functions:
- `evaluate_node_presence(inference, sae_bank, circuit, pos_tokens)` - Two-pass evaluation of posctx node presence and circuit sufficiency.

Classes and methods:
- `_CircuitSufficiencyPatcher` - A forward-pass hook for the circuit-isolation pass.
  - `__init__(self, bank, activator_map, seed_layer, seed_kind, seed_latent_idx)` - Initializes the instance.
  - `__call__(self, model)` - Invokes the instance as a callable.
  - `transform(self, layer_idx, kind, x)` - Handles transform.

#### `src/eval/sufficiency.py`

Evaluation metrics for faithfulness, sufficiency, completeness, minimality, and counterfactual behavior. This module focuses on sufficiency.

Top-level functions:
- `evaluate_sufficiency(inference, sae_bank, avg_acts, circuit, tokens, target_tokens, pos_argmax?, max_layer?, circuit_layers?)` - Measures if the circuit captures the "full story" for a specific prompt.

#### `src/eval/upstream_faithfulness.py`

Evaluation metrics for faithfulness, sufficiency, completeness, minimality, and counterfactual behavior. This module focuses on upstream faithfulness.

Top-level functions:
- `evaluate_upstream_faithfulness(inference, sae_bank, avg_acts, circuit, seed_layer, seed_kind, seed_latent_idx, tokens, pos_argmax?, circuit_layers?)` - Measures how well the circuit's upstream nodes explain the seed latent's activation.

Classes and methods:
- `SeedActivationCapturePatcher` - Subclass of CircuitPatcher that captures the activation of a specific target latent (the seed) at its peak position during the forward pass.
  - `__init__(self, bank, circuit, avg_acts, seed_layer, seed_kind, seed_latent_idx, pos_argmax?, patch_pos_selective?, **kwargs)` - Initializes the instance.
  - `transform(self, layer_idx, kind, x)` - Handles transform.

### `src/model`

#### `src/model/hooks.py`

Model architecture, inference plumbing, hooks, and tokenization for TuringLLM. This module focuses on hooks.

Top-level functions:
- `capture_activations(model, callback?, capture?)` - Captures activations in shape: [B, L, K, T, N] B=Batch, L=Layer, K=Kind (0=attn, 1=mlp, 2=resid), T=Token, N=Neuron.
- `patch(model, layer_idx, kind, value)` - kind: attn, mlp, resid.
- `multi_patch(model, transform)` - transform: (layer_idx, kind, tensor) -> tensor (or None to skip) kind: attn, mlp, resid.
- `stop_grad_at(model, layer_idx, kind)` - Zeros gradients flowing back through a specific submodule during backward.
- `multi_stop_grad(model, stop_grads)` - Context manager to stop gradients at multiple locations simultaneously.

Classes and methods:
- `Activations` - Represents Activations.
  - `__init__(self)` - Initializes the instance.

#### `src/model/inference.py`

Model architecture, inference plumbing, hooks, and tokenization for TuringLLM. This module focuses on inference.

Top-level functions:
- `_fuse_mlp_projections(state_dict)` - Remaps a checkpoint saved with split up_proj_swish / up_proj weights onto the fused gate_up_proj layout used by the current MLP definition.

Classes and methods:
- `Inference` - Represents Inference.
  - `__init__(self, device, compile?)` - Initializes the instance.
  - `enable_compile(self)` - Handles enable compile.
  - `disable_compile(self)` - Handles disable compile.
  - `enable_grad_checkpointing(self)` - Enables gradient checkpointing on all transformer blocks' attn and mlp submodules.
  - `disable_grad_checkpointing(self)` - Handles disable grad checkpointing.
  - `forward(self, tokens, num_gen?, tokenize_final?, activations_callback?, patcher?, return_activations?, all_logits?, grad_enabled?)` - Runs the forward pass.

#### `src/model/tokenizer.py`

Model architecture, inference plumbing, hooks, and tokenization for TuringLLM. This module focuses on tokenizer.

Classes and methods:
- `Tokenizer` - Represents Tokenizer.
  - `__init__(self, model_id?)` - Initializes the instance.
  - `encode(self, text)` - Encodes the requested value.
  - `decode(self, tokens)` - Decodes the requested value.
  - `get_eos_token(self)` - Returns eos token.
  - `get_bos_token(self)` - Returns bos token.
  - `get_pad_token(self)` - Returns pad token.

#### `src/model/turingllm.py`

Model architecture, inference plumbing, hooks, and tokenization for TuringLLM. This module focuses on turingllm.

Classes and methods:
- `TuringLLMConfig` - Represents TuringLLMConfig.
- `RMSNorm` - Represents RMSNorm.
  - `__init__(self, dim, eps?)` - Initializes the instance.
  - `forward(self, x)` - Runs the forward pass.
- `CausalSelfAttention` - Represents CausalSelfAttention.
  - `__init__(self, config)` - Initializes the instance.
  - `_forward_impl(self, x)` - Handles forward impl.
  - `forward(self, x, input_pos?)` - Runs the forward pass.
- `MLP` - Represents MLP.
  - `__init__(self, config)` - Initializes the instance.
  - `_forward_impl(self, x)` - Handles forward impl.
  - `forward(self, x)` - Runs the forward pass.
- `Block` - Represents Block.
  - `__init__(self, config)` - Initializes the instance.
  - `forward(self, x, input_pos)` - Runs the forward pass.
- `TuringLLM` - Represents TuringLLM.
  - `__init__(self)` - Initializes the instance.
  - `forward(self, idx, targets?, input_pos?, return_all_logits?)` - Runs the forward pass.

### `src/native`

#### `src/native/setup.py`

Build glue and tests for native C++/CUDA acceleration kernels. This module focuses on setup.

This module does not define Python functions or classes directly.

### `src/observability`

#### `src/observability/__init__.py`

Logging, timing, tracking, and console utilities for runtime observability. This module focuses on   init  .

This module does not define Python functions or classes directly.

#### `src/observability/circuit_logger.py`

Logging, timing, tracking, and console utilities for runtime observability. This module focuses on circuit logger.

Classes and methods:
- `CircuitLogger` - Per-seed, per-method text logger for circuit discovery.
  - `__init__(self, seed_comp, seed_latent, method_name)` - Initializes the instance.
  - `_w(self, line)` - Handles w.
  - `header(self, seed_layer, seed_kind, seed_latent, n_pos, n_neg)` - Log seed identity and probe dataset sizes.
  - `stage(self, label, n_nodes, n_edges, note?)` - Log node/edge counts at a named stage of the algorithm, with timing.
  - `note(self, text)` - Append a free-form informational line.
  - `eval(self, faithfulness, sufficiency, completeness)` - Log the three evaluation scores with timing.
  - `reject(self, reason)` - Mark this attempt as rejected and record the reason.
  - `accept(self, n_nodes, n_edges)` - Mark this attempt as accepted.
  - `nodes(self, circuit_nodes)` - Log a compact listing of circuit nodes, grouped by role.
  - `cancel(self)` - Prevents the log from being saved to disk.
  - `save(self)` - Flush all buffered lines to the log file.

#### `src/observability/console.py`

Structured console output helpers.

Classes and methods:
- `Console` - Represents Console.
  - `section(self, title)` - Print a prominent section header.
  - `step(self, message)` - Print a top-level pipeline step (no indent).
  - `detail(self, message)` - Print an indented detail line.
  - `success(self, message)` - Print an indented success line with a checkmark.
  - `warn(self, message)` - Print an indented warning line.
  - `error(self, message, prefix?)` - Print an error line with optional prefix.
  - `kv(self, key, value)` - Print an indented key-value pair.
  - `blank(self)` - Print an empty line.

#### `src/observability/timing.py`

Logging, timing, tracking, and console utilities for runtime observability. This module focuses on timing.

Top-level functions:
- `timer(label?, into?)` - Context manager for timing a block of code.

Classes and methods:
- `TimerResult` - Holds the elapsed time after a Timer context exits.
  - `__init__(self)` - Initializes the instance.
  - `elapsed_ms(self)` - Handles elapsed ms.
  - `__repr__(self)` - Returns a debug representation of the instance.

#### `src/observability/tracking.py`

Logging, timing, tracking, and console utilities for runtime observability. This module focuses on tracking.

Classes and methods:
- `Observability` - Represents Observability.
  - `__init__(self)` - Initializes the instance.
  - `start_attempt(self)` - Handles start attempt.
  - `stop_attempt(self)` - Handles stop attempt.
  - `track_forward(self)` - Tracks forward.

### `src/pipeline`

#### `src/pipeline/__init__.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on   init  .

This module does not define Python functions or classes directly.

#### `src/pipeline/candidate_selection.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on candidate selection.

Top-level functions:
- `run_candidate_selection()` - Runs candidate selection.

#### `src/pipeline/cluster_discovery.py`

Cluster Contrast Discovery runner.

Top-level functions:
- `_load_neg_ctx_if_needed()` - Loads neg ctx if needed.
- `_load_latent_stats_if_needed()` - Loads latent stats if needed.
- `_load_seq_repr()` - Loads seq repr.
- `_cluster_center_ids(cluster_result, cluster_idx, n)` - Return the IDs of the N sequences closest to cluster centroid `cluster_idx`.
- `_cf_seq_ids(cluster_result, target_cluster_idx, n)` - Return the IDs of the N sequences closest to the target cluster centroid that are NOT members of that cluster (hard negatives).
- `_print_summary(results)` - Print a compact table of accepted circuits.
- `_save_summary(results, output_dir)` - Persist a JSON summary of accepted circuits.
- `run_cluster_contrast_discovery(inference, bank, loader)` - Full cluster-contrast discovery run.
- `main()` - Handles main.

#### `src/pipeline/component_index.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on component index.

Top-level functions:
- `component_idx(layer_idx, kind_idx, n_kinds)` - Map (layer, kind) to a flat component index.
- `split_component_idx(component_idx_value, n_kinds)` - Inverse of component_idx -> (layer_idx, kind_idx).
- `layer_component_bounds(layer_idx, n_kinds)` - Return inclusive/exclusive component index bounds for one layer.
- `kind_name_for_component(component_idx_value, kinds)` - Return kind name for a flat component index.
- `get_predecessor_components(comp_idx, n_kinds, kinds)` - Returns the list of component indices that causally precede the given component based on the transformer's residual arithmetic.
- `get_all_upstream_components(comp_idx, n_kinds, kinds, min_layer?, include_same_layer?)` - Returns every component index that lies upstream of the given component.

#### `src/pipeline/discovery.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on discovery.

Top-level functions:
- `prepare_discovery_resources()` - Ensure resources needed by discovery are initialized.
- `run_discovery(candidates?, candidates_path?)` - Runs discovery.

#### `src/pipeline/encoding.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on encoding.

Top-level functions:
- `encode_layer_components(bank, layer_idx, activations, *, primary_device, multi_gpu)` - Encode all component kinds for one layer.

#### `src/pipeline/first_pass.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on first pass.

Top-level functions:
- `_update_stores(mid_ctx_warmup, current_batch_last_latents, comp_idx, sequence_ids, latents)` - Updates stores.
- `run_first_pass()` - Runs first pass.

#### `src/pipeline/negative_context.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on negative context.

Top-level functions:
- `build_negative_contexts()` - Builds negative contexts.

#### `src/pipeline/persist.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on persist.

Top-level functions:
- `offload_to_cpu()` - Move stores to CPU memory to free VRAM for subsequent stages.
- `offload_model_and_sae()` - Release model/SAE GPU memory before ANN-heavy negative-context build.
- `reload_model_and_sae()` - Recreate model/SAE resources after the ANN step completes.
- `save_results()` - Saves results.

#### `src/pipeline/run_pipeline.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on run pipeline.

Top-level functions:
- `run()` - Runs the requested value.

#### `src/pipeline/runtime.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on runtime.

Top-level functions:
- `set_runtime(runtime)` - Sets runtime.
- `get_runtime()` - Returns runtime.
- `clear_runtime()` - Handles clear runtime.
- `initialize_runtime()` - Handles initialize runtime.
- `build_runtime()` - Builds runtime.
- `initialize_resources()` - Handles initialize resources.

Classes and methods:
- `PipelineRuntime` - Represents PipelineRuntime.

#### `src/pipeline/second_pass.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on second pass.

Top-level functions:
- `run_second_pass()` - Runs second pass.

#### `src/pipeline/seq_latent_index.py`

End-to-end pipeline stages for collection, negative-context construction, co-activation mining, candidate selection, discovery, and persistence. This module focuses on seq latent index.

Classes and methods:
- `SeqLatentIndexAccumulator` - Accumulates (sequence_id, latent_id) pairs for the top-K active latents per sequence per component during the first pass, then writes one ``outputs/seq_latent_index/shard_{i}.pt`` file per data shard.
  - `__init__(self, shard_id_ranges, top_k_per_component, output_dir)` - Initializes the instance.
  - `update(self, comp_idx, seq_ids, latents)` - Record the top-K latent IDs per sequence for one component.
  - `on_batch_complete(self, batch_max_id)` - Flush any shards whose last sequence ID is strictly less than ``batch_max_id``.
  - `flush_all(self)` - Flush and save all remaining buffered shards.
  - `_shard_indices_for(self, seq_ids_np)` - Return the shard index for each sequence ID (vectorised binary search).
  - `_flush_shard(self, shard_idx)` - Concatenate buffered pairs for each component and write to disk.

### `src/sae`

#### `src/sae/async_encode.py`

Sparse autoencoder components, accelerated Top-K selection, fused ops, and multi-device SAE orchestration. This module focuses on async encode.

Top-level functions:
- `encode_layer_async(bank, layer_idx, activations, primary_device)` - Encode all 3 SAE components for a layer on the target device's stream.

Classes and methods:
- `PendingEncode` - Holds SAE encode results launched on a CUDA stream.
  - `__init__(self, stream, comp_results, primary_device)` - Initializes the instance.
  - `synchronize(self)` - Handles synchronize.

#### `src/sae/bank.py`

Sparse autoencoder components, accelerated Top-K selection, fused ops, and multi-device SAE orchestration. This module focuses on bank.

Classes and methods:
- `SAEBank` - Represents SAEBank.
  - `__init__(self, device?, devices?, load_decoders?, compile?)` - Initializes the instance.
  - `load_saes(self)` - Loads saes.
  - `pin_decoders(self)` - Context manager that moves every SAE decoder to GPU VRAM for the duration of the block, then restores to CPU if they were not originally in VRAM.
  - `_autocast_ctx(self, device)` - Handles autocast ctx.
  - `encode(self, x, kind, layer)` - Encodes the requested value.
  - `encode_layer_kinds_parallel(self, activations, layer)` - Encode all kinds for a layer using per-kind CUDA streams.
  - `decode(self, latents, kind, layer)` - Decodes the requested value.
  - `full_encode(self, x, kind, layer)` - Returns (features [..., d_sae], residual [..., d_model]).
  - `profile_encode(self, x, kind, layer, output_dir?, warmup?, active?)` - Profiles sae.encode() for a single (kind, layer) using torch.profiler.

#### `src/sae/fused_linear_relu.py`

Fused linear + ReLU via cublasLt RELU_BIAS epilogue.

Top-level functions:
- `_load()` - Loads the requested value.
- `is_available()` - Returns True if the cublasLt fused kernel is importable.
- `linear_relu(x, weight, bias)` - Computes relu(x @ weight.T + bias).

#### `src/sae/topk_sae.py`

Sparse autoencoder components, accelerated Top-K selection, fused ops, and multi-device SAE orchestration. This module focuses on topk sae.

Top-level functions:
- `set_topk_backend(backend)` - Switch the top-k implementation at runtime.
- `get_topk_backend()` - Return the name of the currently active top-k backend.
- `_warmup_triton_topk(d_sae, k, device)` - Run a tiny forward pass to trigger Triton JIT compilation and autotuning.

Classes and methods:
- `SAEConfig` - Represents SAEConfig.
- `SAE` - A simplified Top-K Sparse Autoencoder for inference.
  - `__init__(self, d_model, d_sae, k?, device?, compile?)` - Initializes the instance.
  - `_get_bias_eff(self)` - Lazily computes and caches b_enc - W_enc @ b_dec.
  - `encode(self, x)` - Encodes the input tensor x into sparse top-K activations.
  - `decode(self, latents, ensure_device?)` - Decodes the latents back into the original d_model space.
  - `forward(self, x)` - Full pass: encode and then decode.
  - `load(self, path)` - Loads the SAE weights from a file.
  - `move_decoder_to_vram(self)` - Moves the decoder weights back to the same device as the encoder (VRAM).
  - `remove_decoder_from_vram(self, empty_cache?)` - Moves the decoder weights to CPU to save VRAM when only encoding is needed.

#### `src/sae/triton_topk.py`

Triton radix-select top-k for non-negative BF16 tensors.

Top-level functions:
- `is_available()` - Returns True if Triton is importable and CUDA is present.
- `_prune_configs(configs, named_args, **kwargs)` - Drop configs where BLOCK_N doesn't divide N — prevents no-op tile loops.
- `_build_kernel()` - Deferred import so the module loads fine on CPU-only machines.
- `_get_kernel()` - Returns kernel.
- `topk_nonneg_bf16(x, k)` - Top-k over the last dimension for a non-negative BF16 tensor.

### `src/store`

#### `src/store/circuits.py`

Storage layers for latent statistics, contexts, co-activation graphs, circuit outputs, and search caches. This module focuses on circuits.

Classes and methods:
- `CircuitNode` - Represents a single node in a discovered circuit (e.g.
  - `feature_id(self)` - Returns the FeatureID if present in metadata.
  - `weight(self)` - Handles weight.
  - `source(self)` - Handles source.
- `CircuitEdge` - Represents a causal or attribution edge between two nodes.
  - `weight(self)` - Handles weight.
- `Circuit` - A collection of nodes and edges representing a single discovered mechanism.
  - `add_node(self, node)` - Adds a node to the circuit and returns it.
  - `add_edge(self, source_uuid, target_uuid, **metadata)` - Adds an edge between two existing nodes.
- `CircuitStore` - A central store for managing multiple discovered circuits.
  - `__init__(self)` - Initializes the instance.
  - `add_circuit(self, circuit)` - Adds a circuit to the store and returns it.
  - `get_circuit(self, circuit_uuid)` - Retrieves a circuit by its UUID.
  - `save(self, path)` - Persists the entire store using torch.save.
  - `load(self, path)` - Loads circuits from a saved file.

#### `src/store/context.py`

Storage layers for latent statistics, contexts, co-activation graphs, circuit outputs, and search caches. This module focuses on context.

Top-level functions:
- `_load_mid_reservoir_ext()` - Loads mid reservoir ext.
- `compute_seq_scores(top_acts, top_indices, d_sae)` - Returns [d_sae, batch] float32 mean activation score per latent per sequence.

Classes and methods:
- `Context` - Represents Context.
  - `__init__(self, ctx_type, device?)` - Initializes the instance.
  - `allocate(self, device?)` - Handles allocate.
  - `update_component(self, component_idx, sequence_indices, latents, latent_mean_seq?, latent_std_seq?)` - Update stored contexts for one SAE component over a batch.
  - `_update_top(self, component_idx, sequence_indices, latents)` - Updates top.
  - `_update_mid(self, component_idx, sequence_indices, latents, latent_mean_seq, latent_std_seq)` - Updates mid.
  - `save(self, path)` - Saves the requested value.
  - `load(self, path)` - Loads the requested value.
  - `set_device(self, device)` - Sets device.
  - `get_all_sequence_ids(self)` - Returns a sorted list of all unique sequence IDs stored (excludes sentinel 0).
  - `get_sequence_to_latents_map(self)` - Maps sequence ID → list of (component_idx, latent_idx) pairs.
  - `get_sequence_to_latents_csr(self, device?)` - CSR-style mapping: (seq_offsets, seq_targets_global).

#### `src/store/latent_stats.py`

Storage layers for latent statistics, contexts, co-activation graphs, circuit outputs, and search caches. This module focuses on latent stats.

Top-level functions:
- `_load_cuda_ext()` - Load the pre-compiled latent_stats_cuda extension from src/native/.

Classes and methods:
- `LatentStats` - Represents LatentStats.
  - `__init__(self, device?)` - Initializes the instance.
  - `allocate(self, device?)` - Explicitly allocate the large GPU tensors.
  - `update_component(self, component_idx, latents)` - Updates component.
  - `_update_seq_scores(self, component_idx, top_acts, top_indices)` - Welford update for per-sequence activation scores.
  - `_welford_merge(mean_a, m2_a, n_a, mean_b, m2_b, n_b)` - Parallel Welford merge: combines batch stats (B) into global stats (A) in-place.
  - `variance(self, component_idx?)` - Sample variance of a: M2 / max(1, n - 1).
  - `variance_abs(self, component_idx?)` - Sample variance of |a|: M2_abs / max(1, n - 1).
  - `std(self, component_idx?)` - Sample standard deviation of a.
  - `std_abs(self, component_idx?)` - Sample standard deviation of |a|.
  - `std_seq(self, component_idx?)` - Sample standard deviation of per-sequence activation scores.
  - `load(self, path)` - Loads the requested value.
  - `save(self, path)` - Saves the requested value.
  - `set_device(self, device)` - Sets device.

#### `src/store/logit_context.py`

Storage layers for latent statistics, contexts, co-activation graphs, circuit outputs, and search caches. This module focuses on logit context.

Classes and methods:
- `LogitContext` - Stores and updates the tokens most likely to be generated when specific SAE latents are activated, using efficient GPU-vectorized Top-K merging.
  - `__init__(self, device?)` - Initializes the instance.
  - `allocate(self, device?)` - Explicitly allocate the large GPU tensors.
  - `update(self, component_last_indices, final_probs)` - Updates the top token mapping using the final probabilities and active latents.
  - `get_top_tokens(self, component_idx, latent_idx)` - Returns the top stored tokens for a specific latent, filtering out zeros.
  - `load(self, path)` - Loads the requested value.
  - `save(self, path)` - Saves the requested value.
  - `set_device(self, device)` - Sets device.

#### `src/store/neg_context.py`

Negative context builder (ANN retrieval).

Top-level functions:
- `_ann_device()` - Handles ann device.
- `_process_component(comp_idx, top_ctx, mid_ctx, neg_ctx, index, K, n_neg, min_pos_ctx, stats, total_n_seqs, slot_to_id_d, id_to_slot_d)` - Process one SAE component end-to-end without any Python loop over latents.
- `build_neg_ctx(seq_repr, top_ctx, mid_ctx, neg_ctx)` - Populate neg_ctx for all latents with sufficient PosCtx data.

Classes and methods:
- `NegCtxStats` - Represents NegCtxStats.
  - `fill_rate_mean(self)` - Handles fill rate mean.
  - `fill_rate_p10(self)` - Handles fill rate p10.
  - `fill_rate_p50(self)` - Handles fill rate p50.
  - `fill_rate_p90(self)` - Handles fill rate p90.
  - `print_summary(self, n_sequences)` - Prints summary.
  - `save(self, path)` - Saves the requested value.
- `TorchANNIndex` - Exact cosine similarity index.
  - `__init__(self, vecs, device)` - Initializes the instance.
  - `search(self, queries, k)` - Returns (similarities [Q, k], indices [Q, k]) on self.device.

#### `src/store/search_cache.py`

Storage layers for latent statistics, contexts, co-activation graphs, circuit outputs, and search caches. This module focuses on search cache.

Top-level functions:
- `generate_search_cache(top_ctx, bank, loader, output_path?, n_sequences?, component_chunk_size?)` - Vastly optimized search cache generation using a sequence-first approach and vectorized processing.

#### `src/store/seq_repr.py`

Per-sequence representation store.

Classes and methods:
- `SeqRepr` - Represents SeqRepr.
  - `__init__(self, n_seqs, device?)` - Args: n_seqs: Total number of unique sequences in the dataset.
  - `update(self, seq_ids, resid)` - Store pooled residual stream representations for a batch of sequences.
  - `get_repr(self, seq_ids)` - Returns float32 representations for the requested sequence IDs.
  - `print_stats(self)` - Prints stats.
  - `save(self, path)` - Saves the requested value.
  - `load(self, path)` - Loads the requested value.

#### `src/store/top_coactivation.py`

Storage layers for latent statistics, contexts, co-activation graphs, circuit outputs, and search caches. This module focuses on top coactivation.

Classes and methods:
- `TopCoactivation` - Represents TopCoactivation.
  - `__init__(self, device?)` - Initializes the instance.
  - `mode(self)` - Handles mode.
  - `allocate(self, device?)` - Explicitly allocate the large GPU tensors.
  - `set_frequency_factors(self, active_counts, alpha?, epsilon?)` - Sets frequency factors.
  - `prepare_dump(self, sequence_ids)` - Pre-allocate candidate tensors and build the sequence-ID-to-row mapping.
  - `_score_freq_weighted(self, dense, comp_idx)` - Apply the frequency adjustment factor to the mean activations.
  - `_score_raw(self, dense)` - Return the mean activations without any frequency adjustment.
  - `update_batch(self, batch_ids, component_latents)` - Compute per-sequence candidate profiles and write them to the dump tensors.
  - `reduce(self, seq_offsets, seq_targets_global, seq_len?, active_count?)` - Run the C++ post-processing reduction.
  - `_apply_pmi_postprocess(self, active_count, seq_offsets, seq_targets_global, seq_len)` - Post-process top_values (binary firing counts) into PMI log-scores.
  - `_compute_total_tokens_per_target(self, seq_offsets, seq_targets_global, seq_len)` - For each target latent (global ID), count the total token positions across all sequences in its top_ctx set.
  - `load(self, path)` - Loads the requested value.
  - `save(self, path)` - Saves the requested value.
  - `set_device(self, device)` - Sets device.

#### `src/store/utils.py`

Storage layers for latent statistics, contexts, co-activation graphs, circuit outputs, and search caches. This module focuses on utils.

Classes and methods:
- `_AutoAllocTensor` - Descriptor for lazily-allocated tensor attributes on store classes.
  - `__set_name__(self, owner, name)` - Sets name.
  - `__get__(self, obj, objtype)` - Returns the requested value.
  - `__get__(self, obj, objtype)` - Returns the requested value.
  - `__get__(self, obj, objtype?)` - Returns the requested value.
  - `__set__(self, obj, value)` - Sets the requested value.
