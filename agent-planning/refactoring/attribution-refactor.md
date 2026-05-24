# Plan: Attribution Refactor

> **Goal:** Split `src/circuit/instrument/attribution.py` into focused attribution modules while preserving existing public imports, algorithm behavior, and test coverage.
>
> **Created:** 2026-05-23

---

## Phase 1 — Establish Compatibility Baseline

- [x] Record the current public API from `src/circuit/instrument/attribution.py`: `UpstreamScores`, `compute_logit_attribution`, `compute_feature_attribution`, `compute_feature_gradient`, `compute_latent_counterfactual_scores`, `collect_active_feature_nodes`, `compute_direct_effects_matrix`, and `compute_latent_upstream_scores`.
- [x] Search repository imports of `circuit.instrument.attribution` and direct imports from this module to identify compatibility requirements.
- [x] Decide on sibling modules rather than an `attribution/` package to avoid a Python import conflict with the existing `attribution.py` file.
- [x] Run the focused baseline command before refactoring: `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_patcher.py tests/circuit/test_sae_graph.py tests/model/test_hooks.py -q`.
- [x] Note any pre-existing failures, environment issues, or skipped tests in this plan before implementation begins.

### Phase 1 Notes

Current public names to preserve from `circuit.instrument.attribution`:

- Dataclasses: `UpstreamScores`.
- Public attribution/scoring helpers: `compute_logit_attribution`, `compute_feature_attribution`, `compute_feature_gradient`, `compute_latent_counterfactual_scores`, `collect_active_feature_nodes`, `compute_direct_effects_matrix`, and `compute_latent_upstream_scores`.
- Private helpers currently owned by the large module and planned for mechanical extraction: `_find_logit_targets`, `_upstream_anchors_for_target`, `_all_feature_anchors_with_meta`, `_score_grads_into_adj`, `_score_token_grads_into_adj`, and `_compute_partial_one_hop_influence`.

Observed compatibility call sites:

- `tests/circuit/test_attribution.py` imports the main public attribution surface directly from `circuit.instrument.attribution`.
- `tests/circuit/discovery/test_layerwise_gradient_upstream.py` imports `UpstreamScores`.
- `tests/circuit/discovery/test_counterfactual_gradient.py` imports `compute_latent_counterfactual_scores`.
- `src/circuit/discovery/gradient_upstream.py` and `src/circuit/discovery/layerwise_gradient_upstream.py` import `compute_latent_upstream_scores` and `UpstreamScores`.
- `src/circuit/discovery/logit_attribution.py`, `src/circuit/discovery/differential_activation.py`, and `src/circuit/discovery/top_coact_attr.py` import feature/logit attribution helpers.
- `src/circuit/discovery/circuit_tracer_baseline.py` imports `compute_direct_effects_matrix`.
- `src/circuit/discovery/counterfactual_gradient.py` imports `compute_latent_counterfactual_scores`.
- `src/circuit/discovery/top_coact_expansion/hard_negative_coact_sparse_expansion.py` imports `compute_feature_gradient`.

Extraction shape decision:

- Use sibling modules such as `attribution_types.py`, `attribution_active_nodes.py`, and `attribution_logit.py`, not an `attribution/` package, because `src/circuit/instrument/attribution.py` must remain importable as the compatibility facade during the refactor.

Baseline verification:

- Command: `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_patcher.py tests/circuit/test_sae_graph.py tests/model/test_hooks.py -q`
- Result on 2026-05-23: `139 passed in 0.46s`.
- Pre-existing failures/blockers: none observed.

## Phase 2 — Extract Shared Types And Active Node Collection

- [x] Create `src/circuit/instrument/attribution_types.py` for shared dataclasses and type-oriented helpers.
- [x] Move `UpstreamScores` into `attribution_types.py`.
- [x] Create `src/circuit/instrument/attribution_active_nodes.py`.
- [x] Move `collect_active_feature_nodes` into `attribution_active_nodes.py`.
- [x] Update `attribution.py` to import and re-export the moved names.
- [x] Verify with `python -m pytest tests/circuit/test_attribution.py -q`.

### Phase 2 Notes

- Added `src/circuit/instrument/attribution_types.py` for the shared `UpstreamScores` dataclass and explicit `__all__`.
- Added `src/circuit/instrument/attribution_active_nodes.py` for `collect_active_feature_nodes`, preserving the active-count filtering, peak-activation ranking, optional cap, and stable `(layer, kind_idx, latent_index)` ordering.
- Updated `src/circuit/instrument/attribution.py` to import and re-export `UpstreamScores` and `collect_active_feature_nodes`; existing callers can still import both names from `circuit.instrument.attribution`.
- Verification on 2026-05-23: `python -m pytest tests/circuit/test_attribution.py -q` -> `35 passed in 0.24s`.
- Lints/diagnostics reported no errors for `attribution.py`, `attribution_types.py`, or `attribution_active_nodes.py`.

## Phase 3 — Extract Basic Logit And Feature Attribution

- [x] Create `src/circuit/instrument/attribution_logit.py`.
- [x] Move `compute_logit_attribution` into `attribution_logit.py`.
- [x] Create `src/circuit/instrument/attribution_feature.py`.
- [x] Move `compute_feature_attribution` and `compute_feature_gradient` into `attribution_feature.py`.
- [x] Keep shared imports from `FeatureGraph`, `SparseAct`, and `FeatureID` local and explicit in the new modules.
- [x] Update `attribution.py` to import and re-export the moved functions.
- [x] Verify with `python -m pytest tests/circuit/test_attribution.py -q`.

### Phase 3 Notes

- Added `src/circuit/instrument/attribution_logit.py` for `compute_logit_attribution`, preserving the current logit-target scalar construction, leaf-anchor gradient collection, and activation-times-gradient scoring.
- Added `src/circuit/instrument/attribution_feature.py` for `compute_feature_attribution` and `compute_feature_gradient`, preserving target feature validation, upstream anchor collection, candidate-node filtering, and raw-gradient scoring behavior.
- Kept `FeatureGraph`, `SparseAct`, and `FeatureID` imports local and explicit in the extracted modules.
- Updated `src/circuit/instrument/attribution.py` to import and re-export the moved public functions, so existing imports from `circuit.instrument.attribution` continue to work.
- Verification on 2026-05-23: `python -m pytest tests/circuit/test_attribution.py -q` -> `35 passed in 0.22s`.
- Lints/diagnostics reported no errors for `attribution.py`, `attribution_logit.py`, or `attribution_feature.py`.

## Phase 4 — Extract Counterfactual Scoring

- [x] Create `src/circuit/instrument/attribution_counterfactual.py`.
- [x] Move `compute_latent_counterfactual_scores` into `attribution_counterfactual.py`.
- [x] Preserve current patching, graph, and tensor semantics exactly; do not redesign the counterfactual algorithm during extraction.
- [x] Update `attribution.py` to import and re-export the moved function.
- [x] Verify with `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_patcher.py -q`.

### Phase 4 Notes

- Added `src/circuit/instrument/attribution_counterfactual.py` for `compute_latent_counterfactual_scores`, preserving target-scalar validation, upstream pair discovery, leaf-anchor gradient collection, active-count masking, per-scope top-K selection, and activator/inhibitor output semantics.
- Kept the local `pipeline.component_index.component_idx` import inside the scorer to avoid adding module-import coupling.
- Updated `src/circuit/instrument/attribution.py` to import and re-export `compute_latent_counterfactual_scores`, so existing imports from `circuit.instrument.attribution` continue to work.
- Verification on 2026-05-23: `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_patcher.py -q` -> `75 passed in 0.25s`.
- Lints/diagnostics reported no errors for `attribution.py` or `attribution_counterfactual.py`.

## Phase 5 — Extract Autograd Anchor And Edge Scoring Helpers

- [x] Create `src/circuit/instrument/attribution_autograd.py`.
- [x] Move `_upstream_anchors_for_target`, `_all_feature_anchors_with_meta`, `_score_grads_into_adj`, and `_score_token_grads_into_adj`.
- [x] Keep helper names private to the new module unless tests or callers explicitly import them.
- [x] Update direct-effects and upstream-score code to import these helpers from `attribution_autograd.py`.
- [x] Verify with `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_sae_graph.py -q`.

### Phase 5 Notes

- Added `src/circuit/instrument/attribution_autograd.py` for shared anchor collection and edge-scoring helpers used by direct-effects construction.
- Moved `_upstream_anchors_for_target`, `_all_feature_anchors_with_meta`, `_score_grads_into_adj`, and `_score_token_grads_into_adj` without changing causal upstream selection, all-anchor logit scoring, error-node scoring, feature-node scoring, or token-node attribution semantics.
- Kept the moved helper names private and imported them explicitly into `src/circuit/instrument/attribution.py` for the remaining direct-effects code.
- Verification on 2026-05-23: `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_sae_graph.py -q` -> `66 passed in 0.25s`.
- Lints/diagnostics reported no errors for `attribution.py` or `attribution_autograd.py`.

## Phase 6 — Extract Logit Target Selection And Direct Effects Matrix

- [x] Create `src/circuit/instrument/attribution_direct_effects.py`.
- [x] Move `_find_logit_targets`, `_compute_partial_one_hop_influence`, `compute_direct_effects_matrix`, and any private helpers used only by direct-effects matrix construction.
- [x] Keep the printed progress messages and node ordering stable unless tests explicitly permit changes.
- [x] Preserve current feature, error-node, token-node, and logit-sentinel ordering.
- [x] Update `attribution.py` to import and re-export `compute_direct_effects_matrix`.
- [x] Verify with `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_sae_graph.py tests/circuit/test_patcher.py -q`.

### Phase 6 Notes

- Added `src/circuit/instrument/attribution_direct_effects.py` for `_find_logit_targets`, `_compute_partial_one_hop_influence`, and `compute_direct_effects_matrix`.
- Preserved the existing logit target cumulative-probability selection, sparse one-hop logit influence ranking, online Neumann ranking, printed progress messages, and node ordering for feature, error, token, and logit sentinel nodes.
- Kept direct-effects dependencies explicit on active-node collection, autograd helper scoring, Neumann influence, and SAE graph instruments.
- Updated `src/circuit/instrument/attribution.py` to import and re-export `compute_direct_effects_matrix`, so existing imports from `circuit.instrument.attribution` continue to work.
- Verification on 2026-05-23: `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_sae_graph.py tests/circuit/test_patcher.py -q` -> `106 passed in 0.25s`.
- Lints/diagnostics reported no errors for `attribution.py` or `attribution_direct_effects.py`.

## Phase 7 — Extract Latent Upstream Scores

- [x] Create `src/circuit/instrument/attribution_upstream_scores.py`.
- [x] Move `compute_latent_upstream_scores` and any helpers used only by that path.
- [x] Import `UpstreamScores` from `attribution_types.py`.
- [x] Preserve attribution versus absent-gradient scoring semantics exactly.
- [x] Update `attribution.py` to import and re-export `compute_latent_upstream_scores`.
- [x] Verify with `python -m pytest tests/circuit/test_attribution.py tests/circuit/discovery/test_layerwise_gradient_upstream.py tests/circuit/discovery/test_counterfactual_gradient.py -q`.

### Phase 7 Notes

- Added `src/circuit/instrument/attribution_upstream_scores.py` for `compute_latent_upstream_scores`.
- Imported `UpstreamScores` from `src/circuit/instrument/attribution_types.py` in the new module, preserving the split `attribution` and `absent_gradient` return contract.
- Preserved predecessor component expansion, target scalar validation, shared backward pass, active-count masking, attribution top-K by absolute score, and absent-inhibitor top-K by most-negative gradient.
- Updated `src/circuit/instrument/attribution.py` to import and re-export `compute_latent_upstream_scores`, so existing imports from `circuit.instrument.attribution` continue to work.
- Verification on 2026-05-23: `python -m pytest tests/circuit/test_attribution.py tests/circuit/discovery/test_layerwise_gradient_upstream.py tests/circuit/discovery/test_counterfactual_gradient.py -q` -> `94 passed in 0.44s`.
- Lints/diagnostics reported no errors for `attribution.py` or `attribution_upstream_scores.py`.

## Phase 8 — Reduce `attribution.py` To A Compatibility Facade

- [x] Replace `src/circuit/instrument/attribution.py` with a small facade that imports public names from the sibling modules.
- [x] Define `__all__` in `attribution.py` for the stable public API.
- [x] Confirm all existing imports from `circuit.instrument.attribution` still resolve without caller changes.
- [x] Avoid broad test rewrites; only adjust tests if they intentionally inspect private helper ownership.
- [x] Confirm `src/circuit/instrument/attribution.py` is reduced to a small facade, ideally under 100 LOC.

### Phase 8 Notes

- Reduced `src/circuit/instrument/attribution.py` to a 16-line compatibility facade that imports the stable public attribution API from sibling modules.
- Added explicit `__all__` for `UpstreamScores`, `compute_logit_attribution`, `compute_feature_attribution`, `compute_feature_gradient`, `compute_latent_counterfactual_scores`, `compute_direct_effects_matrix`, and `compute_latent_upstream_scores`.
- Repository import search found old-facade imports in circuit discovery modules and tests; all target public names remain available from `circuit.instrument.attribution`.
- Import smoke check on 2026-05-23 confirmed the old facade exposes the expected public names.
- Focused verification on 2026-05-23: `python -m pytest tests/circuit/test_attribution.py -q` -> `35 passed in 0.17s`.
- Lints/diagnostics reported no errors for `attribution.py`.

## Phase 9 — Testing And Verification

- [x] Run focused attribution tests: `python -m pytest tests/circuit/test_attribution.py -q`.
- [x] Run nearby instrument tests: `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_patcher.py tests/circuit/test_sae_graph.py tests/model/test_hooks.py -q`.
- [x] Run discovery tests that depend on attribution behavior: `python -m pytest tests/circuit/discovery/test_counterfactual_gradient.py tests/circuit/discovery/test_layerwise_gradient_upstream.py tests/circuit/discovery/test_top_coact_attr.py -q`.
- [x] Run lints/diagnostics on all newly created and edited files.
- [x] Confirm no algorithm outputs, public imports, printed progress shape, or node ordering changed.
- [x] Confirm no generated artifacts, native binaries, or unrelated local outputs were touched by the refactor.

### Phase 9 Notes

- Focused attribution verification on 2026-05-23: `python -m pytest tests/circuit/test_attribution.py -q` -> `35 passed in 0.16s`.
- Nearby instrument verification on 2026-05-23: `python -m pytest tests/circuit/test_attribution.py tests/circuit/test_patcher.py tests/circuit/test_sae_graph.py tests/model/test_hooks.py -q` -> `139 passed in 0.29s`.
- Discovery attribution verification on 2026-05-23: `python -m pytest tests/circuit/discovery/test_counterfactual_gradient.py tests/circuit/discovery/test_layerwise_gradient_upstream.py tests/circuit/discovery/test_top_coact_attr.py -q` -> `60 passed in 0.30s`.
- Lints/diagnostics reported no errors for `attribution.py` or the extracted sibling attribution modules.
- The refactor preserved the old public import surface through `circuit.instrument.attribution`, including the explicit facade `__all__`.
- No behavior-focused test rewrites were needed; the existing tests cover the moved logit, feature, counterfactual, direct-effects, and upstream-score paths.
- `git status --short` shows this refactor's attribution source/planning changes plus pre-existing unrelated work such as `.gitignore`, `README.md`, multi-device planning, and earlier pass-1/pass-2 refactor files; no generated artifacts, native binaries, or output-run files were touched by this refactor.

---

## Open Questions

- Should private helpers such as `_score_grads_into_adj` remain private module functions, or should direct-effects code expose a small public helper API for tests?
- Should `_find_logit_targets` live in `attribution_logit.py` or `attribution_direct_effects.py` if it is primarily used by direct-effects matrix construction?
- Should repeated autograd anchor collection logic be represented by small dataclasses instead of parallel lists of anchors and metadata?
- Should this eventually become an `attribution/` package after a separate migration removes the `attribution.py` file?

## Risks / Assumptions

- The primary risk is accidental behavior drift in autograd graph retention, anchor ordering, or node ordering.
- Using sibling modules avoids the package/file import conflict and keeps the old `circuit.instrument.attribution` import path stable.
- `compute_direct_effects_matrix` should be extracted late because it is the largest and most coupled path.
- This refactor should not redesign attribution algorithms; it should only move code into clearer ownership boundaries.
