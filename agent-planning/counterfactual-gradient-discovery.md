# Plan: Counterfactual Gradient Discovery

> **Goal:** Implement `CounterfactualGradientDiscovery`, a new `DiscoveryMethod` that runs gradient attribution on **negctx sequences** to discover two classes of circuit node: absent activators (latents whose presence would cause the seed to fire) and present inhibitors (latents whose activation is actively suppressing the seed).
>
> **Created:** 2026-04-12

---

## Background

All existing gradient methods (`logit_attribution`, `sfc_attribution_patching`, `gradient_upstream`) run on posctx — sequences where the seed fires. This method runs on **negctx** — semantically similar sequences where the seed is inactive. The gradient of the seed's activation w.r.t. upstream latents on negctx reveals:

- **Absent activators**: upstream latents with a high positive gradient but low/zero activation on negctx. They would cause the seed to fire if active — their absence explains the suppression.
- **Present inhibitors**: upstream latents with a high negative gradient-activation product (`activation × gradient < 0`) on negctx. They are active and causally suppressing the seed.

### How negctx is accessed

`ProbeDataset.neg_tokens` is already populated by `build_probe_dataset` (via `neg_ctx.ctx_seq_idx[comp_idx, latent_idx]`). No separate loading step is needed — just use `probe_data.neg_tokens[:max_neg_sequences]` directly.

### The zero-activation problem

The seed latent is expected to have very low or zero SAE activation on negctx sequences (that's the definition of negctx). The SAE uses top-k sparsification: if the seed latent never appears in the top-k for any (batch, token) position on negctx, then `f_connected[..., seed_latent_idx]` is identically 0, and the backward through `scatter_` produces zero gradients at the seed's SAE output. The gradient propagation to upstream leaf anchors will be zero — no useful signal.

**Mitigation — encoder direction projection (preferred):** Instead of targeting `f_connected[..., seed_latent_idx]` (which is identically 0 if the seed never appears in SAE top-k on negctx), compute the seed latent's encoder pre-activation directly from the residual stream `x`:

```python
w_seed = sae_bank.saes[seed_kind][seed_layer].W_enc[seed_latent_idx]  # [d_model]
b_seed = sae_bank.saes[seed_kind][seed_layer].b_enc[seed_latent_idx]  # scalar

# x is already in the computation graph via SAEGraphInstrument's identity passthrough
pre_act_neg = x @ w_seed + b_seed  # [B, T] — always non-zero, always differentiable
```

The encoder direction (`W_enc[seed_latent_idx]`) is the pattern the residual stream needs to resemble for the seed to fire — the correct target for "what would need to change to make the seed activate." This is distinct from the decoder direction (`W_dec[:, seed_latent_idx]`), which is what the seed writes into the stream when it fires.

For the target scalar, use an **MSE loss** against the seed's known posctx activation:

```python
# target_act_pos: seed's actual SAE activation on posctx (computed cheaply via no-grad forward)
loss = ((pre_act_neg[batch_idx, pos_argmax] - target_act_pos) ** 2).mean()
target_scalar = -loss  # minimise MSE = maximise pre_act toward posctx level
```

The MSE formulation scales the gradient by the deficit `(pre_act_neg - target_act_pos)` — latents that would help close the gap between negctx and posctx get proportionally larger scores. This is more informative than just maximising `pre_act_neg` unconditionally.

**Implementation:** Subclass `SAEGraphInstrument` as `SeedProjectionInstrument`, which overrides `transform` to store `seed_pre_act = x @ w_seed + b_seed` when `layer_idx == seed_layer and kind == seed_kind`, then calls `super().transform()`. The stored tensor is in the computation graph via the parent class's identity passthrough `(x - x.detach())`.

**Computing `target_act_pos`:** Run one no-grad forward on `probe_data.pos_tokens` to get the seed's actual SAE activation (e.g., via the existing `_calculate_argmax` hook pattern, recording the activation value rather than the argmax). This is one cheap extra pass and is done once per seed before the grad-enabled negctx pass.

**Mitigation — fallback:** If the encoder direction approach has not yet been implemented, add a guard in `_run_negctx_hop`: check if `probe_data.neg_tokens` is non-empty AND if `f_connected[..., seed_latent_idx].sum().abs() < 1e-6`. If so, log a warning and return `({}, {})` rather than running a useless backward.

For Phase 1 implementation, use the fallback guard first. The decoder direction approach can be added as a follow-up.

### Single-pass dual scoring

Run one `SAEGraphInstrument` forward + one `autograd.grad` backward on negctx. From the same gradient tensors, compute two scores:

- **Activator score** for latent U = raw gradient `∂(seed_act)/∂(act_U)` — signal strength regardless of whether U is currently active.
- **Inhibitor score** for latent U = `activation(U) × gradient` where the product is negative — U is both active AND causally suppressing the seed.

This is a single function `compute_latent_counterfactual_scores` that returns two dicts from one backward pass.

### Reuse of existing infrastructure

| Existing component | Reuse |
|---|---|
| `SAEGraphInstrument` | Unchanged |
| `get_predecessor_components` | Unchanged |
| `evaluate_upstream_faithfulness` | Unchanged |
| `CircuitPatcher` with `max_layer` | Unchanged |
| `build_probe_dataset` | Unchanged — `neg_tokens` already in result |

---

## Phase 1 — Config & Pydantic Schema ✅

- [x] Add `CounterfactualGradientConfig` to `src/config.py`:
  ```python
  class CounterfactualGradientConfig(BaseModel):
      model_config = ConfigDict(extra='forbid')
      top_k_activators: int = 8         # top-K absent activators to include
      top_k_inhibitors: int = 8         # top-K present inhibitors to include
      activator_threshold: float = 0.01 # min raw gradient magnitude for activators
      inhibitor_threshold: float = 0.01 # min |activation × gradient| for inhibitors
      min_active_count: int = 1         # skip latents below this global firing count
      max_neg_sequences: int = 4        # max negctx sequences for the grad-enabled pass
      pruning_threshold: float = 0.0    # faithfulness drop threshold for minimality pruning
      min_faithfulness: float = 0.2     # minimum upstream_faithfulness to accept circuit
  ```
- [x] Add `counterfactual_gradient: CounterfactualGradientConfig = Field(default_factory=CounterfactualGradientConfig)` to `DiscoveryConfig`.
- [x] Add to `config.yaml` under `discovery:`:
  ```yaml
  counterfactual_gradient:
    top_k_activators: 8          # Top-K absent activators (positive gradient on negctx)
    top_k_inhibitors: 8          # Top-K present inhibitors (negative act×grad on negctx)
    activator_threshold: 0.01    # Min raw gradient magnitude
    inhibitor_threshold: 0.01    # Min |activation × gradient| for inhibitors
    min_active_count: 1          # Min global lifetime firing count
    max_neg_sequences: 4         # Max negctx sequences for the grad-enabled pass
    pruning_threshold: 0.0       # Faithfulness drop threshold for pruning (0 = disabled)
    min_faithfulness: 0.2        # Min upstream_faithfulness to accept
  ```
- [x] Add `"counterfactual_gradient"` to the comment block listing available methods in `config.yaml`.

## Phase 2 — `compute_latent_counterfactual_scores` in `attribution.py` ✅

Add a new function to `src/circuit/instrument/attribution.py`. This is a single-backward-pass function that produces both activator and inhibitor scores from the same gradient tensors.

- [x] Implement `compute_latent_counterfactual_scores(graph, target_scalar, seed_layer, n_kinds, kinds, top_k_activators, top_k_inhibitors, min_active_count, active_count) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float]]`:

  The caller is responsible for constructing `target_scalar` (encoder projection MSE or fallback). This function scores ALL upstream latents across all layers ≤ `seed_layer` in one backward pass — no BFS or hop restriction needed. This is what distinguishes it from `compute_latent_upstream_scores`, which scores only direct predecessor components.

  1. If `target_scalar.grad_fn is None`, return `({}, {})`.
  2. Iterate over ALL `(layer_p, kind_p)` entries in `graph.activations` where `layer_p <= seed_layer` (not just direct predecessors — the gradient flows through the full upstream graph in one backward pass, so all upstream latents can be scored at once). No `predecessor_comp_indices` argument needed.
  5. Collect leaf anchors (`state_grad.act`) for only these predecessor (layer_p, kind_p) pairs from `graph.activations`.
  6. Run `torch.autograd.grad(target_scalar, anchors, retain_graph=True, allow_unused=True)`.
  7. For each predecessor (layer_p, kind_p):
     - Get `(acts_grad, _, _)` from `graph.get_latents(layer_p, kind_p)` — first return value (detached leaf).
     - Match gradient from step 6 (same iteration order as anchor collection).
     - Compute `attr_act = acts_grad.act * grad` → `[B, T, d_sae]`. Sum → `scores_inhibitor = attr_act.sum(dim=(0,1))` → `[d_sae]`.
     - Compute `scores_activator = grad.sum(dim=(0,1))` → `[d_sae]` (raw gradient, no activation scaling).
     - Apply `min_active_count` filter: `comp_p = component_idx(layer_p, kinds.index(kind_p), n_kinds)`; zero out both score tensors where `active_count[comp_p] < min_active_count`.
     - From `scores_activator`: collect top-`top_k_activators` by `scores_activator` (positive only, `score > 0`).
     - From `scores_inhibitor`: collect top-`top_k_inhibitors` by `scores_inhibitor.abs()` where `scores_inhibitor < 0`.
     - Convert each to `FeatureID(layer=layer_p, kind=kind_p, index=int(idx))` and accumulate.
  8. Merge across all predecessor components. Re-rank activators by raw gradient (descending). Re-rank inhibitors by `|inhibitor_score|` (descending). Slice to `top_k_activators` and `top_k_inhibitors` respectively.
  9. Return `(activator_scores: Dict[FeatureID, float], inhibitor_scores: Dict[FeatureID, float])`.

  **Note:** The raw gradient for activators (`grad.sum(dim=(0,1))`) is NOT scaled by the current activation. This is intentional — a latent with zero activation on negctx but large positive gradient is exactly the "absent activator" we're looking for. Existing `compute_feature_gradient` operates on a candidate list; this function is the vectorised, no-candidate-list equivalent.

## Phase 3 — Implement `CounterfactualGradientDiscovery` ✅

Created `src/circuit/discovery/counterfactual_gradient.py`:

- [x] Define `class CounterfactualGradientDiscovery(DiscoveryMethod)`.
- [x] `__init__`: read all params from `config.discovery.counterfactual_gradient`.
- [x] Implement `discover(seed_comp_idx, seed_latent_idx) -> Optional[Circuit]` with logger wrapper.
- [x] Implement `_discover(seed_comp_idx, seed_latent_idx, logger)`:

  ```
  1. Build probe dataset for seed → probe_data (contains both pos_tokens and neg_tokens).
  2. If probe_data.neg_tokens is empty, reject ("no negctx sequences available").
  3. Add seed node to circuit (role="seed").

  4. Run _run_negctx_hop(seed_comp_idx, seed_latent_idx, neg_tokens, target_act_pos):
     - Disable compile.
     - Run SeedProjectionInstrument forward pass on neg_tokens (grad_enabled=True).
     - Check guard: if instrument.seed_pre_act is None, return {}, {}.
     - Extract pos_argmax_neg: argmax of seed_pre_act over token dimension → [B].
     - Compute target_scalar from instrument.seed_pre_act and target_act_pos (MSE-derived, see Background).
     - Call compute_latent_counterfactual_scores(graph, target_scalar, seed_layer, ...) — scores ALL upstream layers ≤ seed_layer at once, no hop restriction.
     - Free instrument, enable compile, empty CUDA cache.
     - Return (activator_scores, inhibitor_scores).

  5. For each (upstream_fid, score) in activator_scores:
     - Skip if |score| < activator_threshold.
     - Skip if latent_stats.active_count < min_active_count.
     - Add to circuit (role="counterfactual_activator", score in metadata).
     - Add directional edge: upstream_fid → seed_fid, weight=score.

  6. For each (upstream_fid, score) in inhibitor_scores:
     - Skip if |score| < inhibitor_threshold.
     - Skip if latent_stats.active_count < min_active_count.
     - Add to circuit (role="counterfactual_inhibitor", score in metadata).
     - Add directional edge: upstream_fid → seed_fid, weight=score.

  7. If len(circuit.nodes) <= 1, reject ("no activators or inhibitors found").

  8. Evaluation (on pos_tokens, layer-bounded to seed_layer):
     - upstream_faithfulness via evaluate_upstream_faithfulness(... max_layer=seed_layer)
     - Standard faithfulness, sufficiency, completeness with max_layer=seed_layer.
     - Log all four scores. Accept if upstream_faithfulness >= min_faithfulness.
  ```

- [x] Implement `_get_posctx_activation(seed_comp_idx, seed_latent_idx, pos_tokens, pos_argmax) -> float`:
  - Run one no-grad forward on `pos_tokens` using an `activations_callback` hook at `seed_layer`.
  - Records the seed latent's SAE activation at `pos_argmax` positions across the batch.
  - Returns the mean seed activation — used as `target_act_pos` for the MSE loss.

- [x] Implement `_run_negctx_hop(comp_idx, latent_idx, neg_tokens, target_act_pos, logger) -> Tuple[Dict, Dict]`:
  - Wraps `SeedProjectionInstrument` forward + `compute_latent_counterfactual_scores`.
  - Creates `SeedProjectionInstrument` with `w_seed = sae.encoder.weight[seed_latent_idx]` and `b_seed = sae._get_bias_eff()[seed_latent_idx]`.
  - After forward, checks guard: if `instrument.seed_pre_act is None`, returns `({}, {})`.
  - Computes `target_scalar = -MSE(seed_pre_act[at argmax], target_act_pos)`.
  - Checks near-zero guard: `abs(target_scalar.item()) < 1e-8` → returns `({}, {})`.
  - Passes `target_scalar` into `compute_latent_counterfactual_scores`.
  - `self.inference.disable_compile()` before, `self.inference.enable_compile()` in `finally`.
  - `del instrument; torch.cuda.empty_cache(); gc.collect()` in `finally` block.

- [x] Added `SeedProjectionInstrument` class in `src/circuit/discovery/counterfactual_gradient.py`:
  ```python
  class SeedProjectionInstrument(SAEGraphInstrument):
      def __init__(self, bank, seed_layer, seed_kind, w_seed, b_seed):
          super().__init__(bank)
          self.seed_layer = seed_layer
          self.seed_kind = seed_kind
          self.w_seed = w_seed  # [d_model]
          self.b_seed = b_seed  # scalar
          self.seed_pre_act = None  # set during forward: [B, T]

      def transform(self, layer_idx, kind, x):
          result = super().transform(layer_idx, kind, x)
          if layer_idx == self.seed_layer and kind == self.seed_kind:
              w = self.w_seed.to(device=x.device, dtype=x.dtype)
              b = self.b_seed.to(device=x.device, dtype=x.dtype)
              self.seed_pre_act = x @ w + b  # [B, T]
          return result
  ```
  Note: capture `seed_pre_act` from `x` (the pre-SAE residual stream) AFTER calling `super().transform()` to ensure the identity passthrough `(x - x.detach())` is already in the computation graph. The `.to(device, dtype)` handles multi-GPU / mixed-precision cases. `w_seed` and `b_seed` are detached (no SAE encoder gradients).

## Phase 4 — Register in `discovery_window.py` ✅

- [x] Add import: `from circuit.discovery.counterfactual_gradient import CounterfactualGradientDiscovery`
- [x] Add to `METHOD_REGISTRY`: `"counterfactual_gradient": CounterfactualGradientDiscovery`
- [x] Add to `_build_methods` docstring: `"counterfactual_gradient" — negctx gradient attribution for absent activators and present inhibitors`

## Phase 5 — Validation ✅

Implemented as `tests/circuit/discovery/test_counterfactual_gradient.py` — 32 tests, all passing.

**`TestComputeLatentCounterfactualScores` (15 tests)** — oracle and contract tests:
- [x] Activator scores are positive, inhibitor scores are negative (sign contracts).
- [x] Activators and inhibitors are disjoint sets.
- [x] Exact oracle values verified: latent 0 → activator score +1.0, latent 2 → inhibitor score -1.0.
- [x] Zero-gradient latent excluded from both dicts.
- [x] `target_scalar.grad_fn is None` → both dicts empty (guard verified).
- [x] `seed_layer` boundary: layers above `seed_layer` excluded; layers ≤ seed_layer included.
- [x] `top_k_activators` / `top_k_inhibitors` limits respected; `top_k=0` returns empty dict.
- [x] `min_active_count` filtering: zero active_count blocks all latents; full count passes all.
- [x] Return types: both outputs are `Dict[FeatureID, float]`.

**`TestSeedProjectionInstrument` (6 tests)** — instrument behaviour:
- [x] `seed_pre_act` is populated after forward pass (not None).
- [x] `seed_pre_act` has shape `[B, T]`.
- [x] `seed_pre_act` is differentiable (has `grad_fn`).
- [x] Changing `w_seed` changes `seed_pre_act` values (confirms projection is onto `w_seed`).
- [x] Non-matching `seed_layer=99` leaves `seed_pre_act = None`.
- [x] Gradient flows from `seed_pre_act` to upstream leaf anchors when `seed_layer=1` (upstream layer exists).

**`TestCounterfactualGradientDiscovery` (11 tests)** — mock-based integration:
- [x] Rejects empty negctx / empty posctx.
- [x] Rejects when no activators or inhibitors pass threshold.
- [x] Rejects when `upstream_faithfulness < min_faithfulness`.
- [x] Activator nodes have `role="counterfactual_activator"` with positive `attribution_score`.
- [x] Inhibitor nodes have `role="counterfactual_inhibitor"` with negative `attribution_score`.
- [x] Both role types present when both score dicts are non-empty.
- [x] Accepted circuit metadata contains all required keys including `upstream_faithfulness`.
- [x] Nodes with `|score| < threshold` are filtered → circuit rejected.

---

## Open Questions

- **Encoder projection as primary path vs. fallback ordering:** The plan describes the encoder direction MSE approach as preferred and the zero-activation guard as a fallback. In practice, implement the encoder direction approach first (it is not significantly more complex than the `f_connected` approach and avoids the entire zero-activation problem). The fallback guard is a one-liner safety net for unexpected cases.
- **Evaluation on negctx directly:** A natural evaluation would be: "if we set the activators to their posctx activation values and zero the inhibitors on negctx, does the seed fire?" This is a targeted counterfactual intervention on negctx, different from the standard posctx-based evals. Worth adding as an additional metric once the basic method is working.
- **Interaction with GradientUpstreamDiscovery:** It may be informative to run both methods on the same seed and compare the two circuit halves: posctx-gradient finds "what drives the seed," negctx-gradient finds "what gates the seed." Together they form a more complete picture.
- **Depth:** This is a single-hop method (only predecessor components of the seed are scored). Multi-hop (using the discovered activators' own negctx for deeper attribution) is possible but adds complexity. Not in scope for Phase 1.
- **Using posctx pos_argmax vs. negctx argmax for gradient target:** The plan uses `argmax_neg` extracted from the negctx graph (maximum seed activation position on negctx). An alternative is to use the seed's posctx `pos_argmax` as a fixed evaluation position. Negctx argmax is more principled but may pick arbitrary positions when activation is near-zero.

## Risks / Assumptions

- `self.inference.disable_compile()` / `enable_compile()` must bracket the grad-enabled forward pass, exactly as in `logit_attribution.py` and `gradient_upstream.py`.
- `del instrument; torch.cuda.empty_cache()` in `_run_negctx_hop`'s `finally` block is mandatory — same VRAM accumulation risk as GradientUpstreamDiscovery.
- The near-zero-activation guard (`abs(target_scalar.item()) < 1e-6`) may cause many seeds to be rejected if negctx sequences consistently have the seed latent below SAE top-k threshold. This is expected and handled gracefully. The decoder direction approach (open question) would reduce the rejection rate.
- `compute_latent_counterfactual_scores` uses `retain_graph=True`. Since the instrument is freed immediately after the function returns, this is harmless.
- The evaluations run on **posctx sequences** (`probe_data.pos_tokens`), not negctx. The grad-enabled forward runs on negctx. The eval runs on posctx. These are intentionally separate: discovery uses negctx, evaluation tests whether the discovered nodes explain the seed's activation on posctx.
