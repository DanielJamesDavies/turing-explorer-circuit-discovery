# Plan: Gradient Upstream Discovery

> **Goal:** Implement `GradientUpstreamDiscovery`, a new `DiscoveryMethod` that builds a circuit by propagating gradient attribution backwards through the model layer-by-layer, using each discovered latent's own `top_ctx` sequences at each hop rather than always using the seed's context.
>
> **Created:** 2026-04-11

---

## Background

The idea: for a seed latent S at (layer L, kind K), compute `score(U) = activation(U) × ∂activation(S)/∂activation(U)` for all latents U in S's predecessor components. Take top-K, add them to the circuit. For each selected upstream latent U, repeat — but switch to U's own `top_ctx` sequences. This creates a causal upstream tree grounded in each node's most representative context.

### How it fits into the existing codebase

The gradient computation is already handled by two existing primitives:

1. `**SAEGraphInstrument`** (`src/circuit/instrument/sae_graph.py`): instruments a `grad_enabled=True` forward pass and stores per-layer leaf anchors in a `FeatureGraph`.
2. `**compute_feature_attribution`** (`src/circuit/instrument/attribution.py`): takes a `FeatureGraph` + target latent + `pos_argmax` and returns `{FeatureID: activation × gradient}` scores for candidate upstream latents.

`LogitAttribution` already uses these — but it calls `compute_feature_attribution` with a pre-specified candidate list (from `top_coactivation`) to score edges between already-identified nodes. `GradientUpstreamDiscovery` repurposes them for **node discovery**: at each hop, score all latents in the predecessor components (not just co-activation neighbors) to decide which new nodes to add.

### Predecessor component structure

The transformer's residual arithmetic defines which components feed directly into each component at layer L:


| Target kind        | Predecessor components                                    |
| ------------------ | --------------------------------------------------------- |
| `attn` at layer L  | `resid` at layer L-1                                      |
| `mlp` at layer L   | `resid` at layer L-1, `attn` at layer L                   |
| `resid` at layer L | `resid` at layer L-1, `attn` at layer L, `mlp` at layer L |


This is implemented as a helper in Phase 2.

### Key design: vectorised predecessor scoring

Rather than looping over individual candidate latents, we score the full predecessor components in one tensor operation. The `FeatureGraph` stores `attr_act: [B, T, d_sae]` for every (layer, kind). For a predecessor component (layer_p, kind_p):

```
scores[d_sae] = (attr_act * grad).sum(dim=(0,1))  # sum over batch and time
```

This gives per-latent attribution scores for the whole component in O(B × T × d_sae) rather than O(N_candidates × ...). A new function `compute_latent_upstream_scores` in `attribution.py` implements this.

### Argmax extraction from grad-enabled pass

`compute_feature_attribution` needs `pos_argmax` — the token position of peak activation for the target latent. Rather than running a separate no-grad pass to compute this (as `probe_dataset._calculate_argmax` does), we extract it directly from the `FeatureGraph` after the grad-enabled pass:

```python
_, acts_connected, _ = graph.get_latents(target_layer, target_kind)
argmax = acts_connected.act[..., target_latent_idx].argmax(dim=-1)  # [B]
```

`graph.get_latents` returns `(state_grad, state_connected, top_indices)` — the connected activations are the second return value. `state_connected.act` is the original dense `[B, T, d_sae]` feature tensor, still in the computation graph. Using the detached `state_grad` would give the same argmax values but is misleading.

This saves one forward pass per hop.

---

## Phase 1 — Config & Pydantic Schema ✅ Completed

- Add `GradientUpstreamConfig` to `src/config.py`:
  ```python
  class GradientUpstreamConfig(BaseModel):
      model_config = ConfigDict(extra='forbid')
      depth: int = 3                    # number of backward hops from the seed
      top_k_per_hop: int = 8            # top-K upstream latents to select per node per hop
      attribution_threshold: float = 0.01  # min |score| to include a latent
      min_active_count: int = 1         # skip latents below this global firing count
      max_ctx_sequences: int = 4        # max ctx sequences to use per node (probe_batch_size equiv)
      pruning_threshold: float = 0.0    # faithfulness drop threshold for minimality pruning
      min_faithfulness: float = 0.2     # minimum faithfulness score for the circuit to be accepted
  ```
- Add `gradient_upstream: GradientUpstreamConfig = Field(default_factory=GradientUpstreamConfig)` to `DiscoveryConfig`.
- Add to `config.yaml` under `discovery:`:
  ```yaml
  gradient_upstream:
    depth: 3                  # Backward hops from seed
    top_k_per_hop: 8          # Top-K upstream latents per node per hop
    attribution_threshold: 0.01  # Min |activation × gradient| to include a latent
    min_active_count: 1       # Min global lifetime firing count
    max_ctx_sequences: 4      # Max ctx sequences per node for the grad-enabled pass
    pruning_threshold: 0.0    # Faithfulness drop threshold for minimality pruning (0 = disabled)
    min_faithfulness: 0.2     # Minimum faithfulness score for the circuit to be accepted
  ```
- Add `"gradient_upstream"` to the comment block listing available methods in `config.yaml`.

## Phase 2 — `get_predecessor_components` Helper ✅ Completed

- Implement `get_predecessor_components(comp_idx: int, n_kinds: int, kinds: List[str]) -> List[int]`:
  - Decompose `comp_idx` into `(layer, kind_idx)` via `split_component_idx`.
  - `kind = kinds[kind_idx]`
  - Return list of predecessor component indices:
    - If `kind == "attn"` and `layer > 0`: `[component_idx(layer-1, kinds.index("resid"), n_kinds)]`
    - If `kind == "mlp"`: `[component_idx(layer-1, kinds.index("resid"), n_kinds), component_idx(layer, kinds.index("attn"), n_kinds)]` (only resid_(L-1) if layer==0, only resid if attn doesn't exist for that layer)
    - If `kind == "resid"`: resid_(L-1) + attn_L + mlp_L (skip resid if layer==0)
    - If `layer == 0` and the predecessor would be layer -1: return empty list (no predecessors — at the embedding layer).
  - Guard: skip any predecessor component that doesn't exist (e.g., if n_layers=1 or layer=0).

## Phase 3 — `compute_latent_upstream_scores` in `attribution.py` ✅ Completed

Add a new function to `src/circuit/instrument/attribution.py`:

- Implement `compute_latent_upstream_scores(graph, target_layer, target_kind, target_latent_idx, pos_argmax, predecessor_comp_indices, n_kinds, kinds, top_k, min_active_count, active_count) -> Dict[FeatureID, float]`:
  1. Get `(_, target_acts_connected, _)` from `graph.get_latents(target_layer, target_kind)` — second return value.
  2. Compute `target_scalar = target_acts_connected.act[batch_idx, pos_argmax, target_latent_idx].sum()`.
  3. If `target_scalar.grad_fn is None`, return `{}`.
  4. Convert each `comp_idx` in `predecessor_comp_indices` to `(layer_p, kind_p)` using `split_component_idx(comp_idx, n_kinds)` → `(layer_p, kind_idx_p)` → `kind_p = kinds[kind_idx_p]`. Collect leaf anchors (`state_grad.act`) for only these (layer_p, kind_p) pairs from `graph.activations`.
  5. Run `torch.autograd.grad(target_scalar, anchors, retain_graph=True, allow_unused=True)`.
  6. For each predecessor (layer_p, kind_p):
    - Get `(acts_grad, _, _)` from `graph.get_latents(layer_p, kind_p)` — first return value is the detached leaf.
    - Match the corresponding gradient from step 5 by iterating anchors in the same order they were collected.
    - Compute `attr_act = acts_grad.act * grad` → shape `[B, T, d_sae]`.
    - Sum over batch and token dimensions: `scores = attr_act.sum(dim=(0, 1))` → `[d_sae]`.
    - Apply `min_active_count` filter: `comp_p = component_idx(layer_p, kinds.index(kind_p), n_kinds)`; zero out `scores` where `active_count[comp_p] < min_active_count`.
    - Get top-K by `scores.abs()`.
    - Convert to `Dict[FeatureID, float]` using `FeatureID(layer=layer_p, kind=kind_p, index=int(idx))`.
  7. Merge top-K results across all predecessor components, re-rank globally, return top `top_k` overall.
  **Note:** this function is closely analogous to `compute_feature_attribution` but returns vectorised per-component scores instead of per-candidate scores, and selects top-K internally rather than relying on a caller-supplied candidate list.

## Phase 4 — Implement `GradientUpstreamDiscovery` ✅ Completed

Create `src/circuit/discovery/gradient_upstream.py`:

- Define `class GradientUpstreamDiscovery(DiscoveryMethod)`.
- `__init`__: read all params from `config.discovery.gradient_upstream`. Store `self.depth`, `self.top_k_per_hop`, `self.attribution_threshold`, `self.min_active_count`, `self.max_ctx_sequences`, `self.pruning_threshold`, `self.min_faithfulness`.
- Implement `discover(seed_comp_idx, seed_latent_idx) -> Optional[Circuit]` with logger wrapper (same pattern as all other methods).
- Implement `_discover(seed_comp_idx, seed_latent_idx, logger)`:
  - (Discovery logic implemented; evaluation calls will be added in Phase 5)
- Implement `_run_hop(comp_idx, latent_idx, tokens) -> Dict[FeatureID, float]`:
  - Wraps the `SAEGraphInstrument` forward + `compute_latent_upstream_scores` call.
  - Always calls `self.inference.disable_compile()` before and `self.inference.enable_compile()` after (same pattern as `logit_attribution.py`).
  - Deletes instrument and calls `torch.cuda.empty_cache()` after extracting scores — critical, as each hop creates a new computation graph that must be freed before the next hop.

## Phase 5 — Evaluation ✅ Completed

All evaluations share one constraint: **the patcher only intervenes on components at layers ≤ `seed_layer`**. Components at layers > `seed_layer` run completely normally, consuming whatever activations the patched layers produce. This is consistent with the method's scope — it explains the upstream drivers of one specific latent, not the entire model.

### 5a — `upstream_faithfulness` (new metric) ✅ Completed

Measures how well the discovered upstream nodes explain *why the seed latent fires*, independent of the model's final output.

Three forward passes on `seed_pos_tokens`, all with layer-bounded intervention (layers ≤ seed_layer only):

1. **Full model** (no ablation) → record seed latent activation `a_full`
2. **Circuit only** (ablate non-circuit latents in layers ≤ seed_layer) → record `a_circuit`
3. **All ablated** (ablate everything in layers ≤ seed_layer) → record `a_ablated`

```
upstream_faithfulness = (a_circuit − a_ablated) / (a_full − a_ablated)
```

- 1.0 = the circuit upstream nodes fully explain the seed's activation above baseline
- 0.0 = the circuit upstream nodes explain none of it

The seed latent's activation is extracted as the mean over the probe batch at each sequence's peak position (`pos_argmax`). If `a_full − a_ablated ≈ 0` (seed barely activates above baseline), skip evaluation and reject.

Implement as `evaluate_upstream_faithfulness` in `src/eval/upstream_faithfulness.py`. Requires a `max_layer`-bounded `CircuitPatcher` and a lightweight activation capture hook — both described below.

**Extracting the seed latent activation scalar:** `inference.forward` only returns logits, not intermediate SAE activations. To read the seed latent's activation during each of the three passes, use a minimal capture patcher that wraps `CircuitPatcher`: after the SAE encode step at `(seed_layer, seed_kind)`, record `top_acts[batch, pos_argmax, :]` where the seed latent index matches, then let the existing patcher logic continue normally. Concretely, subclass or compose `CircuitPatcher` with an override in `transform` that, at `(seed_layer, seed_kind)`, encodes `x`, reads off `acts[..., seed_latent_idx]` at `pos_argmax`, stores the mean as `self.captured_activation`, then proceeds with the normal patching path. The three passes each use a fresh instance; after each `inference.forward` call, read `.captured_activation`. ✅ Completed (Implemented via `SeedActivationCapturePatcher` subclass in `upstream_faithfulness.py`).

### 5b — Standard evals (faithfulness, sufficiency, completeness) ✅ Completed

Uses the existing `evaluate_faithfulness`, `evaluate_sufficiency`, `evaluate_completeness` functions, which **internally construct their own `CircuitPatcher`**. These functions do not accept an external patcher — to introduce layer-bounding, add a `max_layer: Optional[int] = None` parameter to `CircuitPatcher.__init__`. In `CircuitPatcher.transform`, add an early return before any intervention:

```python
if self.max_layer is not None and layer_idx > self.max_layer:
    return x
```

`CircuitPatcher` already has `patch_kinds` for kind-level filtering — `max_layer` is a direct parallel. Then thread `max_layer=seed_layer` through to each eval function call. The minimal change to the existing eval functions is adding `max_layer: Optional[int] = None` and passing it to their internal `CircuitPatcher(...)` constructor calls.

- **Faithfulness**: ablate non-circuit latents in layers ≤ seed_layer → measure logit for target token at peak position
- **Sufficiency**: same ablation → measure whether target token is still top prediction
- **Completeness**: ablate *circuit* latents in layers ≤ seed_layer (`inverse=True`) → measure logit drop
- Add `max_layer: Optional[int] = None` to `CircuitPatcher.__init`__ and the early-return guard in `transform`.
- Thread `max_layer` through `evaluate_faithfulness`, `evaluate_sufficiency`, `evaluate_completeness` (add optional param, pass to their internal `CircuitPatcher` calls). No logic changes otherwise — these are non-breaking additions.
- Implement `evaluate_upstream_faithfulness(inference, sae_bank, avg_acts, circuit, seed_layer, seed_kind, seed_latent_idx, pos_tokens, pos_argmax) -> float` in `src/eval/upstream_faithfulness.py`. Decorate with `@torch.no_grad()`. Internally uses the activation-capturing `CircuitPatcher` subclass for the three passes described in 5a.
- In `GradientUpstreamDiscovery._discover`, run all four evaluations passing `max_layer=seed_layer`, log all four scores. Accept/reject based on `upstream_faithfulness >= min_faithfulness`.

## Phase 6 — Register in `discovery_window.py` ✅ Completed

- Add import: `from circuit.discovery.gradient_upstream import GradientUpstreamDiscovery`
- Add to `METHOD_REGISTRY`: `"gradient_upstream": GradientUpstreamDiscovery`
- Add to `_build_methods` docstring: `"gradient_upstream" — backwards gradient BFS with per-node context switching`

## Phase 7 — Validation

- Run with `"gradient_upstream"` on a small config (`n_shards: 1`, `n_seeds: 8`, `depth: 2`, `top_k_per_hop: 4`) and confirm: (a) no OOM (each hop's graph is freed), (b) circuits have nodes at multiple layer depths (not just layer L-1), (c) the edges go in the right direction (upstream → downstream).
- Spot-check: for a seed at (layer 8, resid), verify that the discovered upstream nodes include latents from (layer 7, resid), (layer 8, attn), (layer 8, mlp) — all three expected predecessor components.
- Confirm both `upstream_faithfulness` and standard faithfulness are logged per circuit. Verify `upstream_faithfulness` is higher than standard faithfulness (expected: the upstream circuit should be better at explaining its own seed than at explaining the final output).
- Compare circuit size and faithfulness against `logit_attribution` on the same seeds to establish a baseline comparison.

---

## Open Questions

- **Queue ordering (BFS vs priority queue):** The current plan uses strict BFS (process all hop-1 nodes before hop-2). An alternative is a priority queue ordered by attribution score, which would explore the highest-scoring upstream nodes first regardless of depth. This could produce better circuits with fewer hops.
- `**build_probe_dataset` cost per hop:** Each upstream latent's ctx loading calls `build_probe_dataset`, which itself runs a no-grad forward pass to compute `pos_argmax`. With `top_k_per_hop=8` and `depth=3`, this is up to 24 probe builds (plus the 24 grad-enabled hop passes). The argmax-from-graph optimisation (extracting argmax from the grad-enabled pass instead of a separate no-grad pass) could halve this. Not needed for the first implementation but worth noting.
- **Visited set granularity:** The visited set is per (comp_idx, latent_idx). If the same latent appears as an upstream node of multiple nodes at the same depth, it is only enqueued once and its ctx pass runs once. This is correct and efficient but means its edge score is only computed relative to the first node that discovered it. Consider whether multiple edges to the same upstream node from different nodes are wanted.
- **Attribution through SAE reconstruction error:** `SAEGraphInstrument` captures both `f_grad` (SAE feature activations) and `res_anchor` (reconstruction error). `compute_feature_attribution` uses both. For upstream discovery, we primarily care about the feature pathway. Consider using `stop_error_grad=True` in `SAEGraphInstrument` to force all gradient flow through SAE features, producing cleaner latent-to-latent attribution.
- **Layer 0 seeds:** If the seed is at layer 0 (attn or mlp), `get_predecessor_components` returns an empty list and the method produces a single-node circuit. This should be caught early and rejected before running any forward passes.

## Risks / Assumptions

- `self.inference.disable_compile()` / `enable_compile()` must bracket every `SAEGraphInstrument` forward pass, exactly as done in `logit_attribution.py`. Forgetting this causes silent correctness failures with `torch.compile`.
- Each hop creates and destroys a `SAEGraphInstrument`. The `del instrument; torch.cuda.empty_cache()` step in `_run_hop` is mandatory — without it, VRAM accumulates across hops and will OOM on deeper circuits.
- `compute_latent_upstream_scores` uses `retain_graph=True` in `autograd.grad`. Since the graph is freed immediately after score extraction, this is harmless but set explicitly to allow re-use of the graph if the function is called multiple times per hop in future.
- The `pos_argmax` extracted from the grad-enabled pass (argmax over the target latent's activation in `acts_connected`) may differ slightly from the argmax computed by `_calculate_argmax` (which uses a no-grad pass). They should be identical since the SAE encoding is deterministic for a given input.

