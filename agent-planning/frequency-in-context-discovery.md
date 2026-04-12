# Plan: Frequency-in-Context Circuit Discovery

> **Goal:** Add `FrequencyInContextDiscovery` as a new `DiscoveryMethod` that finds circuit members by running no-grad forward passes on the seed's positive probe sequences and selecting latents that fire consistently across them, above a configurable frequency threshold.
>
> **Created:** 2026-04-11

---

## Background

All existing discovery methods either (a) read from pre-computed stores (`coactivation_statistical`, all BFS expansion variants) or (b) use gradients on a small probe batch (`sfc_attribution_patching`, `logit_attribution`, `differential_activation`). This method is uniquely simple: it asks "which latents fire on most of the seed's positive sequences?" — latents firing on >80% of the seed's context are almost certainly participating in whatever computation the seed is part of.

This mirrors the original `latent_connections` codebase's Method 1, but scoped to a per-seed probe dataset rather than a global task dataset. The bet is that `top_ctx` sequences for a latent are task-specific, even without a known task label.

**Key property:** no gradient computation, no co-activation store reads, no ablation patching. The only compute is no-grad forward passes on the positive probe sequences.

### How it fits alongside existing methods

| Method | Requires grads | Requires `top_coactivation` | Forward passes |
|---|---|---|---|
| `coactivation_statistical` | No | Yes | 0 (eval only) |
| `frequency_in_context` | **No** | **No** | N_pos / batch_size |
| `all_top_coact_sparse_expansion` | No | Yes | ~12 (eval) |
| `sfc_attribution_patching` | Yes | Yes | ig_steps × n_submodules |

This means it can be run even with a different (or absent) `top_coactivation` store, making it a useful standalone baseline.

### Implementation overview

```
for each batch of pos_tokens:
    run no-grad forward → activations_callback collects (top_acts, top_indices) per component
    for each component:
        for each sequence in batch:
            if latent j fired at any token in this sequence: firing_count[comp][j] += 1

frequency[comp][j] = firing_count[comp][j] / N_pos
circuit_candidates = {(comp, j) : frequency[comp][j] >= frequency_threshold}
```

Edges are assigned by layer/kind ordering (same convention as `coactivation_statistical`): upstream nodes point to the seed, downstream nodes are pointed to by the seed. Edge weight = firing frequency of the candidate node.

---

## Phase 1 — Config & Pydantic Schema

- [ ] Add a `FrequencyInContextConfig` Pydantic model to `src/config.py`:
  ```python
  class FrequencyInContextConfig(BaseModel):
      model_config = ConfigDict(extra='forbid')
      frequency_threshold: float = 0.8   # fraction of pos sequences latent must fire in
      max_nodes: int = 128               # max circuit nodes (not counting seed)
      min_active_count: int = 1          # skip latents below this global firing count
      pruning_threshold: float = 0.0     # faithfulness drop threshold for minimality pruning
  ```
- [ ] Add `frequency_in_context: FrequencyInContextConfig = Field(default_factory=FrequencyInContextConfig)` to `DiscoveryConfig`.
- [ ] Add to `config.yaml` under `discovery:`:
  ```yaml
  frequency_in_context:
    frequency_threshold: 0.8  # Fraction of positive sequences a latent must fire in to be included
    max_nodes: 128             # Max circuit nodes (excluding seed)
    min_active_count: 1        # Min global lifetime firing count
    pruning_threshold: 0.0     # Faithfulness drop threshold for minimality pruning (0 = disabled)
  ```
- [ ] Add `"frequency_in_context"` to the comment block listing available methods in `config.yaml`.

## Phase 2 — Implement `FrequencyInContextDiscovery`

Create `src/circuit/discovery/frequency_in_context.py`.

- [ ] Define `class FrequencyInContextDiscovery(DiscoveryMethod)` following the same structure as `coactivation_statistical.py`.
- [ ] `__init__`: read `frequency_threshold`, `max_nodes`, `min_active_count`, `pruning_threshold` from `config.discovery.frequency_in_context`.
- [ ] Implement `discover(seed_comp_idx, seed_latent_idx) -> Optional[Circuit]`:
  1. Build probe dataset via `self.build_probe_dataset(...)`.
  2. If `probe_data.pos_tokens.shape[0] == 0`, reject with logger.
  3. Call `firing_freq = self._collect_firing_frequencies(probe_data.pos_tokens)` (see below).
  4. Build the circuit — seed node first, then iterate components and latents: include any `(comp_j, lat_j)` where `firing_freq[comp_j][lat_j] >= self.frequency_threshold` and `latent_stats.active_count[comp_j, lat_j] >= self.min_active_count`, up to `self.max_nodes`.
  5. Assign edges by layer/kind ordering (same logic as `coactivation_statistical.py` lines 143–150), using `firing_freq[comp_j][lat_j]` as the edge weight.
  6. If `len(circuit.nodes) <= 1`, reject.
  7. If `pruning_threshold > 0`, call `prune_non_minimal_nodes(...)`.
  8. Evaluate faithfulness, sufficiency, completeness and accept/reject via `min_faithfulness`.
  9. Populate `circuit.metadata` with `discovery_method`, seed info, `frequency_threshold`, and eval scores.

- [ ] Implement `_collect_firing_frequencies(self, pos_tokens: torch.Tensor) -> torch.Tensor`:
  - Signature: `pos_tokens: [N_pos, seq_len]` → returns `freq: [num_components, d_sae]` float tensor on CPU.
  - Pre-allocate `firing_count = torch.zeros(num_components, d_sae, dtype=torch.float32)`.
  - Iterate over `pos_tokens` in batches of `config.discovery.probe_batch_size` using `torch.no_grad()`.
  - For each batch, use an `activations_callback` (same pattern as `second_pass.py` and `probe_dataset.py`) to collect `(top_acts, top_indices)` for each component at each layer.
  - Per component per batch: for each sequence `b`, scatter a binary "fired at least once in this sequence" indicator:
    ```python
    # binary_fired[b, j] = 1 if latent j fired at any token in sequence b
    fired = torch.zeros(B, d_sae, device=device)
    fired.scatter_(1, top_indices.reshape(B, -1).long(),
                   (top_acts.reshape(B, -1) > 0).float().clamp(max=1))
    firing_count[comp_idx] += fired.sum(dim=0).cpu()
    ```
  - After all batches: `freq = firing_count / N_pos` (divide by total positive sequences).
  - Return `freq` on CPU.

  **Note:** The `activations_callback` fires once per layer. Each layer contributes up to 3 components (attn, mlp, resid at indices `layer*3`, `layer*3+1`, `layer*3+2`). Use `encode_layer_components` (already imported in `second_pass.py`) or encode directly via `self.sae_bank.encode(act, kind, layer_idx)` for each kind. Check how `probe_dataset._calculate_argmax` handles this and follow the same pattern.

## Phase 3 — Register in `discovery_window.py`

- [ ] Add import: `from circuit.discovery.frequency_in_context import FrequencyInContextDiscovery`
- [ ] Add entry to `METHOD_REGISTRY`: `"frequency_in_context": FrequencyInContextDiscovery`
- [ ] Add description to the `_build_methods` docstring: `"frequency_in_context"  — no-grad frequency counting on positive probe sequences`

## Phase 4 — Validation

- [ ] Add `"frequency_in_context"` to `config.yaml` methods list (uncommented) and run on a small shard (`n_shards: 1`, `n_seeds: 16`) to confirm: (a) the method runs without errors, (b) circuits are produced with a reasonable number of nodes, (c) `discovery_method: "frequency_in_context"` appears in the summary table.
- [ ] Verify that for a seed with `n_pos=1` the method either produces a circuit or rejects gracefully (with `n_pos=1`, `frequency_threshold=0.8` means only latents firing in that 1 sequence qualify — this is equivalent to n_pos>0 in that sequence; should not crash).
- [ ] Comment out `"frequency_in_context"` from methods after validation and restore original config.

---

## Open Questions

- **Double forward pass cost**: `build_probe_dataset` already runs a forward pass to compute `pos_argmax` (in `_calculate_argmax`). `_collect_firing_frequencies` runs a second forward pass over the same sequences. If the probe dataset is small (≤16 sequences) this doubles the no-grad pass count. Worth combining in the future — `_calculate_argmax` could optionally return all-component firing counts as a side-product. Not needed for the first implementation.
- **`max_nodes` ordering**: When the number of qualifying latents exceeds `max_nodes`, which do we keep? The natural choice is top-K by descending frequency. Add a sort step before the `max_nodes` slice.
- **Batching approach**: The forward pass in `_collect_firing_frequencies` uses `probe_batch_size` from config. For large `n_pos` (64 sequences), this means `64/4 = 16` batches. If `probe_batch_size` is very small this is still fast since no gradient tape is needed.
- **Should the seed be excluded from freq counting?** By definition the seed latent fires on (nearly) all positive sequences — it would trivially appear in its own frequency list. The circuit build step should skip `(seed_comp_idx, seed_latent_idx)` when iterating candidates, just as other methods do.

## Risks / Assumptions

- `pos_tokens` sequences from the probe dataset are representative of the seed's function. This is the same assumption made by all other discovery methods.
- The `activations_callback` fires once per layer across all 3 kinds simultaneously (attn, mlp, resid). Each kind must be encoded separately via `self.sae_bank.encode(act, kind, layer_idx)`. This matches the pattern in `probe_dataset._calculate_argmax` and `second_pass.coact_callback`.
- The `scatter_` with `clamp(max=1)` correctly collapses "did latent j fire at ANY token in sequence b" into a binary per-sequence flag, which is what we want for frequency counting (not total token count).
- `latent_stats` is loaded before discovery (verified in `run_discovery_window`), so `active_count` is accessible.
