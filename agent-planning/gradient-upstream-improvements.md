# Plan: Gradient Upstream — Inhibitor Roles, Gradient Accumulation, Layer-Restricted Ablation

> **Goal:** Improve `GradientUpstreamDiscovery` with three changes: (1) classify upstream nodes as activators, present inhibitors, or absent inhibitors and only BFS-expand activators; (2) gradient accumulation over microbatches for better attribution signal with bounded VRAM; (3) restrict circuit ablation to only the layers where circuit nodes actually exist, not everything ≤ seed_layer.
>
> **Created:** 2026-04-12

---

## Phase 1 — Config Additions

- [x] Add three new fields to `GradientUpstreamConfig` in `src/config.py`:
  ```python
  hop_batch_size: int = 4              # sequences per microbatch in _run_hop (for gradient accumulation)
  absent_inhibitor_top_k: int = 4      # top-K absent inhibitors to find per hop (0 = disabled)
  absent_inhibitor_threshold: float = 0.01  # min |raw gradient| to flag an absent inhibitor
  ```
- [x] Add the three new fields to `config.yaml` under `gradient_upstream:`:
  ```yaml
  hop_batch_size: 4           # Microbatch size for grad-enabled passes (gradient accumulation)
  absent_inhibitor_top_k: 4   # Top-K absent inhibitors per hop (0 = disabled)
  absent_inhibitor_threshold: 0.01  # Min |gradient| at near-zero activation to flag absent inhibitor
  ```
- [x] Wire new fields into `GradientUpstreamDiscovery.__init__` (same cast/default pattern as existing fields).

---

## Phase 2 — Dual-Score Attribution: Extend `compute_latent_upstream_scores`

The goal is to produce **two outputs from one backward pass**: attribution scores (`acts * grad`) for present latents, and raw gradient scores (`grad` where `acts ≈ 0`) for absent potential inhibitors. Currently only the former is computed.

- [x] Change the return type of `compute_latent_upstream_scores` in `src/circuit/instrument/attribution.py` from `Dict[FeatureID, float]` to a `NamedTuple` (or dataclass):
  ```python
  @dataclass
  class UpstreamScores:
      attribution: Dict[FeatureID, float]       # acts * grad — present latents driving target
      absent_gradient: Dict[FeatureID, float]   # raw grad where acts ≈ 0 — potential absent inhibitors
  ```
- [x] In the inner loop where `attr_act = acts_grad.act * grad_act` is computed, additionally compute absent-inhibitor scores **in the same loop iteration** (no second backward pass needed):
  - `absent_mask = acts_grad.act.abs() < 1e-6`  — `[B, T, d_sae]` bool: latent is inactive
  - `absent_grad_scores = (grad_act * absent_mask.float()).sum(dim=(0, 1))`  — `[d_sae]`; gradient only where activation ≈ 0
  - Apply `active_count` filter to `absent_grad_scores` the same way as `scores`
  - Collect top-K absent inhibitor candidates by `absent_grad_scores.min()` (most negative gradient wins — these are the strongest potential suppressors)
- [x] Build and return both dicts in `UpstreamScores`. The attribution dict is unchanged from current behaviour. The absent_gradient dict is keyed to the same `FeatureID` structure.
- [x] Update the single existing call-site in `_run_hop` to unpack `UpstreamScores`.

---

## Phase 3 — Gradient Accumulation in `_run_hop`

Currently `_run_hop` runs all `max_ctx_sequences` as a single batch. Replace this with microbatch loop accumulating scores across chunks.

- [x] In `_run_hop`, split `tokens` (shape `[N, T]`) into chunks of size `hop_batch_size`:
  ```python
  chunks = tokens.split(self.hop_batch_size, dim=0)
  ```
- [x] For each chunk:
  - Create a **new** `SAEGraphInstrument` (each chunk has its own independent graph)
  - Run `inference.forward(chunk, patcher=instrument, grad_enabled=True, ...)`
  - Extract `pos_argmax` from the chunk's graph (same as current, over `dim=-1` of chunk acts)
  - Call `compute_latent_upstream_scores(...)` → `UpstreamScores`
  - **Accumulate** by summing the float scores: for each `FeatureID` seen, `accumulated_attribution[fid] = accumulated_attribution.get(fid, 0.0) + score`
  - Delete the instrument and call `torch.cuda.empty_cache()` + `gc.collect()` **before the next chunk** (critical: each chunk's graph must be freed before the next chunk's forward pass)
- [x] After the loop, apply global top-K selection across the accumulated dicts:
  - `top_attribution = sorted(accumulated_attribution.items(), key=lambda x: abs(x[1]), reverse=True)[:self.top_k_per_hop]`
  - `top_absent = sorted(accumulated_absent.items(), key=lambda x: x[1])[:self.absent_inhibitor_top_k]` (most negative first)
- [x] Return both as an `UpstreamScores`.

---

## Phase 4 — Role Classification and Selective BFS Enqueuing in `_discover`

Split the returned scores into three roles: `activator`, `active_inhibitor`, `absent_inhibitor`. Only activators get enqueued for further BFS expansion.

**Role definitions:**
- **`activator`**: firing on pos_ctx (`activation > 0`), positive attribution score → drives the seed upward. BFS-expanded.
- **`active_inhibitor`**: firing on pos_ctx (`activation > 0`), negative attribution score → suppresses the seed despite being active. Kept at its natural pos_ctx value in evaluation (patcher preserves it). **Not BFS-expanded** — its upstream is not relevant to why the seed fires. Note: treat with skepticism; a negative gradient at the observed operating point reflects the local slope and may not represent the latent's overall circuit role.
- **`absent_inhibitor`**: silent on pos_ctx (`activation ≈ 0`), strongly negative raw gradient → would suppress the seed if it fired. Its absence is part of what allows the seed to activate. In evaluation the patcher naturally leaves it at zero (its real pos_ctx value). **Not BFS-expanded**.

- [x] In the BFS loop body of `_discover`, after receiving `upstream_scores: UpstreamScores`:
  - For each `(fid, score)` in `upstream_scores.attribution.items()`:
    - If `score > self.attribution_threshold`: role = `"activator"` → add node + edge + **enqueue**
    - If `score < -self.attribution_threshold`: role = `"active_inhibitor"` → add node + edge + **do NOT enqueue**
    - (Scores between `-threshold` and `+threshold` are dropped, as before)
  - For each `(fid, score)` in `upstream_scores.absent_gradient.items()`:
    - Only proceed if `self.absent_inhibitor_top_k > 0`
    - Only if `score < -self.absent_inhibitor_threshold` (negative gradient = would suppress)
    - Only if `fid` not already in `fid_to_uuid`
    - role = `"absent_inhibitor"` → add node + edge + **do NOT enqueue**
    - No context-switch or probe load needed (no BFS expansion)
- [x] Store role in node metadata: `"role": "activator"` / `"active_inhibitor"` / `"absent_inhibitor"`
- [x] Store sign-aware score in metadata (`"attribution_score": score` — preserving the sign for later analysis/visualisation)
- [x] Visited set guard stays per `(comp_idx, latent_idx)` — all roles are added to `visited` to prevent re-discovery from a different ancestor node.

---

## Phase 5 — `circuit_layers`-Restricted Ablation in `CircuitPatcher`

Add a new `circuit_layers: Optional[Set[int]]` parameter to `CircuitPatcher` that restricts intervention to only the specific layers where circuit nodes exist.

- [x] In `src/circuit/instrument/patcher.py`, add `circuit_layers: Optional[Set[int]] = None` to `CircuitPatcher.__init__` signature and store as `self.circuit_layers`.
- [x] In `CircuitPatcher.transform`, add an early return **before** the existing `max_layer` check:
  ```python
  if self.circuit_layers is not None and layer_idx not in self.circuit_layers:
      return x
  ```
  (The `max_layer` guard remains unchanged — both filters compose independently.)
- [x] In `src/eval/upstream_faithfulness.py`, add `circuit_layers` guard to `SeedActivationCapturePatcher.transform` (inherits the field via `**kwargs` → `super().__init__()`).
- [x] Add `circuit_layers: Optional[Set[int]] = None` to `evaluate_upstream_faithfulness` signature; pass to all three `SeedActivationCapturePatcher(...)` instantiations via explicit `max_layer=_max_layer, circuit_layers=circuit_layers` (where `_max_layer=seed_layer` when `circuit_layers` is None for backward compatibility).

---

## Phase 6 — Thread `circuit_layers` Through All Eval Functions

- [x] Add `circuit_layers: Optional[Set[int]] = None` to `evaluate_faithfulness` in `src/eval/faithfulness.py`. Pass it to both internal `CircuitPatcher(...)` calls.
- [x] Add `circuit_layers: Optional[Set[int]] = None` to `evaluate_sufficiency` in `src/eval/sufficiency.py`. Pass it to the internal `CircuitPatcher(...)` call.
- [x] Add `circuit_layers: Optional[Set[int]] = None` to `evaluate_completeness` in `src/eval/completeness.py`. Pass it to both internal `CircuitPatcher(...)` calls.
- [x] Add `circuit_layers: Optional[Set[int]] = None` to `evaluate_minimality` and `prune_non_minimal_nodes` in `src/eval/minimality.py`; thread through to `evaluate_faithfulness` calls inside both.
- [x] In `GradientUpstreamDiscovery._discover`, after BFS completes, compute `circuit_layers` from actual circuit nodes. After pruning, recompute it (pruned nodes may vacate a layer). Replace all `max_layer=seed_layer` in the five eval calls with `circuit_layers=circuit_layers`.

---

## Phase 7 — Validation ✅

Static validation via automated unit tests. Full end-to-end pipeline runs require real model weights and are deferred to the first real training run.

**`TestCircuitLayersFilter` (6 tests) — added to `tests/circuit/test_patcher.py`:**
- [x] Layer in `circuit_layers` is patched (null circuit at that layer modifies x)
- [x] Layer not in `circuit_layers` passes through unchanged (x returned as-is)
- [x] Empty `circuit_layers` set passes every layer through (nothing patched)
- [x] `circuit_layers=None` is backward compatible (all layers patched, same as before)
- [x] Composes with `max_layer`: `max_layer=0, circuit_layers={1}` → nothing patched (both guards must pass)
- [x] Multi-layer set `{0, 1}` correctly patches both layers

**`TestComputeLatentUpstreamScores` (7 tests) — added to `tests/circuit/test_attribution.py`:**
- [x] Return type is `UpstreamScores`, not a plain dict
- [x] Empty predecessor list → empty `UpstreamScores`
- [x] Attribution scores match hand-computed oracle: `acts * grad = [2,3,1] * M[:,0] = [2.0, 0.0, 0.5]`
- [x] `absent_inhibitor_top_k=0` → `absent_gradient` is always `{}`
- [x] Inactive latents with negative gradient appear in `absent_gradient`
- [x] `absent_inhibitor_threshold` gates capture: -1.0 passes at 0.5, filtered at 2.0
- [x] Active and absent latents are disjoint (no key appears in both dicts)

All 13 new tests pass. The 39 pre-existing failures in `test_*_top_coact_sparse_expansion.py` and `test_top_coactivation_modes.py` were confirmed to exist before these changes (verified via `git stash`).

---

## Open Questions

- **Absent inhibitor threshold units:** `absent_inhibitor_threshold` is compared against raw gradient values. These are dimensionally different from attribution scores (`acts * grad`). A separate config default may need tuning empirically before the first real run.
- **`prune_non_minimal_nodes` compatibility:** The minimality pruner in `src/eval/minimality.py` may or may not internally construct a `CircuitPatcher`. It needs inspection to decide whether to add `circuit_layers` threading there or leave it using `max_layer` for now.
- **`evaluate_kind_local_faithfulness`:** This function in `faithfulness.py` also takes `max_layer` and constructs `CircuitPatcher`. It is not called from `gradient_upstream` currently, so threading `circuit_layers` through it can be deferred.

## Risks / Assumptions

- **Score accumulation correctness:** Gradient accumulation is exact (sum is associative over independent sequences), but `pos_argmax` varies per chunk. Each chunk extracts its own argmax for that chunk's sequences. The accumulated score correctly marginalises over all sequence positions from all chunks — this is the desired behaviour (more sequences = better signal).
- **`active_inhibitor` reliability:** A negative attribution score reflects the gradient's sign at the observed operating point only. A latent with a non-monotonic relationship to the seed (e.g. activating at mid-range values, suppressing at high values) may be tagged `active_inhibitor` when it is mechanistically an activator firing slightly above its "sweet spot". The `role` metadata tag flags this for human review; these nodes are never BFS-expanded so the impact on circuit structure is contained.
- **Absent inhibitor noise:** Raw gradient at `acts ≈ 0` is a linear extrapolation and may be unreliable, particularly for ReLU-adjacent SAE activations. `absent_inhibitor_top_k: 4` is intentionally small and the threshold is a safety net. These nodes are never expanded, so their impact is limited to the evaluation passes.
- **No double-backward:** The absent inhibitor scores are extracted from the **same** `grads` tensors computed in `compute_latent_upstream_scores` — no second `autograd.grad` call. This is safe since we're just indexing `grad_act` at positions where `acts ≈ 0`.
- **`circuit_layers` and the "full model" pass in `evaluate_upstream_faithfulness`:** The full model pass uses `full_circuit=True`, which makes the patcher a mathematical identity at any patched layer. With `circuit_layers`, the identity is applied only at those layers — all other layers run completely naturally. This is correct: the full model pass should always equal the unpatched forward.
