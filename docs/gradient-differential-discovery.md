# Gradient-Differential Circuit Discovery

> **Method name:** `gradient_differential`
>
> A combined discovery algorithm that uses a **Seed-to-All Gradient Scan** for
> upstream activators and a **Differential Activation + Binary-Split Ablation**
> pipeline for inhibitors. Produces a circuit with directed causal edges and
> both activator and inhibitor roles.

---

## Motivation

We have two complementary techniques for finding circuit nodes:

| Technique | Strength | Blind spot |
|---|---|---|
| **Seed-to-All Gradient Scan** (pos_ctx) | One backward pass scores every upstream latent at once — extremely efficient for activators | Cannot find inhibitors (dormant in pos_ctx → act × grad = 0). Only sees upstream of the seed. Local linear approximation can miss non-linear effects. |
| **Differential Activation** (pos_ctx vs neg_ctx) | Naturally surfaces inhibitors (latents present only in neg_ctx). No gradient required for the scan itself. | Statistical — a high differential doesn't guarantee a causal relationship. |

By combining them in a funnel we get the best of both:

1. Gradient scan → high-recall upstream activators with causal scores in one pass.
2. Differential scan → high-recall inhibitor candidates cheaply.
3. Ablation on neg_ctx → ground-truth counterfactual confirmation of inhibitors.

---

## Algorithm

### Inputs

- `seed_comp_idx`, `seed_latent_idx` — the seed latent to build a circuit around.
- `pos_tokens` [B_pos, T] — sequences where the seed fires (from `ProbeDataset`).
- `neg_tokens` [B_neg, T] — hard-negative sequences where the seed is expected but absent.
- `pos_argmax` [B_pos] — token position of peak seed activation per positive sequence.

### Phase 1 — Upstream Activator Discovery (Gradient Scan)

**Goal:** Find latents at layers ≤ seed that causally drive the seed's activation.

1. Run an **instrumented** (grad-enabled) forward pass on `pos_tokens` using
   `SAEGraphInstrument`. This populates a `FeatureGraph` with leaf anchors
   for every (layer, kind).

2. Compute a **single backward pass** from the seed's activation:

   ```python
   # seed activation at peak positions
   _, seed_connected, _ = graph.get_latents(seed_layer, seed_kind)
   target = seed_connected.act[batch_idx, pos_argmax, seed_latent_idx].sum()
   grads = torch.autograd.grad(target, graph.all_anchors(), retain_graph=True)
   ```

3. For every upstream latent in the graph, compute `score = act × grad` (summed
   over batch and token positions). This reuses the same logic as
   `compute_feature_attribution` in `src/circuit/attribution.py` with
   `candidate_nodes=None` (score all).

4. **Select activator nodes:** Take the top `n_activator_candidates` by score.
   Each gets role `"activator"` and a directed edge `candidate → seed` with
   weight = attribution score.

**Key implementation notes:**

- `compute_feature_attribution` already supports `candidate_nodes=None` to
  score all upstream latents (see lines 178-190 of `src/circuit/attribution.py`).
  Use this directly rather than reimplementing the gradient collection.
- The instrumented forward from this phase is **reused in Phase 4** for
  downstream edge construction — do not discard the `FeatureGraph`.

### Phase 2 — Inhibitor Candidate Generation (Differential Scan)

**Goal:** Find latents that are differentially active between pos and neg contexts.

1. Collect per-latent total activations on `pos_tokens` — this can be
   extracted cheaply from the `FeatureGraph` already built in Phase 1 (iterate
   `graph.activations`, sum `state_grad.act` per latent index). Alternatively,
   reuse the `_collect_activations` helper from `DifferentialActivation`.

2. Run a **no-grad** forward on `neg_tokens` with a `capture_hook` (identical
   to `DifferentialActivation._collect_activations`) to collect per-latent
   total activations.

3. Compute the differential for every latent seen in either set:

   ```
   delta_j = mean(act_j | pos) − mean(act_j | neg)
   ```

   Normalise by total token count in each set so unequal batch sizes don't
   skew the comparison.

4. **Select inhibitor candidates:** Take every latent with `delta < 0`, ranked
   by most-negative-first. Optionally apply a percentile cutoff (e.g. bottom
   5%) or a fixed count `n_inhibitor_candidates` — whichever admits more
   candidates. A generous candidate set is fine because Phase 3 is cheap.

**Output:** A list of `FeatureID` objects with their delta scores.

### Phase 3 — Inhibitor Confirmation (Binary-Split Ablation on neg_ctx)

**Goal:** Confirm which inhibitor candidates have a true causal suppressive
effect on the seed, and measure the strength of that effect.

The key insight: in `neg_ctx` the seed is suppressed and the inhibitors are
active. If we ablate (zero out) an inhibitor and the seed's activation
*recovers*, that inhibitor is causally confirmed.

#### 3a. Measure baseline seed activation on neg_ctx

Run a **no-grad** forward on `neg_tokens` and record the seed's activation at
each token position. Since the seed is suppressed in hard negatives, this
should be near zero:

```python
baseline_seed_act = seed_activation_on(neg_tokens)  # [B_neg, T] or scalar
```

Use the existing `capture_hook` pattern — hook into the target layer, encode,
and extract the seed latent's activation value.

#### 3b. Binary-split ablation

The naive approach (ablate one candidate at a time) requires N forward passes.
Binary splitting reduces this to O(log₂ N):

```
function find_causal_inhibitors(candidates, neg_tokens, baseline):
    if len(candidates) == 0:
        return []
    if len(candidates) == 1:
        # Test this single candidate
        delta = ablate_and_measure(candidates, neg_tokens) - baseline
        if delta > confirmation_threshold:
            return [(candidates[0], delta)]
        return []

    # Test the full group
    delta = ablate_and_measure(candidates, neg_tokens) - baseline
    if delta <= confirmation_threshold:
        # Ablating all of them doesn't recover the seed — none are causal
        return []

    # At least one is causal — split and recurse
    mid = len(candidates) // 2
    left  = find_causal_inhibitors(candidates[:mid],  neg_tokens, baseline)
    right = find_causal_inhibitors(candidates[mid:],  neg_tokens, baseline)
    return left + right
```

#### 3c. ablate_and_measure

To ablate a set of candidate latents on `neg_tokens`:

1. Build a `CircuitPatcher` where the "circuit" contains only the candidate
   latents and `inverse=True` (ablate only the listed nodes, keep everything
   else). The `avg_acts` for the ablated latents should be zero (hard ablation)
   rather than average — this tests the counterfactual "what if these latents
   were completely absent?"

   Alternatively, build a minimal patching hook that zeros out specific latent
   indices during the forward pass. This avoids constructing a full
   `CircuitPatcher` per test:

   ```python
   def ablation_hook(layer_idx, kind, x, ablate_set):
       top_acts, top_indices = bank.encode(x, kind, layer_idx)
       for fid in ablate_set:
           if fid.layer == layer_idx and fid.kind == kind:
               mask = (top_indices == fid.index)
               top_acts = top_acts.masked_fill(mask, 0.0)
       # re-scatter and decode...
   ```

2. Run a no-grad forward with the ablation hook.
3. Capture the seed's activation the same way as in 3a.
4. Return the mean seed activation across all neg sequences.

The `delta` (recovered - baseline) is the **ablation effect**: how much the
seed's activation increases when the candidate is removed. This becomes the
edge weight for confirmed inhibitors (stored as a negative weight to indicate
inhibition).

#### Cost analysis

| Scenario | Forward passes |
|---|---|
| N candidates, all causal | 2 × log₂(N) (split both halves) |
| N candidates, none causal | 1 (the full-group test fails immediately) |
| N candidates, K causal | ~K × log₂(N/K) + log₂(N) |
| N = 128, K = 8 (typical) | ~25 no-grad forward passes |

Each forward pass is on the `neg_tokens` batch (typically 16 sequences), which
is cheap relative to the instrumented grad-enabled pass.

### Phase 4 — Downstream Edge Construction (Optional)

**Goal:** Find latents at layers > seed that the seed drives.

The `FeatureGraph` from Phase 1 is still alive (retained). For downstream
candidates — activators from the differential scan with positive delta at
layers > seed:

1. For each downstream candidate `fid_d`, compute:

   ```python
   compute_feature_attribution(
       graph,
       target_layer=fid_d.layer,
       target_kind=fid_d.kind,
       target_latent_idx=fid_d.index,
       pos_argmax=pos_argmax,
       candidate_nodes=[seed_fid],
   )
   ```

   This gives `d(downstream_act) / d(seed_act)` × `seed_act`.

2. If the score exceeds `attribution_threshold`, add the node with edge
   `seed → downstream`, weight = attribution score.

**Cost:** One backward pass per downstream candidate. Limit to the top ~32
differential candidates at later layers to keep this manageable.

**When to skip:** If downstream structure isn't needed (e.g. the circuit is
only used for faithfulness evaluation which tests "can we recover the model's
output at the seed's position"), this phase can be disabled via config.

### Phase 5 — Minimality Pruning

Use the existing `prune_non_minimal_nodes` from `src/eval/minimality.py`.
Iteratively remove the least-important non-seed node (by LOO faithfulness
drop) until every remaining node is above the `pruning_threshold`.

### Phase 6 — Evaluation

Run the standard evaluation suite on the complete circuit:

- `evaluate_faithfulness` (global) — `src/eval/faithfulness.py`
- `evaluate_kind_local_faithfulness` with `target_kinds=("attn", "mlp", "resid")`
- `evaluate_sufficiency` — `src/eval/sufficiency.py`
- `evaluate_completeness` — `src/eval/completeness.py`

Reject the circuit if `faithfulness < min_faithfulness`.

Store metadata including `n_activators`, `n_inhibitors`,
`n_gradient_activators`, `n_differential_inhibitors`, `n_ablation_confirmed`.

---

## Config

### Pydantic model (`src/config.py`)

```python
class GradientDifferentialConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    n_activator_candidates: int = 64        # top gradient-scored upstream activators
    n_inhibitor_candidates: int = 128       # max inhibitor candidates from differential scan
    inhibitor_percentile: float = 5.0       # alternative: bottom X% of deltas (whichever admits more)
    confirmation_threshold: float = 0.01    # min seed activation recovery to confirm an inhibitor
    attribution_threshold: float = 0.01     # min |edge weight| to keep a causal edge
    enable_downstream: bool = False         # whether to run Phase 4
    n_downstream_candidates: int = 32       # max downstream candidates if enabled
    pruning_threshold: float = 0.0          # minimality pruning (0 = disabled)
```

### YAML (`config.yaml`)

```yaml
discovery:
  methods:
    - gradient_differential

  gradient_differential:
    n_activator_candidates: 64
    n_inhibitor_candidates: 128
    inhibitor_percentile: 5.0
    confirmation_threshold: 0.01
    attribution_threshold: 0.01
    enable_downstream: false
    n_downstream_candidates: 32
    pruning_threshold: 0.0
```

---

## File Structure

```
src/circuit/discovery/gradient_differential.py    # Main discovery class
```

The class should:

- Inherit from `DiscoveryMethod` (`src/circuit/discovery/base.py`).
- Be registered in `METHOD_REGISTRY` in `src/circuit/discovery_window.py`.
- Follow the same `discover()` / `_discover()` pattern as `DifferentialActivation`.

---

## Key Implementation Details

### Reusing the FeatureGraph for pos activations

Phase 1 builds a `FeatureGraph` via `SAEGraphInstrument`. Phase 2 needs
per-latent activation totals on pos_tokens. Rather than running a second
no-grad forward, extract these from the graph directly:

```python
pos_acts: Dict[FeatureID, float] = {}
for (layer, kind), steps in instrument.graph.activations.items():
    for state_grad, _, top_indices in steps:
        act = state_grad.act.detach()   # [B, T, d_sae]
        active = act > 0
        if not active.any():
            continue
        # sum per unique latent index
        flat_idx = torch.where(active)[-1]  # latent dimension indices
        flat_vals = act[active]
        unique, inverse = flat_idx.unique(return_inverse=True)
        summed = torch.zeros_like(unique, dtype=torch.float32).scatter_add_(
            0, inverse, flat_vals.float()
        )
        for idx, total in zip(unique.tolist(), summed.tolist()):
            fid = FeatureID(layer, kind, int(idx))
            pos_acts[fid] = pos_acts.get(fid, 0.0) + total
```

This avoids an extra forward pass entirely.

### compile / no-compile toggling

The model may be `torch.compile`d. Both `SAEGraphInstrument` (graph-mode
tracing breaks custom autograd) and the ablation forward passes (dynamic
control flow) require eager mode. The pattern used throughout the codebase is:

```python
_was_compiled = self.inference._compiled
self.inference.disable_compile()
try:
    ...
finally:
    if _was_compiled:
        self.inference.enable_compile()
```

Apply this around Phase 1 (instrumented forward) and Phase 3 (ablation
forwards). Phase 2's no-grad forward can run compiled.

### Ablation hook design

For the binary-split ablation, avoid building a full `CircuitPatcher` each
iteration (it pre-computes background tensors for every layer×kind, which is
wasteful when we only want to zero a handful of latents). Instead, write a
lightweight hook:

```python
class LatentAblator:
    """Zeros out a specific set of latent indices during forward."""
    def __init__(self, bank: SAEBank, ablate_fids: Set[FeatureID]):
        self.bank = bank
        # Pre-group by (layer, kind) for O(1) lookup in the hook
        self.ablate_map: Dict[Tuple[int, str], Set[int]] = {}
        for fid in ablate_fids:
            self.ablate_map.setdefault((fid.layer, fid.kind), set()).add(fid.index)

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        indices = self.ablate_map.get((layer_idx, kind))
        if not indices:
            return x
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        # Zero activations for ablated latents
        for idx in indices:
            mask = (top_indices == idx)
            top_acts = top_acts.masked_fill(mask, 0.0)
        # Reconstruct: decode modified + error
        B, T, _ = x.shape
        modified = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=x.dtype)
        modified.scatter_(dim=-1, index=top_indices.long(), src=top_acts.to(x.dtype))
        x_hat_modified = self.bank.decode(modified, kind, layer_idx)
        # Preserve original error term
        original = torch.zeros(B, T, self.bank.d_sae, device=x.device, dtype=x.dtype)
        original_acts, _ = self.bank.encode(x, kind, layer_idx)
        original.scatter_(dim=-1, index=top_indices.long(), src=original_acts.to(x.dtype))
        x_hat_original = self.bank.decode(original, kind, layer_idx)
        error = x - x_hat_original
        return x_hat_modified + error
```

Note: the above double-encodes. A cleaner approach is to encode once, clone
`top_acts`, zero the targets in the clone, decode both, and use the difference:

```python
def transform(self, layer_idx, kind, x):
    indices = self.ablate_map.get((layer_idx, kind))
    if not indices:
        return x
    top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
    ablated_acts = top_acts.clone()
    for idx in indices:
        ablated_acts.masked_fill_(top_indices == idx, 0.0)
    # delta in latent space → delta in activation space
    B, T, _ = x.shape
    d_sae = self.bank.d_sae
    diff = torch.zeros(B, T, d_sae, device=x.device, dtype=x.dtype)
    diff.scatter_(dim=-1, index=top_indices.long(), src=(ablated_acts - top_acts).to(x.dtype))
    return x + self.bank.decode(diff, kind, layer_idx)
```

This encode-once, decode-difference approach is cleaner and cheaper.

### Measuring seed recovery

For both the baseline and ablated runs on `neg_tokens`, capture the seed's
activation using the same `capture_hook` pattern as `ProbeDatasetBuilder._calculate_argmax`:

```python
def measure_seed_activation(inference, bank, neg_tokens, seed_layer, seed_kind_idx, seed_latent_idx):
    result = []
    def hook(layer_idx, activations):
        if layer_idx == seed_layer:
            act = activations[seed_kind_idx]
            top_acts, top_indices = bank.encode(act, bank.kinds[seed_kind_idx], layer_idx)
            is_target = (top_indices == seed_latent_idx)
            seed_act = torch.where(is_target, top_acts, torch.zeros_like(top_acts)).sum(dim=-1)
            result.append(seed_act.mean().item())
    with torch.no_grad():
        inference.forward(neg_tokens, activations_callback=hook, return_activations=False)
    return result[0] if result else 0.0
```

The ablation delta is then `measure_with_ablation - measure_baseline`. A
positive delta means the seed recovered when the candidate was removed →
confirmed inhibitor.

---

## Summary of Forward/Backward Passes per Seed

| Phase | Type | Count | Notes |
|---|---|---|---|
| 1. Gradient scan | Instrumented + grad | 1 fwd + 1 bwd | Scores all upstream activators |
| 2. Differential scan (neg) | No-grad | 1 fwd | Collects neg activations |
| 3a. Ablation baseline | No-grad | 1 fwd | Seed activation on unmodified neg_ctx |
| 3b. Binary-split ablation | No-grad | ~log₂(N) fwd | Confirms inhibitors; N = candidate count |
| 4. Downstream (optional) | Reuse Phase 1 graph | K bwd | One per downstream candidate |
| 5. Minimality pruning | No-grad | M × LOO fwd | M = circuit size; disabled if threshold=0 |
| 6. Evaluation | No-grad | 4 fwd | faith, faith_local, suff, compl |
| **Typical total** | | **~20-30 fwd, 1-2 bwd** | **With N=128 candidates, K=0 downstream** |

---

## Edge Cases to Handle

1. **No neg_tokens available:** Fall back to gradient-scan-only mode (skip
   Phases 2-3). Log a warning. The circuit will contain only activators.

2. **Seed not in TopK on neg_ctx:** Expected — that's why the seed is
   suppressed. `measure_seed_activation` should return 0.0 in this case, which
   is the correct baseline.

3. **Binary split finds zero inhibitors:** The full-group ablation test fails
   immediately (1 forward pass). Proceed with activators only.

4. **All candidates pruned by minimality:** Return `None` if only the seed
   remains (same as other methods).

5. **`torch.compile` active:** Disable before Phases 1 and 3, re-enable after.
   Phase 2 can run compiled.
