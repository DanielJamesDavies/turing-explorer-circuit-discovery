# Plan: Focal Seed Discovery Fixes

> **Goal:** Address the structural mismatch between `counterfactual_gradient` and focal/monosemantic seeds (high cohesion, low coupling) that causes 0 faithful circuits when `focal_monosemantic` is used as a seed criterion.
>
> **Created:** 2026-04-19

---

## Background

When seed criteria are hub-oriented (`rare_hub`, `connectivity`, `pagerank_centrality`), `counterfactual_gradient` works well. When criteria select focal/monosemantic seeds (`focal_monosemantic`), the method consistently produces 0 faithful circuits.

Three contributing factors were identified:

1. **Metric bias at partial data**: `focal_monosemantic` is biased toward under-observed latents at 10% data, preferentially selecting seeds where all discovery signals are weakest.
2. **Threshold scale mismatch**: The absolute `activator_threshold = 0.3` is effectively ~4× more restrictive for focal seeds (lower `a_posctx`) than for hubs.
3. **Acceptance criterion tests sufficiency only**: `cf_faith` requires the circuit to be causally sufficient (injecting activators on neg recreates the seed's firing). AND-gate focal seeds inherently fail sufficiency but would pass necessity.

Additionally, at <10% data, seeds with very few hard negatives silently pass a near-empty neg batch to the gradient pass, corrupting the gradient estimate.

---

## Fix 1 — Mask empty `top_ctx` slots in `focal_monosemantic` ✅

**File:** `src/circuit/feature_selection.py`

**Current code (lines 309–314):**
```python
if "focal_monosemantic" in active:
    cohesion = top_ctx.ctx_seq_val.to(self.device).float().mean(dim=-1)   # [C, D]
    n_partners = (top_coactivation.top_values.to(self.device) > 0).float().sum(dim=-1)
    score = cohesion / (n_partners + 1.0)
    score = score.masked_fill(latent_stats.seq_count.to(self.device) < 5, -1e9)
    all_seeds.append(self._top_k(score, "focal_monosemantic"))
```

**Problem:** `ctx_seq_val` has shape `[C, D, 64]`. For a latent with only 10 stored contexts, slots 11–64 are zero. `.mean(dim=-1)` divides by 64, deflating cohesion by `n_stored / 64`. For a latent with 5 observations, cohesion is deflated 12.8×. Meanwhile, `n_partners` is also small at partial data, so the two biases partially cancel in a noisy, unpredictable way — the metric surfaces under-observed latents by coincidence rather than by their actual properties.

**Fix:** Use a masked mean — sum only filled slots, divide only by the number of filled slots.

```python
if "focal_monosemantic" in active:
    v = top_ctx.ctx_seq_val.to(self.device).float()          # [C, D, 64]
    filled = (v > 0).float()
    cohesion = (v * filled).sum(dim=-1) / filled.sum(dim=-1).clamp(min=1.0)  # [C, D]
    n_partners = (top_coactivation.top_values.to(self.device) > 0).float().sum(dim=-1)
    score = cohesion / (n_partners + 1.0)
    score = score.masked_fill(latent_stats.seq_count.to(self.device) < 5, -1e9)
    all_seeds.append(self._top_k(score, "focal_monosemantic"))
```

**Effect:** Cohesion now correctly measures "mean activation on the contexts where this latent was observed," independent of how full the store is. A latent with 10 strong activations scores the same cohesion regardless of whether it was collected from 10% or 100% of the corpus.

---

## Fix 2 — Minimum partner observation guard in `focal_monosemantic` ✅

**File:** `src/circuit/feature_selection.py`

**Current behaviour:** When `n_partners = 0`, the denominator is 1.0 and the score equals cohesion. There is no coupling signal at all — the criterion silently degenerates into a pure cohesion rank. At 10% data, a substantial population of latents has zero observed partners simply due to data incompleteness. These seeds have no neg_ctx sequences, unreliable posctx means, and produce degenerate gradient passes.

**Fix:** Add a `masked_fill` after the score computation to exclude latents with fewer than a minimum number of observed partners. The threshold should be a config parameter; 4 is a reasonable default.

```python
if "focal_monosemantic" in active:
    v = top_ctx.ctx_seq_val.to(self.device).float()
    filled = (v > 0).float()
    cohesion = (v * filled).sum(dim=-1) / filled.sum(dim=-1).clamp(min=1.0)
    n_partners = (top_coactivation.top_values.to(self.device) > 0).float().sum(dim=-1)
    score = cohesion / (n_partners + 1.0)
    score = score.masked_fill(latent_stats.seq_count.to(self.device) < 5, -1e9)
    score = score.masked_fill(n_partners < 4, -1e9)   # NEW: require observed coupling signal
    all_seeds.append(self._top_k(score, "focal_monosemantic"))
```

**Effect:** `focal_monosemantic` now means "high cohesion *relative to* observed partners, with enough partner observations that the denominator is meaningful." Latents with 0–3 observed partners are excluded — they are ambiguous between genuinely isolated and not yet observed.

**Config addition (optional, `config.yaml`):**
```yaml
seed_criteria_params:
  focal_monosemantic_min_partners: 4  # minimum observed coact partners before the criterion applies
```

---

## Fix 3 — Scale `activator_threshold` relative to `target_act_pos` ✅

**File:** `src/circuit/discovery/counterfactual_gradient.py`

**Current code (lines 224–226):**
```python
for upstream_fid, score in activator_scores.items():
    if score < self.activator_threshold:   # 0.3 absolute
        continue
```

**Problem:** The gradient score for absent activator j is analytically:

$$\text{score}_j \approx 2 \cdot a_\text{posctx} \cdot \bigl(W_{\text{dec},j} \cdot J_\text{resid}(x_\text{neg}) \cdot w_\text{enc,seed}\bigr)$$

This comes from differentiating `target_scalar = -((pre_act_at_peak - a_posctx)^2).mean()` where `pre_act_at_peak ≈ 0` on neg, giving `∂loss/∂(pre_act) ≈ -2·a_posctx`, then the chain rule through the residual stream Jacobian and decoder.

Scores scale **linearly with `target_act_pos`**. A hub with `a_posctx = 4.0` produces activator scores ~4× larger than a focal seed with `a_posctx = 1.0` for identical structural alignment. The absolute threshold 0.3 is therefore 4× more restrictive for focal seeds in terms of structural signal required to pass.

**Fix:** Compute an effective threshold proportional to `target_act_pos` immediately after it is computed (line 207), and use it in the threshold checks at lines 225 and 244.

`target_act_pos` is already available at the right point in `_discover`. The proportionality constant is still the config `activator_threshold` / `inhibitor_threshold` — they now represent a *fraction of the loss-gradient scale* rather than an absolute magnitude.

```python
# After line 207 (target_act_pos is now known):
effective_activator_threshold = self.activator_threshold * max(target_act_pos, 0.1)
effective_inhibitor_threshold = self.inhibitor_threshold * max(target_act_pos, 0.1)
```

Then replace lines 225 and 244:
```python
# Line 225 (was: if score < self.activator_threshold)
if score < effective_activator_threshold:
    continue

# Line 244 (was: if abs(score) < self.inhibitor_threshold)
if abs(score) < effective_inhibitor_threshold:
    continue
```

The `max(..., 0.1)` floor prevents the threshold collapsing to zero for near-inactive seeds while still normalising the scale.

**Effect on each seed type:**

| Seed type | `a_posctx` | Old threshold | New threshold | Raw gradient for true activator |
|---|---|---|---|---|
| Hub | 4.0 | 0.3 | 1.2 | ~8.0 — still passes easily |
| Focal | 1.0 | 0.3 | 0.3 | ~2.0 — now comparably selective |
| Weak focal | 0.3 | 0.3 | 0.1 (floor: 0.03) → uses 0.1 | ~0.6 — proportionally correct |

The net effect: hubs become slightly more selective (preventing false positives from hub-scale gradient noise), and focal seeds are no longer systematically over-pruned.

---

## Fix 4 — Accept circuits via necessity when sufficiency fails

**File:** `src/circuit/discovery/counterfactual_gradient.py`

**Current code (lines 320–324):**
```python
if cf_faith < self.min_faithfulness:
    logger.reject(
        f"counterfactual_faithfulness {cf_faith:.4f} < min_faithfulness {self.min_faithfulness}"
    )
    return None
```

**Problem:** `cf_faith` exclusively tests *sufficiency* — injecting discovered activators on neg must recover posctx-level seed activation. For focal AND-gate seeds this is structurally hard: the seed requires a full conjunction of conditions, and injecting a handful of upstream activations cannot recreate the full residual-stream context that the conjunction requires.

`sup_score` tests *necessity* — suppressing the discovered activators on posctx must silence the seed. For AND-gate seeds this is easier to satisfy: remove one required input and the output drops.

The two scores are:
```
cf_faith  = (a_neg_intervened  - a_baseline) / (a_posctx - a_baseline)  # sufficiency
sup_score = (a_posctx - a_pos_intervened) / (a_posctx - a_baseline)     # necessity
```

**Fix:** Accept if there is moderate evidence from either direction, with a small floor on `cf_faith` when accepting via necessity. The floor guards against spurious `sup_score` caused by residual-stream cascade interference (suppressing an active latent can silence the seed through unrelated pathways).

Replace lines 320–324 with:

```python
passes_sufficiency = cf_faith >= self.min_faithfulness
passes_necessity   = (
    sup_score >= self.min_sup_score
    and cf_faith >= self.min_cf_faith_for_necessity
)

if not (passes_sufficiency or passes_necessity):
    logger.reject(
        f"cf_faith {cf_faith:.4f} (min {self.min_faithfulness}) | "
        f"sup_score {sup_score:.4f} (min {self.min_sup_score}) — both below threshold"
    )
    return None
```

**Config additions (`config.yaml`, `counterfactual_gradient` block):**
```yaml
counterfactual_gradient:
  min_faithfulness: 0.2          # sufficiency threshold (unchanged)
  min_sup_score: 0.1             # necessity threshold (new, lower)
  min_cf_faith_for_necessity: 0.05  # cf_faith floor when accepting via necessity only
```

**Schema additions (`src/config.py`, `CounterfactualGradientConfig`):**
```python
min_sup_score: float = 0.1
min_cf_faith_for_necessity: float = 0.05
```

**Effect:** Hub seeds continue to pass via sufficiency unchanged. Focal seeds that pass necessity (suppressing their activators silences them on posctx) but not sufficiency (injecting activators on neg doesn't recreate firing) are now accepted, provided their `cf_faith` is at least non-trivially positive (≥ 0.05) — ruling out pure residual-stream cascade artefacts.

---

## Fix 5 — `neg_mode` fallback when hard negatives are too sparse

**File:** `src/circuit/discovery/counterfactual_gradient.py`

**Current code (lines 360–366, `_get_neg_tokens`):**
```python
if self.neg_mode == "close":
    tokens = probe_data.neg_tokens[:self.max_neg_sequences]
    if tokens.shape[0] == 0:
        logger.reject("no negctx sequences available (neg_mode=close)")
        return None
    logger.note(f"neg_mode=close: {tokens.shape[0]} hard-negative sequences")
    return tokens
```

**Problem:** If a seed has 1–3 hard negatives (common for rare focal seeds at <10% data), the gradient is averaged over a batch of 1–3 sequences. The score for each activator candidate is dominated by the specific token positions and residual stream state of those 1–3 sequences, which may not be representative. The method doesn't fail loudly — it silently proceeds with a near-useless gradient estimate.

`neg_mode = "distant"` draws from a pool of 512 corpus sequences and selects the most distant from posctx in SAE latent space, filtering to non-activating ones. For focal seeds it's more reliable because distant sequences are structurally diverse — even if the seed's exact causal pathway isn't open on any single distant sequence, the diversity of the pool ensures more gradient paths are represented across the batch.

**Fix:** Add a fallback threshold. If `neg_mode = "close"` produces fewer than `min_neg_for_close` sequences, fall back to `distant` mode rather than running with an insufficient neg batch.

```python
if self.neg_mode == "close":
    tokens = probe_data.neg_tokens[:self.max_neg_sequences]
    if tokens.shape[0] == 0:
        logger.reject("no negctx sequences available (neg_mode=close)")
        return None
    if tokens.shape[0] < self.min_neg_for_close:          # NEW
        logger.note(
            f"neg_mode=close: only {tokens.shape[0]} hard negatives "
            f"(< min_neg_for_close={self.min_neg_for_close}), falling back to distant"
        )
        return self._get_distant_tokens(
            seed_comp_idx, seed_latent_idx,
            pos_tokens_eval, pos_argmax_eval,
            logger,
        )
    logger.note(f"neg_mode=close: {tokens.shape[0]} hard-negative sequences")
    return tokens
```

**Config addition:**
```yaml
counterfactual_gradient:
  min_neg_for_close: 4   # fall back to distant if fewer close hard-negatives are available
```

**Schema addition (`CounterfactualGradientConfig`):**
```python
min_neg_for_close: int = 4
```

**Cost:** One extra pool forward pass (512 no-grad sequences, microbatched) when the fallback fires. For focal seeds at partial data this is acceptable — the alternative is a corrupted gradient estimate that wastes the more expensive grad-enabled pass.

---

## Implementation Order

| Fix | Files touched | Effort | Expected impact |
|---|---|---|---|
| ~~1. Masked cohesion mean~~ ✅ | `feature_selection.py` | 3 lines | Corrects metric for partial data |
| ~~2. Min partner guard~~ ✅ | `feature_selection.py` | 1 line | Stops worst under-sampled seeds from being selected |
| ~~3. Proportional threshold~~ ✅ | `counterfactual_gradient.py` | 4 lines | Most likely to directly increase faithful circuit count for focal seeds |
| 4. Necessity acceptance | `counterfactual_gradient.py`, `config.py`, `config.yaml` | ~15 lines | Unlocks AND-gate focal circuits that satisfy necessity |
| 5. `neg_mode` fallback | `counterfactual_gradient.py`, `config.py`, `config.yaml` | ~12 lines | Prevents silent gradient corruption at partial data |

Fixes 1–3 are pure numerical corrections with no trade-offs. Apply them first. Fix 4 increases recall at the cost of potentially accepting a few more false-positive circuits — worth validating on a small run before enabling globally. Fix 5 adds forward-pass cost only when the fallback fires, which should be infrequent after a full data collection run.
