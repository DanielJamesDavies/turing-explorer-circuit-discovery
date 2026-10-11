# 070 — public-model adapters for the production engine (DAN-34 / DAN-35 / DAN-68 / DAN-74)

The production engine (`_build_mode_method("ablation_gradient", "mask", ...)` → `run_learned_mask`, the src eval
patchers and `src/eval/wcm_eval.py`) now runs unchanged on **Pythia-70m + EleutherAI/sae-pythia-70m-32k** and
**GPT-2 small + jbloom OAI v5 32k**. TuringLLM stays the default and its path is byte-for-byte the code it was.

## How a family is selected

`config.model.family` (`turingllm` default | `pythia-70m` | `gpt2-small`), or per call
`Inference(device, family=...)` / `SAEBank(device=..., family=...)`. Both constructors dispatch in `__new__`, so
any code that builds them picks up `model: {family: pythia-70m}` from the yaml. (049's `setup` also loads the
TuringLLM stores; those are DAN-37's job.)
Extra knobs: `model.dtype` (public families; None = float32, what 033/053 ran at) and `model.sae_preload`.

| Piece | File | What it is |
|---|---|---|
| family registry | `src/model/families.py` | kinds in forward order, `same_layer_upstream`, SAE repos/format/k/normalisation |
| model adapter | `src/model/hf_adapters.py` | `HFSiteModel` (+ Pythia / GPT-2 subclasses), `HFInference` |
| SAE bank | `src/sae/public_bank.py` | `PublicSAEBank(SAEBank)`: same encode/decode contract, lazy loading |
| hook dispatch | `src/model/hooks.py` | `multi_patch` / `capture_activations` / `patch` route to the adapter |
| causal order | `src/eval/ablation_faithfulness.py::upstream_sites` | reads `bank.same_layer_upstream` when present |

**The abstraction.** The engine touches a model through three calls only: `model(tokens, return_all_logits)`,
`multi_patch(model, transform)` and `capture_activations(model, callback)`. `HFSiteModel` implements all three
with PyTorch hooks on the HF modules at the SAEs' training hook points; the HF forward itself is untouched (an
unedited pass is the HF pass, bitwise). The bank keeps the `(top_acts, top_idx)` / `decode(add_bias=False)` /
`saes[kind][layer].encoder.weight` contract by loading the public dictionaries into the repo's own `SAE` class.

**Site activation = the SAE's input.** Every `x` a transform or callback sees is what the dictionary reads. For
GPT-2 that is `x_n = (x − mean)/(std + 1e-5)`; an edit `x_n → x_n'` is applied as `x + (x_n' − x_n)·(std + 1e-5)`
(053's convention). So the engine's seed tap `x @ w + b` and `decode(delta, add_bias=False)` keep their meaning
with no engine change. Callers holding a raw model-space stream use `bank.site_input(x, kind, layer)`.

**Pre-activation API (for the scorer, Agent A).** `bank.preact(x, kind, layer)` → every latent's pre-activation
`[.., d_sae]` (same linear op / dtype / autocast as `encode`, so `relu(preact).topk(k)` reproduces `encode`
exactly — tested at every site of both models); `bank.preact_vector(kind, layer, idx)` → `(w, b)` = (encoder row,
`b_enc − W_enc b_dec`), bitwise what the engine taps (`x @ w + b`); on TuringLLM it returns exactly
`encoder.weight[idx]`, `_get_bias_eff()[idx]`.

## Causal order (evidence)

* **Pythia-70m**: `config.json` has `use_parallel_residual: true`; HF `GPTNeoXLayer.forward` (transformers 5.1.0)
  computes `mlp(post_attention_layernorm(hidden_states))` on the layer INPUT and returns
  `mlp_output + attn_output + hidden_states`. So within a layer: attn, mlp → resid; **mlp(L) does not read
  attn(L)**. The SAEs read `layers.L.attention` (GPTNeoXAttention output[0]), `layers.L.mlp`, `layers.L` (layer
  output) — the modules 033 hooked. Test: an edit at attn(L) leaves mlp(L)'s output bitwise unchanged and shifts
  resid(L) by the decoded delta. Consequences: mlp(L)'s upstream set loses attn(L) (033 and the TuringLLM rule
  included it — inert, since its gate had zero data gradient, but wasted); **attn(0) and mlp(0) have no upstream
  SAE sites at all** (they read only the embedding) and cannot be fitted — the DAN-9 frame must exclude them.
* **GPT-2 small**: sequential block, `attn < resid_mid < mlp < resid`. resid_mid is edited inside the attention
  hook (`a' + (r' − r)`), so the block's residual add yields `r'`: the test checks that a resid_mid edit shifts
  both the ln_2 input and `block_out − mlp_out` (the residual stream) by the decoded delta × std.

## DAN-68

053 replaced resid-mid with a pre-hook on `ln_2`, which changes only the MLP input (`GPT2Block` keeps `residual`
and returns `residual + mlp`). The audit reproduces 053's resid-mid dCE with that splice to 4 decimals. The
correct replacement is similar at L0–L2 (+0.017 / +0.005 / +0.029 vs 053's +0.011 / +0.007 / +0.025) and much
larger from L3 on: L3 +0.0098 → +0.0725, L6 −0.0071 → +0.0908, L9 +0.0358 → +0.1626, L11 +0.0514 → +0.2136. The
correct numbers track resid-post's, as they should. Any 053 resid-mid circuit result is affected.
`hooks.patch(kind="resid")` had the same bug class (norm_2 pre-hook = MLP input only) and no callers: **fixed** to
replace the block output (the site `multi_patch` calls "resid"); its test now asserts the corrected semantics.

## Reconstruction audit (`recon_audit.py` → `recon_<family>.json`)

053's exact text (16 × 64 wikitext tokens, no BOS). `ident` (zero decoded delta through the adapter) is 0.0 at
every site of both models. `ref` = max |code − independent transcription|.

**GPT-2** (clean CE 4.2704) matches `053/sae_recon_32k.json` at every layer 053 measured (0, 3, 6, 9, 11): FVU and
L0 to 4 decimals, dCE to 4 decimals for attn / mlp / resid. resid_mid dCE differs because of DAN-68 (above).
`ref` ≤ 3e-5 everywhere except mlp L11 (7.7e-2): one token whose 32nd and 33rd pre-activations tie to 6 decimals
(0.077279), resolved differently by the two algebraically equal formulas — 2 of 33.5M code entries.
Slots = 32 at every site; non-zeros 31.69–32.00 (SAELens applies ReLU after top-k; 053 measured the same).
**Layer 10 is not in the local HF cache** for any of the four repos (see "Left").

**Pythia-70m** (clean CE 5.0371; all 18 dictionaries present). 033 published no FVU / dCE table, only "exactly 16
non-zeros per position"; that holds at every token of every site (slots 16, non-zeros 16.00), and the codes
equal 033's transcription (`(x − b_dec) W_encᵀ + b_enc → relu → top-16` on the raw safetensors) to ≤ 4.6e-5.

| L | attn FVU / dCE | mlp FVU / dCE | resid FVU / dCE |
|---|---|---|---|
| 0 | 0.147 / +0.342 | 0.015 / +0.018 | 0.088 / +0.324 |
| 1 | 0.250 / +0.030 | 0.154 / +0.070 | 0.151 / +0.164 |
| 2 | 0.236 / +0.120 | 0.017 / +0.110 | 0.044 / +0.302 |
| 3 | 0.284 / +0.134 | 0.247 / +0.142 | 0.047 / +0.403 |
| 4 | 0.293 / +0.022 | 0.081 / +0.111 | 0.133 / +0.355 |
| 5 | 0.287 / +0.035 | 0.061 / +0.146 | 0.158 / +0.471 |

These k = 16 dictionaries are much lossier per site than GPT-2's k = 32 ones (resid dCE 0.16–0.47 nats vs
0.006–0.30). That doesn't affect the method (the SAE error is preserved), but it belongs in the paper's substrate
description.

## Smoke (`smoke.py`)

One target per model from cached wikitext (strongest 16 windows; 16 verified-silent contrast windows), one
WCM[Z+C+A] fit (protocol-v1 settings, steps cut) through `_build_mode_method` + `discover()` with 062's context
injection, then one `wcm_eval.score_circuit` pass. CPU.

| model | target | steps | upstream sites | members | loss | cf (bare) | free0 / freeM_dense / freeN_dense (pre) | φ_cf / φ_pin0 | time |
|---|---|---|---|---|---|---|---|---|---|
| Pythia-70m | 2.resid.27816 (fires 11.3%) | 20 | 8 | 262,144 (all; θ₀ = 4 burn-in) | 257 → 248 | 0.947 | 1.017 / 1.017 / – | – | 75 s fit |
| Pythia-70m | same | 150 | 8 | **31** | 257 → 11.7 | 0.767 | 1.080 / 1.057 / 1.006 | 0.913 / 0.762 | 921 s fit, 24 s score |
| GPT-2 small | 3.resid_mid.29966 (fires 9.0%) | 20 | 13 (L0–2 all + 3.attn) | 425,984 (all) | 418 → 403 | 0.950 | 1.003 / 1.003 / 1.003 | 0.957 / 0.984 | 554 s fit, 95 s score |

The 150-step Pythia fit prunes as the mask should, so gradients reach the gates through the adapter.
`smoke_pythia-70m_150steps.json` and `smoke_gpt2-small.json` hold the full rows. (`smoke_pythia-70m.json` holds
the latest run, which is the 150-step one.) These are engine-runs-end-to-end checks, not results: 12 training
contexts, 4 held out, no λ calibration.

## Left / out of scope (what the engine still needs)

* **GPT-2 layer 10 SAEs** must be downloaded (`v5_32k_layer_10` in attn-out / resid-mid / mlp-out, and
  `v5_32k_layer_10.pt` in resid-post). Everything else for GPT-2 is done and tested.
* DAN-37 data passes: top / mid / neg_ctx stores, `seq_repr`, `latent_stats`, corpus loader per model.
  `capture_activations` already yields SAE-input activations in `bank.kinds` order (`[B, L, K, T, N]`, K = 3 / 4),
  and `bank.encode` the codes, so the collection passes can run through the same API. The tokenizer is
  `families.FAMILIES[name].tokenizer_id`; both SAE families trained without BOS, context 64.
* Probe-dataset builder (`ProbeDatasetBuilder` / `protocol_contexts`) for the new stores; λ calibration (DAN-42);
  k-dependent settings (protocol §9.1: `merge.py` / `pass_rule.py` K = 128, rank clamp already reads `bank.k`).
* Not adapted (not on the WCM path): `stop_grad_at` / `multi_stop_grad` (raise on adapters), gradient
  checkpointing and `torch.compile` (no-ops), `counterfactual_gradient.py:461` and `wcm_eval.py:371` sort sites
  alphabetically (`sorted(upstream_sites(...))`) — harmless for the math, but not causal order for GPT-2's
  resid_mid; the mask hop now sorts by `bank.kinds` order.
