# Circuit-tracer comparison (infra built 2026-08-11; runs pending GPU)

PLACEMENT DECISION (Daniel, 2026-08-11): this is APPENDIX material, not
body. The paper is complete without it; the body is already over its
page budget; a method-vs-method comparison on a third model belongs in
the appendix. Revisit only if the result is striking enough to change a
body claim (e.g. their subgraph passing our exact-forward exams
cleanly, which would be a notable positive result for attribution
graphs). Report as a bonus contribution, not a gate.

TARGET UNDER REVIEW: the GPT-2 + Dunefsky/Chlenski route below is BUILT
and validated, but those transcoders are ReLU (dense-code), which is
outside the architecture domain we established. EleutherAI's
skip-transcoder-Llama-3.2-1B-131k is genuine TopK (k=32, 131,072
latents, all 16 layers, skip connections) AND Llama-3.2-1B is natively
supported by circuit-tracer — strictly better on both axes. Practical
note: meta-llama/Llama-3.2-1B is gated ("manual" approval) on HF;
unsloth/Llama-3.2-1B is an ungated mirror of the same weights.
See "Llama option" at the end.

The well-posed version of "how does our method compare to attribution
graphs". Both methods run on the SAME model and the SAME feature
space, so the comparison isolates ONE variable: the discovery
mechanism.

  theirs  linear attribution on a frozen-attention replacement model
          (circuit-tracer, the reference implementation, unmodified)
  ours    tri-amp mask optimised against the EXACT forward pass

Contrast this with the SFC replication, which held the estimator and
node type fixed and varied the ENDPOINT. Here endpoint and feature
space are held fixed and the mechanism varies.

## Why GPT-2 + these transcoders

circuit-tracer supports Gemma-2/Llama/Qwen/Gemma-3 natively, not
GPT-2 — but its transcoder interface is pluggable (`TranscoderSet` +
`from_pretrained_and_transcoders`) and it is TransformerLens-based,
which has gpt2-small. Meeting on GPT-2 keeps OUR side cheap (124M) and
lets their library stay the canonical implementation of their method.

## What is built (all CPU work, no GPU used)

- `convert_transcoders.py` -> `transcoders_ct/layer_{0..11}.safetensors`
  Dunefsky/Chlenski pickles (`sae_training` format) converted into
  circuit-tracer's SingleLayerTranscoder layout. THREE real differences,
  each asserted numerically per layer (deviations ~1e-5):
    * transpose: source W_enc is [d_model, d_sae], target [d_sae, d_model]
    * input pre-bias fold: the source CENTRES its input (x - b_dec) while
      circuit-tracer's encode does not, so
      b_enc_target = b_enc_src - b_dec_src @ W_enc_src
      (the source's b_dec_out is the target's b_dec, the output bias)
    * LAYERNORM CONVENTION FOLD — see below. This one would have silently
      corrupted BOTH sides.
  meta.json records the hooks and the input convention.

### The LayerNorm convention (a caught silent-corruption bug)

These transcoders were trained against `ln2.hook_normalized` under
TransformerLens fold_ln=TRUE, i.e. PURE normalisation with the LN
affine folded away. circuit-tracer loads its model with fold_ln=FALSE,
where that hook yields the FULL ln_2 output. Measured on GPT-2 layer 6,
the difference is not subtle:

    input fed to transcoder      rel_err vs mlp_out    L0
    pure normalisation (correct)        0.41           68
    full ln_2 output (wrong)            0.79            5

So the raw weights would have been wrong in circuit-tracer out of the
box, and our first draft of ours_gpt2.py was wrong too. Fix: fold the
LN affine into the encoder as well, using x_norm = (x_full - b)/w:
    W_enc <- W_enc / w_ln,  b_enc <- b_enc - (b_ln/w_ln) @ W_enc
after which the transcoders consume the FULL ln_2 output natively and
both sides agree. Verified exact (max |dfeature| ~1e-5, identical L0).
GPT-2 layer 3 has one near-degenerate ln_2 weight (2.6e-4), which
inflates a single encoder column ~4000x; the fold stays algebraically
exact and the per-layer equivalence assert certifies it.

`validate_transcoders.py` is the end-to-end check on REAL activations:
every layer reconstructs its MLP output with FVU 0.003-0.27 (median
0.18, no layer above 0.6), L0 7-117 — healthy transcoder behaviour,
confirming the conversion in practice and not merely on random
vectors.
- `ours_gpt2.py` — our side: GPT-2 + the same transcoders, tri-amp with
  TRANSCODER intervention semantics (below), scan + run, held-out
  48/16, verified-inactive negatives, nulls, anchor-support statistic.
  Writes ours_rows.jsonl and ours_members.jsonl (membership + alphas,
  needed for the overlap comparison).
- `../../dev-notes/data/venv-ct/` — ISOLATED venv for circuit-tracer. Required: it pins
  transformers<=4.57.3 while the main venv runs 5.1.0. Built with
  --system-site-packages so torch is shared rather than re-downloaded.
  The two sides never share a process; they exchange node sets as JSON.
- `../circuit-tracer-src/` — upstream clone (pinned by git SHA at
  install time).

## Transcoder intervention semantics (the one new mechanism)

A per-layer transcoder reads the MLP input (ln_2-normalised residual)
and predicts the MLP output:
    c   = relu(x_in @ W_enc.T + b_enc)
    rec = c @ W_dec + b_dec
    err = mlp_out_true - rec          (preserved, as with our SAEs)
An intervention c -> chat therefore edits the MLP output by
    mlp_out <- mlp_out + (chat - c) @ W_dec
so an unmodified circuit reproduces the model EXACTLY and the
transcoder's own reconstruction error passes through untouched. Same
delta algebra as the SAE harnesses, one level of indirection out.

Node universe: MLP-transcoder features at layers < seed layer.
Attention and embeddings stay live and undecomposed — also true of
circuit-tracer's PLT graphs, so both methods share the universe.

## The experiment (pending GPU)

1. Scan for seed features (firing-rate band) at layers 4/6/8.
2. OURS: tri-amp membership for each seed feature.
3. THEIRS: attribution graph on the same prompts; read the subgraph of
   upstream transcoder features feeding the SAME seed feature (their
   graphs contain feature->feature edges, so an internal root is
   well-posed).
4. COMPARE: node-set overlap, and — the part that matters — score
   THEIR subgraph under OUR exact-forward exams (faithfulness,
   necessity, drive) as a membership set, and vice versa where their
   metrics allow.

Either outcome is publishable: if their subgraph passes our exams,
that is strong positive evidence for attribution graphs; if it does
not, it quantifies what the frozen-attention linear surrogate costs.

## Caveat to measure, not assume

These transcoders are ReLU+L1 (l1=8e-5, 24,576 features), i.e. a
DENSE-code architecture, not Top-K. Per 030-cross-sae /
-topk-2026-08-11, our fitted-amplitude null only discriminates where
anchor support is low, so `ours_gpt2.py` measures the anchor-support
rate per seed and it must be read BEFORE any null claim here. The
overlap-and-exams comparison does not depend on the null, so the
headline experiment stands either way — but the null column may need
to be reported as uninformative on this architecture.

## Status

Built and verified on CPU: conversion (12/12 layers, equivalence
asserted), GPT-2 + all transcoder weights downloaded (1.7 GB cached),
ours_gpt2.py syntax-clean. PENDING: venv-ct install completing, then
a smoke test of both sides, then the runs.


## Llama option (recommended target; supersedes GPT-2 if adopted)

  model        EleutherAI trained against Llama-3.2-1B; ungated mirror
               unsloth/Llama-3.2-1B (meta-llama/... is gate="manual")
  transcoders  EleutherAI/skip-transcoder-Llama-3.2-1B-131k
               TopK k=32, 131,072 latents, d_in 2048, decoder-normalised,
               skip_connection=true, all 16 layers (layers.N.mlp/)
  why          (a) genuine TopK = inside our method's established domain,
                   where the fitted-amplitude null has teeth (anchor
                   support ~0.2% at k=32-scale sparsity);
               (b) circuit-tracer supports Llama-3.2-1B NATIVELY, so
                   their library stays canonical — no GPT-2 port, no
                   misconfiguration risk;
               (c) same `sparsify` weight layout we already handled for
                   the Pythia Top-K SAEs, so the loader is known work.
  cost         1B params. Shallow/mid seeds keep it comfortable: six
               upstream transcoders in bf16 ~3 GB + model ~2 GB on 16 GB.
  carry-over   harness structure, transcoder intervention semantics, the
               conversion/validation discipline (fold checks, end-to-end
               FVU) all transfer unchanged. Only the loader differs.
  skip term    W_skip is a dense linear bypass that belongs to no
               feature. It is untouched by our interventions (delta is
               still (chat - c) @ W_dec), so the algebra is unchanged —
               but it is undecomposed computation, like attention, and
               must be stated as such. It hits both methods identically.

GPT-2 infra is retained either way as a fallback and a possible second
data point on a different dictionary family.

---

# LLAMA TARGET — BUILD COMPLETE, AGREEMENT GATE PASSED (2026-08-11)

Target adopted: Llama-3.2-1B (ungated mirror unsloth/Llama-3.2-1B) with
EleutherAI/skip-transcoder-Llama-3.2-1B-131k — genuine TopK (k=32,
131,072 latents, 16 layers, skip connections), and a model
circuit-tracer supports natively so their library stays canonical.
This supersedes the GPT-2 + ReLU route, which is retained as a fallback
and a possible second dictionary family.

## Files

  llama_loader.py     probe + convert (sparsify -> circuit-tracer keys)
  check_hooks.py      TL hook vs HF MLP input
  check_agreement.py  THE GATE: both stacks, same tokens, same features
  debug_agreement.py  discriminates "my bug" from "their surrogate"
  ours_llama.py       our side (tri-amp, transcoder semantics)
  theirs_llama.py     their side (attribution graph -> seed's row)
  compare.py          overlap + size sweep
  transcoders_llama_ct/  16 converted layers (~35 GB)

## FIVE conventions that had to be MEASURED, not assumed

Each would have produced confident-looking numbers from meaningless
features. This is the main methodological lesson of the exercise.

1. GPT-2 LayerNorm placement (fold_ln): transcoders trained on pure
   normalisation, circuit-tracer loads the full ln_2 output. Wrong
   choice: reconstruction 0.41 -> 0.79, L0 68 -> 5.
2. sparsify input centring: the Llama transcoders subtract b_dec;
   circuit-tracer's encode does not. Probed empirically (FVU 0.166 with
   centring vs 0.233 without) and folded into b_enc.
3. TransformerLens RMSNorm hook placement: ln2.hook_normalized sits
   BEFORE the RMS weight; the HF MLP input includes it. Off by
   2.3-6.3x. Fixed in memory on their side (W_enc <- W_enc * w), so the
   35 GB on-disk set stays single-copy and canonical.
4. BOS prepending: their ensure_tokenized() prepends a special token.
   Our windows now match, or the two methods would analyse different
   sequences at shifted positions.
5. Position 0 zeroing: they deliberately zero the prepended token's
   activations (zero_positions = slice(0,1)). Our anchor selection now
   excludes position 0 so both methods read the same positions.

Also: the skip term is essential (FVU 0.17 with, 0.94 without), and a
top-k "selection changed" assert had to be replaced with a code-value
assert — the fold is algebraically exact, so indices can only differ
where two features are near-tied at the k-th boundary and float noise
(~1e-6) reorders them.

## Agreement gate result (check_agreement.py, all 16 layers)

  identical top-k support on every layer; relative error 2.4e-07 to
  1.3e-06 on positions >= 1; their position 0 confirmed zeroed.
  => any difference in the resulting circuits is a difference in
     METHOD, not in convention.

## Status

Build and verification COMPLETE. Running: our seed scan (layers 4, 6).
Then: our tri-amp runs -> their attribution graphs -> compare.py
(overlap vs chance, plus scoring THEIR subgraph under OUR exact-forward
exams at 1x/2x/4x/8x our circuit size).
