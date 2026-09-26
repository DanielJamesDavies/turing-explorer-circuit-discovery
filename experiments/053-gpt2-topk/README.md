# GPT-2 small + OpenAI v5 TopK SAEs (2026-09-20)

**Why this substrate.** 033's phi_pin control (2026-09-20) showed the method's
free and pinned faithfulness AGREE on genuine Top-K dictionaries (pin0 median
0.789, 24/28 seeds >= 0.5, no depth decay) and COLLAPSE on Gemma Scope 2's
uncapped JumpReLU (phi_pin 0.00-0.39 against free0 0.67-1.13, every circuit).
Top-K bounds what a re-encode of a degraded stream can do; 052's cap+clamp buys
stability but not the identity of the member values. These SAEs are the first
public set that is **Top-K AND covers every site kind**, so the full three-kind
(here four-kind) method can run on a public model without that failure.
Pythia-410M was considered and dropped: no examples/dashboards exist for either
its residual (Lawson) or MLP (EleutherAI) SAEs, and it has no attention SAEs.

## Verified, not assumed (2026-09-20)

Repos `jbloom/GPT2-Small-OAI-v5-{32k,128k}-{resid-post,resid-mid,mlp-out,
attn-out}-SAEs`, 12 layers each. Weights W_enc [768, 32768], W_dec [32768, 768],
b_enc, b_dec (float32). Config: `activation_fn_str topk, k=32`, d_in 768,
d_sae 32768, context_size 64, **prepend_bos False**, **normalize_activations
"layer_norm"**, **apply_b_dec_to_input True**.

Semantics transcribed from SAELens `sae.py`:
  x_n = (x - mean(x)) / (std(x) + 1e-5)    per token (torch unbiased std)
  code = topk_32(relu((x_n - b_dec) @ W_enc + b_enc))
  x_hat = (code @ W_dec + b_dec) * std + mean
=> an edit in MODEL space is the usual delta scaled by the token's std:
  x <- x + ((c_hat - c) @ W_dec) * std       (mean and b_dec cancel)

Hook points in plain HF GPT-2 (no SAELens/TransformerLens dependency, so the
main venv is untouched): attn-out = `block.attn` output[0]; resid-mid =
`block.ln_2` INPUT; mlp-out = `block.mlp` output; resid-post = `block`
output[0]. Within-layer causal order attn-out < resid-mid < mlp-out <
resid-post. NB capture hooks must return None — returning a bare tensor
replaces a tuple-valued module output and breaks the forward.

## Step 1 — reconstruction audit (`sae_recon.py`, `sae_recon_32k.json`)

16 x 64 wikitext tokens, clean CE 4.2704. FVU by site and layer:

  kind         L0      L3      L6      L9      L11
  attn-out     0.038   0.195   0.281   0.253   0.002
  resid-mid    0.036   0.001   0.003   0.010   0.157
  mlp-out      0.030   0.071   0.282   0.277   0.149
  resid-post   0.025   0.002   0.004   0.012   0.157

L0 is **exactly 32.0** everywhere (Top-K confirmed, so 052's cap is a no-op).
dCE +0.004 to +0.30 (largest at resid-post L11). **Identity check 0.0e+00 at
every site**: the delta form with c_hat = c reproduces the stream bitwise, so
the normalisation and hook transcription are exact. Residual reconstruction
beats Gemma Scope 2's (FVU 0.001-0.012 at L3-L9); attention and MLP are worse
mid-stack (0.19-0.28) than Gemma's.

## Step 2 — examples store (`neuronpedia.py`, `check_examples.py`)

These SAEs ship no examples.safetensors, but Neuronpedia hosts top-activating
contexts for all eight sets under the SAME feature indices, public API, no key:
`https://www.neuronpedia.org/api/feature/gpt2-small/{L}-{res_post|res_mid|att|
mlp}_{32k|128k}-oai/{idx}` — ~45 contexts per latent, each with tokens
(STRINGS), per-token values, maxValue, maxValueTokenIndex, plus frac_nonzero
and an auto-interp label. Display strings are mapped back to GPT-2 ids
(space -> "Ġ", newline -> "Ċ"), cached under ~/neuronpedia_cache.

Stored vs recomputed at the stored peak (the 052 check; there 0.007-0.012):

  seed                 n_ctx  frac_nonzero  stored  recomputed  median rel err
  resid-post 6 #100    16     0.0006        3.835   3.768       0.008
  resid-post 9 #2000   16     0.0029        4.626   4.569       0.012
  attn-out   6 #500    16     0.0760        2.677   2.635       0.013
  mlp-out    6 #1234   16     0.0099        8.778   8.529       0.006
  resid-mid  3 #777    16     0.0009        4.965   4.986       0.005

Treat the auto-interp labels as hints only (3 of 8 were wrong without
activation gating in the concept-circuits work).

## Step 3 — first panel (`fit_latent_seed.py`, `latent_circuits_lam1e-3.jsonl`,
`panel_lam1e-3.log`, 2026-09-20)

lam 1e-3, theta0 4 (open start — Top-K, so no cap/clamp and ERROR_MODE=current),
triple floor, 1 fitted null, positives from Neuronpedia, negatives mined:

  seed              n     free0  freeM_d  freeM_tk  freeN_d  freeN_tk  phi_sup  phi_cf  phi_pin  a=1
  resid-mid L3 #777   313  1.132   1.027    1.310    0.892    1.206     1.000   1.470   1.046   1.107
  resid-post L6 #100 3513  1.123   1.231    1.167    1.028    1.164     1.000   0.085   0.144   0.029
  attn-out  L6 #500  1121  1.091   1.327    1.026    0.997    1.027     1.000   0.001   0.056  -0.049
  mlp-out   L6 #1234 1376  0.892   0.772    0.663    0.767    0.745     1.000   0.611   0.801   0.148
  resid-post L9 #2000 3135 0.807   0.891    0.565    0.923    0.518     1.000   0.005   0.074   0.130

- **phi_sup (role-aware) is 1.000 on all five**, permuted and fitted nulls are
  ~0.000 everywhere, and free0 is in band (0.81-1.13) across FOUR site kinds and
  three depths. Members spread over all four kinds in every circuit.
- **Circuits are far too big** (313-3,513 vs Pythia's median 51): lam 1e-3 is too
  loose on this substrate. Sweep running at 3e-3 / 1e-2.
- **phi_pin is bimodal** (1.05, 0.14, 0.06, 0.80, 0.07) and tracks NEITHER size
  nor amplitude dependence: mlp-out L6 has 1,376 members and free0@a=1 = 0.148
  yet pins at 0.80, while resid-post L6 has free0@a=1 = 0.029 and pins at 0.14.
  A proposed rule "free0@a=1 predicts phi_pin" was RETRACTED after testing it on
  the 20 Gemma circuits: Pearson r = -0.229 (counterexamples both ways, e.g.
  res:9 tkfloor open a=1 0.581 / pin 0.064; res:12 tkfloor empty a=1 0.000 /
  pin 0.212).
- Working hypothesis: phi_pin needs a capped dictionary AND a compact circuit.
  Pythia has both (n 51, pin 0.789); GPT-2 is capped but under-pruned; Gemma is
  neither, and its compact circuits (90-307) still fail (0.00-0.37). The lam
  sweep tests this directly.
- PROTOCOL BUG FOUND AND FIXED: Neuronpedia returns contexts strongest-first, so
  splitting without shuffling put the WEAKEST contexts in held-out (a_pos 1.26
  against a stored peak of 3.8; freeM_dense read 6.9). Now shuffled before the
  split, as 052 does. NB **033 has the same ordering** (top-N by activation, then
  `pos[:48] / pos[48:]`), so its held-out scores are measured on its weakest
  contexts — a conservative bias, but it should be stated in the protocol.

## Step 4 — lambda sweep (`latent_circuits_lam{3e-3,1e-2}.jsonl`, `panel_lam*.log`)

  seed (frac_nonzero)      lam 1e-3            lam 3e-3           lam 1e-2
  resid-post L6 (0.0006)   3513 pin .28 cf .09 1041 pin .29 cf .26 330 pin .51 cf .51
  attn-out  L6 (0.076)     1121 pin .06        309 pin -.12        104 pin -.25 cf .02
  resid-post L9 (0.0029)   3135 pin .07        899 pin .05         299 pin .00 free0 .76
  (pin = phi_pin^a topk unless noted)

- **lam 1e-3 is the wrong calibration here**: resid-post L6 improves on EVERY
  axis at 1e-2 (330 members, free0 .98, freeM_d .99, sup 1.000, cf .51, pin .51).
  Size/regularisation does matter, contra the flat read at 3e-3.
- **attn-out L6 is pathological at every lambda** (pin negative, cf ~0,
  free0@a=1 -0.08, alpha median 1.38): it fires on **7.6% of tokens**, 15x above
  the protocol's frequency band (FREQ_LO 1e-4, FREQ_HI 5e-3), so 033's scan would
  never have selected it. Seed selection here was BY HAND from Neuronpedia
  indices, not from the band — a protocol difference from Pythia.
- **resid-post L9 is in band and still fails pinning** at every lambda, so
  out-of-band selection does not explain the GPT-2/Pythia gap on its own.
- Across all 8 fits: **phi_sup (role-aware) 0.956-1.000**, nulls ~0.000, free0
  in band, members spread over all four kinds. Faithfulness and necessity are
  not in doubt; phi_pin and phi_cf are.

## Step 5 — DEDUPE (a real bug) and what the members are (2026-09-20)

**Neuronpedia contexts are NOT distinct.** OpenWebText boilerplate repeats, so a
latent's ~45 contexts contain duplicates at 3-33%: resid-post 9 #2000 has 42
contexts / 28 distinct (one string 7x); resid-post 6 #100 has 41 / 29, with
"Could not subscribe, try again later" SEVEN times against one Yukon River
sentence. Splitting without dedupe puts the same string in train AND held-out.
`neuronpedia.contexts(dedupe=True)` now keeps the strongest of each group (052
does the same). **The lam 1e-3 / 3e-3 / 1e-2 panels above were fitted on
contaminated splits — do not quote their held-out numbers.**

Deduped panel, lam 1e-2 (`latent_circuits_dedup_lam1e-2.jsonl`,
`panel_dedup_lam1e-2.log`):

  seed              sites  n    free0  freeM_d  freeM_tk  phi_sup  phi_cf  phi_pin_0  phi_pin_tk
  resid-mid L3 #777   13    31   0.807  0.896    1.319     1.000    1.409   1.457      1.542
  resid-post L6 #100  27   314   1.305  1.757    1.470     1.000    0.521   0.462      0.620
  attn-out  L6 #500   24   113   1.026  0.798    0.146     1.000    0.025  -0.026     -0.166
  mlp-out   L6 #1234  26    78   1.006  0.863    0.651     0.993    0.477   0.795      0.493
  resid-post L9 #2000 39   308   1.024  1.152    0.409     1.000    0.007   0.000      0.000

- **First fully healthy GPT-2 circuits**: resid-mid L3 at **31 members** and
  mlp-out L6 at **78** — four kinds, faithful, necessary (phi_sup ~1.0), nulls
  0.000, AND they drive and pin. Pythia's median circuit is 51 members.
- Dedupe IMPROVED pinning where duplication was worst (resid-post L6:
  0.23/0.82/0.51 -> 0.46/1.40/0.62) but leaves only 21-29 train contexts, and
  the deeper seeds now overshoot (free0 1.31, freeM_d 1.76). Topping positives
  up by corpus mining is the fix.
- attn-out L6 is pathological at EVERY lambda and with clean contexts
  (pin -0.03, cf 0.03): it fires on 7.6% of tokens, 15x above 033's frequency
  band, so its scan would never select it.

**phi_pin TRACKS phi_cf** (both set members to their CLAIMED values — pin in
positive contexts, cf in silent ones). Measured over all 64 circuits we have:
GPT-2 r = **+0.936** (n=16); Pythia r = +0.335 (n=28, restricted range — nearly
every seed is high on both, medians cf 0.93 / pin 0.79); Gemma r = -0.131
(n=20, low everywhere — its own uncapped-dictionary problem). So the question is
not "why does pinning fail" but **"why do these circuits not DRIVE"** — the
closure-without-drive asymmetry already documented on TuringLLM and on 033's
attention seeds.

**Candidate mechanism — CHAIN LENGTH.** Within GPT-2 drive falls with the number
of edited upstream sites: 13 sites -> cf 1.41, 26 -> 0.48, 27 -> 0.52, 39 ->
0.01. Pythia is a 6-layer model (<= ~17 sites) and its median cf is 0.93. Every
intervening site re-encodes, washing the injected values out. Gemma breaks the
pattern (11-site L3 seeds still read 0.02-0.27) but has the uncapped dictionary
on top. TEST: fit GPT-2 seeds at L1-L2 (4-8 upstream sites) — if drive and
pinning return to Pythia levels, chain length is the mechanism.

## Member analysis (`inspect_circuit.py`)

Ranks members by attribution (grad x activation at the anchor over train
positives) and reads the top drivers/suppressors off Neuronpedia (label, firing
rate, peak context). On resid-post L6 #100 (330 members, contaminated split):
members spread over all four kinds (57/65/71/137), sit **4.7 layers below the
seed on average** reaching back to L0, alpha median 0.94 (p90 1.84), and split
**191 drivers / 139 suppressors** — over 40% inhibitory, which is why role-aware
phi_sup mattered. The strongest drivers were a coherent newsletter-boilerplate
group (subscribe / try again later / promotional offers) while the SEED's
auto-interp label says "size and length in geographical contexts" — the label
describes one of the latent's two firing modes and the circuit explained the
other, over-represented one. Labels are hints; print the firing rate and a
context beside them (3 of 8 were wrong in the concept-circuit work).

## Next

Port the latent-endpoint tri-amp fitter (052 `fit_latent_seed.py`) to this
substrate: site order above, std-scaled delta, `ERROR_MODE=current` (the Top-K
reference semantics — no cap, no clamp), triple floor, and the full
amplitude-aware eval family INCLUDING phi_pin.

**Contexts are MINED, not downloaded** — the same protocol as 033 Pythia
(`crosssae_topk.py` `scan`) and TuringLLM: 20k wikitext windows of 64 tokens,
seeds chosen per layer/kind from a frequency band, positives = the top N_POS
windows by activation with the anchor at the argmax position, negatives =
windows where the latent is EXACTLY zero, split 48 train / 16 held-out. That
machinery is model-agnostic apart from the loader and hook points, both already
written and verified here, so it ports nearly as-is and nothing depends on
Neuronpedia. Neuronpedia stays as (a) an independent check on mined contexts —
it already validated the loader end to end — and (b) auto-interp labels for
case studies, treated as hints only.
