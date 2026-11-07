# Arena C: the WCM adapter for Pythia-70m-deduped + Marks SAEs (DAN-194)

Written before the code (`wcm_marks.py`, `test_wcm_marks.py`). Production is frozen at `WCM_VERSION=v3`;
nothing under `src/`, `experiments/082-*`, `paper/` or `third_party/` is changed. Everything new is in this folder and
registers itself at import time. Each place where our setup is our choice rather than SFC's is tagged **[ours]**.

## 1. Substrate and sites

- **Model:** HF `EleutherAI/pythia-70m-deduped` (WSL HF cache), through our HF stack (`model.hf_adapters.HFSiteModel`,
  plain PyTorch forward hooks; no nnsight, no TransformerLens), float32, eager attention. No BOS (the Pythia tokenizer
  adds none; SFC prepends none).
- **Dictionaries:** the 19 Marks et al. "10_32768" SAEs in
  `experiments/030-cross-sae/dictionaries/pythia-70m-deduped/<site>/10_32768/ae.pt` (the folder SFC's
  `dictionaries/` symlink points at). ReLU, 512 → 32,768, untied, a pre-encoder bias.
- **Family:** `pythia-70m-deduped-marks`, registered in `model.families.FAMILIES` and `hf_adapters._SITE_MODELS` by
  `wcm_marks.register()`. It is a different model and dictionary set from our `pythia-70m` family (non-deduped,
  EleutherAI Top-K).

The sites mirror SFC's submodules exactly (`dictionary_loading_utils.py:35-86`, spec §2.2):

| SFC node | nnsight submodule | our site (layer, kind) | our hook |
|---|---|---|---|
| `embed` | `gpt_neox.embed_in`, output | (0, `embed`) | forward hook on `embed_in` |
| `attn_i` | `gpt_neox.layers[i].attention`, `output[0]` | (i, `attn`) | forward hook on `layers[i].attention`, element 0 |
| `mlp_i` | `gpt_neox.layers[i].mlp`, output | (i, `mlp`) | forward hook on `layers[i].mlp` |
| `resid_i` | `gpt_neox.layers[i]`, `output[0]` | (i, `resid`) | forward hook on `layers[i]`, element 0 |

- **Parallel residual.** Pythia computes `out = x + attn(ln1(x)) + mlp(ln2(x))`. An attn edit enters the layer's sum,
  never the same-layer MLP's input. Within-layer causal order (`same_layer_upstream`): embed → {attn, mlp} → resid,
  with attn and mlp not reading each other.
- **The grid has 24 slots, 19 real.** Our engine indexes sites as (layer, kind) over every layer, so with kinds
  (`embed`, `attn`, `mlp`, `resid`) the slots (1..5, `embed`) exist but have no dictionary. **[ours: an
  implementation device, no effect on any number]** They hold an inert `NullSAE`: its code is always 0, its decode is
  0, so its error is the whole stream and no edit can enter there. The hook protocol still calls them, with the
  layer's input, so per-layer activation tuples stay complete; the result is discarded. The task fits mask the 18
  non-embed sites (§4). On the production latent path (0, embed) is a NullSAE too by default (embedding out of
  scope); `upstream_sites` lists the null slots, which cannot gain members (zero data gradient, the L1 price closes
  them) but count towards its `n_sites`.

## 2. Encode and decode

As `dictionary_learning.AutoEncoder` (`dictionary.py:64-68`), with `from_pretrained(normalize_decoder=True)`:

    f    = ReLU(W_E (x − b_D) + b_E)          = ReLU(x W_E^T + b_eff),  b_eff = b_E − W_E b_D
    x̂    = W_D f + b_D
    ε(x) = x − x̂

- **Decoder normalisation:** as `normalize_decoder()`: if the columns of W_D are not unit norm (`torch.allclose`),
  W_D /= n, W_E *= n, b_E *= n. Measured: every column norm is within 7e-7 of 1, so this is a no-op here, and the
  check is kept for exactness.
- **Mapping into our `sae.topk_sae.SAE`:** `encoder.weight` = W_E [32768, 512], `encoder.bias` = b_E,
  `decoder.weight` = W_D [512, 32768], `decoder_bias` = b_D. Our seed tap `x @ w + b` with (w, b) = (W_E row, b_eff) is
  the latent's own pre-activation, as for every other family.
- **No k.** A ReLU SAE has no Top-K. Our bank contract returns `(acts, idx)` of width `bank.k`; here
  **k = d_sae = 32,768 [ours]**: `acts` is the whole ReLU code and `idx` an expanded arange, so no latent can be
  truncated. Consequences, each stated with the results:
  - **The sparsity-preserving fill** (v2/v3's A and C training terms and the scorer's `A_sp` / `C_sp`) has budget
    K − #active members ≥ 32,768 − #members, so every non-member takes its mean: it reduces exactly to the plain mean
    fill, which is also SFC's ablation of a feature (`ablation.py:55-57`).
  - **Rank-keep** on latent targets (`rank_mode="keep"`) holds the target's clean rank among all 32,768
    pre-activations. Its Top-K motivation does not apply to a ReLU dictionary; it runs unchanged and is stated. (The
    logit objective has no rank-keep: section 4.)
  - **The data pass's PMI** (`top_coactivation`, pass 2) assumes exactly k active latents per token, so it is wrong
    at k = d_sae; `data_pass_marks.py` runs pass 1 + neg_ctx only, which is all the protocol contexts need.

## 3. The intervention

Every masked site applies, exactly as on every other family:

    x ← x + (ĉ − c) W_D^T            (decode of the code difference, no bias)

where c is the clean code and ĉ the circuit's code (members at α · c, non-members at the term's fill: 0, the A mean
or the C mean). The SAE error ε = x − x̂ is never touched, so a circuit that keeps everything at α = 1 reproduces the
model exactly: ĉ − c = 0 and the stream is unchanged. Tested (`test_wcm_marks.py`: splice exactness).

**SFC's ablation differs [their method, not a change of ours]:** SFC writes `decode(f') + ε'`, where the error ε is an
ablatable node (mean-ablated when not in the circuit). WCM keeps every error at its clean value and never counts it.

## 4. The SVA tasks (objective a: the logit objective)

**Files:** SFC's own `third_party/feature-circuits/data/{simple,within_rc,rc,nounpp}_{train,test}.json` (@7fbd82b,
read-only), byte-identical to `data/external/sfc_feature_circuits/` that 081 used.

**Selection = SFC's `load_examples`, re-implemented line for line** (`data_loading_utils.py:8-67`) on the Pythia
tokenizer: `random.Random(12).shuffle` of the file's lines; keep a record only if both answers are one token
(after stripping a BOS), the clean and patch prefixes have equal length, and prefix+answer tokenises as
tok(prefix)+tok(answer); with `use_min_length_only`, restart on a new shortest length and skip longer ones; stop at n.
A test checks that our selection equals SFC's own function on the same file (run in `~/venvs/sfc107`).

| data mode | training (the fit's pool, A mean, C mean) | held-out | label |
|---|---|---|---|
| `sfc` (default) | SFC's discovery set: the first 100 min-length records of `<task>_train.json` | SFC's evaluation set: the first 40 min-length records of `<task>_test.json` | SFC's protocol |
| `sfc_test_all` | as `sfc` | every min-length record of `<task>_test.json` | **[ours]** larger held-out |
| `dedup` | as `sfc` | the min-length test records whose `clean_prefix` is not a training prefix (any record of `<task>_train.json`) | **[ours]**, decision 8; can be empty |

- **Leak (081, and here under Pythia's tokenizer):** SFC's test files are not disjoint from their train files; every
  simple / within_rc / rc test prefix occurs in the train file. `make_tasks` prints, per task, how many of the 40
  evaluation examples equal a discovery example and how many have a prefix in the train file.
- **Answers:** good = `clean_answer`, bad = `patch_answer`, both single tokens, read at the last prefix token.
- **C (contrast) prompts:** each training item's `patch_prefix` (081's "original" contrast). A = the mean over the
  training clean prompts. Both are means over real tokens, computed once per task and shared by the fit and the
  scorer (protocol v2/v3 `eval_ref` by construction, as 081).
- **Metric:** our objective is `logit_diff` = log p(good) − log p(bad) = logit[clean_answer] − logit[patch_answer] at
  the last position. That is SFC's evaluation metric `m_eval` exactly (`ablation.py:176-180`); SFC's discovery metric
  is its negative (spec §3.2). Tested against SFC's code.

**The fit** is 081's engine path (`Engine.fit`, `run_learned_mask(objective="logit")`), unchanged:
Z + C + A (triple floor, γ_C = γ_A = 0.25), free coefficients, AdamW, `logit_norm="gap"`, 400 steps × batch 4,
`binarize`/keep threshold from the production config, the 100 training items cycled without replacement within an
epoch.

**Behavioural WCM = Z + A + C + R, no rank-keep (Daniel).** In numbers (`wcm_marks.V3_LOGIT`, = 062
`driver.WCM_VERSIONS["v3"]` without `WCM_RANK`): `train_fill="sparse"`, `sparse_grad="ste"`, `resample_weight=1.0`,
`resample_k=1`, `rank_weight=0`. R's donor rows are the training items' `patch_prefix` prompts (the C mean's pool);
they replace the clean rows position by position, so the adapter asserts they have the clean prompts' token length
(true on SFC's min-length sets).

*Why no rank-keep.* A behavioural rank-keep (the answer token holds its clean rank among the vocabulary, weight 0.1) was
added and then removed. It made fits collapse: `simple` λ 1e-5 pooled gave Z 0.00 with it,
0.89 without it, and 1.00 without both it and R (`ablate_terms.py`, `out/ablate_terms/rows.jsonl`). Across the first
24 pooled fits, every fit with a large final rank term (5–50) had low Z, while the same members at α = 1 scored
0.6–1.0. The answer's absolute vocabulary rank is arbitrary here (the full model ranks the right verb first on only 10%
of `simple` prompts), and the logit difference already scores the answer against its rival, the wrong verb. Those 24
fits (`out/wcm_tasks/`) are superseded and are not reported.

**Maskable scope (decided): the 18 attn / mlp / resid sites, embedding excluded** (`--sites noembed`, the
default; latent targets: their upstream sites minus embed, via a NullSAE at (0, embed), `MARKS_EMBED=1` to undo). Layers
0-1 stay in, although SFC's evaluator leaves them unablated (`start_layer=2`); `--sites ge2` (12 sites) is SFC's
evaluated scope, the scope-matched row. `--sites all` (19) is kept for diagnostics only.

**A / C means, reported both ways (Daniel):** `--means pooled` = the protocol's [d_sae] means over every
real prompt token; `--means per_position` **[ours: an adaptation for fixed-length templates, matching SFC's
per-position mean ablation]** = [T, d_sae] means, A over the training clean prompts and C over the training patch
prompts, position by position (one prompt length T per task, asserted). The same tensors go to the fit and the scorer.

**Size, two columns (107 rule 6):** distinct (site, latent) members; and SFC's native unit, the position-expanded count
a position-free member costs under SFC's evaluator (members × prompt length, at layers ≥ 2 and over all sites). Error
nodes: 0 (WCM never selects them; under SFC's evaluator WCM is scored with errors kept, `random`'s convention, and
with errors removed, `features_wo_errs`; that scoring is DAN-196's).

## 5. Latent targets (objective b)

- **Root:** a Marks SAE latent (site, index); the normal production latent objective (`objective="pos"`, Z + C + A,
  R, rank-keep), i.e. 062's `driver.py` with `FAMILY=pythia-70m-deduped-marks`, `WCM_VERSION=v3`. No change to the
  driver: `wcm_marks.register(patch_bank_dispatch=True)` adds the family and lets `SAEBank(...)` build the Marks bank
  (a runtime alias of `sae.public_bank.PublicSAEBank` to a subclass that only differs for this family) **[ours:
  runtime registration, no file change]**.
- **Contexts:** protocol v1/v3 contexts come from `circuit.protocol_contexts.build_record`, which reads the 069 data
  pass's stores (top / mid / neg_ctx / seq_repr / latent_stats). They do not exist for this family, so latent targets
  need a data pass first. The pythia-70m smoke corpus (`data_public/pythia-70m`, 4,096 × 64 wikitext tokens) uses
  the same tokenizer as pythia-70m-deduped and can be reused; a real run needs the Pile (the SAEs' training data).
  `data_pass_marks.py` (069's `run_data_pass.py`, unchanged, with the family added to its `--family` choices) runs
  pass 1 + neg_ctx; `driver_marks.py` runs 062's driver unchanged on the family. Both ran end to end (§7).
- **Contrast "silent":** with k = d_sae, "outside the Top-K" means activation exactly 0, the ReLU meaning of silent.

## 6. Every difference from SFC, labelled

| # | what | SFC | WCM side here | label |
|---|---|---|---|---|
| 1 | error terms | nodes; ablatable; counted | kept at clean values; never counted | method difference (stated) |
| 2 | positions | node = (site, position, latent) | member = (site, latent), all positions | method difference; both counts reported |
| 3 | k | none (ReLU) | k = d_sae (whole code) | ours, exact |
| 4 | sparse fill | — | = plain mean fill (consequence of 3) | ours, stated |
| 5 | null embed slots (1..5, embed) | — | inert placeholders | ours, no numeric effect |
| 6 | behavioural WCM | — | Z + A + C + R (donors = training patch prompts), no rank-keep | Daniel's definition, stated |
| 7 | scope | evaluator ablates layers ≥ 2 only, never the embedding (`include_embed=False`) | 18 sites, embed excluded (default); `ge2` = 12 = their scope | ours: the embedding is out because SFC never ablates it and zeroing it explodes the Z empty circuit (§7: 7 of 8 resid targets at layer ≥ 2 vacuous with it, e.g. median e0 1e10 vs clean 0.98; without it median e0 −2.1 to 1.3, though a tail of anchors still explodes, §7) |
| 8 | our ablations | mean over the 40 eval patch prompts, per position | Z; A = training clean mean; C = training patch mean; `pooled` (protocol) and `per_position` rows | pooled = the shared protocol; per_position = ours (fixed-length templates; SFC's convention) |
| 9 | data | 100 train / 40 test, min length | the same sets (`sfc`); larger / dedup held-out | `sfc` = theirs; others ours |
| 10 | model stack | nnsight 0.3.7 | HF forward hooks, torch 2.8, transformers 5.1 | environment; tested equal |
| 11 | precision | float32 | float32, no TF32 | same |

## 7. Measured (RTX 5070 Ti; tests on the CPU)

**Tests** (`test_wcm_marks.py`, 4 pass):
- parity with `dictionary_learning.AutoEncoder`, all 19 dictionaries: max |Δf| 1.5e-5 (real activations), 3.6e-5
  (random); max |Δx̂| 1.5e-5 real / 2.4e-4 random; `normalize_decoder` is a no-op on every file;
- splice: unpatched and open-gate WCM patcher over all 19 sites give max |Δlogit| = 0 (bitwise) vs the raw HF model;
  SFC's own `decode(f) + ε` splice at all 19 sites gives 4.3e-3 on logits of magnitude 1.6e3 (float32 cancellation);
- metric: our logit difference on 8 simple + 8 rc evaluation examples equals SFC's own code (`sfc_metric_ref.py`, in
  `~/venvs/sfc107`, nnsight 0.3.7) to 0.0; our `load_examples` selects the same 100 + 40 prefixes in the same order;
- tiny v3 fit (6 steps, CPU): runs, writes a circuit; R ran (400 → 6 draws, 100 donors), rank-keep ran (keep, 0.1).

**Leak (F8 under Pythia's tokenizer)** of the 40 evaluation examples: identical to a discovery example simple 17,
within_rc 2, rc 0, nounpp 0; prefix in the discovery set 40 / 5 / 0 / 1; prefix in the train file 40 / 40 / 40 / 18.
Min lengths: simple 2, within_rc 5, rc 6, nounpp 5 (train and test agree). The `dedup` held-out (test prefixes not in
the 100 discovery prompts) is empty for simple.

**SUPERSEDED (kept for the record): the first smoke and Finding 1 below were scored before the scorer
fix** (the 1-D sparsity-preserving fill at k = d_sae could overwrite KEPT latents with their means), with the
embedding in scope. Their A / C numbers are wrong; the rerun is "Smoke 2" further down.

**Smoke 1, simple, `sfc` data, 400 steps, v3, 19 sites** (held-out = SFC's 40; fit ≈ 73 s, peak 3.3 GB; score 0.6 s):

| λ | latents (layers ≥ 2) | SFC-unit nodes ≥ 2 | Z | A | C | necessity |
|---|---|---|---|---|---|---|
| 1e-3 | 348 (159) | 318 | −0.10 | −0.01 | 0.01 | 0.94 |
| 1e-4 | 1,925 (1,495) | 2,990 | 0.25 | 0.00 | 0.00 | 1.02 |
| 1e-5 | 2,386 (1,458) | 2,916 | 1.01 | 0.11 | 0.11 | 1.01 |

**Finding 1 (superseded: its A / C scores carry the scorer bug; the position-0 norms below still hold):** Pythia's first token carries
massive activations (resid_2 norm 123 at position 0 vs 13 at position 1; mlp_2 110 vs 6). A mean over the real tokens
of 2-token prompts carries them into every position: the decoded A mean sits 58 units from the anchor's activation at
resid_2 (5.7 with a per-position mean, as SFC uses); rc (6 tokens) 22 vs 7.4. A / C stay ≈ 0 at every size tested
(135-8,668 latents, v2 and v3) while Z reaches 0.86-1.0. Leaving position 0 out of the means does not help
(A −0.04 / 0.00): position 0 then gets the other positions' mean. Per-position means would need a src change (the
floors are [d_sae] per site).

**Finding 2: the zero-ablated empty circuit explodes on 64-token contexts.** Latent target 3.resid.27697 (smoke
corpus, 64 strongest + 64 mid, 64 contrast): clean pre-activation max 11.2; with every upstream code at 0 (errors kept)
6.2e5 (median over positions 5.9e5); position 0 kept clean 7.3e5; only the resid sites zeroed 86. The scorer's e0 is
5.2e5 and the dense A empty circuit 7.2e7, above a_pos (1.9), so the production scorer marks the target vacuous
(faithfulness None). One target, 40-step fit (mechanics only): to be measured on a target panel before the overnight
latent run.

**Finding 2, diagnosed** (`diag_zero_empty.py`, forward passes only; `out/diag_z/results.jsonl`). Panel: 36 latents,
2 per (layer 0-5) × {attn, mlp, resid}, seeded, from those with full top + mid stores; read at the held-out strongest
anchors as the scorer reads. Medians over anchors (pre-activations):

| target | clean | e0 | embed only | attn only | mlp only | resid only | pos 0 clean | no embed |
|---|---|---|---|---|---|---|---|---|
| 2.resid.5586 | 0.56 | 957 | −1.24 | 0.21 | −1.07 | −0.63 | 941 | −1.32 |
| 2.resid.8268 | 0.50 | −41.7 | −0.30 | −0.68 | −0.36 | −0.41 | −42.3 | −0.53 |
| 3.resid.18573 | 0.94 | 3.6e5 | −1.11 | −0.27 | −1.15 | −1.52 | 2.2e5 | −1.03 |
| 3.resid.14336 | 1.27 | 2.8e5 | −0.66 | 0.33 | −0.80 | −0.69 | 1.1e5 | −0.73 |
| 4.resid.22438 | 6.37 | 1.1e8 | −1.62 | 3.57 | −1.56 | −2.18 | 9.5e7 | −2.14 |
| 4.resid.25891 | 0.51 | 4.4e7 | −0.80 | 0.17 | −1.13 | −1.50 | 4.4e7 | −1.91 |
| 5.resid.17608 | 2.71 | 9.0e9 | −1.50 | −1.06 | −1.36 | −1.42 | 8.7e9 | 1.31 |
| 5.resid.18615 | 0.98 | 1.0e10 | −0.99 | −1.21 | −0.83 | −0.44 | 9.4e9 | −0.50 |
| the other 28 (all attn / mlp, layer 0-1 resid) | 0.12-1.94 | −5.9 … 1.98 | | | | | ≈ e0 | normal |

- **Scope:** every resid target at layer ≥ 2 (8 of 8) explodes; no attn / mlp target and no layer 0-1 resid target
  does. **Vacuous under §12.4 (`vacuous_v2`): 12 of 36** — the 7 positive explosions, plus 5 ordinary cases where the
  empty circuit already reaches the clean value (1.attn.2190, 1.resid.4751, 3.mlp.1055, 4.attn.4010, 5.mlp.11057).
- **Cause: the resid SAEs amplify an off-distribution stream, and the Z edit keeps their error.** No single kind
  explodes; zeroing the embedding *together with* the rest does, and leaving the embed site out removes it (position 0
  kept clean does not). With the embed code zeroed the stream is ε_embed + b_D (norm 0.17 vs 0.70). By resid_1 the
  input (norm 37) makes the resid_1 SAE fire 31,220 of 32,768 latents (clean L0 ≈ 50-100) with ‖W_D c‖ = 4.7e3, so the
  edit x − W_D c leaves a stream 125× its input. Each later resid site multiplies it again: resid_2 ×240, resid_3
  ×183, resid_4 ×347 (5.resid.17608: 37 → 4.5e3 → 1.1e6 → 2.1e8 → 7.2e10). Without the embed site the same sites stay
  at norm 4-90.
- **Not LayerNorm:** the LN input std does not collapse; it grows with the stream (anchor median, clean → e0:
  layer 1 0.34 → 1.8, layer 2 0.40 → 2e2, layer 3 0.53 → 5e4, layer 4 0.6 → 1e7, layer 5 0.7 → 3.5e9). Only layer 0's
  falls (0.03 → 0.0077), because the embedding is zeroed.

**Recheck with the embedding out of scope** (`KEYS=… RESULTS=results_noembed.jsonl diag_zero_empty.py`, (0, embed) a
NullSAE): the median e0 is normal on the 4 exploding resid targets (3.resid.18573 −1.03, 4.resid.22438 −2.14,
5.resid.17608 1.31, 5.resid.18615 −0.50), **but 3 of 4 are still vacuous**, because the scorer's read is a mean over
the 14-16 anchors and a few anchors still explode: 4.resid.22438 1 of 16 (1.5e7), 5.resid.17608 3 of 16 (3.3e4,
1.8e8, 4.4e8), 5.resid.18615 2 of 16 above 1e5 (4.8e5, 4.2e8). Same mechanism, started without the embedding: at some
position the resid_0 / resid_1 SAE input leaves the distribution once attn_0 / mlp_0 are zeroed (resid_0 L0 up to
8,970, resid_1 up to 32,342 of 32,768), its x − W_D c amplifies, attention spreads it, and resid_2-4 saturate (L0
≈ 32,500; |x| 7e3 → 2e6 → 1e9). Only 3.resid.18573 is no longer vacuous. Excluding the embedding removes the typical
blow-up but not the tail; the latent arena stays blocked on this (a decision: e.g. a robust read, or Z without the
resid sites).

**Smoke 2 (after the scorer fix), simple and rc, `sfc` data, scope noembed (18 sites), v3, 400 steps.** Held-out =
SFC's 40. Fit 61-68 s (simple ≈ rc), score 0.5-0.7 s. "rank" / "R" = the final rank-keep and resample terms.

| task | means | λ | latents (≥ 2) | Z | A | C | nec | rank | R |
|---|---|---|---|---|---|---|---|---|---|
| simple | pooled | 1e-4 | 1,167 (846) | 0.72 | 0.76 | 0.77 | 1.00 | 5.77 | 0.16 |
| simple | pooled | 1e-3 | 523 (323) | 0.77 | 0.86 | 0.87 | 1.00 | 0.03 | 1.23 |
| simple | per_position | 1e-4 | 931 (588) | 0.81 | 0.78 | 0.80 | 0.99 | 0.00 | 0.45 |
| simple | per_position | 1e-3 | 351 (192) | 0.33 | 0.56 | 0.58 | 1.01 | 0.19 | 1.85 |
| rc | pooled | 1e-4 | 4,082 (2,619) | 0.60 | 0.30 | 0.30 | 0.98 | 10.9 | 1.11 |
| rc | pooled | 1e-3 | 2,317 (1,615) | 0.16 | 0.03 | 0.04 | 1.07 | 1.65 | 0.17 |
| rc | per_position | 1e-4 | 3,797 (2,610) | 0.45 | 0.57 | 0.60 | 1.06 | 8.56 | 0.18 |
| rc | per_position | 1e-3 | 2,726 (2,013) | 0.05 | −0.04 | −0.05 | 1.09 | 3.37 | 0.09 |

With the fix, pooled A / C on simple are as good as Z (0.76-0.87), so most of Finding 1 was the scorer bug. On rc
(6 tokens), per-position A / C beat pooled at λ 1e-4 (0.57 / 0.60 vs 0.30 / 0.30). Nothing passes at these two λ; the
curve needs the smaller ones (3e-6 to 3e-5).

**Donor test for task circuits** (`donor_tasks.py`, tests `test_donor_tasks.py`: self-donor empty circuit = full
model to 0.0 on 40 bases; full circuit recovers 1.0 under A / C / R; draws deterministic). Pools A = training clean
prompts, C = held-out patch prompts, R [ours] = 256 smoke-corpus windows truncated to T; 8 draws; 40 bases. First
scores (behavioural v3 = Z + A + C + R, pooled means, noembed):

| circuit | latents | Z / A / C / nec | prod pass | recovery fit A / C / R | a1 A / C / R | resample_pass |
|---|---|---|---|---|---|---|
| simple 1e-5, **with R** | 2,933 | 0.89 / 0.89 / 0.89 / 1.00 | yes | 0.86 / 0.87 / 0.21 | 0.97 / 0.98 / 0.11 | **yes** |
| simple 1e-5, **no R** | 1,569 | 1.00 / 1.00 / 1.00 / 1.00 | yes | 0.77 / 0.68 / 0.22 | 0.72 / 0.67 / 0.11 | **no** (C) |
| simple 3e-6 | 4,897 | 0.99 / 0.99 / 0.99 / 1.02 | yes | 0.98 / 0.98 / 0.14 | 0.98 / 0.98 / 0.03 | yes |
| simple 3e-5 | 1,471 | 0.98 / 0.94 / 0.94 / 1.00 | yes | 0.95 / 0.95 / 0.00 | 0.82 / 0.83 / −0.00 | yes |
| simple 1e-4 | 1,199 | 0.94 / 0.98 / 0.99 / 1.01 | yes | 0.99 / 0.98 / 0.20 | 0.87 / 0.89 / 0.04 | yes |
| simple 3e-4 | 576 | 0.41 / 0.34 / 0.34 / 1.00 | no | 0.23 / 0.23 / 0.31 | 0.56 / 0.60 / 0.05 | no |
| simple 1e-3 | 319 | 0.31 / 0.01 / 0.01 / 1.00 | no | −0.02 / 0.01 / 0.08 | 0.00 / 0.03 / 0.10 | no |
| nounpp 3e-6 | 17,033 | 1.00 / 0.99 / 0.99 / 0.82 | no (nec) | 1.00 / 1.01 / −0.08 | 1.01 / 1.01 / −0.11 | no |
| nounpp 1e-5 | 10,707 | 0.90 / 0.85 / 0.85 / 0.89 | no (nec) | 0.94 / 0.93 / 0.10 | 0.98 / 0.98 / −0.09 | no |
| nounpp 3e-5 | 10,532 | 0.83 / 0.82 / 0.82 / 0.88 | no (nec) | 0.74 / 0.78 / −0.14 | 0.86 / 0.86 / 0.08 | no |

R vs no R (simple 1e-5, same fit otherwise; `ablate_terms.py --save out/ablate_terms_circ`, variants no_rank = Z+A+C+R
and v2 = Z+A+C): R's circuit is 1.9× larger and scores lower on the mean fills, but its donor recovery is higher
(A 0.86 vs 0.77, C 0.87 vs 0.68), so only the R circuit passes the resample rule. R-pool recovery is ≤ 0.31
everywhere: 2-5-token windows cut from running text are far from the templates (the R pool is ours). Empty leaks: simple
A 0.02 / C −0.01 / R −0.08; nounpp −0.32 / −0.44 / −0.15. Scoring cost: 7-22 s per task group (shared empty draws).

## 8. Fit instability: late blow-ups under the anneal

**Symptom.** At 400 steps the behavioural fits are noisy. On seed refits (`seed_variance.py`, 3 seeds) pass / fail
and size flip up to 2-3× at λ ≥ 3e-5, and the size curves are non-monotone (intermediate λ collapse while a larger λ
passes). Running 3× longer at the same settings does not cure it. simple λ 1e-5, per-position, 1,200 steps: 1,527
latents with Z 0.008, where 400 steps passed. The refit is deterministic (`diag_anneal.py` reproduces 1,527 / 0.008).

**Cause (`diag_anneal.py`, full training curves in `out/diag_anneal/`).** That fit is near-perfect for 850 steps (zero
term ≤ 0.0015 at ~1,900 latents). At step ~900 (T = 0.106) the zero term jumps to 12.2, and it never recovers (0.46 at
the end). The engine's anneal (`binarize="anneal"`) uses m = sigmoid(θ / T), T 1 → 0.05 geometric, reaching 0.05 at the
last step under `anneal_reach_frac` 1.0. Adam moves θ by ~lr per step whatever the gradient's scale, so at a constant
lr the gates move up to 1 / T = 20× further per step late in training, and single steps flip boundary gates wholesale.
50 of the 128 failing fits on record (zacr, seed_variance, size_explore, s1200) show the signature: the zero term's last
value is > 10× its mid-run value. Most of the rest fail already at mid-run (λ genuinely too large).

| simple, λ 1e-5, per-position, 1,200 steps | latents | Z / A / C / nec | zero term: worst spike after step 300 | end |
|---|---|---|---|---|
| run's schedule (constant lr, reach 1.0) | 1,527 | 0.008 / 0.001 / −0.001 / 0.995 | 12.2 (step 900) | 0.46 |
| anneal reaches T = 0.05 at 60%, then holds | 1,694 | 0.905 / 0.870 / 0.882 / 1.002 | 3.4 (step 595) | 0.011 |
| amplitude price `amp_l1` 1e-4 (constant lr; the collapsed fit's max α was 19.6) | 1,485 | 0.817 / 0.859 / 0.846 / 1.015 | 2.0 (step 830) | 0.012 (1.48 within the last 25 steps) |
| **lr cosine 1 → 0.05** (tracks T, so lr / T ≈ constant) | 2,241 | **1.011 / 1.013 / 1.017 / 1.011** | 0.13 (step 425) | 1e-5 |

Under cosine the loss falls smoothly and Z with every amplitude reset to 1 (`free0_a1`) is 0.99. The constant-lr
schedule's "hold at the final temperature" (reach 0.6) helps by giving the fit time to recover, but it still spikes.

**λ path (`warm_path.py`, constant lr, 400 steps per λ, simple per-position).** Each λ starts from the previous λ's
final gates and amplitudes (learned_mask `warm_state` / `return_state`). Sizes become monotone: 4,962 → 2,905 → 2,170 →
1,617 → 869 → 509 → 371 → 154 (λ 3e-6 … 1e-2). Smallest passing: 371 (λ 3e-3), against 895 cold. One λ (1e-3) still
fails at Z 0.68, presumably the same late spike.

**Cosine is not enough; geometric is.** On the cosine grid (`out/wcm_tasks_cos`, per-position complete), simple passes
at every λ up to 1e-2 (smallest 208 latents, size monotone), and its 3-seed refits all pass through λ 1e-3 (constant
lr: 0 of 4 at 1e-3). But rc still collapses at mid λ (3e-4: 5,626 latents, Z 0.09). Its full curve blows up EARLY
(step ~100, zero term 5.2): cosine decays slower than T at first, so lr / T peaks at ~2.5× mid-run. A new
`lr_schedule="geometric"` (lr × 0.05^progress, the anneal's own shape: lr / T constant; learned_mask, default off,
tested) trains smoothly:

| rc, λ 3e-4, per-position, 400 steps | latents | Z / A / C / nec | zero term: worst spike |
|---|---|---|---|
| constant (the run's) | 3,943 | 0.117 / 0.118 / 0.093 / 1.011 | 3.2 (step ~200) |
| cosine | 5,626 | 0.085 / 0.052 / 0.027 / 1.073 | 5.2 (step ~100) |
| **geometric** | 3,754 | **1.020 / 1.052 / 1.058 / 0.983** | 0.04 |

**Warm start, revisited.** Restarting the anneal at T = 1 for each warm fit re-softens the inherited gates (zero term
1e-4 at the end of one λ, 0.03-0.16 at the start of the next; cosine warm path collapsed at λ 1e-3 and 3e-3). Holding
T = 0.05 (`anneal_t0`, learned_mask, default off, tested) at lr 0.05× fixes that but prunes too slowly in 400 steps
(simple λ 3e-4: 2,414 latents vs 771 cold). Cold geometric fits are the candidate; the warm path is parked.

**Geometric grid** (`out/wcm_tasks_geo`, `run_geometric.sh`, 400 steps, the λ grid 3e-6 … 3e-2). Smallest circuit
passing our rule (Z / A / C in [0.75, 1.5], necessity ≥ 0.9), per-position / pooled. "contiguous" = every larger λ
passes too, i.e. no collapse inside the curve:

| task | constant lr (the run) | cosine | **geometric** |
|---|---|---|---|
| simple | 895 / 296 (pooled: 576, 319 fail in between) | 208 / – | **344 / 256**, contiguous |
| within_rc | 1,365 / 1,540 (pooled: 1e-4 fails) | 1,734 / – | **879 / 301**, contiguous |
| rc | 12,441 / 8,658 (4 of 7 smaller λ collapse) | 704 / – (3e-4, 1e-3 collapse) | **1,249 / 2,510**, contiguous |
| nounpp | 16,299 / never | never / – | never / 7,286 |

Size falls monotonically with λ on every geometric curve. Geometric has a smaller lr integral than constant (≈ 0.32×),
so a given λ gives a larger circuit; the curves, not single λ, are the comparison. **nounpp fails on necessity
alone**: Z / A / C stay 0.76-0.99 from 29k down to 347 latents, but necessity is 0.67-0.90 (one fit reaches 0.90).
Removing the circuit does not remove the agreement: other paths carry it. A task property, not the instability.

**Seed refits, geometric** (run seed + donor seeds 1-3 = 4 fits per point; `out/seed_variance_geometric`). Passes out
of 4, and max / min size:

| λ | simple pp | simple pooled | rc pp | rc pooled | rc pp, constant lr (for contrast) |
|---|---|---|---|---|---|
| 1e-5 | 4/4, 1.15 | 4/4, 1.11 | 4/4, 1.06 | 4/4, 1.20 | 4/4, 1.73 |
| 3e-5 | 4/4, 1.13 | 4/4, 1.07 | 4/4, 1.09 | 4/4, 1.11 | **0/4**, 1.95 |
| 1e-4 | 4/4, 1.08 | 4/4, 1.07 | 4/4, 1.23 | 4/4, 1.32 | 1/4, 2.05 |
| 3e-4 | 4/4, 1.08 | 4/4, 1.10 | 4/4, 2.63 | 4/4, 1.33 | **0/4**, 3.40 |
| 1e-3 | 4/4, 1.08 | 4/4, 1.08 | 2/4, 1.78 | 4/4, 1.27 | 1/4, 1.62 |
| 3e-3 | 3/4, 1.86 | 3/4, 1.10 | 1/4, 1.33 | 2/4, 1.62 | – |
| 1e-2 | 1/4, 1.39 | 2/4, 1.19 | 1/4, 1.51 | 0/4, 1.86 | – |

Passing is now reliable down to the size cliff; the remaining disagreement sits at the cliff itself, where
faithfulness is near the 0.75 edge. rc per-position size at 3e-4 still varies (2,218-5,839) although all four pass.

**Latent pipeline:** `data_pass_marks.py` on the 4,096-sequence smoke corpus: pass 1 54 s (batch 64, peak 6.2 GB),
neg_ctx 16 s, records complete (3,572 of 32,768 resid_3 latents have full top + mid stores). `driver_marks.py`,
`WCM_VERSION=v3`, one target: context record built, a 40-step fit 39 s (≈ 1 s / step, so ≈ 6.5 min at 400 steps), eval
7.5 s.
