# 119: strict resample on held-out donors (#3) and TuringLLM SAE ΔCE (#18)

DAN-218, group C. Agent gpu-218c, 2026-11-15. Nothing in `paper/` was edited: the proposed changes are listed below.

Downloads: none. WikiText (`Salesforce/wikitext`, wikitext-103-raw-v1 train shard 0), the Phi-3 tokenizer, Pythia-70m,
GPT-2 and their SAEs were all already in the WSL HF cache (`~/.cache/huggingface/hub`). Every run used `HF_HUB_OFFLINE=1`.

| file | what |
|---|---|
| `strict_heldout.py` | wrapper around 073 `strict_eval.main()` with `score_target` bound to `c_held=C_HELD` |
| `run_strict.sh` | the four local runs (2 models x training / held-out C donors) |
| `analyze_strict.py` | 109 `t4_robustness.strict()` scoring on the given rows, plus the paired comparisons |
| `tl_dce.py` | TuringLLM ΔCE on the public-model protocol (`BOS=1`: sensitivity arm) |
| `out/strict_<model>_<train\|held>/rows.jsonl` | 073 rows (and the copied production `random_pool.pt`) |
| `out/strict_summary.{md,json}` | #3 results |
| `out/tl_dce{,_bos}.{csv,json}`, `out/tl_dce_bos.log`, `out_tl_dce.log` | #18 results and logs |

---

## #3: strict resample test with held-out contrast donors

### How c_held works

- `073 strict_eval.score_target(..., c_held=False)` builds its C donors as `nt_ho if c_held else nt_tr`. Here
  `nt_tr` = the target's 48 training contrast contexts (the resample term's own donors in WCM v4) and `nt_ho` = the
  16 held-out ones. The production records hold 64 contrast rows and no `neg_held_v2`, so the split is 48 / 16.
- `strict_eval.main()` never passes `c_held` and there is no env switch for it. The 082 chain's `strict` arm
  (chain.sh l. 697) therefore ran with training C donors. The later experiments (079, 091, 092, 095, 102) call
  `SE.score_target(G, key, pool, c_held=True)` directly.
- `strict_heldout.py` does the same thing by setting `SE.score_target = partial(score_target, c_held=...)` and then
  calling 073's own `main()`. Nothing else changes:
  - the targets are the chain's nulls list (128 per model);
  - the A donors are the 48 strongest-train contexts;
  - the R/T donors come from the production run's own `random_pool.pt`;
  - `N_DRAWS` is 2;
  - the random variant is the anchor null's draw 0;
  - OUT and CTX_DIR point at the production archive.
- The donor-draw generator is consumed in a fixed order (R indices first, then A, then C), so A and R draws are
  identical between the two local arms (median |Δ rs_A| = |Δ rs_R| = 0.000). Only C changes, so the comparison is a
  clean paired one.
- Scoring follows 109 `t4_robustness.strict()` (copied): pass = Z, A and C in merge.BAND [0.75, 1.5] and production
  necessity >= NEC_MIN; vacuous by merge.vacuous on the production main row. The script reproduces the paper's
  production figures exactly: 66.7 [57.9, 74.6] and 43.0 [34.4, 51.6].

### Results

Hardware: local RTX 5070 Ti, WSL, the same code. The production rows are from the H100 pod.

| model | config | scored | strict pass (A, C) % [95% CI] | given production pass | A / C / T median |
|---|---|---|---|---|---|
| Pythia-70m | production (H100, training C) | 126 | 66.7 [57.9, 74.6] | 80.8 [73.1, 87.5] | 0.90 / 0.88 / 0.71 |
| | local, training C | 126 | 68.3 [60.3, 76.2] | 80.8 [73.1, 87.5] | 0.90 / 0.88 / 0.70 |
| | **local, held-out C** | 126 | **69.0 [61.1, 77.0]** | 81.7 [74.0, 88.5] | 0.90 / 0.88 / 0.70 |
| GPT-2 small | production (H100, training C) | 128 | 43.0 [34.4, 51.6] | 51.9 [42.5, 61.3] | 0.79 / 0.75 / 0.66 |
| | local, training C | 128 | 44.5 [35.9, 53.1] | 52.8 [43.4, 62.3] | 0.78 / 0.75 / 0.67 |
| | **local, held-out C** | 128 | **43.8 [35.2, 52.3]** | 51.9 [42.5, 61.3] | 0.78 / 0.72 / 0.67 |

The other cells are the same in all three configurations:

| model | resample (A, T) | (A, C) at α = 1 | random null |
|---|---|---|---|
| Pythia-70m | 42.1% | 4.8% | 0.0% |
| GPT-2 small | 33.6% | 1.6% | 0.0% |

Paired comparisons on the same targets (bootstrap 95% CI of the difference). "Only" counts targets that pass under one
configuration and fail under the other.

| model | comparison | difference | only first / only second |
|---|---|---|---|
| Pythia-70m | local training C − local held-out C | −0.8 pp [−4.8, +3.2] | 3 / 4 |
| Pythia-70m | H100 training C − local training C (hardware) | −1.6 pp [−4.0, 0.0] | 0 / 2 |
| GPT-2 small | local training C − local held-out C | +0.8 pp [−3.1, +4.7] | 4 / 3 |
| GPT-2 small | H100 training C − local training C (hardware) | −1.6 pp [−4.7, +1.6] | 1 / 3 |

**Finding.** Swapping the C donors for held-out contrast contexts moves the strict pass rate by less than one point:
+0.8 pp on Pythia and −0.8 pp on GPT-2, with both intervals spanning 0. The median C recovery drops slightly, by
0.00 on Pythia (0.876 → 0.880) and 0.03 on GPT-2 (0.753 → 0.724). So the reported 66.7% / 43.0% are **not** inflated
by in-sample C donors. The hardware gap (H100 vs local, same donors) is larger than the donor effect: about 1.6 pp
in the local GPU's favour on both models, from 2 and 3 flipped targets.

### Caveats

- The held-out-donor rates are local-GPU numbers. Paired with the local training-donor run, the donor effect is
  ≤ 0.8 pp. Against the H100 production rows, the combined difference is +2.4 / +0.8 pp, about two thirds of it
  hardware.
- The "production pass" column (82.5 / 82.8%) and the necessity gate come from the H100 main eval. They were not
  recomputed locally. Only the strict rows (Z all-position, A, C, T) were rerun.
- The held-out contrast contexts are the records' as-run held-out slice (rows 48-63). Their disjointness from the
  training rows follows the record builder and was not rechecked here (cf. DAN-157 / DAN-217 for the activating sets).
- The A donors are still training activating contexts (see #2, already fixed). Only C is now out of sample.
- Pythia has one target with no circuit file (`1.mlp.1032`), the same as in production. It is an error row and is not
  scored.

### Proposed paper changes (#3)

Main text l. 1784 (app:resample, production paragraph). **Recommended** version: keep the published H100 numbers and
add the held-out check.

> The strict resample test of the first paragraph was rerun on the public models' null panels only: $66.7\%$ and $43.0\%$ of the targets pass it, against $82.5\%$ and $82.8\%$ under the pass criterion, $4.8\%$ and $1.6\%$ at $\alpha = 1$, and none for random circuits; for TuringLLM it is not available in this run. Its contrast donors there are the target's training contrast contexts, the resample term's own donors; rerun with the held-out contrast contexts as donors instead (same targets, one local GPU for both arms), the pass rate moves by less than a point ($68.3\% \to 69.0\%$ and $44.5\% \to 43.8\%$), so the figure is not inflated by in-sample donors.

Alternative, if the table should show the held-out-donor rows. This needs the 109 t4 strict reader pointed at
`119/out/strict_<model>_held` (the table is generated, so it should not be hand-edited). The tab:run2026-resample
cells would become:

| row | Test pass | Test pass given production pass | A / C / T |
|---|---|---|---|
| Pythia-70m, strict resample (A, C) | 66.7 {[}57.9, 74.6{]} → 69.0 {[}61.1, 77.0{]} | 80.8 {[}73.1, 87.5{]} → 81.7 {[}74.0, 88.5{]} | 0.90 / 0.88 / 0.71 → 0.90 / 0.88 / 0.70 |
| GPT-2 small, strict resample (A, C) | 43.0 {[}34.4, 51.6{]} → 43.8 {[}35.2, 52.3{]} | 51.9 {[}42.5, 61.3{]} unchanged | 0.79 / 0.75 / 0.66 → 0.78 / 0.72 / 0.67 |

The random-null rows (0.0) and the production-pass column are unchanged. The test label should become "strict
resample (A, held-out C)", and l. 1784's numbers would become 69.0% and 43.8%, with a clause naming the local-GPU
rerun.

---

## #18: TuringLLM SAE ΔCE

### Already computed?

Partly. `src/eval/sae/reconstruction_eval.py --ce` wrote `analysis-restyled/sae-eval/tables/sae-reconstruction-eval.csv`
(column `ce_difference` = ΔCE). That is the source of the model card's "55% loss recovered". It is on **12,288
tokens of TuringLLM's own training corpus** (`data/` shards), not on the public protocol. No TuringLLM ΔCE existed on
the WikiText protocol of the public columns (070 `recon_audit.py`: 1,024 tokens, 16 × 64, no BOS, CPU fp32).

### What was run

`tl_dce.py` uses 070's text exactly: the wikitext-103-raw-v1 train shard 0, rows of the first 6,000 longer than 200
characters, joined. It is tokenised with TuringLLM's tokenizer (Phi-3-mini, `add_special_tokens=False`) into 16
windows of 64 tokens. The splice is `reconstruction_eval._SiteSplice` ("recon" / "zero"), the same one that produced
the model-card numbers. Runs were on CPU in float32: no GPU was used for #18.

- TuringLLM's clean CE on this text is **6.77 nats**, against 2.93 on its own corpus.
  - WikiText is far out of distribution for TuringLLM. Its training shards are synthetic, textbook-style text, and
    every document starts with `<s>`.
  - A sensitivity arm with `<s>` prepended to each window (`BOS=1`) gives a clean CE of 6.99 and the same picture:
    resid L11 +0.58, attention ≤ +0.017, MLP ≤ +0.051.
  - So the high CE is a domain effect, not a missing-BOS effect.
- At this CE level, splicing noise is large relative to the small sites' effects. Several non-residual and
  early-residual sites show *negative* ΔCE, down to −0.07 at resid L1.

| | WikiText protocol (1,024 tok, no BOS) | own corpus (12,288 tok, existing CSV) |
|---|---|---|
| clean CE | 6.770 | 2.928 |
| FVU | 0.03 (mlp L0) – 0.30 (mlp L9) | 0.006 – 0.23 (EV 0.77 – 0.99) |
| ΔCE, all 36 sites | −0.07 to +0.55 | −0.001 to +0.24 |
| ΔCE, residual | −0.07 to +0.55 (+0.02 to +0.55 from L3 on; L11 +0.546) | +0.013 to +0.238 (L11) |
| ΔCE, attention | −0.001 to +0.008 | −0.001 to +0.003 |
| ΔCE, MLP | −0.023 to +0.046 | +0.004 to +0.026 |
| attn L10: zero-ablation gap / ΔCE | +0.072 / +0.006 | +0.0016 / +0.0003 |
| attn L11: zero-ablation gap / ΔCE | −0.011 (zeroing *helps*) / −0.001 | +0.0022 / +0.0010 |

The per-site values are in `out/tl_dce.csv` and `out/tl_dce_bos.csv`.

**Finding.** The layer-11 attention SAE is not weak: its reconstruction costs +0.001 nats on TuringLLM's corpus and
−0.001 on WikiText. Its "55% loss recovered" comes from a 0.002-nat denominator. On WikiText, zero-ablating that site
even lowers CE, so loss recovered is undefined there. The same holds for layer-10 attention (84%, denominator 0.0016).
In ΔCE terms, TuringLLM's SAEs look like the public ones:
- the residual sites dominate, rising with depth to about +0.24 (own corpus) or +0.55 (WikiText) at L11;
- attention and MLP sites stay within ±0.05.

### Caveats

- On WikiText, TuringLLM is off-distribution (CE 6.77). Its ΔCE values there are noisier and partly negative, so
  they are not a clean like-for-like comparison with the public models (Pythia and GPT-2 are near their training
  distribution on WikiText).
- The own-corpus ΔCE is in-distribution but uses a different text and token count from the public columns.
- 1,024 tokens is a small sample. Site differences of a few hundredths of a nat at the non-residual sites are within
  noise on this text.

### Proposed paper changes (#18)

1. **tab:model-cards**, TuringLLM "Quality" cell (main.tex l. 634).
   - Current: `explained variance $0.77$--$0.99$; layer-1 attention: 37\% of latents fire; layer-11 attention: 55\% of loss recovered (\cref{fig:sae-quality})`
   - Recommended (own corpus, in distribution):
     `FVU $0.01$--$0.23$; $\Delta$CE $\leq +0.24$ nats, largest at the residual stream ($+0.01$ to $+0.24$), at most $+0.03$ elsewhere, on 12{,}288 tokens of its own corpus; layer-1 attention: 37\% of latents fire (\cref{fig:sae-quality})`
   - Alternative (WikiText protocol, matches the caption exactly):
     `FVU $0.03$--$0.30$; $\Delta$CE $-0.07$ to $+0.55$ nats, largest at the residual stream (up to $+0.55$ at layer 11), within $\pm 0.05$ elsewhere; WikiText is off-distribution for TuringLLM (clean CE $6.77$ nats); layer-1 attention: 37\% of latents fire`
2. **Model-card caption** (l. 640). With the recommended cell, append:
   `For TuringLLM, whose training text is synthetic and on which WikiText is off-distribution (clean CE $6.77$ against $2.93$ nats), FVU and $\Delta$CE are over 12{,}288 tokens of its own corpus; on the WikiText protocol its $\Delta$CE is $-0.07$ to $+0.55$, again largest at the residual stream.`
3. **app:sae-quality** (l. 650). Replace "and the layer-11 attention SAE recovers 55\% of the gap despite explaining 0.96 of variance (layer 10: 84\%)" with:
   `Loss recovered is ill-conditioned at the attention sites of layers 10 and 11, where zero-ablating the site raises cross-entropy by only $0.0016$ and $0.0022$ nats, so their $84\%$ and $55\%$ say nothing about the dictionaries: splicing in the reconstruction costs $+0.0003$ and $+0.0010$ nats.`
   - Keep the 37% firing sentence. "Two sites are weak" becomes "One site is weak".
   - Optionally note the same ill-conditioning in the fig:sae-quality caption, if it shows loss recovered for L10/L11
     attention.
