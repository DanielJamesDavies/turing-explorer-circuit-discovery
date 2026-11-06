# 088 SAE properties of passing vs failing targets

Question: do the TuringLLM targets whose circuits fail share properties in their SAE (encoder, decoder, bias,
frequency, rivals), beyond "the target is weakly active"?

Data: 062 full run, `main/eval.shard*.jsonl`, held == "strong", vacuous_tk dropped: 14,994 targets, 35 sites (no
L0 attn targets), pass 46.4%. PASS = free0_tk, freeM_topk_tk, freeN_topk_tk in [0.8, 1.5] and phi_sup_blind_tk >= 0.9.
A = freeM_topk_tk, C = freeN_topk_tk. Every target, no subsampling.

    wsl: PYTHONPATH=src .venv/bin/python experiments/088-sae-properties/compute.py   # GPU, ~3 min
         PYTHONPATH=src .venv/bin/python experiments/088-sae-properties/analyze.py

- `compute.py`: per target, from `models/TuringLLM/SAE` (one site at a time) the encoder row norm, b_enc,
  b_eff = b_enc - W_enc.b_dec (the bias the engine's pre-activation uses), decoder column norm, encoder-decoder cosine,
  max cosine to every other latent's encoder row / decoder column, and the counts above 0.3 / 0.5 / 0.7; site
  percentiles of the norms and b_eff; from `outputs/latent_stats.pt` the firing frequency and mean activation when
  active; from 512 windows x 63 tokens (shard 2000, position 0 dropped) the site's Top-K cut (128th largest post-ReLU
  pre-activation per token) and the target's pre-activation mean / std / max. -> `out/props.csv`, `out/sites.csv`.
- `analyze.py`: pass vs fail medians; Spearman rho pooled and within depth band (L0-4 / L5-7 / L8-11) x kind cells
  (mean of per-cell rho); partial rho controlling for a_pos_tk (and, separately, for the spec rows' rank_clean, pi A);
  5-fold CV logistic AUC per cell. -> `out/results.md`, `out/table_{all,attn,mlp,resid}.csv`, `out/merged.csv`.

## Results (all kinds; within = mean over 9 cells)

| property | med pass | med fail | rho pass, within | rho A, within | rho C, within | partial rho pass, ctrl a_pos |
|---|---:|---:|---:|---:|---:|---:|
| encoder row norm | 0.86 | 0.65 | +0.27 | +0.38 | +0.47 | +0.13 |
| b_eff, site percentile | 0.40 | 0.56 | -0.25 | -0.32 | -0.40 | -0.12 |
| b_eff / std(pre-act) | -0.87 | -0.68 | -0.23 | -0.26 | -0.34 | -0.11 |
| decoder column norm | 1.00 | 1.00 | 0.00 | +0.03 | +0.04 | -0.02 |
| cos(encoder, decoder) | 0.79 | 0.77 | -0.06 | -0.10 | -0.09 | -0.01 |
| max encoder cosine, other latent | 0.18 | 0.18 | +0.18 | +0.29 | +0.34 | +0.06 |
| max decoder cosine, other latent | 0.18 | 0.22 | +0.05 | +0.17 | +0.17 | -0.03 |
| # rivals, enc or dec cos > 0.5 | 0 | 0 | +0.01 | +0.01 | 0.00 | +0.01 |
| log10 firing frequency | -2.93 | -2.79 | -0.13 | -0.10 | -0.15 | -0.04 |
| mean act when active / site cut | 2.7 | 1.6 | +0.22 | +0.33 | +0.41 | +0.07 |
| a_pos_tk / site cut | 12.4 | 5.9 | +0.31 | +0.42 | +0.50 | +0.20 |
| a_pos_tk (the control) | 5.7 | 2.8 | +0.26 | +0.38 | +0.44 | - |
| rank_clean (spec, pi A) | 2 | 10 | -0.30 | -0.42 | -0.49 | -0.16 |

Per kind (full tables in `out/results.md`):
- **mlp**: the clearest SAE signal is the bias. Passing latents have a more negative effective bias relative to
  their pre-activation spread (b_eff / std: -1.85 pass vs -0.87 fail; within rho -0.44, partial -0.31 after
  a_pos, -0.27 after rank). Failing mlp latents also fire more often (within rho -0.27) and have smaller encoder
  norms (0.63 vs 0.85; partial +0.21). In L0-4 mlp, adding the SAE properties to the activation features lifts the CV
  AUC from 0.65 to 0.84.
- **attn**: failing latents have smaller encoder norms (site percentile 0.51 vs 0.78; partial +0.14) and a less
  negative bias (site percentile 0.49 vs 0.22). They do **not** have more near-duplicate rivals; the reverse holds.
  Passing attention latents have the closer neighbours (encoder NN 0.26 vs 0.21, decoder rivals > 0.3 median 1 vs 0),
  and the effect mostly disappears after a_pos (partial +0.07 / +0.08). Once activation is in the model, the SAE
  properties add nothing in L0-7 attention (CV AUC 0.85 -> 0.85, 0.76 -> 0.75) and +0.03 in L8-11.
- **resid**: weak throughout. Partial rho is |rho| < 0.07 except a_pos/cut (+0.17).

## What stands out

1. There are no near-duplicate rivals. Encoder or decoder cosine > 0.5 to another latent occurs in at most 2% of
   targets, and > 0.7 in none. The Top-K competition that pushes weak targets out is diffuse, not one split twin.
2. Decoder norms are all 1.0 (the decoder is unit-normalised), so they carry no information.
3. Every geometry signal is mostly an activation-scale signal. A larger encoder norm and a more negative bias both
   mean "strong and selective when on". Controlling for a_pos_tk roughly halves each rho. The independent residue
   is about +-0.1 to 0.2, or 0.3 for the mlp bias ratio.
4. a_pos / site cut keeps a partial rho of +0.20 beyond raw a_pos. Where the target sits relative to the site's Top-K
   threshold matters, which fits the "pushed out of the Top-K" mechanism.

## Caveats

Correlational only. The cut comes from generic corpus tokens of one shard, not the target's own contexts. Its
cut is higher on the target's own strong contexts, so a_pos/cut overstates headroom. Cells mix layers, so site
scale can leak into raw values; the site-percentile versions guard against this. A per-cell rho has SE about 0.02-0.03,
and the attn L5-11 cells hold only 81 / 84 passes.
