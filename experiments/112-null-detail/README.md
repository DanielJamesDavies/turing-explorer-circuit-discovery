# 112 — Harder-null appendix detail (DAN-181, remaining part)

Fills the `\pending` in `app:nulls` (main.tex "Harder random circuits.") from the 2026-11-07/08 production run.
CPU only, reads the unpacked archives under `experiments/109-pod-run-analysis/out/archives/` (TuringLLM T6
`pub-turingllm-T6-nulls/.../out_turingllm/harder_nulls`; public `out_<model>/harder_nulls` in the S10 results archives),
the stored null circuits next to them and the production circuits (`main/circuits`).

```
python -X utf8 experiments/112-null-detail/null_detail.py      # ~15 s
```

Writes `out/nulls_detail_*.csv`, `out/nulls_detail_meta.json`, `out/run.log`, `paper/tables/run2026_nulls_detail.tex`,
`paper/figures/run2026_nulls_detail.{png,pdf}` (necessity vs min(Z, A, C) per draw). Draft appendix text:
`draft_paragraph.tex` (cites a `tab:run2026-nulls-detail` label the owner would add).

## Rules
- Rows: 084 `report.frame` (good rows, last per seed/arm/draw/split). Pass and vacuous: `062 merge.passes` (floor 0.75)
  and `merge.vacuous`, as 109 `t4_robustness.nulls`. Bands: 109 `lib109.add_site` (family bands).
- Scored targets = non-vacuous production held-out row: TuringLLM 112 (123 with rows, 11 vacuous), Pythia 126 (127, 1),
  GPT-2 128. Training-split rows use their own vacuous flag (1 TuringLLM target vacuous on training).
- Bootstrap: 4,000 resamples over targets, seed 20261108 (`lib109.boot_ci`). For 0/n the bootstrap interval is
  degenerate ([0, 0]), so the exact one-sided 95% upper bound 1 − 0.05^(1/n) is also given (`exact_upper95_pct`).
- **Moved coefficient** (084's definition): |α − α_inert| > 0.01, α from the circuits' node amplitudes.
  α_inert = the gradient-free AdamW trajectory. The production run is WCM v4: geometric lr, peak 0.05 × 3.15, floor
  0.05, warm-up 0.1, wd 0.05, 400 steps, so **α_inert = 0.7901**, not 0.7975 (0.7975 = protocol v1's constant lr;
  both computed in the script). 0.7901 is also the modal α in the weak-null rows. The run's rows have no
  `alpha_moved` for wcm/anchor/contrib: 084's `alpha_inert()` returns None for a non-constant schedule. So the
  share is recomputed here from the stored circuits. Where a row has it (the weak null, modal fallback), it agrees
  to within 0.001.
- Quantile: 084 `report.quantile` on targets with ≥ 2 draws (27/31/32). With 2 draws it can take only 5 values, so
  the median is 1.0 everywhere; the shares are more informative.
- Jaccard = overlap / (n_drawn + n_ref − overlap), from the row fields `overlap_wcm`, `n_drawn`, `n_ref`.
- Shortfall: 084 tables 8/9 (`n_exact`, `dist0`, `by_bin`), plus the drawn set's contribution relative to the
  production members' (`set.contrib_median/mean` vs the profile row's `set_wcm`), and the top-decile fill
  (`cbin_hist` bins 9–10).

## Results (draw 0, held-out strongest unless stated)

| | TuringLLM | Pythia-70m | GPT-2 small |
|---|---|---|---|
| Production pass held / train | 85.7 / 90.1 | 82.5 / 96.8 | 82.8 / 87.5 |
| Every null (anchor, contrib, weak; fitted and α=1) pass held / train | 0 / 0 | 0 / 0 | 0 / 0 |
| Exact 95% upper bound (all bands) | 2.6% | 2.3% | 2.3% |
| Anchor Z/A/C, nec held; train | 0/0.003/0, 0.04; 0/0.007/0, 0.04 | 0/0/0, 0.28; 0/0/0, 0.27 | 0/0/0, 0.12; 0/0/0, 0.13 |
| Contrib Z/A/C, nec held; train | 0/0.34/0.26, 0.66; 0/0.42/0.30, 0.64 | 0/0/0, 0.29; 0/0/0, 0.30 | 0/0/0, 0.91; 0/0/0, 0.92 |
| Weak (DAN-17 refit) Z/A/C, nec | 0/0/0, 0.006 | – | – |
| α moved, median (wcm / anchor / contrib / weak) | 100 / 99 / 99 / 92% | 99 / 93 / 96 | 99 / 92 / 94 |
| WCM min(Z,A,C) > both draws (anchor, contrib) | 100, 100% | 100, 100 | 100, 100 |
| WCM passes, both draws fail | 81.5% | 77.4 | 81.2 |
| Anchor overlap share / Jaccard (median; max J) | 1.4% / 0.007 (0.10) | 4.5% / 0.022 (0.10) | 1.3% / 0.006 (0.05) |
| Contrib exact-bin share per draw (median) | 68% | 58% | 71% |
| Contrib exact in top member decile (pooled) | 22% | 11% | 22% |
| Contrib set contribution / members': median, mean | 0.87, 0.51 | 0.71, 0.29 | 0.94, 0.39 |
| Contrib nec by band | 0.37 / 0.72 / 0.74 | 0.05 / 0.21 / 0.53 | 0.55 / 1.00 / 1.00 |
| Contrib removal switches target off (Top-K = 0) | 5% | 0% | 36% (52–56% L5–11) |

By band, every null passes 0 in every band. Production per band: TuringLLM 89.5/87.2/80.0, Pythia 90.3/79.2/80.9,
GPT-2 96.2/84.8/65.1 (`out/nulls_detail_bands.csv`).

## Things that are odd or that change earlier text
1. **The weak null's coefficients are not inert in the production run.** The paper's caveat (median α exactly 0.798,
   no gradient) describes the protocol-v1 run. On the TuringLLM panel under v4 the weak null's median α is 0.790,
   which is v4's inert value. But 92% of its coefficients moved > 0.01, symmetrically up and down, with median
   |α − inert| ≈ 0.4. A median at the inert value does not mean the coefficients did not move.
2. **All 123 weak-null references on TuringLLM are refits** (`dan17_source = refit`). None is stored, although
   084's README planned 65 stored ones.
3. **The contribution-matched null keeps high necessity but fails sufficiency.** Necessity is
   (a_pos − a_without set)/(a_pos − e0), measured in the live model and blind to the coefficients (so fitted = α=1 on
   100% of rows). On Top-K SAEs, removing a set that carries enough of the target's contribution pushes the
   target's pre-activation below the k-th largest, and the target goes to exactly 0, so necessity = 1. This is what
   happens for 87% of production circuits and 36% of GPT-2 contrib draws (52–56% in L5–11). Necessity tracks how
   much member contribution a draw recovered (Spearman 0.69/0.54/0.56 with the contribution-mean ratio; 0.63/0.50/0.63
   with the exact-bin share). GPT-2's matching is the closest (exact 71%, median ratio 0.94), so its necessity is
   the highest. Sufficiency fails anyway: Z ≥ 0.75 in 0% of GPT-2 draws and 2% of TuringLLM draws. So necessity is
   not specific once contribution is matched. It shows the target depends on many strong upstream latents, not
   that these latents are the circuit.
4. On GPT-2 the production circuit's necessity is ≥ both contrib draws on only 59% of the two-draw targets, because
   both are often at the ceiling of 1.0.
5. The anchor-active null's members carry about 1% of the production members' contribution (median-ratio 0.009–0.017),
   even though their frequency deciles match exactly. Anchor activity plus frequency does not select contributors.
