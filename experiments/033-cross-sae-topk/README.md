# Cross-SAE replication on GENUINE Top-K dictionaries (2026-08-11)

THE cross-SAE replication for the paper. Supersedes the dense-ReLU arm
(030-cross-sae), which is out of domain: the method's
pre-activation objective exists because Top-K censors the activation,
so Top-K dictionaries are its scope. Nothing here simulates sparsity.

  Model   EleutherAI/pythia-70m
  SAEs    EleutherAI/sae-pythia-70m-32k — trained Top-K, k=16,
          32,768 latents, d_in 512, decoder-normalised, 8.2B Pile
          tokens; resid + attention + mlp for all 6 layers
  Corpus  wikitext-103 windows (64 tokens)
  Protocol 48 train / 16 HELD-OUT contexts; membership, amplitudes,
          floors and pins all fitted on train; negatives VERIFIED
          inactive (zero firing in window)
  Panel   5 layers x 3 host kinds x 2 seeds = 30 analysed seeds
          (32 scanned; 2 excluded by a stated uniform rule, held-out
          a_pos < 1.0, i.e. the exam sits at the noise floor)
  Arms    triamp400 (triple floor + free amplitudes), gate400,
          and 5 amplitude-fitted random nulls per seed (146 draws)

Semantics transcribed from EleutherAI/sparsify: pre_acts =
(x - b_dec) W_enc^T + b_enc (the seed read, raw); code = topk_k(relu(
pre_acts)); decode = code W_dec + b_dec (b_dec cancels in the delta
intervention). Verified on GPU: exactly 16 nonzeros per position.

## Results

  arm          n_med   ampF0  ampFM  sup   cf_amp   ALL-PASS
  triamp400    51      1.06   1.07   1.00  0.95     22/30
  gate400      175     0.83   0.63   1.00  1.04      2/30
  146 nulls    matched max 0.09 / 0.54 / 0.79 / 0.23  0/146

  By host kind (triamp400): resid n=37 pass 10/12 | mlp n=56 pass 7/9
  | attn n=310 pass 5/9 (cf median 0.07 — reconstructs, barely drives)
  Median n by layer: L1 27, L2 40, L3 100, L4 55, L5 138

  ANCHOR SUPPORT (36 measurements): 0.00190-0.00259, median 0.00215.
  NB this is ~4x the analytic k/d_sae = 0.00049 because the null draws
  from the LIVE pool, which is biased toward frequently-firing
  latents. Quote the measured number, not the analytic one.

## Findings

1. **The core panel results replicate on someone else's dictionaries**:
   compact weighted circuits (median 51) faithful on both floors and
   necessary, 22/30; gate-only 2/30 at 3.4x the size with mean-fill
   median 0.63 (the same collapse as at home); size grows with depth.
2. **The null recovers completely: 0/146 draws pass anything**, and no
   draw's zero-fill score even enters the band (max 0.09). Together
   with 0/124 at home this is the null validated on two models and two
   independently trained Top-K dictionary families.
3. **The mechanism is measured**: anchor support 0.21% here vs 6.5-37%
   on dense ReLU (030-cross-sae) — 30-170x. A random latent
   under Top-K is outside the top-k at the anchor almost always, so
   its fitted amplitude multiplies exactly zero and there is nothing
   for the null to exploit at any n.
4. **Per-kind drive structure recurs across architectures**: attention
   seeds reconstruct (1.05/1.05) but barely drive (cf 0.07), the same
   asymmetry the 22-seed home panel found.

## Re-run with the SAE error held clean (2026-09-19, for the 052 Gemma port)

`ERROR_MODE=clean` (see crosssae_topk.py header): each edited site becomes
c_hat W_dec + b_dec + err_CLEAN instead of + err(edited stream). Needed on
uncapped JumpReLU SAEs, where the recomputed error feeds itself and diverges
(052/res_explosion_diag.py). This re-run asks whether it changes TopK results.
triamp400 only + 2 fitted nulls per seed; `rows_clean.jsonl`,
`run_clean.log`, `compare_clean.py` / `compare_clean.log`. 28 paired seeds
(held-out a_pos >= 1.0).

  median           original   clean
  band pass        21/28      18/28   (23/28 verdicts agree: 4 lost, 1 gained)
  ampF0            1.056      1.061
  ampFM            1.063      0.950   (lower on 23/28 seeds)
  sup              1.000      1.000
  cf_amp           0.895      0.776
  members n        51         61      (+-22% per seed, both directions)
  fitted nulls     0/140      0/56 pass (max ampF0 0.004)
  by kind: resid 9/10 -> 8/10, mlp 7/9 -> 6/9, attn 5/9 -> 4/9

**Verdict: the central claims replicate under either form (null dead,
necessity intact, zero-fill faithfulness unchanged), but the clean form is a
DIFFERENT counterfactual, not a drop-in: -3 passes of 28, mean-fill ~0.11
lower, and the losses concentrate at the deepest layer (mlp L5 #7608 F0
0.98 -> 0.61; resid L5 #14086 F0 0.90 -> 0.28, FM 1.10 -> 0.19). On TopK the
recomputed error is well-behaved, so the original form stays the reference
there.** An earlier intermediate form (subtract the clean CODE) was wrong
(leaks x - x_clean through non-members) and is archived as
rows_clean_deltaform.jsonl / run_clean_deltaform.log — do not quote.

## phi_pin CONTROL for the Gemma port (2026-09-20, `rows_pin_panel.jsonl`, `run_pin_panel.log`)

On Gemma 3 270M + Gemma Scope 2 every latent circuit reads free0 0.67-1.13 but
**phi_pin 0.00-0.37**: clamping the SAME members to alpha x their CLEAN values
does not reproduce the seed, though letting them be re-encoded from the edited
stream does. This asks whether that gap is ours or the substrate's. `pin0` /
`pinM` added to this harness (members clamped to alpha x clean position-wise
values, non-members at the zero / posctx-mean floor); triamp400 only, no nulls,
28 seeds with held-out a_pos >= 1.0:

  metric          median
  ampF0           1.050
  ampFM           1.061
  sup             1.000
  pin0            0.789      pin0 >= 0.5: 24/28 | pin0 < 0.2: 1/28
  pinM            0.891
  by kind  resid 0.931 (min 0.497) | mlp 0.865 (0.359) | attn 0.737 (0.196)
  by layer L1 0.971 | L2 0.793 | L3 1.014 | L4 0.711 | L5 0.750

**On genuine Top-K dictionaries free and pinned AGREE, at every layer and kind
(no depth decay). The Gemma collapse is therefore a property of the uncapped
JumpReLU substrate, not of the method.** Top-K bounds what a re-encode of a
degraded stream can produce; 052's cap+clamp restores STABILITY but not the
IDENTITY of the member values (on Gemma the median member is SILENT in the free
counterfactual, 052 `val_ratio_median` 0.00, while <5% are inflated). Quote
phi_pin next to every free score from now on.

## Reporting rules

Quote as: the method, compact weighted circuits, necessity, the
gate-only comparison and the null all replicate on a public model with
public Top-K dictionaries under a held-out, verified-negative
protocol; the weak-seed exclusion is stated and uniform; the null's
validity is an architecture-scoped claim backed by the measured
anchor-support rate.
