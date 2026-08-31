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

## Reporting rules

Quote as: the method, compact weighted circuits, necessity, the
gate-only comparison and the null all replicate on a public model with
public Top-K dictionaries under a held-out, verified-negative
protocol; the weak-seed exclusion is stated and uniform; the null's
validity is an architecture-scoped claim backed by the measured
anchor-support rate.
