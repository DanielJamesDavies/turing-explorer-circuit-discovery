# Eval-substrate sensitivity check (2026-08-01)

Question (Daniel's): do the frozen exam's cf numbers depend on its
store-built negatives? Held the CIRCUITS fixed (R top-64 / top-1024
include from the D2.2 archive; AR16 = 16 latents at D2.3's alpha*) and
re-evaluated cf (and sup) on 16 negatives from each selector mode
(close, random, distant, fused; fixed selection seed), vs the store rows
already banked in D2.2/D2.3. 132 rows, 11 seeds, ~9 min GPU, no
rediscovery. Tables in `summary_tables.md`.

## VERDICT: the frozen exam is substrate-ROBUST

- Medians move by <= 0.077 across all four modes for every set (AR16
  1.043 store vs 0.966-1.035 modes; R64 0.705-0.719; R1024
  1.101-1.116). Median per-(seed,set) max deviation: 0.041 — inside the
  documented run-to-run noise band.
- The HEADLINE survives its hardest test: L11's 16-latent driver scores
  cf_alpha = 1.000 on CLOSE negatives (the near-firing contexts where
  contamination lives), 0.972 random / 1.0 fused.
- a_base = 0.0000 in EVERY AR16 cell on EVERY substrate — the
  "silent context" premise holds by measurement on all modes, not just
  store geometry.
- Known failures stay failures everywhere (substrate-independent):
  L8-mlp AR16 ~ 0.29-0.32 on all modes (vs 0.30 store); L9-attn
  identity failure cf <= 0.20 on all modes. The largest single
  deviation anywhere is 0.199 (L8-resid R64, a low-cf cell where
  relative noise is big); nothing changes sign or verdict.

## Consequence

The store-negative caveat on the exam is RETIRED for cf claims: absolute
numbers can be quoted from the frozen exam without a "store-grade
silence" asterisk, and no exam-v2 fork is needed. (Pre-top-k near-miss
contamination remains real as a DISCOVERY-side phenomenon — D2.1/Phase-1
measurements stand — it just does not move the eval.)
