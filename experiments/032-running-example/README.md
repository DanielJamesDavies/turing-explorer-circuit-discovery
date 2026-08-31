# Running-example tri-amp run (2026-08-10)

Single-seed variant of `029-panel/runner.py` for the paper's
running example: the temperature seed (L3 resid, comp 11, latent
35381, "temperature as a thermodynamic state variable"). Identical
held-out protocol (48 train / 16 held-out), shallow-band config
(triple_w 0.10, lam100 4e-3), no null draws (the panel's 124-draw null
covers the class claim). Scoring code verbatim from the panel runner;
adds per-arm membership dumps (`members_<arm>.json`) for the
budget-churn comparison.

Purpose: real numbers for the running-example sentences in sec 5.2 and
sec 5.5 (readability priority 6 — thread the example through Results).

## Results (rows.jsonl; all held-out)

| arm | n | ampF0 | ampFMd | cf_amp | cf_bare | sup |
|---|---|---|---|---|---|---|
| triamp400 | 192 | 1.12 | 1.272 | 1.415 | 0.84 | 1.0 |
| triamp100 | 250 | 1.128 | 1.384 | 1.245 | 0.828 | 1.0 |
| gate400 | 577 | 0.98 | 0.756 | 0.887 | 0.904 | 1.0 |

Seed stats: a_pos tr 8.479 / ho 7.812; e0_ho 0.000 (clean zero floor);
amp-inject negctx baseline a_base 4.50 (the seed is pre-activation-warm
on its close negatives — below the Top-K cut, consistent with the
unverified-negatives caveat; the cf_amp normaliser works over the
7.81 - 4.50 gap).

Membership overlap (members_triamp400 vs members_triamp100):
shared 166; **Jaccard 0.601**; shared/smaller 0.865. Weighted-in-gate
containment 0.927.

## What the paper quotes

- sec 5.2: 192-member weighted circuit, F0 1.12, FMd 1.27 (just past
  the [0.8, 1.25] band — quoted honestly as a near-pass), sup 1.0,
  drive 1.42 -> 1.25 at quarter budget; gate-only 577 members, FMd
  0.76.
- sec 5.5: both budgets reconstruct (F0 1.12 / 1.13) at n 192 / 250,
  Jaccard 0.60: two samples from the family.

Caveats: one seed, no per-seed null (class-level null is the panel's);
FMd for this seed sits outside the all-pass band at both tri-amp arms,
so the seed is not quoted as an ALL-PASS example, only as the
running-example instantiation.
