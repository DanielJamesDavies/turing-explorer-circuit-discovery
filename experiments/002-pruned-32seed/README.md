# Pruned 8-arm 32-seed run (2026-07-19)

Baseline × schedule lattice, all position-aware, `abs_pctl@90` union +
`abs_pctl@95` restoration rounds, **magnitude-bisection prune ON** (relative
floor: free0 within 0.05 of each raw circuit's own free0), `negative_roles=
include` throughout (post role-fix: PA membership is both-sign / sign-invariant).
30 usable seeds × 8 arms = 240 rows.

- Data: `pruned-32seed-core8.jsonl` (rows carry `size_raw` → `size`, all evals,
  per-arm `secs`). Runner: `pruned-32seed-core8_runner.py`. Aggregate:
  `aggregate.py`. Figure: `compression-closure-frontier.png`.

## Medians (non-collapsed cells; keep% = pruned/raw)

| arm | raw | pruned | keep% | free0 | free0 deep | cf | #collapse |
|---|---|---|---|---|---|---|---|
| abl-ig_mean PA | 308k | 66k | **21%** | 0.950 | **0.950** | 1.43 | 0 |
| cf-ig_mean PA | 309k | 71k | 23% | 0.952 | **0.954** | 1.41 | 0 |
| abl-restoration PA | 398k | 87k | 22% | 0.951 | **0.951** | 1.32 | 0 |
| cf-ig_negctx PA | 252k | 132k | 53% | 0.862 | 0.590 | **0.948** | 2 |
| act-grad | 44k | 27k | 62% | 0.870 | 0.420 | 1.48 | 2 |
| cf-local PA | 180k | 98k | 55% | 0.707 | 0.400 | 1.17 | 6 |
| abl-ig_zero PA | 27k | 15k | 55% | 0.682 | 0.185 | 1.32 | 4 |
| abl-restoration-zero PA | 22k | 6.5k | 30% | 0.346 | **0.000** | 0.90 | 14 |

## Findings

1. **Mean-floor both-sign arms win and compress to ~21–23%** while holding
   free0 ≈ 0.95 at EVERY depth (0 collapses). The keepable circuit at φ0.95 is
   ~66–87k — the "311k is a lot" worry largely dissolves post-prune.
2. **cf/abl convergence confirmed**: abl-ig_mean (0.950, 66k) ≈ cf-ig_mean
   (0.952, 71k) — the gap-vs-drive objective washes out once roles are unified.
3. **Zero-baseline is a NEGATIVE result**: compact raw (abl-ig_zero 27k,
   act-restoration 22k) but they DON'T close — abl-ig_zero deep 0.185,
   act-restoration deep 0.000 (14 collapses). The free0-*coherent* zero baseline
   gives *worse* free0: a fully-zeroed stream can't be reconstructed by path or
   greedy walk at depth.
4. **negctx (cf-ig_negctx) least compressible (53% keep) but best-calibrated**
   (cf 0.948, no overshoot) — its members are uniformly load-bearing.
5. **The tradeoff is real**: nothing is both tiny AND closing. Deep
   latent-endpoint closure needs ~70–90k nodes (the depth-scaled irreducible
   core, now post-prune); smaller circuits sacrifice closure.

## CAVEAT — relative-floor prune degeneracy

The relative floor (base_free0 − 0.05) collapses low-closure cells to 1 node:
when a deep seed's raw free0 ≤ 0.05, any 1-node subset satisfies the floor, so
the bisection prunes everything (`#collapse` column). This ONLY hit the
low-closure arms (0 collapses on the three winners), and the medians above
EXCLUDE collapsed cells, so the low-closure arms' numbers are OPTIMISTIC (true
deep zero-baseline / cf-local numbers are worse). A clean follow-up would rerun
those four arms with an ABSOLUTE φ target (e.g. 0.85) + raw-free0 logging, so
every cell stays interpretable.
