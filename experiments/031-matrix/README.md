# The definitive cross-method matrix (2026-08-10)

Attribution arms (abl-ig_mean PA, cf-ig_mean PA, abl-restoration PA
rounds=sites; abs-p50 selection) on the SAME 22 seeds, probes, held-out
split, and evaluation semantics as 029-panel; mask arms
(gate400/triamp100/triamp400) join from the panel's rows. Each
attribution arm scored at its natural full union AND truncated to the
seed's triamp400 size (global top-n by |attribution|). Data:
rows.jsonl (runner.py). FMd vacuity guard: freeM outside (-1,3)
excluded (seed L11/1829 + the inverted-denominator attn seed).

## The matrix (medians over 22 seeds, held-out)

  arm         n_med    free0  freeM   cf     sup   disc_secs
  abl_ig     591,120   0.99   0.99   1.15   1.00     87
  cf_ig      591,152   0.99   0.99   1.15   1.00     82
  resto      560,077   0.99   0.97   1.14   1.00     33
  abl_ig@n       369   1.08   0.00   0.61   1.00     --
  cf_ig@n        369   0.99   0.00   0.56   1.00     --
  resto@n        369   4.36   0.00   0.66   0.95     --
  gate400      1,216   0.66   0.23   0.75   1.00     65
  triamp100    1,061   0.96   1.00   1.01   1.00     18
  triamp400      369   0.99   1.08   0.89   1.00     67

  ALL-PASS at matched size (both floors in band + sup>0.9, /21):
  attribution truncations 0 - 0 - 0 | gate400 3 | triamp100 12 |
  triamp400 17

## Findings

1. **At natural size every family passes; the price is the size.**
   All three attribution arms reconstruct held-out at ~0.99 on both
   floors — at 10^5-10^6 members. The faithfulness cost measured yet
   again, now cross-method.
2. **Under compactness pressure the families separate absolutely.**
   Truncated to the weighted circuits' size, attribution passes 0/21
   (freeM median exactly 0.00; resto@n free0 explodes to 4.36);
   gate-only 3/21; the weighted circuit 17/21 AT THE SAME MEDIAN SIZE
   (369). Compact faithfulness is only achievable, on this evidence,
   by jointly optimised membership + amplitudes.
3. **Full-union arms have CONVERGED memberships**: abl_ig / cf_ig /
   resto post near-identical scores per seed because abs-p50 unions
   mostly coincide — method identity matters at compact sizes, not at
   the union scale.
4. **Cost inverts the old intuitions**: restoration at rounds=sites is
   now the CHEAPEST attribution arm (33s vs 82-87s); and triamp100 is
   the cheapest arm in the whole matrix (18s median) while passing
   12/21 — discovery-with-optimization is not the expensive option.
5. Caveat: @n rows are naive global-top-n truncations; a bisection
   prune re-targeted to n could do better. It cannot rescue freeM 0.00
   at 369 vs 1.08, but quote the caveat with the table.
