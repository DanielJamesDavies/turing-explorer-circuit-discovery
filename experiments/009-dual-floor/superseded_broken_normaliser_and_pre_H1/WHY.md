# Why these 52 rows are superseded (2026-07-30)

Two independent reasons, either of which alone would invalidate a comparison
against newer rows:

1. **Broken dual normaliser.** Each dual term was divided by its OWN
   fully-closed-mask loss. That quantity is unbounded: at L10 the zero
   floor's closed state drives the seed's pre-activation to ~1.8e5 (zeroing
   all 32 upstream sites is far off-manifold), giving norm_zero = 3.3e10
   against norm_floor = 176 — a ratio of 1.9e8. The zero term entered the
   loss at ~1e-8 relative weight, so **dual silently degenerated into
   negctx-only** and reproduced negctx-only's exact failure (free0 == 0.0,
   negative freeM_topk). Measured ratios: L2 137.7, L5 1.08, L8 1.19,
   L10 1.9e8 — which is why only L10 detonated.
   Fixed by scaling both terms by a shared, bounded mean(target^2).

2. **Pre-H1 numerics.** The transform then decoded `code` and `dense`
   separately and subtracted; it now decodes the difference once. Same
   algebra, better conditioned, but NOT bitwise — membership shifts ~0.5%.

The ZERO-FLOOR and NEGCTX-ONLY arms here are unaffected by (1) — they never
touch dual scaling — so the "negctx-only is dead at depth" finding stands
(free0 exactly 0.0 and negative freeM_topk at L5/L8/L10). Every DUAL arm is
invalid, and so is the reading that "lambda has no leverage at L10".

Do NOT merge these rows with the re-run: (2) means even the unaffected arms
are not bitwise comparable.
