# D4.2 — Driver identifiability (2026-08-01)

11-seed panel, split-half probe resampling (disjoint halves of the
deterministic probe order via a builder proxy; R / abl-ig / direct-mass
per half), precision perturbation (R full-48 under autocast_bf16 vs the
archived fp32 ranking), K* per half, and AMPC alpha stability + frozen-
exam transfer. Rows in rows.jsonl; half-ranking heads archived as
rank_{method}_{half}_{seed}.jsonl.gz; tables in summary_tables.md.

## Verdicts

1. **The pre-registered symmetry holds in DIRECTION but is moderate,
   not absolute.** Non-attn: driver heads reproduce at Jaccard 0.68-0.78
   (@16) across halves vs 0.51-0.61 for full membership — heads are
   tighter than closure, but ~25-30% of head members still swap under
   probe resampling. The honest paper phrasing: "closure membership is
   a family; driver sets are a NARROWER family."
2. **What IS identifiable is the FUNCTION, not the member list.** AMPC
   alpha is stable across halves (within ~15%; often identical) and
   both halves' K=16 sets impose on the frozen exam at cf ~ 0.98-1.11
   on 7/9 non-attn seeds (deep-mlp half-B dips to 0.71-0.74). Two
   half-disjoint discoveries buy two DIFFERENT 16-latent sets that both
   work — solution multiplicity reaches all the way down to drivers,
   but every solution found is real. The single-latent/16-latent
   headline should be framed "A driver, reproducibly constructible",
   not "THE driver".
3. **Attn inverts**: heads LESS stable than membership (0.34-0.52 vs
   0.55-0.61), K* doubles across halves (L3) or stays unreachable (L9).
   Where imposition fails, the head ranking is noise — nothing to
   identify. Consistent with the L9-attn identity failure.
4. **Precision is a non-factor for discovery** (this doubles as D2.6's
   A/B): bf16-autocast R rankings are Jaccard 1.00/1.00/0.99/0.99 at
   K=16/64/256/1024 and 1.00 on full membership vs fp32, with NO
   wall-clock change (median 0.97x; the one 94s->2s pair is D2.2's
   compile warmup, not a speedup). VERDICT: the autocast_bf16 flag can
   be flipped for training-regime consistency at zero cost, or left
   off — it changes nothing measurable at discovery. No panel rerun
   needed either way.
5. **K\* carries a ~1.1-1.7x resampling spread** (2.1x on attn) — the
   D1 caveat quantified: deep K* is order-of-magnitude, and any K*
   comparison needs to clear ~1.5x before it means anything.

## Caveats
- One split (A/B), not a resampling distribution — spreads are 2-point.
- Halves have 24 probes vs the exam's 48: half-discoveries are noisier
  than production ones, so these Jaccards are a LOWER bound on the
  48-probe stability.
- alpha ceiling 8.0 censors the mlp/attn alpha-stability cells.
