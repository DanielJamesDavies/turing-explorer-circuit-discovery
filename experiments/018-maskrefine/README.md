# D3.6 — Mask-refine over attribution support: 24-seed run (2026-08-01)

Daniel's proposal, run at his direction: the learned mask trained ONLY
over an attribution-proposed candidate support (R = abl-restoration PA
ranking; top-50k = MS50, top-10k = MS10) vs the house mask over the full
1.4M-latent space (MF). Same recipe and lambda across arms (same
pressure, different search space). 24 valid seeds = 11 D1 panel + 13
stratified new (degenerate a_pos=0 candidates skipped by a guard added
after pass 1; one junk seed 27/40424 left None-metric rows in
rows.jsonl — excluded from all tables). Engine: `support` parameter in
run_learned_mask (7 unit tests; mask suite 76 green). Member lists
archived per (arm, seed); R rankings archived for the 13 new seeds
(extends the D2.2 archive to 24 seeds). Tables in `summary_tables.md`.

## Verdicts

1. **Shallow (L0-4): the restriction is free or better.** MS50 median
   dfree0 +0.016 (3 wins / 1 loss of 10); even MS10 is +0.009 with 4/0
   wins and 9% smaller circuits. When the mask's answer already lives
   inside the attribution head (containment 65-95% shallow), searching
   only there costs nothing and sometimes regularises (L3-attn: MS10 is
   HALF the size at BETTER free0, 0.99 vs 0.87).
2. **Deep, split by KIND — the run's sharpest finding.**
   - deep MLP: MS50 is a clean WIN — median dfree0 +0.019 at x0.65
     size (both L8-mlp seeds: 31-38% smaller AND better; L10-mlp:
     half the size at equal quality).
   - deep RESID: MS50 loses ~0.12 free0 at slightly LARGER size (0/3
     wins); MS10 collapses outright (free0 0.0 on 5 deep seeds).
   The mask's closure at deep resid genuinely needs latents that
   restoration's top-50k does not contain.
3. **Containment (the original gate) falls with depth**: median 57%
   @10k / 80% @50k, but shallow ~80-95% -> deep ~17-33%. LOW containment
   does NOT automatically mean the restriction fails (deep mlp wins at
   20% containment — the support holds SUBSTITUTES), and HIGH
   containment does not guarantee a win (L5). Containment measures
   overlap of solutions, not sufficiency of the space.
4. **Jaccard(MS50, MF) median 0.43**: the restricted mask finds a
   substantially DIFFERENT circuit, not a filtered copy of the free one
   — consistent with the known solution multiplicity (dtype churn 16.5%,
   lambda swaps). Restriction steers WHICH member of the solution
   family training lands on.
5. **MS10 is a shallow-only tool**: fine to +0.009 at L0-4, collapses
   beyond (deep median dfree0 -0.695; five hard zeros). Support width
   must scale with depth — as closure size itself does.

## Recommended defaults from this data

Support restriction is worth adopting where it wins: shallow seeds
(either width) and deep MLP (50k). It should NOT be the default for
deep resid seeds. If a single rule is wanted: support = R top-50k for
kind != resid or layer <= 4, full dictionary otherwise. A support-width
sweep (100k-200k at deep resid) is the obvious follow-up if a uniform
rule is desired.

## Caveats

- Same lambda across arms: sizes are not matched (the restriction
  changes the lambda->n curve); dfree0 comparisons at these sizes are
  the operating-point comparison, not a matched-size one.
- One R ranking per seed (single 48/48 discovery); support quality
  inherits R's known depth weaknesses.
- free0 is the primary column; cf/sup rows exist for all arms but the
  mask is a closure object — driver-side reads belong to the AMPC line.
- 5 attn seeds, 2 of them weak (a_pos < 1.1) — attn medians are thin.

## Width-sweep addendum (autonomous follow-up, rows_width.jsonl)

The deep-resid loss is a WIDTH problem, not a restriction problem.
MS100/MS200 on the three losing seeds (free0, vs main-run MF / MS50):

| seed | MF | MS50 | MS100 | MS200 |
|---|---:|---:|---:|---:|
| 26/17432 L8-resid | 0.96 (14.6k) | 0.83 | 0.93 | **0.95 (15.9k)** |
| 29/2753 L9-resid | 0.68 (22.0k) | 0.55 | 0.68 | **0.69 (23.5k)** |
| 35/6599 L11-resid | 0.88 (13.4k) | 0.77 | 0.86 | **0.87 (15.9k)** |

MS200 recovers MF-level free0 at comparable size on all three; MS100 is
within 0.03 on two of three. 200k is still a ~7x reduction of the
search space at L11 (200k of 1.43M). REVISED RULE: support width scales
with depth — 10-50k shallow, 50k deep mlp, 100-200k deep resid; with
that schedule the restriction is viable EVERYWHERE, and the D3.6 idea
stands panel-wide.

## FINAL VERDICT (Daniel, 2026-08-01)

Not a paper result. The wins are modest and band/kind-conditional, and
a depth-scaled width schedule is more machinery than the free mask
needs. The run's lasting value is the byproducts: 24-seed R ranking +
member archives, and the containment/Jaccard evidence of solution
multiplicity (underdetermination section).
