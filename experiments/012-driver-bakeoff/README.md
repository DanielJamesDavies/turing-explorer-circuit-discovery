# D1 — Driver bake-off (completed 2026-07-31 14:45; 280 rows, 11 seeds)

Frozen driver exam (48/16 held-out split, pin0 metrics, calibrated
imposition, random controls) over 7 arms x K in {64,256,1024,4096}.
Panel: 8 layer-stratified seeds + L8-mlp, L9-attn (documented), L4-mlp
(sampled). The sampled L1-attn seed had no positive contexts (skipped).
Tables in `summary_tables.md`; per-seed rows in `rows_s*.jsonl`; direct
weights saved per seed (`direct_full_*.pt`).

## Verdicts (read per driver notion — D0.1's split holds panel-wide)

**Intervention-drivers (phi-cf, the headline):**
1. **Restoration round-prefix (R) wins at every K** — median cf 0.71 / 0.93 /
   1.11 at K=64/256/1024, ahead of hybrid (0.62/0.91/0.93) and the flat
   attribution heads (~0.57/0.84/0.93). The greedy-from-the-floor ordering
   is the best driver ranking we have, at zero extra machinery.
2. **Amplified direct-mass is the pound-for-pound champion**: cf_alpha ~
   1.00-1.03 at EVERY K (median), including **1.01 at K=64** where raw
   injection gets 0.58. The alpha law is clean: alpha*(K) = 1.64 -> 1.16 ->
   1.04 -> 1.04 — smaller sets need proportionally more gain. "N latents at
   gain g" is a real driver object, discovered in ~2s/seed.
3. **The hybrid did NOT hold its champion title** under the held-out split
   and global-K budgets: H <= R at every K. The earlier hybrid win was
   per-site-cap-matched and in-distribution; this exam is stricter.

**Pinned-drivers:** everything is pin-dead below K=4096 in medians; at
K=4096 **R leads (0.91; 0.72 in the deep band)** with C second (0.78; 0.26
deep) — the only two arms alive at depth, consistent with D0.1's
direct/near-seed story (restoration's early rounds pick direct-ish members).

**Kind is the decisive axis** (cf@256): resid 0.87-0.98 across real arms;
mlp 0.26-0.55; attn -0.01-0.46. **Only C (0.36) and R (0.46) survive on
attention seeds**; A/D/H collapse there. Any driver method claim must be
kind-split.

**The universally-hard seed keeps its title**: L9-attn 27/6859 — K* hits
the ceiling (full ranking never pins 0.8), cf <= 0.28 for every arm, and
amplification CANNOT rescue it (alpha* pegged at 8.0, cf_alpha 0.26): its
imposition failure is identity, not amplitude. Nothing in any ranking
contains what makes it fire. Case-study candidate.

**mask_inject (MI) scouting verdict: defaults are dead.** Empty circuits on
8/11 seeds; where it produced sets they scored ~0 (one negative sup). The
dual-intervention mask (D3.2) is mandated — it needs its own hyperparameter
regime, not the closure mask's.

**sup stays a sanity gate only**: ~0.94-0.99 at K=64 for every real arm,
0.00-0.04 for random — necessity is cheap everywhere (D0.1 confirmed at
panel scale).

**K\* (pinned-driver size)**: 77 -> 18,296 across L0-L11 (T4), broadly
growing with depth, order 10^2-10^4 vs closure 10^4-10^6 on the same seeds.

## Caveats

- Mid-run fixes (logged in rewrite-tracker): hybrid attr patch, MI floor/
  NPA/None-guard, and DEPTH-AWARE BATCHING introduced mid-panel (deep seeds
  discovered at probe-batch 2/1 vs shallow at 4). Within-seed arm
  comparisons are unaffected (same process); cross-seed attribution
  numerics carry the documented batching jitter.
- K* is ILL-CONDITIONED at depth: recomputation under changed batching
  drifted 46% at L8-resid (7,190 -> 3,875) but 2% at L8-mlp — boundary
  flatness is seed-dependent. Treat deep K* as order-of-magnitude with
  resampling error bars (D4.2).
- MI has rows on only 3 seeds (16 rows) — earlier seeds' MI errored through
  the fix sequence; the empty-at-defaults verdict is from clean guard logs
  on all 11.
- alpha-fit uses collapsed-pin targets and activator-only injection;
  amplified-random control rows exist at K=256 (cf_alpha ~0 — amplification
  does not rescue arbitrary sets).
