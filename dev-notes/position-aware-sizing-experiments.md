# Position-aware allowed-set: sizing experiments

Two planned experiments to address the **menu-size concern** for position-aware /
allowed-set circuits. Context (2026-07-14): position-aware discovery reaches free-φ
0.82–0.98 across seeds L3–L11, but the allowed-set membership is large — **2–11k
latents/site** (5–27% of the 40,960 dictionary at one layer-kind), vs ~12–24/site for the
deployed circuits. mlp/resid dominate; attn stays small. See `position_aware.py`,
`perlayer_counts.py`, and the ⚠ MAJOR section of `paper/paper-updates.md`.

Key distinction to keep in mind: the **"meal"** (latents active at any single position)
stays ~128 regardless; the **"menu"** (the aggregated union) is what's large. These
experiments probe whether the menu is a *bounded, meaningful* object and whether it can be
*shrunk* without losing sufficiency.

---

## Experiment 1 — Menu saturation (full-topctx run)

**Question.** As we add more contexts, does the allowed-set union **saturate** (grow
sub-linearly and plateau — the seed has a finite driver vocabulary) or grow ~linearly
forever (then the flat menu isn't a well-defined object)? And does free-φ sufficiency hold
up on a larger, more representative context set?

**Honest caveat first.** A bigger run does **not** shrink the menu — more sequences means a
*larger* union. What it buys is (a) whether the union plateaus, and (b) whether
position-aware sufficiency generalises beyond the tiny 4-sequence batch. It answers "is the
menu bounded and the result robust," not "is the menu small."

**Method.**
- Pick ~3 seeds spanning depth (e.g. L3 mlp 10/7370, L7 resid 23/29486, L11 mlp 34/9005).
- Use the seed's **full `top_ctx`** (up to ~64 stored sequences) instead of the 4-seq probe
  batch. Process per-site incrementally to bound memory (the `[B,T,d_sae]` attribution is
  large — reduce to per-position top-N indices immediately, discard the full tensor).
- Sweep number of sequences **K ∈ {1, 2, 4, 8, 16, 32, 64}**. For each K, at fixed N (say
  64 and 96): form the position-aware allowed set over K sequences' causal prefixes, then
  measure **(a) union members/site** and **(b) free-φ** (allowed-set zero-ablation eval) on
  those K sequences.

**What we plot / decide.**
- `members/site vs K` — flat tail ⇒ saturates (menu is bounded); straight line ⇒ unbounded.
- `free-φ vs K` — stays ~0.8–0.9 ⇒ sufficiency generalises; decays ⇒ overfit to few contexts.
- Decision: if it saturates AND sufficiency holds, the menu is a legitimate (if large)
  object and we can quote a converged size. If it grows unboundedly, pivot to per-instance /
  per-sequence circuits as the reported object.

**Cost.** One GPU pass per (seed, K); the attribution is one backward per K-batch. Memory is
the watch-out — incremental per-site reduction required.

---

## Experiment 2 — Attribution thresholding (shrink the menu)

**Question.** Top-N keeps a fixed N latents per position regardless of magnitude, so it
sweeps in a long tail of tiny-|attribution| latents. If the per-position attribution is
**peaked** (a few large, many negligible), can we keep sufficiency (~0.8–0.9 free-φ) at a
**fraction** of the size by thresholding on |attribution| instead of taking a fixed top-N?

**Method.** Replace "per-position top-N" with a threshold rule; sweep the threshold; plot the
**size-vs-faithfulness frontier** against the top-N baseline. Three threshold variants to try:
1. **Global absolute** — keep latent at position t if `|attr[t, latent]| >= θ` (sweep θ).
2. **Per-position relative** — keep if `|attr| >= f * max_latent |attr[t, ·]|` (sweep the
   fraction f, e.g. 0.05–0.5). Scale-free across positions/sites.
3. **Cumulative mass** — per position, keep the smallest set covering X% of that position's
   total |attr| mass (sweep X, e.g. 80–99%). Directly targets "how much of the signal."

For each setting: members/site + free-φ. Overlay on the top-N curve (from
`nonoracle_gate.py`).

**What we decide.**
- If any threshold rule gives the **same free-φ at materially fewer latents/site** (e.g. ~0.8
  at hundreds/site vs top-N's thousands), adopt it — the allowed set becomes far more
  interpretable, and `position_aware_top_n` gets replaced/augmented by a threshold config
  knob.
- If thresholding tracks top-N with no size win, the size is intrinsic (the attribution tail
  genuinely matters for stream reconstruction) — then lean on the per-position "meal" framing
  and per-instance circuits for the compact object.

**Implementation note.** Both live in `position_aware_membership` (`src/circuit/instrument/
position_aware.py`): the selection loop currently does `block.abs().topk(n)`. Thresholding
swaps that for a mask; a config knob (`position_aware_select: "top_n" | "abs" | "relative" |
"mass"` + a threshold value) makes it toggleable alongside the existing `position_aware`
flag. Default stays top-N so nothing changes until opted in.

---

## Priority

Run **Exp 2 (thresholding) first** — it directly attacks the size concern and is the cheaper
of the two (same batch, just different selection). **Exp 1 (saturation)** answers whether the
menu is bounded, which matters most if we promote allowed-set circuits to the paper's main
object (decision (b) in `paper-updates.md`). Both should run on the same 3 depth-spanning
seeds so the numbers compose.

---

## Exp 2 — RESULTS (2026-07-14)

Ran all four rules on the three seeds (B=4 positive sequences each; `scratchpad/
thresh_sweep.py`, plot `thresh_frontier.png`). Verdict: **`abs` (global absolute
magnitude cut) wins decisively — it dominates top-N, `relative`, and `mass` on every
seed.** Best rows, free-φ @ mean latents/site:

| Seed | top-N baseline (best) | best `abs` | shrink @ ≥φ |
|---|---|---|---|
| 10/7370 L3 mlp  | 0.976 @ 3,263 (N=96) | **0.988 @ 1,221** (p75); 0.85 @ 553 (p90) | **2.7×** |
| 23/29486 L7 resid | 0.824 @ 2,602 (N=96) | **0.946 @ 1,721** (p50) | 1.5× **and higher φ** |
| 34/9005 L11 mlp | 0.852 @ 6,502 (N=96) | **0.951 @ 4,620** (p50) | 1.4× **and higher φ** |

`abs` threshold = a per-seed percentile of pooled nonzero |attr| (p50/p75 is the sweet
spot; p90 trades φ for a much smaller set on the shallow seed). It must be **calibrated
per seed** (raw attribution units are seed-specific) — one extra sort of the pooled |attr|.

**`relative` and `mass` are dominated** even after a bug fix. Both normalise by a
per-position quantity (row max / row total); a *dead* position (seed doesn't attribute to
it, whole row ~0) divided by ~0 and selected the **entire 40,960-latent dictionary**, which
the union spread across the site (max=40960 everywhere). Fixed in `_selection_mask` with a
live-position guard (dead rows select nothing; `abs` never needed it). After the fix they're
bounded but still track/underperform top-N — per-position normalisation wastes budget on
locally-large-but-globally-tiny latents. **The signal is peaked in a *global* magnitude
sense, so a global cut is the efficient selector.** Guard covered by unit tests.

**Honest caveats.** (1) The win is a *constant factor* (1.4–2.7×), not order-of-magnitude:
deep seeds still need ~1.7k–4.6k/site for ~0.95 φ — **depth drives absolute size**;
thresholding shaves the constant. Only the shallow seed gets genuinely compact
(553/site @ 0.85). (2) `abs` needs per-seed percentile calibration to be usable in
production (current `position_aware_select="abs"` takes a raw threshold).

**Follow-ups.** Wire a percentile-calibrated `abs` into production (add
`position_aware_threshold_mode: "absolute" | "percentile"` so the config takes a percentile
directly, computed per seed). Then Exp 1 (saturation) still worth running to see whether the
`abs`-selected menu plateaus with more contexts.
