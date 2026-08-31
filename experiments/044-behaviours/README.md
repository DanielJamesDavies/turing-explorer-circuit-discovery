# Tri-amp circuits on SFC-style unsupervised behaviours

Goal (Daniel, 2026-08-30): replicate the behaviour-data method of
Sparse Feature Circuits (Marks et al., ICLR 2025 — who use the quanta
clustering of Michaud et al. 2023), then fit tri-amp masks on a few
behaviours and analyse the results.

## Pipeline

1. `cluster_behaviours.py` — activations-variant quanta clustering on
   TuringLLM: sample 64-token windows, keep contexts predicted
   confidently (p ≥ 0.2 at position 62→63), represent each context by
   the concat of per-layer L2-normalised residuals at the anchor,
   spherical k-means K=100. v1: NWIN=8192 (~4k kept, clusters 27–53).
   v2 (current `behaviour_clusters.pt`): NWIN=32768 → 15,942 kept,
   clusters ~100–300.
2. `behaviour_runner.py` — per cluster: metric = mean log P(cluster's
   actual next token | context) at the anchor; fit
   `objective="logit"` (G11 endpoint machinery) over all 36 sites,
   `mask_floor_source="zero"`, `free_amplitude=True` (tri-amp),
   λ=3e-3, 400 steps, 75/25 train/held-out split. Reports EF_ho and
   EF_tr (overfit meter).

## Behaviours picked (v2, auto3 = most coherent with size ≥ 80)

- cluster 5 (n=119): "as well / such →' as'" completion
- cluster 99 (n=111): post-apostrophe contraction/possessive ('t 's 're)
- cluster 95 (n=121): year-digit continuation ("early 1…", "19→9")

## The debugging arc (order matters)

1. First scores looked null: EF −0.14…+0.35, memberships 2.5k–5.5k.
2. EF_tr ≈ EF_ho on all three → NOT overfitting; the "fit" itself
   looked like it failed.
3. `score_zf.py` (score in the fit's zero-fill frame): still ~0.1 →
   fit/score fill-frame mismatch NOT the cause.
4. `ceiling_test.py` (keep ALL latents): EF = 1.000 exactly → the
   evaluator is delta-injection style, SAE error terms are RETAINED;
   perfect EF is achievable (the SFC error-node caveat does not apply
   to this evaluator).
5. **The actual bug**: `behaviour_runner.metric()` scored memberships
   at α=1.0 — it dropped the fitted amplitudes of a tri-amp circuit.
   Every knowledge-arc scorer passes `keep_scales`; this one didn't.
6. `score_amp.py` — with amplitudes, zero-fill (fit frame), held-out:
   **EF 0.987 / 0.961 / 0.935** (clusters 5 / 95 / 99); plain-α EF on
   the same members: 0.095 / 0.106 / 0.007. Mean-fill with amps stays
   low (−0.02 / 0.12 / 0.26): amplitudes are calibrated against zero
   fills — the circuit is frame-specific by construction.
7. `null_amp.py` — matched control: random member ids, same per-site
   counts, circuit's own amplitude values permuted on: EF −0.018…−0.004
   across 2 nulls × 3 clusters. Membership identity, not amplitude
   mass, carries the recovery.

## PICKUP (2026-08-31, run cancelled for the night at Daniel's request)

Expansion batch (`CLUSTERS=49,79,17,47,75,15,9,61 LAM=3e-3 ...
behaviour_runner.py`, runner now fixed to score zero-fill WITH
amplitudes + matched amp-permuted nulls) completed 3 of 8 before
cancellation — results already in behaviour_rows.jsonl:
  behav49 "cru→cial"        n=1134 EF_ho=0.933 (tr 0.998, nulls .02/.06)
  behav79 "as →a"           n=2266 EF_ho=0.920 (tr 0.942, nulls .03/.02)
  behav17 "essential →for"  n=4217 EF_ho=0.957 (tr 0.982, nulls -.00/.02)
TO RESUME: rerun with CLUSTERS=47,75,15,9,61 (the 5 not yet fitted),
then `run_kb.sh` minus its wait loop (or just: make_knowledge.py, then
DATA=knowledge_clusters.pt OUT=knowledge CLUSTERS=0,1,2 runner) for
the three constructed knowledge behaviours (SR→1905 / GR→1915 minimal
pair + →Einstein attribution). make_knowledge.py never ran — no
knowledge_clusters.pt yet. Afterwards: kb0⊖kb1 membership diff, and
overlap of kb0 members with the hand-mapped L4→L9 year-relay latents.

## Caveats / open

- Memberships are large (2,477 / 3,477 / 5,457 at λ=3e-3) — behaviour
  circuits at ~95% EF cost far more nodes than concept circuits;
  no λ sweep yet, so minimal size unknown.
- v1 rows (clusters 3/19 in `behaviour_rows.jsonl`, and the
  score_zf lines for them) are stale: v2 clustering overwrote
  `behaviour_clusters.pt`, so v1 members no longer pair with the
  cluster ids in it. Only clusters 5/95/99 rows are valid.
- Amplitudes: free_amplitude softplus (positive); scoring must pass
  `keep_scales` or the tri-amp circuit is misrepresented.
