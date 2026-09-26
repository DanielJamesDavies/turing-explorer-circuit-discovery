# 059 — Which activating contexts should WCM train on? (the 64 strongest vs strongest + mid-band)

2026-09-24. Asked by Daniel after 058: does training only on peak firing give circuits that explain ordinary firing,
and do mid-band contexts, or just more data, help?

**Setup:** `pool_test.py` and `run_all.sh`.
- The primary config (close C, rank-keep 3e-3, γ 0.25, λ 1e-3, ablation values from the training contexts) and
  the new stratified split (`src/circuit/context_split.py`).
- All arms are scored on the SAME two held-out sets: 16 of the 64 strongest, and 16 of the mid-band reservoir.
- One evaluation reference for all arms: the A ablation value is the mean over the 48 strongest-train contexts;
  C is the close contrast set, split 48 / 16.
- Contexts are built once and injected, so the engine and eval code are unchanged.
- 10.attn.16967 has only 19 stored top contexts and no mid-band pool, so the comparisons use 15 targets.

| Arm | Trains on | Nodes (median) |
|---|---|---|
| D | 32 strongest | 337 |
| A | 48 strongest (the protocol-v1 choice) | 397 |
| B | 32 strongest (= D) + 16 mid-band | 474 |
| C | 48 strongest + 48 mid-band (96) | 536 |

## Results (activation read; `results/summary.md`, paired stats below)

**Held out: the strongest contexts** (target activation median 2.73)

| Arm | free0 | freeM_topk | freeN_topk | Worst-of-3 \|faith − 1\|, median | In band [0.8, 1.25] | Closer to 1 than A |
|---|---|---|---|---|---|---|
| D | 0.891 | 0.892 | 0.891 | 0.19 | 8/15 | 11/15 |
| A | 0.922 | 0.859 | 0.870 | 0.30 | 5/15 | — |
| **B** | **0.952** | **0.918** | **0.903** | **0.14** | **9/15** | **14/15** (median −0.07) |
| C | 0.896 | 0.902 | 0.867 | 0.32 | 5/15 | 8/15 |

**Held out: mid-band contexts** (target activation median 1.31)

| Arm | free0 | freeM_topk | freeN_topk | Worst-of-3 \|faith − 1\|, median | In band | Closer to 1 than A |
|---|---|---|---|---|---|---|
| D | 1.012 | 1.050 | 1.068 | 0.74 | 1/15 | 5/15 |
| A | 1.051 | 0.991 | 0.936 | 0.72 | 1/15 | — |
| **B** | 0.999 | 0.981 | 0.940 | **0.41** | 2/15 | 9/15 |
| C | 1.055 | 1.212 | 0.952 | 0.63 | 4/15 | 9/15 |

- Necessity is 1.000 everywhere.
- Sufficiency to induce on the mid-band set is 1.5–1.8 for every arm. That's an artefact: the injected values
  come from strongest-train contexts, while the denominator is the lower mid-band activation. Not comparable.

**Node overlap (median Jaccard):** D–A 0.59, D–B 0.50, A–B 0.49, A–C 0.44, B–C 0.41.

## Reading

1. **Peak-trained circuits transfer to ordinary firing on average, but not per target.** On mid-band held-out
   contexts every arm's medians are about 1.0, yet the spread is wide (worst-of-3 deviation 0.4–0.7) and only
   1–4/15 are in band. The mid-band targets fire at half the strength, which makes the ratios noisier. It is a
   harder test, not only a worse circuit.
2. **Mixing mid-band contexts in at a fixed budget helps.** B (32 strongest + 16 mid) is closer to 1 than A
   (48 strongest) on 14/15 targets on the strongest held-out set, and on 9/15 on the mid-band set. It halves the
   worst-of-3 deviation on both (0.30 → 0.14; 0.72 → 0.41). The cost is +19% nodes (474 vs 397). Adding 16
   mid-band contexts (D → B) beats adding 16 more strongest ones (D → A).
3. **More data alone does not help at a fixed 400 steps.**
   - C (96 contexts) is no better than A on the strongest set and costs +35% nodes. It improves the mid-band
     medians only.
   - A is worse than D on 11/15 targets.
   - Both are confounded by the step budget: at 400 steps and batch 4, each context is seen about 50× (D),
     33× (A, B) and 17× (C). Our calibration notes say lr and steps act as a budget.
4. **Sizes grow with training-set diversity:** 337 → 397 → 474 → 536.

## Caveats

- 15 targets, one fit per arm. Refit noise is real (same-config Jaccard ≈ 0.66), and band counts move by ±2–3.
  The B-vs-A paired result (14/15) is the strongest signal; the D-vs-A reversal is a warning about noise and the
  step budget.
- The B/C training ablation value (mean over their own training contexts, which include mid-band ones) differs
  from the common evaluation reference, which favours A and D if anything.

## Follow-ups (2026-09-24): E, B2 (noise floor), C800 (budget control)

Full table: `results/summary.md`. Comparisons are on the strongest held-out set unless stated otherwise.

| Arm | Trains on | Nodes | Worst-of-3 dev | In band | Closer to 1 than A | Closer to 1 than B |
|---|---|---|---|---|---|---|
| A | 48 strongest | 397 | 0.30 | 5/15 | — | 1/15 |
| B | 32 + 16 mid | 474 | 0.14 | 9/15 | **14/15** | — |
| **B2** | 32 + 16, resampled | 462 | 0.23 | 6/15 | **11/15** | 5/15 |
| E | 48 + 16 mid | 453 | 0.25 | 6/15 | 6/15 | 2/15 |
| C800 | 48 + 48, 800 steps | **308** | 0.39 | 4/15 | 7/15 | 4/15 |

On the mid-band held-out set, B and B2 are closer to 1 than A on 10/15 and 9/15; E on 9/15; C800 on 7/15.

1. **The 32 + 16 composition survives resampling, but B's sample was lucky.**
   - A second, independent sample of the same composition (B2) still beats A on 11/15 targets (strongest) and
     9/15 (mid-band).
   - But B beats B2 on 10/15, and B vs B2 share only Jaccard 0.52 of their nodes, no more than different arms
     do (0.41–0.59).
   - The sample-to-sample noise is about as large as B's margin over B2. The true effect of mixing in mid-band
     contexts is real but moderate: roughly B2's margin, not B's.
2. **Adding 16 more strongest contexts to 32 + 16 hurts (E).** E is closer to 1 than B on only 2/15 and no better
   than A (6/15) on the strongest held-out set. At 400 steps, 48 training contexts at about 2:1 strongest:mid is
   the best of what we tried.
3. **More steps is not a clean fix; it changes the size regime.** C800 (96 contexts at 800 steps) shrank the
   circuits from 536 to 308 nodes and lost faithfulness (freeN 0.74, sufficiency to induce 0.69). The sparsity
   price acts for twice as long, so steps are a budget for sparsity as well as for fitting, as the calibration
   notes said. "More data plus more steps" would need λ recalibrated to compare at matched size. That belongs
   to the DAN-53 step-budget arm, not a quick control.

**Decision support:** keep 32 strongest + 16 mid-band (48 training contexts) at 400 steps. Describe the evidence
as "beats 48 strongest in two independent samples (14/15 and 11/15 targets)", not as the single-sample margin.

## Matched contrast count (Dm / Em / Cm, 2026-09-24)

The C ablation value averages over as many contrast training contexts as the arm has activating training contexts:
32 / 64 / 96 instead of 48. The contrast pool was extended to 128 per target, and the first 64 were identical to
the cached ones for all 16 targets.

| Pair | Closer to 1, strongest held-out | Closer to 1, mid-band held-out |
|---|---|---|
| Dm vs D | 9/15 (−0.026) | 7/15 (+0.002) |
| Em vs E | 9/15 (−0.011) | 5/15 (+0.029) |
| Cm vs C | 8/15 (−0.001) | 6/15 (+0.003) |
| *B2 vs B (resample noise)* | *5/15 (+0.040)* | *6/15 (+0.026)* |

**No effect beyond resampling noise.** Every matched-vs-unmatched difference is smaller than the difference
between two samples of the same arm. It is also moot for the protocol: 32 + 16 trains on 48 activating contexts,
which already equals the default 48 contrast training contexts (64 retrieved, split 48 / 16).

**Recommendation:** keep the default (`contrast_context_count = None`, i.e. 64 retrieved, 48 train), not "match".
For thin targets, "match" would shrink the contrast set and make the C mean noisier. The option stays available
in config.

## Equal ablation-term weights under the final protocol (Bw / B2w, 2026-09-25)

Daniel switched the primary to γ_C = γ_A = 1, λ = 2e-3 on the strength of 055's eq2, to be confirmed by this refit.
Same contexts as B / B2 (two stratified samples of 32 + 16), only the weights and price changed.

| Arm | Weights, λ | Nodes | Worst-of-3 dev (strongest / mid) | In band (strongest / mid) |
|---|---|---|---|---|
| B | 0.25, 1e-3 | 474 | **0.14** / **0.41** | **9** / 2 |
| Bw | 1, 2e-3 | 478 | 0.28 / 0.75 | 7 / 2 |
| B2 | 0.25, 1e-3 | 462 | **0.23** / **0.35** | **6** / **5** |
| B2w | 1, 2e-3 | 476 | 0.33 / 0.79 | 5 / 2 |

Paired, closer to 1:

| Pair | Strongest held-out | Mid-band held-out |
|---|---|---|
| Bw vs B | **2/15** (+0.054) | 5/15 (+0.046) |
| B2w vs B2 | 6/15 (+0.012) | 6/15 (+0.037) |
| *Noise reference, B2 vs B* | *5/15* | *6/15* |

- **Equal weights do not help under the final protocol.** They are worse in both samples, doubling the typical
  worst miss on the mid-band set, at the same circuit size.
- The 055 eq2 advantage (old contexts, list split) does not carry over.
- Node overlap B–Bw is 0.58, about the refit floor.
- **Recommendation: revert to γ_C = γ_A = 0.25, λ = 1e-3.** Daniel's decision; recorded on DAN-75.

## Would settle it (cheap, local)

- A noise floor: refit A with a different stratification offset, and compare A-vs-A to A-vs-B.
- A budget control: C at 800 steps, so each context is seen as often as in A and B.
