# 056 — Specificity: mechanism for the target, or concept amplifier?

**The concern (Daniel, 2026-09-23).** Faithfulness, necessity and the random-circuit baseline all read the target
alone. A circuit that restores the target by pushing up its whole neighbourhood (concept, topic, context type) passes
all three, yet it is a concept amplifier rather than a mechanism for that latent. The one place this leaks through is
the activation read: if the circuit lifts the target's rivals, the target loses its Top-K place even when its
pre-activation is right. That is exactly what four deep attention pilot targets show (pre-activation reproduced at
1.0–1.6×, activation read 0), so those cases are candidates for amplification, not only measurement artefacts.

**What `specificity.py` measures.** It reads the whole code at the target's site (every latent's pre-activation and
the SAE's actual Top-K set, at the target's anchor) on the 16 held-out contexts, in three runs per ablation method
(Z, A sparsity-preserving, C sparsity-preserving on close contrast contexts): the clean model, the empty circuit, and
the fitted circuit. All runs go through the engine's `CircuitOnlyPatcher`, as in the v3 eval pass.

| Measure | Specific circuit | Concept amplifier |
|---|---|---|
| Target's rank among the site's pre-activations; share of anchors in Top-K | as clean | drops |
| Latents switched on above the clean Top-K cut | few | many |
| Jaccard of the circuit's Top-K set with the clean one (vs the empty circuit's) | high, above empty | low |
| **Siblings** (co-fire with the target in the clean Top-K on ≥ 50% of TRAIN contexts): faithfulness, same form as the target's | clearly below the target's | as high as the target's |

The control group for siblings is every other latent active in the clean held-out Top-K: it says how much of the
site's ordinary activity the circuit restores in general. A flag is raised when a target is restored (≥ 0.5) and its
siblings reach at least 0.8× its faithfulness.

**Arms:** `old` (the 15k circuits), `close` and `eq2` from 055, on the 16 targets in `055-close-contrast/seeds.txt`.

## Run (not yet run)

```
PYTHONPATH=src python experiments/056-specificity/specificity.py
```

- `SMOKE=1`: first target of the first arm only, to `results/specificity_smoke.jsonl`.
- `SUMMARY=1`: re-print the summary from `results/specificity.jsonl`.
- `ARMS`, `SEEDS`, `SIB_FRAC` (0.5), `MIN_DEN` (0.05) are set through the environment.

Estimated cost: 3 arms × 16 targets × (clean + 3 × (empty + circuit)) held-out runs plus one train run each. With
the engine start-up this should take about 15–20 minutes locally.

## Results (2026-09-23; 3 arms × 16 targets × 3 ablations, `results/specificity.jsonl`, `results/summary.md`)

**Most circuits are target-specific.** Median target faithfulness (pre-activation) 0.82–1.04 against siblings
0.27–0.36 and other active site latents 0.09–0.29, in every arm and ablation. About half the targets are clean:
target ~1, siblings 0.2–0.5, few switch-ons (0.mlp.16000, 1.resid.12137, 5.mlp.11680, 8.attn.37097,
11.resid.8702, 0.mlp.5196).

**But circuits switch on many latents under mean ablation.** Latents lifted above the clean Top-K cut, median:

| | Z | A | C |
|---|---|---|---|
| empty circuit | 7,956 | 18 | 18 |
| old | 1,633 | 391 | 366 |
| close | 903 | 489 | 491 |
| eq2 | 1,774 | 636 | 390 |

Under Z the circuit restores the code relative to the empty circuit; under A and C it adds 20–35× the empty
circuit's switch-ons. The target still stays in the Top-K for most targets, but two failure modes appear:

- **Site-wide inflation** (10.attn.16967, 5.attn.34661, 2.attn.33479, 10.attn.36603): thousands of latents pushed
  above the cut (up to 10,235), so the target loses its Top-K place although its pre-activation is right
  (10.attn.16967 eq2: pre-activation faithfulness 1.24, 7,226 switch-ons, 0% in Top-K). The Top-K-censored deep
  attention targets of the pilot are this, not a measurement quirk.
- **Concept amplifier** (6.resid.18234 in close and eq2; borderline 11.mlp.30743): siblings restored as well as or
  better than the target (1.29–1.86 vs ~1.0), the whole site inflated (1.3–1.9), 4,000–10,000 switch-ons, yet
  free0 / freeM / freeN all ~1.0: it passes every score in the current battery.

Equal weights (eq2) do not fix it: they rescue 2.attn.33479's Top-K place but inflate 6.resid.18234 and
10.attn.16967 further.

## Fix: an off-target term in WCM's loss (2026-09-23)

**Why the objective allows it.** WCM's loss reads the target's pre-activation only, w_t·x + b_t, which constrains
the stream at the target's site along one direction of 1,024. Everything else there is free, so the cheapest route
to the target may lift hundreds of other latents. Coefficient size does not drive it (switch-ons vs α median / p90 /
max: Spearman ρ 0.10–0.40; worst inflaters have median α ≈ 1.0–1.1), so pricing or capping α would not fix it.

**The term** (`learned_mask.py`: `offtarget_weight` γ_S, `offtarget_mode`; opt-in, off = bit-identical, learned-mask
tests 137 pass plus one pre-existing failure). Inside each of the Z / C / A terms:
γ_S · Σ_{j≠t} relu(pre_j(circuit) − ref_j)², with ref_j = max(clean, empty circuit under that ablation), and in
**cut** mode also the clean Top-K cut at that anchor, so only switch-ons are charged. References come from one
no-grad pass per ablation before training.

**Calibration.** The sum runs over ~40k latents: γ_S = 0.25 ballooned circuits (0.mlp.16000 33 → 843 nodes);
0.01 and 0.001 in "all" mode gave 239 and 72. In "all" mode the penalty is dominated by harmless sub-threshold drift,
which is why clean targets grow; cut mode removes that.

**Results** (16 targets, C ablation unless stated; faithfulness on close contrast contexts):

| Arm | Nodes | free0 / freeM / freeN | All 3 | Switched on (A / C) | Site inflation (A / C) | Amplifier flags |
|---|---|---|---|---|---|---|
| close | 398 | 0.97 / 0.92 / 0.79 | 6 | 488 / 491 | 0.29 / 0.23 | 3 |
| otall3 (1e-3, all) | 520 | 0.96 / 0.97 / 0.90 | 6 | 70 / 73 | 0.00 / 0.04 | 1 |
| otcut3 (1e-3, cut) | 401 | 0.96 / 0.94 / 0.85 | 7 | 174 / 178 | 0.09 / 0.11 | 3 |
| otcut3e3 (3e-3, cut) | 438 | 1.00 / 0.91 / 0.83 | 6 | 246 / 183 | 0.14 / 0.05 | 3 |
| otcut2 (1e-2, cut) | 439 | 0.97 / 0.85 / 0.80 | 5 | 216 / 183 | 0.09 / 0.09 | 0 |

(Amplifier flag here = a (target, A or C) pair with target pre-activation faithfulness ≥ 0.5, ≥ 10 siblings, and
sibling median ≥ 0.8× the target; `results/summary.md` uses the looser version without the 10-sibling minimum.)

Every off-target arm cuts switch-ons 3–7× and site inflation 2–6× without a faithfulness cost beyond noise. Part of
otcut2's lower freeM/freeN is overshoot (2.attn.33479 1.44, 6.resid.18234 1.31), not under-restoration. Per-target
results move non-monotonically with γ_S (7.mlp.28744 switch-ons 111 → 1,080 → 2,424 as γ_S rises; 3.mlp.23075 freeM
1.27 → 0.58 → 0.85), so one refit per target cannot rank the weights.

**Previously censored targets** (out of Top-K under close):
- 2.attn.33479: rescued in every off-target arm (100% in Top-K under Z / A / C; otcut3 0.94 / 1.00 / 1.02).
- 5.attn.34661: rescued by otcut2 (100%, 1.04 / 1.12 / 1.11) and mostly by otall3 (94%); not by otcut3.
- 10.attn.36603: fixed under Z and A in every arm (freeM 0.81–0.95); under C only otcut3e3 partly (81%, 0.58), and
  the C failure comes with few switch-ons (45–91), so it is not rival inflation.
- 10.attn.16967: partial at best (in Top-K on 40% of anchors, activation faithfulness 0.5–0.7).

**Decision (2026-09-23):** cut mode becomes part of WCM, provisional γ_S = 3e-3; the weight is settled in the DAN-64
grid; specificity (switch-ons vs the empty circuit, sibling ratio, Top-K membership) joins the evaluation battery.

**Implications (first pass, before the fix).** (1) The evaluation battery needs specificity: switch-ons relative to the empty circuit, the sibling
ratio, and the target's Top-K membership. (2) Next test: the firing-margin objective (`margin_topk`, built
2026-08-30), which penalises lifting rivals because they raise the Top-K cut. (3) If that fails, a one-sided
off-target loss term measured against the empty circuit.

## Protocol-v1 circuits (2026-09-25, `spec059.py`, `results/v059/`)

The diagnostic runs on 059's circuits for the protocol training set (32 strongest + 16 mid-band), each weight setting
fitted on two stratified samples: B / B2 (γ 0.25, λ 1e-3) and Bw / B2w (γ 1, λ 2e-3). It uses 059's contexts, with
held-out strongest contexts and siblings drawn from the 48 strongest-train contexts.

| Arm | Target pre-faith (C) | Sibling median (C) | Lifted above cut (Z / A / C) | Amplifier flags (target × π) |
|---|---|---|---|---|
| B | 0.90 | 0.36 | 631 / 259 / 235 | 6 |
| B2 | 0.86 | 0.39 | 771 / 371 / 342 | 8 |
| Bw | 0.88 | 0.38 | 818 / 474 / 267 | 5 |
| B2w | 0.90 | 0.36 | 1514 / 534 / 453 | 8 |

- **The weights don't change specificity.** Flags are 5–8 for every arm. The resample difference (B vs B2: 6 vs 8)
  is as large as the weight difference.
- **The typical circuit is specific:** siblings are restored to about 0.36–0.39 against the target's 0.86–0.90. A
  minority are amplifiers, concentrated in the same targets as before: 5.attn.34661 (siblings 1.3 vs target 0.74),
  7.mlp.28744 (siblings 1.3–3.8), 10.attn.36603 and 11.mlp.30743 under A. Several of these are near-threshold or
  deep attention targets.
- Consistent with the earlier conclusion: specificity is reported per circuit, never claimed for the method.
