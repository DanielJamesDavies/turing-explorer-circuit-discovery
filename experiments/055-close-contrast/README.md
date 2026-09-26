# 055 — Which contrast contexts should WCM train and score on? (DAN-64, local pilot)

The 15k production circuits were fitted with `floor_negctx_mode="store"`: the C term of WCM\[Z+C+A\] read the
pre-built kNN contrast store. That store is broken (DAN-64):

- its query is the centroid of the target's activating-context representations, taken from `seq_repr`, which holds
  only 200,000 of the corpus's 24.8M sequences;
- when none of a target's 128 activating contexts is in that sample the query is the zero vector, every similarity
  ties at 0, and the search returns the same 64 sequences. **37.6% of the 15k targets** got that one shared list;
- the remaining targets' queries come from a median of **1** of their 128 activating contexts;
- silence was never checked, and the selector rejects 14–51% of close candidates for making the target fire.

`NegContextSelector` (`src/utils/neg_context_selector.py`) already implements the intended flow, and the production
config never switched to it: representations for the target's top contexts (read from `seq_repr`, otherwise run
through the model), ranking of the corpus pool by max cosine similarity, the target's own activating contexts
excluded, and each candidate kept only if the target stays out of Top-K.

## What this experiment runs

`refit.py` refits the same 16 pilot targets (8 whose store row was the shared fallback list, 8 that retrieved,
spread over depth; `seeds.txt`) through the **production WCM path** — `AblationGradientDiscovery`,
`attribution_mode="mask"`, the `config-h100-triamp.yaml` learned-mask values — once per contrast source:

| Arm | C during fitting |
|---|---|
| `old` | the stored 15k circuits (H100, store contrast contexts) |
| `store` | local refit, same store contexts — the **noise floor**: refits swap nodes on their own |
| `close` | selector, nearest silent contexts |
| `random` | selector, random silent contexts |
| `distant` | selector, most dissimilar silent contexts |

`run_all.sh` (close is fitted first, by hand) then `run_store.sh` score **every arm against every contrast source**
(close / random / distant / store) with the v3 eval pass (`CTR_SOURCE`, new in `amp_eval_pass_v2.py`): activation
read, held-out contexts, every upstream site ablated. `analyse.py` writes `results/summary.md`: sizes, the
train-source × eval-source matrix per metric, the fallback/retrieved split, the paired change against `old`, and
node overlap (Jaccard) between arms.

The matrix matters because each arm wins the metric it trained on (the home-turf effect seen throughout this
programme), so no arm may be judged on its own contrast set alone.

## Results (2026-09-22/23; `results/summary.md`)

Headline scores are the three faithfulness ratios on held-out contexts, activation read: free0 (Z), freeM_topk (A,
sparsity-preserving), freeN_topk (C, sparsity-preserving, on close contrast contexts). "All 3" = all three in
[0.8, 1.25] (illustrative band; the pass rule is DAN-8). Results are bimodal per target (a target holds near 1 or
collapses near 0), so counts are more honest than medians. Two targets can never pass: 0.mlp.5196 has no valid free0
ratio, and 10.attn.16967 is Top-K-censored (see 056).

| Arm | γ_C = γ_A | λ | γ_S (mode) | Nodes | free0 / freeM / freeN | All 3 |
|---|---|---|---|---|---|---|
| old (15k, store, H100) | 0.25 | 1e-3 | – | 354 | 0.97 / 0.83 / 0.80 | 5 |
| store (local noise control) | 0.25 | 1e-3 | – | 346 | 0.93 / 0.92 / 0.68 | 5 |
| close | 0.25 | 1e-3 | – | 398 | 0.97 / 0.92 / 0.79 | 6 |
| random | 0.25 | 1e-3 | – | 341 | 0.99 / 0.84 / 0.76 | 4 |
| distant | 0.25 | 1e-3 | – | 377 | 1.01 / 0.87 / 0.78 | 6 |
| eq1 | 1 | 1e-3 | – | 497 | 0.91 / 0.90 / 0.91 | 7 |
| eq2 | 1 | 2e-3 | – | 303 | 0.98 / 0.96 / 0.91 | 7 |
| otall3 | 0.25 | 1e-3 | 1e-3 (all) | 520 | 0.96 / 0.97 / 0.90 | 6 |
| otcut3 | 0.25 | 1e-3 | 1e-3 (cut) | 401 | 0.96 / 0.94 / 0.85 | 7 |
| otcut3e3 | 0.25 | 1e-3 | 3e-3 (cut) | 438 | 1.00 / 0.91 / 0.83 | 6 |
| otcut2 | 0.25 | 1e-3 | 1e-2 (cut) | 439 | 0.97 / 0.85 / 0.80 | 5 |

**Contrast source.** A proper C barely moves membership or the headline scores. Node overlap with the old circuits is
0.56–0.60 against 0.66 for a same-C refit (the noise floor), and every headline median moves by ≤ 0.04 against
close. What the source does change is robustness off the training set: each arm is best on the contrast set it
trained on (dense freeN, the home-turf effect), and the old circuits are worst everywhere but the store. Decision
(Daniel, 2026-09-23): **adopt close** (NegContextSelector) now; keep the existing 15k fits for now; refit the 15k
later, since publishing on the broken C "feels off".

**Floor weights.** Equal weights (γ_C = γ_A = 1) reach the same or slightly better all-three count at 24% fewer nodes
when λ is doubled (eq2: 7/16 at 303 nodes vs 6/16 at 398), trading free0 failures for freeM/freeN rescues on
individual targets. +1 target is within the noise floor (the two store fits differ by 1). To be settled in the DAN-64
grid.

**Off-target term** (added to the engine for 056; `offtarget_weight`, `offtarget_mode`): no faithfulness cost within
noise, large specificity gain (see `../056-specificity/README.md`). Per-target results swing non-monotonically with
γ_S, so choosing between 1e-3, 3e-3 and 1e-2 needs more targets. Provisional default: cut mode, γ_S = 3e-3.

**Failures cluster** on a fixed set of hard targets, mostly deep attention and several from the fallback group, plus
a population split: store-fallback targets (activating contexts absent from the 200k seq_repr sample, i.e. rarer
latents) score far lower on freeM than retrieved ones (0.03–0.73 vs 0.91–1.08).
