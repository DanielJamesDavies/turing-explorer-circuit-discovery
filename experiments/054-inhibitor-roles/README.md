# 054 — Which role definition holds at set level? (DAN-72)

WCM is unsigned: a node is kept because replacing it with its ablation value
spoils the reproduction of the target. A *role* says which way: an
**excitatory** node props the target up (without it the target comes out too
low), an **inhibitory** node holds it down (without it the target comes out too
high). The v3 eval pass labels roles by gradient × activation on the unmodified
model; on the 60-circuit pilot, setting all such "inhibitors" to zero *lowered*
the target (median release −0.55), so the labels fail as a set.

`role_tests.py` compares role definitions on the same 60 pilot circuits, scored
with the v3 evaluation code (engine `CircuitOnlyPatcher`, every upstream site
ablated):

| Definition | What it measures |
|---|---|
| `G` | gradient × activation on the unmodified model, train slice (current rule) |
| `F_pi` | d(pre-activation)/d(log α) at the **fitted** circuit, circuit-only execution under π ∈ {Z, A, C} |
| `L_pi` | leave-one-out: a(circuit) − a(circuit without node i), node sent to its ablation value under π (24 circuits, 2 per layer, first 16 train contexts) |

**Set-level test**, on the 16 held-out activating contexts, under each π:
`release` = removing all the definition's inhibitory nodes should *raise* the
target; `remove_exc` = removing its excitatory nodes should *lower* it. Both are
reported on the pre-activation (`_pre`) and activation (`_tk`) reads.

Run: `PYTHONPATH=src python experiments/054-inhibitor-roles/role_tests.py`
(→ `results/roles_pilot.jsonl`, summary printed at the end; `SUMMARY=1` re-prints it).

Smoke test (5.resid.13677, 368 nodes): 13 s without LOO, 100 s with LOO over
Z, A, C. Early observation: under zero ablation, removing large groups sends the
stream off-distribution (pre-activation ×60 while the activation dies), so the
set-level test is only meaningful under the mean ablations A and C.

## Results (2026-09-22; 60 pilot circuits, LOO on 24; log `results/roles_pilot.log`)

**No definition holds at set level.** Share of circuits where removing all the
definition's inhibitory nodes raises the target (held-out, pre-activation read):

| Definition | under Z | under A | under C |
|---|---|---|---|
| G (gradient, unmodified model) | 0.38 | 0.12 | 0.03 |
| F (fitted-effect sign) | 0.32 | 0.22 | 0.13 |
| L (leave-one-out vs ablation value) | 0.29 | 0.42 | 0.46 |

Median release is negative for every definition (best: L_C −0.035, L_A −0.081,
i.e. about zero). Removing the excitatory nodes lowers the target in 73–100% of
circuits, as expected. Inhibitory share of nodes: 25–39% depending on the
definition. Definitions agree on 55–90% of nodes (F_A|F_C 0.90, L_A|L_C 0.71,
anything against Z ~0.55–0.65); roles under Z disagree with roles under A/C.

**Why: node effects are strongly non-additive.** Summing each node's
leave-one-out effect predicts the group effect badly:

| π | release predicted (sum of single inhibitory effects) | actual | remove_exc predicted | actual |
|---|---|---|---|---|
| Z | +0.98 | −0.70 | −9.37 | −0.98 |
| A | +0.55 | −0.08 | −3.52 | −0.96 |
| C | +0.59 | −0.04 | −4.19 | −0.91 |

Single inhibitory effects add up to a large predicted rise, yet removing them
together does nothing or lowers the target; excitatory effects add up to 4–9×
the target, yet removing them together only silences it (saturation at −1).
Single-node effects are small (median |effect| 0.4–1.6% of a_pos) and
compensate each other: the same collective, redundant picture as faithfulness.

**Implication.** A node's role is well defined *individually* (leave-one-out
against the ablation value, best under A/C), but "the inhibitory nodes" are not a
separable set, so any claim that treats them as a group (release, role-aware
necessity, the inhibitory branch of role-aware sufficiency to induce) is not
supported. Recommendation for protocol v1: role-blind necessity and role-blind
sufficiency to induce as primary; per-node roles only as descriptive leave-one-out
signs under A/C, with the non-additivity stated.

## Looking at the nodes (`inspect_nodes.py` → `results/node_report.md`)

8 circuits across depth (0.mlp.16000, 1.mlp.8495 "multiple", 3.attn.35598
"Shakespeare", 4.mlp.20758 "modern", 6.attn.8246 "coordinating conjunctions",
7.mlp.15696 "drilling", 9.attn.21759 "(economic) zones", 10.mlp.21748 "such");
6 most inhibitory and 6 most excitatory nodes by L_C, plus 4 where G and L_C
disagree. Each node: its own top contexts, firing on the target's activating vs
contrast contexts, direct logit effect.

**Excitatory nodes are clean and interpretable.** Almost all are same-concept
latents with ~100% peak-token consistency ("Shakespeare" for the Shakespeare
target, "modern/contemporary", "zones", "such", "dr/illing/subsurface/oil" for
drilling), firing far more on the activating than the contrast contexts (e.g.
3.3 vs 0.3), positive under Z, A and C alike, and often promoting the target's
own tokens in the logits.

**Inhibitory nodes are mostly not semantic inhibitors.** Four kinds:
1. **Ubiquitous hub latents** that fire about equally on activating and contrast
   contexts and recur across unrelated circuits: 0.attn.712 ("like / 1 / ("),
   0.attn.5663 ("symbols"), 0.attn.15350 ("."), 0.attn.18179 ("Cov / tutorial").
   Their role *flips* between circuits (0.attn.5663: excitatory for one target,
   inhibitory for two others) and between ablation methods (L_Z +0.9 vs L_C −0.02).
   These look like stream-maintenance latents, not brakes.
2. **Context-type detectors** that fire *more* on the contrast contexts than on
   the activating ones and not at the anchor (e.g. the "modern" circuit's top
   inhibitors, 0.mlp.39049 / 15578 / 3223, 0.resid.16354: ~4.5 vs ~10, garbled or
   non-English/code-like peak tokens). The closest thing to a genuine inhibitor.
3. **Redundant same-concept latents**: in the "such" circuit, the top
   "inhibitors" are strong "such" latents at layers 8–9 (100% "such"), i.e. copies
   of the excitatory concept. Negative leave-one-out here most likely reflects
   normalisation or Top-K competition among redundant copies, not inhibition.
4. **High-α, uninterpretable correction nodes** (α 3–4.6, peak-token consistency
   ~3%, very large fitted effects), which can land on either side.

This explains the non-additivity: "the inhibitory set" mixes hub latents the
stream needs with a few real context detectors, so removing it wholesale breaks
the reconstruction instead of releasing a brake. Next test: restrict to kind 2
(fires more on contrast than activating contexts, not a hub) and check whether
*that* subset passes the set-level release test.

## Kind-2 subset test (`kind2_test.py` → `results/kind2.jsonl`, `results/kind2_hub05.jsonl`)

Kind 2 = inhibitory by leave-one-out under C, fires more on the target's contrast
contexts than its activating ones (mean per-sequence max), and not a hub (in
fewer than `HUB_FRAC` of all production circuits, 049 `fanout_latents.csv`).
Same 24 LOO circuits, held-out contexts, pre-activation read, release as a share
of a_pos. Control = same-size random draws from the *other* LOO-inhibitory nodes
(5 draws, median).

| | hub < 0.01 | hub < 0.05 |
|---|---|---|
| circuits with ≥ 1 kind-2 node | 11 / 24 (sets of 1–7) | 16 / 24 (median 4) |
| release under C: kind 2 | +0.0016, rises in 73% | +0.0035, rises in 75% |
| release under C: random control | +0.0032, rises in 82% | +0.0103, rises in 75% |
| release under A: kind 2 | −0.0007, rises in 45% | −0.0007, rises in 44% |
| release under A: random control | +0.0029, rises in 64% | +0.0034, rises in 62% |
| hub inhibitors, under C | −0.0026, rises in 50% | −0.0031, rises in 50% |
| kind 2 additivity under C: predicted vs actual | +0.0041 vs +0.0016 | +0.0152 vs +0.0035 |
| share inhibitory: contrast-preferring non-hubs vs the rest | 0.17 vs 0.14 | 0.33 vs 0.18 |

**Kind 2 fails.** Context-detector inhibitors are no better as a brake set than
random inhibitory nodes of the same number: under C they release about as often
as the control, by less, and under A they don't release at all. The effects are
tiny either way (under 1% of a_pos). Small sets release more reliably than the
full inhibitory set (73–82% vs 46%), because a few nodes stay closer to additive,
but that holds for any small set. There is a descriptive association: at the
0.05 threshold, contrast-preferring non-hub nodes are inhibitory more often (0.33
vs 0.18). It has no causal weight at set level.

**Conclusion for the paper:** no signed-role claim survives: not for the gradient
rule, not for the fitted sign or leave-one-out, and not for the most
semantically plausible subset. Roles stay descriptive (per-node leave-one-out
under A/C). The finding to report is non-additivity (§ above), which supports the
collective faithfulness picture.

**Data observation (for DAN-64):** four unrelated targets (0.mlp.16000,
3.attn.35598, 4.mlp.20758, 9.attn.21759) show the *same* first three contrast
contexts ("…value of an art collection…", "…ulf to Twitter…", "…A Pareto Im…"),
so the stored contrast contexts may fall back to a shared generic set for some
targets. Needs checking.
