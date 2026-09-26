# 057 — Wired weighted circuits: direct-effect edges for WCM

2026-09-24. Turns a WCM circuit (a set of upstream latents with coefficients, a "star" onto the target) into a
wired graph using the SFC direct-effect edge construction. Until now this was done only for attribution circuits
(`analysis-restyled/edge-pilot/`, `src/circuit/instrument/edge_attribution.py`).

## Why the attribution-circuit code doesn't transfer

`attach_direct_edges` linearises around the **unmodified model**. A weighted circuit is faithful only in its own
**circuit-only run**:
- members at α × live value;
- every other upstream latent at the ablation value: zero, or the A / C mean kept k-sparse;
- the SAE error recomputed from the edited stream.

Edges measured on the clean run describe the full model. Rescaling them by α doesn't fix this, because the
linearisation point is still wrong.

## Construction (`src/circuit/instrument/weighted_edges.py`)

`WeightedCircuitGraph` replays the circuit exactly as `CircuitOnlyPatcher(keep_scales=…)` does, in three modes:
- **total:** node = α·live + a zero tap, so the gradient gives g_u, the total gradient of the metric.
- **stop:** node = a leaf, so circuit nodes are gradient terminals while model ops, the fill and the SAE errors
  stay live. This gives direct Jacobians.
- **pin:** every node is fixed except one live node, and one node can be overridden. Used for the causal check.

The metric m is the target's pre-activation at its anchor, averaged over contexts. This is what WCM trains on.

With b_u defined as u's **floor**, the value latent u takes when dropped from the circuit (this includes the k-sparse
fill slot it frees):

    A_u        = Σ g_u (v_u − b_u)                       node attribution
    w(u→d)     = Σ g_d · ∂live_d/∂leaf_u · (v_u − b_u)   member edge (direct path, total onward)
    w(u→tgt)   = Σ h_u (v_u − b_u)                       edge to the target (h = direct gradient)
    A_u        = w(u→tgt) + Σ_d w(u→d)                   exact chain rule → implementation check

The coefficients enter twice: α_u through v_u − b_u, and α_d through ∂(α_d f_d)/∂·. Non-members are constants in
this run, so every edge connects circuit nodes.

**One edge set per ablation method.** WCM trains under Z, A and C jointly, so edges are computed under each.
*Consensus* edges are those above threshold with the same sign under all three.

**Thresholds:** θ × Σ|A_u| with θ ∈ {1e-2, 1e-3}.

## Checks (per circuit × ablation method)

| Check | What it tests |
|---|---|
| **replay** | This run's target pre-activation vs `CircuitOnlyPatcher`'s. Tests that the instrument reproduces the evaluated circuit. |
| **conservation** | Σ\|Σ out-edges − A_u\| / Σ\|A_u\|. Tests the code (bf16 autocast sets the floor). |
| **causal edges** | Top-10 edges. Pin every node at its circuit value except d (live) and u (→ floor), then take g_d · Δd. Tests that edges are not just gradients. |
| **causal nodes** | Top-10 nodes, each dropped with everything else live. Tests leave-one-out linearity, which is known to fail (054); reported, not a pass criterion. |

## Smoke results (Z, rkeep3e3fix)

- **0.mlp.16000:** 35 nodes, all at one site, so no member edges exist. Replay 0.2%, node r 0.99.
- **11.resid.8702:** 553 nodes.
  - Replay 0.18%, conservation 0.32%.
  - **Edge causal r 0.96, sign 10/10.**
  - Node causal r −0.02. Dropping one large-α node under Z leaves a near-empty deep stream that RMSNorm amplifies:
    one drop moves m by −4,772 against a linear +1.1. This is the leave-one-out non-additivity, not a code fault.
    The pinned edge test avoids it because it holds the stream.
  - 4 edges at θ 1e-2 (depth 3). Σ edges-to-target 8.5 of ΣA 36.7, so most attribution routes through other
    nodes.

## Full run

    PYTHONPATH=src python experiments/057-wcm-edges/wcm_edges.py          # ARM=rkeep3e3fix, 16 targets × Z/A/C
    SUMMARY=1 PYTHONPATH=src python experiments/057-wcm-edges/wcm_edges.py

Outputs:
- `results/edges_<arm>.jsonl`;
- `results/summary_<arm>.md`;
- matrices in `data/<arm>/<target>.pt`: `nodes` [(layer, kind, idx, α)], and per π `A`, `E[d, u]`, `Et`.

## Results: main 16, rkeep3e3fix (primary config v1), 2026-09-24

About 15 minutes locally on an RTX 5070 Ti. Batched VJPs worked everywhere; there was no sequential fallback.
Files: `results/summary_rkeep3e3fix.md` (driver) and `results/analyse_rkeep3e3fix.md` (`analyse.py`). Two
targets are layer-0 MLP with all nodes at one site, so they have no member edges; the 14 others carry the stats.

**The construction is sound.**
- Replay is within 0.4% of the evaluator everywhere.
- Conservation error is ≤ 0.9% everywhere, median 0.2–0.3%.
- Pinned causal edge check, median over targets: r = 0.93 (Z), 0.98 (A), 0.99 (C); sign agreement median 1.0.
- Weak cases are mostly Z: 7.mlp.28744 (0.53), 8.attn.37097 Z (0.29), 5.mlp.11680 Z (0.57). One is C:
  10.attn.16967 C (0.67, sign 4/10). Zero ablation leaves a thin stream that makes the first-order picture
  rougher.
- Node leave-one-out r has median 0.68–0.75 but many negatives. This is the known non-additivity, reported
  only.

**What the wiring says.**

| Median over 14 targets | Z | A | C |
|---|---|---|---|
| Direct share (Σ edges to target / ΣA) | 0.33 | 0.22 | 0.37 |
| Member-edge mass / node mass | 4.0 | 3.3 | 3.4 |
| Share of edge mass in the top 10 edges | 3.7% | 5.3% | 5.1% |
| Edges needed for half the edge mass | 1,758 | 1,323 | 1,167 |
| Edges ≥ 1e-2 · Σ\|A\| | 10 | 7 | 7.5 |
| Edges ≥ 1e-3 · Σ\|A\| | 474 | 320 | 403 |
| Max depth at 1e-3 | 10.5 | 9.5 | 9 |

1. **The circuits are internally multi-step, not stars.** Only about a third of a circuit's attribution reaches
   the target directly; the rest routes through other circuit nodes (depth ≈ 10 at 1e-3).
2. **The wiring is dense and diffuse.** Edge mass is 3–4× node mass, so positive and negative routes cancel
   heavily. The top 10 edges hold about 5% of it, and half of it needs over 1,000 edges. This is the edge-level
   version of the collective-tail finding: no sparse diagram carries the computation. A 1e-2 threshold leaves
   about 7–10 edges, which is a skeleton, not the mechanism.
3. **The wiring depends on the ablation method.**
   - Pairwise Jaccard of the 1e-3 edge sets across Z/A/C: median 0.28.
   - Top-20 edge overlap: 27%.
   - Consensus (all three, same sign): median 82 of 839 edges in the union at 1e-3; 0–5 edges at 1e-2.
   - The most ablation-stable targets are 1.resid.12137 (0.58), 6.resid.18234 (0.48), 2.attn.33479 (0.48) and
     11.mlp.30743 (0.46). These are the candidates for a rendered "wired circuit" figure (consensus edges only).
   - The least stable are 7.mlp.28744 (0.11), 10.attn.16967 (0.12) and 10.attn.36603 (0.14).

**For the paper:** WCM circuits can be wired validly. The honest object is the consensus edge set, with its share
of edge mass stated. A sparse wiring diagram drawn from one ablation method would overclaim, just as a single-run
node set does. This is a 16-target pilot, so don't quote it yet.

## Figures (`render.py`, 2026-09-24)

`figures/wiring_consensus.png/.pdf` is a 2 × 2 panel; there is also one `wiring_<target>` per target. The four
most ablation-stable targets are shown.

**What is drawn:**
- Consensus edges only: ≥ 1e-3 · Σ|A| under Z, A and C, with one sign.
- Each edge is drawn at its weakest normalised weight across the three.
- The selection is a backbone grown from the target: the strongest consensus edge into an already-drawn node, at
  most 4 in-edges per node, 22 nodes and 28 edges. Every drawn node has a consensus path to the target.
- Colours follow the Lab Bright theme: the target is honey and nodes are blue; blue edges are positive, red
  negative. Serif font (diagram).

**Share of edge mass carried by the whole consensus set:**

| Target | Consensus share |
|---|---|
| 1.resid.12137 | 44% |
| 2.attn.33479 | 27% |
| 6.resid.18234 | 15% |
| 11.mlp.30743 | 14% |

The panel titles state this share, so the drawing isn't read as the whole mechanism.

**Readings (pilot, descriptive):**
- **11.mlp.30743** has a residual-stream relay: a chain of mostly negative consensus edges runs up the residual
  stream, one hop per layer, from L0–L2 to L10 and into the target.
- **6.resid.18234** also relays through each residual layer (L0 → L5). This target was the pure concept amplifier
  in the plain-WCM specificity pass (056), and it still is under rkeep3e3fix:
  - A: target 1.30 vs sibling median 1.09;
  - C: 1.26 vs 1.08.

  Siblings are restored at ≥ 0.8× the target, so it is flagged, and the other active latents are restored more
  still (1.55 / 1.82). Its wiring is a real relay, but it relays the neighbourhood, not the target specifically.
  Don't use it as a showcase; swap in 3.mlp.23075 (Jaccard 0.41) if a fourth panel is needed.
- **Shallow targets (1.resid.12137, 2.attn.33479)** are fed by a few layer-0 hubs (R0/28723, R0/29422; M0/2152,
  R0/2152) with attention latents feeding those hubs.
- **Same-index pairs across the layer-0 MLP and residual dictionaries recur:** M0/2152 → R0/2152 and
  M0/35468 → R0/35468 here, and many more in the unfiltered top edges. This looks like an SAE-bank property (aligned
  dictionaries at layer 0?). Not investigated.

### What the 11.mlp.30743 relay is (`describe.py` → `results/describe_11.mlp.30743.md`)

**Every chain node is an "iv" string detector.** Each has 100% of its top contexts peaking on the subword "iv":
herb*iv*ores, omn*iv*ores, der*iv*ative, p*iv*otal. The chain is R2/1164, R3/21272, R4/22454, R5/38187, R6/1448,
R7/14260 and R9/37889 (α 1.2 → 2.8, rising with depth). Every one:
- fires 9–26 at the target's anchor;
- sits at about 0 on its close contrast contexts;
- promotes "otal" first-order.

The target also peaks on "iv" in 64/64 contexts. All four sampled windows are "p**iv**otal", but the 64 weren't
checked for pivotal-only.

**The flow is token identity carried up the residual stream.** Each layer's residual dictionary has its own "iv"
latent, and each one hands on to the next. It is a real, clean, ablation-stable relay, but it is transport of the
current token, not a computed concept. This is the string-detector control from the relativity work, and it
applies here.

The one non-"iv" contributor in the backbone is 10.resid.36122. It fires on word-internal fragments of long
Latinate words (pre, inter, …; 12% consistency) and has a positive direct edge to the target.

**The red edges are not inhibition.**
- An edge's sign is sign(u moves d) × sign(d's marginal effect on the target).
- Every chain node's own attribution is negative at the fitted operating point: A_u = −0.02 to −0.04 of Σ|A|
  under Z, A and C. Pushing any of them higher would lower the target slightly.
- So the member edges come out negative, even though each "iv" latent presumably drives the next.
- A plausible cause (inference, untested): the fit overshoots slightly (target pre-activation 17.0 vs 16.65 at the
  anchor under Z) with weight decay holding α. At a stationary point, overshoot times the marginal effect is
  balanced by the decay, which makes the marginal effect negative.

### Is it ONE feature relayed? Yes (`check_relay.py` → `results/check_relay.md/.json`)

The sample is 4,096 random corpus sequences: 262,144 positions, of which 115 are "iv" tokens (token ids 440 and
20444).

**Behaviour.**
- Each chain latent fires on 88–99% of all "iv" tokens (recall). Its precision on "iv" is 0.38–0.59; the rest are
  mid-word fragments (in, ov, v, av, riv, …).
- Consecutive chain latents co-fire at the same positions: Pearson r of activations 0.87–0.95, Jaccard 0.30–0.47.
  The first and last link, R2 → R9, still has r = 0.83.
- The target is narrower: 79 of its 116 firings are "iv" (precision 0.68, recall 0.69), consistent with firing on
  "pivotal" rather than every "iv". Each chain latent correlates with the target at r = 0.62–0.81.

**Geometry.**
- Decoder cosine between consecutive chain latents is 0.38–0.68, against a random cross-dictionary baseline of
  median |cos| 0.023 (99th percentile 0.093).
- Each downstream latent is its upstream latent's nearest neighbour among all 40,960 latents of its dictionary in
  4 of 6 links, and second nearest in the other 2.
- Read-write alignment, cos(downstream encoder row, upstream decoder column), is 0.23–0.47.

**Census: the circuit contains every layer's best "iv" detector.**
- For every upstream residual dictionary, R0 through R10, the latent with the best F1 as an "iv" detector on the
  corpus is a circuit member. The circuit keeps only 11–54 of 40,960 residual latents per layer.
- The consensus backbone draws R2–R7 and R9 (both of R9's two strong "iv" latents are members). R0/19408,
  R1/9353, R8/40171 and R10/5650 are members whose edges fall below the drawn consensus backbone.
- Coefficients tend to grow with depth along the relay: α 1.0 at R0, 2.1 at R4–R5, 2.6–2.7 at R6–R7, 2.8–3.4 at
  R9–R10.

**Reading.** WCM, which is never told about tokens or layers, recovered the residual stream carrying one feature
(the identity of the current subword) through eleven successive dictionaries. Its edges wire each layer's copy to
the next, and the copies are the same feature by geometry and by behaviour on held-out text.

It's one circuit (a case study). It's token identity, the most basic thing the residual stream carries, not a
computed concept. It's a pilot on the primary v1 config.

Precedents to cite for cross-layer feature matching: Laptev et al. 2025 (feature flow); Lindsey et al. 2024
(crosscoders). What's new is recovering the relay as causal direct effects inside a discovered circuit.

**Figure fix to consider:** colour an edge by sign(u moves d), from the Jacobian term without g_d, and show each
node's marginal effect separately. As drawn, red mixes "suppresses d" with "d is past its useful level".

**Next:**
- Compare with the attribution-circuit edge statistics (edge-pilot: ~9 edges/node, depth ~4 local; ~40/node,
  depth ~10 path) at matched thresholds.
- Decide whether the edge-mass share of the consensus set should be reported per circuit.
